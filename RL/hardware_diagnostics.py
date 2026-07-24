"""Device/hardware diagnostics: startup printout, periodic sampling for the
progress display, and reproducibility metadata (git commit, package
versions, GPU/CPU/RAM specs) recorded into every run's manifest.

No dependency on the training loop -- this is pure inspection, safe to call
from the benchmark, training, or comparison scripts alike.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


@dataclass
class GitInfo:
    commit: str | None
    branch: str | None
    dirty: bool
    diff: str | None  # uncommitted diff, truncated


def get_git_info(max_diff_chars: int = 20_000) -> GitInfo:
    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    dirty = bool(status)
    diff = _run(["git", "diff", "HEAD"]) if dirty else None
    if diff and len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + "\n... (truncated)"
    return GitInfo(commit=commit, branch=branch, dirty=dirty, diff=diff)


@dataclass
class GPUInfo:
    available: bool
    name: str | None = None
    cuda_version: str | None = None
    total_memory_gb: float | None = None
    driver_utilization_pct: float | None = None  # from nvidia-smi, None if unavailable


def get_gpu_info() -> GPUInfo:
    if not torch.cuda.is_available():
        return GPUInfo(available=False)
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    utilization = None
    out = _run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
    if out:
        try:
            utilization = float(out.splitlines()[0])
        except ValueError:
            pass
    return GPUInfo(
        available=True, name=name, cuda_version=torch.version.cuda,
        total_memory_gb=round(total, 2), driver_utilization_pct=utilization,
    )


@dataclass
class SystemInfo:
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu: GPUInfo
    cpu_count_logical: int
    cpu_count_physical: int
    ram_total_gb: float
    platform: str
    mujoco_version: str
    stable_baselines3_version: str


def get_system_info() -> SystemInfo:
    import mujoco
    import stable_baselines3

    gpu = get_gpu_info()
    return SystemInfo(
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        gpu=gpu,
        cpu_count_logical=psutil.cpu_count(logical=True) or 0,
        cpu_count_physical=psutil.cpu_count(logical=False) or 0,
        ram_total_gb=round(psutil.virtual_memory().total / (1024**3), 2),
        platform=platform.platform(),
        mujoco_version=mujoco.__version__,
        stable_baselines3_version=stable_baselines3.__version__,
    )


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[hardware] --device cuda requested but CUDA is not available; falling back to cpu.")
        return "cpu"
    return requested


def print_startup_diagnostics(device: str, n_envs: int, batch_size: int, rollout_size: int) -> SystemInfo:
    info = get_system_info()
    print("=" * 72)
    print("HARDWARE / DEVICE DIAGNOSTICS")
    print("=" * 72)
    print(f"  selected device        : {device}")
    print(f"  CUDA available         : {info.cuda_available}  (CUDA {info.cuda_version})")
    if info.gpu.available:
        print(f"  GPU                    : {info.gpu.name}  ({info.gpu.total_memory_gb} GB VRAM)")
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"  GPU memory allocated   : {allocated:.3f} GB (at process start)")
    else:
        print("  GPU                    : none detected")
    print(f"  PyTorch version        : {info.torch_version}")
    print(f"  MuJoCo version         : {info.mujoco_version}")
    print(f"  Stable-Baselines3 ver. : {info.stable_baselines3_version}")
    print(f"  CPU cores (logical)    : {info.cpu_count_logical}  (physical: {info.cpu_count_physical})")
    print(f"  RAM total              : {info.ram_total_gb} GB")
    print(f"  parallel environments  : {n_envs}")
    print(f"  PPO batch size         : {batch_size}")
    print(f"  rollout buffer size    : {rollout_size}  (n_steps * n_envs)")
    print("-" * 72)
    print(
        "  NOTE: MuJoCo physics stepping runs on CPU in every worker process "
        "(no GPU-accelerated simulator such as MJX is wired in). Only the PPO "
        "policy/value network's forward/backward passes and the optimizer step "
        "run on the selected device. For a small MLP policy on a 33-dim "
        "observation, GPU transfer overhead often outweighs any compute "
        "speedup versus CPU -- the benchmark mode measures this directly "
        "rather than assuming CUDA is faster."
    )
    print("=" * 72)
    return info


@dataclass
class HardwareSample:
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    gpu_utilization_pct: float | None
    gpu_memory_used_gb: float | None
    gpu_memory_total_gb: float | None


def sample_hardware() -> HardwareSample:
    """Cheap-ish point-in-time sample. Call this every few seconds, not every env step."""
    cpu_percent = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    gpu_util = None
    gpu_mem_used = None
    gpu_mem_total = None
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        gpu_mem_total = total_bytes / (1024**3)
        gpu_mem_used = (total_bytes - free_bytes) / (1024**3)
        out = _run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        if out:
            try:
                gpu_util = float(out.splitlines()[0])
            except ValueError:
                pass
    return HardwareSample(
        cpu_percent=cpu_percent,
        ram_percent=vm.percent,
        ram_used_gb=round(vm.used / (1024**3), 2),
        gpu_utilization_pct=gpu_util,
        gpu_memory_used_gb=round(gpu_mem_used, 3) if gpu_mem_used is not None else None,
        gpu_memory_total_gb=round(gpu_mem_total, 3) if gpu_mem_total is not None else None,
    )


def system_info_as_dict() -> dict:
    info = get_system_info()
    d = asdict(info)
    return d


def pip_freeze() -> str | None:
    return _run([sys.executable, "-m", "pip", "freeze"])
