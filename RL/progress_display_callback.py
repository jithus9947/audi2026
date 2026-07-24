"""Periodic (not per-step) console progress line + manifest update.

Hardware polling and manifest writes are real I/O, so this only fires every
``poll_interval_s`` wall-clock seconds (default 7s) rather than every env
step -- polling per-step would itself become the bottleneck at the FPS this
pipeline runs at.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from RL.hardware_diagnostics import sample_hardware
from RL.run_management import write_json_atomic


class ProgressDisplayCallback(BaseCallback):
    def __init__(
        self,
        target_timesteps: int,
        run_dir: Path,
        manifest: dict,
        task_metrics_callback,
        device: str,
        n_envs: int,
        poll_interval_s: float = 7.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_timesteps = max(target_timesteps, 1)
        self.run_dir = Path(run_dir)
        self.manifest = manifest
        self.task_metrics_callback = task_metrics_callback
        self.device = device
        self.n_envs = n_envs
        self.poll_interval_s = poll_interval_s
        self._start_time = 0.0
        self._start_timesteps = 0
        self._last_poll = 0.0
        self.last_checkpoint: str | None = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._start_timesteps = self.num_timesteps
        self._last_poll = 0.0  # force an immediate first print

    def _on_step(self) -> bool:
        now = time.time()
        if now - self._last_poll < self.poll_interval_s:
            return True
        self._last_poll = now

        elapsed = now - self._start_time
        done_this_run = self.num_timesteps - self._start_timesteps
        fps = int(done_this_run / elapsed) if elapsed > 0 else 0
        remaining = max(self.target_timesteps - self.num_timesteps, 0)
        eta_s = remaining / fps if fps > 0 else float("inf")
        pct = 100.0 * self.num_timesteps / self.target_timesteps if self.target_timesteps else 0.0
        n_steps = getattr(self.model, "n_steps", 1) or 1
        iteration = self.n_calls // n_steps

        hw = sample_hardware()
        buf = self.task_metrics_callback._episode_buffers if self.task_metrics_callback else {}
        mean_reward = float(np.mean([e["r"] for e in self.model.ep_info_buffer])) if self.model.ep_info_buffer else None
        mean_forward = float(np.mean(buf["forward_distance"])) if buf.get("forward_distance") else None
        mean_landing_error = float(np.mean(buf["landing_error"])) if buf.get("landing_error") else None
        hit_rate = float(np.mean(buf["target_hit"])) if buf.get("target_hit") else None

        print(
            f"[progress] iter={iteration} step={self.num_timesteps:,}/{self.target_timesteps:,} "
            f"({pct:5.1f}%) fps={fps} eta={_fmt_duration(eta_s)} elapsed={_fmt_duration(elapsed)} | "
            f"cpu={hw.cpu_percent:.0f}% ram={hw.ram_percent:.0f}% "
            f"gpu={_fmt_pct(hw.gpu_utilization_pct)} gpu_mem={_fmt_gb(hw.gpu_memory_used_gb)} | "
            f"reward={_fmt_num(mean_reward)} fwd_dist={_fmt_num(mean_forward)}m "
            f"landing_err={_fmt_num(mean_landing_error)}m hit_rate={_fmt_num(hit_rate)}"
        )

        self.manifest["progress"] = {
            "iteration": iteration,
            "global_timestep": self.num_timesteps,
            "percent_complete": round(pct, 2),
            "fps": fps,
            "elapsed_s": round(elapsed, 1),
            "eta_s": None if eta_s == float("inf") else round(eta_s, 1),
            "cpu_percent": hw.cpu_percent,
            "ram_percent": hw.ram_percent,
            "gpu_utilization_pct": hw.gpu_utilization_pct,
            "gpu_memory_used_gb": hw.gpu_memory_used_gb,
            "mean_reward": mean_reward,
            "mean_forward_distance_m": mean_forward,
            "mean_landing_error_m": mean_landing_error,
            "target_hit_rate": hit_rate,
            "last_checkpoint": self.last_checkpoint,
            "device": self.device,
            "n_envs": self.n_envs,
        }
        write_json_atomic(self.run_dir / "run_manifest.json", self.manifest)
        return True


def _fmt_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def _fmt_pct(value):
    return "?" if value is None else f"{value:.0f}%"


def _fmt_gb(value):
    return "?" if value is None else f"{value:.2f}GB"


def _fmt_num(value):
    return "?" if value is None else f"{value:.2f}"
