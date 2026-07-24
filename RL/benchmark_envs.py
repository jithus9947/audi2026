#!/usr/bin/env python3
"""Benchmark parallel-environment counts end to end (real PPO rollout collection
+ real PPO update, not just raw env.step() throughput) and recommend the
fastest stable configuration.

Never assumes more environments is faster: each candidate count is measured
independently, a failing/OOM/crashing config is caught and reported without
aborting the rest of the sweep, and the recommendation is whichever
successful config had the highest measured end-to-end FPS.

    python RL/benchmark_envs.py --benchmark-envs 8 16 24 32 --benchmark-iterations 10 --device auto

Also invoked via ``python RL/train_target_throw.py --benchmark-envs 8 16 24 32``.
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.g1_target_throw_env import G1TargetThrowEnv
from RL.hardware_diagnostics import sample_hardware, select_device


def valid_batch_size(rollout_size: int, preferred=(512, 256, 128, 64)) -> int:
    """Largest preferred batch size that divides the rollout buffer evenly.

    n_steps and n_envs are always divisors of rollout_size = n_steps * n_envs,
    so this always terminates with a clean divisor (worst case rollout_size
    itself, i.e. one giant minibatch).
    """
    for candidate in preferred:
        if rollout_size % candidate == 0:
            return candidate
    return rollout_size


def _make_env(rank: int, seed: int, target_distance: float, target_half_size: float):
    def _init():
        env = G1TargetThrowEnv(
            learned_release=True,
            target_pos=(target_distance, 0.0, 0.01),
            target_half_size=target_half_size,
            verbose_init=False,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def run_single_benchmark(
    n_envs: int,
    n_steps: int,
    iterations: int,
    device: str,
    target_distance: float,
    target_half_size: float,
    seed: int,
    warmup_iterations: int = 1,
    start_method: str | None = None,
) -> dict:
    result = {"n_envs": n_envs, "n_steps": n_steps, "success": False}
    vec_env = None
    try:
        env_fns = [_make_env(i, seed, target_distance, target_half_size) for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
        rollout_size = n_steps * n_envs
        batch_size = valid_batch_size(rollout_size)
        model = PPO(
            "MlpPolicy", vec_env, n_steps=n_steps, batch_size=batch_size,
            device=device, verbose=0, seed=seed,
        )

        model.learn(total_timesteps=warmup_iterations * rollout_size)  # warm-up, not timed

        start = time.time()
        model.learn(total_timesteps=iterations * rollout_size, reset_num_timesteps=False)
        elapsed = time.time() - start

        hw = sample_hardware()
        timed_timesteps = iterations * rollout_size
        result.update(
            success=True,
            rollout_size=rollout_size,
            batch_size=batch_size,
            elapsed_s=round(elapsed, 2),
            fps=round(timed_timesteps / elapsed, 1) if elapsed > 0 else 0.0,
            timed_timesteps=timed_timesteps,
            cpu_percent=hw.cpu_percent,
            ram_percent=hw.ram_percent,
            gpu_utilization_pct=hw.gpu_utilization_pct,
            gpu_memory_used_gb=hw.gpu_memory_used_gb,
        )
    except Exception as exc:  # noqa: BLE001 - a failing config must not abort the sweep
        result["error"] = f"{type(exc).__name__}: {exc}"
        if warmup_iterations:  # only print full traceback in verbose/manual runs
            traceback.print_exc(limit=3)
    finally:
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:
                pass
    return result


def benchmark_env_counts(
    env_counts: list[int],
    n_steps: int,
    iterations: int,
    device: str,
    target_distance: float,
    target_half_size: float,
    seed: int,
) -> tuple[list[dict], dict | None]:
    results = []
    for n_envs in env_counts:
        print(f"\n--- Benchmarking n_envs={n_envs} (n_steps={n_steps}, {iterations} timed iterations) ---")
        result = run_single_benchmark(n_envs, n_steps, iterations, device, target_distance, target_half_size, seed)
        results.append(result)
        if result["success"]:
            print(
                f"  OK  fps={result['fps']:.0f}  rollout={result['rollout_size']} "
                f"batch={result['batch_size']}  cpu={result['cpu_percent']:.0f}%  ram={result['ram_percent']:.0f}%  "
                f"gpu={result['gpu_utilization_pct']}  gpu_mem={result['gpu_memory_used_gb']}GB"
            )
        else:
            print(f"  FAILED: {result['error']}")

    successful = [r for r in results if r["success"]]
    recommended = max(successful, key=lambda r: r["fps"]) if successful else None

    print("\n=== Benchmark summary ===")
    for r in results:
        if r["success"]:
            print(f"envs={r['n_envs']:<4} fps={r['fps']:<10.0f}cpu={r['cpu_percent']:<6.0f}ram={r['ram_percent']:<6.0f}gpu={r['gpu_utilization_pct']}")
        else:
            print(f"envs={r['n_envs']:<4} FAILED ({r['error']})")
    if recommended:
        print(f"\nrecommended_envs={recommended['n_envs']}  (measured fps={recommended['fps']:.0f})")
    else:
        print("\nAll configurations failed -- check the errors above.")
    return results, recommended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-envs", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--benchmark-iterations", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--target-half-size", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-method", default=None, choices=[None, "fork", "spawn", "forkserver"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"Benchmarking on device={device}, envs={args.benchmark_envs}, {args.benchmark_iterations} timed iterations each.")
    results, recommended = benchmark_env_counts(
        args.benchmark_envs, args.n_steps, args.benchmark_iterations, device,
        args.target_distance, args.target_half_size, args.seed,
    )
    if recommended:
        print(
            "\nSuggested production command:\n"
            f"  python RL/train_target_throw.py --n-envs {recommended['n_envs']} --n-steps {args.n_steps} "
            f"--batch-size {recommended['batch_size']} --device {device} --iterations 1000 --checkpoint-every 100"
        )


if __name__ == "__main__":
    main()
