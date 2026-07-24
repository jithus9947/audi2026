#!/usr/bin/env python3
"""Deterministic evaluation of a trained target-throw PPO policy.

Runs N episodes with a fixed target distance (must match what the model was
trained on -- this script does not implement or allow a distance curriculum)
and reports the same task metrics TensorBoard sees during training: throw
distance, landing error, longitudinal/lateral error, undershoot/overshoot,
target-hit rate, and the accuracy-tier rates.

    python RL/evaluate_throw.py --model RL/runs/target_throw/final_model.zip --episodes 100
    python RL/evaluate_throw.py --model RL/runs/target_throw_5m_finetune/best_landing_error_model.zip \\
        --episodes 20 --render --debug-events
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO

from envs.g1_target_throw_env import G1TargetThrowEnv


def _fmt(value):
    return "None" if value is None else f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--target-half-size", type=float, default=0.35)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--render", action="store_true", help="Show the MuJoCo window, one episode at a time, real-time speed.")
    parser.add_argument("--debug-events", action="store_true", help="Print one line per episode.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")

    env = G1TargetThrowEnv(
        learned_release=True,
        target_pos=(args.target_distance, 0.0, 0.01),
        target_half_size=args.target_half_size,
        verbose_init=True,
    )
    model = PPO.load(str(args.model), device=args.device)

    viewer_ctx = None
    if args.render:
        import mujoco.viewer

        viewer_ctx = mujoco.viewer.launch_passive(env.model, env.data)

    metrics = {
        "reward": [], "throw_distance": [], "landing_error": [], "longitudinal_error": [],
        "lateral_error": [], "undershoot": [], "overshoot": [], "hit": [], "released": [], "fell": [],
        "within_025m": [], "within_050m": [], "within_075m": [], "within_100m": [], "within_150m": [], "within_200m": [],
    }
    termination_counts: dict[str, int] = {}

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if viewer_ctx is not None:
                    viewer_ctx.sync()
                    time.sleep(env.control_dt)

            metrics["reward"].append(total_reward)
            metrics["throw_distance"].append(info["throw_distance"])
            metrics["released"].append(1.0 if info["ball_released"] else 0.0)
            metrics["fell"].append(1.0 if info.get("robot_fell") else 0.0)
            reason = info.get("termination_reason") or "unknown"
            termination_counts[reason] = termination_counts.get(reason, 0) + 1

            if info.get("landing_error") is not None:
                metrics["landing_error"].append(info["landing_error"])
                metrics["longitudinal_error"].append(info["longitudinal_error"])
                metrics["lateral_error"].append(info["lateral_error"])
                metrics["undershoot"].append(info["undershoot"])
                metrics["overshoot"].append(info["overshoot"])
                metrics["hit"].append(1.0 if info["square_target_hit"] else 0.0)
                for tier in ("025", "050", "075", "100", "150", "200"):
                    metrics[f"within_{tier}m"].append(1.0 if info[f"within_{tier}m"] else 0.0)

            if args.debug_events:
                landing = info.get("landing_position")
                print(
                    f"episode={episode} released={info['ball_released']} "
                    f"release_speed={_fmt(info['release_speed'])} release_angle={_fmt(info['release_angle_deg'])} "
                    f"landing={landing} forward_distance={_fmt(info['landing_forward_distance'])} "
                    f"longitudinal_error={_fmt(info['longitudinal_error'])} lateral_error={_fmt(info['lateral_error'])} "
                    f"landing_error={_fmt(info['landing_error'])} target_hit={info['square_target_hit']} "
                    f"termination={reason}"
                )
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    n = args.episodes
    print("\n=== Evaluation summary ===")
    print(f"model: {args.model}")
    print(f"episodes: {n}  target_distance: {args.target_distance} m")
    print(f"mean_reward: {np.mean(metrics['reward']):.2f}")
    print(f"release_rate: {np.mean(metrics['released']):.2f}")
    print(f"fall_rate: {np.mean(metrics['fell']):.2f}")
    print(f"mean_throw_distance_m: {np.mean(metrics['throw_distance']):.3f}")
    valid = len(metrics["landing_error"])
    print(f"valid_landing_rate: {valid / n:.2f}")
    if valid:
        print(f"mean_landing_error_m: {np.mean(metrics['landing_error']):.3f}")
        print(f"median_landing_error_m: {np.median(metrics['landing_error']):.3f}")
        print(f"mean_longitudinal_error_m: {np.mean(metrics['longitudinal_error']):.3f}")
        print(f"mean_lateral_error_m: {np.mean(metrics['lateral_error']):.3f}")
        print(f"mean_undershoot_m: {np.mean(metrics['undershoot']):.3f}")
        print(f"mean_overshoot_m: {np.mean(metrics['overshoot']):.3f}")
        print(f"target_hit_rate: {np.mean(metrics['hit']):.2f}")
        for tier in ("025", "050", "075", "100", "150", "200"):
            print(f"within_{tier}m_rate: {np.mean(metrics[f'within_{tier}m']):.2f}")
    print(f"termination_reasons: {termination_counts}")


if __name__ == "__main__":
    main()
