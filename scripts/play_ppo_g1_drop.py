#!/usr/bin/env python3
"""Play a trained PPO G1 ball-drop policy in the MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "policies" / "g1_ball_drop_ppo" / "best_model.zip",
        help="Path to the trained PPO .zip model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier; use 0.5 for slow motion.",
    )
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be greater than zero")
    if not args.model.is_file():
        parser.error(f"PPO model not found: {args.model}. Train it first or pass --model PATH.")

    model = PPO.load(str(args.model))
    env = G1FixedBodyThrowEnv(learned_release=True)
    obs, _ = env.reset(seed=args.seed)
    episode = 1
    print("PPO viewer running. Close the MuJoCo window or press Ctrl+C to stop.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(env.control_dt / args.speed)

            if terminated or truncated:
                landing = info["landing_error"]
                result = f"{landing:.3f} m" if landing is not None else "no landing"
                print(
                    f"Episode {episode}: success={info['success']}, "
                    f"landing error={result}, fell={info['robot_fell']}"
                )
                episode += 1
                obs, _ = env.reset(seed=args.seed + episode - 1)


if __name__ == "__main__":
    main()
