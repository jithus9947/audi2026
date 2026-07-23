#!/usr/bin/env python3
"""Visualize a trained PPO policy in the MuJoCo viewer.

Mirrors scripts/play_baseline_g1_throw.py but drives the arm with the trained
PPO policy instead of the scripted baseline.

Usage
-----
    python RL/play_policy.py                              # most recently trained run
    python RL/play_policy.py --run-name ppo_g1_throw_main
    python RL/play_policy.py --deterministic false          # sample instead of argmax-mean action
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer

RL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RL_ROOT.parent
for p in (PROJECT_ROOT, RL_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=None, help="Defaults to RL/runs/latest.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", type=str, default="true", choices=["true", "false"])
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    run_name = args.run_name or (RL_ROOT / "runs" / "latest.txt").read_text().strip()
    run_dir = RL_ROOT / "runs" / run_name
    model_path = run_dir / "model" / "final_model.zip"
    vecnorm_path = run_dir / "model" / "vecnormalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run RL/train.py first.")

    env = G1FixedBodyThrowEnv(learned_release=True)

    # VecNormalize is only used here as a holder for the training-time obs
    # running stats (mean/var) -- we step `env` directly for the viewer, we
    # never call vec_env.step()/reset().
    vec_env = VecNormalize.load(str(vecnorm_path), DummyVecEnv([lambda: env]))
    vec_env.training = False

    model = PPO.load(str(model_path), device="cpu")
    deterministic = args.deterministic == "true"

    obs, _ = env.reset(seed=args.seed)
    print(f"Policy viewer running (run='{run_name}'). Close the MuJoCo window or press Ctrl+C to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            norm_obs = vec_env.normalize_obs(obs)
            action, _ = model.predict(norm_obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(env.control_dt)

            if terminated or truncated:
                episode += 1
                print(
                    f"episode {episode}: success={info['success']} "
                    f"landing_error={info['landing_error']} "
                    f"best_dist={info['best_dist']:.3f} m "
                    f"release={info['release_time']}"
                )
                obs, _ = env.reset(seed=args.seed + episode)


if __name__ == "__main__":
    main()
