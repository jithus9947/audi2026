#!/usr/bin/env python3
"""Evaluate a trained PPO ball-drop policy with the baseline metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv


def evaluate(model_path: Path, episodes: int = 20, base_seed: int = 42):
    model = PPO.load(str(model_path))
    successes = falls = 0
    landing_errors, completion_times, smoothness = [], [], []

    for episode in range(episodes):
        env = G1FixedBodyThrowEnv(learned_release=True)
        obs, _ = env.reset(seed=base_seed + episode)
        previous_action = np.zeros(env.action_space.shape, dtype=np.float32)
        action_changes = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            action_changes.append(float(np.linalg.norm(action - previous_action)))
            previous_action = action.copy()
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        successes += int(info['success'])
        falls += int(info['robot_fell'])
        smoothness.append(float(np.mean(action_changes)))
        if info['landing_error'] is not None:
            landing_errors.append(info['landing_error'])
        if info['completion_time'] is not None:
            completion_times.append(info['completion_time'])
        landing = f"{info['landing_error']:.3f} m" if info['landing_error'] is not None else 'no landing'
        print(
            f"Episode {episode + 1}/{episodes}: success={info['success']}, "
            f"landing error={landing}, fell={info['robot_fell']}",
            flush=True,
        )

    print('PPO EVALUATION SUMMARY')
    print(f'Episodes: {episodes}')
    print(f'Success rate: {100 * successes / episodes:.1f}%')
    print(f'Average landing error: {np.mean(landing_errors):.3f} m' if landing_errors else 'Average landing error: no landings')
    print(f'Average completion time: {np.mean(completion_times):.3f} s' if completion_times else 'Average completion time: no completions')
    print(f'Robot fell: {falls}/{episodes}')
    print(f'Action smoothness: {np.mean(smoothness):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT / 'policies' / 'g1_ball_drop_ppo' / 'best_model.zip')
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
