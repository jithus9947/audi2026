#!/usr/bin/env python3
"""Train a PPO policy for the Unitree G1 ball-drop task."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv


def make_drop_env():
    return G1FixedBodyThrowEnv(learned_release=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-envs', type=int, default=1)
    parser.add_argument('--output', type=Path, default=ROOT / 'policies' / 'g1_ball_drop_ppo')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train_env = make_vec_env(make_drop_env, n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_env(make_drop_env, n_envs=1, seed=args.seed + 10_000)
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output),
        log_path=str(args.output / 'evaluations'),
        eval_freq=max(5_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.output / 'tensorboard'),
    )
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    model.save(str(args.output / 'final_model'))
    print(f'PPO training complete. Models saved in: {args.output}')


if __name__ == '__main__':
    main()
