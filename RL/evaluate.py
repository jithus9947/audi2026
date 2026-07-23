#!/usr/bin/env python3
"""Evaluate a trained PPO policy against the deterministic baseline controller.

Both controllers are rolled out on the identical G1FixedBodyThrowEnv, with the
same per-episode seeds and the same metric set, so the comparison is
apples-to-apples:

    - success rate
    - average episode reward (raw env reward, not VecNormalize-scaled)
    - average landing error (only episodes that actually land the ball)
    - average best XY distance reached during the episode
    - average release time (only episodes that release the ball)
    - average action smoothness: mean ||a_t - a_{t-1}|| across the episode,
      the metric the project README flags as missing before a final PPO
      comparison. The env is fixed-body (no legs/free root joint -- see
      assets/unitree_g1/scene_throw.xml), so a "fall" metric does not apply
      to this task; that is noted explicitly in RL/README.md instead of
      being faked here.

Usage
-----
    python RL/evaluate.py --run-name ppo_g1_throw_20260722_180000 --num-episodes 50
    python RL/evaluate.py  # uses the most recently trained run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RL_ROOT.parent
for p in (PROJECT_ROOT, RL_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv  # noqa: E402
from baselines.baseline_controller import BaselineController  # noqa: E402
from env_utils import make_env  # noqa: E402


def _episode_record(reward: float, info: dict, smooth_deltas: list) -> dict:
    return dict(
        reward=float(reward),
        success=bool(info.get("success", False)),
        landing_error=info.get("landing_error"),
        best_dist=float(info.get("best_dist", float("nan"))),
        release_time=info.get("release_time"),
        released=bool(info.get("released", False)),
        action_smoothness=float(np.mean(smooth_deltas)) if smooth_deltas else 0.0,
    )


def evaluate_baseline(num_episodes: int, base_seed: int) -> list:
    env = G1FixedBodyThrowEnv(learned_release=True)
    controller = BaselineController(env.n_arm)
    episodes = []
    for ep in range(num_episodes):
        env.reset(seed=base_seed + ep)
        done = False
        ep_reward = 0.0
        prev_action = np.zeros(env.n_arm + 1)
        smooth_deltas = []
        info = {}
        while not done:
            t = env.step_count * env.control_dt
            action = controller.act(t)
            smooth_deltas.append(float(np.linalg.norm(action - prev_action)))
            prev_action = action
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        episodes.append(_episode_record(ep_reward, info, smooth_deltas))
    return episodes


def evaluate_ppo(model_path: Path, vecnorm_path: Path, num_episodes: int, base_seed: int) -> list:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env_fn = make_env(rank=0, seed=base_seed, log_dir=None, learned_release=True, monitor=False)
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(str(model_path), device="cpu")

    action_dim = vec_env.action_space.shape[0]
    episodes = []
    for ep in range(num_episodes):
        vec_env.seed(base_seed + ep)
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        prev_action = np.zeros(action_dim)
        smooth_deltas = []
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            smooth_deltas.append(float(np.linalg.norm(action[0] - prev_action)))
            prev_action = action[0]
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += float(reward[0])
            done = bool(dones[0])
            info = infos[0]
        episodes.append(_episode_record(ep_reward, info, smooth_deltas))
    vec_env.close()
    return episodes


def aggregate_metrics(episodes: list) -> dict:
    n = len(episodes)
    rewards = [e["reward"] for e in episodes]
    successes = [e["success"] for e in episodes]
    best_dists = [e["best_dist"] for e in episodes]
    smoothness = [e["action_smoothness"] for e in episodes]
    landing_errors = [e["landing_error"] for e in episodes if e["landing_error"] is not None]
    release_times = [e["release_time"] for e in episodes if e["release_time"] is not None]
    return dict(
        num_episodes=n,
        success_rate=float(np.mean(successes)),
        avg_reward=float(np.mean(rewards)),
        avg_best_dist=float(np.mean(best_dists)),
        avg_landing_error=float(np.mean(landing_errors)) if landing_errors else None,
        avg_release_time=float(np.mean(release_times)) if release_times else None,
        avg_action_smoothness=float(np.mean(smoothness)),
        episodes=episodes,
    )


def _fmt(x, suffix=""):
    return "n/a" if x is None else f"{x:.3f}{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=None, help="Defaults to RL/runs/latest.txt")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000, help="Base eval seed, offset from the training seed")
    args = parser.parse_args()

    run_name = args.run_name or (RL_ROOT / "runs" / "latest.txt").read_text().strip()
    run_dir = RL_ROOT / "runs" / run_name
    model_path = run_dir / "model" / "final_model.zip"
    vecnorm_path = run_dir / "model" / "vecnormalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run RL/train.py first.")

    print(f"Evaluating baseline controller ({args.num_episodes} episodes)...")
    baseline_metrics = aggregate_metrics(evaluate_baseline(args.num_episodes, args.seed))

    print(f"Evaluating PPO policy from run '{run_name}' ({args.num_episodes} episodes)...")
    ppo_metrics = aggregate_metrics(evaluate_ppo(model_path, vecnorm_path, args.num_episodes, args.seed))

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison = dict(
        run_name=run_name,
        num_episodes=args.num_episodes,
        seed=args.seed,
        baseline=baseline_metrics,
        ppo=ppo_metrics,
    )
    (results_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    print("\n" + "=" * 62)
    print("BASELINE vs PPO")
    print("=" * 62)
    print(f"{'metric':<26}{'baseline':>16}{'ppo':>16}")
    print(f"{'success rate':<26}{baseline_metrics['success_rate']*100:>15.1f}%{ppo_metrics['success_rate']*100:>15.1f}%")
    print(f"{'avg reward':<26}{baseline_metrics['avg_reward']:>16.3f}{ppo_metrics['avg_reward']:>16.3f}")
    print(f"{'avg landing error (m)':<26}{_fmt(baseline_metrics['avg_landing_error']):>16}{_fmt(ppo_metrics['avg_landing_error']):>16}")
    print(f"{'avg best dist (m)':<26}{baseline_metrics['avg_best_dist']:>16.3f}{ppo_metrics['avg_best_dist']:>16.3f}")
    print(f"{'avg release time (s)':<26}{_fmt(baseline_metrics['avg_release_time']):>16}{_fmt(ppo_metrics['avg_release_time']):>16}")
    print(f"{'avg action smoothness':<26}{baseline_metrics['avg_action_smoothness']:>16.4f}{ppo_metrics['avg_action_smoothness']:>16.4f}")
    print(f"\nSaved to {results_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
