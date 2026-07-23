#!/usr/bin/env python3
"""Train a PPO policy for the Unitree G1 ball-drop task.

Usage
-----
    python RL/train.py
    python RL/train.py --timesteps 2000000 --n-envs 16 --run-name my_run

Produces, under RL/runs/<run-name>/:
    model/final_model.zip       trained SB3 PPO policy
    model/vecnormalize.pkl      observation-normalization running stats
    checkpoints/                periodic checkpoints (policy + vecnormalize)
    monitor/monitor_<i>.csv     per-worker episode logs (reward, length, success, ...)
    tensorboard/                live training curves (reward, success rate, losses)
    metadata.json               config, package/git versions, wall-clock time
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import logging

RL_ROOT = Path(__file__).resolve().parent
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from config import PPOConfig  # noqa: E402
from env_utils import make_env  # noqa: E402


def git_commit_hash(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def build_train_vec_env(cfg: PPOConfig, monitor_dir: Path):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    env_fns = [
        make_env(rank=i, seed=cfg.seed, log_dir=monitor_dir, learned_release=cfg.learned_release)
        for i in range(cfg.n_envs)
    ]
    vec_cls = SubprocVecEnv if cfg.n_envs > 1 else DummyVecEnv
    vec_env = vec_cls(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec_env


def make_success_rate_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class SuccessRateCallback(BaseCallback):
        """Logs rolling success rate / best-distance to TensorBoard.

        SB3's Monitor already records these per episode (see MONITOR_INFO_KEYWORDS
        in env_utils.py); this callback just surfaces the rolling mean the same
        way SB3 surfaces rollout/ep_rew_mean.
        """

        def _on_step(self) -> bool:
            buf = self.model.ep_info_buffer
            if buf:
                successes = [ep.get("success") for ep in buf if "success" in ep]
                best_dists = [ep.get("best_dist") for ep in buf if "best_dist" in ep]
                if successes:
                    self.logger.record("rollout/success_rate", float(np.mean(successes)))
                if best_dists:
                    self.logger.record("rollout/best_dist_mean", float(np.mean(best_dists)))
            return True

    return SuccessRateCallback()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=None, help="Defaults to a timestamped name")
    parser.add_argument("--timesteps", type=int, default=None, help="Overrides PPOConfig.total_timesteps")
    parser.add_argument("--n-envs", type=int, default=None, help="Overrides PPOConfig.n_envs")
    parser.add_argument("--seed", type=int, default=None, help="Overrides PPOConfig.seed")
    parser.add_argument("--device", default=None, help="cpu | cuda | auto")
    args = parser.parse_args()

    cfg = PPOConfig()
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device

    run_name = args.run_name or datetime.now().strftime("ppo_g1_throw_%Y%m%d_%H%M%S")
    run_dir = RL_ROOT / "runs" / run_name
    monitor_dir = run_dir / "monitor"
    checkpoint_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    model_dir = run_dir / "model"
    for d in (monitor_dir, checkpoint_dir, tb_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Prepare logging: tee stdout/stderr to a run-specific log file so the
    # entire training session (prints, SB3 output, tracebacks) is captured.
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_fh = open(run_dir / "train.log", "a")

    class _Tee:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def write(self, data):
            try:
                self.a.write(data)
            except Exception:
                pass
            try:
                self.b.write(data)
            except Exception:
                pass

        def flush(self):
            try:
                self.a.flush()
            except Exception:
                pass
            try:
                self.b.flush()
            except Exception:
                pass

    sys.stdout = _Tee(orig_stdout, log_fh)
    sys.stderr = _Tee(orig_stderr, log_fh)

    # Also configure the Python logging module to emit INFO+ to the same
    # run log (and console). SB3 uses prints for some output; logging helps
    # capture anything using the logging API.
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(orig_stdout), logging.FileHandler(run_dir / "train.log")])

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(cfg.seed)

    train_env = build_train_vec_env(cfg, monitor_dir)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.lr_schedule(),
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=cfg.policy_kwargs(),
        tensorboard_log=str(tb_dir),
        seed=cfg.seed,
        device=cfg.device,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg.checkpoint_every // cfg.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_g1_throw",
        save_vecnormalize=True,
    )
    callbacks = CallbackList([checkpoint_callback, make_success_rate_callback()])

    metadata = {
        "run_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit_hash(RL_ROOT),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config": cfg.to_dict(),
        "observation_space": str(train_env.observation_space),
        "action_space": str(train_env.action_space),
    }

    start = time.time()
    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks, tb_log_name="PPO")
    finally:
        elapsed = time.time() - start
        # restore stdout/stderr and close the run log so subsequent prints
        # go to the console as normal and the file is flushed/closed.
        try:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
        except Exception:
            pass
        try:
            log_fh.close()
        except Exception:
            pass

        model.save(str(model_dir / "final_model"))
        train_env.save(str(model_dir / "vecnormalize.pkl"))
        metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
        metadata["wall_clock_seconds"] = elapsed
        metadata["timesteps_trained"] = int(model.num_timesteps)
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (RL_ROOT / "runs" / "latest.txt").write_text(run_name)
        train_env.close()

    print(f"\nDone. Run artifacts saved under {run_dir}")
    print(f"Wall-clock training time: {elapsed:.1f}s for {model.num_timesteps} timesteps")


if __name__ == "__main__":
    main()
