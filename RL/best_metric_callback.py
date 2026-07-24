"""Deterministic periodic evaluation that tracks *several* "best" models.

SB3's built-in EvalCallback only tracks best-by-mean-reward. The task here
has several outcomes worth tracking independently (a policy that throws far
but inaccurately vs. one that's accurate but short are both "improving" in
different senses), so this runs its own small deterministic eval sweep every
``eval_every_iterations`` and, whenever a tracked metric improves, saves a
new immutable timestamped copy (``best_<metric>_<timestamp>_value<v>_steps<s>.zip``)
plus updates a convenience alias (``best_<metric>_model.zip``) that always
points at the current best. The alias is overwritten; the timestamped copies
never are -- so nothing is ever silently lost even though there's a
"latest-best" file that's easy to point evaluation scripts at.

In addition to the 4 independently-tracked metrics, ``best_overall`` selects
ONE model using the exact lexicographic priority requested for the "real"
best model: highest target-hit rate, then lowest landing error, then lowest
longitudinal error, then lowest fall rate, then highest valid-landing rate,
then lowest lateral error. This is deliberately NOT ep_rew_mean -- reward can
climb from stability terms alone even while landing accuracy is flat (this is
exactly what happened in the run that plateaued at ~3.1 m).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from RL.run_management import atomic_model_save, best_model_filename, timestamp_now

EVAL_CSV_COLUMNS = [
    "timestamp", "global_step", "iteration", "mean_reward", "mean_landing_error_m",
    "target_hit_rate", "mean_forward_distance_m", "mean_abs_longitudinal_error_m",
    "mean_abs_lateral_error_m", "fall_rate", "valid_landing_rate", "n_episodes",
]

# (metric key, "higher is better"?)
TRACKED_METRICS = (
    ("reward", True),
    ("landing_error", False),
    ("target_hit_rate", True),
    ("forward_distance", True),
)


def _overall_priority_key(values: dict) -> tuple:
    """Lower is better in every component (tuple comparison is lexicographic,
    so this directly encodes: hit_rate desc, landing_error asc, longitudinal
    error asc, fall_rate asc, valid_landing_rate desc, lateral_error asc)."""
    return (
        -values["target_hit_rate"],
        values["landing_error"],
        values["abs_longitudinal_error"],
        values["fall_rate"],
        -values["valid_landing_rate"],
        values["abs_lateral_error"],
    )


class BestMetricEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env_fn: Callable[[], object],
        save_dir: Path,
        eval_every_iterations: int,
        n_eval_episodes: int = 50,
        seed: int = 10_000,
        eval_csv_path: Path | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.save_dir = Path(save_dir)
        self.eval_csv_path = Path(eval_csv_path) if eval_csv_path else self.save_dir.parent / "csv" / "evaluations.csv"
        self.eval_every_iterations = max(eval_every_iterations, 1)
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.best = {
            "reward": -np.inf,
            "landing_error": np.inf,
            "target_hit_rate": -np.inf,
            "forward_distance": -np.inf,
        }
        self.best_overall_key: tuple | None = None
        self.best_paths: dict[str, Path] = {}

    def _on_step(self) -> bool:
        n_steps = getattr(self.model, "n_steps", 1) or 1
        eval_every_calls = self.eval_every_iterations * n_steps
        if self.n_calls % eval_every_calls != 0:
            return True
        self._run_eval()
        return True

    def _run_eval(self) -> None:
        env = self.eval_env_fn()
        rewards, landing_errors, hits, forward_distances = [], [], [], []
        abs_longitudinal_errors, abs_lateral_errors, falls, valid_landings = [], [], [], []
        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset(seed=self.seed + ep)
            total_reward = 0.0
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
            falls.append(1.0 if info.get("robot_fell") else 0.0)
            valid = info.get("landing_error") is not None
            valid_landings.append(1.0 if valid else 0.0)
            if valid:
                landing_errors.append(info["landing_error"])
                forward_distances.append(info["landing_forward_distance"])
                hits.append(1.0 if info.get("square_target_hit") else 0.0)
                abs_longitudinal_errors.append(abs(info["longitudinal_error"]))
                abs_lateral_errors.append(abs(info["lateral_error"]))

        values = {
            "reward": float(np.mean(rewards)) if rewards else -np.inf,
            "landing_error": float(np.mean(landing_errors)) if landing_errors else np.inf,
            "target_hit_rate": float(np.mean(hits)) if hits else 0.0,
            "forward_distance": float(np.mean(forward_distances)) if forward_distances else 0.0,
            "abs_longitudinal_error": float(np.mean(abs_longitudinal_errors)) if abs_longitudinal_errors else np.inf,
            "abs_lateral_error": float(np.mean(abs_lateral_errors)) if abs_lateral_errors else np.inf,
            "fall_rate": float(np.mean(falls)) if falls else 1.0,
            "valid_landing_rate": float(np.mean(valid_landings)) if valid_landings else 0.0,
        }
        for key in ("reward", "landing_error", "target_hit_rate", "forward_distance", "fall_rate", "valid_landing_rate"):
            tag = key if key in ("target_hit_rate", "fall_rate", "valid_landing_rate") else f"mean_{key}"
            self.logger.record(f"eval/{tag}", values[key])

        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = timestamp_now()

        self.eval_csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.eval_csv_path.exists()
        with open(self.eval_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_CSV_COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "global_step": self.num_timesteps,
                    "iteration": self.n_calls // (getattr(self.model, "n_steps", 1) or 1),
                    "mean_reward": values["reward"],
                    "mean_landing_error_m": values["landing_error"],
                    "target_hit_rate": values["target_hit_rate"],
                    "mean_forward_distance_m": values["forward_distance"],
                    "mean_abs_longitudinal_error_m": values["abs_longitudinal_error"],
                    "mean_abs_lateral_error_m": values["abs_lateral_error"],
                    "fall_rate": values["fall_rate"],
                    "valid_landing_rate": values["valid_landing_rate"],
                    "n_episodes": len(rewards),
                }
            )

        for metric, higher_is_better in TRACKED_METRICS:
            value = values[metric]
            improved = value > self.best[metric] if higher_is_better else value < self.best[metric]
            if not improved:
                continue
            self.best[metric] = value
            timestamped_name = best_model_filename(metric, timestamp, value, self.num_timesteps)
            timestamped_path = atomic_model_save(self.model, self.save_dir / timestamped_name)
            alias_path = atomic_model_save(self.model, self.save_dir / f"best_{metric}_model.zip")
            self.best_paths[metric] = timestamped_path
            if self.verbose:
                print(f"[best-model] new best {metric}={value:.3f} -> {timestamped_path.name} (alias {alias_path.name} updated)")

        overall_key = _overall_priority_key(values)
        if self.best_overall_key is None or overall_key < self.best_overall_key:
            self.best_overall_key = overall_key
            timestamped_name = best_model_filename("overall", timestamp, values["landing_error"], self.num_timesteps)
            timestamped_path = atomic_model_save(self.model, self.save_dir / timestamped_name)
            alias_path = atomic_model_save(self.model, self.save_dir / "best_overall_model.zip")
            self.best_paths["overall"] = timestamped_path
            if self.verbose:
                print(
                    f"[best-model] new best_overall (hit_rate={values['target_hit_rate']:.2f}, "
                    f"landing_error={values['landing_error']:.2f}, fall_rate={values['fall_rate']:.2f}) "
                    f"-> {timestamped_path.name} (alias {alias_path.name} updated)"
                )

        if self.verbose:
            print(
                f"[eval] reward={values['reward']:.2f} landing_error={values['landing_error']:.2f}m "
                f"hit_rate={values['target_hit_rate']:.2f} forward_distance={values['forward_distance']:.2f}m "
                f"fall_rate={values['fall_rate']:.2f} valid_landing_rate={values['valid_landing_rate']:.2f}"
            )
