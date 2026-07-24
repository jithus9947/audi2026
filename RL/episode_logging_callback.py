"""Per-episode CSV logging and optional one-line-per-throw debug printing.

One row is appended per completed episode (any of the parallel envs), using
the rich info dict envs/g1_target_throw_env.py populates at episode end.
Kept as a single callback (rather than a separate CSV file + separate debug
file) since both just react to the same "episode ended" event; --debug-events
in RL/train_target_throw.py flips ``print_debug`` on. Printing is off by
default because per-episode I/O across 16 parallel envs measurably drops FPS.
"""
from __future__ import annotations

import csv
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

CSV_COLUMNS = [
    "global_step",
    "ppo_iteration",
    "episode_number",
    "environment_id",
    "target_distance_m",
    "episode_reward",
    "episode_length",
    "termination_reason",
    "ball_released",
    "release_position_x",
    "release_position_y",
    "release_position_z",
    "release_velocity_x",
    "release_velocity_y",
    "release_velocity_z",
    "release_speed_mps",
    "release_forward_speed_mps",
    "release_vertical_speed_mps",
    "release_lateral_speed_mps",
    "release_angle_deg",
    "flight_time_s",
    "max_ball_height_m",
    "valid_landing",
    "landing_x",
    "landing_y",
    "landing_z",
    "landing_forward_distance_m",
    "longitudinal_error_m",
    "lateral_error_m",
    "landing_error_m",
    "undershoot_m",
    "overshoot_m",
    "target_hit",
    "within_025m",
    "within_050m",
    "within_075m",
    "within_100m",
    "within_150m",
    "within_200m",
    "pelvis_error",
    "feet_contact_valid",
    "self_collision_count",
    "bad_ground_contact",
    "fall_detected",
]


class EpisodeLoggingCallback(BaseCallback):
    def __init__(self, csv_path: Path, print_debug: bool = False, flush_every: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.print_debug = print_debug
        self.flush_every = flush_every
        self._episode_number = 0
        self._rows_since_flush = 0
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.csv_path.exists()
        self._file = open(self.csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        if is_new:
            self._writer.writeheader()
            self._file.flush()

    def _on_step(self) -> bool:
        n_steps = getattr(self.model, "n_steps", 1) or 1
        iteration = self.n_calls // n_steps
        for env_id, info in enumerate(self.locals.get("infos", [])):
            if "episode" not in info:
                continue
            self._episode_number += 1
            row = {
                "global_step": self.num_timesteps,
                "ppo_iteration": iteration,
                "episode_number": self._episode_number,
                "environment_id": env_id,
                "target_distance_m": info.get("target_distance_m"),
                "episode_reward": info["episode"]["r"],
                "episode_length": info["episode"]["l"],
                "termination_reason": info.get("termination_reason"),
                "ball_released": info.get("ball_released"),
                "release_position_x": _xyz(info.get("release_position"), 0),
                "release_position_y": _xyz(info.get("release_position"), 1),
                "release_position_z": _xyz(info.get("release_position"), 2),
                "release_velocity_x": _xyz(info.get("release_velocity"), 0),
                "release_velocity_y": _xyz(info.get("release_velocity"), 1),
                "release_velocity_z": _xyz(info.get("release_velocity"), 2),
                "release_speed_mps": info.get("release_speed"),
                "release_forward_speed_mps": info.get("release_forward_speed"),
                "release_vertical_speed_mps": info.get("release_vertical_speed"),
                "release_lateral_speed_mps": info.get("release_lateral_speed"),
                "release_angle_deg": info.get("release_angle_deg"),
                "flight_time_s": info.get("flight_time"),
                "max_ball_height_m": info.get("max_ball_height"),
                "valid_landing": info.get("landing_error") is not None,
                "landing_x": _xyz(info.get("landing_position"), 0),
                "landing_y": _xyz(info.get("landing_position"), 1),
                "landing_z": info.get("landing_z"),
                "landing_forward_distance_m": info.get("landing_forward_distance"),
                "longitudinal_error_m": info.get("longitudinal_error"),
                "lateral_error_m": info.get("lateral_error"),
                "landing_error_m": info.get("landing_error"),
                "undershoot_m": info.get("undershoot"),
                "overshoot_m": info.get("overshoot"),
                "target_hit": info.get("square_target_hit"),
                "within_025m": info.get("within_025m"),
                "within_050m": info.get("within_050m"),
                "within_075m": info.get("within_075m"),
                "within_100m": info.get("within_100m"),
                "within_150m": info.get("within_150m"),
                "within_200m": info.get("within_200m"),
                "pelvis_error": info.get("pelvis_position_drift_m"),
                "feet_contact_valid": info.get("reward_raw", {}).get("feet_contact", 0.0) > 0.0,
                "self_collision_count": info.get("self_collision_count"),
                "bad_ground_contact": info.get("reward_raw", {}).get("bad_ground_contact", 0.0) < 0.0,
                "fall_detected": info.get("robot_fell"),
            }
            self._writer.writerow(row)
            self._rows_since_flush += 1
            if self._rows_since_flush >= self.flush_every:
                self._file.flush()
                self._rows_since_flush = 0

            if self.print_debug:
                landing = (row["landing_x"], row["landing_y"])
                print(
                    f"env={env_id} released={row['ball_released']} "
                    f"release_speed={_fmt(row['release_speed_mps'])} "
                    f"release_angle={_fmt(row['release_angle_deg'])} "
                    f"landing={landing} "
                    f"forward_distance={_fmt(row['landing_forward_distance_m'])} "
                    f"longitudinal_error={_fmt(row['longitudinal_error_m'])} "
                    f"lateral_error={_fmt(row['lateral_error_m'])} "
                    f"landing_error={_fmt(row['landing_error_m'])} "
                    f"target_hit={row['target_hit']} "
                    f"termination={row['termination_reason']}"
                )
        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()


def _xyz(vec, index):
    if vec is None:
        return None
    return vec[index]


def _fmt(value):
    if value is None:
        return "None"
    return f"{value:.3f}"
