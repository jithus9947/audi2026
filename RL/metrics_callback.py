"""TensorBoard logging for everything SB3 doesn't emit on its own: the
per-term reward breakdown (envs/reward_components.py, both weighted
``reward_components/*`` and unweighted ``reward_raw/*``) and the full set of
task-level throw/landing metrics needed to diagnose undershoot/overshoot,
target-hit rate, release characteristics, and termination reasons.

SB3 already logs rollout/ep_rew_mean, rollout/ep_len_mean and the PPO loss
terms (train/loss, train/value_loss, train/policy_gradient_loss,
train/entropy_loss, train/approx_kl, train/clip_fraction) plus time/fps --
this callback only adds what is missing.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

TERMINATION_REASONS = ("success", "landed", "fall", "timeout", "no_release", "out_of_bounds")


class TaskMetricsCallback(BaseCallback):
    """Aggregates per-step reward terms and per-episode task outcomes."""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._component_values: dict[str, list[float]] = {}
        self._raw_values: dict[str, list[float]] = {}
        self._episode_buffers: dict[str, deque] = {
            "forward_distance": deque(maxlen=window),
            "throw_distance": deque(maxlen=window),
            "landing_error": deque(maxlen=window),
            "longitudinal_error": deque(maxlen=window),
            "abs_longitudinal_error": deque(maxlen=window),
            "lateral_error": deque(maxlen=window),
            "abs_lateral_error": deque(maxlen=window),
            "undershoot": deque(maxlen=window),
            "overshoot": deque(maxlen=window),
            "target_hit": deque(maxlen=window),
            "within_025m": deque(maxlen=window),
            "within_050m": deque(maxlen=window),
            "within_075m": deque(maxlen=window),
            "within_100m": deque(maxlen=window),
            "within_150m": deque(maxlen=window),
            "within_200m": deque(maxlen=window),
            "released": deque(maxlen=window),
            "valid_landing": deque(maxlen=window),
            "fall": deque(maxlen=window),
            "target_distance": deque(maxlen=window),
            "release_speed": deque(maxlen=window),
            "release_forward_speed": deque(maxlen=window),
            "release_vertical_speed": deque(maxlen=window),
            "release_lateral_speed": deque(maxlen=window),
            "release_angle_deg": deque(maxlen=window),
            "flight_time": deque(maxlen=window),
            "max_ball_height": deque(maxlen=window),
            "release_desired_speed": deque(maxlen=window),
            "release_speed_deficit": deque(maxlen=window),
            "action_clip_fraction": deque(maxlen=window),
            "shoulder_saturation_rate": deque(maxlen=window),
            "elbow_saturation_rate": deque(maxlen=window),
            "wrist_saturation_rate": deque(maxlen=window),
            "hand_speed_at_release": deque(maxlen=window),
            "max_hand_speed_before_release": deque(maxlen=window),
        }
        self._termination_counts = {reason: deque(maxlen=window) for reason in TERMINATION_REASONS}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key, value in info.get("reward_components", {}).items():
                self._component_values.setdefault(key, []).append(value)
            for key, value in info.get("reward_raw", {}).items():
                self._raw_values.setdefault(key, []).append(value)

            if "episode" not in info:
                continue  # Monitor only sets this on the terminal step of an episode.

            buf = self._episode_buffers
            buf["throw_distance"].append(float(info.get("throw_distance", 0.0)))
            buf["released"].append(1.0 if info.get("ball_released") else 0.0)
            buf["fall"].append(1.0 if info.get("robot_fell") else 0.0)

            landing_error = info.get("landing_error")
            valid_landing = landing_error is not None
            buf["valid_landing"].append(1.0 if valid_landing else 0.0)
            if valid_landing:
                buf["forward_distance"].append(float(info["landing_forward_distance"]))
                buf["landing_error"].append(float(landing_error))
                buf["longitudinal_error"].append(float(info["longitudinal_error"]))
                buf["abs_longitudinal_error"].append(abs(float(info["longitudinal_error"])))
                buf["lateral_error"].append(float(info["lateral_error"]))
                buf["abs_lateral_error"].append(abs(float(info["lateral_error"])))
                buf["undershoot"].append(float(info["undershoot"]))
                buf["overshoot"].append(float(info["overshoot"]))
                buf["target_hit"].append(1.0 if info.get("square_target_hit") else 0.0)
                buf["within_025m"].append(1.0 if info.get("within_025m") else 0.0)
                buf["within_050m"].append(1.0 if info.get("within_050m") else 0.0)
                buf["within_075m"].append(1.0 if info.get("within_075m") else 0.0)
                buf["within_100m"].append(1.0 if info.get("within_100m") else 0.0)
                buf["within_150m"].append(1.0 if info.get("within_150m") else 0.0)
                buf["within_200m"].append(1.0 if info.get("within_200m") else 0.0)
                buf["target_distance"].append(float(info["target_distance_m"]))
            if info.get("release_speed") is not None:
                buf["release_speed"].append(float(info["release_speed"]))
                buf["release_forward_speed"].append(float(info["release_forward_speed"]))
                buf["release_vertical_speed"].append(float(info["release_vertical_speed"]))
                buf["release_lateral_speed"].append(float(info["release_lateral_speed"]))
                buf["release_angle_deg"].append(float(info["release_angle_deg"]))
            if info.get("release_desired_speed_mps") is not None:
                buf["release_desired_speed"].append(float(info["release_desired_speed_mps"]))
                buf["release_speed_deficit"].append(float(info["release_speed_deficit_mps"]))
            if info.get("mean_hand_speed_at_release") is not None:
                buf["hand_speed_at_release"].append(float(info["mean_hand_speed_at_release"]))
                buf["max_hand_speed_before_release"].append(float(info["max_hand_speed_before_release"]))
            if info.get("flight_time") is not None:
                buf["flight_time"].append(float(info["flight_time"]))
            buf["max_ball_height"].append(float(info.get("max_ball_height", 0.0)))
            buf["action_clip_fraction"].append(float(info.get("action_clip_fraction", 0.0)))
            buf["shoulder_saturation_rate"].append(float(info.get("shoulder_saturation_rate", 0.0)))
            buf["elbow_saturation_rate"].append(float(info.get("elbow_saturation_rate", 0.0)))
            buf["wrist_saturation_rate"].append(float(info.get("wrist_saturation_rate", 0.0)))

            reason = info.get("termination_reason")
            for r in TERMINATION_REASONS:
                self._termination_counts[r].append(1.0 if reason == r else 0.0)
        return True

    def _on_rollout_end(self) -> None:
        for key, values in self._component_values.items():
            if values:
                self.logger.record(f"reward_components/{key}", float(np.mean(values)))
        for key, values in self._raw_values.items():
            if values:
                self.logger.record(f"reward_raw/{key}", float(np.mean(values)))
        self._component_values = {}
        self._raw_values = {}

        buf = self._episode_buffers
        if buf["throw_distance"]:
            self.logger.record("task/mean_throw_distance_m", float(np.mean(buf["throw_distance"])))
        if buf["released"]:
            self.logger.record("task/release_rate", float(np.mean(buf["released"])))
        if buf["fall"]:
            self.logger.record("task/fall_rate", float(np.mean(buf["fall"])))
        if buf["valid_landing"]:
            self.logger.record("task/valid_landing_rate", float(np.mean(buf["valid_landing"])))
        if buf["forward_distance"]:
            self.logger.record("task/mean_forward_distance_m", float(np.mean(buf["forward_distance"])))
        if buf["landing_error"]:
            errs = list(buf["landing_error"])
            self.logger.record("task/mean_landing_error_m", float(np.mean(errs)))
            self.logger.record("task/median_landing_error_m", float(np.median(errs)))
            self.logger.record("task/min_landing_error_m", float(np.min(errs)))
        if buf["longitudinal_error"]:
            self.logger.record("task/mean_longitudinal_error_m", float(np.mean(buf["longitudinal_error"])))
            self.logger.record("task/mean_absolute_longitudinal_error_m", float(np.mean(buf["abs_longitudinal_error"])))
        if buf["lateral_error"]:
            self.logger.record("task/mean_lateral_error_m", float(np.mean(buf["lateral_error"])))
            self.logger.record("task/mean_absolute_lateral_error_m", float(np.mean(buf["abs_lateral_error"])))
        if buf["undershoot"]:
            self.logger.record("task/mean_undershoot_m", float(np.mean(buf["undershoot"])))
        if buf["overshoot"]:
            self.logger.record("task/mean_overshoot_m", float(np.mean(buf["overshoot"])))
        if buf["target_hit"]:
            self.logger.record("task/target_hit_rate", float(np.mean(buf["target_hit"])))
        # "025"/"050"/.../"200" are landing-error thresholds in METERS (0.25m,
        # 0.50m, 0.75m, 1.00m, 1.50m, 2.00m) -- the _error_rate suffix makes
        # that explicit so "200m" can't be misread as 200 metres.
        for tier in ("025", "050", "075", "100", "150", "200"):
            key = f"within_{tier}m"
            if buf[key]:
                self.logger.record(f"task/{key}_error_rate", float(np.mean(buf[key])))
        if buf["target_distance"]:
            self.logger.record("task/target_distance_m", float(np.mean(buf["target_distance"])))
        if buf["release_speed"]:
            self.logger.record("task/mean_release_speed_mps", float(np.mean(buf["release_speed"])))
            self.logger.record("task/mean_release_forward_speed_mps", float(np.mean(buf["release_forward_speed"])))
            self.logger.record("task/mean_release_vertical_speed_mps", float(np.mean(buf["release_vertical_speed"])))
            self.logger.record("task/mean_release_lateral_speed_mps", float(np.mean(buf["release_lateral_speed"])))
            self.logger.record("task/mean_release_angle_deg", float(np.mean(buf["release_angle_deg"])))
        if buf["flight_time"]:
            self.logger.record("task/mean_flight_time_s", float(np.mean(buf["flight_time"])))
        if buf["max_ball_height"]:
            self.logger.record("task/mean_max_ball_height_m", float(np.mean(buf["max_ball_height"])))
        if buf["release_desired_speed"]:
            self.logger.record("task/mean_release_desired_speed_mps", float(np.mean(buf["release_desired_speed"])))
            self.logger.record("task/mean_release_speed_deficit_mps", float(np.mean(buf["release_speed_deficit"])))
        if buf["hand_speed_at_release"]:
            self.logger.record("task/mean_hand_speed_at_release", float(np.mean(buf["hand_speed_at_release"])))
            self.logger.record("task/max_hand_speed_before_release", float(np.mean(buf["max_hand_speed_before_release"])))
        self.logger.record("task/action_clip_fraction", float(np.mean(buf["action_clip_fraction"])) if buf["action_clip_fraction"] else 0.0)
        self.logger.record("task/shoulder_saturation_rate", float(np.mean(buf["shoulder_saturation_rate"])) if buf["shoulder_saturation_rate"] else 0.0)
        self.logger.record("task/elbow_saturation_rate", float(np.mean(buf["elbow_saturation_rate"])) if buf["elbow_saturation_rate"] else 0.0)
        self.logger.record("task/wrist_saturation_rate", float(np.mean(buf["wrist_saturation_rate"])) if buf["wrist_saturation_rate"] else 0.0)

        for reason, values in self._termination_counts.items():
            if values:
                self.logger.record(f"task/termination_{reason}", float(np.mean(values)))
