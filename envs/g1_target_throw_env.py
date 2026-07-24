"""G1 task: Level 0 (fixed pelvis) + Level 1 option 1 (hit a marked target).

Subclasses G1FixedBodyThrowEnv (envs/g1_fixed_body_throw_env.py) without
touching it or the scene XML -- exactly the pattern already used by
envs/g1_throw_distance_env.py. Only __init__, reset(), step() and
_compute_reward() are overridden here; everything else (physics, action
interface, ball hold/release, fall detection) stays whatever the base class
already does.

Two things live here that earlier revisions didn't have:

1. An explicit release / first-landing event state machine
   (_update_event_state), so "the ball has been thrown" and "the ball has
   landed" are each detected exactly once per episode from real physical
   evidence (hand separation + speed + confirmation steps; flight time +
   descent/contact), rather than inferred implicitly from the base class's
   landed flag on every step.

2. Reward assembled from continuous per-step components (unchanged from the
   previous revision) plus one-time components paid out only on the exact
   release/landing step (envs/reward_components.py: release_speed_reward,
   premature_release_penalty_raw,
   compute_landing_outcome), so the ball's accuracy is scored from its first
   landing position only -- never from mid-flight or held-in-hand position,
   and never repeated every step.
"""
from __future__ import annotations

import mujoco
import numpy as np

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv
from envs.reward_components import (
    EventDetectionConfig,
    RewardShapingParams,
    RewardWeights,
    action_smoothness_penalty,
    axis_decomposition,
    compute_landing_outcome,
    desired_release_speed,
    feet_and_ground_contact_reward,
    hand_forward_reward,
    hand_rotation_penalty,
    pelvis_fixed_reward,
    premature_release_penalty_raw,
    projectile_straightness_reward,
    release_speed_reward,
    self_collision_penalty,
)

SATURATION_EPS = 1e-3  # rad; joint target within this of its clipped bound counts as "saturated"
ACTION_BOUND_EPS = 0.98  # normalized action magnitude counted as "at the bound"

DISTANCE_TOLERANCE_M = 0.01


class G1TargetThrowEnv(G1FixedBodyThrowEnv):
    """Throw the ball at a small square target ``target_pos`` away, pelvis fixed."""

    def __init__(
        self,
        *args,
        target_pos=(5.0, 0.0, 0.01),
        target_half_size: float = 0.35,
        episode_time: float = 3.0,
        pelvis_pos_tolerance: float = 0.03,
        pelvis_turn_tolerance: float = 0.09,
        reward_weights: RewardWeights | None = None,
        event_config: EventDetectionConfig | None = None,
        shaping: RewardShapingParams | None = None,
        verbose_init: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            target_pos=target_pos,
            episode_time=episode_time,
            success_radius=target_half_size,
            **kwargs,
        )
        self.target_half_size = float(target_half_size)
        self.pelvis_pos_tolerance = float(pelvis_pos_tolerance)
        self.pelvis_turn_tolerance = float(pelvis_turn_tolerance)
        self.weights = reward_weights or RewardWeights()
        self.event_config = event_config or EventDetectionConfig()
        self.shaping = shaping or RewardShapingParams()

        arm_body_ids = {int(self.model.jnt_bodyid[j]) for j in self.arm_joint_ids}
        if self.hand_joint_ids.size:
            arm_body_ids |= {int(self.model.jnt_bodyid[j]) for j in self.hand_joint_ids}
        self.arm_body_ids = arm_body_ids

        # Joint-group masks for actuator saturation diagnostics (item 7): which
        # of the 7 arm joints (in self.arm_joint_names / arm_actuator_ids order)
        # belong to each named group, by substring match on the existing joint
        # names -- no new names invented, just classifying what's already there.
        self._shoulder_mask = np.array(["shoulder" in n for n in self.arm_joint_names])
        self._elbow_mask = np.array(["elbow" in n for n in self.arm_joint_names])
        self._wrist_mask = np.array(["wrist" in n for n in self.arm_joint_names])

        left_foot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        right_foot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.foot_body_ids = {left_foot, right_foot}
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "throw_target_geom")

        self.pelvis_home_pos = self.data.xpos[self.base_body_id].copy()
        self.pelvis_home_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_reward_components: dict[str, float] = {}
        self.last_reward_raw: dict[str, float] = {}
        self.launch_xy = np.zeros(2)

        self._reset_event_state()

        if verbose_init:
            self._print_init_diagnostics(target_pos)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_init_diagnostics(self, requested_target_pos) -> None:
        robot_origin_xy = self.pelvis_home_pos[:2]
        target_xy = np.asarray(self.target_pos[:2])
        expected_distance = float(np.linalg.norm(np.asarray(requested_target_pos[:2]) - np.zeros(2)))
        actual_distance = float(np.linalg.norm(target_xy - robot_origin_xy))
        ok = abs(actual_distance - expected_distance) <= DISTANCE_TOLERANCE_M
        print(
            "[G1TargetThrowEnv init] "
            f"robot_origin_xy={robot_origin_xy.tolist()} "
            f"target_center_xy={target_xy.tolist()} "
            f"target_half_size={self.target_half_size:.3f}m "
            f"ball_radius={self.ball_radius:.3f}m "
            f"expected_target_distance={expected_distance:.3f}m "
            f"actual_target_distance={actual_distance:.3f}m "
            f"distance_ok={ok}"
        )
        if not ok:
            print(
                "[G1TargetThrowEnv init] WARNING: actual target distance does not match "
                "the requested target_pos within tolerance -- check robot_origin assumptions."
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_event_state(self) -> None:
        self.ball_released = False
        self.release_recorded = False
        self.release_time: float | None = None
        self.release_position: np.ndarray | None = None
        self.release_velocity: np.ndarray | None = None
        self.release_speed: float | None = None
        self.release_hand_position: np.ndarray | None = None
        self.release_forward_speed: float | None = None
        self.release_vertical_speed: float | None = None
        self.release_lateral_speed: float | None = None
        self.release_angle_deg: float | None = None
        self._release_confirm_counter = 0
        self._release_event_this_step = False

        # Item 6: release-timing diagnostics -- was release before or after
        # peak forward hand speed, and what were the arm joint velocities at
        # that instant.
        self._max_hand_forward_velocity_before_release = 0.0
        self.release_hand_forward_velocity: float | None = None
        self.max_hand_forward_velocity_before_release: float | None = None
        self.release_hand_velocity_diff_from_peak: float | None = None
        self.release_arm_joint_velocities: np.ndarray | None = None
        self.release_desired_speed: float | None = None
        self.release_speed_deficit: float | None = None

        # Item 7: action/actuator saturation running accumulators (rates are
        # computed on demand in step() from these).
        self._saturation_steps = {"shoulder": 0, "elbow": 0, "wrist": 0}
        self._saturation_streak = {"shoulder": 0, "elbow": 0, "wrist": 0}
        self._saturation_max_streak = {"shoulder": 0, "elbow": 0, "wrist": 0}
        self._action_clip_fraction_sum = 0.0
        self._saturation_total_steps = 0
        self._max_arm_actuator_force = 0.0

        self.landing_recorded = False
        self.first_landing_position: np.ndarray | None = None
        self.first_landing_time: float | None = None
        self.flight_time: float | None = None
        self.landing_forward_distance: float | None = None
        self.landing_lateral_offset: float | None = None
        self.target_distance: float | None = None
        self.episode_landing_outcome: dict | None = None
        self._landing_event_this_step = False

        self.out_of_bounds = False
        self._settle_counter = 0
        self.termination_reason: str | None = None
        self._pending_terminated = False
        self._pending_truncated = False
        self.self_collision_count = 0
        self.max_ball_height = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.pelvis_home_pos = self.data.xpos[self.base_body_id].copy()
        self.pelvis_home_quat = self.data.xquat[self.base_body_id].copy()
        self.last_reward_components = {}
        self.last_reward_raw = {}
        self.launch_xy = self._ball_pos()[:2].copy()
        self._reset_event_state()
        return obs, info

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    def _ball_touching_floor_or_target(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = (contact.geom1, contact.geom2)
            if self.ball_geom_id not in geoms:
                continue
            other_geom = geoms[0] if geoms[1] == self.ball_geom_id else geoms[1]
            if other_geom in (self.floor_geom_id, self.target_geom_id):
                return True
        return False

    def _update_event_state(self) -> None:
        t = self.step_count * self.control_dt
        ball_pos = self._ball_pos()
        ball_xy = ball_pos[:2]
        ball_linvel = self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 3].copy()
        self.max_ball_height = max(self.max_ball_height, float(ball_pos[2]))

        self._release_event_this_step = False
        self._landing_event_this_step = False

        hand_pos = self.data.xpos[self.hold_body_id]
        hand_vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.hold_body_id, hand_vel6, 0)
        hand_forward_velocity = float(hand_vel6[3])  # rot:lin ordering, index 3 = world-x linear

        if not self.release_recorded:
            self._max_hand_forward_velocity_before_release = max(
                self._max_hand_forward_velocity_before_release, hand_forward_velocity
            )
            dist_from_hand = float(np.linalg.norm(ball_pos - hand_pos))
            ball_speed = float(np.linalg.norm(ball_linvel))
            attached = bool(self.hold_eq_id >= 0 and self.data.eq_active[self.hold_eq_id])
            condition = (
                (not attached)
                and dist_from_hand >= self.event_config.release_min_hand_distance
                and ball_speed >= self.event_config.release_min_speed
            )
            self._release_confirm_counter = self._release_confirm_counter + 1 if condition else 0
            if not self.ball_released and self._release_confirm_counter >= self.event_config.release_confirm_steps:
                self.ball_released = True
            if self.ball_released and not self.release_recorded:
                self.release_recorded = True
                self.release_time = t
                self.release_position = ball_pos.copy()
                self.release_velocity = ball_linvel.copy()
                self.release_speed = ball_speed
                self.release_hand_position = hand_pos.copy()
                self.release_forward_speed = float(ball_linvel[0])
                self.release_vertical_speed = float(ball_linvel[2])
                self.release_lateral_speed = float(ball_linvel[1])
                horizontal_speed = float(np.hypot(ball_linvel[0], ball_linvel[1]))
                self.release_angle_deg = (
                    float(np.degrees(np.arctan2(ball_linvel[2], horizontal_speed)))
                    if horizontal_speed > 1e-6
                    else 90.0
                )
                self.release_hand_forward_velocity = hand_forward_velocity
                self.max_hand_forward_velocity_before_release = self._max_hand_forward_velocity_before_release
                self.release_hand_velocity_diff_from_peak = (
                    self._max_hand_forward_velocity_before_release - hand_forward_velocity
                )
                self.release_arm_joint_velocities = self.data.qvel[self.arm_qvel_adr].copy()
                self._release_event_this_step = True

        if self.ball_released and not self.landing_recorded:
            time_since_release = t - self.release_time
            near_ground = ball_pos[2] <= (self.ball_radius + self.event_config.ground_contact_tolerance)
            settling = abs(float(ball_linvel[2])) <= self.event_config.allowed_vertical_velocity
            contact_confirmed = self._ball_touching_floor_or_target()
            if time_since_release >= self.event_config.min_flight_time and (
                contact_confirmed or (near_ground and settling)
            ):
                self.landing_recorded = True
                self.first_landing_position = ball_xy.copy()
                self.first_landing_time = t
                self.flight_time = time_since_release
                forward, lateral, distance = axis_decomposition(
                    ball_xy, self.pelvis_home_pos[:2], np.asarray(self.target_pos[:2])
                )
                self.landing_forward_distance = forward
                self.landing_lateral_offset = lateral
                self.target_distance = distance
                self._landing_event_this_step = True

        if not self.landing_recorded and not self.out_of_bounds:
            if float(np.linalg.norm(ball_xy - self.launch_xy)) > self.event_config.out_of_bounds_radius:
                self.out_of_bounds = True

    def _update_saturation_tracking(self, action: np.ndarray) -> None:
        """Item 7: is the policy physically maxed out, or is there headroom left?

        ``action`` here is what SB3 already clipped to the Box's [-1, 1]
        before calling step() (the true pre-clip policy output isn't visible
        from inside the env -- SB3 clips in its own rollout loop upstream).
        What IS fully visible, and more physically meaningful anyway, is
        whether the resulting joint TARGET sits at its clipped safety-margin
        bound -- that's what "the arm has no more range to give" actually
        means, independent of whether the policy's raw output was exactly at
        +-1 or far beyond it.
        """
        self._saturation_total_steps += 1
        arm_action = action[: self.n_arm]
        self._action_clip_fraction_sum += float(np.mean(np.abs(action) >= ACTION_BOUND_EPS))

        arm_targets = self.nominal_ctrl[self.arm_actuator_ids] + self.action_scale * arm_action
        at_lower = arm_targets <= self.arm_joint_lower + SATURATION_EPS
        at_upper = arm_targets >= self.arm_joint_upper - SATURATION_EPS
        saturated = at_lower | at_upper

        for group, mask in (("shoulder", self._shoulder_mask), ("elbow", self._elbow_mask), ("wrist", self._wrist_mask)):
            group_saturated = bool(np.any(saturated & mask)) if mask.any() else False
            if group_saturated:
                self._saturation_steps[group] += 1
                self._saturation_streak[group] += 1
                self._saturation_max_streak[group] = max(self._saturation_max_streak[group], self._saturation_streak[group])
            else:
                self._saturation_streak[group] = 0

        arm_forces = np.abs(self.data.actuator_force[self.arm_actuator_ids])
        if arm_forces.size:
            self._max_arm_actuator_force = max(self._max_arm_actuator_force, float(np.max(arm_forces)))

    def _decide_termination(self) -> None:
        terminated = False
        truncated = False
        reason = self.termination_reason
        if self.robot_fell:
            terminated = True
            reason = "fall"
        elif self.landing_recorded:
            self._settle_counter += 1
            if self._settle_counter >= self.event_config.settle_steps_after_landing:
                terminated = True
                hit = bool(self.episode_landing_outcome and self.episode_landing_outcome.get("target_hit"))
                reason = "success" if hit else "landed"
        elif self.out_of_bounds:
            terminated = True
            reason = "out_of_bounds"
        elif self.step_count * self.control_dt >= self.episode_time:
            truncated = True
            reason = "no_release" if not self.ball_released else "timeout"
        self._pending_terminated = terminated
        self._pending_truncated = truncated
        self.termination_reason = reason

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _throw_distance(self) -> float:
        return float(np.linalg.norm(self._ball_pos()[:2] - self.launch_xy))

    def _compute_reward(self, action, landed):
        self._update_event_state()
        self._update_saturation_tracking(action)
        w = self.weights
        raw: dict[str, float] = {}
        terms: dict[str, float] = {}

        raw["pelvis_fixed"] = pelvis_fixed_reward(self)
        raw["hand_forward"] = hand_forward_reward(self)
        raw["hand_rotation"] = hand_rotation_penalty(self)
        raw["self_collision"] = self_collision_penalty(self)
        if raw["self_collision"] < 0:
            self.self_collision_count += 1
        feet_r, bad_ground_r = feet_and_ground_contact_reward(self)
        raw["feet_contact"] = feet_r
        raw["bad_ground_contact"] = bad_ground_r
        raw["projectile_straightness"] = projectile_straightness_reward(self) if self.ball_released else 0.0
        raw["action_smoothness"] = action_smoothness_penalty(self, action)

        terms["pelvis_fixed"] = w.pelvis_fixed * raw["pelvis_fixed"]
        terms["hand_forward"] = w.hand_forward * raw["hand_forward"]
        terms["hand_rotation"] = w.hand_rotation * raw["hand_rotation"]
        terms["self_collision"] = w.self_collision * raw["self_collision"]
        terms["feet_contact"] = w.feet_contact * raw["feet_contact"]
        terms["bad_ground_contact"] = w.bad_ground_contact * raw["bad_ground_contact"]
        terms["projectile"] = w.projectile_straightness * raw["projectile_straightness"]
        terms["action_smoothness"] = w.action_smoothness * raw["action_smoothness"]
        terms["fall_penalty"] = -w.fall_penalty if self.robot_fell else 0.0

        terms["release_speed"] = 0.0
        terms["premature_release_penalty"] = 0.0
        if self._release_event_this_step:
            static_target_distance = float(
                np.linalg.norm(np.asarray(self.target_pos[:2]) - self.pelvis_home_pos[:2])
            )
            self.release_desired_speed = desired_release_speed(
                static_target_distance,
                float(self.release_position[2]),
                np.radians(self.release_angle_deg),
                self.shaping,
            )
            self.release_speed_deficit = self.release_desired_speed - self.release_forward_speed
            raw["release_speed"] = release_speed_reward(self.release_forward_speed, self.release_desired_speed)
            terms["release_speed"] = w.release_speed * raw["release_speed"]
            # "_magnitude" (not "_penalty") for the RAW value here on purpose --
            # this dict already has a same-named "premature_release_penalty" key
            # in `terms` holding the actual signed (negative) reward term, and
            # reusing the identical key name for an unsigned [0,1] magnitude in
            # `raw` was exactly the ambiguity flagged during review: a reader
            # could mistake the positive raw magnitude for a positive reward.
            raw["premature_release_magnitude"] = premature_release_penalty_raw(self.release_time, self.event_config)
            terms["premature_release_penalty"] = -w.premature_release_penalty * raw["premature_release_magnitude"]

        for key in (
            "distance_progress",
            "target_dense",
            "target_longitudinal",
            "target_lateral",
            "accuracy_progress",
            "undershoot_penalty",
            "overshoot_penalty",
            "target_hit_bonus",
            "near_target_bonus",
        ):
            terms[key] = 0.0

        if self._landing_event_this_step:
            outcome = compute_landing_outcome(
                self.landing_forward_distance,
                self.landing_lateral_offset,
                self.target_distance,
                self.target_half_size,
                self.ball_radius,
                self.shaping,
            )
            self.episode_landing_outcome = outcome
            terms["distance_progress"] = w.distance_progress * outcome["distance_progress_raw"]
            terms["target_dense"] = w.target_dense * outcome["target_dense_raw"]
            terms["target_longitudinal"] = w.target_longitudinal * outcome["target_longitudinal_raw"]
            terms["target_lateral"] = w.lateral_accuracy * outcome["lateral_accuracy_raw"]
            terms["accuracy_progress"] = w.accuracy_progress * outcome["accuracy_progress_raw"]
            terms["undershoot_penalty"] = -w.undershoot_penalty * outcome["undershoot"]
            terms["overshoot_penalty"] = -w.overshoot_penalty * outcome["overshoot"]
            terms["target_hit_bonus"] = w.target_hit_bonus if outcome["target_hit"] else 0.0
            terms["near_target_bonus"] = outcome["near_target_bonus"]
            raw["distance_progress"] = outcome["distance_progress_raw"]
            raw["target_dense"] = outcome["target_dense_raw"]
            raw["target_longitudinal"] = outcome["target_longitudinal_raw"]
            raw["lateral_accuracy"] = outcome["lateral_accuracy_raw"]
            raw["accuracy_progress"] = outcome["accuracy_progress_raw"]

        self.last_reward_raw = raw
        self.last_reward_components = terms
        self._decide_termination()
        return float(sum(terms.values()))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, _base_terminated, _base_truncated, info = super().step(action)
        terminated = self._pending_terminated
        truncated = self._pending_truncated

        info["reward_components"] = dict(self.last_reward_components)
        info["reward_raw"] = dict(self.last_reward_raw)
        info["throw_distance"] = self._throw_distance()
        info["termination_reason"] = self.termination_reason
        info["self_collision_count"] = self.self_collision_count
        info["max_ball_height"] = self.max_ball_height

        info["ball_released"] = self.ball_released
        info["release_time"] = self.release_time
        info["release_position"] = None if self.release_position is None else self.release_position.tolist()
        info["release_velocity"] = None if self.release_velocity is None else self.release_velocity.tolist()
        info["release_speed"] = self.release_speed
        info["release_forward_speed"] = self.release_forward_speed
        info["release_vertical_speed"] = self.release_vertical_speed
        info["release_lateral_speed"] = self.release_lateral_speed
        info["release_angle_deg"] = self.release_angle_deg
        info["release_desired_speed_mps"] = self.release_desired_speed
        info["release_speed_deficit_mps"] = self.release_speed_deficit

        # Item 6: release-timing diagnostics.
        info["release_hand_forward_velocity"] = self.release_hand_forward_velocity
        info["max_hand_forward_velocity_before_release"] = self.max_hand_forward_velocity_before_release
        info["release_hand_velocity_diff_from_peak"] = self.release_hand_velocity_diff_from_peak
        info["release_arm_joint_velocities"] = (
            None if self.release_arm_joint_velocities is None else self.release_arm_joint_velocities.tolist()
        )
        info["released_before_peak_hand_speed"] = (
            None if self.release_hand_velocity_diff_from_peak is None else self.release_hand_velocity_diff_from_peak > 1e-6
        )

        # Item 7: action/actuator saturation, running rates over the episode so far.
        n = max(self._saturation_total_steps, 1)
        info["action_clip_fraction"] = self._action_clip_fraction_sum / n
        info["shoulder_saturation_rate"] = self._saturation_steps["shoulder"] / n
        info["elbow_saturation_rate"] = self._saturation_steps["elbow"] / n
        info["wrist_saturation_rate"] = self._saturation_steps["wrist"] / n
        info["shoulder_saturation_max_streak"] = self._saturation_max_streak["shoulder"]
        info["elbow_saturation_max_streak"] = self._saturation_max_streak["elbow"]
        info["wrist_saturation_max_streak"] = self._saturation_max_streak["wrist"]
        info["max_arm_actuator_force"] = self._max_arm_actuator_force
        info["mean_hand_speed_at_release"] = self.release_hand_forward_velocity
        info["max_hand_speed_before_release"] = self.max_hand_forward_velocity_before_release

        info["landing_recorded"] = self.landing_recorded
        info["flight_time"] = self.flight_time
        info["pelvis_position_drift_m"] = float(
            np.linalg.norm(self.data.xpos[self.base_body_id][:2] - self.pelvis_home_pos[:2])
        )

        if self.episode_landing_outcome is not None:
            outcome = self.episode_landing_outcome
            landing_error = outcome["landing_error"]
            info["landing_position"] = list(self.first_landing_position)
            info["landing_z"] = float(self.ball_radius)
            info["landing_forward_distance"] = self.landing_forward_distance
            info["target_distance_m"] = self.target_distance
            info["longitudinal_error"] = outcome["longitudinal_error"]
            info["lateral_error"] = outcome["lateral_error"]
            info["landing_error"] = landing_error
            info["undershoot"] = outcome["undershoot"]
            info["overshoot"] = outcome["overshoot"]
            info["square_target_hit"] = outcome["target_hit"]
            info["within_025m"] = landing_error <= 0.25
            info["within_050m"] = landing_error <= 0.50
            info["within_075m"] = landing_error <= 0.75
            info["within_100m"] = landing_error <= 1.00
            info["within_150m"] = landing_error <= 1.50
            info["within_200m"] = landing_error <= 2.00
        else:
            info["landing_position"] = None
            info["landing_z"] = None
            info["landing_forward_distance"] = None
            info["target_distance_m"] = float(
                np.linalg.norm(np.asarray(self.target_pos[:2]) - self.pelvis_home_pos[:2])
            )
            info["longitudinal_error"] = None
            info["lateral_error"] = None
            info["landing_error"] = None
            info["undershoot"] = None
            info["overshoot"] = None
            info["square_target_hit"] = False
            info["within_025m"] = False
            info["within_050m"] = False
            info["within_075m"] = False
            info["within_100m"] = False
            info["within_150m"] = False
            info["within_200m"] = False

        return obs, reward, terminated, truncated, info
