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
from gymnasium import spaces

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv
from envs.hand_geometry import compute_ball_hold_offset, get_palm_frame, print_hand_geometry_report
from envs.reward_components import (
    EventDetectionConfig,
    MotionPhaseConfig,
    RewardShapingParams,
    RewardWeights,
    action_smoothness_penalty,
    axis_decomposition,
    ball_palm_alignment_reward,
    compute_landing_outcome,
    desired_release_speed,
    feet_and_ground_contact_reward,
    hand_forward_reward,
    hand_rotation_penalty,
    joint_acceleration_penalty,
    palm_orientation_reward,
    pelvis_fixed_reward,
    post_throw_velocity_penalty,
    premature_release_penalty_raw,
    projectile_straightness_reward,
    recovery_pose_reward,
    release_speed_reward,
    self_collision_penalty,
    wrist_roll_interval_reward,
    wrist_yaw_reference,
    wrist_yaw_release_reward,
)
from envs.right_arm_config import (
    ACTIVE_ARM_JOINT_NAMES,
    FIXED_ARM_JOINT_NAMES,
    RightArmCalibration,
    load_right_arm_config,
)

SATURATION_EPS = 1e-3  # rad; joint target within this of its clipped bound counts as "saturated"
ACTION_BOUND_EPS = 0.98  # normalized action magnitude counted as "at the bound"

DISTANCE_TOLERANCE_M = 0.01

PHASES = ("ready", "throw", "follow_through", "recovery", "hold")
PHASE_INDEX = {name: i / (len(PHASES) - 1) for i, name in enumerate(PHASES)}
PHASE_FEATURE_NAMES = (
    "normalized_episode_time",
    "normalized_time_since_release",
    "is_released",
    "is_recovering",
    "phase_index",
)


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


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
        motion_phases: MotionPhaseConfig | None = None,
        right_arm_calibration: RightArmCalibration | None = None,
        right_arm_config_path=None,
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
        self.motion_phases = motion_phases or MotionPhaseConfig()

        # Right-arm joint reduction: verify the joint-name assumptions against
        # the LIVE model (not just trust the constants) -- see
        # envs/right_arm_config.py for why these five are "active" (policy
        # controls them) and these two are "fixed" (held constant, never in
        # the action space).
        active_and_fixed = set(ACTIVE_ARM_JOINT_NAMES) | set(FIXED_ARM_JOINT_NAMES)
        if active_and_fixed != set(self.arm_joint_names):
            raise RuntimeError(
                "envs/right_arm_config.py's ACTIVE_ARM_JOINT_NAMES + FIXED_ARM_JOINT_NAMES "
                f"({sorted(active_and_fixed)}) do not match the live model's arm_joint_names "
                f"({sorted(self.arm_joint_names)}) -- the joint-name assumptions need updating."
            )
        if right_arm_calibration is not None:
            self.right_arm_calibration = right_arm_calibration
        else:
            self.right_arm_calibration = load_right_arm_config(right_arm_config_path)
        self.right_arm_calibration.validate()
        calib = self.right_arm_calibration

        self._active_arm_local_idx = np.array(
            [self.arm_joint_names.index(n) for n in ACTIVE_ARM_JOINT_NAMES], dtype=np.int64
        )
        self._fixed_arm_local_idx = np.array(
            [self.arm_joint_names.index(n) for n in FIXED_ARM_JOINT_NAMES], dtype=np.int64
        )
        self._wrist_roll_local_idx = self.arm_joint_names.index("right_wrist_roll_joint")
        self._wrist_yaw_local_idx = self.arm_joint_names.index("right_wrist_yaw_joint")

        # Re-anchor "nominal" (the ctrl the base class returns to every reset
        # and every step's action=0 baseline, see G1FixedBodyThrowEnv.step's
        # arm_targets = nominal_ctrl + action_scale*action) for the three
        # calibrated joints -- an INSTANCE-ARRAY override, the same pattern
        # already used for hold_relpose/model.eq_data below, not a base-class
        # edit. This makes wrist_roll's resting/action=0 angle the calibrated
        # ~1.652 rad (not the model's raw keyframe 0.0) and locks the two
        # fixed joints to their calibrated constants (default 0.0, i.e.
        # unchanged from the keyframe, until the calibration tool sets
        # otherwise).
        self.nominal_ctrl[self.arm_actuator_ids[self._wrist_roll_local_idx]] = calib.wrist_roll_nominal
        for name, value in calib.fixed_joints.items():
            idx = self.arm_joint_names.index(name)
            self.nominal_ctrl[self.arm_actuator_ids[idx]] = value

        # Per-active-joint ACTION clamp (not a ctrl-space clamp -- the base
        # class's step() hard-clips action to [-1,1] and computes ctrl from
        # it internally, with no hook to reclip ctrl afterward, so the only
        # way to guarantee a ctrl target narrower than nominal+-action_scale
        # is to constrain the action that produces it). shoulder_pitch/roll/
        # elbow keep the full [-1,1] range; wrist_roll and wrist_yaw get the
        # action interval that maps, through nominal_ctrl + action_scale*a,
        # exactly onto their calibrated radian ranges.
        self._active_action_low = np.full(len(ACTIVE_ARM_JOINT_NAMES), -1.0)
        self._active_action_high = np.full(len(ACTIVE_ARM_JOINT_NAMES), 1.0)
        wrist_roll_active_idx = list(ACTIVE_ARM_JOINT_NAMES).index("right_wrist_roll_joint")
        wrist_yaw_active_idx = list(ACTIVE_ARM_JOINT_NAMES).index("right_wrist_yaw_joint")
        wrist_roll_nominal_ctrl = self.nominal_ctrl[self.arm_actuator_ids[self._wrist_roll_local_idx]]
        wrist_yaw_nominal_ctrl = self.nominal_ctrl[self.arm_actuator_ids[self._wrist_yaw_local_idx]]
        self._active_action_low[wrist_roll_active_idx] = np.clip(
            (calib.wrist_roll_min - wrist_roll_nominal_ctrl) / self.action_scale, -1.0, 1.0
        )
        self._active_action_high[wrist_roll_active_idx] = np.clip(
            (calib.wrist_roll_max - wrist_roll_nominal_ctrl) / self.action_scale, -1.0, 1.0
        )
        self._active_action_low[wrist_yaw_active_idx] = np.clip(
            (calib.wrist_yaw_release_min - wrist_yaw_nominal_ctrl) / self.action_scale, -1.0, 1.0
        )
        self._active_action_high[wrist_yaw_active_idx] = np.clip(
            (calib.wrist_yaw_release_max - wrist_yaw_nominal_ctrl) / self.action_scale, -1.0, 1.0
        )

        # Item 5/8 (reduced action space): policy now outputs 5 active-arm
        # values + release, not 7+1 -- this changes action_space.shape, so
        # (like the observation_space extension below) a checkpoint trained
        # on the previous 8-dim action cannot be loaded/resumed here.
        # PPO.load's shape check (RL/train_target_throw.py's --resume-from
        # handling) already refuses ANY observation/action shape mismatch
        # with a clear error -- this is not new code, just a case that
        # generic guard now also covers.
        self.action_space = spaces.Box(-1, 1, shape=(len(ACTIVE_ARM_JOINT_NAMES) + 1,), dtype=np.float32)

        # Item 2/3/4/6: replace the hand-picked ball hold offset with one
        # derived from the actual hand mesh geometry (envs/hand_geometry.py),
        # so the ball rests just past the visual hand mesh instead of
        # overlapping it, plus the calibrated lateral (and any forward/
        # vertical) nudge on top. This overwrites the INSTANCE's hold_relpose
        # attribute and the compiled model's own weld constraint data
        # (model.eq_data) -- runtime data, the same kind of override the base
        # class itself already does for target_pos in reset() -- not the XML
        # source and not any base-class code.
        hold_offset, hold_quat = compute_ball_hold_offset(
            self.model, self.hold_body_id, self.ball_radius, self.motion_phases.ball_contact_margin,
            extra_offset_local=(calib.ball_offset_forward, calib.ball_offset_lateral, calib.ball_offset_vertical),
        )
        self.hold_relpose = np.concatenate([hold_offset, hold_quat])
        self.model.eq_data[self.hold_eq_id][3:10] = self.hold_relpose

        # Item 5/8: ready/rest poses are NORMALIZED ACTIONS (see
        # MotionPhaseConfig docstring); ctrl versions are what's actually
        # commanded to the position actuators (= target joint angles,
        # directly, since these are MuJoCo <position> actuators), clipped to
        # the same safety margin as everything else. Computed AFTER the
        # nominal_ctrl re-anchoring above, so an all-zero pose action here
        # correctly resolves to the calibrated rest pose (wrist_roll ~1.652,
        # fixed joints at their calibrated constants), not the raw keyframe.
        self.ready_pose_action = np.array(self.motion_phases.ready_pose, dtype=np.float64)
        self.rest_pose_action = np.array(self.motion_phases.rest_pose, dtype=np.float64)
        self.ready_pose_ctrl = np.clip(
            self.nominal_ctrl[self.arm_actuator_ids] + self.action_scale * self.ready_pose_action,
            self.arm_joint_lower, self.arm_joint_upper,
        )
        self.rest_pose_ctrl = np.clip(
            self.nominal_ctrl[self.arm_actuator_ids] + self.action_scale * self.rest_pose_action,
            self.arm_joint_lower, self.arm_joint_upper,
        )

        arm_body_ids = {int(self.model.jnt_bodyid[j]) for j in self.arm_joint_ids}
        if self.hand_joint_ids.size:
            arm_body_ids |= {int(self.model.jnt_bodyid[j]) for j in self.hand_joint_ids}
        self.arm_body_ids = arm_body_ids

        # Joint-group masks for actuator saturation diagnostics (item 7): which
        # of the 7 arm joints (in self.arm_joint_names / arm_actuator_ids order)
        # belong to each named group, by substring match on the existing joint
        # names -- no new names invented, just classifying what's already there.
        # Note shoulder_yaw/wrist_pitch (FIXED) still match "shoulder"/"wrist"
        # here; harmless -- their action is always exactly 0 so arm_targets
        # always equals nominal_ctrl for them, which is never at a joint-limit
        # bound by construction, so they never register as saturated.
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
        self.throw_direction_world = np.array([1.0, 0.0, 0.0])

        # Item 10: extend the observation with phase/timing features. This
        # changes observation_space.shape, so a checkpoint trained on the
        # PREVIOUS (un-extended) observation cannot be loaded/resumed here --
        # PPO.load's own shape check will refuse it with a clear error rather
        # than silently mismatching (see RL/train_target_throw.py's
        # --resume-from handling). A fresh training run is required after
        # this change; see the final report for why that's unavoidable here.
        base_obs_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(base_obs_dim + len(PHASE_FEATURE_NAMES),), dtype=np.float32
        )

        self._reset_event_state()

        if verbose_init:
            print_hand_geometry_report(self.model, self.hold_body_id, self.ball_geom_id)
            self._print_init_diagnostics(target_pos)
            self._print_right_arm_calibration_report()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_right_arm_calibration_report(self) -> None:
        calib = self.right_arm_calibration
        print("=" * 72)
        print("RIGHT-ARM JOINT CONFIGURATION")
        print("=" * 72)
        print(f"active (policy-controlled, 5-dim action): {ACTIVE_ARM_JOINT_NAMES}")
        print(f"fixed  (held constant, never in action space): {calib.fixed_joints}")
        print(
            f"wrist_roll: nominal={calib.wrist_roll_nominal:.4f} rad  "
            f"range=[{calib.wrist_roll_min:.4f}, {calib.wrist_roll_max:.4f}]  "
            f"action_clamp=[{self._active_action_low[list(ACTIVE_ARM_JOINT_NAMES).index('right_wrist_roll_joint')]:.4f}, "
            f"{self._active_action_high[list(ACTIVE_ARM_JOINT_NAMES).index('right_wrist_roll_joint')]:.4f}]"
        )
        print(
            f"wrist_yaw:  hold={calib.wrist_yaw_hold:.4f}  "
            f"release_range=[{calib.wrist_yaw_release_min:.4f}, {calib.wrist_yaw_release_max:.4f}]  "
            f"release_value={calib.wrist_yaw_release_value:.4f}  ramp_duration={calib.wrist_yaw_ramp_duration:.3f}s  "
            f"action_clamp=[{self._active_action_low[list(ACTIVE_ARM_JOINT_NAMES).index('right_wrist_yaw_joint')]:.4f}, "
            f"{self._active_action_high[list(ACTIVE_ARM_JOINT_NAMES).index('right_wrist_yaw_joint')]:.4f}]"
        )
        print(
            f"ball_offset_local (added to geometry-derived offset): "
            f"forward+={calib.ball_offset_forward:.4f}  lateral={calib.ball_offset_lateral:.4f} "
            f"(positive = robot's own left)  vertical+={calib.ball_offset_vertical:.4f}"
        )
        print("=" * 72)

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

        # Items 5-10: motion-phase state machine.
        self.phase = "ready"
        self._recovery_start_ctrl: np.ndarray | None = None
        self._recovery_start_time: float | None = None
        self._hold_start_time: float | None = None
        self.ball_to_palm_distance: float = 0.0  # continuously updated while held
        self.palm_roll_at_release: float | None = None
        self.palm_pitch_at_release: float | None = None
        self.palm_yaw_at_release: float | None = None
        self.palm_normal_alignment_at_release: float | None = None

    def reset(self, seed=None, options=None):
        # _reset_event_state() first (not after, as in earlier revisions):
        # super().reset() internally calls self._get_obs() (virtual dispatch
        # -> MY override, which reads self.phase/self.release_recorded/etc),
        # so those need valid values BEFORE super().reset() runs, not after.
        self._reset_event_state()
        obs, info = super().reset(seed=seed, options=options)

        # wrist_roll's calibrated nominal (~1.652 rad) sits ~1.65 rad from the
        # model's raw keyframe (~0 rad) -- letting the PD controller travel
        # that whole distance from a standing start (base class's reset()
        # already commanded ctrl=nominal_ctrl, i.e. the calibrated target,
        # one line above) measurably OVERSHOOTS (observed: transient peak
        # ~2.26 rad, well outside the calibrated [1.62, 1.70] interval,
        # before settling ~0.14s later). Rather than trying to script a
        # smooth ramp through the action interface -- which can't represent
        # ctrl values outside nominal_ctrl +- action_scale (0.5 rad) anyway,
        # so it can't reach all the way back to ~0 to blend from -- start the
        # episode with the calibrated joints' PHYSICAL qpos already at their
        # target, the same direct-qpos-teleport pattern already used for ball
        # placement (_place_ball_in_hand). No transient, no overshoot, and
        # the ball is re-placed afterward so it matches the corrected hand
        # orientation, not the pre-teleport one.
        wrist_roll_qpos_adr = self.arm_qpos_adr[self._wrist_roll_local_idx]
        wrist_roll_qvel_adr = self.arm_qvel_adr[self._wrist_roll_local_idx]
        self.data.qpos[wrist_roll_qpos_adr] = self.right_arm_calibration.wrist_roll_nominal
        self.data.qvel[wrist_roll_qvel_adr] = 0.0
        for name, value in self.right_arm_calibration.fixed_joints.items():
            idx = self.arm_joint_names.index(name)
            self.data.qpos[self.arm_qpos_adr[idx]] = value
            self.data.qvel[self.arm_qvel_adr[idx]] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._place_ball_in_hand()
        mujoco.mj_forward(self.model, self.data)  # propagate the re-placed ball's qpos into xpos/xmat

        self.pelvis_home_pos = self.data.xpos[self.base_body_id].copy()
        self.pelvis_home_quat = self.data.xquat[self.base_body_id].copy()
        self.last_reward_components = {}
        self.last_reward_raw = {}
        self.launch_xy = self._ball_pos()[:2].copy()
        to_target = np.asarray(self.target_pos[:2]) - self.pelvis_home_pos[:2]
        norm = float(np.linalg.norm(to_target))
        direction_xy = to_target / norm if norm > 1e-6 else np.array([1.0, 0.0])
        self.throw_direction_world = np.array([direction_xy[0], direction_xy[1], 0.0])
        obs = self._get_obs()
        return obs, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        base_obs = super()._get_obs()
        t = self.step_count * self.control_dt
        time_since_release = (t - self.release_time) if self.release_recorded else 0.0
        extra = np.array(
            [
                float(min(t / max(self.episode_time, 1e-6), 1.0)),
                float(min(time_since_release, 2.0) / 2.0),
                1.0 if self.ball_released else 0.0,
                1.0 if self.phase in ("recovery", "hold") else 0.0,
                PHASE_INDEX[self.phase],
            ],
            dtype=np.float32,
        )
        return np.concatenate([base_obs, extra]).astype(np.float32)

    # ------------------------------------------------------------------
    # Right-arm action reduction (5 active joints -> the base class's 7-arm
    # action interface)
    # ------------------------------------------------------------------

    def _expand_active_action(self, reduced_action: np.ndarray) -> np.ndarray:
        """5 active-joint actions + release -> the 7-arm + release array the
        base class's step() expects. The two FIXED joints (shoulder_yaw,
        wrist_pitch) always get action 0 here -- since nominal_ctrl was
        re-anchored to their calibrated constants in __init__, action=0 means
        "stay at the calibrated value", for every phase, every step,
        unconditionally (this call happens before _phase_action_override,
        which only ever touches this via the SAME full-width array so the
        fixed slots stay 0 through ready/recovery/hold too). wrist_roll and
        wrist_yaw get clamped to their calibrated action ranges (computed
        once in __init__ from the calibrated radian bounds) BEFORE expansion
        -- the base class clips action to [-1,1] unconditionally and offers
        no hook to reclip the resulting ctrl afterward, so constraining the
        action itself is the only way to guarantee the narrower radian range.
        """
        reduced = np.asarray(reduced_action, dtype=np.float64)
        n_active = len(self._active_arm_local_idx)
        active = np.clip(reduced[:n_active], self._active_action_low, self._active_action_high)
        full = np.zeros(self.n_arm + 1, dtype=np.float64)
        full[self._active_arm_local_idx] = active
        full[self._fixed_arm_local_idx] = 0.0
        full[-1] = reduced[-1]
        return full

    # ------------------------------------------------------------------
    # Motion phases (items 5-9)
    # ------------------------------------------------------------------

    def _update_phase(self, t: float) -> None:
        mp = self.motion_phases
        if self.release_recorded:
            dt_release = t - self.release_time
            if dt_release < mp.follow_through_duration:
                new_phase = "follow_through"
            elif dt_release < mp.follow_through_duration + mp.recovery_duration:
                new_phase = "recovery"
            else:
                new_phase = "hold"
        elif t < mp.ready_pose_duration:
            new_phase = "ready"
        else:
            new_phase = "throw"

        if new_phase == "recovery" and self.phase != "recovery":
            self._recovery_start_ctrl = self.data.ctrl[self.arm_actuator_ids].copy()
            self._recovery_start_time = t
        if new_phase == "hold" and self.phase != "hold":
            self._hold_start_time = t
        self.phase = new_phase

    def _phase_action_override(self, action: np.ndarray, t: float) -> np.ndarray:
        """Ready/recovery/hold are scripted -- the policy has NO control over
        the arm during these phases, by construction, regardless of what it
        outputs (item 9: "do not allow the policy to continue sending
        uncontrolled right-arm actions after the throw has completed"). Only
        "throw" and "follow_through" pass the policy's action through
        unchanged.
        """
        action = np.array(action, dtype=np.float64, copy=True)
        mp = self.motion_phases
        if self.phase == "ready":
            action[: self.n_arm] = self.ready_pose_action
            action[-1] = -1.0  # force "do not release" well below the 0.5 trigger threshold
        elif self.phase == "recovery":
            progress = (t - self._recovery_start_time) / max(mp.recovery_duration, 1e-6)
            alpha = _smoothstep(progress)
            blended_ctrl = (1.0 - alpha) * self._recovery_start_ctrl + alpha * self.rest_pose_ctrl
            action[: self.n_arm] = np.clip(
                (blended_ctrl - self.nominal_ctrl[self.arm_actuator_ids]) / self.action_scale, -1.0, 1.0
            )
        elif self.phase == "hold":
            action[: self.n_arm] = self.rest_pose_action
        return action

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
        # NOTE: self.phase is deliberately NOT updated here. It's updated in
        # step() BEFORE physics runs (so the phase that picks the action
        # override is the same phase the reward below is computed against --
        # updating it again here, after physics, with a step-later timestamp
        # could flip the phase mid-step and make the applied action and the
        # reward for it inconsistent).
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
            self.ball_to_palm_distance = dist_from_hand  # item 12: continuously tracked while held
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
                palm_frame = get_palm_frame(self.model, self.data, self.hold_body_id, self.ball_body_id)
                self.palm_roll_at_release = palm_frame.roll_deg
                self.palm_pitch_at_release = palm_frame.pitch_deg
                self.palm_yaw_at_release = palm_frame.yaw_deg
                self.palm_normal_alignment_at_release = float(np.dot(palm_frame.normal_vector, [0.0, 0.0, 1.0]))
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
            # Settle countdown only starts once the scripted follow_through ->
            # recovery -> hold sequence has actually reached "hold" -- if it
            # started counting from the landing step instead, an episode
            # could terminate before recovery ever finishes (landing often
            # happens well before release_time + follow_through_duration +
            # recovery_duration), meaning "the arm returns to rest" would
            # never actually get observed, rewarded, or verified within the
            # episode. episode_time's own truncation below is still the
            # backstop if hold is somehow never reached.
            if self.phase == "hold":
                self._settle_counter += 1
                if self._settle_counter >= self.event_config.settle_steps_after_landing:
                    terminated = True
                    hit = bool(self.episode_landing_outcome and self.episode_landing_outcome.get("target_hit"))
                    reason = "success" if hit else "landed"
            else:
                self._settle_counter = 0
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

        # Item 11: movement-quality terms, each gated to the phase(s) it's
        # actually meaningful in (see envs/reward_components.py docstrings).
        # Kept small (RewardWeights defaults ~0.02-0.15) so they shape HOW the
        # throw looks without competing with the landing-accuracy terms below.
        raw["joint_acceleration"] = joint_acceleration_penalty(self)
        terms["joint_acceleration"] = w.joint_acceleration * raw["joint_acceleration"]

        terms["palm_orientation"] = 0.0
        if self.phase in ("throw", "follow_through"):
            raw["palm_orientation"] = palm_orientation_reward(self)
            terms["palm_orientation"] = w.palm_orientation * raw["palm_orientation"]

        terms["ball_palm_alignment"] = 0.0
        if not self.ball_released:
            raw["ball_palm_alignment"] = ball_palm_alignment_reward(self)
            terms["ball_palm_alignment"] = w.ball_palm_alignment * raw["ball_palm_alignment"]

        terms["recovery_pose"] = 0.0
        if self.phase in ("recovery", "hold"):
            raw["recovery_pose"] = recovery_pose_reward(self)
            terms["recovery_pose"] = w.recovery_pose * raw["recovery_pose"]

        # Item 8: wrist_roll must stay in its calibrated interval regardless
        # of phase (the action clamp already keeps the COMMANDED target
        # inside it -- this scores the REALIZED qpos). wrist_yaw's smooth
        # release-ramp reward is deliberately gated to "throw" only -- "do
        # not reward release yaw during ready/backswing" (item 8's own
        # words); ready is scripted to 0 anyway and follow_through/recovery/
        # hold are scripted back toward 0/rest, so there's no ramp to score
        # there either.
        raw["wrist_roll_interval"] = wrist_roll_interval_reward(self)
        terms["wrist_roll_interval"] = w.wrist_roll_interval * raw["wrist_roll_interval"]

        terms["wrist_yaw_release"] = 0.0
        if self.phase == "throw":
            raw["wrist_yaw_release"] = wrist_yaw_release_reward(self)
            terms["wrist_yaw_release"] = w.wrist_yaw_release * raw["wrist_yaw_release"]

        terms["post_throw_velocity"] = 0.0
        if self.phase == "hold":
            raw["post_throw_velocity"] = post_throw_velocity_penalty(self)
            terms["post_throw_velocity"] = w.post_throw_velocity * raw["post_throw_velocity"]

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
        # Expand the policy's 5-active-joint action to the base class's
        # 7-arm interface FIRST, so everything downstream (phase override,
        # reward, diagnostics) sees the same full-width array it always has.
        full_action = self._expand_active_action(action)

        # Phase decided BEFORE physics, from the PRE-step clock -- this is
        # what picks the action override (items 5-9), so it must be settled
        # before super().step() runs, not after.
        t = self.step_count * self.control_dt
        self._update_phase(t)
        effective_action = self._phase_action_override(full_action, t)

        obs, reward, _base_terminated, _base_truncated, info = super().step(effective_action)
        terminated = self._pending_terminated
        truncated = self._pending_truncated

        info["reward_components"] = dict(self.last_reward_components)
        info["reward_raw"] = dict(self.last_reward_raw)
        info["throw_distance"] = self._throw_distance()
        info["termination_reason"] = self.termination_reason
        info["self_collision_count"] = self.self_collision_count
        info["max_ball_height"] = self.max_ball_height
        info["phase"] = self.phase
        info["ball_to_palm_distance"] = self.ball_to_palm_distance
        info["palm_roll_at_release"] = self.palm_roll_at_release
        info["palm_pitch_at_release"] = self.palm_pitch_at_release
        info["palm_yaw_at_release"] = self.palm_yaw_at_release
        info["palm_normal_alignment"] = self.palm_normal_alignment_at_release
        info["recovery_start_time"] = self._recovery_start_time
        info["hold_start_time"] = self._hold_start_time
        current_arm_qpos = self.data.qpos[self.arm_qpos_adr]
        info["final_arm_pose_error"] = float(np.linalg.norm(current_arm_qpos - self.rest_pose_ctrl))
        hand_vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.hold_body_id, hand_vel6, 0)
        info["final_hand_linear_speed"] = float(np.linalg.norm(hand_vel6[3:]))
        info["final_hand_angular_speed"] = float(np.linalg.norm(hand_vel6[:3]))

        # Right-arm joint-reduction diagnostics: confirm the hard constraints
        # actually hold in the REALIZED physics, not just in the commanded
        # action (see validation item 10 in the change report).
        calib = self.right_arm_calibration
        wrist_roll_rad = float(current_arm_qpos[self._wrist_roll_local_idx])
        wrist_yaw_rad = float(current_arm_qpos[self._wrist_yaw_local_idx])
        info["wrist_roll_rad"] = wrist_roll_rad
        info["wrist_yaw_rad"] = wrist_yaw_rad
        info["wrist_roll_in_range"] = bool(calib.wrist_roll_min <= wrist_roll_rad <= calib.wrist_roll_max)
        info["wrist_yaw_in_release_range"] = bool(
            calib.wrist_yaw_release_min <= wrist_yaw_rad <= calib.wrist_yaw_release_max
        )
        info["fixed_joint_deviation"] = float(
            np.max(np.abs(current_arm_qpos[self._fixed_arm_local_idx] - self.nominal_ctrl[self.arm_actuator_ids[self._fixed_arm_local_idx]]))
        ) if self._fixed_arm_local_idx.size else 0.0

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
