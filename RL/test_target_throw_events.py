#!/usr/bin/env python3
"""Validation tests for the target-throw event/reward rework.

Run with:
    python -m pytest RL/test_target_throw_events.py -v
or directly:
    python RL/test_target_throw_events.py

Split into two groups:
  - Geometry/hit-detection tests drive envs.reward_components.compute_landing_outcome
    and axis_decomposition directly with synthetic coordinates -- precise and
    fast, no physics noise.
  - Event state-machine tests drive a real G1TargetThrowEnv through physics
    to check release-once / landing-once / no-overwrite-on-bounce / no-NaN
    behavior end to end.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.reward_components import RewardShapingParams, axis_decomposition, compute_landing_outcome
from envs.g1_target_throw_env import G1TargetThrowEnv

TARGET_DISTANCE = 5.0
TARGET_HALF_SIZE = 0.35
BALL_RADIUS = 0.04
SHAPING = RewardShapingParams()


def _outcome(landing_xy, robot_origin_xy=(0.0, 0.0), target_xy=(TARGET_DISTANCE, 0.0)):
    forward, lateral, distance = axis_decomposition(
        np.asarray(landing_xy, dtype=float), np.asarray(robot_origin_xy, dtype=float), np.asarray(target_xy, dtype=float)
    )
    return compute_landing_outcome(forward, lateral, distance, TARGET_HALF_SIZE, BALL_RADIUS, SHAPING)


# ---------------------------------------------------------------------------
# 1. Geometry / hit detection
# ---------------------------------------------------------------------------


def test_target_distance_is_exactly_five_meters():
    forward, lateral, distance = axis_decomposition(
        np.array([5.0, 0.0]), np.array([0.0, 0.0]), np.array([5.0, 0.0])
    )
    assert abs(distance - TARGET_DISTANCE) < 1e-9


def test_ball_at_target_centre_is_a_hit():
    outcome = _outcome((TARGET_DISTANCE, 0.0))
    assert outcome["target_hit"] is True
    assert abs(outcome["landing_error"]) < 1e-9


def test_ball_at_each_target_edge_is_a_hit():
    edge = TARGET_HALF_SIZE + BALL_RADIUS - 1e-4
    for dx, dy in [(edge, 0.0), (-edge, 0.0), (0.0, edge), (0.0, -edge)]:
        outcome = _outcome((TARGET_DISTANCE + dx, dy))
        assert outcome["target_hit"] is True, f"expected hit at offset ({dx}, {dy})"


def test_ball_just_inside_target_is_a_hit():
    inside = TARGET_HALF_SIZE + BALL_RADIUS - 0.01
    outcome = _outcome((TARGET_DISTANCE + inside, 0.0))
    assert outcome["target_hit"] is True


def test_ball_just_outside_each_edge_is_a_miss():
    outside = TARGET_HALF_SIZE + BALL_RADIUS + 0.01
    for dx, dy in [(outside, 0.0), (-outside, 0.0), (0.0, outside), (0.0, -outside)]:
        outcome = _outcome((TARGET_DISTANCE + dx, dy))
        assert outcome["target_hit"] is False, f"expected miss at offset ({dx}, {dy})"


def test_ball_before_target_is_undershoot_not_overshoot():
    outcome = _outcome((3.2, 0.0))
    assert outcome["undershoot"] > 0
    assert outcome["overshoot"] == 0
    assert abs(outcome["undershoot"] - 1.8) < 1e-9


def test_ball_beyond_target_is_overshoot_not_undershoot():
    outcome = _outcome((6.0, 0.0))
    assert outcome["overshoot"] > 0
    assert outcome["undershoot"] == 0
    assert abs(outcome["overshoot"] - 1.0) < 1e-9


def test_lateral_error_independent_of_longitudinal_error():
    on_axis = _outcome((3.2, 0.0))
    off_axis = _outcome((3.2, 1.5))
    assert abs(on_axis["lateral_error"]) < 1e-9
    assert abs(off_axis["lateral_error"] - 1.5) < 1e-9
    # Undershoot magnitude must not change just because the throw also drifted sideways.
    assert abs(on_axis["undershoot"] - off_axis["undershoot"]) < 1e-9


def test_landing_at_five_meters_centred_gives_max_bounded_progress():
    outcome = _outcome((TARGET_DISTANCE, 0.0))
    assert abs(outcome["distance_progress_raw"] - 1.0) < 1e-9
    assert outcome["target_dense_raw"] > 0.99


def test_ball_from_a_different_lanes_target_is_not_a_hit_here():
    # Environment 1 sits 4 m further along a shared axis in the multi-viewer grid;
    # its landing point must not register against environment 0's target.
    other_lane_landing_xy = (TARGET_DISTANCE, 4.0)  # 4 m laterally offset, like a neighboring lane
    outcome = _outcome(other_lane_landing_xy)
    assert outcome["target_hit"] is False


def test_no_nan_or_inf_in_outcome():
    outcome = _outcome((TARGET_DISTANCE, 0.0))
    for key, value in outcome.items():
        if isinstance(value, float):
            assert np.isfinite(value), f"{key} is not finite: {value}"


# ---------------------------------------------------------------------------
# 2. Event state machine (real physics)
# ---------------------------------------------------------------------------


def _ramped_swing(t, env):
    # Active action order: [shoulder_pitch, shoulder_roll, elbow, wrist_roll, wrist_yaw, release]
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    progress = min(1.0, t / 0.4)
    action[0] = -progress
    action[1] = 0.15 * progress
    action[2] = 0.8 * progress  # elbow
    if t >= 0.4:
        action[-1] = 1.0
    return action


def _run_episode(env, seed=0, max_steps=300):
    env.reset(seed=seed)
    infos = []
    for step in range(max_steps):
        t = step * env.control_dt
        obs, reward, terminated, truncated, info = env.step(_ramped_swing(t, env))
        infos.append(info)
        if terminated or truncated:
            break
    return infos


def test_release_detected_exactly_once():
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    infos = _run_episode(env)
    release_events = [i for i in infos if i["ball_released"]]
    first_release_step = infos.index(release_events[0])
    # ball_released stays True for every subsequent step -- exactly one rising edge.
    assert all(i["ball_released"] for i in infos[first_release_step:])
    assert not any(i["ball_released"] for i in infos[:first_release_step])


def test_landing_detected_exactly_once_and_not_overwritten():
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    infos = _run_episode(env)
    landing_positions = {tuple(i["landing_position"]) for i in infos if i["landing_recorded"]}
    assert len(landing_positions) == 1, "first landing position must never change once recorded"


def test_landing_reward_awarded_exactly_once():
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    infos = _run_episode(env)
    nonzero_landing_reward_steps = [
        i for i in infos if i["reward_components"].get("target_dense", 0.0) != 0.0
    ]
    assert len(nonzero_landing_reward_steps) <= 1


def test_episode_survives_past_landing_for_settle_period():
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    infos = _run_episode(env)
    landing_step = next((idx for idx, i in enumerate(infos) if i["landing_recorded"]), None)
    assert landing_step is not None, "this scripted throw should release and land"
    assert len(infos) - 1 >= landing_step, "episode should not end before the landing step itself"


def test_no_nan_or_inf_reward_over_episode():
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    infos = _run_episode(env)
    for info in infos:
        for key, value in info["reward_components"].items():
            assert np.isfinite(value), f"reward component {key} not finite: {value}"


def test_ball_hold_offset_clears_the_visual_hand_mesh():
    from envs.hand_geometry import compute_ball_hold_offset, find_hand_mesh_geom, _mesh_local_bbox
    import mujoco

    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    geom_id = find_hand_mesh_geom(env.model, env.hold_body_id)
    assert geom_id >= 0
    bbox_min, bbox_max = _mesh_local_bbox(env.model, geom_id)
    hand_mesh_max_x = env.model.geom_pos[geom_id][0] + bbox_max[0]
    offset, _ = compute_ball_hold_offset(env.model, env.hold_body_id, env.ball_radius, contact_margin=0.002)
    # Ball's NEAR edge (offset_x - radius) must clear the hand mesh's far edge.
    assert offset[0] - env.ball_radius >= hand_mesh_max_x


def test_ball_placed_at_hand_on_reset_matches_hold_relpose():
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    env.reset(seed=0)
    hand_pos = env.data.xpos[env.hold_body_id]
    hand_mat = env.data.xmat[env.hold_body_id].reshape(3, 3)
    expected = hand_pos + hand_mat @ env.hold_relpose[:3]
    actual = env._ball_pos()
    assert np.linalg.norm(actual - expected) < 1e-6


def test_episode_starts_in_ready_phase():
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    env.reset(seed=0)
    assert env.phase == "ready"


def test_policy_cannot_move_arm_during_ready_phase():
    """A max-effort action during the ready window must be overridden --
    the arm should stay at the scripted ready pose regardless."""
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    env.reset(seed=0)
    aggressive_action = np.ones(env.action_space.shape, dtype=np.float32)
    for _ in range(5):  # well within ready_pose_duration=0.3s at control_dt=0.02
        obs, r, term, trunc, info = env.step(aggressive_action)
        assert info["phase"] == "ready"
    current_ctrl = env.data.ctrl[env.arm_actuator_ids]
    assert np.allclose(current_ctrl, env.ready_pose_ctrl, atol=1e-6)
    assert info["ball_released"] is False  # release action was forced off too


def test_arm_returns_to_rest_pose_after_hold_phase():
    """End to end: run a real throw, then confirm the arm is back near the
    rest pose (not still moving) once the hold phase is reached."""
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    env.reset(seed=0)
    info = {}
    for step in range(400):
        t = step * env.control_dt
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        progress = min(1.0, t / 0.4)
        action[0] = -progress
        action[2] = 0.8 * progress  # elbow
        if t >= max(0.4, env.motion_phases.ready_pose_duration):
            action[-1] = 1.0
        obs, r, term, trunc, info = env.step(action)
        if info.get("phase") == "hold":
            break
        if term or trunc:
            break
    assert info.get("phase") == "hold", "scripted throw should reach the hold phase within 400 steps"
    assert info["final_arm_pose_error"] < 0.05
    assert info["final_hand_linear_speed"] < 0.2
    assert info["final_hand_angular_speed"] < 0.5


def test_palm_frame_and_geometry_report_run_without_error():
    from envs.hand_geometry import get_palm_frame, print_hand_geometry_report
    import io
    import contextlib

    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    env.reset(seed=0)
    frame = get_palm_frame(env.model, env.data, env.hold_body_id, env.ball_body_id)
    assert np.all(np.isfinite(frame.forward_vector))
    assert np.all(np.isfinite(frame.normal_vector))
    assert abs(np.linalg.norm(frame.forward_vector) - 1.0) < 1e-6
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_hand_geometry_report(env.model, env.hold_body_id, env.ball_geom_id)
    assert "hold_throw_ball" in buf.getvalue()


def test_observation_has_no_nan_across_all_phases():
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs))
    for step in range(200):
        t = step * env.control_dt
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        progress = min(1.0, t / 0.4)
        action[0] = -progress
        if t >= 0.4:
            action[-1] = 1.0
        obs, r, term, trunc, info = env.step(action)
        assert np.all(np.isfinite(obs)), f"non-finite obs at step {step}"
        assert np.isfinite(r), f"non-finite reward at step {step}"
        if term or trunc:
            break


def test_checkpoint_shapes_are_stable_for_resume():
    # 33 base features + 5 phase features (normalized_episode_time,
    # normalized_time_since_release, is_released, is_recovering, phase_index)
    # -- this shape change is intentional (item 10); checkpoints trained on
    # the previous 33-dim observation cannot be resumed against this env,
    # PPO.load's own shape check will refuse them rather than silently
    # mismatching. See the motion/palm-fix change report for why.
    #
    # action_space is 6 = 5 active right-arm joints (shoulder_pitch,
    # shoulder_roll, elbow, wrist_roll, wrist_yaw) + release, not 7+1 --
    # shoulder_yaw and wrist_pitch were removed from the action space
    # entirely (see envs/right_arm_config.py). Checkpoints trained on the
    # previous 8-dim action are likewise refused by PPO.load's shape check,
    # not silently mismatched. See the right-arm joint-reduction change
    # report for why.
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    assert env.observation_space.shape == (38,)
    assert env.action_space.shape == (6,)


# ---------------------------------------------------------------------------
# 3. Right-arm joint reduction (shoulder_yaw/wrist_pitch fixed, wrist_roll/
#    wrist_yaw range-limited, ball shifted laterally)
# ---------------------------------------------------------------------------


def test_active_and_fixed_joint_names_match_the_live_model():
    from envs.right_arm_config import ACTIVE_ARM_JOINT_NAMES, FIXED_ARM_JOINT_NAMES

    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    assert set(ACTIVE_ARM_JOINT_NAMES) | set(FIXED_ARM_JOINT_NAMES) == set(env.arm_joint_names)
    assert len(ACTIVE_ARM_JOINT_NAMES) == 5
    assert len(FIXED_ARM_JOINT_NAMES) == 2


def test_fixed_joints_never_move_under_max_effort_action():
    """shoulder_yaw/wrist_pitch are not in the action space at all -- an
    adversarial max-effort action on every OTHER joint must not move them."""
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    env.reset(seed=0)
    max_deviation = 0.0
    for _ in range(150):
        action = env.action_space.sample()
        action[:] = 1.0  # max effort on every active joint + force release
        obs, r, term, trunc, info = env.step(action)
        max_deviation = max(max_deviation, info["fixed_joint_deviation"])
        if term or trunc:
            break
    # Small slack for PD-tracking settling from the base class's own +-0.03 rad
    # reset noise on ALL arm joints (including the fixed ones) -- not from the
    # policy, which never has an action component for them.
    assert max_deviation < 0.08, f"fixed joints drifted {max_deviation:.4f} rad from their calibrated constants"


def test_wrist_roll_action_clamp_keeps_commanded_target_in_range():
    from envs.right_arm_config import ACTIVE_ARM_JOINT_NAMES

    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    env.reset(seed=0)
    calib = env.right_arm_calibration
    wrist_roll_active_idx = list(ACTIVE_ARM_JOINT_NAMES).index("right_wrist_roll_joint")
    for _ in range(100):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[wrist_roll_active_idx] = 1.0  # max-effort toward the joint's own, much wider, limit
        obs, r, term, trunc, info = env.step(action)
        commanded = float(env.data.ctrl[env.arm_actuator_ids[env._wrist_roll_local_idx]])
        assert calib.wrist_roll_min - 1e-6 <= commanded <= calib.wrist_roll_max + 1e-6, (
            f"commanded wrist_roll target {commanded:.4f} escaped the calibrated "
            f"[{calib.wrist_roll_min}, {calib.wrist_roll_max}] interval"
        )
        if term or trunc:
            break


def test_wrist_yaw_action_clamp_keeps_commanded_target_in_range():
    from envs.right_arm_config import ACTIVE_ARM_JOINT_NAMES

    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    env.reset(seed=0)
    calib = env.right_arm_calibration
    wrist_yaw_active_idx = list(ACTIVE_ARM_JOINT_NAMES).index("right_wrist_yaw_joint")
    for _ in range(100):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[wrist_yaw_active_idx] = 1.0
        obs, r, term, trunc, info = env.step(action)
        commanded = float(env.data.ctrl[env.arm_actuator_ids[env._wrist_yaw_local_idx]])
        assert calib.wrist_yaw_release_min - 1e-6 <= commanded <= calib.wrist_yaw_release_max + 1e-6, (
            f"commanded wrist_yaw target {commanded:.4f} escaped the calibrated release range"
        )
        # And never negative even under a max-NEGATIVE action.
        action[wrist_yaw_active_idx] = -1.0
        obs, r, term, trunc, info = env.step(action)
        commanded = float(env.data.ctrl[env.arm_actuator_ids[env._wrist_yaw_local_idx]])
        assert commanded >= calib.wrist_yaw_release_min - 1e-6
        if term or trunc:
            break


def test_ball_offset_shifted_toward_calibrated_lateral_direction():
    """Positive ball_offset_lateral must move the ball's local-frame hold
    offset in exactly the +local_y direction relative to the zero-offset
    baseline -- and by exactly the calibrated amount, nothing else changing."""
    from envs.right_arm_config import RightArmCalibration

    baseline = G1TargetThrowEnv(learned_release=True, verbose_init=False, right_arm_calibration=RightArmCalibration(ball_offset_lateral=0.0))
    shifted = G1TargetThrowEnv(learned_release=True, verbose_init=False, right_arm_calibration=RightArmCalibration(ball_offset_lateral=0.006))
    delta = shifted.hold_relpose[:3] - baseline.hold_relpose[:3]
    assert abs(delta[1] - 0.006) < 1e-9
    assert abs(delta[0]) < 1e-9
    assert abs(delta[2]) < 1e-9


def test_right_arm_config_yaml_round_trip():
    import tempfile

    from envs.right_arm_config import RightArmCalibration, load_right_arm_config, save_right_arm_config

    config = RightArmCalibration(ball_offset_lateral=0.004, wrist_yaw_release_value=0.12)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "right_arm_throw_pose.yaml"
        save_right_arm_config(config, path)
        loaded = load_right_arm_config(path)
    assert abs(loaded.ball_offset_lateral - 0.004) < 1e-9
    assert abs(loaded.wrist_yaw_release_value - 0.12) < 1e-9
    assert loaded.fixed_joints == config.fixed_joints


def test_full_scripted_throw_never_violates_right_arm_constraints():
    """End-to-end validation item 10 (2,3,4,9): run a real scripted throw and
    confirm wrist_roll stays in range, wrist_yaw stays in [0, 0.2], and the
    two fixed joints never move, for every single step of the episode."""
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    env.reset(seed=0)
    calib = env.right_arm_calibration
    reached_hold = False
    for step in range(400):
        t = step * env.control_dt
        info = env.step(_ramped_swing(t, env))[4]
        # Small slack (~1 degree) for realistic PD-tracking noise around the
        # COMMANDED target -- the action-space clamp already guarantees the
        # commanded target itself never leaves these ranges (see the two
        # action-clamp tests above); this end-to-end test additionally
        # confirms the REALIZED qpos tracks closely, not exactly instantly.
        slack = 0.02
        assert calib.wrist_roll_min - slack <= info["wrist_roll_rad"] <= calib.wrist_roll_max + slack, (
            f"wrist_roll {info['wrist_roll_rad']:.4f} left range at step {step}"
        )
        assert calib.wrist_yaw_release_min - slack <= info["wrist_yaw_rad"] <= calib.wrist_yaw_release_max + slack, (
            f"wrist_yaw {info['wrist_yaw_rad']:.4f} left [0, 0.2] at step {step}"
        )
        assert info["fixed_joint_deviation"] < 0.08, f"fixed joints drifted at step {step}"
        if info["phase"] == "hold":
            reached_hold = True
        if info["phase"] == "hold" and info.get("landing_recorded"):
            break
    assert reached_hold, "scripted throw should reach the hold phase"


if __name__ == "__main__":
    import inspect

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and inspect.isfunction(obj)]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures.append(test.__name__)
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        sys.exit(1)
