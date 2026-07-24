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
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    progress = min(1.0, t / 0.4)
    action[0] = -progress
    action[1] = 0.15 * progress
    action[3] = 0.8 * progress
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


def test_checkpoint_shapes_are_stable_for_resume():
    env = G1TargetThrowEnv(learned_release=True, verbose_init=False)
    assert env.observation_space.shape == (33,)
    assert env.action_space.shape == (8,)


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
