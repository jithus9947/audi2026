"""Modular, independently-tunable reward terms for the G1 target-throw task.

Every function returns a *raw*, roughly unit-scaled value (+-1, 0..1, or a
two-state flag). All magnitude tuning happens in ``RewardWeights`` -- no
function bakes its own scale in, so ``weight * raw_function(env)`` is always
the full story and weights can be re-tuned without touching logic or
double-counting an already-scaled number.

Two families of reward:

  Continuous, per-step (paid out every step, small by design so a policy
  that just stands there can't out-earn actually throwing):
    pelvis_fixed_reward, hand_forward_reward, hand_rotation_penalty,
    self_collision_penalty, feet_and_ground_contact_reward,
    projectile_straightness_reward, action_smoothness_penalty.

  One-time, event-triggered (paid out exactly once, on the step release or
  first-landing is confirmed -- see G1TargetThrowEnv's event state machine):
    release_speed_reward (at release)
    compute_landing_outcome(...) -> distance_progress, target_dense,
    target_longitudinal, target_lateral, accuracy_progress,
    undershoot_penalty, overshoot_penalty, target_hit_bonus/near_target_bonus
    (at first landing).

``EventDetectionConfig`` holds the thresholds for the release/landing state
machine (envs/g1_target_throw_env.py); ``RewardShapingParams`` holds the
scale constants (decay rates, desired speeds, bonus tiers) the landing-event
formulas use. Keeping thresholds/scales/weights in three separate dataclasses
means every number the task depends on is named, defaulted, and overridable
from one place instead of scattered through the reward logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class RewardWeights:
    """Per-term multipliers over the raw component functions.

    Values mostly follow the rebalanced defaults from the first training run
    (see the module docstring in the previous revision) plus the requested
    landing/event-based terms layered on top. ``pelvis_fixed``/``feet_contact``
    stay small -- they pay out every step, so at full weight a "do nothing"
    episode out-earns an actual throw attempt (verified empirically: a
    150-step do-nothing episode scored ~300 at weight 1.0 vs. ~76 for a real
    throw attempt). Landing-event terms are one-time per episode, so they can
    carry much larger weights without creating that same exploit.

    target_longitudinal/target_lateral (logged tags) are dense per-axis
    breakdowns of the same landing error that target_dense already scores in
    combination. target_lateral reuses the ``lateral_accuracy`` weight
    (rather than getting its own) so the same lateral-accuracy signal isn't
    counted under two different multipliers; target_longitudinal gets its
    own modest weight since there's no equivalent duplicate. Keep both
    modest (default 1.0) to avoid double-rewarding the same landing accuracy
    three different ways (target_dense + target_longitudinal + target_lateral
    all respond to the same landing).
    """

    # Continuous, per-step. pelvis_fixed/feet_contact cut hard (0.20 -> 0.03 /
    # 0.02): a "do nothing" episode runs the FULL ~150-step episode collecting
    # these every step, while a throw ends after ~50-65 steps (landing +
    # settle) -- verified empirically that with these two at 0.10/0.05 (an
    # earlier attempt at this same rebalance) plus the new 5x undershoot
    # penalty below, the CURRENT policy's actual throws (~1.85m undershoot)
    # scored only ~22-27 total vs. ~22.4 for never releasing at all -- a real
    # risk of PPO regressing toward inaction (which item 8 explicitly
    # forbids: 100% release rate must be preserved). At 0.03/0.02 the
    # do-nothing baseline drops to ~7.5, comfortably below any real throw.
    pelvis_fixed: float = 0.03
    hand_forward: float = 0.50
    hand_rotation: float = 0.25
    self_collision: float = 2.0
    feet_contact: float = 0.02
    bad_ground_contact: float = 5.0
    projectile_straightness: float = 0.75
    action_smoothness: float = 0.05
    fall_penalty: float = 10.0

    # One-time, at release.
    release_speed: float = 1.0
    premature_release_penalty: float = 2.0

    # One-time, at first landing. target_dense/target_longitudinal/undershoot_penalty
    # raised (4x/5x/5x) after a real training run showed the policy converging
    # to a stable ~3.1 m short throw: the per-step stability terms above were
    # comparable in scale to these one-time landing terms once diluted by
    # episode-length averaging, so there was little pressure to trade a "safe"
    # short throw for a riskier, more accurate one. overshoot_penalty raised
    # to a similar (slightly lower) magnitude so overshooting isn't free once
    # the policy does start throwing harder.
    distance_progress: float = 3.0
    target_dense: float = 16.0        # was 4.0 (4x)
    target_longitudinal: float = 5.0  # was 1.0 (5x)
    lateral_accuracy: float = 2.0     # also used for the "target_lateral" logged tag; unchanged, lateral error already good
    accuracy_progress: float = 2.0
    undershoot_penalty: float = 3.75  # was 0.75 (5x)
    overshoot_penalty: float = 3.5    # was 1.0; kept slightly below undershoot per spec ("symmetric or slightly weaker")
    target_hit_bonus: float = 15.0


@dataclass
class EventDetectionConfig:
    """Thresholds for the release/first-landing state machine."""

    release_min_hand_distance: float = 0.12  # m, ball must clear the hand by this much
    release_min_speed: float = 0.5           # m/s, ball must already be moving
    release_confirm_steps: int = 2           # consecutive steps the above must hold
    min_windup_time: float = 0.35            # s; releasing before this incurs premature_release_penalty.
    # Without this, PPO discovers that releasing on ~step 3 (before the arm
    # has moved at all) still gets the ball some velocity "for free" from the
    # weld constraint's correction physics rather than a genuine swing --
    # confirmed by rendering the actual policy (release at t=0.06s, arm still
    # at the hip). This threshold makes that a worse strategy than winding up.
    min_flight_time: float = 0.15            # s after release before landing can register
    allowed_vertical_velocity: float = 0.05  # m/s, |vz| below this counts as "settled" near the floor
    ground_contact_tolerance: float = 0.02   # m, extra slack above ball radius for "near ground"
    settle_steps_after_landing: int = 15     # extra steps kept alive after first landing for bounce/roll logging
    out_of_bounds_radius: float = 25.0       # m from launch point; ball beyond this ends the episode


@dataclass
class RewardShapingParams:
    """Scale constants used by the one-time landing/release reward formulas."""

    target_reward_scale: float = 1.0   # target_dense decay length, meters
    longitudinal_scale: float = 1.0    # target_longitudinal decay length, meters
    lateral_scale: float = 0.5         # target_lateral / lateral_accuracy decay length, meters
    lateral_penalty_weight: float = 0.5  # folded into target_lateral, see compute_landing_outcome
    # release_speed_reward now derives its desired speed from projectile
    # physics (desired_release_speed()) using the ACTUAL release angle/height
    # each episode, not this fixed constant. A hardcoded desired_forward_speed
    # of 4.0 m/s was the actual bug behind the 3.1 m plateau: real release
    # speeds cluster at 4.6-5.0 m/s, so the reward was already saturated at
    # 1.0 the whole time -- zero gradient to throw harder. Kept only as the
    # fallback when the physics solution is degenerate (near-zero angle).
    desired_forward_speed_fallback: float = 7.0
    min_release_angle_deg: float = 5.0   # guard against div-by-~0 in the physics solution at very flat angles
    max_desired_release_speed: float = 15.0  # sane upper clamp on the physics-derived desired speed
    gravity: float = 9.81
    initial_reference_error: float | None = None  # None -> use the configured target distance
    near_target_tiers: tuple[tuple[float, float], ...] = (
        (0.25, 10.0),
        (0.50, 7.0),
        (1.00, 4.0),
        (1.50, 1.5),
    )


def _body_name(model, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""


# ---------------------------------------------------------------------------
# Continuous, per-step components.
# ---------------------------------------------------------------------------


def pelvis_fixed_reward(env) -> float:
    """Level 0: +1 while the pelvis stays put and does not turn, else -1."""
    pelvis_pos = env.data.xpos[env.base_body_id]
    pelvis_quat = env.data.xquat[env.base_body_id]
    pos_drift = float(np.linalg.norm(pelvis_pos[:2] - env.pelvis_home_pos[:2]))
    quat_diff = np.empty(3)
    mujoco.mju_subQuat(quat_diff, pelvis_quat, env.pelvis_home_quat)
    turn_angle = float(np.linalg.norm(quat_diff))
    stable = pos_drift <= env.pelvis_pos_tolerance and turn_angle <= env.pelvis_turn_tolerance
    return 1.0 if stable else -1.0


def hand_forward_reward(env, scale: float = 1.0) -> float:
    """Reward the wrist moving forward (+x), penalize it moving backward."""
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, env.hold_body_id, vel, 0)
    forward_speed = float(vel[3])  # world-frame linear x, index 3:6 is the linear part
    return float(np.tanh(forward_speed / scale))


def hand_rotation_penalty(env, scale: float = 3.0) -> float:
    """Penalize wrist rotation that is not the intended forward swing."""
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, env.hold_body_id, vel, 0)
    lateral_angular = np.array([vel[0], vel[2]])  # angular part is vel[0:3]
    return -float(np.tanh(np.linalg.norm(lateral_angular) / scale))


def self_collision_penalty(env) -> float:
    """-1 this step if the right arm chain contacts any other body part."""
    arm_ids = env.arm_body_ids
    exempt_ids = env.arm_body_ids | {env.ball_body_id}
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        b1 = env.model.geom_bodyid[contact.geom1]
        b2 = env.model.geom_bodyid[contact.geom2]
        if b1 in arm_ids and b2 not in exempt_ids:
            return -1.0
        if b2 in arm_ids and b1 not in exempt_ids:
            return -1.0
    return 0.0


def feet_and_ground_contact_reward(env) -> tuple[float, float]:
    """(+1 if standing on >=1 foot, -1 if any other body part touches the floor)."""
    if env.floor_geom_id < 0:
        return 0.0, 0.0
    feet_touching = False
    bad_touch = False
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if g1 == env.floor_geom_id:
            other_body = env.model.geom_bodyid[g2]
        elif g2 == env.floor_geom_id:
            other_body = env.model.geom_bodyid[g1]
        else:
            continue
        if other_body in env.foot_body_ids:
            feet_touching = True
        elif other_body != env.ball_body_id:
            bad_touch = True
    feet_reward = 1.0 if feet_touching and not bad_touch else 0.0
    bad_reward = -1.0 if bad_touch else 0.0
    return feet_reward, bad_reward


def projectile_straightness_reward(env) -> float:
    """Reward a clean forward+upward release: not sideways, not dumped down.

    Reads the ball free-joint qvel directly (first 3 dof = linear velocity)
    rather than the base env's ``_ball_vel`` helper, whose returned slice is
    the free joint's angular component, not linear -- not something this
    module touches or relies on being changed.
    """
    linvel = env.data.qvel[env.ball_qvel_adr : env.ball_qvel_adr + 3]
    forward, lateral, up = float(linvel[0]), float(linvel[1]), float(linvel[2])
    straightness = max(0.0, 1.0 - abs(lateral) / (abs(forward) + 1e-3))
    arc = 0.5 * np.tanh(forward / 3.0) + 0.5 * np.tanh(max(up, 0.0) / 2.0)
    return float(straightness * arc)


def action_smoothness_penalty(env, action: np.ndarray) -> float:
    """Small, roughly [-1, 0]-scaled penalty on large/jerky arm commands."""
    arm_action = action[: env.n_arm]
    delta = action - env.prev_action
    n = max(len(arm_action), 1)
    return float(
        -0.5 * np.linalg.norm(arm_action) / np.sqrt(n)
        - 0.5 * np.linalg.norm(delta) / np.sqrt(max(len(delta), 1))
    )


# ---------------------------------------------------------------------------
# One-time, event-triggered components.
# ---------------------------------------------------------------------------


def desired_release_speed(
    target_distance: float,
    release_height: float,
    release_angle_rad: float,
    shaping: RewardShapingParams,
) -> float:
    """Projectile-range solution: the launch speed that reaches ``target_distance``
    from ``release_height`` at ``release_angle_rad``, landing at ground level.

    Standard range-with-height formula solved for v, given R (target_distance),
    h (release_height), theta (release_angle_rad), g (gravity):

        R = v*cos(theta)/g * (v*sin(theta) + sqrt((v*sin(theta))**2 + 2*g*h))

    solved for v:

        v = R * sqrt(g / (2*cos(theta)*(R*sin(theta) + h*cos(theta))))

    Uses the CURRENT episode's actual release angle/height (not a fixed
    assumption), so "desired speed" always means "what this exact throw
    would have needed to reach the target" -- verified against the
    suggested 6.5-8.0 m/s range: at target_distance=5m, release_height=0.55m,
    release_angle=21.5 deg (this task's observed average), this formula
    gives ~7.5 m/s, right in that range.
    """
    angle = max(release_angle_rad, np.radians(shaping.min_release_angle_deg))
    s, c = np.sin(angle), np.cos(angle)
    denom = 2.0 * c * (target_distance * s + max(release_height, 0.0) * c)
    if denom <= 1e-6 or target_distance <= 0:
        return shaping.desired_forward_speed_fallback
    v = target_distance * np.sqrt(shaping.gravity / denom)
    return float(np.clip(v, 0.5, shaping.max_desired_release_speed))


def release_speed_reward(release_forward_speed: float, desired_speed: float) -> float:
    """Raw in [0, 1]; smooth linear ramp to the physics-derived desired speed,
    clamped only at the top so overshooting it isn't extra-rewarded (no hard
    binary threshold -- continuous everywhere below and at the target)."""
    if desired_speed <= 0:
        return 0.0
    return float(np.clip(release_forward_speed / desired_speed, 0.0, 1.0))


def premature_release_penalty_raw(release_time: float, event_config: EventDetectionConfig) -> float:
    """Raw in [0, 1]: how far short of the minimum wind-up time release happened.

    0 if release_time >= min_windup_time (no penalty for a real wind-up);
    grows linearly to 1 the earlier release happens before that, capped so an
    instant (t=0) release is the worst case rather than an unbounded penalty.
    """
    if event_config.min_windup_time <= 0:
        return 0.0
    shortfall = max(0.0, event_config.min_windup_time - release_time)
    return float(np.clip(shortfall / event_config.min_windup_time, 0.0, 1.0))


def axis_decomposition(landing_xy: np.ndarray, robot_origin_xy: np.ndarray, target_xy: np.ndarray):
    """Decompose a landing point into (forward_distance, lateral_offset, target_distance).

    forward_distance / lateral_offset are measured along / perpendicular to
    the robot-origin -> target centreline, so they stay meaningful even if a
    lane's target isn't exactly on the world +x axis.
    """
    to_target = target_xy - robot_origin_xy
    target_distance = float(np.linalg.norm(to_target))
    forward_axis = to_target / target_distance if target_distance > 1e-9 else np.array([1.0, 0.0])
    lateral_axis = np.array([-forward_axis[1], forward_axis[0]])
    relative = landing_xy - robot_origin_xy
    forward_distance = float(np.dot(relative, forward_axis))
    lateral_offset = float(np.dot(relative, lateral_axis))
    return forward_distance, lateral_offset, target_distance


def compute_landing_outcome(
    landing_forward_distance: float,
    lateral_offset: float,
    target_distance: float,
    target_half_size: float,
    ball_radius: float,
    shaping: RewardShapingParams,
) -> dict:
    """Everything needed for reward + logging from one first-landing event.

    Returns raw (unweighted) values under ``*_raw`` plus the already-derived
    diagnostic numbers (errors, hit flags) so the caller can weight, log, and
    fill the info dict from a single source of truth.
    """
    longitudinal_error = landing_forward_distance - target_distance
    lateral_error = lateral_offset
    landing_error = float(np.hypot(longitudinal_error, lateral_error))

    distance_progress_raw = float(np.clip(landing_forward_distance / target_distance, 0.0, 1.0)) if target_distance > 0 else 0.0
    target_dense_raw = float(np.exp(-landing_error / max(shaping.target_reward_scale, 1e-6)))
    target_longitudinal_raw = float(np.exp(-abs(longitudinal_error) / max(shaping.longitudinal_scale, 1e-6)))
    target_lateral_raw = float(
        np.exp(-abs(lateral_error) / max(shaping.lateral_scale, 1e-6))
        - shaping.lateral_penalty_weight * abs(lateral_error)
    )
    lateral_accuracy_raw = float(np.exp(-abs(lateral_error) / max(shaping.lateral_scale, 1e-6)))

    reference_error = shaping.initial_reference_error if shaping.initial_reference_error is not None else target_distance
    accuracy_progress_raw = (
        float(np.clip((reference_error - landing_error) / reference_error, 0.0, 1.0)) if reference_error > 0 else 0.0
    )

    undershoot = max(0.0, target_distance - landing_forward_distance)
    overshoot = max(0.0, landing_forward_distance - target_distance)

    target_hit = bool(
        abs(longitudinal_error) <= target_half_size + ball_radius
        and abs(lateral_error) <= target_half_size + ball_radius
    )
    near_target_bonus = 0.0
    if not target_hit:
        for threshold, bonus in shaping.near_target_tiers:
            if landing_error <= threshold:
                near_target_bonus = bonus
                break

    return {
        "longitudinal_error": longitudinal_error,
        "lateral_error": lateral_error,
        "landing_error": landing_error,
        "distance_progress_raw": distance_progress_raw,
        "target_dense_raw": target_dense_raw,
        "target_longitudinal_raw": target_longitudinal_raw,
        "target_lateral_raw": target_lateral_raw,
        "lateral_accuracy_raw": lateral_accuracy_raw,
        "accuracy_progress_raw": accuracy_progress_raw,
        "undershoot": undershoot,
        "overshoot": overshoot,
        "target_hit": target_hit,
        "near_target_bonus": near_target_bonus,
    }


# ---------------------------------------------------------------------------
# Startup validation / reporting (item 12).
# ---------------------------------------------------------------------------


def print_reward_weight_report(
    weights: RewardWeights,
    shaping: RewardShapingParams,
    target_distance: float,
    target_half_size: float,
    ball_radius: float,
    assumed_release_angle_deg: float = 25.0,
    assumed_release_height: float = 0.55,
) -> None:
    """Print the full weight table plus sanity warnings. Call this ONCE
    (e.g. in the training script's main(), before spawning parallel workers)
    -- not per-env-instance, or it prints once per subprocess.
    """
    print("=" * 72)
    print("REWARD WEIGHT CONFIGURATION")
    print("=" * 72)
    print("Continuous, per-step:")
    for name in ("pelvis_fixed", "hand_forward", "hand_rotation", "self_collision",
                 "feet_contact", "bad_ground_contact", "projectile_straightness",
                 "action_smoothness", "fall_penalty"):
        print(f"  {name:28s} {getattr(weights, name):8.3f}")
    print("One-time, at release:")
    for name in ("release_speed", "premature_release_penalty"):
        print(f"  {name:28s} {getattr(weights, name):8.3f}")
    print("One-time, at first landing:")
    for name in ("distance_progress", "target_dense", "target_longitudinal", "lateral_accuracy",
                 "accuracy_progress", "undershoot_penalty", "overshoot_penalty", "target_hit_bonus"):
        print(f"  {name:28s} {getattr(weights, name):8.3f}")
    print("-" * 72)

    warnings: list[str] = []

    # Sign sanity: penalty weights must be positive (they're negated at the
    # call site) -- a negative weight here would silently flip a penalty into
    # a reward for undershooting/overshooting/falling/colliding.
    for name in ("undershoot_penalty", "overshoot_penalty", "fall_penalty", "self_collision",
                 "bad_ground_contact", "premature_release_penalty"):
        value = getattr(weights, name)
        if value < 0:
            warnings.append(f"{name} is NEGATIVE ({value}) -- this will flip the penalty into a reward.")

    # Stability-vs-task dominance: compare a per-step term's PER-EPISODE
    # ceiling (weight * ~150 steps) against a one-time term's single payout,
    # since that's the comparison that actually matters for the do-nothing
    # exploit (a sparse one-time term looks tiny next to a per-step term's
    # raw weight, but pays out once vs. every step).
    approx_episode_steps = 150
    stability_ceiling = approx_episode_steps * (weights.pelvis_fixed + weights.feet_contact)
    task_ceiling = weights.target_dense + weights.target_longitudinal + weights.undershoot_penalty * 2.0
    if stability_ceiling > task_ceiling:
        warnings.append(
            f"pelvis_fixed+feet_contact accumulated over ~{approx_episode_steps} steps "
            f"({stability_ceiling:.2f}) exceeds a typical one-time landing payout ({task_ceiling:.2f}) -- "
            "risk of a 'never release' exploit outscoring any real throw. Lower pelvis_fixed/feet_contact "
            "or shorten episode_time."
        )

    # Release-speed saturation: would the reward already be maxed at a speed
    # this task can plausibly achieve, using a representative angle/height?
    representative_desired = desired_release_speed(
        target_distance, assumed_release_height, np.radians(assumed_release_angle_deg), shaping
    )
    if representative_desired <= 5.5:
        warnings.append(
            f"desired_release_speed at a representative {assumed_release_angle_deg}deg/{assumed_release_height}m "
            f"release is only {representative_desired:.2f} m/s -- release_speed_reward would saturate at a very "
            "achievable speed, giving little incentive to throw harder. Check target_distance/assumed angle."
        )

    # Hit-bonus tolerance sanity: is the hit box big enough to ever plausibly
    # register given typical release-speed noise, or so tiny it's practically
    # unreachable at this target distance?
    hit_half_width = target_half_size + ball_radius
    if hit_half_width < 0.05:
        warnings.append(
            f"target hit tolerance ({hit_half_width:.3f}m half-width) is extremely tight for a {target_distance}m "
            "throw -- target_hit_bonus may be practically unreachable; consider near_target_tiers as the primary "
            "shaping signal instead."
        )

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("No reward-weight warnings.")
    print("=" * 72)
