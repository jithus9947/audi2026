#!/usr/bin/env python3
"""Single-environment visual debugging: target, release point, ball
trajectory, first-landing point, and the landing-to-target error line, drawn
live in the MuJoCo window.

Markers are injected into the viewer's ad-hoc scene (``viewer.user_scn`` +
``mujoco.mjv_initGeom``/``mjv_connector``) every frame -- nothing is added to
the compiled model, so this cannot affect training physics and touches no
XML file. Evaluation/debugging only.

    python RL/visual_debug.py --model RL/runs/target_throw/final_model.zip
    python RL/visual_debug.py  # no --model: random actions, useful to check geometry alone
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer
import numpy as np

from envs.g1_target_throw_env import G1TargetThrowEnv
from envs.hand_geometry import get_palm_frame

TARGET_RGBA = np.array([0.1, 0.9, 0.2, 0.5], dtype=np.float32)
RELEASE_RGBA = np.array([0.9, 0.9, 0.1, 1.0], dtype=np.float32)
LANDING_RGBA = np.array([0.9, 0.1, 0.1, 1.0], dtype=np.float32)
TRAJECTORY_RGBA = np.array([0.2, 0.6, 1.0, 0.8], dtype=np.float32)
ERROR_LINE_RGBA = np.array([1.0, 1.0, 1.0, 0.9], dtype=np.float32)
PALM_AXIS_RGBA = (
    np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),  # forward (local +x): red
    np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32),  # local +y: green
    np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32),  # normal (local +z): blue
)
EXPECTED_BALL_RGBA = np.array([1.0, 0.6, 0.0, 0.6], dtype=np.float32)
IDENTITY_MAT = np.eye(3).flatten()


def _add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, dtype=np.float64), IDENTITY_MAT, rgba)
    scn.ngeom += 1


def _add_line(scn, start, end, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), IDENTITY_MAT, rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64))
    scn.ngeom += 1


def _add_palm_frame(scn, position, rot_mat, length=0.12):
    """Item 14: palm coordinate frame -- red=forward (local +x, the axis the
    hold offset and hand mesh both align with), green=local +y, blue=surface
    normal convention (local +z). See envs/hand_geometry.py for why these
    conventions were chosen (no independent palm DOF in this model)."""
    for axis in range(3):
        direction = rot_mat[:, axis]
        _add_line(scn, position, position + direction * length, 2.5, PALM_AXIS_RGBA[axis])


def _add_box_outline(scn, center_xy, half_size, z, rgba):
    corners = [
        (center_xy[0] - half_size, center_xy[1] - half_size),
        (center_xy[0] + half_size, center_xy[1] - half_size),
        (center_xy[0] + half_size, center_xy[1] + half_size),
        (center_xy[0] - half_size, center_xy[1] + half_size),
    ]
    for i in range(4):
        a = (*corners[i], z)
        b = (*corners[(i + 1) % 4], z)
        _add_line(scn, a, b, 3.0, rgba)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=None, help="PPO .zip; omit to run random actions.")
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--target-half-size", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = G1TargetThrowEnv(
        learned_release=True,
        target_pos=(args.target_distance, 0.0, 0.01),
        target_half_size=args.target_half_size,
        verbose_init=True,
    )
    model = None
    if args.model is not None:
        from stable_baselines3 import PPO

        model = PPO.load(str(args.model), device="cpu")

    obs, _ = env.reset(seed=args.seed)
    episode = 1
    trajectory: list[np.ndarray] = []
    last_phase = env.phase
    print("Visual debug running. Close the MuJoCo window or press Ctrl+C to stop.")
    print(f"[episode {episode}] phase=ready")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if info["ball_released"]:
                trajectory.append(env._ball_pos().copy())
            if info["phase"] != last_phase:
                print(f"[episode {episode}] phase={last_phase} -> {info['phase']} (t={env.step_count * env.control_dt:.3f}s)")
                last_phase = info["phase"]

            viewer.user_scn.ngeom = 0
            target_xy = np.asarray(env.target_pos[:2])
            _add_box_outline(viewer.user_scn, target_xy, env.target_half_size, env.target_pos[2] + 0.01, TARGET_RGBA)
            _add_sphere(viewer.user_scn, (*target_xy, env.target_pos[2] + 0.02), 0.03, TARGET_RGBA)

            # Item 14: palm coordinate frame + expected vs actual ball centre.
            hand_pos = env.data.xpos[env.hold_body_id]
            hand_mat = env.data.xmat[env.hold_body_id].reshape(3, 3)
            _add_palm_frame(viewer.user_scn, hand_pos, hand_mat)
            if not info["ball_released"]:
                expected_ball_pos = hand_pos + hand_mat @ env.hold_relpose[:3]
                _add_sphere(viewer.user_scn, expected_ball_pos, env.ball_radius * 0.6, EXPECTED_BALL_RGBA)

            if info["release_position"] is not None:
                _add_sphere(viewer.user_scn, info["release_position"], 0.03, RELEASE_RGBA)
            for i in range(1, len(trajectory)):
                _add_line(viewer.user_scn, trajectory[i - 1], trajectory[i], 2.0, TRAJECTORY_RGBA)
            if info["landing_position"] is not None:
                landing_xyz = (*info["landing_position"], info["landing_z"])
                _add_sphere(viewer.user_scn, landing_xyz, 0.04, LANDING_RGBA)
                _add_line(viewer.user_scn, landing_xyz, (*target_xy, env.target_pos[2] + 0.02), 2.0, ERROR_LINE_RGBA)

            viewer.sync()
            time.sleep(env.control_dt / args.speed)

            if terminated or truncated:
                print(
                    f"Episode {episode}: released={info['ball_released']} "
                    f"landing_error={info['landing_error']} target_hit={info['square_target_hit']} "
                    f"longitudinal_error={info['longitudinal_error']} lateral_error={info['lateral_error']} "
                    f"final_arm_pose_error={info['final_arm_pose_error']:.4f} "
                    f"final_hand_linear_speed={info['final_hand_linear_speed']:.4f} "
                    f"termination={info['termination_reason']}"
                )
                episode += 1
                trajectory = []
                last_phase = None
                obs, _ = env.reset(seed=args.seed + episode - 1)
                print(f"[episode {episode}] phase=ready")


if __name__ == "__main__":
    main()
