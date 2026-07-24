#!/usr/bin/env python3
"""Interactive visual calibration for the right-arm throw pose and ball offset
(item 7 of the right-arm joint-reduction change).

    python RL/calibrate_right_arm_pose.py
    python RL/calibrate_right_arm_pose.py --load configs/right_arm_throw_pose.yaml

Opens a live MuJoCo window with keyboard controls for the 5 policy-controlled
joints (shoulder_pitch/roll, elbow, wrist_roll, wrist_yaw), the 2 fixed
joints (shoulder_yaw, wrist_pitch), and the ball's local-frame hold offset
(forward/lateral/vertical). Every change is applied live -- direct
``data.ctrl``/weld-relpose writes via ``mujoco.mj_step``, exactly the same
runtime-override mechanism G1TargetThrowEnv itself uses, nothing in the
protected base class or XML is touched -- and the current numeric state is
printed to the terminal after every change. 'P' saves the current state to
YAML (default configs/right_arm_throw_pose.yaml); 'R' resets to the state the
tool started from.

This does NOT run the policy or the phase state machine -- it drives
data.ctrl directly via a raw mj_step loop, bypassing G1TargetThrowEnv.step()
entirely, so joint targets are exactly and only what you set here. It uses a
G1TargetThrowEnv instance purely as a convenient, already-wired-up source of
body/joint/actuator ids and the same hold-offset geometry math the real
training env uses.

Controls (radians per keypress = 0.01 for joints, meters = 0.001 for ball
offsets; hold Shift on the letter for a 10x coarse step):
    shoulder_pitch   Q / A
    shoulder_roll    W / S
    elbow            E / D
    wrist_roll       R / F
    wrist_yaw        T / G
    shoulder_yaw*    Y / H   (*fixed joint)
    wrist_pitch*     U / J   (*fixed joint)
    ball forward     I / K
    ball lateral     O / L   (positive = robot's own left, see hand_geometry.py)
    ball vertical    ]  / [
    P   save to YAML
    C   print current config
    X   reset to the pose this session started from
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer
import numpy as np

from envs.g1_target_throw_env import G1TargetThrowEnv
from envs.hand_geometry import compute_ball_hold_offset
from envs.right_arm_config import (
    ACTIVE_ARM_JOINT_NAMES,
    DEFAULT_CONFIG_PATH,
    FIXED_ARM_JOINT_NAMES,
    load_right_arm_config,
    save_right_arm_config,
)

JOINT_STEP = 0.01
BALL_STEP = 0.001
COARSE_MULTIPLIER = 10.0

# GLFW key codes for the letters used below (uppercase ASCII == GLFW keycode
# for A-Z; mujoco.viewer's key_callback receives the GLFW code, not a str).
KEY = {c: ord(c) for c in "QWERTYUIOPASDFGHJKLZXCVBNM"}
KEY_LBRACKET = ord("[")
KEY_RBRACKET = ord("]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load", type=Path, default=None, help="YAML to start from. Default: configs/right_arm_throw_pose.yaml if it exists, else built-in defaults.")
    parser.add_argument("--save-path", type=Path, default=DEFAULT_CONFIG_PATH, help="Where 'P' saves to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib = load_right_arm_config(args.load) if args.load is not None else load_right_arm_config()

    env = G1TargetThrowEnv(learned_release=True, verbose_init=False, right_arm_calibration=calib)
    env.reset(seed=0)

    # Working joint-angle state (radians), keyed by joint name, seeded from
    # the env's own resolved starting ctrl (already reflects the loaded
    # calibration's wrist_roll_nominal / fixed_joints / wrist_yaw hold value).
    joint_targets = {
        name: float(env.data.ctrl[env.arm_actuator_ids[env.arm_joint_names.index(name)]])
        for name in ACTIVE_ARM_JOINT_NAMES + FIXED_ARM_JOINT_NAMES
    }
    ball_offset = {
        "forward": calib.ball_offset_forward,
        "lateral": calib.ball_offset_lateral,
        "vertical": calib.ball_offset_vertical,
    }
    initial_joint_targets = dict(joint_targets)
    initial_ball_offset = dict(ball_offset)

    def apply_joints() -> None:
        for name, value in joint_targets.items():
            aid = env.arm_actuator_ids[env.arm_joint_names.index(name)]
            env.data.ctrl[aid] = value
            env.data.qpos[env.arm_qpos_adr[env.arm_joint_names.index(name)]] = value
            env.data.qvel[env.arm_qvel_adr[env.arm_joint_names.index(name)]] = 0.0

    def apply_ball_offset() -> None:
        hold_offset, hold_quat = compute_ball_hold_offset(
            env.model, env.hold_body_id, env.ball_radius, env.motion_phases.ball_contact_margin,
            extra_offset_local=(ball_offset["forward"], ball_offset["lateral"], ball_offset["vertical"]),
        )
        env.hold_relpose = np.concatenate([hold_offset, hold_quat])
        env.model.eq_data[env.hold_eq_id][3:10] = env.hold_relpose
        mujoco.mj_forward(env.model, env.data)
        env._place_ball_in_hand()
        mujoco.mj_forward(env.model, env.data)

    def print_status() -> None:
        print("-" * 72)
        print("ACTIVE joints (rad):")
        for name in ACTIVE_ARM_JOINT_NAMES:
            print(f"  {name:30s} {joint_targets[name]:8.4f}")
        print("FIXED joints (rad):")
        for name in FIXED_ARM_JOINT_NAMES:
            print(f"  {name:30s} {joint_targets[name]:8.4f}")
        wr = joint_targets["right_wrist_roll_joint"]
        wy = joint_targets["right_wrist_yaw_joint"]
        print(
            f"wrist_roll in calibrated [{calib.wrist_roll_min:.3f},{calib.wrist_roll_max:.3f}]: "
            f"{'YES' if calib.wrist_roll_min <= wr <= calib.wrist_roll_max else 'NO -- out of range'}"
        )
        print(
            f"wrist_yaw in calibrated [{calib.wrist_yaw_release_min:.3f},{calib.wrist_yaw_release_max:.3f}]: "
            f"{'YES' if calib.wrist_yaw_release_min <= wy <= calib.wrist_yaw_release_max else 'NO -- out of range'}"
        )
        print(f"ball_offset_local: forward={ball_offset['forward']:+.4f}  lateral={ball_offset['lateral']:+.4f}  vertical={ball_offset['vertical']:+.4f}")
        ball_dist = float(np.linalg.norm(env._ball_pos() - env.data.xpos[env.hold_body_id]))
        print(f"ball-to-hand-origin distance: {ball_dist:.4f} m")
        print("-" * 72)

    def save() -> None:
        calib.wrist_roll_nominal = joint_targets["right_wrist_roll_joint"]
        calib.wrist_yaw_hold = 0.0  # hold value stays 0 by design; release_value is what's being posed here
        calib.wrist_yaw_release_value = joint_targets["right_wrist_yaw_joint"] if joint_targets["right_wrist_yaw_joint"] > 0 else calib.wrist_yaw_release_value
        calib.fixed_joints["right_shoulder_yaw_joint"] = joint_targets["right_shoulder_yaw_joint"]
        calib.fixed_joints["right_wrist_pitch_joint"] = joint_targets["right_wrist_pitch_joint"]
        calib.ball_offset_forward = ball_offset["forward"]
        calib.ball_offset_lateral = ball_offset["lateral"]
        calib.ball_offset_vertical = ball_offset["vertical"]
        try:
            calib.validate()
        except ValueError as exc:
            print(f"NOT SAVED -- calibration is invalid: {exc}")
            return
        path = save_right_arm_config(calib, args.save_path)
        print(f"Saved to {path}")

    def reset_to_initial() -> None:
        joint_targets.update(initial_joint_targets)
        ball_offset.update(initial_ball_offset)
        apply_joints()
        apply_ball_offset()
        print("Reset to the pose this session started from.")

    def on_key(keycode: int) -> None:
        step = JOINT_STEP
        ball_step = BALL_STEP
        joint_deltas = {
            KEY["Q"]: ("right_shoulder_pitch_joint", step), KEY["A"]: ("right_shoulder_pitch_joint", -step),
            KEY["W"]: ("right_shoulder_roll_joint", step), KEY["S"]: ("right_shoulder_roll_joint", -step),
            KEY["E"]: ("right_elbow_joint", step), KEY["D"]: ("right_elbow_joint", -step),
            KEY["R"]: ("right_wrist_roll_joint", step), KEY["F"]: ("right_wrist_roll_joint", -step),
            KEY["T"]: ("right_wrist_yaw_joint", step), KEY["G"]: ("right_wrist_yaw_joint", -step),
            KEY["Y"]: ("right_shoulder_yaw_joint", step), KEY["H"]: ("right_shoulder_yaw_joint", -step),
            KEY["U"]: ("right_wrist_pitch_joint", step), KEY["J"]: ("right_wrist_pitch_joint", -step),
        }
        ball_deltas = {
            KEY["I"]: ("forward", ball_step), KEY["K"]: ("forward", -ball_step),
            KEY["O"]: ("lateral", ball_step), KEY["L"]: ("lateral", -ball_step),
            KEY_RBRACKET: ("vertical", ball_step), KEY_LBRACKET: ("vertical", -ball_step),
        }
        if keycode in joint_deltas:
            name, delta = joint_deltas[keycode]
            lower = env.arm_joint_lower[env.arm_joint_names.index(name)]
            upper = env.arm_joint_upper[env.arm_joint_names.index(name)]
            joint_targets[name] = float(np.clip(joint_targets[name] + delta, lower, upper))
            apply_joints()
            print_status()
        elif keycode in ball_deltas:
            axis, delta = ball_deltas[keycode]
            ball_offset[axis] += delta
            apply_ball_offset()
            print_status()
        elif keycode == KEY["P"]:
            save()
        elif keycode == KEY["C"]:
            print_status()
        elif keycode == KEY["X"]:
            reset_to_initial()

    print(__doc__)
    print_status()
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        print("Calibration viewer running. Close the window or Ctrl+C to stop.")
        while viewer.is_running():
            mujoco.mj_step(env.model, env.data)
            viewer.sync()


if __name__ == "__main__":
    main()
