"""Which right-arm joints the policy controls, which are held fixed, and the
calibrated pose/ball-offset numbers around them.

Ground truth for the joint split (verified against the live model, not
assumed -- see G1TargetThrowEnv.__init__, which asserts ACTIVE_ARM_JOINT_NAMES
+ FIXED_ARM_JOINT_NAMES == set(env.arm_joint_names) at construction time):

  ACTIVE  (policy-controlled, 5-dim action): shoulder_pitch, shoulder_roll,
      elbow, wrist_roll, wrist_yaw.
  FIXED   (never in the action space, held at a calibrated constant for the
      whole episode): shoulder_yaw, wrist_pitch.

wrist_roll and wrist_yaw are "active" but additionally range-limited far
tighter than the arm's general safety-margined joint range -- see
G1TargetThrowEnv._active_action_low/_active_action_high, which derives the
narrow per-joint action clamp from these calibrated radian bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ACTIVE_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
)
FIXED_ARM_JOINT_NAMES = (
    "right_shoulder_yaw_joint",
    "right_wrist_pitch_joint",
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "right_arm_throw_pose.yaml"


@dataclass
class RightArmCalibration:
    """Everything a human calibrated by hand (or the defaults, until they do).

    ball_offset_forward/vertical are DELTAS added on top of the
    geometrically-computed hold offset from envs/hand_geometry.py (which
    already correctly clears the visual hand mesh) -- not absolute
    positions. Default 0.0 means "use the geometry as computed". lateral has
    no geometric baseline (the mesh-centering logic only centers the
    forward/vertical cross-section), so it is an absolute local-frame offset.

    lateral's sign convention: POSITIVE moves the ball toward the robot's own
    LEFT side. This was verified two independent ways (see the change report):
    algebraically (dot(hand local +y, world +y) = +0.96 at a representative
    extended-throw pose, with the robot facing world +x where +y is the
    robot's left per REP-103) and visually (rendered image with the two
    candidate offsets colour-coded, robot's own left arm/side confirmed by
    the un-mirrored torso logo placement).
    """

    fixed_joints: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in FIXED_ARM_JOINT_NAMES}
    )
    wrist_roll_nominal: float = 1.652
    wrist_roll_min: float = 1.62
    wrist_roll_max: float = 1.70
    wrist_yaw_hold: float = 0.0
    wrist_yaw_release_min: float = 0.0
    wrist_yaw_release_max: float = 0.2
    wrist_yaw_release_value: float = 0.15
    wrist_yaw_ramp_duration: float = 0.35
    ball_offset_forward: float = 0.0
    ball_offset_lateral: float = 0.006
    ball_offset_vertical: float = 0.0

    def validate(self) -> None:
        if set(self.fixed_joints.keys()) != set(FIXED_ARM_JOINT_NAMES):
            raise ValueError(
                f"fixed_joints keys {sorted(self.fixed_joints.keys())} != "
                f"expected {sorted(FIXED_ARM_JOINT_NAMES)}"
            )
        if not (self.wrist_roll_min <= self.wrist_roll_nominal <= self.wrist_roll_max):
            raise ValueError(
                f"wrist_roll_nominal ({self.wrist_roll_nominal}) must lie within "
                f"[{self.wrist_roll_min}, {self.wrist_roll_max}]"
            )
        if self.wrist_roll_min > self.wrist_roll_max:
            raise ValueError("wrist_roll_min must be <= wrist_roll_max")
        if not (0.0 <= self.wrist_yaw_release_min <= self.wrist_yaw_release_max <= 0.2):
            raise ValueError(
                f"wrist_yaw release range [{self.wrist_yaw_release_min}, "
                f"{self.wrist_yaw_release_max}] must lie within [0.0, 0.2]"
            )
        if not (self.wrist_yaw_release_min <= self.wrist_yaw_release_value <= self.wrist_yaw_release_max):
            raise ValueError(
                f"wrist_yaw_release_value ({self.wrist_yaw_release_value}) must lie within "
                f"[{self.wrist_yaw_release_min}, {self.wrist_yaw_release_max}]"
            )


def load_right_arm_config(path: Path | str | None = None) -> RightArmCalibration:
    """Load calibration from YAML, or return defaults if the file doesn't exist yet
    (e.g. before anyone has run the calibration tool -- see
    RL/calibrate_right_arm_pose.py)."""
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.is_file():
        return RightArmCalibration()
    with open(resolved) as f:
        data = yaml.safe_load(f) or {}
    wrist_roll = data.get("wrist_roll", {})
    wrist_yaw = data.get("wrist_yaw", {})
    ball_offset = data.get("ball_offset_local", {})
    config = RightArmCalibration(
        fixed_joints={**{name: 0.0 for name in FIXED_ARM_JOINT_NAMES}, **data.get("fixed_joints", {})},
        wrist_roll_nominal=float(wrist_roll.get("nominal", 1.652)),
        wrist_roll_min=float(wrist_roll.get("minimum", 1.62)),
        wrist_roll_max=float(wrist_roll.get("maximum", 1.70)),
        wrist_yaw_hold=float(wrist_yaw.get("hold", 0.0)),
        wrist_yaw_release_min=float(wrist_yaw.get("release_minimum", 0.0)),
        wrist_yaw_release_max=float(wrist_yaw.get("release_maximum", 0.2)),
        wrist_yaw_release_value=float(wrist_yaw.get("release_value", 0.15)),
        wrist_yaw_ramp_duration=float(wrist_yaw.get("ramp_duration", 0.35)),
        ball_offset_forward=float(ball_offset.get("forward", 0.0)),
        ball_offset_lateral=float(ball_offset.get("lateral", 0.006)),
        ball_offset_vertical=float(ball_offset.get("vertical", 0.0)),
    )
    config.validate()
    return config


def save_right_arm_config(config: RightArmCalibration, path: Path | str | None = None) -> Path:
    config.validate()
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "active_joints": list(ACTIVE_ARM_JOINT_NAMES),
        "fixed_joints": dict(config.fixed_joints),
        "wrist_roll": {
            "nominal": config.wrist_roll_nominal,
            "minimum": config.wrist_roll_min,
            "maximum": config.wrist_roll_max,
        },
        "wrist_yaw": {
            "hold": config.wrist_yaw_hold,
            "release_minimum": config.wrist_yaw_release_min,
            "release_maximum": config.wrist_yaw_release_max,
            "release_value": config.wrist_yaw_release_value,
            "ramp_duration": config.wrist_yaw_ramp_duration,
        },
        "ball_offset_local": {
            "forward": config.ball_offset_forward,
            "lateral": config.ball_offset_lateral,
            "vertical": config.ball_offset_vertical,
        },
    }
    with open(resolved, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return resolved
