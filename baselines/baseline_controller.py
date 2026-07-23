"""Deterministic open-loop G1 forward-throw controller.

Edit ``SAFE_START_ACTION``, ``THROW_END_ACTION`` and the two time values below
to tune the baseline motion.  Each action value is in [-1, 1], not radians.
"""

import numpy as np

# Action order: shoulder pitch, shoulder roll, shoulder yaw, elbow,
# wrist roll, wrist pitch, wrist yaw.  The environment converts these into
# limited joint targets, so keeping every value within [-1, 1] is required.
#
# This safe start pose keeps the hand clear of the robot's thighs.  Do not use
# a backward/downward shoulder pose here: it can make the hand hit the legs.
SAFE_START_ACTION = np.array(
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], dtype=np.float32
)

# Forward end pose. Change these seven values to tune the throwing direction.
THROW_END_ACTION = np.array(
    [-1.00, 0.80, 0.05, -0.85, 0.23, 0.15, -0.01], dtype=np.float32
)

# Motion timing in seconds. Reducing FORWARD_SWING_END makes the throw faster.
FORWARD_SWING_START = 0.12
RELEASE_TIME = 0.50


class BaselineController:
    """Move from a safe pose into a forward swing, then release the ball."""

    def __init__(
        self,
        n_arm,
        forward_swing_start=FORWARD_SWING_START,
        release_time=RELEASE_TIME,
        safe_start_action=SAFE_START_ACTION,
        throw_end_action=THROW_END_ACTION,
    ):
        self.n_arm = n_arm
        self.forward_swing_start = float(forward_swing_start)
        self.release_time = float(release_time)
        if not 0 <= self.forward_swing_start < self.release_time:
            raise ValueError("Require 0 <= forward_swing_start < release_time")
        self.safe_start_action = np.clip(np.asarray(safe_start_action, dtype=np.float32), -1, 1)
        self.throw_end_action = np.clip(np.asarray(throw_end_action, dtype=np.float32), -1, 1)

    @staticmethod
    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def act(self, t):
        """Return the arm command and release command for time ``t``."""
        action = np.zeros(self.n_arm + 1, dtype=np.float32)
        count = min(self.n_arm, len(self.throw_end_action))
        if t < self.forward_swing_start:
            # Hold this upright, leg-clear start pose before the swing.
            action[:count] = self.safe_start_action[:count]
        else:
            # A fast forward swing gives the released ball forward velocity.
            progress = self.smoothstep(
                (float(t) - self.forward_swing_start)
                / (self.release_time - self.forward_swing_start)
            )
            action[:count] = (
                self.safe_start_action[:count]
                + (self.throw_end_action[:count] - self.safe_start_action[:count]) * progress
            )
        # The last action controls the weld that holds the ball at the wrist.
        action[-1] = 1.0 if t >= self.release_time else 0.0
        return action
