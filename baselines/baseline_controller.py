"""Deterministic open-loop G1 forward-throw controller."""

import numpy as np


class BaselineController:
    """Wind the arm back, swing it forward, then release the ball."""

    def __init__(self, n_arm, windup_end=0.30, release_time=0.56):
        self.n_arm = n_arm
        self.windup_end = float(windup_end)
        self.release_time = float(release_time)
        # Seven action values map in order to the G1 right shoulder, elbow,
        # and wrist joints. These stay inside the environment joint limits.
        self.windup_action = np.array(
            [0.35, 0.20, -0.10, 0.45, 0.00, 0.05, 0.00], dtype=np.float32
        )
        self.throw_action = np.array(
            [-1.000, 0.800, 0.050, -0.850, 0.234, 0.150, -0.014],
            dtype=np.float32,
        )

    @staticmethod
    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def act(self, t):
        """Return the arm command and release command for time ``t``."""
        action = np.zeros(self.n_arm + 1, dtype=np.float32)
        count = min(self.n_arm, len(self.throw_action))
        if t < self.windup_end:
            # Move smoothly into a compact backswing.
            progress = self.smoothstep(float(t) / self.windup_end)
            action[:count] = self.windup_action[:count] * progress
        else:
            # A fast forward swing gives the released ball forward velocity.
            progress = self.smoothstep(
                (float(t) - self.windup_end) / (self.release_time - self.windup_end)
            )
            action[:count] = (
                self.windup_action[:count]
                + (self.throw_action[:count] - self.windup_action[:count]) * progress
            )
        # The last action controls the weld that holds the ball at the wrist.
        action[-1] = 1.0 if t >= self.release_time else 0.0
        return action
