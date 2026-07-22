"""Deterministic ball-drop reference controller for PPO comparisons."""

import numpy as np


class BaselineController:
    """A repeatable arm-placement motion followed by a controlled release."""

    def __init__(self, n_arm, placement_start=0.12, release_time=0.733):
        self.n_arm = n_arm
        self.placement_start = float(placement_start)
        self.release_time = float(release_time)
        self.placement_action = np.array(
            [-1.000, 0.800, 0.050, -0.850, 0.234, 0.150, -0.014],
            dtype=np.float32,
        )

    @staticmethod
    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def act(self, t):
        """Move the ball to the front-center landing zone, then release it."""
        action = np.zeros(self.n_arm + 1, dtype=np.float32)
        progress = self.smoothstep(
            (float(t) - self.placement_start) / (self.release_time - self.placement_start)
        )
        count = min(self.n_arm, len(self.placement_action))
        action[:count] = self.placement_action[:count] * progress
        # The last action dimension controls ball release when learned_release=True.
        action[-1] = 1.0 if t >= self.release_time else 0.0
        return action
