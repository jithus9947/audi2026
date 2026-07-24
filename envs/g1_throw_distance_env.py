"""G1 throwing task that rewards distance instead of target accuracy.

Subclasses G1FixedBodyThrowEnv (envs/g1_fixed_body_throw_env.py) without
touching it. Same robot, same ball, same physics -- only ``_compute_reward``
and episode length change: the goal is "throw the ball as far as possible"
rather than "land it inside an 18cm circle".
"""
from __future__ import annotations

import numpy as np

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv


class G1ThrowDistanceEnv(G1FixedBodyThrowEnv):
    """Reward = horizontal distance the ball travels from its start position.

    ``distance_reward_scale`` and ``max_reward_distance`` shape the terminal
    bonus; ``max_reward_distance`` caps the reward per episode so a single
    unrealistic simulation glitch can't dominate a PPO batch.
    """

    def __init__(
        self,
        *args,
        episode_time: float = 3.0,
        distance_reward_scale: float = 2.0,
        max_reward_distance: float = 6.0,
        **kwargs,
    ):
        super().__init__(*args, episode_time=episode_time, **kwargs)
        self.distance_reward_scale = float(distance_reward_scale)
        self.max_reward_distance = float(max_reward_distance)
        self.launch_xy = np.zeros(2, dtype=np.float64)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.launch_xy = self._ball_pos()[:2].copy()
        return obs, info

    def _throw_distance(self) -> float:
        return float(np.linalg.norm(self._ball_pos()[:2] - self.launch_xy))

    def _compute_reward(self, action, landed):
        dist = min(self._throw_distance(), self.max_reward_distance)
        reward = (
            -0.002 * np.linalg.norm(action[: self.n_arm])
            - 0.001 * np.linalg.norm(action - self.prev_action)
        )
        if self.robot_fell:
            return reward - 5.0
        if landed:
            return reward + dist * self.distance_reward_scale
        if self.released:
            reward += 0.02 * dist
        return reward

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["throw_distance"] = self._throw_distance()
        return obs, reward, terminated, truncated, info
