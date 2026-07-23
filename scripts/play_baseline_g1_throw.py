#!/usr/bin/env python3
"""Visualize the deterministic open-loop G1 throw in MuJoCo."""

from pathlib import Path
import sys
import time

import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.baseline_controller import BaselineController
from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv


def main():
    env = G1FixedBodyThrowEnv(learned_release=True)
    # The target belongs to the PPO drop environment. This baseline is a free
    # forward throw, so hide the non-colliding target marker in this viewer.
    target_geom = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, "throw_target_geom"
    )
    if target_geom >= 0:
        env.model.geom_rgba[target_geom, 3] = 0.0
    controller = BaselineController(env.n_arm)
    obs, _ = env.reset(seed=42)

    print("Baseline viewer running. Close the MuJoCo window or press Ctrl+C to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t = env.step_count * env.control_dt
            action = controller.act(t)
            obs, _, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(env.control_dt)

            if terminated or truncated:
                print(
                    f"throw complete: release={info['release_time']:.2f} s, "
                    f"fell={info['robot_fell']}"
                )
                obs, _ = env.reset(seed=42)


if __name__ == "__main__":
    main()
