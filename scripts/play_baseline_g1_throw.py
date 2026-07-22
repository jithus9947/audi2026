#!/usr/bin/env python3
"""Visualize the deterministic baseline G1 throw in MuJoCo."""

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
                    f"episode complete: best distance={info['best_dist']:.3f} m, "
                    f"release={info['release_time']:.2f} s"
                )
                obs, _ = env.reset(seed=42)


if __name__ == "__main__":
    main()
