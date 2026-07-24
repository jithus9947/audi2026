#!/usr/bin/env python3
"""Live MuJoCo view of N independent target-throw lanes in one window.

Each lane is a fully independent G1TargetThrowEnv (same physics as training,
built from the untouched assets/unitree_g1/scene_throw.xml). Their qpos is
mirrored every frame into one merged display-only MjModel/MjData (built at
startup via mujoco.MjSpec.attach, never written to disk, never touching any
existing asset file) so all lanes render side by side in a single screen.

The merged model is never stepped -- only mj_forward (kinematics) is called
on it for rendering, so lanes cannot physically interact; all real dynamics
happen in each lane's own independent env.

This process polls the training run directory for the newest PPO checkpoint
and hot-reloads it, so the viewer keeps showing the current policy while
RL/train_target_throw.py trains in another process.
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

TARGET_MARKER_RGBA = (0.1, 0.9, 0.2, 0.65)


def lane_offset(i: int, cols: int, row_spacing: float, col_spacing: float) -> tuple[float, float]:
    """World (x, y) offset of lane ``i`` in the row/col grid used everywhere below."""
    row, col = divmod(i, cols)
    x = row * row_spacing
    y = (col - (cols - 1) / 2.0) * col_spacing
    return x, y


def build_arena_spec(n_robots: int, rows: int, cols: int, row_spacing: float, col_spacing: float,
                      target_distance: float, target_half_size: float):
    unit_template = mujoco.MjSpec.from_file(str(ROOT / "assets" / "unitree_g1" / "scene_throw.xml"))
    unit_template.delete(unit_template.geom("floor"))
    for key in list(unit_template.keys):
        unit_template.delete(key)

    parent = mujoco.MjSpec()
    parent.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    span_x = rows * row_spacing
    span_y = cols * col_spacing
    parent.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[span_x, span_y, 0.05],
        rgba=[0.2, 0.3, 0.4, 1.0],
    )

    prefixes = []
    for i in range(n_robots):
        x, y = lane_offset(i, cols, row_spacing, col_spacing)
        frame = parent.worldbody.add_frame(pos=[x, y, 0])
        prefix = f"r{i}_"
        parent.attach(unit_template.copy(), prefix=prefix, frame=frame)
        prefixes.append(prefix)

    for prefix in prefixes:
        target_body = parent.body(f"{prefix}throw_target")
        target_body.pos = [target_distance, 0.0, 0.01]
        target_body.add_geom(
            name=f"{prefix}target_marker",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[target_half_size, target_half_size, 0.004],
            rgba=list(TARGET_MARKER_RGBA),
            contype=0,
            conaffinity=0,
        )

    model = parent.compile()
    return model, prefixes


class Lane:
    """One independent physics simulation + its qpos slice in the merged model.

    Each lane's env is its own single-robot simulation, unaware of the 16-lane
    grid -- its pelvis and ball free joints report qpos as if it were alone at
    the world origin. Free-joint qpos is an absolute world position (not
    relative to the attach frame), so the lane's (x, y) grid offset has to be
    added back in by hand before the qpos is copied into the merged display
    model, or every lane would render on top of the others at the origin.
    """

    def __init__(self, env: G1TargetThrowEnv, merged_qpos_slice: slice, xy_offset: tuple[float, float], seed: int):
        self.env = env
        self.qpos_slice = merged_qpos_slice
        self.xy_offset = np.array(xy_offset, dtype=np.float64)
        self.seed = seed
        pelvis_joint_id = env.model.body_jntadr[env.base_body_id]
        self.pelvis_qpos_adr = int(env.model.jnt_qposadr[pelvis_joint_id])
        self.ball_qpos_adr = int(env.ball_qpos_adr)
        self.env.reset(seed=seed)
        self.obs = self.env._get_obs()
        self.episodes = 0
        self.hits = 0

    def step(self, deterministic_action):
        action = deterministic_action if deterministic_action is not None else np.zeros(
            self.env.action_space.shape, dtype=np.float32
        )
        self.obs, _, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.episodes += 1
            if info.get("square_target_hit"):
                self.hits += 1
            self.env.reset(seed=self.seed + self.episodes)
            self.obs = self.env._get_obs()

    def offset_qpos(self) -> np.ndarray:
        qpos = self.env.data.qpos.copy()
        qpos[self.pelvis_qpos_adr : self.pelvis_qpos_adr + 2] += self.xy_offset
        qpos[self.ball_qpos_adr : self.ball_qpos_adr + 2] += self.xy_offset
        return qpos


def latest_checkpoint(run_dir: Path):
    checkpoints_dir = run_dir / "checkpoints"
    candidates = list(checkpoints_dir.glob("*.zip")) if checkpoints_dir.is_dir() else []
    final = run_dir / "final_model.zip"
    if final.is_file():
        candidates.append(final)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "RL" / "runs" / "target_throw")
    parser.add_argument("--n-robots", type=int, default=16)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--row-spacing", type=float, default=10.0, help="Meters between lane rows (throw depth).")
    parser.add_argument("--col-spacing", type=float, default=4.0, help="Meters between lanes side by side.")
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--target-half-size", type=float, default=0.35)
    parser.add_argument("--reload-every", type=float, default=3.0, help="Seconds between checkpoint polls.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.rows * args.cols < args.n_robots:
        parser.error("rows*cols must be >= n-robots")

    print("Building merged viewing arena for", args.n_robots, "lanes...")
    model, prefixes = build_arena_spec(
        args.n_robots, args.rows, args.cols, args.row_spacing, args.col_spacing,
        args.target_distance, args.target_half_size,
    )
    data = mujoco.MjData(model)

    single_model_nq = None
    lanes: list[Lane] = []
    for i in range(args.n_robots):
        env = G1TargetThrowEnv(
            learned_release=True,
            target_pos=(args.target_distance, 0.0, 0.01),
            target_half_size=args.target_half_size,
            verbose_init=(i == 0),  # one diagnostic print is enough across all 16 lanes
        )
        if single_model_nq is None:
            single_model_nq = env.model.nq
        qpos_slice = slice(i * single_model_nq, (i + 1) * single_model_nq)
        offset = lane_offset(i, args.cols, args.row_spacing, args.col_spacing)
        lanes.append(Lane(env, qpos_slice, offset, seed=args.seed + i * 1000))

    policy = None
    policy_path = None
    last_reload = 0.0

    print("Waiting for the first PPO checkpoint (random actions until then)...")
    print("Close the MuJoCo window or press Ctrl+C to stop viewing.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [args.rows * args.row_spacing / 2.0, 0.0, 1.0]
        viewer.cam.distance = max(args.rows * args.row_spacing, args.cols * args.col_spacing) * 0.9
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -35

        while viewer.is_running():
            now = time.time()
            if now - last_reload > args.reload_every:
                last_reload = now
                candidate = latest_checkpoint(args.run_dir)
                if candidate is not None and candidate != policy_path:
                    try:
                        from stable_baselines3 import PPO

                        policy = PPO.load(str(candidate), device="cpu")
                        policy_path = candidate
                        print(f"Loaded checkpoint: {candidate.name}")
                    except Exception as exc:  # noqa: BLE001 - viewer keeps running either way
                        print(f"Could not load checkpoint {candidate}: {exc}")

            for lane in lanes:
                if policy is not None:
                    action, _ = policy.predict(lane.obs, deterministic=True)
                else:
                    action = None
                lane.step(action)
                data.qpos[lane.qpos_slice] = lane.offset_qpos()

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(lanes[0].env.control_dt)

    total_hits = sum(l.hits for l in lanes)
    total_eps = sum(l.episodes for l in lanes)
    print(f"Viewer closed. Across all lanes: {total_hits}/{total_eps} target hits.")


if __name__ == "__main__":
    main()
