#!/usr/bin/env python3
"""Item 13: per-episode motion plots -- joint angles, palm orientation, hand
speed, ball-to-palm distance, and ball-held state over time, with each motion
phase (ready/throw/follow_through/recovery/hold) marked.

    python RL/plot_episode_motion.py --model RL/runs/target_throw_5m/<run>/models/final_model.zip --episodes 20
    python RL/plot_episode_motion.py --episodes 5   # random actions, no --model needed

Writes one PNG per episode plus one across-episode summary PNG to
--output-dir (default: a new timestamped folder so nothing is overwritten).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.g1_target_throw_env import G1TargetThrowEnv
from envs.hand_geometry import get_palm_frame
from RL.run_management import timestamp_now

PHASE_COLORS = {
    "ready": "#cfe8ff",
    "throw": "#ffe0b3",
    "follow_through": "#ffd1dc",
    "recovery": "#d9f2d9",
    "hold": "#e6e6e6",
}


def record_episode(env: G1TargetThrowEnv, model, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    rows = {
        "t": [], "phase": [], "shoulder_pitch": [], "shoulder_roll": [], "shoulder_yaw": [],
        "elbow": [], "wrist_roll": [], "wrist_pitch": [], "wrist_yaw": [],
        "palm_roll": [], "palm_pitch": [], "palm_yaw": [],
        "hand_speed_forward": [], "hand_speed_total": [],
        "ball_to_palm_distance": [], "ball_held": [],
    }
    info = {}
    terminated = truncated = False
    while not (terminated or truncated):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        t = env.step_count * env.control_dt
        qpos = env.data.qpos[env.arm_qpos_adr]
        frame = get_palm_frame(env.model, env.data, env.hold_body_id, env.ball_body_id)
        hand_vel6 = np.zeros(6)
        import mujoco

        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, env.hold_body_id, hand_vel6, 0)

        rows["t"].append(t)
        rows["phase"].append(info["phase"])
        for i, name in enumerate(["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]):
            rows[name].append(float(qpos[i]) if i < len(qpos) else np.nan)
        rows["palm_roll"].append(frame.roll_deg)
        rows["palm_pitch"].append(frame.pitch_deg)
        rows["palm_yaw"].append(frame.yaw_deg)
        rows["hand_speed_forward"].append(float(hand_vel6[3]))
        rows["hand_speed_total"].append(float(np.linalg.norm(hand_vel6[3:])))
        rows["ball_to_palm_distance"].append(0.0 if info["ball_released"] else info["ball_to_palm_distance"])
        rows["ball_held"].append(0.0 if info["ball_released"] else 1.0)

    for key in rows:
        if key != "phase":
            rows[key] = np.array(rows[key])
    rows["info"] = info
    return rows


def _shade_phases(ax, t, phase):
    start = 0
    for i in range(1, len(phase) + 1):
        if i == len(phase) or phase[i] != phase[start]:
            ax.axvspan(t[start], t[i - 1] if i < len(phase) else t[-1], color=PHASE_COLORS.get(phase[start], "white"), alpha=0.5, lw=0)
            start = i


def plot_episode(record: dict, episode_idx: int, save_path: Path) -> None:
    t = record["t"]
    phase = record["phase"]
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    ax = axes[0]
    for name in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw"]:
        ax.plot(t, np.degrees(record[name]), label=name)
    _shade_phases(ax, t, phase)
    ax.set_ylabel("shoulder (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Episode {episode_idx}: joint angles, palm orientation, hand speed, ball-palm distance")

    ax = axes[1]
    ax.plot(t, np.degrees(record["elbow"]), label="elbow", color="tab:red")
    for name in ["wrist_roll", "wrist_pitch", "wrist_yaw"]:
        ax.plot(t, np.degrees(record[name]), label=name)
    _shade_phases(ax, t, phase)
    ax.set_ylabel("elbow/wrist (deg)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(t, record["palm_roll"], label="palm roll")
    ax.plot(t, record["palm_pitch"], label="palm pitch")
    ax.plot(t, record["palm_yaw"], label="palm yaw")
    _shade_phases(ax, t, phase)
    ax.set_ylabel("palm orientation (deg)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    ax.plot(t, record["hand_speed_forward"], label="hand forward speed")
    ax.plot(t, record["hand_speed_total"], label="hand total speed", alpha=0.7)
    _shade_phases(ax, t, phase)
    ax.set_ylabel("speed (m/s)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[4]
    ax.plot(t, record["ball_to_palm_distance"], label="ball-to-palm distance", color="tab:orange")
    ax.fill_between(t, 0, record["ball_held"] * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0, alpha=0.15, color="tab:blue", label="ball held")
    _shade_phases(ax, t, phase)
    ax.set_ylabel("distance (m)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)

    info = record["info"]
    fig.text(
        0.5, 0.005,
        f"release={info.get('release_time')}  landing_error={info.get('landing_error')}  "
        f"final_arm_pose_error={info.get('final_arm_pose_error'):.4f}  termination={info.get('termination_reason')}  "
        f"[phase colors: ready={PHASE_COLORS['ready']} throw={PHASE_COLORS['throw']} "
        f"follow_through={PHASE_COLORS['follow_through']} recovery={PHASE_COLORS['recovery']} hold={PHASE_COLORS['hold']}]",
        ha="center", fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def plot_summary(records: list[dict], save_path: Path) -> None:
    release_times = [r["info"].get("release_time") for r in records if r["info"].get("release_time") is not None]
    pose_errors = [r["info"].get("final_arm_pose_error") for r in records]
    hold_starts = [r["info"].get("hold_start_time") for r in records if r["info"].get("hold_start_time") is not None]
    landing_errors = [r["info"].get("landing_error") for r in records if r["info"].get("landing_error") is not None]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(release_times, bins=10, color="tab:orange")
    axes[0].set_title("release time (s)")
    axes[1].hist(pose_errors, bins=10, color="tab:green")
    axes[1].set_title("final arm pose error (rad)")
    axes[2].hist(landing_errors, bins=10, color="tab:blue")
    axes[2].set_title("landing error (m)")
    fig.suptitle(f"Summary across {len(records)} episodes (hold phase reached: {len(hold_starts)}/{len(records)})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--target-half-size", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (ROOT / "RL" / "motion_plots" / timestamp_now())
    output_dir.mkdir(parents=True, exist_ok=True)

    env = G1TargetThrowEnv(
        learned_release=True, target_pos=(args.target_distance, 0.0, 0.01),
        target_half_size=args.target_half_size, verbose_init=False,
    )
    model = None
    if args.model is not None:
        from stable_baselines3 import PPO

        model = PPO.load(str(args.model), device="cpu")

    records = []
    for ep in range(args.episodes):
        record = record_episode(env, model, seed=args.seed + ep)
        records.append(record)
        plot_episode(record, ep, output_dir / f"episode_{ep:02d}.png")
        print(f"episode {ep}: saved {output_dir / f'episode_{ep:02d}.png'}")

    plot_summary(records, output_dir / "summary.png")
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
