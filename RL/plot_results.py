#!/usr/bin/env python3
"""Generate the two figures expected by the "Graphs & baseline comparison" evidence:

    training_curves.png    episode reward and success-rate learning curves
    baseline_vs_ppo.png    baseline vs trained-PPO bars on every evaluate.py metric

Usage
-----
    python RL/plot_results.py --run-name ppo_g1_throw_20260722_180000
    python RL/plot_results.py  # uses the most recently trained run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RL_ROOT = Path(__file__).resolve().parent

# Palette slots from the project's data-viz reference (categorical order is
# fixed, never cycled): slot 1 blue = baseline / single-series curves,
# slot 2 orange = PPO.
COLOR_BASELINE = "#2a78d6"
COLOR_PPO = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_monitor_episodes(monitor_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(Path(monitor_dir).glob("*.monitor.csv")):
        frames.append(pd.read_csv(f, skiprows=1))
    if not frames:
        raise FileNotFoundError(f"No monitor CSVs found in {monitor_dir}")
    df = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    df["timesteps"] = df["l"].cumsum()
    return df


def plot_training_curves(monitor_dir: Path, out_path: Path, window: int = 20):
    df = load_monitor_episodes(monitor_dir)
    reward_roll = df["r"].rolling(window, min_periods=1).mean()
    success_roll = df["success"].astype(float).rolling(window, min_periods=1).mean() * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

    ax = axes[0]
    ax.scatter(df["timesteps"], df["r"], s=6, color=COLOR_BASELINE, alpha=0.15, linewidths=0)
    ax.plot(df["timesteps"], reward_roll, color=COLOR_BASELINE, linewidth=2)
    ax.set_ylabel("Episode reward")
    ax.set_title("PPO training reward", loc="left", color=INK_PRIMARY)
    ax.grid(axis="y", linewidth=0.75)

    ax = axes[1]
    ax.plot(df["timesteps"], success_roll, color=COLOR_BASELINE, linewidth=2)
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Training timesteps")
    ax.set_ylim(-5, 105)
    ax.set_title(f"Rolling success rate (window = {window} episodes)", loc="left", color=INK_PRIMARY)
    ax.grid(axis="y", linewidth=0.75)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_baseline_vs_ppo(comparison: dict, out_path: Path):
    baseline, ppo = comparison["baseline"], comparison["ppo"]

    metrics = [
        ("Success rate (%)", baseline["success_rate"] * 100, ppo["success_rate"] * 100, "{:.0f}"),
        ("Avg reward", baseline["avg_reward"], ppo["avg_reward"], "{:.2f}"),
        (
            "Avg landing error (m)",
            baseline["avg_landing_error"],
            ppo["avg_landing_error"],
            "{:.3f}",
        ),
        (
            "Avg action smoothness",
            baseline["avg_action_smoothness"],
            ppo["avg_action_smoothness"],
            "{:.3f}",
        ),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 4.2))
    bar_width = 0.55

    for ax, (title, b_val, p_val, fmt) in zip(axes, metrics):
        values = [b_val if b_val is not None else 0.0, p_val if p_val is not None else 0.0]
        colors = [COLOR_BASELINE, COLOR_PPO]
        bars = ax.bar([0, 1], values, width=bar_width, color=colors)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "PPO"])
        ax.set_title(title, fontsize=9.5, loc="left", color=INK_PRIMARY)
        ax.grid(axis="y", linewidth=0.75)
        ax.set_axisbelow(True)
        top = max(values) if max(values) > 0 else 1.0
        ax.set_ylim(0, top * 1.25)
        for bar, raw in zip(bars, (b_val, p_val)):
            label = "n/a" if raw is None else fmt.format(raw)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.03,
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK_PRIMARY,
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BASELINE),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_PPO),
    ]
    fig.legend(
        handles,
        ["Baseline (scripted)", f"PPO ({comparison['num_episodes']} eval episodes)"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle("Baseline vs trained PPO", y=1.12, fontsize=12, color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--window", type=int, default=20, help="Rolling-mean window, in episodes")
    args = parser.parse_args()

    run_name = args.run_name or (RL_ROOT / "runs" / "latest.txt").read_text().strip()
    run_dir = RL_ROOT / "runs" / run_name
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(run_dir / "monitor", plots_dir / "training_curves.png", window=args.window)

    comparison_path = run_dir / "results" / "comparison.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text())
        plot_baseline_vs_ppo(comparison, plots_dir / "baseline_vs_ppo.png")
    else:
        print(f"Skipping baseline-vs-PPO plot: {comparison_path} not found. Run RL/evaluate.py first.")


if __name__ == "__main__":
    main()
