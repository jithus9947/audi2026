#!/usr/bin/env python3
"""Scan every run_manifest.json under RL/runs/ and print a comparison table.

    python RL/compare_runs.py
    python RL/compare_runs.py --base-dir RL/runs/target_throw_5m --csv comparison.csv
"""
from __future__ import annotations

import argparse
import csv as csv_module
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = [
    "run_id", "requested_iterations", "completed_iterations", "n_envs", "seed",
    "final_global_timestep", "runtime_s", "device", "resumed", "exit_status",
    "mean_reward", "mean_forward_distance_m", "mean_landing_error_m", "target_hit_rate",
    "best_checkpoint", "final_model_path",
]


def load_manifests(base_dir: Path) -> list[dict]:
    rows = []
    for manifest_path in sorted(base_dir.rglob("run_manifest.json")):
        try:
            data = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        final_metrics = data.get("final_metrics") or data.get("progress") or {}
        best_paths = data.get("best_model_paths") or {}
        best_checkpoint = best_paths.get("landing_error") or best_paths.get("reward") or ""
        rows.append(
            {
                "run_id": data.get("run_id", manifest_path.parent.name),
                "requested_iterations": data.get("requested_iterations"),
                "completed_iterations": data.get("completed_iterations"),
                "n_envs": data.get("n_envs"),
                "seed": data.get("seed"),
                "final_global_timestep": data.get("final_global_timestep"),
                "runtime_s": data.get("runtime_s"),
                "device": data.get("device"),
                "resumed": data.get("resumed"),
                "exit_status": data.get("exit_status"),
                "mean_reward": final_metrics.get("mean_reward"),
                "mean_forward_distance_m": final_metrics.get("mean_forward_distance_m"),
                "mean_landing_error_m": final_metrics.get("mean_landing_error_m"),
                "target_hit_rate": final_metrics.get("target_hit_rate"),
                "best_checkpoint": best_checkpoint,
                "final_model_path": data.get("final_model_path", ""),
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No run_manifest.json files found.")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in COLUMNS}
    header = "  ".join(c.ljust(widths[c]) for c in COLUMNS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "RL" / "runs")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    rows = load_manifests(args.base_dir)
    print_table(rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
