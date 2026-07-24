#!/usr/bin/env python3
"""Report-only disk usage per training run. Never deletes anything --
this repo has no automated cleanup command; any removal is a manual `rm`
the user runs themselves after reviewing this report.

    python RL/storage_report.py
    python RL/storage_report.py --base-dir RL/runs/target_throw_5m
"""
from __future__ import annotations

import argparse
from pathlib import Path


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1] / "RL" / "runs")
    args = parser.parse_args()

    if not args.base_dir.is_dir():
        print(f"No such directory: {args.base_dir}")
        return

    total = 0
    print(f"{'run':<55}{'checkpoints':>12}{'tensorboard':>12}{'csv':>10}{'total':>10}")
    for run_dir in sorted(p for p in args.base_dir.iterdir() if p.is_dir()):
        checkpoints = dir_size_bytes(run_dir / "checkpoints") if (run_dir / "checkpoints").is_dir() else 0
        tensorboard = dir_size_bytes(run_dir / "tensorboard") if (run_dir / "tensorboard").is_dir() else 0
        csv_size = dir_size_bytes(run_dir / "csv") if (run_dir / "csv").is_dir() else 0
        run_total = dir_size_bytes(run_dir)
        total += run_total
        n_checkpoints = len(list((run_dir / "checkpoints").glob("*.zip"))) if (run_dir / "checkpoints").is_dir() else 0
        print(
            f"{run_dir.name:<55}{human(checkpoints):>12}{human(tensorboard):>12}"
            f"{human(csv_size):>10}{human(run_total):>10}  ({n_checkpoints} checkpoints)"
        )
    print("-" * 99)
    print(f"total experiment storage: {human(total)}")


if __name__ == "__main__":
    main()
