"""Run-directory layout, unique naming, and crash-safe (atomic) saving.

Every training invocation gets its own timestamped directory under
RL/runs/<run-name>/ -- nothing is ever overwritten. All "safe save" helpers
here follow the same pattern: write to a temporary path in the same
directory, then os.replace() it onto the final name. os.replace is atomic on
POSIX and Windows within the same filesystem, so a crash mid-write leaves
either the old file (untouched) or a stray ``*.part`` file -- never a
half-written file at the name anything else expects to read.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

RUN_SUBDIRS = ("checkpoints", "best_models", "tensorboard", "csv", "logs", "config", "models")


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def unique_path(path: Path) -> Path:
    """Return ``path`` if free, else ``path_2``, ``path_3``, ... (never overwrite)."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def create_run_directory(
    base_dir: Path,
    iterations: int,
    n_envs: int,
    seed: int,
    resumed: bool = False,
    run_label: str | None = None,
) -> Path:
    """RL/runs/<run-name>/<timestamp>_[finetune_]iter<N>_env<E>_seed<S>[_<label>]/"""
    parts = [timestamp_now()]
    if resumed:
        parts.append("finetune")
    parts.append(f"iter{iterations}")
    parts.append(f"env{n_envs}")
    parts.append(f"seed{seed}")
    if run_label:
        parts.append(run_label)
    run_dir = unique_path(base_dir / "_".join(parts))
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in RUN_SUBDIRS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def atomic_model_save(model, final_path: Path) -> Path:
    """Save an SB3 model so ``final_path`` only ever exists fully-written.

    SB3's save() only appends ".zip" when the given path has NO suffix at
    all (stable_baselines3.common.save_util.open_path_pathlib checks
    ``path.suffix == ""``) -- a temp name like "foo.part" already has a
    suffix, so SB3 would happily write literally to "foo.part" with no zip
    extension. Using "_part" (underscore, not a dot) keeps the temp path
    suffix-less so SB3's own ".zip" appending behaves as expected.
    """
    final_path = Path(final_path)
    if final_path.suffix != ".zip":
        final_path = final_path.with_suffix(".zip")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_stem = str(final_path.with_suffix("")) + "_part"
    model.save(tmp_stem)  # SB3 appends ".zip" since tmp_stem has no suffix
    tmp_zip = Path(tmp_stem + ".zip")
    os.replace(tmp_zip, final_path)
    return final_path


def write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def checkpoint_filename(global_steps: int, iteration: int, resumed: bool = False, local_iteration: int | None = None) -> str:
    if resumed and local_iteration is not None:
        return f"checkpoint_local_iter{local_iteration:04d}_global_steps{global_steps:08d}.zip"
    return f"checkpoint_iter{iteration:04d}_steps{global_steps:08d}.zip"


def final_model_filename(timestamp: str, iterations: int, n_envs: int, total_steps: int, seed: int) -> str:
    return f"final_model_{timestamp}_iter{iterations}_env{n_envs}_steps{total_steps:08d}_seed{seed}.zip"


def interrupted_model_filename(timestamp: str, global_steps: int) -> str:
    return f"interrupted_model_{timestamp}_global_steps{global_steps:08d}.zip"


def best_model_filename(metric_name: str, timestamp: str, value: float, global_steps: int) -> str:
    safe_value = f"{value:.4f}".replace("-", "neg").replace(".", "p")
    return f"best_{metric_name}_{timestamp}_value{safe_value}_steps{global_steps:08d}.zip"


class Tee:
    """stdout duplicator so console output is also captured to a log file.

    Proxies isatty()/fileno()/anything else to the FIRST stream (the real
    terminal). Without this, rich/tqdm's progress_bar=True can't detect a
    live terminal (AttributeError or a False isatty()) and silently renders
    no progress bar at all -- write/flush are the only methods that need to
    actually fan out to both streams.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return self.streams[0].isatty()

    def fileno(self):
        return self.streams[0].fileno()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)
