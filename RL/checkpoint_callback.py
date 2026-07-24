"""Descriptive, atomic, never-delete checkpointing.

Replaces SB3's built-in CheckpointCallback (which names files
``{prefix}_{steps}_steps.zip`` and has no atomic-write guarantee) with one
that names files ``checkpoint_iter{N}_steps{S}.zip`` (or
``checkpoint_local_iter{N}_global_steps{S}.zip`` when resumed, so local
progress in *this* run and the absolute step count are both visible), saves
via run_management.atomic_model_save, and never deletes an older checkpoint.
"""
from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from RL.run_management import atomic_model_save, checkpoint_filename


class DescriptiveCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_dir: Path,
        save_every_iterations: int,
        resumed: bool = False,
        previous_timesteps: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_every_iterations = max(save_every_iterations, 1)
        self.resumed = resumed
        self.previous_timesteps = previous_timesteps
        self.saved_paths: list[Path] = []

    def _on_step(self) -> bool:
        n_steps = getattr(self.model, "n_steps", 1) or 1
        save_every_calls = self.save_every_iterations * n_steps
        if self.n_calls % save_every_calls != 0:
            return True
        local_iteration = self.n_calls // n_steps
        name = checkpoint_filename(
            self.num_timesteps,
            local_iteration,
            resumed=self.resumed,
            local_iteration=local_iteration if self.resumed else None,
        )
        path = atomic_model_save(self.model, self.save_dir / name)
        self.saved_paths.append(path)
        if self.verbose:
            print(f"[checkpoint] saved {path}")
        return True
