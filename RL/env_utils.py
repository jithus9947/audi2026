"""Env factory shared by train.py and evaluate.py.

Only imports from envs/ and baselines/ -- never edits them -- so the rest of
the audi2026 project stays untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv  # noqa: E402

# Extra per-episode fields recorded by SB3's Monitor wrapper alongside reward
# and length. These are exactly the fields evaluate.py / plot_results.py need
# to reconstruct success rate and best-distance curves without touching the
# env or evaluation_scripts/.
MONITOR_INFO_KEYWORDS = ("success", "best_dist", "release_time")


def make_env(
    rank: int,
    seed: int,
    log_dir: Optional[Path] = None,
    learned_release: bool = True,
    monitor: bool = True,
) -> Callable[[], "gym.Env"]:
    """Return a thunk building one (optionally Monitor-wrapped) env instance."""

    def _init():
        env = G1FixedBodyThrowEnv(learned_release=learned_release)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        if monitor:
            from stable_baselines3.common.monitor import Monitor

            filename = None
            if log_dir is not None:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                filename = str(Path(log_dir) / f"monitor_{rank}")
            env = Monitor(env, filename=filename, info_keywords=MONITOR_INFO_KEYWORDS)
        return env

    return _init
