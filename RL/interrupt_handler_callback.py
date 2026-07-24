"""Graceful Ctrl+C / SIGTERM handling.

Signal handlers only set a flag (signal handlers must stay minimal/async-safe
-- no file I/O, no model saving inside them). The callback's ``_on_step``
checks that flag every rollout step and returns False when set, which makes
SB3's ``model.learn()`` stop the collection loop and return normally *from
the main thread*. The training script then does the actual emergency save,
CSV flush, and manifest update after ``model.learn()`` returns -- see
RL/train_target_throw.py's try/finally around the call.
"""
from __future__ import annotations

import signal

from stable_baselines3.common.callbacks import BaseCallback


class InterruptHandlerCallback(BaseCallback):
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.interrupted = False
        self._previous_handlers = {}

    def _on_training_start(self) -> None:
        self._previous_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._handle_signal)
        self._previous_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.interrupted = True
        # Restore default handling so a second Ctrl+C force-kills if the
        # graceful path is somehow stuck (e.g. a hung subprocess env).
        signal.signal(signum, self._previous_handlers.get(signum, signal.SIG_DFL))

    def _on_step(self) -> bool:
        if self.interrupted:
            if self.verbose:
                print("\n[interrupt] stop requested, finishing current rollout step and saving...")
            return False
        return True

    def _on_training_end(self) -> None:
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)
