#!/usr/bin/env python3
"""Single command: train PPO on the target-throw task and serve TensorBoard.

    python RL/run.py

Starts RL/train_target_throw.py (PPO, 16 parallel envs by default) and
`tensorboard` as background subprocesses. Headless by default -- production
training does NOT open the tiled MuJoCo viewer automatically (pass --viewer
if you explicitly want it; it costs FPS). Ctrl+C stops both subprocesses.

Nothing in envs/g1_fixed_body_throw_env.py, baselines/baseline_controller.py,
baselines/play_step_throw.py, or scripts/play_*.py is touched by this.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from RL.hardware_diagnostics import select_device
from RL.run_management import create_run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="target_throw_5m")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--n-envs", "--num-envs", dest="n_envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--target-half-size", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    parser.add_argument("--viewer", action="store_true", help="Also open the tiled MuJoCo viewer (costs FPS and RAM). OFF by default.")
    parser.add_argument("--no-viewer", action="store_true", help="Deprecated no-op: headless is already the default.")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the TensorBoard URL.")
    parser.add_argument(
        "--viewer-envs", type=int, default=None,
        help="Lanes shown in the viewer, independent of --n-envs. The viewer builds its OWN separate "
        "MuJoCo simulations (not a window into the training workers), so viewer-envs + n-envs are "
        "concurrent processes that share RAM/CPU. Default: min(n-envs, 8) -- pairing e.g. --n-envs 32 "
        "with --viewer-envs 32 means 64 simultaneous MuJoCo simulations at startup, which can OOM-kill "
        "the viewer on a laptop-class machine. Keep this small; it's for a visual sanity check, not "
        "for watching literally every training worker.",
    )
    parser.add_argument("--viewer-rows", type=int, default=None, help="Default: auto, near-square for --viewer-envs.")
    parser.add_argument("--viewer-cols", type=int, default=None, help="Default: auto, near-square for --viewer-envs.")
    parser.add_argument(
        "--viewer-delay", type=float, default=8.0,
        help="Seconds to wait after starting training before launching the viewer, so the training "
        "workers finish loading first instead of competing for RAM during simultaneous startup.",
    )
    parser.add_argument("--resume-from", type=Path, default=None, help="Existing .zip checkpoint to fine-tune from.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit run directory; default is a fresh timestamped one.")
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def _default_grid(n_envs: int) -> tuple[int, int]:
    cols = min(8, n_envs)
    rows = -(-n_envs // cols)  # ceil division
    return rows, cols


def main() -> None:
    args = parse_args()
    if args.no_viewer:
        print("--no-viewer is deprecated and has no effect: headless is already the default. Use --viewer to opt in.")

    viewer_envs = args.viewer_envs if args.viewer_envs is not None else min(args.n_envs, 8)
    if args.viewer_rows is not None and args.viewer_cols is not None:
        viewer_rows, viewer_cols = args.viewer_rows, args.viewer_cols
    else:
        viewer_rows, viewer_cols = _default_grid(viewer_envs)
    if args.viewer and viewer_envs == args.n_envs and args.n_envs > 8:
        print(
            f"[warning] --viewer-envs matches --n-envs ({args.n_envs}): that's {args.n_envs * 2} concurrent "
            f"MuJoCo simulations at startup (training workers + viewer's own lanes). Consider a smaller "
            f"--viewer-envs (e.g. 8) if you hit an OOM kill."
        )

    device = select_device(args.device)
    resumed = args.resume_from is not None
    if args.output_dir:
        run_dir = args.output_dir
        for sub in ("checkpoints", "best_models", "tensorboard", "csv", "logs", "config", "models"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_directory(
            ROOT / "RL" / "runs" / args.run_name, args.iterations, args.n_envs, args.seed,
            resumed=resumed, run_label=args.run_label,
        )
    rollout_size = args.n_steps * args.n_envs
    tb_url = f"http://localhost:{args.tensorboard_port}"

    print("=" * 72)
    print(f"run_id              : {run_dir.name}")
    print(f"run_directory       : {run_dir}")
    print(f"tensorboard_url     : {tb_url}")
    print(f"device              : {device}")
    print(f"n_envs              : {args.n_envs}")
    print(f"rollout_size        : {rollout_size}  (n_steps={args.n_steps} * n_envs={args.n_envs})")
    print(f"batch_size          : {args.batch_size}")
    print(f"checkpoint_dir      : {run_dir / 'checkpoints'}")
    print(f"final_model_pattern : models/final_model_<timestamp>_iter<N>_env{args.n_envs}_steps<S>_seed{args.seed}.zip")
    if args.viewer:
        print(f"viewer              : tiled, {viewer_envs} lanes ({viewer_rows}x{viewer_cols}), starts {args.viewer_delay:.0f}s after training")
    else:
        print("viewer              : disabled (headless production default)")
    print("=" * 72)

    train_cmd = [
        sys.executable, str(ROOT / "RL" / "train_target_throw.py"),
        "--run-name", args.run_name,
        "--output", str(run_dir),
        "--n-envs", str(args.n_envs),
        "--n-steps", str(args.n_steps),
        "--batch-size", str(args.batch_size),
        "--iterations", str(args.iterations),
        "--checkpoint-every", str(args.checkpoint_every),
        "--target-distance", str(args.target_distance),
        "--target-half-size", str(args.target_half_size),
        "--seed", str(args.seed),
        "--device", args.device,
        "--continue-in-place",  # we already created run_dir above; don't let it create a second one
    ]
    if args.resume_from:
        train_cmd += ["--resume-from", str(args.resume_from)]
    if args.learning_rate is not None:
        train_cmd += ["--learning-rate", str(args.learning_rate)]
    print("Starting training:", " ".join(train_cmd))
    train_proc = subprocess.Popen(train_cmd, cwd=str(ROOT))

    tb_log_path = run_dir / "logs" / "tensorboard.log"
    tb_cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(run_dir / "tensorboard"),
        "--port", str(args.tensorboard_port),
    ]
    tb_log = open(tb_log_path, "w")
    print("Starting TensorBoard:", " ".join(tb_cmd), f"(log: {tb_log_path})")
    tb_proc = subprocess.Popen(tb_cmd, cwd=str(ROOT), stdout=tb_log, stderr=subprocess.STDOUT)
    if not args.no_browser:
        time.sleep(3)
        try:
            webbrowser.open(tb_url)
        except Exception:
            pass

    shutdown_done = False

    def shutdown(*_args):
        nonlocal shutdown_done
        if shutdown_done:
            return
        shutdown_done = True
        print("\nShutting down training and TensorBoard...")
        for proc in (train_proc, tb_proc):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in (train_proc, tb_proc):
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        tb_log.close()

    signal.signal(signal.SIGINT, lambda *_: (shutdown(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (shutdown(), sys.exit(0)))

    try:
        if args.viewer:
            if args.viewer_delay > 0:
                print(f"Waiting {args.viewer_delay:.0f}s for training workers to finish loading before starting the viewer...")
                time.sleep(args.viewer_delay)
            viewer_cmd = [
                sys.executable, str(ROOT / "RL" / "multi_viewer.py"),
                "--run-dir", str(run_dir),
                "--n-robots", str(viewer_envs),
                "--rows", str(viewer_rows),
                "--cols", str(viewer_cols),
                "--target-distance", str(args.target_distance),
                "--target-half-size", str(args.target_half_size),
            ]
            print("Starting live viewer:", " ".join(viewer_cmd))
            subprocess.run(viewer_cmd, cwd=str(ROOT))
        else:
            print("Headless production run. Waiting for training to finish (Ctrl+C to stop everything)...")
            train_proc.wait()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
