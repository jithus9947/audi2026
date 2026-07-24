#!/usr/bin/env python3
"""Train (or resume/fine-tune) PPO on the G1 Level 0 + Level 1 target-throw task.

Level 0 (compulsory): keep the pelvis fixed while throwing.
Level 1 option 1: land the ball inside a small square target, a fixed
distance away (default 5.0 m -- do not curriculum this closer during
training).

Does not touch envs/g1_fixed_body_throw_env.py, baselines/baseline_controller.py,
baselines/play_step_throw.py, or any scripts/play_*.py.

Every invocation gets its own timestamped run directory under
RL/runs/<run-name>/ -- nothing from a previous run is ever overwritten.

Fresh run:
    python RL/train_target_throw.py --iterations 4000 --n-envs 16 --checkpoint-every 500

Benchmark parallel-env counts first (does not train):
    python RL/train_target_throw.py --benchmark-envs 8 16 24 32 --benchmark-iterations 10

Fine-tune from an existing checkpoint (new run directory, timesteps continue):
    python RL/train_target_throw.py \\
        --resume-from RL/runs/sanity_check/final_model.zip \\
        --iterations 1000 --checkpoint-every 100 --eval-interval 50 \\
        --learning-rate 1e-4
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.g1_target_throw_env import G1TargetThrowEnv
from envs.reward_components import RewardShapingParams, RewardWeights, print_reward_weight_report
from RL.benchmark_envs import benchmark_env_counts, valid_batch_size
from RL.best_metric_callback import BestMetricEvalCallback
from RL.checkpoint_callback import DescriptiveCheckpointCallback
from RL.episode_logging_callback import EpisodeLoggingCallback
from RL.hardware_diagnostics import get_git_info, print_startup_diagnostics, select_device, system_info_as_dict
from RL.interrupt_handler_callback import InterruptHandlerCallback
from RL.metrics_callback import TaskMetricsCallback
from RL.progress_display_callback import ProgressDisplayCallback
from RL.run_management import (
    Tee,
    atomic_model_save,
    create_run_directory,
    final_model_filename,
    interrupted_model_filename,
    timestamp_now,
    write_json_atomic,
)


def make_env(rank: int, seed: int, target_distance: float, target_half_size: float, verbose_init: bool):
    def _init():
        env = G1TargetThrowEnv(
            learned_release=True,
            target_pos=(target_distance, 0.0, 0.01),
            target_half_size=target_half_size,
            verbose_init=verbose_init and rank == 0,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="target_throw_5m", help="Experiment family; runs live under RL/runs/<run-name>/.")
    parser.add_argument("--run-label", default=None, help="Optional short label appended to the run directory name.")
    parser.add_argument("--n-envs", "--num-envs", dest="n_envs", type=int, default=16, help="Parallel environments (subprocesses).")
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout steps per env per iteration.")
    parser.add_argument("--iterations", type=int, default=4000, help="PPO rollout+update cycles for THIS run.")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Iterations between checkpoints.")
    parser.add_argument("--eval-interval", type=int, default=50, help="Iterations between deterministic best-model evals.")
    parser.add_argument("--eval-episodes", type=int, default=50, help="More episodes = less noisy best-model selection; 50-100 recommended.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=None, help="Default 3e-4 fresh / keep loaded schedule when resuming unless set.")
    parser.add_argument("--ent-coef", type=float, default=None, help="Default 0.01 fresh / keep loaded value when resuming unless set. Raise temporarily to re-inject exploration into a converged/plateaued policy.")
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--target-distance", type=float, default=5.0, help="Meters to the target. Fixed -- no curriculum.")
    parser.add_argument("--target-half-size", type=float, default=0.35, help="Half-side of the square target, meters.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-threads", type=int, default=None, help="torch.set_num_threads(); default leaves PyTorch's own heuristic.")
    parser.add_argument("--start-method", default=None, choices=[None, "fork", "spawn", "forkserver"])
    parser.add_argument("--cpu-affinity", action="store_true", help="Best-effort: pin the main process off 2 cores, leaving headroom for OS/TensorBoard/viewer. Linux only, no-op elsewhere.")
    parser.add_argument("--render-mode", default="none", choices=["none", "single", "tiled"], help="'single'/'tiled' just print pointers to visual_debug.py/multi_viewer.py -- training itself is always headless (SubprocVecEnv can't render inline).")
    parser.add_argument("--output", type=Path, default=None, help="Explicit run directory (bypasses the automatic timestamped naming). Still never overwrites an existing dir's checkpoints.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Existing .zip checkpoint to continue training from.")
    parser.add_argument("--continue-in-place", action="store_true", help="When resuming, write into --output instead of a fresh timestamped dir. Default is a NEW run directory.")
    parser.add_argument("--debug-events", action="store_true", help="Print one line per completed throw (reduces FPS). OFF by default.")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--progress-poll-interval", type=float, default=7.0, help="Seconds between console progress lines / manifest updates.")
    parser.add_argument("--benchmark-envs", type=int, nargs="+", default=None, help="If set, benchmark these env counts and exit instead of training.")
    parser.add_argument("--benchmark-iterations", type=int, default=10)
    return parser.parse_args()


def maybe_set_cpu_affinity(n_envs: int) -> None:
    if not hasattr(__import__("os"), "sched_setaffinity"):
        print("[cpu-affinity] not supported on this platform, ignoring --cpu-affinity.")
        return
    import os

    total = os.cpu_count() or n_envs
    reserved = min(2, max(total - n_envs, 0))
    usable = max(total - reserved, 1)
    try:
        os.sched_setaffinity(0, set(range(usable)))
        print(f"[cpu-affinity] pinned main process to {usable}/{total} cores, leaving {reserved} for OS/TensorBoard/viewer.")
    except OSError as exc:
        print(f"[cpu-affinity] failed to set affinity ({exc}); continuing without it.")


def main() -> None:
    args = parse_args()

    if args.benchmark_envs:
        device = select_device(args.device)
        results, recommended = benchmark_env_counts(
            args.benchmark_envs, args.n_steps, args.benchmark_iterations, device,
            args.target_distance, args.target_half_size, args.seed,
        )
        if recommended:
            print(
                "\nSuggested production command:\n"
                f"  python RL/train_target_throw.py --run-name {args.run_name} --n-envs {recommended['n_envs']} "
                f"--n-steps {args.n_steps} --batch-size {recommended['batch_size']} --device {device} "
                f"--iterations {args.iterations} --checkpoint-every {args.checkpoint_every}"
            )
        return

    device = select_device(args.device)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    if args.cpu_affinity:
        maybe_set_cpu_affinity(args.n_envs)
    if args.render_mode != "none":
        print(
            f"--render-mode {args.render_mode}: training remains headless (SubprocVecEnv workers can't render "
            f"inline). Use `python RL/visual_debug.py` (single) or `python RL/multi_viewer.py` (tiled) "
            f"separately, pointed at this run's checkpoints, to watch the policy."
        )

    resumed = args.resume_from is not None
    base_dir = ROOT / "RL" / "runs" / args.run_name
    if args.output and (not resumed or args.continue_in_place):
        run_dir = args.output
        run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("checkpoints", "best_models", "tensorboard", "csv", "logs", "config", "models"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_directory(base_dir, args.iterations, args.n_envs, args.seed, resumed=resumed, run_label=args.run_label)
    run_id = run_dir.name
    csv_path = args.csv_path or (run_dir / "csv" / "episodes.csv")

    console_log = open(run_dir / "logs" / "training_console.log", "a")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, console_log)

    try:
        print(f"Run directory: {run_dir}")

        reward_weights = RewardWeights()
        print_reward_weight_report(
            reward_weights, RewardShapingParams(), args.target_distance, args.target_half_size,
            ball_radius=0.04,  # matches assets/unitree_g1/scene_throw.xml throw_ball_geom size
        )

        env_fns = [
            make_env(i, args.seed, args.target_distance, args.target_half_size, verbose_init=True)
            for i in range(args.n_envs)
        ]
        train_env = SubprocVecEnv(env_fns, start_method=args.start_method)

        rollout_size = args.n_steps * args.n_envs
        if rollout_size % args.batch_size != 0:
            print(
                f"[warning] --batch-size {args.batch_size} does not evenly divide rollout_size {rollout_size}; "
                f"SB3 will use a smaller final minibatch each epoch. A clean divisor would be "
                f"{valid_batch_size(rollout_size)}."
            )
        timesteps_per_iteration = rollout_size
        additional_timesteps = args.iterations * timesteps_per_iteration

        previous_timesteps = 0
        source_run_dir = None
        if resumed:
            if not args.resume_from.is_file():
                raise SystemExit(f"--resume-from checkpoint not found: {args.resume_from}")
            custom_objects = {"learning_rate": args.learning_rate} if args.learning_rate is not None else {}
            try:
                model = PPO.load(str(args.resume_from), env=train_env, device=device, custom_objects=custom_objects)
            except ValueError as exc:
                raise SystemExit(
                    "Checkpoint is INCOMPATIBLE with the current observation/action space "
                    f"(obs={train_env.observation_space}, action={train_env.action_space}). "
                    f"Refusing to silently load a mismatched checkpoint. Original error: {exc}"
                ) from exc
            previous_timesteps = model.num_timesteps
            source_run_dir = str(args.resume_from.resolve().parent.parent)
            # PPO.load() restores tensorboard_log from the SOURCE run's saved
            # state (e.g. RL/runs/sanity_check/tensorboard), not this new run
            # directory -- without this override, new logs would keep going
            # to the old run's folder while TensorBoard watches the new
            # (empty) one and never appears to update.
            model.tensorboard_log = str(run_dir / "tensorboard")
            if args.ent_coef is not None:
                # ent_coef is read as a plain attribute each train() call (unlike
                # learning_rate, which goes through a schedule function), so a
                # direct assignment after load takes effect immediately -- no
                # custom_objects indirection needed.
                print(f"Overriding ent_coef: {model.ent_coef} -> {args.ent_coef}")
                model.ent_coef = args.ent_coef
            print(f"Resumed from {args.resume_from} at {previous_timesteps:,} timesteps.")
            reset_num_timesteps = False
        else:
            model = PPO(
                "MlpPolicy", train_env,
                learning_rate=args.learning_rate if args.learning_rate is not None else 3e-4,
                n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                gamma=0.99, gae_lambda=0.95, ent_coef=args.ent_coef if args.ent_coef is not None else 0.01, target_kl=args.target_kl,
                verbose=1, seed=args.seed, device=device,
                tensorboard_log=str(run_dir / "tensorboard"),
            )
            reset_num_timesteps = True

        expected_final_timesteps = previous_timesteps + additional_timesteps
        system_info = print_startup_diagnostics(device, args.n_envs, args.batch_size, rollout_size)
        git_info = get_git_info()
        print(f"git commit: {git_info.commit}  branch: {git_info.branch}  dirty: {git_info.dirty}")
        print(f"timesteps_per_iteration = {args.n_steps} * {args.n_envs} = {timesteps_per_iteration:,}")
        print(f"additional_timesteps = {args.iterations} * {timesteps_per_iteration:,} = {additional_timesteps:,}")
        print(f"expected_final_timesteps = {previous_timesteps:,} + {additional_timesteps:,} = {expected_final_timesteps:,}")

        start_timestamp = timestamp_now()
        manifest = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "start_timestamp": start_timestamp,
            "end_timestamp": None,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "requested_iterations": args.iterations,
            "completed_iterations": None,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "rollout_size": rollout_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": device,
            "seed": args.seed,
            "target_distance": args.target_distance,
            "source_checkpoint": str(args.resume_from) if args.resume_from else None,
            "source_run_dir": source_run_dir,
            "resumed": resumed,
            "initial_global_timestep": previous_timesteps,
            "final_global_timestep": None,
            "final_model_path": None,
            "best_model_paths": {},
            "tensorboard_path": str(run_dir / "tensorboard"),
            "csv_paths": {"episodes": str(csv_path), "evaluations": str(run_dir / "csv" / "evaluations.csv")},
            "git": asdict(git_info),
            "reward_weights": asdict(reward_weights),
            "exit_status": "running",
            "hardware": system_info_as_dict(),
        }
        write_json_atomic(run_dir / "run_manifest.json", manifest)
        write_json_atomic(run_dir / "config" / "training_config.json", manifest["args"])
        write_json_atomic(run_dir / "config" / "reward_config.json", manifest["reward_weights"])
        write_json_atomic(run_dir / "config" / "hardware_info.json", manifest["hardware"])

        checkpoint_callback = DescriptiveCheckpointCallback(
            save_dir=run_dir / "checkpoints", save_every_iterations=args.checkpoint_every,
            resumed=resumed, previous_timesteps=previous_timesteps, verbose=1,
        )
        metrics_callback = TaskMetricsCallback()
        episode_logging_callback = EpisodeLoggingCallback(csv_path=csv_path, print_debug=args.debug_events)
        best_metric_callback = BestMetricEvalCallback(
            eval_env_fn=lambda: G1TargetThrowEnv(
                learned_release=True, target_pos=(args.target_distance, 0.0, 0.01),
                target_half_size=args.target_half_size, verbose_init=False,
            ),
            save_dir=run_dir / "best_models", eval_every_iterations=args.eval_interval,
            n_eval_episodes=args.eval_episodes, verbose=1,
        )
        interrupt_callback = InterruptHandlerCallback(verbose=1)
        progress_callback = ProgressDisplayCallback(
            target_timesteps=expected_final_timesteps, run_dir=run_dir, manifest=manifest,
            task_metrics_callback=metrics_callback, device=device, n_envs=args.n_envs,
            poll_interval_s=args.progress_poll_interval,
        )

        print(
            f"Training PPO: {args.iterations} iterations x {args.n_steps} steps x {args.n_envs} envs "
            f"= {additional_timesteps:,} new timesteps (target distance fixed at {args.target_distance} m)."
        )
        try:
            model.learn(
                total_timesteps=additional_timesteps,
                reset_num_timesteps=reset_num_timesteps,
                callback=CallbackList(
                    [checkpoint_callback, metrics_callback, episode_logging_callback, best_metric_callback, interrupt_callback, progress_callback]
                ),
                progress_bar=True,
                tb_log_name="PPO",
            )
            interrupted = interrupt_callback.interrupted
        except KeyboardInterrupt:
            interrupted = True
            episode_logging_callback._on_training_end()  # normal on_training_end() won't run on a raised exception

        end_timestamp = timestamp_now()
        final_timestep = model.num_timesteps
        completed_iterations = (final_timestep - previous_timesteps) // timesteps_per_iteration

        if interrupted:
            emergency_name = interrupted_model_filename(end_timestamp, final_timestep)
            emergency_path = atomic_model_save(model, run_dir / "models" / emergency_name)
            manifest["exit_status"] = "interrupted"
            manifest["final_model_path"] = str(emergency_path)
            print(f"\n[interrupted] emergency checkpoint saved: {emergency_path}")
            print("Resume with:")
            print(
                f"  python RL/train_target_throw.py --resume-from {emergency_path} "
                f"--run-name {args.run_name} --iterations <remaining_iterations> "
                f"--n-envs {args.n_envs} --device {device}"
            )
        else:
            final_name = final_model_filename(end_timestamp, completed_iterations, args.n_envs, final_timestep, args.seed)
            final_descriptive_path = atomic_model_save(model, run_dir / "models" / final_name)
            atomic_model_save(model, run_dir / "models" / "final_model.zip")
            atomic_model_save(model, run_dir / "models" / "latest_model.zip")
            manifest["exit_status"] = "completed"
            manifest["final_model_path"] = str(final_descriptive_path)
            print(f"\nTraining complete. Final model: {final_descriptive_path}")

        buf = metrics_callback._episode_buffers
        import numpy as np

        manifest["end_timestamp"] = end_timestamp
        manifest["completed_iterations"] = int(completed_iterations)
        manifest["final_global_timestep"] = int(final_timestep)
        manifest["runtime_s"] = None  # filled below once we can parse start/end timestamps
        manifest["best_model_paths"] = {k: str(v) for k, v in best_metric_callback.best_paths.items()}
        manifest["final_metrics"] = {
            "mean_reward": float(np.mean([e["r"] for e in model.ep_info_buffer])) if model.ep_info_buffer else None,
            "mean_forward_distance_m": float(np.mean(buf["forward_distance"])) if buf.get("forward_distance") else None,
            "mean_landing_error_m": float(np.mean(buf["landing_error"])) if buf.get("landing_error") else None,
            "target_hit_rate": float(np.mean(buf["target_hit"])) if buf.get("target_hit") else None,
        }
        from datetime import datetime

        fmt = "%Y%m%d_%H%M%S"
        manifest["runtime_s"] = (datetime.strptime(end_timestamp, fmt) - datetime.strptime(start_timestamp, fmt)).total_seconds()
        write_json_atomic(run_dir / "run_manifest.json", manifest)

        print(f"Run artifacts: {run_dir}")
    finally:
        sys.stdout = real_stdout
        console_log.close()


if __name__ == "__main__":
    main()
