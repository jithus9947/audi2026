# Humanoid Robot Fundamentals

A MuJoCo, Gymnasium, and Stable-Baselines3 project for the Unitree G1 29-DoF humanoid.

## Ball-drop task

The robot must move its right arm forward and release a 50 g ball so its first ground contact falls in a target zone directly in front of the robot.

- Ball radius: 4 cm
- Target centre: `(0.45 m, 0.0 m)` on the floor
- Success radius: 18 cm
- Hand model: rigid five-finger visual hand; finger control is intentionally out of scope
- Arm safety: all arm commands are clipped to MuJoCo joint ranges with an 0.08 rad margin

The repository includes the Unitree G1 model assets and the generated
`assets/unitree_g1/scene_throw.xml` ball scene.

## Setup

```bash
git clone https://github.com/jithus9947/audi2026.git
cd audi2026
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, use:

```powershell
git clone https://github.com/jithus9947/audi2026.git
cd audi2026
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No separate Unitree or MuJoCo Menagerie clone is required. The G1 XML, meshes,
textures, license, and generated ball scene are already under
`assets/unitree_g1/`.

Only regenerate the scene after intentionally changing its source or hand
attachment settings:

```bash
python scripts/create_g1_throw_scene.py --hand-body right_wrist_yaw_link
```

## Run and evaluate the baseline

The deterministic baseline smoothly extends the right arm, keeps the ball 16 cm ahead of the wrist for clearance, and releases it at about 0.74 s.

```bash
python evaluation_scripts/eval.py
```

Current deterministic baseline result (10 episodes):

| Metric | Result |
| --- | ---: |
| Success rate | 100% |
| Average landing error | 0.081 m |
| Average reward | 10.46 |
| Average release time | 0.740 s |

To visualize the baseline, use the command for your operating system.

macOS requires MuJoCo's `mjpython` launcher:

```bash
mjpython scripts/play_baseline_g1_throw.py
```

Linux:

```bash
python scripts/play_baseline_g1_throw.py
```

Windows PowerShell:

```powershell
python scripts\play_baseline_g1_throw.py
```

Close the MuJoCo window or press `Ctrl+C` to stop it.

## Reinforcement learning pipeline

The repository also includes an `RL/` pipeline for training and evaluating a PPO agent on the same G1 ball-drop task.

- `RL/train.py` trains PPO and saves run artifacts under `RL/runs/<run-name>/`.
- `RL/evaluate.py` evaluates both the scripted baseline and the trained PPO policy under the same task settings.
- `RL/plot_results.py` renders training curves and baseline-vs-PPO comparison figures.
- `RL/README.md` contains the full PPO pipeline documentation and usage examples.

Run the RL pipeline with:

```bash
python RL/train.py
python RL/evaluate.py
python RL/plot_results.py
```

Use TensorBoard to inspect training logs:

```bash
tensorboard --logdir RL/runs
```

### Tune the free-throw baseline

Edit the values at the top of `baselines/baseline_controller.py`, then restart
the viewer. The seven values use this order:

```text
right shoulder pitch, right shoulder roll, right shoulder yaw,
right elbow, right wrist roll, right wrist pitch, right wrist yaw
```

`SAFE_START_ACTION` is the pose before the throw and should keep the hand well
away from the legs; it uses normalised values from `-1` to `1`.
`THROW_END_JOINT_TARGET_RAD` is the forward end pose in radians. The viewer
prints if any supplied joint value is clipped to a G1 safety limit. Increase
`RELEASE_TIME` for a slower arm swing or reduce it slightly for a faster throw.
Keep `FORWARD_SWING_START < RELEASE_TIME`.

The baseline viewer uses a wider, baseline-only arm action range so the stated
radian targets can actually be reached. PPO continues to use its original
action scaling and is not affected.

## Level 0 + Level 1 target-throw RL task

`envs/g1_target_throw_env.py` adds a new task on top of the existing pipeline
without changing `envs/g1_fixed_body_throw_env.py`, `baselines/baseline_controller.py`,
`baselines/play_step_throw.py`, or any `scripts/play_*.py`:

- **Level 0 (compulsory):** the pelvis must stay fixed (no drifting, no turning) while throwing.
- **Level 1, option 1:** land the ball inside a small square target 5 m away.

Every requirement is its own reward term in `envs/reward_components.py`
(pelvis stability, forward/backward arm motion, side-to-side wrist rotation,
self-collision, foot/ground contact, projectile straightness) plus a set of
one-time landing-event terms scored only from the ball's first confirmed
landing position (distance progress, dense landing accuracy, undershoot/
overshoot, lateral accuracy, target-hit bonus) -- see the event state machine
in `envs/g1_target_throw_env.py` for how release/landing are each detected
exactly once per episode.

Every invocation creates its own timestamped run directory under
`RL/runs/<run-name>/<timestamp>_iter<N>_env<E>_seed<S>/` -- nothing from a
previous run (checkpoints, best models, TensorBoard logs, CSVs) is ever
overwritten. See `docs/training_pipeline.md`-style summary below for the
full layout, or just run `python RL/compare_runs.py` to see every run you've
done so far.

Start everything -- PPO training (headless by default) + TensorBoard -- with
one command:

```bash
python RL/run.py
```

TensorBoard opens automatically at `http://localhost:6006`. Pass `--viewer`
to also open the tiled MuJoCo window (costs FPS -- off by default for
production training). Ctrl+C stops both subprocesses cleanly (saves an
emergency checkpoint first).

Benchmark parallel-env counts on your hardware before a long run (does not
assume more envs = faster; measures each, recommends the best):

```bash
python RL/benchmark_envs.py --benchmark-envs 8 16 24 32 --benchmark-iterations 10 --device auto
```

Run pieces individually if preferred:

```bash
python RL/train_target_throw.py --iterations 4000 --n-envs 16 --checkpoint-every 500
tensorboard --logdir RL/runs/target_throw_5m/<run-id>/tensorboard
python RL/multi_viewer.py --run-dir RL/runs/target_throw_5m/<run-id>
python RL/compare_runs.py                 # table of every run's final metrics
python RL/storage_report.py               # disk usage per run (report-only, no deletion)
```

### Fine-tune an existing checkpoint

Continues the timestep count instead of resetting it, and writes to a new
run directory so the original run is never overwritten:

```bash
python RL/train_target_throw.py \
  --resume-from RL/runs/target_throw/final_model.zip \
  --output RL/runs/target_throw_5m_finetune \
  --iterations 1000 --checkpoint-every 100 --eval-interval 50 \
  --learning-rate 1e-4 --debug-events
```

Or via the single-command launcher: `python RL/run.py --resume-from RL/runs/target_throw/final_model.zip --iterations 1000`.

### Evaluate and debug

```bash
python RL/evaluate_throw.py --model RL/runs/target_throw/final_model.zip --episodes 100
python RL/evaluate_throw.py --model RL/runs/target_throw/best_models/best_landing_error_model.zip --episodes 20 --render --debug-events
python RL/visual_debug.py --model RL/runs/target_throw/final_model.zip   # draws target box, release/landing points, trajectory, error line
python -m pytest RL/test_target_throw_events.py -v                       # geometry/event unit tests
```

Each run directory also gets `episodes.csv` (one row per completed episode),
`run_manifest.json` (args + reward weights + resume provenance), and
`best_models/` (best-by-reward, best-by-landing-error, best-by-hit-rate,
best-by-forward-distance, each tracked and saved independently).

## Evaluation metrics

Both the baseline and PPO should be evaluated with the same metrics:

- success rate;
- first-ground-contact XY distance to the target;
- completion time;
- robot fall status;
- action smoothness.

The current evaluator records success, landing error, release time, and reward. Add fall and smoothness reporting before final PPO comparison.

## PPO training

After the baseline scene has been generated, train PPO with the same 8-action
interface: seven right-arm targets and one ball-release command.

```bash
python scripts/train_ppo_g1_drop.py --timesteps 200000
```

Models and evaluation checkpoints are written under `policies/g1_ball_drop_ppo/`.
Use the best checkpoint for the final comparison:

```bash
python evaluation_scripts/evaluate_ppo.py \
  --model policies/g1_ball_drop_ppo/best_model.zip \
  --episodes 20
```

The PPO evaluator reports success rate, landing error, completion time, robot
falls, and mean action change (smoothness), matching the intended comparison
metrics.

To watch the learned PPO policy on macOS, use `mjpython`:

```bash
mjpython scripts/play_ppo_g1_drop.py \
  --model policies/g1_ball_drop_ppo/best_model.zip
```

On Linux:

```bash
python scripts/play_ppo_g1_drop.py \
  --model policies/g1_ball_drop_ppo/best_model.zip
```

On Windows PowerShell:

```powershell
python scripts\play_ppo_g1_drop.py `
  --model policies\g1_ball_drop_ppo\best_model.zip
```

Use `--speed 0.5` for slow motion. The viewer continually resets and plays new
episodes; it does not train while the window is open.
