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
