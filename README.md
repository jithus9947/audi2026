# Humanoid Robot Fundamentals

A MuJoCo, Gymnasium, and Stable-Baselines3 project for the Unitree G1 29-DoF humanoid.

## Ball-drop task

The robot must move its right arm forward and release a 50 g ball so its first ground contact falls in a target zone directly in front of the robot.

- Ball radius: 4 cm
- Target centre: `(0.45 m, 0.0 m)` on the floor
- Success radius: 18 cm
- Hand model: rigid five-finger visual hand; finger control is intentionally out of scope
- Arm safety: all arm commands are clipped to MuJoCo joint ranges with an 0.08 rad margin

The target and ball scene are generated as `assets/unitree_g1/scene_throw.xml` from the Menagerie `scene.xml` model.

## Setup

```bash
git clone https://github.com/jithus9947/audi2026.git
cd audi2026
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Fetch the G1 assets once:

```bash
mkdir -p external assets/unitree_g1
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git external/mujoco_menagerie
git -C external/mujoco_menagerie sparse-checkout set unitree_g1
git -C external/mujoco_menagerie archive HEAD unitree_g1 | tar -x -C assets --strip-components=1
```

Generate the ball-drop scene:

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

To visualize the same controller on macOS, use MuJoCo's `mjpython` launcher:

```bash
mjpython scripts/play_baseline_g1_throw.py
```

Close the MuJoCo window or press `Ctrl+C` to stop it.

## Evaluation metrics

Both the baseline and PPO should be evaluated with the same metrics:

- success rate;
- first-ground-contact XY distance to the target;
- completion time;
- robot fall status;
- action smoothness.

The current evaluator records success, landing error, release time, and reward. Add fall and smoothness reporting before final PPO comparison.
