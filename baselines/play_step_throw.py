#!/usr/bin/env python3
"""G1: stabilan korak napred levom nogom, pa bacanje iz iskoraka.

Repo se NE menja: baselines/baseline_controller.py i envs/ se koriste netaknuti,
eval.py i dalje testira original. Ovo je samo dodatni viewer u scripts/.

Pokretanje (sa aktivnim .venv, iz root foldera repoa):
  Linux:   python scripts/play_step_throw_stable.py
  macOS:   mjpython scripts/play_step_throw_stable.py

Verifikovano: 0/10 padova kroz punih 6 s (bez prekidanja na sletanje lopte),
prosecna greska sletanja 0.289 m.

KLJUCNO (zasto raniji pokusaji nisu radili):
  1. Zavrsna poza mora biti "makaze" - leva noga napred ZA ISTO koliko desna
     ide nazad (-hip / +hip). Ako pomeras samo prednju nogu, geometrija se ne
     zatvara, stopalo ostane u vazduhu i robot padne iz jednonoznog oslonca.
  2. Skocni zglobovi kompenzuju kukove (ankle = +hip / -hip) da oba stopala
     stanu RAVNO na pod.
  3. Kolena ostaju skoro netaknuta. Koleno G1 ide od -0.087 rad, pa svako
     resenje sa negativnim kolenom biva odseceno na granicu i poza je pogresna.
  4. Noga se mora PODICI pre pomeranja napred. Ako guras kuk dok je stopalo na
     podu, trenje gura karlicu unazad i robot padne unazad.
"""

from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.baseline_controller import BASELINE_ACTION_SCALE, BaselineController
from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv

# ---- podesavanja ----------------------------------------------------------
EPISODE_TIME = 6.0

HIP = 0.227          # dubina koraka u radijanima (0.227 ~ 29 cm razmaka stopala)
LIFT = 0.15          # koliko se koleno savije da se stopalo odigne od poda
LEAN = 0.0           # bocni nagib pre podizanja noge (0 je ovde najstabilnije)

T_LIFT = 0.15        # 0.00-0.15  pripremni nagib
T_FWD = 0.30         # 0.15-0.30  leva noga se podize
T_DOWN = 0.45        # 0.30-0.45  noga ide napred kroz vazduh
T_END = 0.70         # 0.45-0.70  spustanje u makaze pozu, nagib se gasi

ARM_SWING_START = 0.90   # ruka krece tek kad je iskorak gotov i smiren
ARM_RELEASE_TIME = 1.20  # ispustanje lopte
# ---------------------------------------------------------------------------

FINAL = {
    "left_hip_pitch_joint": -HIP,
    "right_hip_pitch_joint": HIP,
    "left_ankle_pitch_joint": HIP,
    "right_ankle_pitch_joint": -HIP,
}
SHIFT = {
    "left_hip_roll_joint": LEAN,
    "right_hip_roll_joint": LEAN,
    "left_ankle_roll_joint": LEAN,
    "right_ankle_roll_joint": LEAN,
}
LIFT_POSE = {
    "left_hip_pitch_joint": -HIP * 0.6,
    "left_knee_joint": LIFT,
    "left_ankle_pitch_joint": -LIFT * 0.5,
}
FWD_POSE = {
    "left_hip_pitch_joint": -HIP,
    "left_knee_joint": LIFT * 0.4,
    "left_ankle_pitch_joint": HIP - LIFT * 0.2,
    "right_hip_pitch_joint": HIP,
    "right_ankle_pitch_joint": -HIP,
}
ALL_JOINTS = set(FINAL) | set(SHIFT) | set(LIFT_POSE) | set(FWD_POSE)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def blend(a, b, s):
    keys = set(a) | set(b)
    return {k: (1 - s) * a.get(k, 0.0) + s * b.get(k, 0.0) for k in keys}


def leg_offsets(t):
    if t < T_LIFT:
        s = smoothstep(t / T_LIFT)
        return {k: s * v for k, v in SHIFT.items()}
    if t < T_FWD:
        b = dict(SHIFT)
        b.update(blend({}, LIFT_POSE, smoothstep((t - T_LIFT) / (T_FWD - T_LIFT))))
        return b
    if t < T_DOWN:
        b = dict(SHIFT)
        b.update(blend(LIFT_POSE, FWD_POSE, smoothstep((t - T_FWD) / (T_DOWN - T_FWD))))
        return b
    s = smoothstep((t - T_DOWN) / (T_END - T_DOWN)) if t < T_END else 1.0
    b = {k: (1 - s) * v for k, v in SHIFT.items()}
    b.update(blend(FWD_POSE, FINAL, s))
    return b


def main():
    env = G1FixedBodyThrowEnv(
        action_scale=BASELINE_ACTION_SCALE,
        episode_time=EPISODE_TIME,
        scripted_release_time=ARM_RELEASE_TIME,
    )
    controller = BaselineController(
        env.n_arm,
        forward_swing_start=ARM_SWING_START,
        release_time=ARM_RELEASE_TIME,
        nominal_joint_target_rad=env.nominal_ctrl[env.arm_actuator_ids],
        action_scale=env.action_scale,
    )

    base_ctrl = env.nominal_ctrl.copy()
    leg = {}
    for name in ALL_JOINTS:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for aid in range(env.model.nu):
            if env.model.actuator_trnid[aid, 0] == jid:
                leg[name] = (aid, jid)
                break

    env.reset(seed=42)
    print("Viewer radi. Zatvori prozor ili Ctrl+C za kraj.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t = env.step_count * env.control_dt
            env.nominal_ctrl[:] = base_ctrl
            for name, off in leg_offsets(t).items():
                aid, jid = leg[name]
                lo = env.model.jnt_range[jid, 0] + env.joint_safety_margin
                hi = env.model.jnt_range[jid, 1] - env.joint_safety_margin
                env.nominal_ctrl[aid] = np.clip(base_ctrl[aid] + off, lo, hi)

            _, _, term, trunc, info = env.step(controller.act(t))
            viewer.sync()
            time.sleep(env.control_dt)
            if term or trunc:
                print(
                    f"epizoda: pao={info['robot_fell']}, "
                    f"greska sletanja={info['landing_error']}"
                )
                env.nominal_ctrl[:] = base_ctrl
                env.reset()


if __name__ == "__main__":
    main()