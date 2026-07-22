#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
xml_path=ROOT/'assets'/'unitree_g1'/'scene_with_hands.xml'
if not xml_path.exists(): raise FileNotFoundError('Copy Unitree G1 assets to assets/unitree_g1 first.')
model=mujoco.MjModel.from_xml_path(str(xml_path))
print('\nJoints:')
for i in range(model.njnt):
    name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i); print(f'{i:02d}: {str(name):35s} qpos={model.jnt_qposadr[i]} qvel={model.jnt_dofadr[i]}')
print('\nActuators:')
for i in range(model.nu):
    name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i); trnid=model.actuator_trnid[i,0]; jname=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(trnid)) if trnid>=0 else None; print(f'{i:02d}: {str(name):35s} joint={jname}')
