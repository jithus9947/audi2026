#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import time, mujoco, mujoco.viewer
xml_path=ROOT/'assets'/'unitree_g1'/'scene.xml'
if not xml_path.exists(): raise FileNotFoundError('Copy Unitree G1 assets to assets/unitree_g1 first.')
model=mujoco.MjModel.from_xml_path(str(xml_path)); data=mujoco.MjData(model)
if model.nkey>0: mujoco.mj_resetDataKeyframe(model,data,0)
else: mujoco.mj_resetData(model,data)
mujoco.mj_forward(model,data)
with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running(): mujoco.mj_step(model,data); viewer.sync(); time.sleep(model.opt.timestep)
