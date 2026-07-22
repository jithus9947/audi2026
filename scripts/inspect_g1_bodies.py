#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
xml_path=ROOT/'assets'/'unitree_g1'/'scene_with_hands.xml'
if not xml_path.exists(): raise FileNotFoundError('Copy Unitree G1 assets to assets/unitree_g1 first.')
model=mujoco.MjModel.from_xml_path(str(xml_path))
print('Candidate right hand / wrist body names:')
for i in range(model.nbody):
    name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i)
    if name and ('right' in name.lower() or 'wrist' in name.lower() or 'hand' in name.lower()): print(i,name)
print('\nUse one with: python scripts/create_g1_throw_scene.py --hand-body BODY_NAME')
