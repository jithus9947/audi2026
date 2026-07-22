#!/usr/bin/env python3
import mujoco
xml = """<mujoco><worldbody><body name='box' pos='0 0 1'><joint type='free'/><geom type='box' size='0.1 0.1 0.1'/></body></worldbody></mujoco>"""
model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
for _ in range(100): mujoco.mj_step(model,data)
print("MuJoCo simulation test completed successfully.")
print("Final box position:", data.qpos[:3])
