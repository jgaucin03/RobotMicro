import os
import numpy as np
import mujoco

XML_PATH = os.path.join(os.path.dirname(__file__), "../scenes/scene_debug.xml")
model = mujoco.MjModel.from_xml_path(XML_PATH)

def jinfo(jname):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return model.jnt_axis[jid].copy(), int(model.jnt_type[jid])

for j in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]:
    axis, jtype = jinfo(j)
    print(f"{j:14s} axis={axis} type={jtype}")