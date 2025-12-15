#!/usr/bin/env python3
import os
import numpy as np
import mujoco

# Paths / names must match your project
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH    = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

GRIPPER_JOINT_NAMES = [
    "link6_Grip-Driver",
    "link6_Grip-Link",
]

# Your "open" configuration for planning
GRIPPER_OPEN_QPOS = np.array([0.7, -0.7], dtype=float)

# Your best goal arm configuration (from planner / manual tuning)
Q_GOAL_ARM = np.array([
    -3.11,  # shoulder_pan
    -1.88,  # shoulder_lift
     1.26,  # elbow
    -1.73,  # wrist_1
    -1.57,  # wrist_2
    -1.85,  # wrist_3
], dtype=float)


def get_q_indices(model, joint_names):
    q_idx = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"Joint '{jn}' not found in model.")
        q_idx.append(model.jnt_qposadr[jid])
    return np.asarray(q_idx, dtype=int)


def main():
    print(f"Loading: {XML_PATH}")
    if not os.path.exists(XML_PATH):
        print("XML file not found.")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    arm_q_idx    = get_q_indices(model, ARM_JOINT_NAMES)
    grip_q_idx   = get_q_indices(model, GRIPPER_JOINT_NAMES)

    # Set arm to q_goal, gripper to "open"
    data.qpos[arm_q_idx]  = Q_GOAL_ARM
    data.qpos[grip_q_idx] = GRIPPER_OPEN_QPOS

    mujoco.mj_forward(model, data)

    print("\nq_goal (arm):", Q_GOAL_ARM)
    print("gripper (open):", GRIPPER_OPEN_QPOS)
    print(f"Number of contacts at (q_goal + open gripper): {data.ncon}")

    for i in range(data.ncon):
        con = data.contact[i]
        g1  = con.geom1
        g2  = con.geom2
        dist = con.dist
        name1 = model.geom(g1).name
        name2 = model.geom(g2).name
        print(f"  contact[{i}] {name1} <-> {name2} dist={dist:.9e}")


if __name__ == "__main__":
    main()