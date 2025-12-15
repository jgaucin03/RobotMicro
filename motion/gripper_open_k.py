#!/usr/bin/env python3
# motion/debug_gripper_open_k.py
#
# Minimal debug script:
#   - Loads scene_debug.xml
#   - Runs ONLY the "open gripper" phase for a few dozen steps
#   - Prints gripper qpos/qvel each step, plus actuator ctrl
#
# Goal: understand why the gripper visually never seems to open.

import os
import numpy as np
import mujoco

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH    = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

GRIPPER_JOINT_NAMES = [
    "link6_Grip-Driver",
    "link6_Grip-Link",
]

# The open pose we *want* (same as in pick_cup_k.py)
GRIPPER_OPEN_QPOS   = np.array([0.7, -0.7], dtype=float)
OPEN_STEPS          = 80
N_DEBUG_STEPS       = 120   # run a bit past full-open

DT_SCALE = 1.0      # keep dt as-is


def get_gripper_indices(model):
    grip_q_idx   = []
    grip_dof_idx = []
    for jn in GRIPPER_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Gripper joint '{jn}' not found in model.")
        grip_q_idx.append(model.jnt_qposadr[jid])
        grip_dof_idx.append(model.jnt_dofadr[jid])
    return np.array(grip_q_idx, dtype=int), np.array(grip_dof_idx, dtype=int)


def main():
    print(f"Loading: {XML_PATH}")
    if not os.path.exists(XML_PATH):
        print("XML file not found.")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # Optionally scale timestep
    original_dt = model.opt.timestep
    model.opt.timestep = original_dt * DT_SCALE
    dt = model.opt.timestep
    print(f"dt = {dt:.6f}")

    # Gripper joints
    grip_q_idx, grip_dof_idx = get_gripper_indices(model)

    # Look for the gripper position actuator (if present)
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_servo")
    if act_id >= 0:
        print(f"Found actuator 'gripper_servo' with id {act_id}")
    else:
        print("No 'gripper_servo' actuator found.")

    mujoco.mj_forward(model, data)
    q_grip_initial = data.qpos[grip_q_idx].copy()
    print(f"Initial gripper qpos from XML: {q_grip_initial}")

    # Print joint ranges for gripper joints (to see if 0.7/-0.7 is out of range)
    print("Gripper joint ranges:")
    for jn in GRIPPER_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        lo, hi = model.jnt_range[jid]
        print(f"  {jn}: [{lo:.3f}, {hi:.3f}]")

    # Debug loop: only do the "open" interpolation, no path following, no closing
    q_grip_prev = data.qpos[grip_q_idx].copy()

    print("\n=== BEGIN GRIPPER OPEN DEBUG ===")
    for step in range(N_DEBUG_STEPS):
        # Interpolation factor
        alpha = min(1.0, step / max(1, OPEN_STEPS))

        # Interpolate from *current* pose to target open pose (same logic as pick_cup_k)
        q_curr   = data.qpos[grip_q_idx].copy()
        q_target = (1.0 - alpha) * q_curr + alpha * GRIPPER_OPEN_QPOS

        # Teleport gripper in qpos
        data.qpos[grip_q_idx] = q_target

        # Teleport-aware velocity: set qvel from finite difference or zero on teleport
        # Here we always compute finite-difference from previous qpos.
        q_grip_new = data.qpos[grip_q_idx].copy()
        data.qvel[grip_dof_idx] = (q_grip_new - q_grip_prev) / dt
        q_grip_prev = q_grip_new.copy()

        # (Optional) explicitly set gripper control; by default ctrl is 0, which
        # for a position actuator means "target = 0".
        if act_id >= 0:
            # For pure diagnosis, leave this as-is; we're *not* trying to fix yet.
            ctrl_val = data.ctrl[act_id]
        else:
            ctrl_val = None

        # Step physics ONCE
        mujoco.mj_step(model, data)

        # After physics step, read back state
        q_after  = data.qpos[grip_q_idx].copy()
        qvel_after = data.qvel[grip_dof_idx].copy()

        # Print detailed info
        if act_id >= 0:
            ctrl_str = f"{data.ctrl[act_id]:.4f}"
        else:
            ctrl_str = "N/A"

        print(
            f"step {step:03d}: alpha={alpha:5.3f} | "
            f"q_before={q_target} -> q_after={q_after} | "
            f"qvel={qvel_after} | ctrl(gripper_servo)={ctrl_str}"
        )

    print("=== END GRIPPER OPEN DEBUG ===")


if __name__ == "__main__":
    main()