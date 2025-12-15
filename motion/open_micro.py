#!/usr/bin/env python3
"""
open_microwave_verbose.py

Combined Planner and Player script with high verbosity.
Goal: one-shot waypoint conformity test with simple stage logic.

Stages:
  - Stage 0  (gripper OPEN): move to handle approach (your current waypoints).
  - Stage 01 (gripper CLOSE while holding pose): placeholder "close at handle".
  - Stage 02 (gripper CLOSED): run the next waypoints (your current "stage 1" here).

Outputs:
  CSV columns: 6 arm joints + gripper_ctrl + t

Notes:
  - This is a quick test harness, not a perfect controller.
  - Uses a simple velocity-limited retimer per segment (linear interpolation).
  - Player uses time column to pick closest setpoint by sim time.

"""

import os
import time
import numpy as np
import mujoco
from mujoco import viewer

# ==========================================
# 1. CONFIGURATION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")
CSV_PATH = os.path.join(CURRENT_DIR, "plan_microwave_verbose.csv")

# Joint & Actuator Names
ARM_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3"
]
ARM_ACTUATORS = [
    "shoulder_pan_servo", "shoulder_lift_servo", "elbow_servo",
    "wrist_1_servo", "wrist_2_servo", "wrist_3_servo"
]
GRIPPER_ACTUATOR = "gripper_servo"

# Control Settings
CTRL_HZ = 100.0
CTRL_DT = 1.0 / CTRL_HZ

# Conservative velocity limits for safety during debug
V_MAX = np.array([0.8, 0.8, 0.8, 1.0, 1.0, 1.5], dtype=float)

# Gripper States
GRIP_OPEN = 0.7
GRIP_CLOSED = 0.0

# Stage-01 "close at handle" timing
GRIP_CLOSE_TIME = 0.8   # seconds to close (placeholder)
GRIP_SETTLE_TIME = 0.25 # seconds hold after closing (placeholder)

# ==========================================
# 2. PLANNER LOGIC
# ==========================================

# --------------------------
# Stage 0 (your CURRENT waypoints): approach handle with gripper OPEN
# --------------------------
MANUAL_WAYPOINTS_STAGE_0 = [
    np.array([0.0,    0.0,  0.0,     0.0,    0.0,    0.0], dtype=float),
    np.array([-1.51,  0.0,  0.0,     0.0,    0.0,    0.0], dtype=float),
    np.array([-1.51, -1.1,  0.0,     0.0,    0.0,    0.0], dtype=float),
]

# --------------------------
# Stage 01 (PLACEHOLDERS): hold at handle pose while gripper CLOSES
# - We will HOLD the final Stage-0 pose, so there are no arm waypoints here.
# - We *do* generate a gripper ramp from OPEN -> CLOSED.
# --------------------------
# (No manual arm waypoints needed; arm hold generated programmatically.)

# --------------------------
# Stage 02 (your "stage 1" waypoints): execute after grasp, gripper CLOSED
# --------------------------
MANUAL_WAYPOINTS_STAGE_02 = [
    np.array([-1.38, -1.1,  0.0314, -0.66,   0.0314, 0.0], dtype=float),
]

def retime_waypoints_simple(waypoints, dt, vmax):
    """
    Simple linear interpolation respecting velocity limits.
    Returns a trajectory with shape (N,6).
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 6:
        raise ValueError(f"Waypoints must be (K,6). Got: {waypoints.shape}")

    print(f"[PLANNER] Retiming {len(waypoints)} waypoints...")
    traj_chunks = [waypoints[0][None, :]]

    for i in range(len(waypoints) - 1):
        q_start = waypoints[i]
        q_end = waypoints[i + 1]
        dist = np.abs(q_end - q_start)

        max_dist_idx = int(np.argmax(dist))
        max_dist_val = float(dist[max_dist_idx])

        if max_dist_val < 1e-6:
            steps = 1
            time_needed = 0.0
        else:
            time_needed = float(np.max(dist / vmax))
            steps = int(np.ceil(time_needed / dt))
            steps = max(5, steps)  # Force minimum smoothness

        print(
            f"  > Segment {i}->{i+1}: "
            f"Max dist {max_dist_val:.4f} rad on joint[{max_dist_idx}] | "
            f"time_needed={time_needed:.4f}s | steps={steps}"
        )

        segment = np.linspace(q_start, q_end, steps, endpoint=True)
        traj_chunks.append(segment[1:, :])  # skip start to avoid dupes

    out = np.vstack(traj_chunks)
    print(f"[PLANNER] Retimed trajectory length: {out.shape[0]} steps")
    return out

def generate_hold_segment(q_hold, seconds, dt):
    """
    Hold a pose for N steps.
    Returns (arm_hold, t_local)
    """
    steps = int(np.ceil(seconds / dt))
    steps = max(1, steps)
    arm_hold = np.repeat(q_hold[None, :], steps, axis=0)
    t_local = np.arange(steps, dtype=float) * dt
    return arm_hold, t_local

def generate_gripper_ramp(g0, g1, seconds, dt):
    """
    Linear ramp for gripper from g0 -> g1 over time.
    Returns (g, t_local)
    """
    steps = int(np.ceil(seconds / dt))
    steps = max(1, steps)
    g = np.linspace(float(g0), float(g1), steps, endpoint=True)
    t_local = np.arange(steps, dtype=float) * dt
    return g, t_local

def generate_csv():
    print(f"\n=== GENERATING PLAN ===")
    print(f"Target CSV: {CSV_PATH}")
    print(f"CTRL_DT={CTRL_DT:.6f}s (CTRL_HZ={CTRL_HZ:.1f})")
    print(f"V_MAX={V_MAX}")

    # --------------------------
    # Stage 0 retime (arm motion, gripper OPEN)
    # --------------------------
    print("\n[PLANNER] --- STAGE 0: approach handle (gripper OPEN) ---")
    w0 = np.asarray(MANUAL_WAYPOINTS_STAGE_0, dtype=float)
    traj0 = retime_waypoints_simple(w0, CTRL_DT, V_MAX)
    grip0 = np.full((traj0.shape[0],), GRIP_OPEN, dtype=float)

    # --------------------------
    # Stage 01 (arm hold, gripper CLOSE ramp + settle)
    # --------------------------
    print("\n[PLANNER] --- STAGE 01: close gripper at handle (arm HOLD) ---")
    q_hold = traj0[-1].copy()
    print(f"[PLANNER] Holding pose (end of stage 0): {q_hold}")

    g_ramp, _ = generate_gripper_ramp(GRIP_OPEN, GRIP_CLOSED, GRIP_CLOSE_TIME, CTRL_DT)
    arm_ramp = np.repeat(q_hold[None, :], g_ramp.shape[0], axis=0)

    settle_steps = int(np.ceil(GRIP_SETTLE_TIME / CTRL_DT))
    settle_steps = max(0, settle_steps)
    arm_settle = np.repeat(q_hold[None, :], settle_steps, axis=0) if settle_steps > 0 else np.zeros((0, 6))
    g_settle = np.full((settle_steps,), GRIP_CLOSED, dtype=float) if settle_steps > 0 else np.zeros((0,), dtype=float)

    print(f"[PLANNER] Gripper ramp steps: {g_ramp.shape[0]} ({g_ramp.shape[0]*CTRL_DT:.3f}s)")
    print(f"[PLANNER] Settle steps:      {settle_steps} ({settle_steps*CTRL_DT:.3f}s)")

    traj01 = np.vstack([arm_ramp, arm_settle])
    grip01 = np.concatenate([g_ramp, g_settle])

    # --------------------------
    # Stage 02 retime (arm motion, gripper CLOSED)
    # - IMPORTANT: ensure stage02 starts from the end of stage01 (q_hold)
    # --------------------------
    print("\n[PLANNER] --- STAGE 02: execute after grasp (gripper CLOSED) ---")
    w2 = np.asarray(MANUAL_WAYPOINTS_STAGE_02, dtype=float)

    # If stage02 has only 1 waypoint, we still want *some* motion possibility later.
    # For now: we stitch a start waypoint = q_hold -> waypoint(s).
    # If the first waypoint equals q_hold, this will retime to nearly nothing.
    w2_stitched = np.vstack([q_hold[None, :], w2])
    print(f"[PLANNER] Stage02 stitched waypoints: {w2_stitched.shape[0]}")
    print(f"  start: {w2_stitched[0]}")
    print(f"  end:   {w2_stitched[-1]}")

    traj02 = retime_waypoints_simple(w2_stitched, CTRL_DT, V_MAX)
    # Drop duplicate first row because stage01 already ends at q_hold
    if traj02.shape[0] > 1:
        traj02 = traj02[1:, :]
    grip02 = np.full((traj02.shape[0],), GRIP_CLOSED, dtype=float)

    # --------------------------
    # Stitch all stages
    # --------------------------
    q_traj = np.vstack([traj0, traj01, traj02])
    g_traj = np.concatenate([grip0, grip01, grip02])

    # Time column
    t_col = (np.arange(q_traj.shape[0], dtype=float) * CTRL_DT).reshape(-1, 1)

    full_data = np.hstack([q_traj, g_traj.reshape(-1, 1), t_col])
    header_str = ",".join(ARM_JOINT_NAMES + ["gripper_ctrl", "t"])
    np.savetxt(CSV_PATH, full_data, delimiter=",", header=header_str, comments="")

    print("\n=== PLAN SUMMARY ===")
    print(f"[PLANNER] Total steps: {q_traj.shape[0]}")
    print(f"[PLANNER] Total duration: {float(t_col[-1][0]):.3f}s")
    print(f"[PLANNER] Final Joint Config: {q_traj[-1]}")
    print(f"[PLANNER] Final Gripper Ctrl: {float(g_traj[-1]):.4f}")
    print("[PLANNER] CSV written successfully.")
    return full_data

# ==========================================
# 3. PLAYER LOGIC
# ==========================================

def simulate_plan():
    print(f"\n=== STARTING SIMULATION ===")
    print(f"Loading Model: {XML_PATH}")

    if not os.path.exists(XML_PATH):
        print(f"ERROR: XML file not found at {XML_PATH}")
        return

    # Load Model
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to load MuJoCo model: {e}")
        return

    # Load Plan
    print(f"Reading Plan: {CSV_PATH}")
    try:
        plan_data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    except Exception as e:
        print(f"Failed to read plan CSV: {e}")
        return

    q_ref = plan_data[:, :6]
    g_ref = plan_data[:, 6]
    t_ref = plan_data[:, 7]

    print(f"Loaded {len(t_ref)} plan rows.")
    print(f"Time range: t[0]={t_ref[0]:.3f}s -> t[-1]={t_ref[-1]:.3f}s")

    # Get IDs
    arm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS]
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)

    if any(a < 0 for a in arm_ids) or grip_id < 0:
        print("ERROR: One or more actuator names not found in the model.")
        print(f"  arm_ids={arm_ids}")
        print(f"  grip_id={grip_id}")
        return

    # Initialize Robot at start position
    print(f"\n[PLAYER] Initializing robot to start configuration:")
    print(f"  q_start = {q_ref[0]}")
    print(f"  g_start = {float(g_ref[0]):.4f}")

    # NOTE: This is a quick test harness; we set qpos for the first pose just to start clean.
    # The rest is actuator-driven.
    data.qpos[:6] = q_ref[0]
    data.ctrl[arm_ids] = q_ref[0]
    data.ctrl[grip_id] = float(g_ref[0])
    mujoco.mj_forward(model, data)

    # Diagnostics configs
    VERBOSE_INTERVAL = 50  # Print status every N steps

    # Simulation Loop
    with viewer.launch_passive(model, data) as v:
        start_wall = time.time()
        sim_step = 0

        print("\n[PLAYER] Viewer launched. Executing plan...")

        while v.is_running():
            sim_t = float(data.time)

            # Simple real-time sync: try not to run faster than wall time
            if (time.time() - start_wall) < sim_t:
                time.sleep(0.001)
                continue

            # Find closest index where t_ref >= sim_t (hold last if finished)
            indices = np.where(t_ref >= sim_t)[0]
            if len(indices) > 0:
                idx = int(indices[0])
            else:
                idx = int(len(t_ref) - 1)

            target_q = q_ref[idx]
            target_g = float(g_ref[idx])

            # Apply Control
            data.ctrl[arm_ids] = target_q
            data.ctrl[grip_id] = target_g

            # Verbose Logging
            if sim_step % VERBOSE_INTERVAL == 0:
                j0_err = float(data.qpos[0] - target_q[0])
                print(
                    f"[Step {sim_step}] sim_t={sim_t:.3f}s | plan_idx={idx}/{len(t_ref)-1} | "
                    f"g={target_g:.3f} | j0_err={j0_err:.4f}"
                )

            # Step Physics
            mujoco.mj_step(model, data)

            # Sync Viewer occasionally
            if sim_step % 30 == 0:
                v.sync()

            sim_step += 1

            # Optional stop behavior
            if sim_t > float(t_ref[-1]) + 5.0 and sim_step % 1000 == 0:
                print("[PLAYER] Plan finished. Holding final pose...")


if __name__ == "__main__":
    generate_csv()
    simulate_plan()