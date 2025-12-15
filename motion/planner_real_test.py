#!/usr/bin/env python3
# TEST SUCCESSFUL!!!!
# motion/planner_real.py
#
# Robot-real "workspace probing" planner:
# - Purpose: generate a slow, safe, actuator-driven waypoint path to explore reachable joint-space
#   configurations in the real (physical) workspace.
# - Uses scene_debug.xml (same model), but different output filename and intent.
# - Unwraps joint angles to avoid 3.14 <-> -3.13 discontinuities.
# - Retimes with velocity limits at a fixed control rate (CTRL_DT).
# - Automatically scales velocity limits to hit a *target total duration* (e.g., ~30s).
# - Outputs CSV with 8 columns: 6 arm joints + gripper_ctrl + t.
#
# NEW (UR3 "real joint" frame export):
# - Your physical UR3 joint angle "zero" differs from the MuJoCo model for:
#     shoulder_lift: real = mjc - pi/2
#     wrist_1:      real = mjc - pi/2
#   Others are congruent.
#
# - This script now writes TWO versions:
#     1) path_real_*            : REAL-ROBOT joint coordinates (with offsets applied)
#     2) path_real_mjcframe_*   : MuJoCo frame joint coordinates (for easy sim playback)
#
# Output files:
#   - motion/path_real_latest.csv                 (REAL, always overwritten)
#   - motion/path_real_<unix_timestamp>.csv       (REAL, archived)
#   - motion/path_real_mjcframe_latest.csv        (MJC, always overwritten)
#   - motion/path_real_mjcframe_<unix_timestamp>.csv (MJC, archived)

import os
import time
import numpy as np
import mujoco

from planlib import (
    unwrap_waypoints,
    retime_waypoints,
    write_plan_csv,
    append_time_column,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

OUT_LATEST_REAL = os.path.join(CURRENT_DIR, "path_real_latest.csv")
OUT_LATEST_MJC  = os.path.join(CURRENT_DIR, "path_real_mjcframe_latest.csv")

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# -------------------------
# Manual waypoints (single stage) IN MuJoCo frame
# -------------------------
# NOTE: fixed your typo:
#   np.array([2.7,  -1.5   1.2, ...])  ->  np.array([2.7, -1.5, 1.2, ...])
MANUAL_WAYPOINTS_STAGE_1 = [
    np.array([0.0,  0.0,   0.0,   0.0,   0.0,   0.0], dtype=float),
    np.array([2.7,  0.0,   0.0,   0.0,   0.0,   0.0], dtype=float),

    np.array([2.7,  0.0,   0.0,  -1.7,   0.0,   0.0], dtype=float),
    np.array([2.7,  0.0,   0.0,  -1.7,  -1.7,   0.0], dtype=float),
    np.array([2.7,  0.0,   0.0,  -1.7,  -1.7,  -1.7], dtype=float),

    np.array([2.7,  0.0,   1.2,  -1.7,  -1.7,  -1.7], dtype=float),
    np.array([2.7, -1.5,   1.2,  -1.7,  -1.7,  -1.7], dtype=float),
]

# -------------------------
# Control discretization
# -------------------------
CTRL_HZ = 100.0
CTRL_DT = 1.0 / CTRL_HZ

# -------------------------
# “Slow test” settings
# -------------------------
TARGET_TOTAL_TIME = 30.0
V_MAX_BASE = np.array([0.6, 0.6, 0.7, 1.2, 1.2, 1.5], dtype=float)
AUTO_SCALE_ITERS = 2

# -------------------------
# Gripper schedule
# -------------------------
GRIP_OPEN = 0.7

# -------------------------
# UR3 REAL joint offsets relative to MuJoCo frame
# -------------------------
# Your statement: when MuJoCo joint angle = 0,
#   real shoulder_lift reads -pi/2
#   real wrist_1 reads      -pi/2
#
# That means: real = mjc + offset, where offset = -pi/2 on those joints.
REAL_JOINT_OFFSETS = np.array([
    0.0,               # shoulder_pan
    -0.5 * np.pi,      # shoulder_lift  (aka shoulder_tilt in your note)
    0.0,               # elbow
    -0.5 * np.pi,      # wrist_1
    0.0,               # wrist_2
    0.0,               # wrist_3
], dtype=float)


def estimate_duration_seconds(num_steps, ctrl_dt):
    return float(num_steps) * float(ctrl_dt)


def autoscale_vmax_to_target(Wu, ctrl_dt, vmax_base, target_time_s, iters=2):
    """
    Heuristic auto-scaling:
      1) Retimes using current vmax
      2) Measures resulting duration ~= steps * ctrl_dt
      3) Scales vmax by factor (duration / target) to move duration toward target

    Reasoning:
      Duration is approximately proportional to 1/vmax, so:
        if current_duration < target_duration => we need to slow down => reduce vmax
        scale factor k = target / current
        new_vmax = old_vmax / k = old_vmax * (current / target)

    Returns:
      traj, vmax_used, duration_s
    """
    vmax = vmax_base.copy()

    traj = None
    duration_s = None

    for k in range(int(iters)):
        print(f"\n=== AUTO-SCALE PASS {k+1}/{iters} ===")
        print(f"Using V_MAX = {vmax} rad/s")

        traj = retime_waypoints(Wu, ctrl_dt=ctrl_dt, vmax=vmax, use_smoothstep=True)
        duration_s = estimate_duration_seconds(traj.shape[0], ctrl_dt)

        print(f"[auto-scale] retimed steps = {traj.shape[0]}")
        print(f"[auto-scale] duration     = {duration_s:.3f} s (target={target_time_s:.3f} s)")

        if duration_s <= 1e-6:
            print("[auto-scale] duration too small to scale; leaving V_MAX unchanged.")
            break

        err = abs(duration_s - target_time_s)
        if err < 0.5:
            print(f"[auto-scale] within 0.5s of target (err={err:.3f}s). Done.")
            break

        ratio = duration_s / target_time_s
        ratio = float(np.clip(ratio, 0.05, 20.0))
        vmax = vmax * ratio

        print(f"[auto-scale] scaling factor (duration/target) = {ratio:.4f}")
        print(f"[auto-scale] updated V_MAX = {vmax} rad/s")

    return traj, vmax, duration_s


def apply_real_offsets(traj_mjc, offsets):
    """
    Convert MuJoCo-frame joint angles -> Real-robot joint angles.
      real = mjc + offsets
    """
    if traj_mjc.ndim != 2 or traj_mjc.shape[1] != 6:
        raise ValueError(f"apply_real_offsets expected (N,6), got {traj_mjc.shape}")
    return traj_mjc + offsets[None, :]


def main():
    print(f"Loading: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Stage 1 manual (MuJoCo frame)
    W1 = np.asarray(MANUAL_WAYPOINTS_STAGE_1, dtype=float)
    if W1.ndim != 2 or W1.shape[1] != 6:
        raise ValueError(f"Expected MANUAL_WAYPOINTS_STAGE_1 shape (K,6), got {W1.shape}")

    print("\n=== INPUT WAYPOINTS (RAW, MuJoCo frame) ===")
    print(f"Waypoints: {W1.shape[0]}")
    print(f"q_start = {W1[0]}")
    print(f"q_end   = {W1[-1]}")
    print("All waypoints:")
    for i in range(W1.shape[0]):
        print(f"  W1[{i}] = {W1[i]}")

    # Unwrap to avoid discontinuities across pi boundaries (still MuJoCo frame)
    print("\n=== UNWRAP WAYPOINTS (MuJoCo frame) ===")
    W1u = unwrap_waypoints(W1, discont=np.pi)
    print(f"Unwrapped waypoints: {W1u.shape[0]}")
    print(f"q_start (unwrapped) = {W1u[0]}")
    print(f"q_end   (unwrapped) = {W1u[-1]}")
    print("All unwrapped waypoints:")
    for i in range(W1u.shape[0]):
        print(f"  W1u[{i}] = {W1u[i]}")

    # Retiming with auto-scaling to hit ~TARGET_TOTAL_TIME (MuJoCo frame)
    print("\n=== RETIMING (MuJoCo frame, VELOCITY-LIMITED, AUTO-SCALED FOR SLOW TEST) ===")
    print(f"CTRL_DT           = {CTRL_DT:.6f} s (CTRL_HZ={CTRL_HZ:.1f})")
    print(f"TARGET_TOTAL_TIME = {TARGET_TOTAL_TIME:.3f} s")
    print(f"V_MAX_BASE        = {V_MAX_BASE} rad/s")

    traj_mjc, vmax_used, duration_s = autoscale_vmax_to_target(
        W1u,
        ctrl_dt=CTRL_DT,
        vmax_base=V_MAX_BASE,
        target_time_s=TARGET_TOTAL_TIME,
        iters=AUTO_SCALE_ITERS,
    )

    print("\n=== TRAJECTORY SUMMARY (MuJoCo frame) ===")
    print(f"traj steps = {traj_mjc.shape[0]}")
    print(f"duration   = {duration_s:.3f} s  (target={TARGET_TOTAL_TIME:.3f} s)")
    print(f"V_MAX used = {vmax_used}")
    print(f"q_start = {traj_mjc[0]}")
    print(f"q_end   = {traj_mjc[-1]}")

    # Convert to REAL robot joint coordinates
    print("\n=== REAL JOINT FRAME CONVERSION ===")
    print("Offsets applied as: real = mjc + offsets")
    print(f"Offsets (rad) = {REAL_JOINT_OFFSETS}")
    print("  shoulder_pan   offset = 0")
    print("  shoulder_lift  offset = -pi/2")
    print("  elbow          offset = 0")
    print("  wrist_1        offset = -pi/2")
    print("  wrist_2        offset = 0")
    print("  wrist_3        offset = 0")

    traj_real = apply_real_offsets(traj_mjc, REAL_JOINT_OFFSETS)

    print("\n=== TRAJECTORY SUMMARY (REAL joint frame) ===")
    print(f"q_start_real = {traj_real[0]}")
    print(f"q_end_real   = {traj_real[-1]}")
    print("Sanity check (real - mjc) on first row:")
    print(f"  (traj_real[0] - traj_mjc[0]) = {traj_real[0] - traj_mjc[0]}")

    # Gripper column (constant open) + time
    grip = np.full((traj_mjc.shape[0],), GRIP_OPEN, dtype=float)

    # Build BOTH plans (same gripper, same time), just different joint columns
    # Plan format: [q1..q6, gripper_ctrl, t]
    header = ARM_JOINT_NAMES + ["gripper_ctrl"]

    plan_mjc = np.hstack([traj_mjc, grip[:, None]])
    plan_real = np.hstack([traj_real, grip[:, None]])

    # Append time column
    plan_mjc, _t = append_time_column(plan_mjc, CTRL_DT, start_time=0.0)
    plan_real, _t2 = append_time_column(plan_real, CTRL_DT, start_time=0.0)
    header = header + ["t"]

    print("\n=== COMPILED PLAN SUMMARY (MJC FRAME) ===")
    print(f"plan rows = {plan_mjc.shape[0]}")
    print(f"plan cols = {plan_mjc.shape[1]}  (6 joints + gripper_ctrl + t)")
    mid = plan_mjc.shape[0] // 2
    print("Sample rows (first, mid, last):")
    print(f"  row[0]   = {plan_mjc[0]}")
    print(f"  row[mid] = {plan_mjc[mid]}")
    print(f"  row[-1]  = {plan_mjc[-1]}")

    print("\n=== COMPILED PLAN SUMMARY (REAL FRAME) ===")
    print(f"plan rows = {plan_real.shape[0]}")
    print(f"plan cols = {plan_real.shape[1]}  (6 joints + gripper_ctrl + t)")
    mid2 = plan_real.shape[0] // 2
    print("Sample rows (first, mid, last):")
    print(f"  row[0]   = {plan_real[0]}")
    print(f"  row[mid] = {plan_real[mid2]}")
    print(f"  row[-1]  = {plan_real[-1]}")

    # Save timestamped + latest for BOTH
    out_ts = int(time.time())

    out_real_ts = os.path.join(CURRENT_DIR, f"path_real_{out_ts}.csv")
    out_mjc_ts  = os.path.join(CURRENT_DIR, f"path_real_mjcframe_{out_ts}.csv")

    write_plan_csv(out_real_ts, plan_real, header)
    write_plan_csv(OUT_LATEST_REAL, plan_real, header)

    write_plan_csv(out_mjc_ts, plan_mjc, header)
    write_plan_csv(OUT_LATEST_MJC, plan_mjc, header)

    print(f"\n[RESULT] Saved REAL-frame plan CSV: {out_real_ts}")
    print(f"[RESULT] Also wrote REAL latest: {OUT_LATEST_REAL}")
    print(f"[RESULT] Saved MJC-frame plan CSV: {out_mjc_ts}")
    print(f"[RESULT] Also wrote MJC latest: {OUT_LATEST_MJC}")
    print(f"[RESULT] CTRL_HZ={CTRL_HZ:.1f} => approx duration = {plan_mjc.shape[0] * CTRL_DT:.2f} s")


if __name__ == "__main__":
    main()