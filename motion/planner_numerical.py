#!/usr/bin/env python3
# motion/planner_numerical.py
#
# Robot-real planner:
# - Builds multi-stage waypoint lists (Stage 1, Stage 2, Stage 3...).
# - Optionally IK-refines the final waypoint of each stage to a site position.
# - Unwraps joint angles to avoid 3.14 <-> -3.13 discontinuities.
# - Retimes with velocity limits at a fixed control rate (CTRL_DT).
# - Outputs CSV with 8 columns: 6 arm joints + gripper_ctrl + t.

import os
import time
import numpy as np
import mujoco

from planlib import (
    mj_id,
    unwrap_waypoints,
    retime_waypoints,
    ik_to_site_position,
    collision_summary,
    write_plan_csv,
    append_time_column,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

EE_SITE_NAME = "ee_site"

# -------------------------
# Stage end-site IK targets
# -------------------------
# Stages (sites are optional; if None, we only use manual waypoint end)
PLAN_STAGE_1_SITE = "cup_grip_target_site_back"
PLAN_STAGE_2_SITE = "cup_grip_target_site_front"

# If you want, you can IK-refine stage 3 end to microwave target,
# but since you're hand-authoring the in-microwave motion, leave it off by default.
PLAN_STAGE_3_SITE = None  # e.g. "cup_target_site"
PLAN_STAGE_4_SITE = None
PLAN_STAGE_5_SITE = None

# -------------------------
# Manual waypoints
# -------------------------
MANUAL_WAYPOINTS_STAGE_1 = [
    np.array([0.0,  0.0,   0.0,   0.0,   0.0,   0.0], dtype=float),
    np.array([2.7,  0.0,   0.0,   0.0,   0.0,   0.0], dtype=float),
    np.array([2.7,  0.0,   0.0,  -1.7,  -1.7,  -1.7], dtype=float),
    np.array([2.7,  0.0,   1.54, -1.7,  -1.7,  -1.7], dtype=float),
    np.array([2.5, -1.92,  1.54, -1.7,  -1.7,  -1.7], dtype=float),
    np.array([2.7, -1.92,  1.54, -1.73, -1.48, -1.79], dtype=float),
]

MANUAL_WAYPOINTS_STAGE_2 = [
    np.array([2.7,  -1.92,  1.54, -1.73, -1.48, -1.79], dtype=float),
    np.array([3.17, -1.85,  1.23, -1.79, -1.57, -1.82], dtype=float),
]

MANUAL_WAYPOINTS_STAGE_3 = [
    # put the cup into the microwave
    # Start (this will be overwritten to exactly stage2 end after IK refine)
    np.array([3.17, -1.85, 1.23, -1.79, -1.57, -1.82], dtype=float),

    # You wrote "shoulder_lift: -1.85 -1.79" in the first line; treat that as an intermediate step
    np.array([3.17, -1.79, 1.23, -1.79, -1.57, -1.82], dtype=float),

    np.array([3.17, -1.48, 1.23, -1.79, -1.57, -1.82], dtype=float),

    np.array([-1.57, -1.48, 1.23, -1.79, -1.57, -1.82], dtype=float),
    np.array([-1.57, -1.48, 1.26, -1.79, -1.57, -1.82], dtype=float),
    np.array([-1.57, -1.48, 0.88, -1.79, -1.57, -1.82], dtype=float),

    np.array([-1.45, -1.48, 0.88, -1.79, -1.57, -1.82], dtype=float),
    np.array([-1.45, -1.48, 0.503, -1.79, -1.57, -1.82], dtype=float),

    np.array([-1.29, -1.48, 0.0628, -1.79, -1.57, -1.82], dtype=float),

    np.array([-1.26, -1.85, 0.0628, -0.88, -1.57, -1.82], dtype=float),
]

MANUAL_WAYPOINTS_STAGE_4 = [
    # stage open the gripper
    # stage out of the microwave: intermediate step
    np.array([-1.26, -1.85, 0.0628, -0.88, -1.57, -1.82], dtype=float),
    np.array([-1.26, -1.38, 0.0628, -0.88, -1.57, -1.82], dtype=float),
]

MANUAL_WAYPOINTS_STAGE_5 = [
    # close the gripper (tuck / retreat)
    np.array([-1.26, -1.38, 0.911, -0.88, -1.57, -1.82], dtype=float),
    np.array([-1.76, -1.38, 0.911, -0.88, -1.57, -1.82], dtype=float),
    np.array([-1.76, -1.38, 1.82,  -0.88, -1.57, -1.82], dtype=float),
    np.array([-1.76,  0.942, 1.82, -0.88, -1.57, -1.82], dtype=float),
]

# -------------------------
# Control discretization (robot-real)
# -------------------------
CTRL_HZ = 100.0
CTRL_DT = 1.0 / CTRL_HZ

# Velocity limits (rad/s) — set these to what feels “safe”
V_MAX = np.array([0.6, 0.6, 0.7, 1.2, 1.2, 1.5], dtype=float)

# -------------------------
# Gripper schedule
# -------------------------
GRIP_OPEN = 0.7
GRIP_CLOSED = 0.0

# Close at end of stage 2
GRIP_CLOSE_TIME = 1.0  # seconds to close

# Hold closed a bit before moving (helps settle contacts)
GRIP_SETTLE_TIME = 0.25  # seconds holding pose after closing before stage 3 move

# Open at end of stage 3 (release cup)
GRIP_OPEN_TIME = 0.75    # seconds to open

# Optional settle after opening before moving away
GRIP_OPEN_SETTLE_TIME = 0.15

# Optional "tuck fingers closed" after stage 4 before retreat motion stage 5
GRIP_TUCK_CLOSE_TIME = 0.50
GRIP_TUCK_SETTLE_TIME = 0.10


def site_pos(model, data, site_name):
    sid = mj_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[sid].copy()


def ik_refine_to_site(model, data, q_seed, target_site, label):
    mujoco.mj_forward(model, data)
    tgt = site_pos(model, data, target_site)
    print(f"\n[IK] Refining final waypoint to site '{target_site}' ({label})")
    q_best, err = ik_to_site_position(
        model, data,
        arm_joint_names=ARM_JOINT_NAMES,
        ee_site_name=EE_SITE_NAME,
        target_pos=tgt,
        q_seed=q_seed,
        max_steps=250,
        tol=1.0e-3,
        step_scale=0.8,
        damping=1e-2,
        verbose=True,
    )
    ok, ncon, worst, _ = collision_summary(model, data, pen_thresh=-1e-3)
    print(f"[IK] final_err={err:.6f} m, collision_free? {ok} (ncon={ncon}, worst_dist={worst:.6e})")
    return q_best, err


def _maybe_ik_refine_stage_end(model, data, Wu, site_name, label):
    if site_name is None:
        return Wu, None
    q_seed = Wu[-1].copy()
    q_ik, err = ik_refine_to_site(model, data, q_seed, site_name, label)
    Wu[-1] = q_ik
    return Wu, err


def main():
    print(f"Loading: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Stage waypoints
    W1 = np.asarray(MANUAL_WAYPOINTS_STAGE_1, dtype=float)
    W2 = np.asarray(MANUAL_WAYPOINTS_STAGE_2, dtype=float)
    W3 = np.asarray(MANUAL_WAYPOINTS_STAGE_3, dtype=float)
    W4 = np.asarray(MANUAL_WAYPOINTS_STAGE_4, dtype=float)
    W5 = np.asarray(MANUAL_WAYPOINTS_STAGE_5, dtype=float)

    # --- Build a single waypoint stack to unwrap across ALL stage boundaries ---
    # stitch as: W1, W2[1:], W3[1:], W4[1:], W5[1:]
    W_all = np.vstack([W1, W2[1:], W3[1:], W4[1:], W5[1:]])
    W_all = unwrap_waypoints(W_all, discont=np.pi)

    # Split back into stages (unwrapped)
    n1 = W1.shape[0]
    n2 = W2.shape[0]
    n3 = W3.shape[0]
    n4 = W4.shape[0]
    n5 = W5.shape[0]

    # indices inside W_all:
    # W1: n1
    # W2[1:]: (n2-1)
    # W3[1:]: (n3-1)
    # W4[1:]: (n4-1)
    # W5[1:]: (n5-1)
    idx = 0
    W1u = W_all[idx:idx + n1]
    idx += n1

    W2_tail = W_all[idx:idx + (n2 - 1)]
    idx += (n2 - 1)
    W2u = np.vstack([W1u[-1], W2_tail])

    W3_tail = W_all[idx:idx + (n3 - 1)]
    idx += (n3 - 1)
    W3u = np.vstack([W2u[-1], W3_tail])

    W4_tail = W_all[idx:idx + (n4 - 1)]
    idx += (n4 - 1)
    W4u = np.vstack([W3u[-1], W4_tail])

    W5_tail = W_all[idx:idx + (n5 - 1)]
    idx += (n5 - 1)
    W5u = np.vstack([W4u[-1], W5_tail])

    # Optional IK-refine stage ends to sites (mostly for initial grasp targets)
    W1u, _ = _maybe_ik_refine_stage_end(model, data, W1u, PLAN_STAGE_1_SITE, "stage_1_back")
    W2u, _ = _maybe_ik_refine_stage_end(model, data, W2u, PLAN_STAGE_2_SITE, "stage_2_front")

    # IMPORTANT: downstream stages must start from IK-refined predecessors
    W3u[0] = W2u[-1].copy()
    W4u[0] = W3u[-1].copy()
    W5u[0] = W4u[-1].copy()

    # Optional IK refine for later stages (OFF by default)
    W3u, _ = _maybe_ik_refine_stage_end(model, data, W3u, PLAN_STAGE_3_SITE, "stage_3_end")
    W4u, _ = _maybe_ik_refine_stage_end(model, data, W4u, PLAN_STAGE_4_SITE, "stage_4_end")
    W5u, _ = _maybe_ik_refine_stage_end(model, data, W5u, PLAN_STAGE_5_SITE, "stage_5_end")

    # Re-enforce boundaries after any optional IK
    W4u[0] = W3u[-1].copy()
    W5u[0] = W4u[-1].copy()

    print("\n=== WAYPOINT SUMMARY (UNWRAPPED + (OPTIONAL) IK-REFINED) ===")
    print(f"Stage1 waypoints: {W1u.shape[0]}")
    print(f"Stage2 waypoints: {W2u.shape[0]}")
    print(f"Stage3 waypoints: {W3u.shape[0]}")
    print(f"Stage4 waypoints: {W4u.shape[0]}")
    print(f"Stage5 waypoints: {W5u.shape[0]}")
    print(f"Stage1 end q: {W1u[-1]}")
    print(f"Stage2 end q: {W2u[-1]}")
    print(f"Stage3 end q: {W3u[-1]}")
    print(f"Stage4 end q: {W4u[-1]}")
    print(f"Stage5 end q: {W5u[-1]}")

    # Retiming
    print("\n=== RETIMING (VELOCITY-LIMITED) ===")
    print(f"CTRL_DT = {CTRL_DT:.6f} s (CTRL_HZ={CTRL_HZ:.1f})")
    print(f"V_MAX   = {V_MAX} rad/s")

    traj1 = retime_waypoints(W1u, ctrl_dt=CTRL_DT, vmax=V_MAX, use_smoothstep=True)
    traj2 = retime_waypoints(W2u, ctrl_dt=CTRL_DT, vmax=V_MAX, use_smoothstep=True)
    traj3 = retime_waypoints(W3u, ctrl_dt=CTRL_DT, vmax=V_MAX, use_smoothstep=True)
    traj4 = retime_waypoints(W4u, ctrl_dt=CTRL_DT, vmax=V_MAX, use_smoothstep=True)
    traj5 = retime_waypoints(W5u, ctrl_dt=CTRL_DT, vmax=V_MAX, use_smoothstep=True)

    print(f"Stage1 steps: {traj1.shape[0]}")
    print(f"Stage2 steps: {traj2.shape[0]}")
    print(f"Stage3 steps: {traj3.shape[0]}")
    print(f"Stage4 steps: {traj4.shape[0]}")
    print(f"Stage5 steps: {traj5.shape[0]}")

    # ---------------------------------------------------------
    # Stitch arm trajectories (avoid duplicates at boundaries)
    # ---------------------------------------------------------
    arm12 = np.vstack([traj1, traj2[1:]])                 # stage1 + stage2 motion
    arm3 = traj3[1:]                                      # stage3 motion
    arm4 = traj4[1:]                                      # stage4 motion
    arm5 = traj5[1:]                                      # stage5 motion

    print(f"Total arm steps (pre-gripper-holds): {(arm12.shape[0] + arm3.shape[0] + arm4.shape[0] + arm5.shape[0])}")
    print(f"q_start = {arm12[0]}")
    print(f"q_end (stage5 end) = {traj5[-1]}")

    # ---------------------------------------------------------
    # Gripper ctrl trajectory (piecewise)
    # ---------------------------------------------------------
    # Plan segments:
    #   A) arm12 motion: grip OPEN
    #   B) close ramp at end of stage2 (hold pose): OPEN->CLOSED
    #   C) settle closed (hold pose)
    #   D) arm3 motion: grip CLOSED
    #   E) open ramp at end of stage3 (hold pose): CLOSED->OPEN
    #   F) settle open (hold pose)
    #   G) arm4 motion: grip OPEN
    #   H) tuck close ramp at end of stage4 (hold pose): OPEN->CLOSED
    #   I) tuck settle closed (hold pose)
    #   J) arm5 motion: grip CLOSED

    # A) stage1+2 motion
    grip12 = np.full((arm12.shape[0],), GRIP_OPEN, dtype=float)

    # B) close ramp (hold pose at stage2 end)
    close_steps = int(np.ceil(GRIP_CLOSE_TIME / CTRL_DT))
    close_steps = max(1, close_steps)
    close_ramp = np.linspace(GRIP_OPEN, GRIP_CLOSED, close_steps, endpoint=True)
    arm_close = np.repeat(arm12[-1][None, :], close_steps, axis=0)

    # C) settle closed
    settle_steps = int(np.ceil(GRIP_SETTLE_TIME / CTRL_DT))
    settle_steps = max(0, settle_steps)
    arm_settle_closed = np.repeat(arm12[-1][None, :], settle_steps, axis=0) if settle_steps > 0 else np.zeros((0, 6))
    grip_settle_closed = np.full((settle_steps,), GRIP_CLOSED, dtype=float) if settle_steps > 0 else np.zeros((0,))

    # D) stage3 motion (closed)
    grip3 = np.full((arm3.shape[0],), GRIP_CLOSED, dtype=float)

    # E) open ramp (hold pose at stage3 end)
    open_steps = int(np.ceil(GRIP_OPEN_TIME / CTRL_DT))
    open_steps = max(1, open_steps)
    open_ramp = np.linspace(GRIP_CLOSED, GRIP_OPEN, open_steps, endpoint=True)
    arm_open = np.repeat(traj3[-1][None, :], open_steps, axis=0)

    # F) settle open
    open_settle_steps = int(np.ceil(GRIP_OPEN_SETTLE_TIME / CTRL_DT))
    open_settle_steps = max(0, open_settle_steps)
    arm_settle_open = np.repeat(traj3[-1][None, :], open_settle_steps, axis=0) if open_settle_steps > 0 else np.zeros((0, 6))
    grip_settle_open = np.full((open_settle_steps,), GRIP_OPEN, dtype=float) if open_settle_steps > 0 else np.zeros((0,))

    # G) stage4 motion (open)
    grip4 = np.full((arm4.shape[0],), GRIP_OPEN, dtype=float)

    # H) tuck close ramp (hold pose at stage4 end)
    tuck_steps = int(np.ceil(GRIP_TUCK_CLOSE_TIME / CTRL_DT))
    tuck_steps = max(1, tuck_steps)
    tuck_ramp = np.linspace(GRIP_OPEN, GRIP_CLOSED, tuck_steps, endpoint=True)
    arm_tuck = np.repeat(traj4[-1][None, :], tuck_steps, axis=0)

    # I) tuck settle closed
    tuck_settle_steps = int(np.ceil(GRIP_TUCK_SETTLE_TIME / CTRL_DT))
    tuck_settle_steps = max(0, tuck_settle_steps)
    arm_tuck_settle = np.repeat(traj4[-1][None, :], tuck_settle_steps, axis=0) if tuck_settle_steps > 0 else np.zeros((0, 6))
    grip_tuck_settle = np.full((tuck_settle_steps,), GRIP_CLOSED, dtype=float) if tuck_settle_steps > 0 else np.zeros((0,))

    # J) stage5 motion (closed)
    grip5 = np.full((arm5.shape[0],), GRIP_CLOSED, dtype=float)

    # Full arm + grip
    arm_full = np.vstack([
        arm12,
        arm_close,
        arm_settle_closed,
        arm3,
        arm_open,
        arm_settle_open,
        arm4,
        arm_tuck,
        arm_tuck_settle,
        arm5,
    ])

    grip_full = np.concatenate([
        grip12,
        close_ramp,
        grip_settle_closed,
        grip3,
        open_ramp,
        grip_settle_open,
        grip4,
        tuck_ramp,
        grip_tuck_settle,
        grip5,
    ])

    assert arm_full.shape[0] == grip_full.shape[0], "arm/grip length mismatch"

    plan = np.hstack([arm_full, grip_full[:, None]])
    header = ARM_JOINT_NAMES + ["gripper_ctrl"]

    # --- append timestamp column (seconds) ---
    plan, _t = append_time_column(plan, CTRL_DT, start_time=0.0)
    header = header + ["t"]

    print("\n=== COMPILED PLAN SUMMARY (ARM+GRIPPER_CTRL+t) ===")
    print(f"plan rows = {plan.shape[0]}")
    print(f"plan cols = {plan.shape[1]}  (6 joints + gripper_ctrl + t)")
    mid = plan.shape[0] // 2
    print("Sample rows (first, mid, last):")
    print(f"  row[0]   = {plan[0]}")
    print(f"  row[mid] = {plan[mid]}")
    print(f"  row[-1]  = {plan[-1]}")

    out_ts = int(time.time())
    out_csv = os.path.join(CURRENT_DIR, f"plan_stage12345_robotreal_{out_ts}.csv")
    write_plan_csv(out_csv, plan, header)

    # Convenience copy
    latest_csv = os.path.join(CURRENT_DIR, "path_latest_with_gripper.csv")
    write_plan_csv(latest_csv, plan, header)

    print(f"\n[RESULT] Saved plan CSV: {out_csv}")
    print(f"[RESULT] Also wrote: {latest_csv}")
    print(f"[RESULT] CTRL_HZ={CTRL_HZ:.1f} => approx plan duration = {plan.shape[0] * CTRL_DT:.2f} s")


if __name__ == "__main__":
    main()