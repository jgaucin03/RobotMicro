#!/usr/bin/env python3
# motion/planner_numerical_k.py
#
# Multi-stage numerical planner producing a single CSV with:
#   [6 arm joints + gripper_ctrl] per row.
#
# Stage 1: plan to CUP_GRIP_SITE_NAME_BACK using manual waypoints + optional IK refine
# Stage 2: plan to CUP_GRIP_SITE_NAME_FRONT using manual waypoints + optional IK refine
#
# Features:
# - Piecewise planner uses straight-line edges if collision-free; else RRT patches
# - Wrapped-joint handling to avoid 2pi “whip” moves (shoulder_pan wrap)
# - Optional local IK refinement at end of each stage to hit a site (position-only)
# - Compiles one combined CSV:
#     * initial gripper open ramp to 0.7 (arm held at q_start)
#     * stage trajectories with gripper held open
#     * final gripper close ramp to 0.0 (arm held at final pose)
#
# NOTE: The gripper ctrl here is intended to drive your actuator "gripper_servo"
#       in playback (data.ctrl[gripper_act_id] = gripper_ctrl).

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import mujoco
import numpy as np

from RobotMicro.motion.planlibk import (
    get_site_id,
    get_joint_ids_and_ranges,
    get_qpos_indices,
    world_pos_of_site,
    get_forbidden_pairs,
    is_state_collision_free,
    solve_ik_multi_seed,
    unwrap_target_near,
    plan_piecewise_waypoints,
    compile_with_gripper_ctrl,
    save_plan_csv,
)

np.set_printoptions(precision=5, suppress=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH    = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

EE_SITE_NAME = "ee_site"

# ---------------------------------------------------------------------------
# Arm joints (must match CSV order & model joint names)
# ---------------------------------------------------------------------------
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# Gripper joints (qpos indices only used for collision-check posture)
GRIPPER_JOINT_NAMES = [
    "link6_Grip-Driver",
    "link6_Grip-Link",
]

# Collision-check posture for gripper (OPEN)
GRIPPER_OPEN_QPOS = np.array([0.7, -0.7], dtype=float)

# ---------------------------------------------------------------------------
# Plan stages (sites)
# ---------------------------------------------------------------------------
PLAN_STAGE_1 = "cup_grip_target_site_back"
PLAN_STAGE_2 = "cup_grip_target_site_front"

# ---------------------------------------------------------------------------
# Manual waypoint lists
# NOTE: waypoint[0] is allowed to be redundant; we override it with q_start anyway.
# ---------------------------------------------------------------------------
MANUAL_WAYPOINTS_STAGE_1 = [
    np.array([0.0,  0.0,   0.0,   0.0,   0.0,   0.0], dtype=float),
    np.array([2.7,  0.0,   0.0,   0.0,   0.0,   0.0], dtype=float),
    np.array([2.7,  0.0,   0.0,  -1.7,  -1.7,  -1.7], dtype=float),
    np.array([2.7,  0.0,   1.54, -1.7,  -1.7,  -1.7], dtype=float),
    np.array([2.5, -1.92,  1.54, -1.7,  -1.7,  -1.7], dtype=float),
    np.array([2.7, -1.92,  1.54, -1.73, -1.48, -1.79], dtype=float),
    np.array([2.7, -1.92,  1.54, -1.73, -1.48, -1.79], dtype=float),
]

MANUAL_WAYPOINTS_STAGE_2 = [
    np.array([2.7,  -1.92,  1.54, -1.73, -1.48, -1.79], dtype=float),
    np.array([3.17, -1.85,  1.23, -1.79, -1.57, -1.82], dtype=float),
]

# ---------------------------------------------------------------------------
# Wrapped-joint config:
# We treat shoulder_pan as wrap-enabled so 3.17 can become ~-3.113 (near pi wrap)
# ---------------------------------------------------------------------------
WRAP_JOINTS = {"shoulder_pan"}  # easy to extend later


# ---------------------------------------------------------------------------
# Gripper ctrl compilation
# ---------------------------------------------------------------------------
GRIPPER_CTRL_START   = 0.0   # assume initial ctrl at start of plan
GRIPPER_CTRL_OPEN    = 0.7
GRIPPER_CTRL_CLOSED  = 0.0

OPEN_RAMP_STEPS  = 120   # initial open ramp duration
CLOSE_RAMP_STEPS = 120   # final close ramp duration

# ---------------------------------------------------------------------------
# Piecewise planning
# ---------------------------------------------------------------------------
STRAIGHTLINE_POINTS = 60

# Optional IK refine at end of each stage to hit that stage’s site
ENABLE_STAGE_END_IK_REFINE = True

# Add seeds around last manual waypoint for IK refine
IK_PERTURB_NUM   = 4
IK_PERTURB_SIGMA = 0.25


@dataclass
class PlanStage:
    name: str
    site_name: str
    manual_waypoints: List[np.ndarray]


def build_wrap_mask() -> np.ndarray:
    return np.array([jn in WRAP_JOINTS for jn in ARM_JOINT_NAMES], dtype=bool)


def refine_goal_with_local_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_sid: int,
    arm_q_idx: np.ndarray,
    arm_dof_idx: np.ndarray,
    joint_ranges: np.ndarray,
    target_world: np.ndarray,
    q_seed: np.ndarray,
) -> np.ndarray:
    """
    Run multi-seed local IK around q_seed and return best solution.
    """
    seeds = []

    seeds.append(q_seed.copy())
    seeds.append(np.zeros_like(q_seed))

    # Jitter around seed
    rng = np.random.default_rng(0)
    for _ in range(IK_PERTURB_NUM):
        noise = rng.normal(loc=0.0, scale=IK_PERTURB_SIGMA, size=q_seed.shape)
        q = q_seed + noise
        q = np.clip(q, joint_ranges[:, 0], joint_ranges[:, 1])
        seeds.append(q)

    q_best, err_best, steps_best = solve_ik_multi_seed(
        model=model,
        data=data,
        ee_sid=ee_sid,
        arm_q_idx=arm_q_idx,
        arm_dof_idx=arm_dof_idx,
        seeds=seeds,
        target_world=target_world,
        verbose=True,
    )

    print("\n=== LOCAL IK BEST ===")
    print(f"  best_err  = {err_best:.6f}")
    print(f"  best_q    = {q_best}")
    print(f"  best_steps= {steps_best}")
    return q_best


def main():
    print(f"Loading: {XML_PATH}")
    if not os.path.exists(XML_PATH):
        print("XML file not found.")
        return

    # For reproducible sampling
    np.random.seed(0)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # IDs
    ee_sid   = get_site_id(model, EE_SITE_NAME)
    sid_s1   = get_site_id(model, PLAN_STAGE_1)
    sid_s2   = get_site_id(model, PLAN_STAGE_2)

    # Indices + ranges
    j_ids, arm_q_idx, arm_dof_idx, joint_ranges = get_joint_ids_and_ranges(model, ARM_JOINT_NAMES)
    grip_q_idx = get_qpos_indices(model, GRIPPER_JOINT_NAMES)

    wrap_mask = build_wrap_mask()
    print("\n=== WRAP MASK ===")
    for jn, w in zip(ARM_JOINT_NAMES, wrap_mask):
        print(f"  {jn:>14}: wrap={bool(w)}")

    # Start config: arm zeros
    q_start = np.zeros(len(ARM_JOINT_NAMES), dtype=float)

    # Collision setup
    forbidden_pairs, ignore_gid = get_forbidden_pairs(model)

    # Check start collision-free with planning gripper posture
    print("\nChecking collision at start (with planning gripper posture)...")
    cs_start = is_state_collision_free(
        model, data, q_start, arm_q_idx,
        forbidden_pairs, ignore_gid,
        grip_q_idx=grip_q_idx,
        grip_planning_qpos=GRIPPER_OPEN_QPOS,
        verbose=True,
    )
    print(f"  Start collision-free? {cs_start}")
    if not cs_start:
        print("Start is in collision; fix initial pose/scene first.")
        return

    # Build stages
    stages = [
        PlanStage(name="stage_1_back",  site_name=PLAN_STAGE_1, manual_waypoints=MANUAL_WAYPOINTS_STAGE_1),
        PlanStage(name="stage_2_front", site_name=PLAN_STAGE_2, manual_waypoints=MANUAL_WAYPOINTS_STAGE_2),
    ]

    # Keep a running arm trajectory (dense)
    arm_traj_all: List[np.ndarray] = []

    # Stage 1 target in world
    target_s1 = world_pos_of_site(model, data, sid_s1)
    target_s2 = world_pos_of_site(model, data, sid_s2)
    print("\n=== TARGET WORLD POSITIONS ===")
    print(f"  {PLAN_STAGE_1}: {target_s1}")
    print(f"  {PLAN_STAGE_2}: {target_s2}")

    # Seed for piecewise (stage start pose)
    q_stage_start = q_start.copy()

    for si, stg in enumerate(stages):
        print("\n" + "#" * 70)
        print(f"### PLANNING {stg.name} -> site '{stg.site_name}' ###")
        print("#" * 70)

        # Copy manual waypoints, override first with actual stage start
        wps = [wp.copy() for wp in stg.manual_waypoints]
        wps[0] = q_stage_start.copy()

        # If this is stage 2+, unwrap any "far" manual waypoint values near current start
        # (particularly shoulder_pan) so interpolation doesn't go the long way.
        for i in range(1, len(wps)):
            wps[i] = unwrap_target_near(wps[i-1], wps[i], wrap_mask)

        print("\nManual waypoints (after unwrap + start override):")
        for i, wp in enumerate(wps):
            print(f"  wp[{i}] = {wp}")

        # Piecewise planning between manual waypoints
        dense = plan_piecewise_waypoints(
            model=model,
            data=data,
            arm_q_idx=arm_q_idx,
            joint_ranges=joint_ranges,
            forbidden_pairs=forbidden_pairs,
            ignore_gid=ignore_gid,
            wrap_mask=wrap_mask,
            waypoints=wps,
            grip_q_idx=grip_q_idx,
            grip_planning_qpos=GRIPPER_OPEN_QPOS,
            rng=np.random.default_rng(0),
            points_per_straight_segment=STRAIGHTLINE_POINTS,
        )

        if dense is None:
            print(f"\n[RESULT] Failed to plan {stg.name}.")
            return

        print(f"\n[{stg.name}] Dense segment length: {len(dense)}")

        # Optional IK refine at stage end to hit the site
        if ENABLE_STAGE_END_IK_REFINE:
            target_world = target_s1 if stg.site_name == PLAN_STAGE_1 else target_s2
            q_seed = dense[-1].copy()

            print("\nChecking collision at IK-refined goal (with planning gripper posture)...")
            # IK can move; we refine first, then collision-check after
            q_ik = refine_goal_with_local_ik(
                model=model,
                data=data,
                ee_sid=ee_sid,
                arm_q_idx=arm_q_idx,
                arm_dof_idx=arm_dof_idx,
                joint_ranges=joint_ranges,
                target_world=target_world,
                q_seed=q_seed,
            )

            # Collision check IK result
            ok = is_state_collision_free(
                model, data, q_ik, arm_q_idx,
                forbidden_pairs, ignore_gid,
                grip_q_idx=grip_q_idx,
                grip_planning_qpos=GRIPPER_OPEN_QPOS,
                verbose=True,
            )
            print(f"  IK goal collision-free? {ok}")
            if ok:
                print(f"\n[OK] Using IK-refined final goal for {stg.name}.")
                dense.append(q_ik.copy())
            else:
                print(f"\n[WARN] IK-refined goal was in collision. Keeping original final waypoint.")

        # Concatenate to global trajectory, avoid duplicate join if same pose
        if not arm_traj_all:
            arm_traj_all.extend(dense)
        else:
            # If last equals first (within small tolerance), skip the first of this segment
            if np.linalg.norm(arm_traj_all[-1] - dense[0]) < 1e-9:
                arm_traj_all.extend(dense[1:])
            else:
                # “Natural resolve” between stages: insert a short interpolation bridge
                print("\n[INFO] Stage boundary mismatch; inserting a short bridge segment.")
                bridge = [arm_traj_all[-1].copy(), dense[0].copy()]
                bridge_dense = []
                for s in np.linspace(0.0, 1.0, 40, endpoint=False):
                    bridge_dense.append(bridge[0] + s * (bridge[1] - bridge[0]))
                bridge_dense.append(bridge[1])
                arm_traj_all.extend(bridge_dense[1:])
                arm_traj_all.extend(dense[1:])

        # Update stage start for next stage
        q_stage_start = arm_traj_all[-1].copy()

    # Final summary before compilation
    print("\n=== FINAL ARM TRAJECTORY SUMMARY ===")
    print(f"  total_steps (arm only) = {len(arm_traj_all)}")
    print(f"  q_start                = {arm_traj_all[0]}")
    print(f"  q_end                  = {arm_traj_all[-1]}")

    # Compile with gripper ctrl:
    # - Open ramp at beginning (arm held at q_start)
    # - Arm trajectory with gripper held open
    # - Close ramp at end (arm held at q_end)
    plan = compile_with_gripper_ctrl(
        arm_traj=arm_traj_all,
        gripper_ctrl_for_arm=GRIPPER_CTRL_START,
        open_ramp_steps=OPEN_RAMP_STEPS,
        close_ramp_steps=CLOSE_RAMP_STEPS,
        gripper_open=GRIPPER_CTRL_OPEN,
        gripper_closed=GRIPPER_CTRL_CLOSED,
        start_hold_arm=arm_traj_all[0].copy(),
        end_hold_arm=arm_traj_all[-1].copy(),
    )

    print("\n=== COMPILED PLAN SUMMARY (ARM+GRIPPER_CTRL) ===")
    print(f"  plan rows = {plan.shape[0]}")
    print(f"  plan cols = {plan.shape[1]}  (6 joints + gripper_ctrl)")
    print("\nSample rows (first, mid, last):")
    print("  row[0]   =", plan[0])
    print("  row[mid] =", plan[plan.shape[0] // 2])
    print("  row[-1]  =", plan[-1])

    # Save
    ts = int(time.time())
    csv_name = f"plan_stage12_{ts}.csv"
    csv_path = os.path.join(CURRENT_DIR, csv_name)
    save_plan_csv(csv_path, plan, ARM_JOINT_NAMES, gripper_col_name="gripper_ctrl")

    # Optional convenience copy name
    latest_path = os.path.join(CURRENT_DIR, "path_latest_with_gripper.csv")
    try:
        # overwrite if exists
        if os.path.exists(latest_path):
            os.remove(latest_path)
        # simple copy
        import shutil
        shutil.copyfile(csv_path, latest_path)
        print(f"[RESULT] Also wrote: {latest_path}")
    except Exception as e:
        print(f"[WARN] Could not write {latest_path}: {e}")


if __name__ == "__main__":
    main()