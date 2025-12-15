#!/usr/bin/env python3
# motion/planlib.py
#
# Shared planning library for UR3 + gripper tasks:
# - Joint index/range lookup
# - Collision checking with forbidden geom pairs
# - Wrapped-joint metrics + interpolation (prevents 2pi “whip” motion)
# - Unconstrained local IK (position-only) + multi-seed wrapper
# - Joint-space RRT
# - Piecewise planner: straight-line edge if possible else RRT patch
# - Compile a combined CSV with [6 arm joints + gripper_ctrl] per row
#
# NOTE: This library assumes the planner evaluates collisions with the gripper
#       held in a specified "planning posture" (typically OPEN qpos).
#       Execution can (and should) drive gripper via actuator ctrl.

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import mujoco
import numpy as np

np.set_printoptions(precision=5, suppress=True)


# ---------------------------------------------------------------------------
# Defaults (safe-ish; override from caller as needed)
# ---------------------------------------------------------------------------

DEEP_PEN_THRESH = -1e-3

IK_MAX_STEPS           = 600
IK_BASE_STEP_SIZE      = 0.5
IK_MAX_STEP_NORM       = 0.20
IK_ERR_TOL             = 1e-3
IK_STALL_IMPROVE_TOL   = 1e-8
IK_REG_WEIGHT          = 1e-4

RRT_MAX_ITERS          = 3000
RRT_STEP_SIZE          = 0.02
RRT_GOAL_THRESH        = 0.08
RRT_GOAL_BIAS          = 0.2
EDGE_COLL_STEPS        = 20

# How many interpolation points to create for “easy” straight-line segments
STRAIGHTLINE_POINTS = 60

# For reproducible RRT sampling (caller can reseed)
DEFAULT_SEED = 0

DEBUG_COLLISIONS_ON_FAIL = True


# ===========================================================================
# Basic model helpers
# ===========================================================================

def get_site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"Site '{name}' not found in model.")
    return sid


def get_joint_ids_and_ranges(model: mujoco.MjModel, joint_names: Sequence[str]):
    """
    For the given list of joint names, return:
        j_ids:       [n] joint ids
        q_indices:   [n] indices into qpos
        dof_indices: [n] indices into qvel (and Jacobian columns)
        ranges:      [n,2] joint limits (or [-2pi,2pi] if not set)
    """
    j_ids       = []
    q_indices   = []
    dof_indices = []
    ranges      = []

    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Joint '{jn}' not found in model.")
        j_ids.append(jid)
        q_indices.append(model.jnt_qposadr[jid])
        dof_indices.append(model.jnt_dofadr[jid])

        rng = model.jnt_range[jid].copy()
        if np.allclose(rng, 0.0):
            rng = np.array([-2.0 * np.pi, 2.0 * np.pi], dtype=float)
        ranges.append(rng)

    return (
        np.array(j_ids, dtype=int),
        np.array(q_indices, dtype=int),
        np.array(dof_indices, dtype=int),
        np.array(ranges, dtype=float),
    )


def get_qpos_indices(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    q_idx = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Joint '{jn}' not found in model.")
        q_idx.append(model.jnt_qposadr[jid])
    return np.asarray(q_idx, dtype=int)


def get_actuator_id(model: mujoco.MjModel, actuator_name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if aid < 0:
        raise ValueError(f"Actuator '{actuator_name}' not found in model.")
    return aid


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    if gid < 0:
        return f"<none:{gid}>"
    nm = model.geom(gid).name
    return nm if nm is not None else f"<geom:{gid}>"


def world_pos_of_site(model: mujoco.MjModel, data: mujoco.MjData, sid: int) -> np.ndarray:
    mujoco.mj_forward(model, data)
    return np.array(data.site_xpos[sid])


# ===========================================================================
# Wrapped joints: distance + interpolation utilities
# ===========================================================================

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    # maps to (-pi, pi]
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def diff_with_wrap(q_to: np.ndarray, q_from: np.ndarray, wrap_mask: np.ndarray) -> np.ndarray:
    """
    Compute per-joint difference (q_to - q_from), wrapping selected joints
    into [-pi, pi] for “shortest direction”.
    """
    d = q_to - q_from
    if wrap_mask is not None and np.any(wrap_mask):
        d_wrap = wrap_to_pi(d)
        d = np.where(wrap_mask, d_wrap, d)
    return d


def interp_with_wrap(q0: np.ndarray, q1: np.ndarray, s: float, wrap_mask: np.ndarray) -> np.ndarray:
    """
    Interpolate from q0 to q1 using wrapped difference for wrap joints.
    """
    d = diff_with_wrap(q1, q0, wrap_mask)
    return q0 + s * d


def dist_with_wrap(q1: np.ndarray, q2: np.ndarray, wrap_mask: np.ndarray) -> float:
    d = diff_with_wrap(q1, q2, wrap_mask)
    return float(np.linalg.norm(d))


def unwrap_target_near(q_ref: np.ndarray, q_target: np.ndarray, wrap_mask: np.ndarray) -> np.ndarray:
    """
    Adjust q_target by +/-2pi on wrapped joints so it is “near” q_ref.
    Useful when user specifies e.g. 3.17 but wants continuation past pi -> -pi.
    """
    out = q_target.copy()
    if wrap_mask is None or not np.any(wrap_mask):
        return out

    for i, w in enumerate(wrap_mask):
        if not w:
            continue
        # choose k in {-1,0,1} that minimizes distance to ref
        candidates = [out[i] - 2*np.pi, out[i], out[i] + 2*np.pi]
        best = min(candidates, key=lambda v: abs(v - q_ref[i]))
        out[i] = best
    return out


# ===========================================================================
# Collision setup + checks
# ===========================================================================

def get_forbidden_pairs(model: mujoco.MjModel):
    """
    Returns:
        forbidden_id_pairs: list of (geom1_id, geom2_id) forbidden contacts
        ignore_gid: set of geom ids ignored in checks (empty by default)
    Adjust geometry name lists to match your scene_debug.xml.
    """
    arm_geoms = [
        "base_link_geom",
        "link1_geom",
        "link2_geom",
        "link3_geom",
        "link4_geom",
        "link5_geom",
        "link6_geom",
    ]

    jaw_geoms_visual = [
        "Left_Grip_geom",
        "Right_Grip_geom",
    ]

    jaw_geoms_collision = [
        "Left_Grip_col",
        "Right_Grip_col",
    ]

    table_geoms = [
        "ur3_table",
        "micro_table",
    ]

    microwave_geoms = [
        "micro_Base_geom",
        "micro_door_geom",
    ]

    floor_geoms = ["floor"]

    cup_geom = "cup_col"

    forbidden_pairs_by_name = [
        # Self-collision approximations
        ("base_link_geom", "link1_geom"),
        ("link1_geom", "link3_geom"),
        ("link1_geom", "link4_geom"),
        ("link1_geom", "link5_geom"),
        ("link1_geom", "link6_geom"),
        ("link2_geom", "link4_geom"),
        ("link2_geom", "link5_geom"),
        ("link2_geom", "link6_geom"),
    ]

    # Robot vs environment
    arm_and_jaws = arm_geoms + jaw_geoms_visual + jaw_geoms_collision
    for obs in table_geoms + microwave_geoms + floor_geoms:
        for link in arm_and_jaws:
            forbidden_pairs_by_name.append((obs, link))

    # Cup vs non-gripper parts (allow jaw collision geoms to grip)
    cup_blockers = [
        "base_link_geom",
        "link1_geom",
        "link2_geom",
        "link3_geom",
        "link4_geom",
        "link5_geom",
    ] + jaw_geoms_visual

    for link in cup_blockers:
        forbidden_pairs_by_name.append((cup_geom, link))

    # Build name->gid
    name_to_gid: Dict[str, int] = {}
    for gid in range(model.ngeom):
        nm = model.geom(gid).name
        if nm:
            name_to_gid[nm] = gid

    forbidden_pairs: List[Tuple[int, int]] = []
    missing: List[Tuple[str, str]] = []

    for n1, n2 in forbidden_pairs_by_name:
        if n1 in name_to_gid and n2 in name_to_gid:
            forbidden_pairs.append((name_to_gid[n1], name_to_gid[n2]))
        else:
            missing.append((n1, n2))

    print("\n=== COLLISION SETUP ===")
    print(f"  Total geoms in model: {model.ngeom}")
    print(f"  Forbidden pairs (resolved): {len(forbidden_pairs)}")
    print(f"  Missing-name pairs (ignored): {len(missing)}")
    for (g1, g2) in forbidden_pairs[:20]:
        print(f"    forbid: {geom_name(model, g1)} <-> {geom_name(model, g2)}")
    if len(forbidden_pairs) > 20:
        print(f"    ... {len(forbidden_pairs) - 20} more")

    ignore_gid = set()
    return forbidden_pairs, ignore_gid


def is_state_collision_free(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_arm: np.ndarray,
    arm_q_idx: np.ndarray,
    forbidden_pairs: Sequence[Tuple[int, int]],
    ignore_gid: set,
    grip_q_idx: Optional[np.ndarray] = None,
    grip_planning_qpos: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> bool:
    """
    Set arm qpos, optionally set gripper qpos to a planning posture, forward,
    then check contacts against forbidden_pairs.
    """
    data.qpos[arm_q_idx] = q_arm.copy()

    if grip_q_idx is not None and grip_planning_qpos is not None:
        data.qpos[grip_q_idx] = grip_planning_qpos

    mujoco.mj_forward(model, data)

    if data.ncon == 0:
        return True

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        dist = con.dist

        if g1 in ignore_gid or g2 in ignore_gid:
            continue

        if (g1, g2) in forbidden_pairs or (g2, g1) in forbidden_pairs:
            if dist < DEEP_PEN_THRESH:
                if verbose or DEBUG_COLLISIONS_ON_FAIL:
                    print("\n[COLLISION] Forbidden contact:")
                    print(f"  {geom_name(model, g1)} <-> {geom_name(model, g2)}")
                    print(f"  dist = {dist:.6e}")
                return False

    return True


def check_edge_collision_free(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_from: np.ndarray,
    q_to: np.ndarray,
    arm_q_idx: np.ndarray,
    forbidden_pairs: Sequence[Tuple[int, int]],
    ignore_gid: set,
    wrap_mask: np.ndarray,
    grip_q_idx: Optional[np.ndarray] = None,
    grip_planning_qpos: Optional[np.ndarray] = None,
    steps: int = EDGE_COLL_STEPS,
) -> bool:
    """
    Discretize edge using wrapped interpolation and collision-check each sample.
    """
    for s in np.linspace(0.0, 1.0, steps):
        q = interp_with_wrap(q_from, q_to, float(s), wrap_mask)
        if not is_state_collision_free(
            model, data, q, arm_q_idx,
            forbidden_pairs, ignore_gid,
            grip_q_idx=grip_q_idx,
            grip_planning_qpos=grip_planning_qpos,
        ):
            return False
    return True


# ===========================================================================
# IK
# ===========================================================================

def solve_ik_unconstrained(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_sid: int,
    arm_q_idx: np.ndarray,
    arm_dof_idx: np.ndarray,
    q_init: np.ndarray,
    target_world: np.ndarray,
    neutral_q: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Position-only IK: gradient step dq ~ J^T * error + small regularization.
    """
    q = q_init.copy()
    if neutral_q is None:
        neutral_q = np.zeros_like(q)

    data.qpos[arm_q_idx] = q
    mujoco.mj_forward(model, data)

    prev_err = None
    errors = []

    if verbose:
        print("\n--- UNCONSTRAINED IK (single seed) ---")

    for step in range(IK_MAX_STEPS):
        mujoco.mj_forward(model, data)
        x_ee = np.array(data.site_xpos[ee_sid])
        err = target_world - x_ee
        err_norm = float(np.linalg.norm(err))
        errors.append(err_norm)

        if verbose:
            print(f"  [IK] step {step}: err={err_norm:.6f}, q_arm={q}")

        if err_norm < IK_ERR_TOL:
            if verbose:
                print("  [IK] Converged to tolerance.")
            break

        if prev_err is not None:
            improve = prev_err - err_norm
            if improve < IK_STALL_IMPROVE_TOL:
                if verbose:
                    print(
                        f"  [IK] Stalled at step {step}, "
                        f"improvement {improve:.3e} < {IK_STALL_IMPROVE_TOL:g}."
                    )
                break
        prev_err = err_norm

        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, ee_sid)
        J = jacp[:, arm_dof_idx]  # 3x6

        dq_task = IK_BASE_STEP_SIZE * (J.T @ err)
        dq_reg  = IK_REG_WEIGHT * (neutral_q - q)
        dq = dq_task + dq_reg

        nrm = float(np.linalg.norm(dq))
        if nrm > IK_MAX_STEP_NORM:
            dq *= (IK_MAX_STEP_NORM / nrm)

        q = q + dq
        data.qpos[arm_q_idx] = q

    return q, float(errors[-1]), len(errors)


def solve_ik_multi_seed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_sid: int,
    arm_q_idx: np.ndarray,
    arm_dof_idx: np.ndarray,
    seeds: Sequence[np.ndarray],
    target_world: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, int]:
    best_q = None
    best_err = np.inf
    best_steps = 0
    best_idx = -1

    neutral_q = np.zeros_like(seeds[0])

    for i, q0 in enumerate(seeds):
        if verbose:
            print("\n=================================")
            print(f"=== IK SEED {i} (q0={q0}) ===")
            print("=================================")

        q_sol, err, steps = solve_ik_unconstrained(
            model, data, ee_sid,
            arm_q_idx, arm_dof_idx,
            q0, target_world,
            neutral_q=neutral_q,
            verbose=verbose,
        )

        if err < best_err:
            best_err = err
            best_q = q_sol.copy()
            best_steps = steps
            best_idx = i

    if verbose:
        print("\n=== BEST IK ACROSS SEEDS ===")
        print(f"  best_seed_idx = {best_idx}")
        print(f"  best_err      = {best_err:.6f}")
        print(f"  best_q        = {best_q}")

    return best_q, float(best_err), int(best_steps)


# ===========================================================================
# RRT
# ===========================================================================

@dataclass
class RRTNode:
    q: np.ndarray
    parent: int


def rrt_plan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_q_idx: np.ndarray,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    joint_ranges: np.ndarray,
    forbidden_pairs: Sequence[Tuple[int, int]],
    ignore_gid: set,
    wrap_mask: np.ndarray,
    grip_q_idx: Optional[np.ndarray] = None,
    grip_planning_qpos: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[List[np.ndarray]]:
    """
    Joint-space RRT with wrapped-joint metric.
    """
    print("\n--- RRT PLANNING IN JOINT SPACE ---")
    dim = q_start.shape[0]
    nodes: List[RRTNode] = [RRTNode(q=q_start.copy(), parent=-1)]

    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)

    def sample_config() -> np.ndarray:
        if rng.random() < RRT_GOAL_BIAS:
            return q_goal.copy()
        qs = []
        for j in range(dim):
            lo, hi = joint_ranges[j]
            qs.append(rng.uniform(lo, hi))
        return np.array(qs, dtype=float)

    def nearest_node(q_rand: np.ndarray) -> int:
        dists = [dist_with_wrap(n.q, q_rand, wrap_mask) for n in nodes]
        return int(np.argmin(dists))

    def steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
        d = diff_with_wrap(q_to, q_from, wrap_mask)
        dist = float(np.linalg.norm(d))
        if dist < step_size:
            return q_to.copy()
        return q_from + d * (step_size / dist)

    for it in range(RRT_MAX_ITERS):
        q_rand = sample_config()
        idx_near = nearest_node(q_rand)
        q_near = nodes[idx_near].q

        q_new = steer(q_near, q_rand, RRT_STEP_SIZE)

        if not check_edge_collision_free(
            model, data, q_near, q_new,
            arm_q_idx, forbidden_pairs, ignore_gid,
            wrap_mask=wrap_mask,
            grip_q_idx=grip_q_idx,
            grip_planning_qpos=grip_planning_qpos,
        ):
            continue

        nodes.append(RRTNode(q=q_new, parent=idx_near))
        dist_goal = dist_with_wrap(q_new, q_goal, wrap_mask)
        print(f"  [RRT] iter={it}, nodes={len(nodes)}, dist_to_goal={dist_goal:.4f}")

        if dist_goal < RRT_GOAL_THRESH:
            print("  [RRT] Reached goal region.")

            # attempt direct connect to exact goal
            goal_idx = len(nodes) - 1
            if check_edge_collision_free(
                model, data, q_new, q_goal,
                arm_q_idx, forbidden_pairs, ignore_gid,
                wrap_mask=wrap_mask,
                grip_q_idx=grip_q_idx,
                grip_planning_qpos=grip_planning_qpos,
            ):
                nodes.append(RRTNode(q=q_goal.copy(), parent=goal_idx))
                goal_idx = len(nodes) - 1

            # reconstruct
            path: List[np.ndarray] = []
            cur = goal_idx
            while cur >= 0:
                path.append(nodes[cur].q.copy())
                cur = nodes[cur].parent
            path.reverse()

            print(f"  [RRT] Path found with {len(path)} waypoints.")
            return path

    print("  [RRT] Failed to find a path within iteration limit.")
    return None


# ===========================================================================
# Piecewise planning utilities
# ===========================================================================

def densify_waypoints(
    waypoints: Sequence[np.ndarray],
    wrap_mask: np.ndarray,
    points_per_segment: int,
) -> List[np.ndarray]:
    """
    Densify a waypoint list by wrapped linear interpolation.
    """
    if len(waypoints) <= 1:
        return [waypoints[0].copy()]

    out: List[np.ndarray] = []
    for i in range(len(waypoints) - 1):
        q0 = waypoints[i]
        q1 = waypoints[i + 1]
        for s in np.linspace(0.0, 1.0, points_per_segment, endpoint=False):
            out.append(interp_with_wrap(q0, q1, float(s), wrap_mask))
    out.append(waypoints[-1].copy())
    return out


def plan_piecewise_waypoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_q_idx: np.ndarray,
    joint_ranges: np.ndarray,
    forbidden_pairs: Sequence[Tuple[int, int]],
    ignore_gid: set,
    wrap_mask: np.ndarray,
    waypoints: Sequence[np.ndarray],
    grip_q_idx: Optional[np.ndarray] = None,
    grip_planning_qpos: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    points_per_straight_segment: int = STRAIGHTLINE_POINTS,
) -> Optional[List[np.ndarray]]:
    """
    For each consecutive pair of waypoints:
      - if straight-line edge is collision-free => use wrapped interpolation
      - else => run RRT to bridge them, then densify RRT path edges
    Returns a dense list of joint configurations.
    """
    if len(waypoints) < 2:
        return [waypoints[0].copy()]

    full: List[np.ndarray] = []

    print("\n=== BEGIN PIECEWISE PLANNING ===")
    for seg_i in range(len(waypoints) - 1):
        qA = waypoints[seg_i].copy()
        qB = waypoints[seg_i + 1].copy()

        print("\n" + "=" * 50)
        print(f"=== SEGMENT {seg_i}: waypoint[{seg_i}] -> waypoint[{seg_i+1}] ===")
        print("=" * 50)
        print(f"  qA = {qA}")
        print(f"  qB = {qB}")

        csA = is_state_collision_free(
            model, data, qA, arm_q_idx,
            forbidden_pairs, ignore_gid,
            grip_q_idx=grip_q_idx,
            grip_planning_qpos=grip_planning_qpos,
            verbose=True,
        )
        csB = is_state_collision_free(
            model, data, qB, arm_q_idx,
            forbidden_pairs, ignore_gid,
            grip_q_idx=grip_q_idx,
            grip_planning_qpos=grip_planning_qpos,
            verbose=True,
        )

        print(f"  Collision-free A? {csA}")
        print(f"  Collision-free B? {csB}")

        if not csA or not csB:
            print("  [SEGMENT] One endpoint is in collision; aborting piecewise plan.")
            return None

        print("  [SEGMENT] Trying straight-line edge...")
        ok_edge = check_edge_collision_free(
            model, data, qA, qB,
            arm_q_idx, forbidden_pairs, ignore_gid,
            wrap_mask=wrap_mask,
            grip_q_idx=grip_q_idx,
            grip_planning_qpos=grip_planning_qpos,
        )
        print(f"  [SEGMENT] Straight-line edge collision-free? {ok_edge}")

        if ok_edge:
            print(f"  [SEGMENT] Using straight-line interpolation with {points_per_straight_segment} points.")
            dense = densify_waypoints([qA, qB], wrap_mask, points_per_straight_segment)
        else:
            print("  [SEGMENT] Straight line blocked. Calling RRT...")
            path = rrt_plan(
                model, data, arm_q_idx,
                q_start=qA,
                q_goal=qB,
                joint_ranges=joint_ranges,
                forbidden_pairs=forbidden_pairs,
                ignore_gid=ignore_gid,
                wrap_mask=wrap_mask,
                grip_q_idx=grip_q_idx,
                grip_planning_qpos=grip_planning_qpos,
                rng=rng,
            )
            if path is None:
                print("  [SEGMENT] RRT failed for this segment.")
                return None

            # Densify each RRT edge
            dense = []
            for i in range(len(path) - 1):
                dense.extend(densify_waypoints([path[i], path[i+1]], wrap_mask, points_per_straight_segment)[:-1])
            dense.append(path[-1].copy())
            print(f"  [SEGMENT] RRT path densified to {len(dense)} points.")

        # Concatenate, avoid duplicate join
        if not full:
            full.extend(dense)
        else:
            full.extend(dense[1:])

    return full


# ===========================================================================
# Plan compilation: arm path + gripper ctrl column
# ===========================================================================

def ramp_values(v0: float, v1: float, steps: int) -> np.ndarray:
    if steps <= 1:
        return np.array([v1], dtype=float)
    return np.linspace(v0, v1, steps, dtype=float)


def compile_with_gripper_ctrl(
    arm_traj: Sequence[np.ndarray],
    gripper_ctrl_for_arm: float,
    open_ramp_steps: int,
    close_ramp_steps: int,
    gripper_open: float,
    gripper_closed: float,
    start_hold_arm: np.ndarray,
    end_hold_arm: np.ndarray,
) -> np.ndarray:
    """
    Build a numeric array of shape (N, 7):
      [q1..q6, gripper_ctrl]
    with:
      - an initial gripper open ramp while holding start_hold_arm
      - then arm trajectory with constant gripper ctrl
      - then final gripper close ramp while holding end_hold_arm
    """
    rows: List[np.ndarray] = []

    # Open ramp
    open_vals = ramp_values(gripper_ctrl_for_arm, gripper_open, open_ramp_steps)
    for u in open_vals:
        rows.append(np.concatenate([start_hold_arm.copy(), np.array([u], dtype=float)]))

    # Arm motion (keep gripper at OPEN by default)
    for q in arm_traj:
        rows.append(np.concatenate([q.copy(), np.array([gripper_open], dtype=float)]))

    # Close ramp (hold end pose)
    close_vals = ramp_values(gripper_open, gripper_closed, close_ramp_steps)
    for u in close_vals:
        rows.append(np.concatenate([end_hold_arm.copy(), np.array([u], dtype=float)]))

    return np.vstack(rows)


def save_plan_csv(
    csv_path: str,
    plan: np.ndarray,
    arm_joint_names: Sequence[str],
    gripper_col_name: str = "gripper_ctrl",
):
    assert plan.ndim == 2 and plan.shape[1] == len(arm_joint_names) + 1
    header = ",".join(list(arm_joint_names) + [gripper_col_name])
    np.savetxt(csv_path, plan, delimiter=",", header=header, comments="")
    print(f"\n[RESULT] Saved plan CSV: {csv_path}")
    print(f"         shape = {plan.shape} rows x cols")