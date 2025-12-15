#!/usr/bin/env python3
# motion/plan_numerical_k.py
#
# UR3 IK + motion planning for cup gripping:
#   * (NEW) Optional manual waypoint chain (piecewise planning)
#   * (NEW) For each gap: try straight-line edge first, else RRT
#   * (NEW) Optional local IK refinement at the end (from last waypoint) to hit CUP_GRIP_SITE_NAME
#
# Obstacles:
#   - Microwave body + door
#   - UR3 table + microwave table + floor
#   - Cup body, but ONLY against non-gripper geoms (arm should not hit/knock it)
#   - Basic self-collision pairs for UR3 links
#
# Grasp:
#   - Contacts between Left_Grip_col / Right_Grip_col and cup_col are allowed,
#     so the jaws can actually grip the cup.
#
# IMPORTANT:
#   - Collision checks are evaluated with a fixed "planning gripper posture".
#     By default this is slightly LESS open than the "wide-open" pose to avoid
#     micro_table <-> Right_Grip_col barely-penetrating contacts.

import mujoco
import numpy as np
import os
import time
from dataclasses import dataclass

np.set_printoptions(precision=5, suppress=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH    = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

EE_SITE_NAME       = "ee_site"
CUP_GRIP_SITE_NAME = "cup_grip_target_site_back"   # <-- back target

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

# ---------------------------------------------------------------------------
# Gripper posture used DURING PLANNING collision checks
# ---------------------------------------------------------------------------

# Your true wide-open pose (use in playback)
GRIPPER_OPEN_QPOS = np.array([0.7, -0.7], dtype=float)

# (NEW) Planning posture: slightly less open to avoid table<->jaw edge-case hits.
# If you want to force wide-open planning again, set this equal to GRIPPER_OPEN_QPOS.
GRIPPER_PLAN_QPOS = np.array([0.45, -0.45], dtype=float)

# ---------------------------------------------------------------------------
# Manual waypoint chain (NEW)
# ---------------------------------------------------------------------------

USE_MANUAL_WAYPOINT_CHAIN = True

# Your requested manual chain for BACK target (in order).
MANUAL_WAYPOINTS_BACK = [
    np.array([0.0,  0.0,   0.0,   0.0,   0.0,   0.0], dtype=float),  # start (redundant; will be overridden by q_start)
    np.array([2.7,  0.0,   0.0,   0.0,   0.0,   0.0], dtype=float),
    np.array([2.7,  0.0,   0.0,  -1.7,  -1.7,  -1.7], dtype=float),
    np.array([2.7,  0.0,   1.54, -1.7,  -1.7,  -1.7], dtype=float),
    np.array([2.5, -1.92,  1.54, -1.7,  -1.7,  -1.7], dtype=float),
    np.array([2.7, -1.92,  1.54, -1.73, -1.48, -1.79], dtype=float),  # your final “sure-fire” pose
    np.array([2.7, -1.92,  1.54, -1.73, -1.48, -1.79], dtype=float),  # your final “sure-fire” pose
]

# (Optional) The older FRONT pose you mentioned (kept here for reference)
MANUAL_GOAL_FRONT = np.array(
    [-3.11, -1.88, 1.26, -1.73, -1.57, -1.85],
    dtype=float,
)

# ---------------------------------------------------------------------------
# Local IK refinement (NEW)
# ---------------------------------------------------------------------------

USE_LOCAL_IK_FINAL = True          # refine from last manual waypoint to hit target site
LOCAL_IK_SEEDS_AROUND_FINAL = 4    # random perturb seeds around last waypoint
LOCAL_IK_NOISE_SIGMA = 0.15        # rad
LOCAL_IK_ACCEPT_ERR = 5e-3         # m; if IK does worse than this, keep manual final

# ---------------------------------------------------------------------------
# Straight-line preference between waypoints (NEW)
# ---------------------------------------------------------------------------

TRY_STRAIGHT_LINE_FIRST = True
STRAIGHT_LINE_STEPS     = 60       # interpolation points if straight-line is collision-free

# ---------------------------------------------------------------------------
# IK parameters
# ---------------------------------------------------------------------------

IK_MAX_STEPS           = 600
IK_BASE_STEP_SIZE      = 0.5
IK_MAX_STEP_NORM       = 0.20
IK_ERR_TOL             = 1e-3
IK_STALL_IMPROVE_TOL   = 1e-8
IK_REG_WEIGHT          = 1e-4

# ---------------------------------------------------------------------------
# RRT parameters
# ---------------------------------------------------------------------------

RRT_MAX_ITERS   = 3000
RRT_STEP_SIZE   = 0.02
RRT_GOAL_THRESH = 0.08
RRT_GOAL_BIAS   = 0.25

EDGE_COLL_STEPS = 25

# ---------------------------------------------------------------------------
# Collision parameters
# ---------------------------------------------------------------------------

# Default “hard” penetration threshold
DEEP_PEN_THRESH = -1e-3

# (NEW) Optional pair-specific tolerance (helps with the micro_table <-> jaw hairline case)
# If you want zero special-casing, set this dict empty.
PAIR_PEN_THRESH_BY_NAME = {
    ("micro_table", "Right_Grip_col"): -2e-3,
    ("micro_table", "Left_Grip_col"):  -2e-3,
}

# For reproducible random sampling
np.random.seed(0)
DEBUG_COLLISIONS_ON_FAIL = True


# ======================================================================
# Utility helpers
# ======================================================================

def geom_name(model, gid: int) -> str:
    if gid < 0:
        return f"<none:{gid}>"
    nm = model.geom(gid).name
    return nm if nm is not None else f"<geom:{gid}>"


def get_site_id(model, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"Site '{name}' not found in model.")
    return sid


def get_joint_ids_and_ranges(model, joint_names):
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


def get_gripper_q_indices(model):
    q_idx = []
    for jn in GRIPPER_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"Gripper joint '{jn}' not found in model.")
        q_idx.append(model.jnt_qposadr[jid])
    return np.asarray(q_idx, dtype=int)


def world_pos_of_site(model, data, sid):
    mujoco.mj_forward(model, data)
    return np.array(data.site_xpos[sid])


def build_pair_pen_thresh_map(model):
    """
    Resolve PAIR_PEN_THRESH_BY_NAME into a dict keyed by (gid1, gid2) and symmetric.
    """
    name_to_gid = {}
    for gid in range(model.ngeom):
        nm = model.geom(gid).name
        if nm:
            name_to_gid[nm] = gid

    pair_map = {}
    missing = []
    for (n1, n2), thr in PAIR_PEN_THRESH_BY_NAME.items():
        if n1 in name_to_gid and n2 in name_to_gid:
            g1 = name_to_gid[n1]
            g2 = name_to_gid[n2]
            pair_map[(g1, g2)] = thr
            pair_map[(g2, g1)] = thr
        else:
            missing.append((n1, n2))

    if PAIR_PEN_THRESH_BY_NAME:
        print("\n=== PAIR-SPECIFIC PEN THRESH ===")
        print(f"  requested: {len(PAIR_PEN_THRESH_BY_NAME)}")
        print(f"  resolved:  {len(pair_map)//2}")
        print(f"  missing:   {len(missing)}")
        for (g1, g2), thr in list(pair_map.items())[:6]:
            print(f"    {geom_name(model, g1)} <-> {geom_name(model, g2)} : {thr:.2e}")

    return pair_map


def get_forbidden_pairs(model):
    arm_geoms = [
        "base_link_geom",
        "link1_geom",
        "link2_geom",
        "link3_geom",
        "link4_geom",
        "link5_geom",
        "link6_geom",
    ]

    jaw_geoms_visual = ["Left_Grip_geom", "Right_Grip_geom"]
    jaw_geoms_collision = ["Left_Grip_col", "Right_Grip_col"]

    table_geoms = ["ur3_table", "micro_table"]
    microwave_geoms = ["micro_Base_geom", "micro_door_geom"]
    floor_geoms = ["floor"]

    cup_geom = "cup_col"

    forbidden_pairs_by_name = [
        ("base_link_geom", "link1_geom"),
        ("link1_geom", "link3_geom"),
        ("link1_geom", "link4_geom"),
        ("link1_geom", "link5_geom"),
        ("link1_geom", "link6_geom"),
        ("link2_geom", "link4_geom"),
        ("link2_geom", "link5_geom"),
        ("link2_geom", "link6_geom"),
    ]

    arm_and_jaws = arm_geoms + jaw_geoms_visual + jaw_geoms_collision
    for obs in table_geoms + microwave_geoms + floor_geoms:
        for link in arm_and_jaws:
            forbidden_pairs_by_name.append((obs, link))

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

    name_to_gid = {}
    for gid in range(model.ngeom):
        nm = model.geom(gid).name
        if nm:
            name_to_gid[nm] = gid

    forbidden_pairs = []
    missing = []
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


def is_state_collision_free(model, data,
                            q_arm,
                            arm_q_idx,
                            forbidden_pairs,
                            ignore_gid,
                            pair_pen_thresh_map,
                            grip_q_idx=None,
                            grip_plan_qpos=None,
                            verbose=False):
    data.qpos[arm_q_idx] = q_arm.copy()

    if grip_q_idx is not None and grip_plan_qpos is not None:
        data.qpos[grip_q_idx] = grip_plan_qpos

    mujoco.mj_forward(model, data)

    if data.ncon == 0:
        return True

    for i in range(data.ncon):
        con  = data.contact[i]
        g1   = con.geom1
        g2   = con.geom2
        dist = con.dist

        if g1 in ignore_gid or g2 in ignore_gid:
            continue

        if (g1, g2) in forbidden_pairs or (g2, g1) in forbidden_pairs:
            thr = pair_pen_thresh_map.get((g1, g2), DEEP_PEN_THRESH)
            if dist < thr:
                if verbose or DEBUG_COLLISIONS_ON_FAIL:
                    print("\n[COLLISION] Forbidden contact:")
                    print(f"  {geom_name(model, g1)} <-> {geom_name(model, g2)}")
                    print(f"  dist = {dist:.6e} (thr={thr:.6e})")
                return False

    return True


def check_edge_collision_free(model, data,
                              q_from,
                              q_to,
                              arm_q_idx,
                              forbidden_pairs,
                              ignore_gid,
                              pair_pen_thresh_map,
                              grip_q_idx=None,
                              grip_plan_qpos=None,
                              steps=EDGE_COLL_STEPS):
    for s in np.linspace(0.0, 1.0, steps):
        q_interp = (1.0 - s) * q_from + s * q_to
        if not is_state_collision_free(
            model, data, q_interp,
            arm_q_idx, forbidden_pairs, ignore_gid,
            pair_pen_thresh_map,
            grip_q_idx=grip_q_idx,
            grip_plan_qpos=grip_plan_qpos,
        ):
            return False
    return True


# ======================================================================
# IK
# ======================================================================

def solve_ik_unconstrained(
    model,
    data,
    ee_sid,
    arm_q_idx,
    arm_dof_idx,
    q_init,
    target_world,
    neutral_q=None,
    verbose=True,
):
    q_arm = q_init.copy()

    if neutral_q is None:
        neutral_q = np.zeros_like(q_arm)

    data.qpos[arm_q_idx] = q_arm
    mujoco.mj_forward(model, data)

    errors   = []
    prev_err = None

    if verbose:
        print("\n--- UNCONSTRAINED IK (single seed) ---")

    for step in range(IK_MAX_STEPS):
        mujoco.mj_forward(model, data)
        x_ee = np.array(data.site_xpos[ee_sid])
        err = target_world - x_ee
        err_norm = np.linalg.norm(err)
        errors.append(err_norm)

        if verbose:
            print(f"  [IK] step {step}: err={err_norm:.6f}, q_arm={q_arm}")

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
        J_arm = jacp[:, arm_dof_idx]

        dq_task = IK_BASE_STEP_SIZE * (J_arm.T @ err)
        dq_reg  = IK_REG_WEIGHT * (neutral_q - q_arm)
        dq = dq_task + dq_reg

        nrm = np.linalg.norm(dq)
        if nrm > IK_MAX_STEP_NORM:
            dq *= (IK_MAX_STEP_NORM / nrm)

        q_arm = q_arm + dq
        data.qpos[arm_q_idx] = q_arm

    if verbose:
        print(f"  [IK] final_err={errors[-1]:.6f} after {len(errors)} steps")
    return q_arm, errors[-1], len(errors)


def solve_ik_local_around(
    model, data,
    ee_sid, arm_q_idx, arm_dof_idx,
    q_seed,
    joint_ranges,
    target_world,
):
    """
    (NEW) Local multi-seed IK around q_seed (small perturbations).
    """
    seeds = [q_seed.copy()]
    for _ in range(LOCAL_IK_SEEDS_AROUND_FINAL):
        noise = np.random.normal(0.0, LOCAL_IK_NOISE_SIGMA, size=q_seed.shape)
        q = np.clip(q_seed + noise, joint_ranges[:, 0], joint_ranges[:, 1])
        seeds.append(q)

    best_q = None
    best_err = np.inf
    best_steps = None
    neutral_q = np.zeros_like(q_seed)

    print("\n=== LOCAL IK AROUND FINAL WAYPOINT ===")
    print(f"  base_seed = {q_seed}")
    print(f"  num_seeds = {len(seeds)}")

    for i, q0 in enumerate(seeds):
        print("\n---------------------------------")
        print(f"  Local IK seed {i}: q0={q0}")
        q_sol, err, steps = solve_ik_unconstrained(
            model, data, ee_sid, arm_q_idx, arm_dof_idx,
            q_init=q0, target_world=target_world,
            neutral_q=neutral_q, verbose=True,
        )
        if err < best_err:
            best_err = err
            best_q = q_sol.copy()
            best_steps = steps

    print("\n=== LOCAL IK BEST ===")
    print(f"  best_err  = {best_err:.6f}")
    print(f"  best_q    = {best_q}")
    print(f"  best_steps= {best_steps}")

    return best_q, best_err, best_steps


# ======================================================================
# RRT
# ======================================================================

@dataclass
class RRTNode:
    q: np.ndarray
    parent: int


def rrt_plan(
    model,
    data,
    arm_q_idx,
    q_start,
    q_goal,
    joint_ranges,
    forbidden_pairs,
    ignore_gid,
    pair_pen_thresh_map,
    grip_q_idx=None,
    grip_plan_qpos=None,
    *,
    max_iters=RRT_MAX_ITERS,
    step_size=RRT_STEP_SIZE,
    goal_thresh=RRT_GOAL_THRESH,
    goal_bias=RRT_GOAL_BIAS,
):
    print("\n--- RRT PLANNING IN JOINT SPACE ---")

    dim   = q_start.shape[0]
    nodes = [RRTNode(q=q_start.copy(), parent=-1)]

    def sample_config():
        if np.random.rand() < goal_bias:
            return q_goal.copy()
        qs = []
        for j in range(dim):
            lo, hi = joint_ranges[j]
            qs.append(np.random.uniform(lo, hi))
        return np.array(qs, dtype=float)

    def nearest_node(q_rand):
        dists = [np.linalg.norm(n.q - q_rand) for n in nodes]
        return int(np.argmin(dists))

    def steer(q_from, q_to):
        v    = q_to - q_from
        dist = np.linalg.norm(v)
        if dist < step_size:
            return q_to.copy()
        return q_from + v * (step_size / dist)

    for it in range(max_iters):
        q_rand   = sample_config()
        idx_near = nearest_node(q_rand)
        q_near   = nodes[idx_near].q

        q_new = steer(q_near, q_rand)

        if not check_edge_collision_free(
            model, data, q_near, q_new,
            arm_q_idx, forbidden_pairs, ignore_gid,
            pair_pen_thresh_map,
            grip_q_idx=grip_q_idx,
            grip_plan_qpos=grip_plan_qpos,
        ):
            continue

        nodes.append(RRTNode(q=q_new, parent=idx_near))

        dist_to_goal = np.linalg.norm(q_new - q_goal)
        print(f"  [RRT] iter={it}, nodes={len(nodes)}, dist_to_goal={dist_to_goal:.4f}")

        if dist_to_goal < goal_thresh:
            print("  [RRT] Reached goal region.")

            if check_edge_collision_free(
                model, data, q_new, q_goal,
                arm_q_idx, forbidden_pairs, ignore_gid,
                pair_pen_thresh_map,
                grip_q_idx=grip_q_idx,
                grip_plan_qpos=grip_plan_qpos,
            ):
                nodes.append(RRTNode(q=q_goal.copy(), parent=len(nodes) - 1))
                goal_idx = len(nodes) - 1
            else:
                goal_idx = len(nodes) - 1

            path = []
            cur = goal_idx
            while cur >= 0:
                path.append(nodes[cur].q.copy())
                cur = nodes[cur].parent
            path.reverse()

            print(f"  [RRT] Path found with {len(path)} waypoints.")
            return path

    print("  [RRT] Failed to find a path within iteration limit.")
    return None


# ======================================================================
# Piecewise planning through manual waypoints (NEW)
# ======================================================================

def straight_line_path(q0, q1, steps):
    pts = []
    for s in np.linspace(0.0, 1.0, steps):
        pts.append((1.0 - s) * q0 + s * q1)
    return pts


def plan_piecewise(
    model, data,
    arm_q_idx,
    joint_ranges,
    forbidden_pairs,
    ignore_gid,
    pair_pen_thresh_map,
    grip_q_idx,
    grip_plan_qpos,
    waypoints,
):
    """
    Connect waypoints[i] -> waypoints[i+1] sequentially.
    - try straight-line edge first (if enabled)
    - else run RRT for that segment
    """
    full_path = []

    for seg in range(len(waypoints) - 1):
        qA = waypoints[seg].copy()
        qB = waypoints[seg + 1].copy()

        print("\n==================================================")
        print(f"=== SEGMENT {seg}: waypoint[{seg}] -> waypoint[{seg+1}] ===")
        print("==================================================")
        print(f"  qA = {qA}")
        print(f"  qB = {qB}")

        okA = is_state_collision_free(
            model, data, qA, arm_q_idx,
            forbidden_pairs, ignore_gid, pair_pen_thresh_map,
            grip_q_idx=grip_q_idx, grip_plan_qpos=grip_plan_qpos, verbose=True,
        )
        okB = is_state_collision_free(
            model, data, qB, arm_q_idx,
            forbidden_pairs, ignore_gid, pair_pen_thresh_map,
            grip_q_idx=grip_q_idx, grip_plan_qpos=grip_plan_qpos, verbose=True,
        )
        print(f"  Collision-free A? {okA}")
        print(f"  Collision-free B? {okB}")

        if not okA:
            print("  [SEGMENT] Start waypoint is in collision. Aborting.")
            return None
        if not okB:
            print("  [SEGMENT] Goal waypoint is in collision. Aborting.")
            return None

        if TRY_STRAIGHT_LINE_FIRST:
            print("  [SEGMENT] Trying straight-line edge...")
            ok_edge = check_edge_collision_free(
                model, data, qA, qB,
                arm_q_idx, forbidden_pairs, ignore_gid, pair_pen_thresh_map,
                grip_q_idx=grip_q_idx, grip_plan_qpos=grip_plan_qpos,
                steps=EDGE_COLL_STEPS,
            )
            print(f"  [SEGMENT] Straight-line edge collision-free? {ok_edge}")

            if ok_edge:
                seg_path = straight_line_path(qA, qB, STRAIGHT_LINE_STEPS)
                print(f"  [SEGMENT] Using straight-line interpolation with {len(seg_path)} points.")
                # stitch
                if len(full_path) == 0:
                    full_path.extend(seg_path)
                else:
                    full_path.extend(seg_path[1:])
                continue

        print("  [SEGMENT] Running RRT for this gap...")
        seg_path = rrt_plan(
            model=model,
            data=data,
            arm_q_idx=arm_q_idx,
            q_start=qA,
            q_goal=qB,
            joint_ranges=joint_ranges,
            forbidden_pairs=forbidden_pairs,
            ignore_gid=ignore_gid,
            pair_pen_thresh_map=pair_pen_thresh_map,
            grip_q_idx=grip_q_idx,
            grip_plan_qpos=grip_plan_qpos,
        )
        if seg_path is None:
            print(f"  [SEGMENT] RRT failed on segment {seg}.")
            return None

        if len(full_path) == 0:
            full_path.extend(seg_path)
        else:
            full_path.extend(seg_path[1:])

        print(f"  [SEGMENT] Segment path length = {len(seg_path)}")
        print(f"  [SEGMENT] Full path so far   = {len(full_path)}")

    return full_path


# ======================================================================
# Main
# ======================================================================

def main():
    print(f"Loading: {XML_PATH}")
    if not os.path.exists(XML_PATH):
        print("XML file not found.")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    ee_sid   = get_site_id(model, EE_SITE_NAME)
    grip_sid = get_site_id(model, CUP_GRIP_SITE_NAME)

    j_ids, arm_q_idx, arm_dof_idx, joint_ranges = get_joint_ids_and_ranges(
        model, ARM_JOINT_NAMES
    )

    grip_q_idx = get_gripper_q_indices(model)

    pair_pen_thresh_map = build_pair_pen_thresh_map(model)

    # Save XML default arm config
    q_xml_default = data.qpos[arm_q_idx].copy()

    # Start config: arm zeros, gripper planning posture
    q_start = np.zeros(len(ARM_JOINT_NAMES), dtype=float)
    data.qpos[arm_q_idx]  = q_start
    data.qpos[grip_q_idx] = GRIPPER_PLAN_QPOS
    mujoco.mj_forward(model, data)

    print("\n=== START CONFIGURATION ===")
    print("  q_start (arm):", q_start)
    print("  gripper (plan posture):", GRIPPER_PLAN_QPOS)
    print("  gripper (wide-open playback pose):", GRIPPER_OPEN_QPOS)

    target_world = world_pos_of_site(model, data, grip_sid)
    print(f"\nTarget world position ({CUP_GRIP_SITE_NAME}): {target_world}")

    # ------------------------------------------------------------------
    # Collision setup
    # ------------------------------------------------------------------
    forbidden_pairs, ignore_gid = get_forbidden_pairs(model)

    cs_start = is_state_collision_free(
        model, data, q_start, arm_q_idx,
        forbidden_pairs, ignore_gid, pair_pen_thresh_map,
        grip_q_idx=grip_q_idx,
        grip_plan_qpos=GRIPPER_PLAN_QPOS,
        verbose=True,
    )
    print(f"\nCollision-free start? {cs_start}")

    if not cs_start:
        print("[ABORT] Start config is in collision.")
        return

    # ------------------------------------------------------------------
    # Build the waypoint list
    # ------------------------------------------------------------------
    if USE_MANUAL_WAYPOINT_CHAIN:
        print("\n=== USING MANUAL WAYPOINT CHAIN ===")
        manual = [q.copy() for q in MANUAL_WAYPOINTS_BACK]

        # enforce first waypoint = q_start (even if you included zeros already)
        manual[0] = q_start.copy()

        print("Manual waypoints:")
        for i, q in enumerate(manual):
            print(f"  wp[{i}] = {q}")

        # Optional local IK refinement for the final target site
        q_final_manual = manual[-1].copy()
        q_goal = q_final_manual.copy()
        q_goal_ik = None
        ik_err = None

        if USE_LOCAL_IK_FINAL:
            print("\n=== LOCAL IK REFINEMENT ENABLED ===")
            q_goal_ik, ik_err, _ = solve_ik_local_around(
                model, data, ee_sid, arm_q_idx, arm_dof_idx,
                q_seed=q_final_manual,
                joint_ranges=joint_ranges,
                target_world=target_world,
            )

            print("\nChecking collision at IK-refined goal (with planning gripper posture)...")
            ok_goal_ik = is_state_collision_free(
                model, data, q_goal_ik, arm_q_idx,
                forbidden_pairs, ignore_gid, pair_pen_thresh_map,
                grip_q_idx=grip_q_idx, grip_plan_qpos=GRIPPER_PLAN_QPOS, verbose=True,
            )
            print(f"  IK goal collision-free? {ok_goal_ik}")

            if ok_goal_ik and ik_err is not None and ik_err <= LOCAL_IK_ACCEPT_ERR:
                print(f"\n[OK] Using IK-refined final goal (err={ik_err:.6f} m).")
                q_goal = q_goal_ik.copy()
            else:
                print(f"\n[NOTE] Keeping manual final goal (IK err={ik_err}, ok={ok_goal_ik}).")

        # Append q_goal only if it differs meaningfully from manual last
        if np.linalg.norm(q_goal - manual[-1]) > 1e-6:
            manual.append(q_goal.copy())
            print("\nAppended IK-refined q_goal as final waypoint:")
            print(f"  wp[{len(manual)-1}] = {manual[-1]}")

        waypoints = manual

    else:
        print("\n=== NO MANUAL WAYPOINTS: FALLING BACK TO ORIGINAL GLOBAL IK+RRT ===")
        # Original behavior (kept as fallback)
        # Multi-seed IK (global) then single RRT
        q_manual = MANUAL_GOAL_FRONT.copy()
        seeds = [q_start.copy(), q_xml_default.copy(), q_manual.copy()]
        for _ in range(4):
            noise = np.random.normal(0.0, 0.25, size=q_manual.shape)
            q = np.clip(q_manual + noise, joint_ranges[:, 0], joint_ranges[:, 1])
            seeds.append(q)

        print("\nIK seeds being used:")
        for i, s in enumerate(seeds):
            print(f"  seed[{i}]: {s}")

        # Use local IK routine (multi) as a drop-in
        q_goal, ik_err, _ = solve_ik_local_around(
            model, data, ee_sid, arm_q_idx, arm_dof_idx,
            q_seed=q_manual,
            joint_ranges=joint_ranges,
            target_world=target_world,
        )
        waypoints = [q_start.copy(), q_goal.copy()]

    # ------------------------------------------------------------------
    # Piecewise plan through waypoints
    # ------------------------------------------------------------------
    print("\n=== BEGIN PIECEWISE PLANNING ===")
    path = plan_piecewise(
        model=model,
        data=data,
        arm_q_idx=arm_q_idx,
        joint_ranges=joint_ranges,
        forbidden_pairs=forbidden_pairs,
        ignore_gid=ignore_gid,
        pair_pen_thresh_map=pair_pen_thresh_map,
        grip_q_idx=grip_q_idx,
        grip_plan_qpos=GRIPPER_PLAN_QPOS,
        waypoints=waypoints,
    )

    if path is None:
        print("\n[RESULT] Planning failed (some segment could not be connected).")
        return

    # ------------------------------------------------------------------
    # Evaluate EE error at final waypoint
    # ------------------------------------------------------------------
    data.qpos[arm_q_idx]  = path[-1]
    data.qpos[grip_q_idx] = GRIPPER_PLAN_QPOS
    mujoco.mj_forward(model, data)
    ee_final  = np.array(data.site_xpos[ee_sid])
    final_err = np.linalg.norm(target_world - ee_final)

    print("\n=== FINAL PATH SUMMARY ===")
    print(f"  num_waypoints = {len(path)}")
    print(f"  final q_arm   = {path[-1]}")
    print(f"  final EE pos  = {ee_final}")
    print(f"  final EE error to target site = {final_err:.6f} m")

    print("\nSome path waypoints (first, middle, last):")
    print("  q[0]   =", path[0])
    if len(path) > 2:
        print("  q[mid] =", path[len(path) // 2])
    print("  q[-1]  =", path[-1])

    # ------------------------------------------------------------------
    # Save path to CSV for playback
    # ------------------------------------------------------------------
    path_array = np.stack(path, axis=0)
    timestamp  = int(time.time())
    csv_name   = f"cup_grip_path_{timestamp}.csv"
    csv_path   = os.path.join(CURRENT_DIR, csv_name)

    header = ",".join(ARM_JOINT_NAMES)
    np.savetxt(csv_path, path_array, delimiter=",", header=header, comments="")

    print(f"\n[RESULT] Saved joint path to CSV: {csv_path}")


if __name__ == "__main__":
    main()