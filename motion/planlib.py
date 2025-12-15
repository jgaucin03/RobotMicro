#!/usr/bin/env python3
# motion/planlib.py
#
# Common planning utilities for UR3 + gripper.
# - Name->index helpers
# - Simple DLS IK to match a site position (translation only)
# - Velocity-limited retiming of joint waypoints at a fixed control rate
# - Plan compilation (6 arm joints + gripper_ctrl) -> CSV

import os
import numpy as np
import mujoco


def mj_id(model, objtype, name: str) -> int:
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise ValueError(f"Name not found: {name}")
    return i


def get_joint_qpos_dof_indices(model, joint_names):
    qpos_idx = []
    dof_idx = []
    for jn in joint_names:
        jid = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qpos_idx.append(model.jnt_qposadr[jid])
        dof_idx.append(model.jnt_dofadr[jid])
    return np.asarray(qpos_idx, dtype=int), np.asarray(dof_idx, dtype=int)


def get_actuator_ids(model, actuator_names):
    ids = []
    for an in actuator_names:
        aid = mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
        ids.append(aid)
    return np.asarray(ids, dtype=int)

def append_time_column(plan, ctrl_dt, start_time=0.0):
    plan = np.asarray(plan, dtype=float)
    t = (start_time + np.arange(plan.shape[0], dtype=float) * float(ctrl_dt))[:, None]
    return np.hstack([plan, t]), t[:, 0]

def collision_summary(model, data, pen_thresh=-1e-3):
    # After mj_forward, contacts are populated.
    ncon = data.ncon
    worst = 0.0
    worst_pair = None
    for i in range(ncon):
        c = data.contact[i]
        if c.dist < worst:
            worst = c.dist
            worst_pair = (c.geom1, c.geom2)
    ok = (worst >= pen_thresh)
    return ok, ncon, worst, worst_pair


def unwrap_waypoints(waypoints, discont=np.pi):
    """
    Ensures continuity across +/-pi by unwrapping each joint trajectory.
    This is what prevents "3.14 -> -3.13" jumps.
    """
    W = np.asarray(waypoints, dtype=float).copy()
    # unwrap along the waypoint axis (axis=0), per joint column
    W = np.unwrap(W, axis=0, discont=discont)
    return W


def smoothstep(u):
    # C1 smooth (zero velocity at ends)
    return 3*u*u - 2*u*u*u


def retime_segment(q0, q1, ctrl_dt, vmax, use_smoothstep=True):
    """
    Create intermediate setpoints from q0->q1 such that per-joint velocity <= vmax.
    Returns an array of shape (K,6) INCLUDING q0 but EXCLUDING q1 (so segments stitch cleanly).
    """
    dq = q1 - q0
    vmax = np.asarray(vmax, dtype=float)
    vmax = np.maximum(vmax, 1e-6)

    # duration required so that max(|dq_j|/T) <= vmax_j
    T = float(np.max(np.abs(dq) / vmax))
    if T < 1e-9:
        return q0[None, :]

    steps = int(np.ceil(T / ctrl_dt))
    steps = max(1, steps)

    out = []
    for k in range(steps):
        u = k / steps
        s = smoothstep(u) if use_smoothstep else u
        out.append(q0 + s * dq)
    return np.asarray(out, dtype=float)


def retime_waypoints(waypoints, ctrl_dt, vmax, use_smoothstep=True):
    """
    Velocity-limited retiming over multiple waypoints.
    Returns (N,6) INCLUDING the final waypoint.
    """
    W = np.asarray(waypoints, dtype=float)
    if W.shape[0] == 1:
        return W.copy()

    traj = []
    for i in range(W.shape[0] - 1):
        seg = retime_segment(W[i], W[i+1], ctrl_dt, vmax, use_smoothstep=use_smoothstep)
        traj.append(seg)
    traj.append(W[-1][None, :])
    return np.vstack(traj)


def ik_to_site_position(
    model,
    data,
    arm_joint_names,
    ee_site_name,
    target_pos,
    q_seed,
    max_steps=200,
    tol=1e-3,
    step_scale=0.8,
    damping=1e-2,
    verbose=True,
):
    """
    Damped least squares IK (translation only) for a site.
    Returns q_best, best_err.
    """
    ee_sid = mj_id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    arm_q_idx, arm_dof_idx = get_joint_qpos_dof_indices(model, arm_joint_names)

    # set seed
    data.qpos[arm_q_idx] = np.asarray(q_seed, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    best_q = data.qpos[arm_q_idx].copy()
    best_err = float("inf")

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    for it in range(max_steps):
        cur = data.site_xpos[ee_sid].copy()
        err = np.asarray(target_pos, dtype=float) - cur
        e = float(np.linalg.norm(err))

        if e < best_err:
            best_err = e
            best_q = data.qpos[arm_q_idx].copy()

        if verbose:
            print(f"  [IK] step {it:03d}: err={e:.6f}, q_arm={data.qpos[arm_q_idx]}")

        if e <= tol:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, ee_sid)
        J = jacp[:, arm_dof_idx]  # (3,6)

        # DLS: dq = J^T (J J^T + λ^2 I)^-1 err
        A = J @ J.T + (damping**2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)

        # apply
        q = data.qpos[arm_q_idx].copy()
        q_new = q + step_scale * dq
        data.qpos[arm_q_idx] = q_new
        data.qvel[arm_dof_idx] = 0.0
        mujoco.mj_forward(model, data)

    # return best
    data.qpos[arm_q_idx] = best_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return best_q, best_err


def write_plan_csv(csv_path, plan, header):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        np.savetxt(f, plan, delimiter=",", fmt="%.8f")