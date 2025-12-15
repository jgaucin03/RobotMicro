#!/usr/bin/env python3
# motion/diagnose_path_k.py
#
# Diagnostics script for MuJoCo dynamics when replaying a precomputed UR3 path.
#
# It emulates pick_cup_k.py's logic (open gripper, follow joint path, close
# gripper) but:
#   - runs headless (no viewer),
#   - only for the first N simulation steps,
#   - prints contact forces, joint velocities / accelerations,
#   - flags suspiciously large values and penetrations.
#
# Usage:
#   python diagnose_path_k.py                 # uses default CSV, 200 steps
#   python diagnose_path_k.py mypath.csv 400  # specific CSV, 400 steps

import os
import sys
import numpy as np
import mujoco

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH    = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

# Default CSV path (same convention as pick_cup_k.py)
DEFAULT_CSV_PATH = os.path.join(CURRENT_DIR, "path_latest.csv")

# Names must match your XML
EE_SITE_NAME         = "ee_site"
CUP_GRIP_SITE_NAME   = "cup_grip_target_site"
CUP_TARGET_SITE_NAME = "cup_target_site"   # not strictly needed here

# UR3 arm joints, in the same order as in the CSV
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# Gripper joints (from your setup)
GRIPPER_JOINT_NAMES = [
    "link6_Grip-Driver",
    "link6_Grip-Link",
]

# Approximate "open" / "closed" configuration for the gripper
GRIPPER_OPEN_QPOS   = np.array([0.723, -0.723], dtype=float)
GRIPPER_CLOSED_QPOS = np.array([0.0,   0.0  ], dtype=float)

# -------------------- Playback / physics tuning (match pick_cup_k) -----------

# For diagnostics we keep dt = model.opt.timestep (no additional scaling).
DT_SCALE       = 1.0
OPEN_STEPS     = 80
CLOSE_STEPS    = 80

# Base geometric densification (minimum number of interpolation points per
# CSV segment before adaptive densification kicks in).
STEPS_PER_SEGMENT = 80  # was 50; denser path -> smaller per-step jumps

# We now update the arm *every* sim step. Since per-step deltas are small (due
# to adaptive densification), this is much smoother than big jumps every 4 steps.
CTRL_SUBSTEPS  = 1

# Upper bound on per-step change in any arm joint (radians).
# This is enforced by adaptive densification so that "teleports" cannot be huge.
# Rough intuition: 0.01 rad ~ 0.57 deg. With dt ~ 0.0005, that's ~20 rad/s max
# implied instantaneous velocity from teleporting.
MAX_Q_DELTA_PER_STEP = 0.01

# -------------------- Diagnostics thresholds --------------------------------

VEL_WARN    = 5.0     # rad/s
ACC_WARN    = 50.0    # rad/s^2
FORCE_WARN  = 100.0   # N (tune as needed)
PEN_THRESH  = -1e-3   # "deep" penetration
MAX_CONTACTS_PRINT = 10  # limit per step to avoid spam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_site_id(model, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"Site '{name}' not found in model.")
    return sid


def get_arm_indices(model):
    arm_q_idx = []
    arm_dof_idx = []
    for jn in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Joint '{jn}' not found in model.")

        arm_q_idx.append(model.jnt_qposadr[jid])
        arm_dof_idx.append(model.jnt_dofadr[jid])

    return np.array(arm_q_idx, dtype=int), np.array(arm_dof_idx, dtype=int)


def get_gripper_indices_and_default(model, data):
    grip_q_idx   = []
    grip_dof_idx = []
    for jn in GRIPPER_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Gripper joint '{jn}' not found in model.")

        grip_q_idx.append(model.jnt_qposadr[jid])
        grip_dof_idx.append(model.jnt_dofadr[jid])

    grip_q_idx   = np.array(grip_q_idx, dtype=int)
    grip_dof_idx = np.array(grip_dof_idx, dtype=int)

    mujoco.mj_forward(model, data)
    q_gripper_open = data.qpos[grip_q_idx].copy()

    return grip_q_idx, grip_dof_idx, q_gripper_open


def load_joint_path_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")

    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != len(ARM_JOINT_NAMES):
        raise ValueError(
            f"CSV has {arr.shape[1]} columns, expected {len(ARM_JOINT_NAMES)}."
        )
    return arr


def densify_path(path,
                 steps_per_segment=STEPS_PER_SEGMENT,
                 max_q_delta=MAX_Q_DELTA_PER_STEP):
    """
    Geometric + adaptive densification.

    - steps_per_segment: minimum number of interpolation points per CSV segment.
    - max_q_delta:      adaptive limit on max per-step change in *any* joint.

    For each segment [q0 -> q1], we compute a base number of steps from
    steps_per_segment, then increase that count until the maximum |q1 - q0|
    divided by the number of steps is <= max_q_delta.
    """
    if path.shape[0] == 1:
        return path.copy()

    dense = []
    for i in range(path.shape[0] - 1):
        q0 = path[i]
        q1 = path[i + 1]

        diff = np.max(np.abs(q1 - q0))
        # At least steps_per_segment, but more if needed to keep per-step delta small.
        n_steps = steps_per_segment
        if max_q_delta > 0.0 and diff > 0.0:
            n_steps = max(
                n_steps,
                int(np.ceil(diff / max_q_delta))
            )

        # Interpolate from q0 to q1 with n_steps increments (not including q1).
        for s in np.linspace(0.0, 1.0, n_steps, endpoint=False):
            dense.append((1.0 - s) * q0 + s * q1)

    # Add the final waypoint
    dense.append(path[-1])
    return np.asarray(dense, dtype=float)


def geom_name(model, gid):
    if gid < 0:
        return f"<none:{gid}>"
    nm = model.geom(gid).name
    return nm if nm is not None else f"<geom:{gid}>"


def print_contact_diagnostics(model, data, step_idx):
    ncon = data.ncon
    if ncon == 0:
        print(f"  [step {step_idx}] no contacts")
        return

    print(f"  [step {step_idx}] num_contacts = {ncon}")
    force = np.zeros(6, dtype=np.float64)
    nprint = min(ncon, MAX_CONTACTS_PRINT)

    for i in range(nprint):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)

        g1 = con.geom1
        g2 = con.geom2
        dist = con.dist
        normal_force = force[0]  # along contact normal

        n1 = geom_name(model, g1)
        n2 = geom_name(model, g2)

        flags = []
        if dist < PEN_THRESH:
            flags.append("PENETRATION")
        if abs(normal_force) > FORCE_WARN:
            flags.append("HIGH_FORCE")

        flag_str = f" [{' | '.join(flags)}]" if flags else ""
        print(
            f"    contact[{i}] {n1} <-> {n2}: "
            f"dist={dist:.6e}, fn={normal_force:.3f}{flag_str}"
        )

    if ncon > nprint:
        print(f"    ... {ncon - nprint} more contacts not printed")


def print_joint_diagnostics(data, arm_dof_idx, step_idx, dt):
    qvel = data.qvel[arm_dof_idx].copy()
    qacc = data.qacc[arm_dof_idx].copy()

    max_vel = np.max(np.abs(qvel))
    max_acc = np.max(np.abs(qacc))

    j_max_vel = np.argmax(np.abs(qvel))
    j_max_acc = np.argmax(np.abs(qacc))

    vel_flag = " (HIGH_VEL)" if max_vel > VEL_WARN else ""
    acc_flag = " (HIGH_ACC)" if max_acc > ACC_WARN else ""

    print(
        f"  [step {step_idx}] max |qvel|={max_vel:.3f} rad/s"
        f" on joint {ARM_JOINT_NAMES[j_max_vel]}{vel_flag}, "
        f"max |qacc|={max_acc:.3f} rad/s^2"
        f" on joint {ARM_JOINT_NAMES[j_max_acc]}{acc_flag}"
    )


def report_path_stats(joint_path, dt):
    """
    Print worst-case per-step joint delta & implied "teleport" velocity.
    This is a quick sanity check to see how aggressive the path is.
    """
    if joint_path.shape[0] < 2:
        print("Path stats: single waypoint only (no motion).")
        return

    diffs = np.diff(joint_path, axis=0)
    max_delta = float(np.max(np.abs(diffs)))
    max_vel_est = max_delta / dt if dt > 0.0 else float("inf")

    print("\n--- PATH STATS (after densify) ---")
    print(f"  num steps = {joint_path.shape[0]}")
    print(f"  max per-step |Δq|  = {max_delta:.5f} rad "
          f"({np.degrees(max_delta):.2f} deg)")
    print(f"  implied |qvel| if treated as teleport = {max_vel_est:.2f} rad/s")
    if max_delta > MAX_Q_DELTA_PER_STEP * 1.01:
        print("  WARNING: max per-step delta exceeds MAX_Q_DELTA_PER_STEP;"
              " consider lowering it or increasing STEPS_PER_SEGMENT.")
    else:
        print("  OK: per-step deltas are bounded by MAX_Q_DELTA_PER_STEP.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Parse CLI args
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        csv_path = DEFAULT_CSV_PATH

    if len(sys.argv) >= 3:
        max_steps = int(sys.argv[2])
    else:
        max_steps = 200

    print(f"Using joint path CSV: {csv_path}")
    print(f"Max simulation steps for diagnostics: {max_steps}")

    # Load model and data (we want dt before densifying, so we do this first).
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    original_dt = model.opt.timestep
    model.opt.timestep = original_dt * DT_SCALE
    dt = model.opt.timestep
    print(f"Original dt = {original_dt:.6f}, scaled dt = {dt:.6f}")

    print("DOF index -> joint mapping:")
    for i in range(model.nv):
        j = model.dof_jntid[i]
        jname = model.joint(j).name
        print(f"  DOF {i}: joint {jname}")

    # Load joint path and densify with both a base STEPS_PER_SEGMENT and
    # an adaptive maximum per-step delta.
    joint_path_raw = load_joint_path_from_csv(csv_path)
    joint_path     = densify_path(
        joint_path_raw,
        steps_per_segment=STEPS_PER_SEGMENT,
        max_q_delta=MAX_Q_DELTA_PER_STEP,
    )
    print(
        f"Loaded path with {joint_path_raw.shape[0]} waypoints; "
        f"densified to {joint_path.shape[0]} steps."
    )

    report_path_stats(joint_path, dt)

    # Sites (for debugging if needed)
    ee_sid         = get_site_id(model, EE_SITE_NAME)
    grip_sid       = get_site_id(model, CUP_GRIP_SITE_NAME)
    cup_target_sid = get_site_id(model, CUP_TARGET_SITE_NAME)

    # Indices
    arm_q_idx, arm_dof_idx = get_arm_indices(model)
    grip_q_idx, grip_dof_idx, q_gripper_open_model = get_gripper_indices_and_default(
        model, data
    )

    # Override model-derived "open" with your explicit open pose
    q_gripper_open   = GRIPPER_OPEN_QPOS.copy()
    q_gripper_closed = GRIPPER_CLOSED_QPOS.copy()

    if len(q_gripper_open) != len(q_gripper_closed):
        raise ValueError("Gripper open/closed configuration dimension mismatch.")

    mujoco.mj_forward(model, data)

    # State machine (same as pick_cup_k)
    phase = "open_gripper"
    open_step_count  = 0
    close_step_count = 0
    path_step_idx    = 0
    sim_step_counter = 0

    # For velocity finite-difference baseline
    q_arm_prev  = data.qpos[arm_q_idx].copy()
    q_grip_prev = data.qpos[grip_q_idx].copy()

    print("\n=== BEGIN DIAGNOSTICS ===")
    for step_idx in range(max_steps):
        sim_step_counter += 1
        print(f"\n--- SIM STEP {step_idx} (phase={phase}) ---")

        teleported_arm  = False
        teleported_grip = False

        # -------------------------
        # Phase logic (mirror pick_cup_k)
        # -------------------------
        if phase == "open_gripper":
            alpha = min(1.0, open_step_count / max(1, OPEN_STEPS))
            q_curr   = data.qpos[grip_q_idx].copy()
            q_target = (1.0 - alpha) * q_curr + alpha * q_gripper_open
            data.qpos[grip_q_idx] = q_target
            teleported_grip = True

            open_step_count += 1
            if alpha >= 1.0:
                phase = "follow_path"
                print("[FSM] Transition: open_gripper -> follow_path")

        elif phase == "follow_path":
            if path_step_idx < joint_path.shape[0]:
                if sim_step_counter % CTRL_SUBSTEPS == 0:
                    q_target = joint_path[path_step_idx]
                    data.qpos[arm_q_idx] = q_target
                    teleported_arm = True
                    print(f"  Applied path index {path_step_idx}")
                    path_step_idx += 1
            else:
                phase = "close_gripper"
                print("[FSM] Transition: follow_path -> close_gripper")
                close_step_count = 0

        elif phase == "close_gripper":
            alpha = min(1.0, close_step_count / max(1, CLOSE_STEPS))
            q_target = (1.0 - alpha) * q_gripper_open + alpha * q_gripper_closed
            data.qpos[grip_q_idx] = q_target
            teleported_grip = True

            close_step_count += 1
            if alpha >= 1.0:
                phase = "done"
                print("[FSM] Transition: close_gripper -> done")

        elif phase == "done":
            # Hold pose
            pass

        # -------------------------
        # Velocity consistency / finite difference
        # -------------------------
        q_arm_new  = data.qpos[arm_q_idx].copy()
        q_grip_new = data.qpos[grip_q_idx].copy()

        if teleported_arm:
            # We artificially zero arm velocities when we teleport so that qacc
            # from the physics step reflects the "impulse" needed to enforce
            # constraints, not our finite-difference artifact.
            data.qvel[arm_dof_idx] = 0.0
            q_arm_prev = q_arm_new.copy()
        else:
            data.qvel[arm_dof_idx] = (q_arm_new - q_arm_prev) / dt
            q_arm_prev = q_arm_new.copy()

        if teleported_grip:
            data.qvel[grip_dof_idx] = 0.0
            q_grip_prev = q_grip_new.copy()
        else:
            data.qvel[grip_dof_idx] = (q_grip_new - q_grip_prev) / dt
            q_grip_prev = q_grip_new.copy()

        # Step physics
        mujoco.mj_step(model, data)

        # Hard check for NaN / Inf / huge accelerations
        if not np.isfinite(data.qacc).all():
            print(f"\n*** NON-FINITE qacc detected at step {step_idx}, "
                  f"t={step_idx*dt:.4f} s ***")
            for i in range(model.nv):
                if not np.isfinite(data.qacc[i]):
                    j = model.dof_jntid[i]
                    jname = model.joint(j).name
                    print(f"  DOF {i}: joint {jname}, qacc={data.qacc[i]}")
            break

        HARD_ACC_LIMIT = 1e4
        max_abs_qacc = np.max(np.abs(data.qacc))
        if max_abs_qacc > HARD_ACC_LIMIT:
            print(f"\n*** HUGE qacc detected at step {step_idx}, "
                  f"t={step_idx*dt:.4f} s ***")
            for i in range(model.nv):
                if abs(data.qacc[i]) == max_abs_qacc:
                    j = model.dof_jntid[i]
                    jname = model.joint(j).name
                    print(f"  DOF {i}: joint {jname}, qacc={data.qacc[i]}")
            print_contact_diagnostics(model, data, step_idx)
            break

        # Now qacc and contacts are valid for this step
        print_joint_diagnostics(data, arm_dof_idx, step_idx, dt)
        print_contact_diagnostics(model, data, step_idx)

    print("\n=== END DIAGNOSTICS ===")


if __name__ == "__main__":
    main()