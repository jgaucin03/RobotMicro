#!/usr/bin/env python3
# motion/pick_cup_k.py
#
# Playback script:
# - Loads a precomputed plan CSV.
#   * 6 columns: [arm joints]            -> script will do gripper open/close ramps itself
#   * 7 columns: [arm joints, grip_ctrl] -> script uses embedded gripper ctrl from CSV
#
# - Executes arm path kinematically (teleport arm qpos) while stepping physics.
# - Gripper is actuator-driven ("gripper_servo") and NOT teleported.
#
# Physics is stepped normally; the arm joints are teleported in qpos and we keep
# qvel consistent to avoid injecting fake energy.

import os
import sys
import time
import numpy as np
import mujoco
from mujoco import viewer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH    = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

# Default CSV path (you can rename your generated file to this, or pass via CLI)
DEFAULT_CSV_PATH = os.path.join(CURRENT_DIR, "path_latest.csv")

# Names must match your XML
EE_SITE_NAME         = "ee_site"
CUP_GRIP_SITE_NAME   = "cup_grip_target_site_back"  # only used for debugging
CUP_TARGET_SITE_NAME = "cup_target_site"            # only used for debugging

# UR3 arm joints, in the same order as in the CSV
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# Gripper joints (must match your XML joint names)
GRIPPER_JOINT_NAMES = [
    "link6_Grip-Driver",
    "link6_Grip-Link",
]

# ---------------------------------------------------------------------------
# Actuator names (must match your XML)
# ---------------------------------------------------------------------------

ARM_ACTUATOR_NAMES = [
    "shoulder_pan_servo",
    "shoulder_lift_servo",
    "elbow_servo",
    "wrist_1_servo",
    "wrist_2_servo",
    "wrist_3_servo",
]

GRIPPER_ACTUATOR_NAME = "gripper_servo"

# ---------------------------------------------------------------------------
# Gripper configs (ACTUATOR CTRL targets) used only for 6-col CSV mode
# ---------------------------------------------------------------------------

# You widened ctrlrange to [-0.8, 0.8], so 0.7 is valid.
GRIPPER_OPEN_CTRL   = 0.7
GRIPPER_CLOSED_CTRL = 0.0

# ---------------------------------------------------------------------------
# Playback / physics tuning
# ---------------------------------------------------------------------------

DT_SCALE    = 1.0   # keep MuJoCo dt as-is

# Only used when CSV has 6 cols (arm-only):
OPEN_STEPS  = 80    # steps to open gripper (ctrl ramp)
CLOSE_STEPS = 80    # steps to close gripper (ctrl ramp)

# Only densify 6-col CSVs (planner-produced 7-col CSVs are already dense)
STEPS_PER_SEGMENT = 15   # densify path waypoints -> smooth path

# Only advance along the plan every N physics steps.
# NOTE: if CSV includes gripper_ctrl (7 cols) we will override this to 1 automatically.
CTRL_SUBSTEPS = 4

# Wall-clock slow motion factor. 1.0 ~ real-time. Larger = visually slower.
# This does NOT affect physics dt.
SLOWMO_FACTOR = 0.0

# Diagnostics control
VERBOSE_EVERY_N_STEPS = 1000  # 0 to disable periodic prints

VEL_WARN   = 5.0        # rad/s
ACC_WARN   = 200.0      # rad/s^2
FORCE_WARN = 100.0      # N
PEN_THRESH = -1e-3      # penetration threshold
MAX_CONTACTS_PRINT = 5

# Hard safety threshold for accelerations to detect explosion
HARD_ACC_LIMIT = 1e5    # rad/s^2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_site_id(model, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"Site '{name}' not found in model.")
    return sid


def get_arm_indices(model):
    arm_q_idx   = []
    arm_dof_idx = []
    for jn in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Arm joint '{jn}' not found in model.")
        arm_q_idx.append(model.jnt_qposadr[jid])
        arm_dof_idx.append(model.jnt_dofadr[jid])
    return np.array(arm_q_idx, dtype=int), np.array(arm_dof_idx, dtype=int)


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


def get_actuator_ids(model, actuator_names):
    ids = []
    for an in actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
        if aid < 0:
            raise ValueError(f"Actuator '{an}' not found in model.")
        ids.append(aid)
    return np.array(ids, dtype=int)


def load_plan_from_csv(csv_path):
    """
    Returns:
      arm_path_raw: (N,6)
      grip_ctrl_raw: None OR (N,)
      ncols: int
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")

    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]

    ncols = arr.shape[1]
    if ncols == 6:
        return arr, None, 6
    if ncols == 7:
        return arr[:, :6], arr[:, 6].copy(), 7

    raise ValueError(f"CSV has {ncols} columns, expected 6 or 7.")


def densify_arm_path(path6, steps_per_segment=STEPS_PER_SEGMENT):
    """
    Densify only arm path (6 cols). For 7-col planner outputs, do NOT densify here.
    """
    if path6.shape[0] == 1:
        return path6.copy()

    dense = []
    for i in range(path6.shape[0] - 1):
        q0 = path6[i]
        q1 = path6[i + 1]
        for s in np.linspace(0.0, 1.0, steps_per_segment, endpoint=False):
            dense.append((1.0 - s) * q0 + s * q1)
    dense.append(path6[-1])
    return np.asarray(dense, dtype=float)


def geom_name(model, gid):
    if gid < 0:
        return f"<none:{gid}>"
    nm = model.geom(gid).name
    return nm if nm is not None else f"<geom:{gid}>"


def print_contact_diagnostics(model, data, step_idx):
    ncon = data.ncon
    if ncon == 0:
        print(f"[step {step_idx}] contacts: none")
        return

    print(f"[step {step_idx}] contacts: {ncon}")
    force = np.zeros(6, dtype=np.float64)
    nprint = min(ncon, MAX_CONTACTS_PRINT)
    for i in range(nprint):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)
        n1 = geom_name(model, con.geom1)
        n2 = geom_name(model, con.geom2)
        dist = con.dist
        fn = force[0]

        flags = []
        if dist < PEN_THRESH:
            flags.append("PEN")
        if abs(fn) > FORCE_WARN:
            flags.append("HIGH_FORCE")

        flag_str = f" [{' | '.join(flags)}]" if flags else ""
        print(
            f"  contact[{i}] {n1} <-> {n2}: "
            f"dist={dist:.6e}, fn={fn:.2f}{flag_str}"
        )

    if ncon > nprint:
        print(f"  ... {ncon - nprint} more contacts not printed")


def print_joint_diagnostics(model, data, arm_dof_idx, step_idx):
    qvel = data.qvel[arm_dof_idx].copy()
    qacc = data.qacc[arm_dof_idx].copy()

    max_vel = float(np.max(np.abs(qvel)))
    max_acc = float(np.max(np.abs(qacc)))
    j_max_vel = int(np.argmax(np.abs(qvel)))
    j_max_acc = int(np.argmax(np.abs(qacc)))

    vel_flag = " (HIGH_VEL)" if max_vel > VEL_WARN else ""
    acc_flag = " (HIGH_ACC)" if max_acc > ACC_WARN else ""

    print(
        f"[step {step_idx}] max |qvel|={max_vel:.3f} rad/s on {ARM_JOINT_NAMES[j_max_vel]}{vel_flag}, "
        f"max |qacc|={max_acc:.3f} rad/s^2 on {ARM_JOINT_NAMES[j_max_acc]}{acc_flag}"
    )


def clamp_to_ctrlrange(model, act_id: int, u: float) -> float:
    lo, hi = model.actuator_ctrlrange[act_id]
    return float(np.clip(u, lo, hi))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global CTRL_SUBSTEPS

    # Determine CSV path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = DEFAULT_CSV_PATH

    print(f"Using joint path CSV: {csv_path}")

    arm_raw, grip_ctrl_raw, ncols = load_plan_from_csv(csv_path)

    # If 6-col, densify. If 7-col, assume already dense/time-discretized.
    if ncols == 6:
        arm_path = densify_arm_path(arm_raw, steps_per_segment=STEPS_PER_SEGMENT)
        grip_ctrl = None
        print(f"Loaded arm-only path with {arm_raw.shape[0]} waypoints; densified to {arm_path.shape[0]} steps.")
    else:
        arm_path = arm_raw.copy()
        grip_ctrl = grip_ctrl_raw.copy()
        print(f"Loaded arm+gripper plan with {arm_path.shape[0]} rows (no densify).")
        # IMPORTANT: planner rows already represent control updates; don't stretch them by CTRL_SUBSTEPS=4
        if CTRL_SUBSTEPS != 1:
            print(f"[INFO] CSV has gripper_ctrl; overriding CTRL_SUBSTEPS {CTRL_SUBSTEPS} -> 1 for faithful timing.")
            CTRL_SUBSTEPS = 1

    # Load model and data
    print(f"Loading: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # Optionally scale timestep
    original_dt = model.opt.timestep
    model.opt.timestep = original_dt * DT_SCALE
    dt = model.opt.timestep
    print(f"Original dt = {original_dt:.6f}, scaled dt = {dt:.6f}")

    # Sites
    ee_sid         = get_site_id(model, EE_SITE_NAME)
    grip_sid       = get_site_id(model, CUP_GRIP_SITE_NAME)
    cup_target_sid = get_site_id(model, CUP_TARGET_SITE_NAME)

    # Indices
    arm_q_idx, arm_dof_idx   = get_arm_indices(model)
    grip_q_idx, grip_dof_idx = get_gripper_indices(model)

    # Actuators
    arm_act_ids = get_actuator_ids(model, ARM_ACTUATOR_NAMES)
    grip_act_id = get_actuator_ids(model, [GRIPPER_ACTUATOR_NAME])[0]

    print("\n=== ACTUATOR SETUP ===")
    print("Arm actuators:")
    for n, aid in zip(ARM_ACTUATOR_NAMES, arm_act_ids):
        lo, hi = model.actuator_ctrlrange[aid]
        print(f"  {n}: id={aid}, ctrlrange=[{lo:.3f}, {hi:.3f}]")
    lo, hi = model.actuator_ctrlrange[grip_act_id]
    print(f"Gripper actuator: {GRIPPER_ACTUATOR_NAME}: id={grip_act_id}, ctrlrange=[{lo:.3f}, {hi:.3f}]")

    mujoco.mj_forward(model, data)
    q_grip_initial = data.qpos[grip_q_idx].copy()
    print(f"\nInitial gripper qpos from XML: {q_grip_initial}")

    # Initialize arm controls to current qpos
    data.ctrl[arm_act_ids] = data.qpos[arm_q_idx].copy()

    # Initialize gripper ctrl reasonably (driver joint qpos is index 0 of gripper qpos vector)
    data.ctrl[grip_act_id] = clamp_to_ctrlrange(model, grip_act_id, float(data.qpos[grip_q_idx][0]))
    print(f"Initial gripper ctrl set to driver qpos: {float(data.ctrl[grip_act_id]):.4f}")

    # Gripper ctrl targets (used only in 6-col mode)
    grip_open_u   = clamp_to_ctrlrange(model, grip_act_id, GRIPPER_OPEN_CTRL)
    grip_closed_u = clamp_to_ctrlrange(model, grip_act_id, GRIPPER_CLOSED_CTRL)
    print(f"\nGripper OPEN ctrl target:   {GRIPPER_OPEN_CTRL} (clamped -> {grip_open_u:.4f})")
    print(f"Gripper CLOSED ctrl target: {GRIPPER_CLOSED_CTRL} (clamped -> {grip_closed_u:.4f})")

    # State machine
    if ncols == 7:
        phase = "follow_plan"
    else:
        phase = "open_gripper"

    open_step_count  = 0
    close_step_count = 0
    plan_idx         = 0
    sim_step_counter = 0
    step_idx         = 0

    # For teleport-aware velocity consistency (ARM only)
    q_arm_prev = data.qpos[arm_q_idx].copy()

    print("\n=== BEGIN PICK-CUP PLAYBACK ===")
    print(f"Mode: {'7-col (embedded gripper_ctrl)' if ncols == 7 else '6-col (script ramps gripper)'}")
    print(f"CTRL_SUBSTEPS = {CTRL_SUBSTEPS}")

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            sim_step_counter += 1
            step_idx += 1

            teleported_arm = False

            # ----------------------------------------------------------
            # Phase logic
            # ----------------------------------------------------------
            if phase == "follow_plan":
                if plan_idx < arm_path.shape[0]:
                    if sim_step_counter % CTRL_SUBSTEPS == 0:
                        q_target = arm_path[plan_idx]

                        # Arm teleport + keep servo ctrls aligned
                        data.qpos[arm_q_idx] = q_target
                        data.ctrl[arm_act_ids] = q_target
                        teleported_arm = True

                        # Gripper ctrl from CSV
                        u = float(grip_ctrl[plan_idx])
                        data.ctrl[grip_act_id] = clamp_to_ctrlrange(model, grip_act_id, u)

                        print(f"[PLAN] Applied row {plan_idx} at step {step_idx} (grip_ctrl={float(data.ctrl[grip_act_id]):.4f})")
                        plan_idx += 1
                else:
                    phase = "done"
                    print(f"[FSM] Transition: follow_plan -> done at step {step_idx}")

            elif phase == "open_gripper":
                alpha = min(1.0, open_step_count / max(1, OPEN_STEPS))
                u = (1.0 - alpha) * float(data.ctrl[grip_act_id]) + alpha * grip_open_u
                data.ctrl[grip_act_id] = clamp_to_ctrlrange(model, grip_act_id, u)

                if open_step_count == 0:
                    print(f"[FSM] Starting to open gripper (ctrl) at step {step_idx}")
                open_step_count += 1

                if alpha >= 1.0:
                    phase = "follow_path"
                    print(f"[FSM] Transition: open_gripper -> follow_path at step {step_idx}")
                    print(f"[FSM] Gripper ctrl now open at u = {float(data.ctrl[grip_act_id]):.4f}")

            elif phase == "follow_path":
                # Hold open while moving
                data.ctrl[grip_act_id] = grip_open_u

                if plan_idx < arm_path.shape[0]:
                    if sim_step_counter % CTRL_SUBSTEPS == 0:
                        q_target = arm_path[plan_idx]
                        data.qpos[arm_q_idx] = q_target
                        data.ctrl[arm_act_ids] = q_target
                        teleported_arm = True

                        print(f"[FSM] Applied path index {plan_idx} at step {step_idx}")
                        plan_idx += 1
                else:
                    phase = "close_gripper"
                    close_step_count = 0
                    print(f"[FSM] Transition: follow_path -> close_gripper at step {step_idx}")

            elif phase == "close_gripper":
                alpha = min(1.0, close_step_count / max(1, CLOSE_STEPS))
                u = (1.0 - alpha) * grip_open_u + alpha * grip_closed_u
                data.ctrl[grip_act_id] = clamp_to_ctrlrange(model, grip_act_id, u)

                if close_step_count == 0:
                    print(f"[FSM] Starting to close gripper (ctrl) at step {step_idx}")

                close_step_count += 1
                if alpha >= 1.0:
                    phase = "done"
                    print(f"[FSM] Transition: close_gripper -> done at step {step_idx}")
                    print(f"[FSM] Gripper ctrl now closed at u = {float(data.ctrl[grip_act_id]):.4f}")

            elif phase == "done":
                # Hold last controls
                pass

            # ----------------------------------------------------------
            # Velocity consistency (teleport-aware) - ARM only
            # ----------------------------------------------------------
            q_arm_new = data.qpos[arm_q_idx].copy()
            if teleported_arm:
                data.qvel[arm_dof_idx] = 0.0
                q_arm_prev = q_arm_new.copy()
            else:
                data.qvel[arm_dof_idx] = (q_arm_new - q_arm_prev) / dt
                q_arm_prev = q_arm_new.copy()

            # Step physics
            mujoco.mj_step(model, data)

            # NaN / explosion guard
            if not np.isfinite(data.qacc).all():
                print(f"\n*** NON-FINITE qacc detected at step {step_idx}, t={step_idx*dt:.4f} s ***")
                for i in range(model.nv):
                    if not np.isfinite(data.qacc[i]):
                        j = model.dof_jntid[i]
                        jname = model.joint(j).name
                        print(f"  DOF {i}: joint {jname}, qacc={data.qacc[i]}")
                print("Aborting playback due to unstable simulation.")
                break

            max_abs_qacc = float(np.max(np.abs(data.qacc)))
            if max_abs_qacc > HARD_ACC_LIMIT:
                print(f"\n*** HUGE qacc ({max_abs_qacc:.3e}) at step {step_idx}, t={step_idx*dt:.4f} s ***")
                print_joint_diagnostics(model, data, arm_dof_idx, step_idx)
                print_contact_diagnostics(model, data, step_idx)
                print("Aborting playback due to unstable simulation.")
                break

            # Optional diagnostics every N steps
            if VERBOSE_EVERY_N_STEPS and (step_idx % VERBOSE_EVERY_N_STEPS == 0):
                print_joint_diagnostics(model, data, arm_dof_idx, step_idx)
                print_contact_diagnostics(model, data, step_idx)
                driver_q = float(data.qpos[grip_q_idx][0])
                print(f"[step {step_idx}] gripper ctrl={float(data.ctrl[grip_act_id]):.4f}, driver qpos={driver_q:.4f}")

            # Sync viewer
            v.sync()

            # Slow-motion
            if SLOWMO_FACTOR > 0.0:
                time.sleep(dt * SLOWMO_FACTOR)

    print("=== END PICK-CUP PLAYBACK ===")


if __name__ == "__main__":
    main()