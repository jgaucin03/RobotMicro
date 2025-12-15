#!/usr/bin/env python3
# motion/pick_cup.py
#
# Robot-real playback:
# - Loads a plan CSV:
#     * 7-col legacy: [q1..q6, gripper_ctrl]
#     * 8-col new:    [q1..q6, gripper_ctrl, t]   <-- matches your planner_numerical.py
# - Drives arm and gripper via actuators (data.ctrl), NO qpos teleporting
# - Uses the CSV time column if present to schedule setpoint updates
# - Physics steps at sim dt
# - Optional slew-rate limiting to enforce vmax at runtime too
# - Verbose diagnostics and contact prints
#
# NEW:
# - DEBUG_DOOR_JOINT_OPEN: if nonzero, sets door_joint to that angle BEFORE arm playback,
#   by writing qpos and stepping a few frames to settle.

import os
import sys
import time
import numpy as np
import mujoco
from mujoco import viewer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")
DEFAULT_CSV_PATH = os.path.join(CURRENT_DIR, "path_latest_with_gripper.csv")

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

ARM_ACTUATOR_NAMES = [
    "shoulder_pan_servo",
    "shoulder_lift_servo",
    "elbow_servo",
    "wrist_1_servo",
    "wrist_2_servo",
    "wrist_3_servo",
]

GRIPPER_ACTUATOR_NAME = "gripper_servo"

# ----------------------------
# DEBUG / "FUNNY SILLY THING"
# ----------------------------
# If nonzero, we will set the microwave door joint to this value BEFORE arm starts moving,
# so it doesn't interfere with the robot.
#
# Example: open door by -90 degrees:
DEBUG_DOOR_JOINT_OPEN = -1.57
DEBUG_DOOR_JOINT_NAME = "door_joint"
DEBUG_DOOR_SETTLE_STEPS = 50  # physics steps to settle after moving door qpos

# Fallback control rate (only used if CSV has no time column)
CTRL_HZ_FALLBACK = 100.0
CTRL_DT_FALLBACK = 1.0 / CTRL_HZ_FALLBACK

# Safety: runtime slew rate limits (rad/s)
V_MAX = np.array([0.6, 0.6, 0.7, 1.2, 1.2, 1.5], dtype=float)

# Diagnostics
VERBOSE_EVERY_N_STEPS = 1000
VEL_WARN = 2.0
ACC_WARN = 50.0
FORCE_WARN = 100.0
PEN_THRESH = -1e-3
MAX_CONTACTS_PRINT = 6
HARD_ACC_LIMIT = 1e5

# Make playback match real-time wall clock (optional)
REALTIME = True
SLOWMO_FACTOR = 1.0  # 1.0 = real-time, 2.0 = 2x slower, etc.


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


def load_plan_csv(csv_path):
    """
    Supports:
      - 7 cols: q1..q6, grip
      - 8 cols: q1..q6, grip, t   (t is last column)  <-- matches your planner
    Returns:
      arm_plan: (N,6)
      grip_plan: (N,)
      t_plan: (N,) or None
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")

    # Try loading with header skip. If file has blank lines, loadtxt can choke; handle that.
    try:
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV {csv_path}: {e}")

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.shape[1] == 7:
        arm = arr[:, 0:6].astype(float)
        grip = arr[:, 6].astype(float)
        t = None
        return arm, grip, t

    if arr.shape[1] == 8:
        arm = arr[:, 0:6].astype(float)
        grip = arr[:, 6].astype(float)
        t = arr[:, 7].astype(float)  # time is LAST column
        return arm, grip, t

    raise ValueError(
        f"CSV has {arr.shape[1]} columns; expected 7 (q1..q6,grip) or 8 (q1..q6,grip,t)."
    )


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
        print(f"  contact[{i}] {n1} <-> {n2}: dist={dist:.6e}, fn={fn:.2f}{flag_str}")

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


def clamp_ctrlrange_if_any(model, act_id, u):
    lo, hi = model.actuator_ctrlrange[act_id]
    # Some models leave ctrlrange at [0,0] meaning "unspecified" in practice.
    if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):
        return float(np.clip(u, lo, hi))
    return float(u)


def slew_limit(prev_u, target_u, vmax, dt):
    du_max = vmax * dt
    du = target_u - prev_u
    du = np.clip(du, -du_max, du_max)
    return prev_u + du


def compute_plan_dt_from_time(t_plan):
    """
    Returns a robust estimate of plan_dt from time column.
    We use median(diff) to ignore occasional jitter.
    """
    if t_plan is None or len(t_plan) < 2:
        return None
    dt = np.diff(t_plan)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    return float(np.median(dt))


def maybe_open_door_before_playback(model, data):
    """
    If DEBUG_DOOR_JOINT_OPEN is nonzero, set door_joint qpos to that value
    before starting the arm plan. We do this by directly setting qpos (teleport),
    then forward + a few mj_step calls to settle contacts.
    """
    if DEBUG_DOOR_JOINT_OPEN == 0.0:
        print("[DEBUG_DOOR] DEBUG_DOOR_JOINT_OPEN == 0.0, not touching the door.")
        return

    try:
        door_jid = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, DEBUG_DOOR_JOINT_NAME)
    except Exception as e:
        print(f"[DEBUG_DOOR] Could not find joint '{DEBUG_DOOR_JOINT_NAME}': {e}")
        print("[DEBUG_DOOR] Skipping door debug.")
        return

    qadr = int(model.jnt_qposadr[door_jid])
    dofadr = int(model.jnt_dofadr[door_jid])
    jtype = int(model.jnt_type[door_jid])

    # Sanity: expect hinge (1 dof) for microwave door.
    # We won't hard-fail if it isn't; we will still attempt to set qpos[qadr].
    print("\n=== DEBUG_DOOR: PRE-OPEN MICROWAVE DOOR ===")
    print(f"[DEBUG_DOOR] joint='{DEBUG_DOOR_JOINT_NAME}' id={door_jid} jtype={jtype} qposadr={qadr} dofadr={dofadr}")
    print(f"[DEBUG_DOOR] setting qpos[{qadr}] from {float(data.qpos[qadr]):.6f} -> {float(DEBUG_DOOR_JOINT_OPEN):.6f}")

    data.qpos[qadr] = float(DEBUG_DOOR_JOINT_OPEN)

    # Forward once and then settle a few physics steps
    mujoco.mj_forward(model, data)

    # Optional: zero velocities on that dof to avoid a snap impulse (helps stability)
    if 0 <= dofadr < data.qvel.shape[0]:
        print(f"[DEBUG_DOOR] zeroing qvel[{dofadr}] (was {float(data.qvel[dofadr]):.6f})")
        data.qvel[dofadr] = 0.0

    # Let contacts settle a bit
    for k in range(int(DEBUG_DOOR_SETTLE_STEPS)):
        mujoco.mj_step(model, data)

    print(f"[DEBUG_DOOR] settled for {DEBUG_DOOR_SETTLE_STEPS} steps at sim_dt={float(model.opt.timestep):.6f}s")
    print("=== DEBUG_DOOR: DONE ===\n")


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = DEFAULT_CSV_PATH

    print(f"Using plan CSV: {csv_path}")
    arm_plan, grip_plan, t_plan = load_plan_csv(csv_path)

    if t_plan is None:
        print(f"Loaded plan rows={arm_plan.shape[0]}, cols=7 (6 joints + gripper_ctrl)")
    else:
        print(f"Loaded plan rows={arm_plan.shape[0]}, cols=8 (6 joints + gripper_ctrl + t)")
        print(f"Time range: t[0]={t_plan[0]:.6f} s, t[-1]={t_plan[-1]:.6f} s")

    print(f"Loading: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    sim_dt = float(model.opt.timestep)

    # Determine how often to update control.
    # If time column exists: schedule by "t_plan" (in sim time).
    # Else: fall back to CTRL_DT_FALLBACK and steps_per_ctrl.
    plan_dt = compute_plan_dt_from_time(t_plan)
    if plan_dt is None:
        plan_dt = CTRL_DT_FALLBACK
        using_time = False
    else:
        using_time = True

    steps_per_ctrl = int(round(plan_dt / sim_dt))
    steps_per_ctrl = max(1, steps_per_ctrl)
    eff_ctrl_dt = steps_per_ctrl * sim_dt

    print("\n=== TIMING SETUP ===")
    print(f"sim dt               = {sim_dt:.6f} s")
    if using_time:
        print("control scheduling    = from CSV time column (t is last column)")
        print(f"plan_dt (median diff) = {plan_dt:.6f} s")
    else:
        print("control scheduling    = fallback fixed-rate (CSV had no time col)")
        print(f"CTRL_HZ fallback      = {CTRL_HZ_FALLBACK:.1f} Hz  (CTRL_DT={CTRL_DT_FALLBACK:.6f} s)")
        print(f"plan_dt (fallback)    = {plan_dt:.6f} s")

    print(f"steps_per_ctrl        = {steps_per_ctrl}  => effective ctrl dt = {eff_ctrl_dt:.6f} s")
    if t_plan is not None:
        print(f"approx duration (t)   = {t_plan[-1] - t_plan[0]:.2f} s")
    else:
        print(f"approx duration       = {arm_plan.shape[0] * eff_ctrl_dt:.2f} s")

    arm_q_idx, arm_dof_idx = get_joint_qpos_dof_indices(model, ARM_JOINT_NAMES)
    arm_act_ids = get_actuator_ids(model, ARM_ACTUATOR_NAMES)
    grip_act_id = get_actuator_ids(model, [GRIPPER_ACTUATOR_NAME])[0]

    mujoco.mj_forward(model, data)

    # DEBUG: door pre-open (teleport + settle) before viewer starts + before arm moves
    maybe_open_door_before_playback(model, data)

    print("\n=== ACTUATOR SETUP ===")
    for n, aid in zip(ARM_ACTUATOR_NAMES, arm_act_ids):
        lo, hi = model.actuator_ctrlrange[aid]
        print(f"  {n}: id={aid}, ctrlrange=[{lo:.3f}, {hi:.3f}]")
    lo, hi = model.actuator_ctrlrange[grip_act_id]
    print(f"  {GRIPPER_ACTUATOR_NAME}: id={grip_act_id}, ctrlrange=[{lo:.3f}, {hi:.3f}]")

    # Initialize ctrl to current qpos so no sudden jump
    data.ctrl[arm_act_ids] = data.qpos[arm_q_idx].copy()
    data.ctrl[grip_act_id] = clamp_ctrlrange_if_any(model, grip_act_id, float(data.ctrl[grip_act_id]))
    print(f"\nInitial arm qpos: {data.qpos[arm_q_idx].copy()}")
    print(f"Initial arm ctrl: {data.ctrl[arm_act_ids].copy()}")
    print(f"Initial gripper ctrl: {float(data.ctrl[grip_act_id]):.4f}")

    plan_idx = 0
    step_idx = 0
    ctrl_tick = 0

    # Control scheduling state
    # - If using CSV time: next_ctrl_time is set by t_plan[plan_idx]
    # - else: use steps_per_ctrl ticking
    t_sim = 0.0
    if using_time:
        # Normalize so first row triggers at current sim time
        t0 = float(t_plan[0])
        next_ctrl_time = 0.0  # compare against normalized (t_plan - t0)
        print("\n[CTRL] Using CSV time schedule (normalized to start at 0).")
    else:
        next_ctrl_time = None
        print("\n[CTRL] Using fixed ctrl tick schedule.")

    # For realtime pacing
    t0_wall = time.time()
    t0_sim = 0.0

    print("\n=== BEGIN ROBOT-REAL PLAYBACK (ACTUATOR-DRIVEN) ===")
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            step_idx += 1
            t_sim = step_idx * sim_dt

            # Decide whether to apply the next plan row
            do_apply = False

            if plan_idx >= arm_plan.shape[0]:
                do_apply = False
            else:
                if using_time:
                    # compare to normalized plan time
                    t_norm = t_sim
                    # apply as long as we are past the scheduled time (handles dt mismatch)
                    if t_norm + 1e-12 >= next_ctrl_time:
                        do_apply = True
                else:
                    if (step_idx % steps_per_ctrl) == 0:
                        do_apply = True

            if do_apply and plan_idx < arm_plan.shape[0]:
                ctrl_tick += 1

                q_ref = arm_plan[plan_idx].copy()
                g_ref = float(grip_plan[plan_idx])

                # Slew-rate limit the arm ctrl (extra safety)
                prev_u = data.ctrl[arm_act_ids].copy()
                u_target = q_ref
                u_new = slew_limit(prev_u, u_target, V_MAX, eff_ctrl_dt)
                data.ctrl[arm_act_ids] = u_new

                # Gripper ctrl (clamped if ctrlrange exists)
                data.ctrl[grip_act_id] = clamp_ctrlrange_if_any(model, grip_act_id, g_ref)

                if using_time:
                    # schedule next time
                    t0 = float(t_plan[0])
                    if plan_idx + 1 < len(t_plan):
                        next_ctrl_time = float(t_plan[plan_idx + 1] - t0)
                    else:
                        next_ctrl_time = float("inf")

                    print(
                        f"[PLAN] Applied row {plan_idx} at step {step_idx} "
                        f"(t_sim={t_sim:.4f}s, t_ref={(float(t_plan[plan_idx]-t0)):.4f}s, grip_ctrl={float(data.ctrl[grip_act_id]):.4f})"
                    )
                else:
                    print(
                        f"[PLAN] Applied row {plan_idx} at step {step_idx} "
                        f"(grip_ctrl={float(data.ctrl[grip_act_id]):.4f})"
                    )

                plan_idx += 1

            # Step physics
            mujoco.mj_step(model, data)

            # Stability guards
            if not np.isfinite(data.qacc).all():
                print(f"\n*** NON-FINITE qacc detected at step {step_idx}, t={t_sim:.4f} s ***")
                break

            max_abs_qacc = float(np.max(np.abs(data.qacc)))
            if max_abs_qacc > HARD_ACC_LIMIT:
                print(f"\n*** HUGE qacc ({max_abs_qacc:.3e}) at step {step_idx}, t={t_sim:.4f} s ***")
                print_joint_diagnostics(model, data, arm_dof_idx, step_idx)
                print_contact_diagnostics(model, data, step_idx)
                print("Aborting playback due to unstable simulation.")
                break

            # Periodic diagnostics
            if VERBOSE_EVERY_N_STEPS and (step_idx % VERBOSE_EVERY_N_STEPS == 0):
                print_joint_diagnostics(model, data, arm_dof_idx, step_idx)
                print_contact_diagnostics(model, data, step_idx)
                print(f"[step {step_idx}] arm ctrl={data.ctrl[arm_act_ids].copy()}")
                print(f"[step {step_idx}] arm qpos={data.qpos[arm_q_idx].copy()}")
                print(f"[step {step_idx}] gripper ctrl={float(data.ctrl[grip_act_id]):.4f}")
                if using_time:
                    print(f"[step {step_idx}] next_ctrl_time={next_ctrl_time:.6f} (normalized)")

            v.sync()

            # Realtime pacing (optional)
            if REALTIME:
                t_wall_target = t0_wall + (t_sim - t0_sim) * SLOWMO_FACTOR
                now = time.time()
                if t_wall_target > now:
                    time.sleep(t_wall_target - now)

    print("=== END ROBOT-REAL PLAYBACK ===")


if __name__ == "__main__":
    main()