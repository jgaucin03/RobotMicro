# motion/precheck.py
import mujoco
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

MAX_STEPS   = 10
ACC_THRESH  = 1e3   # tune
VEL_THRESH  = 1e2
PEN_THRESH  = -0.005

# The 6 UR3 joints (from your qpos layout)
UR3_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

def print_contacts(m, d):
    min_dist = None
    print(f"  [Contacts] {d.ncon} active:")
    for i in range(d.ncon):
        con = d.contact[i]
        name1 = m.geom(con.geom1).name or "<unnamed>"
        name2 = m.geom(con.geom2).name or "<unnamed>"
        if min_dist is None or con.dist < min_dist:
            min_dist = con.dist
        flag = ""
        if con.dist < PEN_THRESH:
            flag = "  <-- DEEP PENETRATION"
        print(f"    {i}: {name1} <--> {name2}  dist={con.dist:.6f}{flag}")
    if min_dist is not None:
        print(f"  Min contact distance this step: {min_dist:.6f}")
    print()
    return min_dist

def set_ur3_start_pose(m, d, start_pose):
    """
    Set only the UR3 joints, by name, leaving door/button/cup/gripper as-is.
    """
    assert len(start_pose) == len(UR3_JOINT_NAMES)
    for angle, jname in zip(start_pose, UR3_JOINT_NAMES):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"UR3 joint '{jname}' not found in model")

        qadr = m.jnt_qposadr[jid]
        # All UR3 joints are hinges -> 1 qpos each
        d.qpos[qadr] = angle

def sync_ur3_ctrl_to_qpos(m, d):
    """
    Set actuator ctrl for UR3 joints to their current qpos, so any PD actuators
    don't try to yank the arm somewhere else during the precheck.
    """
    if m.nu == 0:
        return  # no actuators

    # actuator_trnid[actuator, 0] is the joint id
    for jname in UR3_JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue

        qadr = m.jnt_qposadr[jid]
        qval = d.qpos[qadr]

        found = False
        for aid in range(m.nu):
            if m.actuator_trnid[aid, 0] == jid:
                d.ctrl[aid] = qval
                found = True

        if not found:
            # Not fatal; just means that joint isn't directly actuated
            print(f"[WARN] No actuator found for UR3 joint '{jname}' (jid={jid})")

def run_precheck():
    print(f"Loading: {XML_PATH}")
    if not os.path.exists(XML_PATH):
        print("XML not found!")
        return False

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    # optional: set a smaller timestep here
    # m.opt.timestep = 0.0005

    # Start pose for UR3 only (6 DOFs)
    start_pose = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

    # IMPORTANT: only modify UR3 joints, not qpos[:6]
    set_ur3_start_pose(m, d, start_pose)

    # Sync UR3 actuators (if present) to avoid large PD errors
    sync_ur3_ctrl_to_qpos(m, d)

    # Recompute kinematics & contacts
    mujoco.mj_forward(m, d)

    # Debug: where is the cup now?
    cup_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_col")
    if cup_gid >= 0:
        cup_pos = d.geom_xpos[cup_gid]
        print("\n[DEBUG] cup_col world_pos after start_pose:", cup_pos)

    print("\n--- INITIAL STATIC CONTACTS ---")
    print_contacts(m, d)

    print("--- PRECHECK STEPS ---\n")

    for step in range(MAX_STEPS):
        try:
            mujoco.mj_step(m, d)
        except Exception as e:
            print(f"[FAIL] Simulation crashed at step {step}: {e}")
            return False

        max_acc = float(np.max(np.abs(d.qacc)))
        max_vel = float(np.max(np.abs(d.qvel)))
        print(f"=== STEP {step} ===")
        print(f"  Max |qacc|={max_acc:.3e}, Max |qvel|={max_vel:.3e}")
        min_dist = print_contacts(m, d)

        bad_dyn = (max_acc > ACC_THRESH or max_vel > VEL_THRESH)
        bad_pen = (min_dist is not None and min_dist < PEN_THRESH)

        if bad_dyn or bad_pen:
            print(f"[FAIL] Explosion / deep penetration detected at step {step}")
            return False

    print("[PASS] Precheck completed without explosions.")
    return True


if __name__ == "__main__":
    ok = run_precheck()
    if not ok:
        print("Precheck failed. Aborting main simulation.")
    else:
        print("Precheck passed. Safe to run main controller.")