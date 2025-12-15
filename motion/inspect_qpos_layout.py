# motion/inspect_qpos_layout.py
import mujoco
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

def main():
    print(f"Loading model: {XML_PATH}")
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    print("\n--- JOINT ↔ QPOS LAYOUT ---")
    print(f"nq = {m.nq}, nv = {m.nv}, njnt = {m.njnt}\n")

    for jid in range(m.njnt):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        jtype = m.jnt_type[jid]
        qadr  = m.jnt_qposadr[jid]
        vadr  = m.jnt_dofadr[jid]

        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            qspan = "[%d : %d) (7 qpos)" % (qadr, qadr + 7)
            tname = "free "
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            qspan = "[%d : %d) (4 qpos)" % (qadr, qadr + 4)
            tname = "ball "
        else:  # hinge or slide
            qspan = "[%d] (1 qpos)" % qadr
            tname = "hinge" if jtype == mujoco.mjtJoint.mjJNT_HINGE else "slide"

        print(f"  joint {jid:2d}: name='{jname}'  type={tname:5s}  qpos{qspan}  qvel[{vadr}]")

    print("\n--- FIRST 16 qpos (XML default) ---")
    mujoco.mj_forward(m, d)
    np.set_printoptions(precision=6, suppress=True)
    print(d.qpos[:16])

if __name__ == "__main__":
    main()