# motion/analyze_cup_micro_contacts.py
import mujoco
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

def main():
    print(f"Loading model: {XML_PATH}")
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    # IDs for geoms of interest
    cup_col_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_col")
    micro_table_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "micro_table")
    floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    print("\n--- GEOM POSITIONS ---")
    for gid, label in [(cup_col_gid, "cup_col"),
                       (micro_table_gid, "micro_table"),
                       (floor_gid, "floor")]:
        pos = d.geom_xpos[gid]
        print(f"{label}: world_pos={pos}")

    print("\n--- CONTACTS BEFORE STEP ---")
    print_contacts_for_geom(m, d, cup_col_gid)

    # One physics step
    mujoco.mj_step(m, d)

    print("\n--- CONTACTS AFTER 1 STEP ---")
    print_contacts_for_geom(m, d, cup_col_gid)

def print_contacts_for_geom(m, d, target_gid):
    found = False
    for i in range(d.ncon):
        con = d.contact[i]
        if con.geom1 == target_gid or con.geom2 == target_gid:
            found = True
            g1 = m.geom(con.geom1).name or "<unnamed>"
            g2 = m.geom(con.geom2).name or "<unnamed>"
            print(f"  Contact {i}: {g1} <--> {g2}")
            print(f"    dist={con.dist:.6f}")
            print(f"    normal={con.frame[0:3]}")
            print(f"    pos={con.pos}")
    if not found:
        print("  (No contacts involving cup_col)")

if __name__ == "__main__":
    main()