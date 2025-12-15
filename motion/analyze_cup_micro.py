#!/usr/bin/env python3
import mujoco
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

CUP_GEOM_NAME = "cup_col"
TABLE_GEOM_NAME = "micro_table"
FLOOR_GEOM_NAME = "floor"

def main():
    print(f"Loading model: {XML_PATH}")
    if not os.path.exists(XML_PATH):
        print("XML not found!")
        return

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    mujoco.mj_forward(m, d)

    # --- Lookup geom IDs ---
    def get_geom(name):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            print(f"Geom '{name}' not found.")
        return gid

    cup_gid   = get_geom(CUP_GEOM_NAME)
    table_gid = get_geom(TABLE_GEOM_NAME)
    floor_gid = get_geom(FLOOR_GEOM_NAME)

    print("\n--- GEOM INFO ---")
    np.set_printoptions(precision=6, suppress=True)

    if cup_gid >= 0:
        print(f"\n[{CUP_GEOM_NAME}]")
        print("  type:", m.geom_type[cup_gid])
        print("  size (radius, half-height for cylinder):", m.geom_size[cup_gid])
        print("  local pos:", m.geom_pos[cup_gid])
        print("  world pos (center):", d.geom_xpos[cup_gid])
        print("  world xmat:\n", d.geom_xmat[cup_gid].reshape(3, 3))

        # For cylinder: axis is local z-axis in world frame
        radius, half_h = m.geom_size[cup_gid][0], m.geom_size[cup_gid][1]
        cup_center = d.geom_xpos[cup_gid]
        R_cup = d.geom_xmat[cup_gid].reshape(3, 3)
        axis_world = R_cup[:, 2]  # local z-axis in world frame
        cup_top = cup_center + axis_world * half_h
        cup_bottom = cup_center - axis_world * half_h

        print("  approx top of cylinder:", cup_top)
        print("  approx bottom of cylinder:", cup_bottom)

    if table_gid >= 0:
        print(f"\n[{TABLE_GEOM_NAME}]")
        print("  type:", m.geom_type[table_gid])
        print("  size (box half-extents):", m.geom_size[table_gid])
        print("  local pos:", m.geom_pos[table_gid])
        print("  world pos (center):", d.geom_xpos[table_gid])
        print("  world xmat:\n", d.geom_xmat[table_gid].reshape(3, 3))

        table_center = d.geom_xpos[table_gid]
        table_half_extents = m.geom_size[table_gid].copy()
        # Assuming table local z points "up" (check xmat if uncertain)
        R_table = d.geom_xmat[table_gid].reshape(3, 3)
        table_axis_z = R_table[:, 2]
        table_top = table_center + table_axis_z * table_half_extents[2]
        print("  approx top surface point (center):", table_top)

    if cup_gid >= 0 and table_gid >= 0:
        # Rough vertical gap (assuming z is "up" in world)
        gap_z = cup_bottom[2] - table_top[2]
        print(f"\nEstimated vertical gap (cup_bottom_z - table_top_z): {gap_z:.6f} m")

    if floor_gid >= 0:
        print(f"\n[{FLOOR_GEOM_NAME}]")
        print("  type:", m.geom_type[floor_gid])
        print("  size:", m.geom_size[floor_gid])
        print("  local pos:", m.geom_pos[floor_gid])
        print("  world pos:", d.geom_xpos[floor_gid])
        print("  world xmat:\n", d.geom_xmat[floor_gid].reshape(3, 3))

    # --- Contacts involving cup_col ---
    print("\n--- CONTACTS INVOLVING cup_col ---")
    if cup_gid < 0:
        print("No cup_col geom; skipping contact analysis.")
        return

    if d.ncon == 0:
        print("No contacts at all.")
        return

    for i in range(d.ncon):
        con = d.contact[i]
        g1 = con.geom1
        g2 = con.geom2

        if g1 != cup_gid and g2 != cup_gid:
            continue

        other_gid = g2 if g1 == cup_gid else g1
        other_name = m.geom(other_gid).name

        dist = con.dist
        pos = np.array(con.pos)
        frame = np.array(con.frame).reshape(3, 3)
        normal = frame[:, 0]   # MuJoCo: first column is contact normal

        print(f"\nContact {i}: {CUP_GEOM_NAME} <--> {other_name}")
        print(f"  dist: {dist:.6f} (negative = penetration)")
        print("  contact position (world):", pos)
        print("  contact normal (world):", normal)

        # Project the vector from cup center to contact point onto the normal
        cup_center = d.geom_xpos[cup_gid]
        rel = pos - cup_center
        proj = np.dot(rel, normal)
        print("  (pos - cup_center)·normal:", proj)

        # If you want, you can approximate how far 'inside' each surface is
        # by comparing proj and dist.

    print("\nDone.\n")


if __name__ == "__main__":
    main()