import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")

def main():
    print(f"Loading: {XML_PATH}")
    if not os.path.exists(XML_PATH):
        print("XML not found!")
        return

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    # Initial Pose
    start_pose = [0, -1.57, 1.57, -1.57, -1.57, 0]
    d.qpos[:6] = start_pose
    d.ctrl[:6] = start_pose
    
    # 1. PRINT INITIAL CONTACTS (Before physics even starts)
    mujoco.mj_forward(m, d)
    print(f"\n--- INITIAL STATIC CONTACTS ({d.ncon} detected) ---")
    for i in range(d.ncon):
        con = d.contact[i]
        g1 = m.geom(con.geom1).name
        g2 = m.geom(con.geom2).name
        dist = con.dist
        print(f"  Contact {i}: {g1} <--> {g2} (Dist: {dist:.5f})")
        if dist < -0.001:
            print(f"     [CRITICAL WARNING] Deep overlap detected!")

    print("\n--- STARTING SLOW-MOTION SIMULATION ---")
    
    # We will step manually without the viewer first to catch the exact moment of explosion
    for step in range(20):
        try:
            mujoco.mj_step(m, d)
        except Exception as e:
            print(f"Sim crashed at step {step}: {e}")
            break

        # Check for huge accelerations
        max_acc = np.max(np.abs(d.qacc))
        max_vel = np.max(np.abs(d.qvel))
        
        print(f"Step {step}: Max Acc={max_acc:.2f}, Max Vel={max_vel:.2f}")

        # If things are exploding, find out why
        if max_acc > 1000 or max_vel > 100:
            print(f"\n[EXPLOSION DETECTED AT STEP {step}]")
            
            # Find which joint is exploding
            bad_dof = np.argmax(np.abs(d.qacc))
            print(f"  Exploding DOF ID: {bad_dof}")
            
            # Print forces involved
            print("  Active Contacts causing this:")
            for i in range(d.ncon):
                con = d.contact[i]
                # If contact force is huge
                # Note: MuJoCo doesn't store explicit force in d.contact, 
                # we look at the resulting constraint force in d.efc_force if needed, 
                # but distance is usually the smoking gun.
                if con.dist < -0.005: 
                    g1 = m.geom(con.geom1).name
                    g2 = m.geom(con.geom2).name
                    print(f"    VIOLENT OVERLAP: {g1} <--> {g2} (Dist: {con.dist})")
            break

if __name__ == "__main__":
    main()