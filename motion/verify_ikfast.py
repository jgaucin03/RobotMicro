import mujoco
import numpy as np
import os
from ur_ikfast import ur_kinematics

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Point this to your FIXED XML (the one with the hardcoded DH params)
XML_PATH = os.path.join(CURRENT_DIR, "../ur3/UR3.xml")

def main():
    # 1. Load MuJoCo Model
    if not os.path.exists(XML_PATH):
        print("XML not found.")
        return
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    
    # 2. Initialize Analytic Solver
    ur3_arm = ur_kinematics.URKinematics('ur3')
    
    print("\n--- UR3 IKFast Verification ---")
    
    # 3. Pick a random valid pose (by using FK on random angles)
    # This ensures the target is actually reachable
    test_angles = [0.5, -1.0, 1.5, -1.0, 1.0, 0.5]
    
    # Calculate Target Pose using IKFast's Math
    # Returns (x, y, z, qx, qy, qz, qw)
    target_pose_quat = ur3_arm.forward(test_angles)
    target_pos = target_pose_quat[:3]
    print(f"Target Position: {target_pos}")
    
    # 4. Set MuJoCo to the same angles
    d.qpos[:6] = test_angles
    mujoco.mj_forward(m, d)
    
    # 5. Get MuJoCo's End Effector Position
    # Note: We need the position of the "ee_site" or "tool0" frame
    # Ensure your XML has a site at the very tip (Wrist 3 + d6 offset)
    ee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    if ee_id == -1:
        print("Error: 'ee_site' not found in XML. Cannot verify.")
        return
        
    mujoco_pos = d.site_xpos[ee_id]
    print(f"MuJoCo Position: {mujoco_pos}")
    
    # 6. Compare
    diff = np.linalg.norm(target_pos - mujoco_pos)
    print(f"Difference:      {diff:.6f} meters")
    
    if diff < 0.001:
        print("\n✅ SUCCESS: XML matches Analytic Solver.")
    else:
        print("\n❌ FAILURE: XML Mismatch.")
        print("   Your XML kinematic chain does not match standard UR3 parameters.")

if __name__ == "__main__":
    main()