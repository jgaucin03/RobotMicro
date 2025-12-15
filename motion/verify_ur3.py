import mujoco
import numpy as np
import os

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# XML_PATH = os.path.join(CURRENT_DIR, "../scenes/scene_debug.xml")
XML_PATH = os.path.join(CURRENT_DIR, "../ur3/UR3.xml")

# Standard UR3 (CB3) DH Lengths (Meters)
# Source: Universal Robots Specification
UR3_SPECS = {
    "Upper Arm (Shoulder->Elbow)": 0.24365,
    "Forearm   (Elbow->Wrist1)":   0.21325,
    "Wrist 1   (Wrist1->Wrist2)":  0.11235, # This is the main diff vs UR3e (which is 0.131)
    "Wrist 2   (Wrist2->Wrist3)":  0.08535
}

def main():
    if not os.path.exists(XML_PATH):
        print("XML not found.")
        return

    print(f"Loading Model: {XML_PATH}")
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    print("\n--- KINEMATIC VERIFICATION ---")
    
    # Get Joint Anchor positions (The center of rotation in World Space)
    # Note: These names must match your XML joint names
    joints = ["shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    positions = {}
    
    for jname in joints:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid == -1:
            print(f"Error: Could not find joint '{jname}'")
            return
        # xanchor gives the global position of the joint center
        positions[jname] = d.xanchor[jid]

    # Calculate actual lengths in your model
    # Distance formula: norm(pos_b - pos_a)
    
    # 1. Upper Arm
    len_upper = np.linalg.norm(positions["elbow"] - positions["shoulder_lift"])
    
    # 2. Forearm
    len_fore  = np.linalg.norm(positions["wrist_1"] - positions["elbow"])
    
    # 3. Wrist 1 Length (Distance between Wrist 1 and Wrist 2 axes)
    # Note: In UR robots, Wrist 1 and Wrist 2 are offset by d4 along the rotation axis of Wrist 1
    len_w1    = np.linalg.norm(positions["wrist_2"] - positions["wrist_1"])
    
    # 4. Wrist 2 Length
    len_w2    = np.linalg.norm(positions["wrist_3"] - positions["wrist_2"])

    measured = {
        "Upper Arm (Shoulder->Elbow)": len_upper,
        "Forearm   (Elbow->Wrist1)":   len_fore,
        "Wrist 1   (Wrist1->Wrist2)":  len_w1,
        "Wrist 2   (Wrist2->Wrist3)":  len_w2
    }

    # Compare
    all_passed = True
    print(f"{'Segment':<30} | {'XML Value':<10} | {'Standard UR3':<12} | {'Diff':<10}")
    print("-" * 70)
    
    for name, standard_val in UR3_SPECS.items():
        actual_val = measured[name]
        diff = abs(actual_val - standard_val)
        status = "OK" if diff < 0.001 else "FAIL" # 1mm tolerance
        if diff >= 0.001: all_passed = False
        
        print(f"{name:<30} | {actual_val:.5f} m | {standard_val:.5f} m    | {diff:.5f} ({status})")

    print("-" * 70)
    
    if all_passed:
        print("\n✅ SUCCESS: Your XML matches the Standard UR3 kinematics.")
        print("   You can safely use the 'ur3_kinematics.py' closed-form solver.")
    else:
        print("\n❌ WARNING: Mismatch detected.")
        print("   If 'Wrist 1' is ~0.131, you have a UR3e model, not a UR3.")
        print("   If other values are off, your XML <body pos=...> offsets are wrong.")

if __name__ == "__main__":
    main()