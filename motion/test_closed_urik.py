import numpy as np
from ur_analytic_ik import ur3

def main():
    print("--- UR3 Analytic IK Test ---")

    # 1. Define a Target Pose (4x4 Homogeneous Matrix)
    # This example target is:
    #   Position: X=0.3, Y=0.1, Z=0.3 (in front and slightly left of robot)
    #   Rotation: Identity (Gripper pointing same as base frame, usually sideways)
    # Note: UR3 base frame has Z up. 
    
    # Let's make a rotation that points the gripper DOWN (-Z world)
    # We rotate 180 degrees around Y axis (or X axis)
    # World Z+ is up. End Effector Z+ is "out of gripper".
    # So we want EE Z+ to align with World Z-.
    
    # Rotation Matrix for pointing DOWN:
    # [[ 1,  0,  0],
    #  [ 0, -1,  0],
    #  [ 0,  0, -1]]
    
    ee_pose = np.eye(4)
    ee_pose[0:3, 0:3] = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    ee_pose[0, 3] = 0.3  # X
    ee_pose[1, 3] = 0.1  # Y
    ee_pose[2, 3] = 0.3  # Z

    print("\nTarget End Effector Pose:")
    print(ee_pose)

    # 2. Call the Solver
    # This returns a (N, 6) numpy array of joint solutions
    solutions = ur3.inverse_kinematics(ee_pose)
    
    num_solutions = solutions.shape[0]
    print(f"\nFound {num_solutions} analytical solutions.")

    # 3. Print and Validate Solutions
    if num_solutions == 0:
        print("Error: Target likely out of reach.")
        return

    print("\nValid Joint Configurations (radians):")
    print(f"{'Sol #':<6} {'Base':<8} {'Shoulder':<8} {'Elbow':<8} {'W1':<8} {'W2':<8} {'W3':<8}")
    print("-" * 65)

    for i, q in enumerate(solutions):
        # Format for readability
        q_str = "  ".join([f"{angle:6.2f}" for angle in q])
        print(f"{i:<6} [{q_str}]")

    # 4. Check Forward Kinematics (Verification)
    # Let's take the first solution and plug it back in
    q_test = solutions[0]
    pose_verify = ur3.forward_kinematics(q_test)
    
    print("\n--- Verification (FK of Solution 0) ---")
    pos_diff = np.linalg.norm(pose_verify[0:3, 3] - ee_pose[0:3, 3])
    print(f"Position Error: {pos_diff:.6f} m")
    
    # Check if rotation matches
    # Trace of R_target * R_actual.T should be 3 for perfect match
    r_diff = np.trace(ee_pose[0:3, 0:3] @ pose_verify[0:3, 0:3].T)
    print(f"Rotation Match Score (3.0 is perfect): {r_diff:.6f}")

    if pos_diff < 1e-4 and abs(r_diff - 3.0) < 1e-4:
        print("\nSUCCESS: Solver is accurate.")
    else:
        print("\nWARNING: Verification failed. Check units/conventions.")

if __name__ == "__main__":
    main()