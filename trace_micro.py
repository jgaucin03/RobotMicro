# RobotMicro/trace_micro.py

import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the scene we just built
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# --- CONFIGURATION ---
SQUARE_CENTER = np.array([0.35, 0.0, 0.3]) # Roughly in front of the microwave
SQUARE_SIZE = 0.15 # 15cm box
SPEED = 0.5 # Radians per second for the path cycle

# Get IDs for the "End Effector" site and the "Target" body
site_id = model.site('ee_site').id
mocap_id = model.body('target').mocapid[0]
dof = model.nv # Degrees of freedom

def get_square_target(t):
    """Returns an XYZ point tracing a square over time."""
    cycle = (t * SPEED) % 4.0
    
    x = SQUARE_CENTER[0]
    y = SQUARE_CENTER[1]
    z = SQUARE_CENTER[2]
    r = SQUARE_SIZE / 2
    
    # Simple logic to trace 4 edges
    if cycle < 1.0: # Bottom edge (Left to Right)
        offset = (cycle - 0.5) * SQUARE_SIZE
        return np.array([x, y + offset, z - r])
    elif cycle < 2.0: # Right edge (Bottom to Top)
        offset = (cycle - 1.5) * SQUARE_SIZE
        return np.array([x, y + r, z + offset])
    elif cycle < 3.0: # Top edge (Right to Left)
        offset = (1.5 - (cycle - 2.0) - 1.0) * SQUARE_SIZE
        return np.array([x, y + offset, z + r])
    else: # Left edge (Top to Bottom)
        offset = (1.5 - (cycle - 3.0) - 1.0) * SQUARE_SIZE
        return np.array([x, y - r, z + offset])

# --- MAIN LOOP ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        t = time.time() - start_time
        
        # 1. Update Target Position (The Green Sphere)
        target_pos = get_square_target(t)
        data.mocap_pos[mocap_id] = target_pos
        
        # 2. INVERSE KINEMATICS (Calculate joint velocities)
        # We want the 'ee_site' to be at 'target_pos'
        
        # Get current EE position and Jacobian
        site_pos = data.site_xpos[site_id]
        jacp = np.zeros((3, dof)) # Position Jacobian
        jacr = np.zeros((3, dof)) # Rotation Jacobian (ignored for now, we just trace position)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        
        # Calculate Error (Delta X)
        error = target_pos - site_pos
        
        # Jacobian Pseudo-Inverse with Damping (DLS)
        # dq = J_pinv * error
        # This tells us how to move joints to reduce the error
        J = jacp # Use only position Jacobian
        lambda_val = 0.1 # Damping factor to prevent jerky movements near singularities
        J_T = J.T
        # Formula: J_T * inv(J * J_T + lambda^2 * I) * error
        dq = J_T @ np.linalg.inv(J @ J_T + lambda_val**2 * np.eye(3)) @ error
        
        # 3. Apply Control
        # We scale dq to make it a velocity command
        q = data.qpos.copy()
        
        # Simple Integration: New Joint Pos = Old + Velocity * Gain
        # We manually set qpos because we aren't using torque actuators here
        data.qpos = q + dq * 1.0 
        
        # Step Physics
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Sleep to keep it viewable
        time.sleep(0.01)