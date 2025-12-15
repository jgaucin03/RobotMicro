import mujoco
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
# Update these paths to point to your actual files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_XML_PATH = os.path.join(CURRENT_DIR, "../ur3_v4/UR3.xml")
NEW_XML_PATH = os.path.join(CURRENT_DIR, "../ur3/UR3.xml")

# Mapping: New Body Name -> Old Body Name containing the mesh
# We also track the joint associated with this body to establish the anchor point
LINK_MAP = [
    # (NewBody, OldBody, JointName, NextJointName)
    ("link1", "link1-v4", "shoulder_pan", "shoulder_lift"),
    ("link2", "link2-v3", "shoulder_lift", "elbow"),
    ("link3", "link3-v3", "elbow", "wrist_1"),
    ("link4", "link4-v3", "wrist_1", "wrist_2"),
    ("link5", "link5-v3", "wrist_2", "wrist_3"),
    ("link6", "link6-v3", "wrist_3", None) # End of chain
]

def get_body_pose(model, data, body_name):
    """Returns global position and rotation matrix of a body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1: raise ValueError(f"Body {body_name} not found.")
    return np.array(data.xpos[bid]), np.array(data.xmat[bid]).reshape(3, 3)

def get_joint_anchor(model, data, joint_name):
    """Returns global position of a joint anchor."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid == -1: raise ValueError(f"Joint {joint_name} not found.")
    return np.array(data.xanchor[jid])

def main():
    print(f"--- Visual Transfer Tool ---")
    print(f"Old Source: {OLD_XML_PATH}")
    print(f"New Target: {NEW_XML_PATH}")

    if not os.path.exists(OLD_XML_PATH) or not os.path.exists(NEW_XML_PATH):
        print("Error: Files not found.")
        return

    # 1. Load OLD Model (Visual Truth)
    m_old = mujoco.MjModel.from_xml_path(OLD_XML_PATH)
    d_old = mujoco.MjData(m_old)
    mujoco.mj_forward(m_old, d_old)

    # 2. Load NEW Model (Kinematic Truth)
    m_new = mujoco.MjModel.from_xml_path(NEW_XML_PATH)
    d_new = mujoco.MjData(m_new)
    mujoco.mj_forward(m_new, d_new)

    print("\n--- GENERATED XML GEOMS ---")
    print("Replace the <geom> lines in your NEW file with these:\n")

    for new_name, old_name, joint_name, next_joint_name in LINK_MAP:
        # A. ANALYSIS OF OLD FILE (Where the mesh should be)
        # In the Old file, the mesh is at the Body Origin.
        # We find the relationship between the Mesh (Old Body) and the Joint Anchor.
        pos_mesh_old, rot_mesh_old = get_body_pose(m_old, d_old, old_name)
        pos_joint_old = get_joint_anchor(m_old, d_old, joint_name)
        
        # Vector from Joint to Mesh in Global Frame
        v_joint_to_mesh_global = pos_mesh_old - pos_joint_old

        # B. ANALYSIS OF NEW FILE (Where the physics is)
        # In the New file, the Body Origin IS the Joint Anchor (DH convention).
        pos_body_new, rot_body_new = get_body_pose(m_new, d_new, new_name)
        
        # C. CALCULATION
        # We want to place the mesh at: NewBodyPos + (JointToMesh Vector)
        # But we must account for the rotation difference between Old and New bodies.
        
        # 1. Calculate Target Global Position of the Mesh
        # We assume the "Vector from Joint to Mesh" should be preserved visually,
        # but rotated to match the new link's orientation if the frame definition changed.
        # actually, simpler: The mesh is a rigid object. We need to attach it to the new frame.
        
        # We calculate the transform required to move from New Body Frame to Old Mesh Frame.
        # This is tricky because the robot might be in different poses (q=0 might differ).
        # HEURISTIC: We align the Joint Axis.
        
        # New approach:
        # Calculate local offset of the mesh relative to the joint in the OLD file.
        # Apply that local offset to the joint in the NEW file, adjusted for frame rotation.
        
        # Since this is hard to generalize programmatically without perfect alignment,
        # we will calculate the RELATIVE transform of the Old Mesh in the Old Joint Frame.
        
        # Old Joint Frame (constructed from global joint axis)
        jid_old = mujoco.mj_name2id(m_old, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        xaxis_old = np.array(d_old.xaxis[jid_old]) # Joint Axis
        # We need a full frame. Let's use the Body's rotation but align Z to axis?
        # Actually, let's use the Body frame directly.
        
        # Transform: Mesh_Global = Body_Old_Global (Since mesh is at 0,0,0 in Old body)
        # Transform: Joint_Old_Global = ...
        
        # Let's try the direct "Inverse Transform" method assuming frames correspond roughly.
        # T_offset = inv(T_new_body) * T_old_mesh
        # This ONLY works if the robot is in the exact same pose.
        # Since we can't guarantee q=0 is the same, we rely on the Joint Anchor being the invariant.
        
        # Vector: Old Joint -> Old Mesh (Global)
        vec_global = pos_mesh_old - pos_joint_old
        
        # We convert this vector into the NEW Body's Local Frame.
        # Local_Pos = inv(Rot_New_Body) * vec_global
        # This assumes the New Body and Old Body are "aligned" rotationally. 
        # They are NOT (DH uses different axes).
        
        # Manual Override based on UR standards + Old XML data:
        # We can extract the exact offset from the Old XML's <joint pos> tag!
        # Old XML: <joint pos="x y z"> -> Joint is at (x,y,z) relative to Mesh.
        # Therefore: Mesh is at (-x, -y, -z) relative to Joint.
        
        j_pos_local_old = m_old.jnt_pos[jid_old] # The 'pos' attribute in XML
        mesh_offset_from_joint = -1 * j_pos_local_old
        
        # Now, rotation.
        # Old Joint Axis in Mesh Frame: m_old.jnt_axis[jid_old]
        # New Joint Axis in Body Frame: (0,0,1) (Always Z for DH)
        
        old_axis = m_old.jnt_axis[jid_old]
        target_axis = np.array([0, 0, 1])
        
        # Find rotation R that maps old_axis to target_axis
        # We want: R * old_axis = target_axis
        # So the Mesh must be rotated by inv(R) to align its internal axis with New Z.
        
        # This aligns the Z-axis (Joint), but the mesh can still spin. 
        # We need to align the "Next Joint" vector to lock the spin.
        if next_joint_name:
            jid_next_old = mujoco.mj_name2id(m_old, mujoco.mjtObj.mjOBJ_JOINT, next_joint_name)
            j_pos_next_local_old = m_old.jnt_pos[jid_next_old] # Next joint in Old Mesh Frame
            
            # Vector from Current Joint to Next Joint (in Old Mesh Frame)
            bone_vec_old = j_pos_next_local_old - j_pos_local_old
            
            # This 'bone_vec_old' represents the arm direction in the mesh.
            # In the NEW DH frame, the arm direction is along the X-axis (usually) or Y.
            # We need to see where the next joint IS in the new frame.
            
            # New Frame Next Joint Pos (Local):
            jid_next_new = mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_JOINT, next_joint_name)
            # The next joint is the origin of the NEXT body.
            # So we get the local pos of the next body in current body frame.
            bid_new = mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_BODY, new_name)
            bid_next_new = mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_BODY, LINK_MAP[LINK_MAP.index((new_name, old_name, joint_name, next_joint_name))+1][0])
            
            p_curr = np.array(m_new.body_pos[bid_new])
            p_next = np.array(m_new.body_pos[bid_next_new]) 
            # This is tricky because p_next is in p_curr's frame? 
            # No, body_pos is in Parent frame.
            # Actually, New XML uses relative coords.
            # The "Next Joint" is at the origin of the "Next Link".
            # The position of "Next Link" in "Current Link" is simply `body_pos`.
            
            bone_vec_target = p_next # Target vector in New Frame
            
            # Optimization: Find Rotation M such that:
            # M * old_axis ~= (0,0,1)
            # M * bone_vec_old ~= bone_vec_target
            
            # We construct coordinate frames and find the transform.
            # Frame A (Old Mesh): Z=old_axis, X=projection of bone_vec
            # Frame B (New Body): Z=(0,0,1), X=projection of bone_vec_target
            
            def make_frame(z_axis, vec_hint):
                z = z_axis / np.linalg.norm(z_axis)
                x = vec_hint - np.dot(vec_hint, z) * z
                if np.linalg.norm(x) < 1e-6: x = np.array([1,0,0]) # Fallback
                x = x / np.linalg.norm(x)
                y = np.cross(z, x)
                return np.column_stack((x, y, z))

            frame_old = make_frame(old_axis, bone_vec_old)
            frame_new = make_frame(target_axis, bone_vec_target)
            
            # We want R such that R * frame_old = frame_new
            # R = frame_new * frame_old^T
            rot_mat = frame_new @ frame_old.T
            
            # The Position offset:
            # We need to take the vector (-joint_pos_old) and rotate it into the new frame
            final_pos = rot_mat @ mesh_offset_from_joint
            
            # Convert to Euler
            r = R.from_matrix(rot_mat)
            euler = r.as_euler('xyz', degrees=False)
            
            # Print Result
            mesh_name = f"{new_name}_vis"
            mesh_file = old_name # The mesh name in asset
            
            # Special manual tweaks for Link 1 (Base) often needed due to Z-rotation
            if new_name == "link1":
                # Link 1 often requires a specific 180 flip in UR robots
                 pass

            print(f'')
            print(f'<geom name="{new_name}_vis" type="mesh" mesh="{old_name}"')
            print(f'      pos="{final_pos[0]:.5f} {final_pos[1]:.5f} {final_pos[2]:.5f}"')
            print(f'      euler="{euler[0]:.5f} {euler[1]:.5f} {euler[2]:.5f}"')
            print(f'      rgba="0.7 0.7 0.7 1"/>')
            
        else:
            # Link 6 (No next joint). Use simpler alignment or copy Link 5's relative logic
            # For Wrist 3, we just align the axis.
            print(f'')
            print(f'<geom name="{new_name}_vis" type="mesh" mesh="{old_name}" pos="0 0 -0.082" euler="1.57 0 0" rgba="0.7 0.7 0.7 1"/>')

if __name__ == "__main__":
    main()