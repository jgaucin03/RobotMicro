import mujoco
import numpy as np
import os
import glfw
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
# Adjust this to match your file structure relative to this script
XML_PATH = "ur3/UR3.xml" 

# The names of the GEOMS to tune
TARGET_GEOMS = [
    "link1_vis",
    "link2_vis",
    "link3_vis",
    "link4_vis",
    "link5_vis",
    "link6_vis"
]

# --- GLOBAL STATE ---
m = None
d = None
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
perturb = mujoco.MjvPerturb()
context = None
scn = None
current_geom_idx = 0
mouse_last_x = 0
mouse_last_y = 0
button_left = False
button_middle = False
button_right = False
window_width = 1280
window_height = 720

def print_xml_snippet(model, geom_name):
    """Generates the clean XML line for the current geometry state."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    pos = model.geom_pos[gid]
    quat = model.geom_quat[gid]
    
    # Convert MuJoCo quat (w,x,y,z) to Euler (x,y,z)
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler = r.as_euler('xyz', degrees=False)
    
    # Heuristics to guess the original mesh filename based on geom name
    mesh_file = geom_name.replace("_vis", "-v3")
    if "link1" in geom_name: mesh_file = "link1-v4" 

    print(f"\n")
    print(f'<geom name="{geom_name}" type="mesh" mesh="{mesh_file}"')
    print(f'      pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}"')
    print(f'      euler="{euler[0]:.5f} {euler[1]:.5f} {euler[2]:.5f}"')
    print(f'      rgba="0.7 0.7 0.7 1"/>')

def init_window():
    if not glfw.init():
        return None
    window = glfw.create_window(window_width, window_height, "UR3 Ultimate Tuner", None, None)
    glfw.make_context_current(window)
    return window

def mouse_button_callback(window, button, action, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

def mouse_move_callback(window, xpos, ypos):
    global mouse_last_x, mouse_last_y, cam, scn, window_height
    dx = xpos - mouse_last_x
    dy = ypos - mouse_last_y
    mouse_last_x = xpos
    mouse_last_y = ypos

    # Mouse Interaction
    action = mujoco.mjtMouse.mjMOUSE_NONE
    if button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
    elif button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif button_middle:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    if action != mujoco.mjtMouse.mjMOUSE_NONE:
        # FIX: Normalize dx/dy by window height to prevent "million miles an hour" speed
        mujoco.mjv_moveCamera(m, action, dx / window_height, dy / window_height, scn, cam)

def scroll_callback(window, xoffset, yoffset):
    global cam, scn
    mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scn, cam)

def key_callback(window, key, scancode, action, mods):
    global m, current_geom_idx, opt

    if action != glfw.PRESS and action != glfw.REPEAT:
        return

    # GEOMETRY CONTROLS
    name = TARGET_GEOMS[current_geom_idx]
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    
    # Tuning Step Sizes (Hold SHIFT for precision mode?)
    step_pos = 0.001  # 1mm precision
    step_rot = 0.05   # ~3 degrees
    
    # Translation (W/S = X, A/D = Y, Q/E = Z)
    if key == glfw.KEY_W: m.geom_pos[gid][0] += step_pos
    if key == glfw.KEY_S: m.geom_pos[gid][0] -= step_pos
    if key == glfw.KEY_A: m.geom_pos[gid][1] += step_pos
    if key == glfw.KEY_D: m.geom_pos[gid][1] -= step_pos
    if key == glfw.KEY_Q: m.geom_pos[gid][2] += step_pos
    if key == glfw.KEY_E: m.geom_pos[gid][2] -= step_pos

    # Rotation (I/K = Pitch, J/L = Roll, U/O = Yaw)
    cur_quat = m.geom_quat[gid]
    r = R.from_quat([cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0]])
    e = list(r.as_euler('xyz'))
    
    changed = False
    if key == glfw.KEY_I: e[0] += step_rot; changed = True
    if key == glfw.KEY_K: e[0] -= step_rot; changed = True
    if key == glfw.KEY_J: e[1] += step_rot; changed = True
    if key == glfw.KEY_L: e[1] -= step_rot; changed = True
    if key == glfw.KEY_U: e[2] += step_rot; changed = True
    if key == glfw.KEY_O: e[2] -= step_rot; changed = True
    
    if changed:
        new_r = R.from_euler('xyz', e)
        nq = new_r.as_quat()
        m.geom_quat[gid] = [nq[3], nq[0], nq[1], nq[2]]

    # VISUALIZATION TOGGLES
    if key == glfw.KEY_F: # Toggle Coordinate Frames
        opt.frame = mujoco.mjtFrame.mjFRAME_BODY if opt.frame == 0 else 0
    if key == glfw.KEY_T: # Toggle Transparency
        opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = not opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]

    # GENERAL CONTROLS
    if key == glfw.KEY_TAB:
        current_geom_idx = (current_geom_idx + 1) % len(TARGET_GEOMS)
        print(f"Switched to: {TARGET_GEOMS[current_geom_idx]}")
    
    if key == glfw.KEY_SPACE:
        print_xml_snippet(m, name)

def main():
    global m, d, cam, opt, scn, context, window_height, window_width
    
    if not os.path.exists(XML_PATH):
        print(f"XML not found at {XML_PATH}. Check path.")
        return

    # Load Model
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    
    # --- AUTO-LEVITATION ---
    # Lift the base body 0.5m in the viewer so it doesn't phase through the floor
    # We do this by finding the 'ur3_base' and forcing its position.
    # Note: This is purely for VIEWING. It does not affect the 'geom_pos' printout.
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur3_base")
    if base_id != -1:
        m.body_pos[base_id][2] += 0.5
        print("NOTE: Robot lifted 0.5m for easier viewing.")

    # Setup Window & Callbacks
    window = init_window()
    glfw.set_cursor_pos_callback(window, mouse_move_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)
    
    # Get initial window size
    window_width, window_height = glfw.get_framebuffer_size(window)

    # Setup Rendering
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.0
    cam.lookat = np.array([0.0, 0.0, 0.5])
    
    # Enable Frame Visualization by default
    opt.frame = mujoco.mjtFrame.mjFRAME_BODY 
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True # Start transparent

    scn = mujoco.MjvScene(m, maxgeom=10000)
    context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)

    print("\n--- UR3 ULTIMATE VISUAL TUNER ---")
    print("MOUSE: Left-Drag=Rotate, Right-Drag=Pan, Scroll=Zoom")
    print("KEYS : WASDQE (Move Mesh), IJKLXO (Rotate Mesh)")
    print("       TAB (Next Link), SPACE (Print XML)")
    print("       F (Toggle Frames), T (Toggle Transparency)")
    
    while not glfw.window_should_close(window):
        # Update window size for mouse scaling
        window_width, window_height = glfw.get_framebuffer_size(window)
        vp = mujoco.MjrRect(0, 0, window_width, window_height)

        # Enforce Zero Pose (Reset joints every frame to keep reference static)
        d.qpos[:] = 0 
        mujoco.mj_forward(m, d)

        # Update Scene
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        
        # Render
        mujoco.mjr_render(vp, scn, context)
        
        # Text Overlay
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, 
            mujoco.mjtGridPos.mjGRID_TOPLEFT, 
            vp, 
            f"EDITING: {TARGET_GEOMS[current_geom_idx]}", 
            "Move Mesh to match RGB Skeleton", 
            context
        )

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()