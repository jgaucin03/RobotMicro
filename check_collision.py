import trimesh
import numpy as np

def check_microwave_lite():
    print("Loading meshes...")
    base_mesh = trimesh.load('Microwave/meshes/Base.stl')
    door_mesh = trimesh.load('Microwave/meshes/BaseJoint.stl')

    # 1. Apply MuJoCo scale (0.001) manually
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= 0.001
    base_mesh.apply_transform(scale_matrix)
    door_mesh.apply_transform(scale_matrix)

    # 2. Visualization Colors
    base_mesh.visual.face_colors = [200, 200, 200, 100] 
    door_mesh.visual.face_colors = [0, 0, 255, 150]

    # 3. Check 1: Bounding Box Intersection (Fast pre-check)
    print("\n--- DIAGNOSIS ---")
    # bounds is [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    base_box = base_mesh.bounds
    door_box = door_mesh.bounds
    
    # Check if boxes overlap
    overlap = not (np.any(base_box[0] > door_box[1]) or np.any(door_box[0] > base_box[1]))
    print(f"Bounding Boxes Overlapping? {overlap}")

    # 4. Check 2: Convex Hull Issue (The MuJoCo Killer)
    print("Computing Convex Hulls (this mimics MuJoCo's default behavior)...")
    base_hull = base_mesh.convex_hull
    
    # Check if the door's centroid is inside the Base's "shrink-wrapped" volume
    door_center = door_mesh.centroid
    # .contains relies on ray-tracing, not FCL
    is_inside_hull = base_hull.contains([door_center])[0]
    
    print(f"Is Door Center inside Base Convex Hull? {is_inside_hull}")
    
    if is_inside_hull:
        print("\n*** VERDICT: CONFIRMED ***")
        print("The microwave Base is effectively a SOLID BLOCK in the simulation.")
        print("MuJoCo has 'shrink-wrapped' the base, filling the hollow inside.")
        print("The door is spawning inside this solid volume, causing the explosion.")
    elif overlap:
        print("\n*** VERDICT: GEOMETRY OVERLAP ***")
        print("The Convex Hulls are fine, but the raw meshes are touching.")
        print("You likely need to adjust the hinge position by 1-2mm.")
    else:
        print("\n*** VERDICT: CLEAR ***")
        print("No obvious geometric issues detected. Check Joint definitions.")

    # 5. Visualize
    print("Opening visualizer (Close window to exit)...")
    # We show the Convex Hull of the base to prove the point
    if is_inside_hull:
        base_hull.visual.face_colors = [255, 0, 0, 50] # Red ghost
        (base_hull + door_mesh).show()
    else:
        (base_mesh + door_mesh).show()

if __name__ == "__main__":
    check_microwave_lite()