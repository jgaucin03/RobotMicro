import os
import shutil
import re
import mujoco
from robot_descriptions.ur3_description import URDF_PATH

# --- CONFIGURATION ---
EXPORT_DIR = "ur3_flat"
OUTPUT_MJCF = "ur3.xml"

# 1. Setup paths
cwd = os.getcwd()
export_path = os.path.join(cwd, EXPORT_DIR)
source_urdf_dir = os.path.dirname(URDF_PATH)
source_package_root = os.path.dirname(os.path.dirname(URDF_PATH)) 
source_mesh_dir = os.path.join(source_package_root, "meshes", "ur3")

print(f"📍 URDF Source: {URDF_PATH}")
print(f"📍 Mesh Source: {source_mesh_dir}")

# 2. Create the Flat Directory
if os.path.exists(export_path):
    shutil.rmtree(export_path)
os.makedirs(export_path)
print(f"📂 Created build directory: {export_path}")

# 3. Copy ALL meshes to the flat directory
# We look in both 'visual' and 'collision' folders
mesh_folders = ["visual", "collision"]
mesh_files_map = {} # Stores 'original_name' -> 'new_flat_name'

print("📦 Flattening meshes...")
for subfolder in mesh_folders:
    search_path = os.path.join(source_mesh_dir, subfolder)
    if not os.path.exists(search_path):
        continue
        
    for filename in os.listdir(search_path):
        if filename.endswith(('.stl', '.dae')):
            src = os.path.join(search_path, filename)
            dst = os.path.join(export_path, filename)
            shutil.copy(src, dst)
            mesh_files_map[filename] = filename
            print(f"   - Copied {filename}")

# 4. Read and Rewrite URDF
with open(URDF_PATH, 'r') as f:
    urdf_content = f.read()

print("🔧 Rewriting URDF to use local filenames...")

# Regex to find any mesh filename attribute and strip the path
# Matches: filename="ANYTHING/base.stl" -> Replaces with: filename="base.stl"
def path_stripper(match):
    full_path = match.group(1)
    filename = os.path.basename(full_path)
    return f'filename="{filename}"'

fixed_urdf = re.sub(r'filename="([^"]+)"', path_stripper, urdf_content)

# 5. Save the patched URDF inside the flat directory
temp_urdf_path = os.path.join(export_path, "ur3_flat.urdf")
with open(temp_urdf_path, "w") as f:
    f.write(fixed_urdf)
print(f"💾 Saved flat URDF: {temp_urdf_path}")

# 6. Load from the FLAT directory
print("🔄 Compiling MJCF...")
try:
    # Load the URDF from the folder where the meshes now live
    model = mujoco.MjModel.from_xml_path(temp_urdf_path)
    
    # Save the final MJCF back to your root (or keep in flat dir)
    output_path = os.path.join(cwd, OUTPUT_MJCF)
    mujoco.mj_saveLastXML(output_path, model)
    
    print("---------------------------------------------------")
    print(f"🎉 SUCCESS! MJCF saved to: {output_path}")
    print("---------------------------------------------------")
    print("Run this to view:")
    print(f"python -m mujoco.viewer --mjcf={OUTPUT_MJCF}")
    
except Exception as e:
    print(f"❌ Conversion failed: {e}")