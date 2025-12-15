import os

# --- Configuration ---
OUTPUT_FILE = "scene.xml"

# We define the UR3 meshes relative to where scene.xml will live (the root folder)
UR3_MESH_DIR = "ur3_flat/"
MICRO_MESH_DIR = "Microwave/meshes/"

xml_content = f"""
<mujoco model="UR3_Microwave_Scene">
    <compiler angle="radian" meshdir="./"/>
    
    <option timestep="0.002" integrator="RK4"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <!-- Floor Texture -->
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance=".2"/>

        <!-- UR3 Meshes (Paths adjusted to point to ur3_flat folder) -->
        <mesh name="ur3_base"     file="{UR3_MESH_DIR}base.stl"/>
        <mesh name="ur3_shoulder" file="{UR3_MESH_DIR}shoulder.stl"/>
        <mesh name="ur3_upperarm" file="{UR3_MESH_DIR}upperarm.stl"/>
        <mesh name="ur3_forearm"  file="{UR3_MESH_DIR}forearm.stl"/>
        <mesh name="ur3_wrist1"   file="{UR3_MESH_DIR}wrist1.stl"/>
        <mesh name="ur3_wrist2"   file="{UR3_MESH_DIR}wrist2.stl"/>
        <mesh name="ur3_wrist3"   file="{UR3_MESH_DIR}wrist3.stl"/>

        <!-- Microwave Meshes (Paths adjusted to point to Microwave folder) -->
        <mesh name="micro_Base"      file="{MICRO_MESH_DIR}Base.stl" scale="0.001 0.001 0.001"/>
        <mesh name="micro_BaseJoint" file="{MICRO_MESH_DIR}BaseJoint.stl" scale="0.001 0.001 0.001"/>
    </asset>

    <worldbody>
        <!-- Lighting -->
        <light pos="0 0 2" dir="0 0 -1" directional="true" castshadow="true"/>
        <light pos="1 1 2" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>

        <!-- Floor -->
        <geom name="floor" size="2 2 .05" type="plane" material="grid" condim="3"/>

        <!-- The Target Marker (Green Sphere) - The robot will try to touch this -->
        <!-- mocap="true" means we can move this object freely with python code -->
        <body name="target" pos="0.4 0 0.3" mocap="true">
            <geom type="sphere" size="0.02" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
            <site name="target_site" rgba="0 1 0 1"/>
        </body>

        <!-- MICROWAVE (Placed in front of robot) -->
        <body name="Microwave" pos="0.5 0 0" euler="0 0 3.1415"> 
            <!-- Rotated 180 degrees (pi) to face the robot -->
            <geom type="mesh" mesh="micro_Base" rgba="0.8 0.8 0.8 1"/>
            <body name="Microwave_Door" pos="0 0 0">
                 <!-- Adjust axis/pos based on your original XML if needed -->
                <joint name="door_joint" type="hinge" axis="0 0 1" pos="-0.377 0 0.165"/>
                <geom type="mesh" mesh="micro_BaseJoint" rgba="0.2 0.2 0.2 1"/>
            </body>
        </body>

        <!-- UR3 ROBOT -->
        <body name="ur3_base" pos="0 0 0">
            <geom type="mesh" mesh="ur3_base" rgba="0.7 0.7 0.7 1"/>
            <body name="shoulder_link" pos="0 0 0.1519">
                <inertial pos="0 0 0" mass="2" diaginertia="0.00809316 0.00809316 0.005625"/>
                <joint name="shoulder_pan_joint" axis="0 0 1" range="-6.28 6.28"/>
                <geom type="mesh" mesh="ur3_shoulder" rgba="0.7 0.7 0.7 1"/>
                
                <body name="upper_arm_link" pos="0 0.1198 0" quat="0.707107 0 0.707107 0">
                    <inertial pos="0 0 0.121825" mass="3.42" diaginertia="0.0217285 0.0217285 0.00961875"/>
                    <joint name="shoulder_lift_joint" axis="0 1 0" range="-6.28 6.28"/>
                    <geom type="mesh" mesh="ur3_upperarm" rgba="0.7 0.7 0.7 1"/>
                    
                    <body name="forearm_link" pos="0 -0.0925 0.24365">
                        <inertial pos="0 0 0.106625" mass="1.26" diaginertia="0.00654681 0.00654681 0.00354375"/>
                        <joint name="elbow_joint" axis="0 1 0" range="-3.14 3.14"/>
                        <geom type="mesh" mesh="ur3_forearm" rgba="0.7 0.7 0.7 1"/>
                        
                        <body name="wrist_1_link" pos="0 0 0.21325" quat="0.707107 0 0.707107 0">
                            <inertial pos="0 0 0" quat="0.5 0.5 -0.5 0.5" mass="0.8" diaginertia="0.002 0.002 0.002"/>
                            <joint name="wrist_1_joint" axis="0 1 0" range="-6.28 6.28"/>
                            <geom type="mesh" mesh="ur3_wrist1" rgba="0.7 0.7 0.7 1"/>
                            
                            <body name="wrist_2_link" pos="0 0.08505 0">
                                <inertial pos="0 0 0" quat="0.5 0.5 -0.5 0.5" mass="0.8" diaginertia="0.002 0.002 0.002"/>
                                <joint name="wrist_2_joint" axis="0 0 1" range="-6.28 6.28"/>
                                <geom type="mesh" mesh="ur3_wrist2" rgba="0.7 0.7 0.7 1"/>
                                
                                <body name="wrist_3_link" pos="0 0 0.08535">
                                    <inertial pos="0 0 0" quat="0.5 0.5 -0.5 0.5" mass="0.35" diaginertia="0.001 0.001 0.001"/>
                                    <joint name="wrist_3_joint" axis="0 1 0" range="-6.28 6.28"/>
                                    <geom type="mesh" mesh="ur3_wrist3" rgba="0.7 0.7 0.7 1"/>
                                    
                                    <!-- End Effector Site: This is the specific point we will control -->
                                    <site name="ee_site" pos="0 0.1 0" size="0.01" rgba="1 0 0 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>
"""

with open(OUTPUT_FILE, "w") as f:
    f.write(xml_content)

print(f"✅ Generated {OUTPUT_FILE} successfully!")