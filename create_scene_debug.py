# RobotMicro/create_scene_debug.py

import os

OUTPUT_FILE = "scene_debug.xml"

# --- CONFIGURATION ---
UR3_FOLDER = "ur3/meshes/" 

xml_content = f"""
<mujoco model="UR3_Gripper_Debug_Scene">
    <compiler angle="radian" meshdir="./" balanceinertia="true"/>
    
    <option timestep="0.002" integrator="RK4"/>
    
    <default>
        <geom solimp="0.99 0.99 0.001" solref="0.001 1" margin="0.001" friction="1 0.005 0.0001"/>
        <joint armature="0.01" damping="1" stiffness="0"/>
    </default>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance=".2"/>
        <material name="table_mat" rgba="0.6 0.4 0.2 1"/>
        <material name="gripper_mat" rgba="0.2 0.2 0.2 1"/>

        <mesh name="base_link-v3" file="ur3_v2/meshes/base_link-v3.stl" scale="0.001 0.001 0.001"/>
        <mesh name="link1-v4"     file="ur3_v2/meshes/link1-v4.stl"     scale="0.001 0.001 0.001"/>
        <mesh name="link2-v3"     file="ur3_v2/meshes/link2-v3.stl"     scale="0.001 0.001 0.001"/>
        <mesh name="link3-v3"     file="ur3_v2/meshes/link3-v3.stl"     scale="0.001 0.001 0.001"/>
        <mesh name="link4-v3"     file="ur3_v2/meshes/link4-v3.stl"     scale="0.001 0.001 0.001"/>
        <mesh name="link5-v3"     file="ur3_v2/meshes/link5-v3.stl"     scale="0.001 0.001 0.001"/>
        <mesh name="link6-v3"     file="ur3_v2/meshes/link6-v3.stl"     scale="0.001 0.001 0.001"/>
        
        <mesh name="Left_Grip" file="{UR3_FOLDER}Left_Grip.stl" scale="0.001 0.001 0.001"/>
        <mesh name="Right_Grip" file="{UR3_FOLDER}Right_Grip.stl" scale="0.001 0.001 0.001"/>
    </asset>

    <worldbody>
        <light pos="0 0 2" dir="0 0 -1" directional="true" castshadow="true"/>
        <light pos="1 1 2" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>
        <geom name="floor" size="2 2 .05" type="plane" material="grid" condim="3"/>

        <geom name="table_base" type="box" size="0.4 0.3 0.01" pos="0 0 0.01" material="table_mat" mass="10"/>

        <body name="ur3_base" pos="0 0 0.02">
            <geom type="mesh" mesh="base_link-v3" rgba="0.7 0.7 0.7 1"/>
            <inertial mass="4.81" pos="-5e-05 -0.016 0.003" fullinertia="0.006 0.006 0.008 0 0 0"/>
            
            <body name="link1" pos="0.00017 -0.0211 0.1046" euler="1.571 0 0">
                <joint name="shoulder_pan" axis="0 1 0" pos="-0.0001 -0.0608 -0.004"/>
                <geom type="mesh" mesh="link1-v4" rgba="0.7 0.7 0.7 1"/>
                <inertial mass="0.31" pos="-0.0002 0.049 -0.009" fullinertia="0.0002 0.0004 0.0002 0 0 0"/>

                <body name="link2" pos="0.0062 0.122 0.1108" euler="-1.57 0 -3.14">
                    <joint name="shoulder_lift" axis="0 1 0" pos="0.006 -0.0608 -0.117"/>
                    <geom type="mesh" mesh="link2-v3" rgba="0.7 0.7 0.7 1"/>
                    <inertial mass="8.39" pos="0.003 0.006 0.052" fullinertia="0.051 0.052 0.007 0 0 0"/>

                    <body name="link3" pos="0.1004 0.0052 0.2402" euler="-1.572 0.059 1.602">
                        <joint name="elbow" axis="-0.03 -0.99 0.048" pos="0.115 0.048 -0.003"/>
                        <geom type="mesh" mesh="link3-v3" rgba="0.7 0.7 0.7 1"/>
                        <inertial mass="7.52" pos="0.002 0.006 0.001" fullinertia="0.005 0.055 0.057 0 0 0"/>

                        <body name="link4" pos="-0.0938 0.0851 0.0067" euler="-0.001 0 -0.079">
                            <joint name="wrist_1" axis="-0.04 0.99 0.0003" pos="-0.0008 -0.038 -0.001"/>
                            <geom type="mesh" mesh="link4-v3" rgba="0.7 0.7 0.7 1"/>
                            <inertial mass="2.54" pos="-0.001 0.0002 -0.001" fullinertia="0.001 0.002 0.002 0 0 0"/>

                            <body name="link5" pos="-0.0821 0.00004 -0.0009" euler="-2.62 1.49 2.62">
                                <joint name="wrist_2" axis="0.048 -0.007 -0.99" pos="-0.002 0.002 0.036"/>
                                <geom type="mesh" mesh="link5-v3" rgba="0.7 0.7 0.7 1"/>
                                <inertial mass="2.54" pos="-0.0009 0.001 -0.001" fullinertia="0.002 0.001 0.002 0 0 0"/>

                                <body name="link6" pos="0.0003 0.0626 -0.0087" euler="1.61 0.13 2.70">
                                    <joint name="wrist_3" axis="-0.14 -0.01 -0.98" pos="0.003 -0.0001 0.017"/>
                                    <geom type="mesh" mesh="link6-v3" rgba="0.7 0.7 0.7 1"/>
                                    
                                    <site name="ee_site" pos="0 0.05 0" size="0.01" rgba="1 0 0 1"/>
                                    
                                    <body name="Left_Grip" pos="-0.00289 -0.00054 -0.0273" euler="-1.498 -0.517 -2.967">
                                        <joint name="link6_Grip-Driver" type="hinge" axis="0 0 1" pos="0.016 -0.066 0.022"/>
                                        <geom name="Left_Grip_geom" type="mesh" mesh="Left_Grip" material="gripper_mat"/>
                                        <inertial mass="0.217" pos="0.029 -0.128 0.023" fullinertia="0.0004 4.9e-05 0.0004 0 0 0"/>
                                    </body>
                                    
                                    <body name="Right_Grip" pos="-0.00286 -0.00054 -0.0271" euler="-1.498 -0.517 -2.967">
                                        <joint name="link6_Grip-Link" type="hinge" axis="0 0 1" pos="-0.016 -0.066 0.017"/>
                                        <geom name="Right_Grip_geom" type="mesh" mesh="Right_Grip" material="gripper_mat"/>
                                        <inertial mass="0.239" pos="-0.028 -0.123 0.023" fullinertia="0.0004 5.2e-05 0.0005 0 0 0"/>
                                    </body>
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

print(f"✅ Generated {OUTPUT_FILE} successfully! Run with: python -m mujoco.viewer --mjcf={OUTPUT_FILE}")