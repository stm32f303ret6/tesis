#!/usr/bin/env python3
"""
Smooth startup script that:
1. Loads the robot model from model/robot.xml
2. Waits for user-specified delay to allow camera setup
3. Sets all tilt motors to 0 using cubic easing over 2 seconds
4. Transitions to height control movements from height_control.py
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from ik import parallel_scara_ik, solve_leg_ik_3dof

# Load model
model = mujoco.MjModel.from_xml_path("model/world.xml")
data = mujoco.MjData(model)
ik_mode = 2

# Tilt motor indices (FL, RL, FR, RR)
TILT_INDICES = [2, 5, 8, 11]

# Initialize parameters from height_control.py
L1, L2 = 0.045, 0.06
max_reach = L1 + L2
min_height = max_reach * 0.15
max_height = max_reach * 0.9
height_range = max_height - min_height
period = 3.0

def cubic_ease(t):
    """Cubic easing function (smoothstep): 3t^2 - 2t^3"""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return t * t * (3.0 - 2.0 * t)

def set_leg_height(height, lateral_offset=0.0):
    """Set leg height with optional lateral (sideways) offset for 3DOF control."""
    target_3d = np.array([0.0, lateral_offset, height])

    res = solve_leg_ik_3dof(target_3d,
                           L1=0.045,
                           L2=0.06,
                           base_dist=0.021,
                           mode=ik_mode)
    if not res:
        print(f"3DOF IK failed at height={height}, lateral={lateral_offset}")
        return False

    tilt, ang1L, ang1R = res

    # Set front legs (FL, FR)
    for i in range(6, 12, 3):
        data.ctrl[i]   = ang1L
        data.ctrl[i+1] = ang1R + np.pi
        data.ctrl[i+2] = tilt

    # Set rear legs (RL, RR)
    for i in range(0, 6, 3):
        data.ctrl[i]   = -ang1L
        data.ctrl[i+1] = -ang1R - np.pi
        data.ctrl[i+2] = tilt

    return True

# Get initial tilt values from the model's keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
initial_tilt_values = [data.ctrl[i] for i in TILT_INDICES]
target_tilt = 0.0

# Get robot body ID for camera tracking
robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Delay before motion
    camera_setup_delay = 5.0  # seconds — adjust as needed
    print(f"Waiting {camera_setup_delay} seconds for camera setup...")
    delay_start = time.time()
    while time.time() - delay_start < camera_setup_delay and viewer.is_running():
        # Keep updating the viewer so you can move the camera
        mujoco.mj_step(model, data)
        viewer.sync()

    start_time = time.time()
    tilt_transition_duration = 2.0  # seconds

    print("Phase 1: Smoothly setting tilt motors to 0 using cubic easing...")

    while viewer.is_running():
        current_time = time.time() - start_time

        if current_time < tilt_transition_duration:
            # Phase 1: Smooth tilt transition to 0
            t = current_time / tilt_transition_duration
            eased_t = cubic_ease(t)
            for idx, initial_val in zip(TILT_INDICES, initial_tilt_values):
                data.ctrl[idx] = initial_val * (1.0 - eased_t) + target_tilt * eased_t

        else:
            # Phase 2: Height control movements
            if abs(current_time - tilt_transition_duration) < 0.01:
                print("Phase 2: Starting height control movements...")

            movement_time = current_time - tilt_transition_duration
            height_factor = 0.3 * (1 + math.sin(2 * math.pi * movement_time / period))
            target_height = min_height + height_range * height_factor
            lateral_factor = 0.015 * math.sin(4 * math.pi * movement_time / period)

            set_leg_height(target_height, lateral_factor)

        mujoco.mj_step(model, data)
        robot_pos = data.xpos[robot_body_id]
        viewer.cam.lookat[:] = robot_pos
        viewer.sync()

print("Demo finished")

