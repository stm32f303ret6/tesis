#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from ik import solve_leg_ik_3dof

model = mujoco.MjModel.from_xml_path("model/world_static_test.xml")
data = mujoco.MjData(model)
ik_mode = 2

def set_leg_height(height, lateral_offset=0.0):
    """Set leg height with optional lateral (sideways) offset for 3DOF control."""
    # Target position in leg frame [x, y, z]
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
    
    # Set front legs (FL, FR) - indices 6-11
    for i in range(6, 12, 3):
        data.ctrl[i]   = ang1L         # Left shoulder
        data.ctrl[i+1] = ang1R + np.pi # Right shoulder (offset by π)
        data.ctrl[i+2] = tilt          # Tilt joint

    # Set rear legs (RL, RR) - indices 0-5
    for i in range(0, 6, 3):
        data.ctrl[i]   = -ang1L        # Left shoulder (mirrored)
        data.ctrl[i+1] = -ang1R - np.pi # Right shoulder (mirrored and offset)
        data.ctrl[i+2] = tilt          # Tilt joint
    
    return True

# Calculate height limits based on config parameters
L1, L2 = 0.045, 0.06
max_reach = L1 + L2
min_height = max_reach * 0.15  # 25% of max reach as minimum
max_height = max_reach * 0.9   # 90% of max reach as maximum
height_range = max_height - min_height
period = 3.0  # Period in seconds

# Get robot body ID for camera tracking
robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        current_time = time.time() - start_time
        
        # Sinusoidal height variation
        height_factor = 0.3 * (1 + math.sin(2 * math.pi * current_time / period))
        target_height = min_height + height_range * height_factor
        
        # Optional: add lateral oscillation to demonstrate 3DOF capability
        lateral_factor = 0.015 * math.sin(4 * math.pi * current_time / period)  # 5mm amplitude, 2x frequency
        
        # Use 3DOF IK with optional lateral motion
        set_leg_height(target_height, lateral_factor)
        mujoco.mj_step(model, data)
        
        # Update camera to follow robot
        robot_pos = data.xpos[robot_body_id]
        viewer.cam.lookat[:] = robot_pos
        viewer.sync()

print("Demo finished")
