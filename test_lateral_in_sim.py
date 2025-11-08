#!/usr/bin/env python3
"""Test lateral movement in MuJoCo simulation by tracking robot body position."""

import time
import mujoco
import numpy as np

from gait_controller import DiagonalGaitController, GaitParameters
from joystick_input import VelocityCommand
from ik import solve_leg_ik_3dof

# Setup
IK_PARAMS = dict(L1=0.045, L2=0.06, base_dist=0.021, mode=2)

LEG_CONTROL = {
    "FL": {"indices": (0, 1, 2), "sign": -1.0, "offset": -np.pi},
    "FR": {"indices": (6, 7, 8), "sign": 1.0, "offset": np.pi},
    "RL": {"indices": (3, 4, 5), "sign": -1.0, "offset": -np.pi},
    "RR": {"indices": (9, 10, 11), "sign": 1.0, "offset": np.pi},
}

GAIT_PARAMS = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)

model = mujoco.MjModel.from_xml_path("model/world.xml")
data = mujoco.MjData(model)
robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")

def apply_leg_angles(leg, angles):
    """Apply IK angles to leg actuators."""
    tilt, ang_left, ang_right = angles
    config = LEG_CONTROL[leg]
    idx_left, idx_right, idx_tilt = config["indices"]
    sign = config["sign"]
    offset = config["offset"]

    data.ctrl[idx_left] = sign * ang_left
    data.ctrl[idx_right] = sign * ang_right + offset
    data.ctrl[idx_tilt] = tilt

# Initialize
controller = DiagonalGaitController(GAIT_PARAMS, robot_width=0.1)
controller.reset()

# Test lateral movement
vel = VelocityCommand(vx=0.0, vy=0.1, omega=0.0)

print("=== Testing Lateral Movement in Simulation ===\n")
print(f"Velocity command: vy = {vel.vy} m/s")
print(f"Running for 5 seconds...\n")

# Track body position
body_positions = []
timestamps = []

start_time = time.time()
sim_time = 0.0
duration = 5.0

while sim_time < duration:
    step_start = time.time()

    # Update gait controller
    leg_targets = controller.update(model.opt.timestep, vel)

    # Solve IK and apply to robot
    for leg, target in leg_targets.items():
        result = solve_leg_ik_3dof(target, **IK_PARAMS)
        if result:
            apply_leg_angles(leg, result)

    # Step simulation
    mujoco.mj_step(model, data)

    # Record body position
    robot_pos = data.xpos[robot_body_id].copy()
    body_positions.append(robot_pos)
    timestamps.append(sim_time)

    # Print every 0.5 seconds
    if len(timestamps) % 250 == 0:
        print(f"t={sim_time:4.2f}s  Body position: x={robot_pos[0]:7.4f}, y={robot_pos[1]:7.4f}, z={robot_pos[2]:7.4f}")

    sim_time += model.opt.timestep

    # Real-time pacing
    time_until_next = model.opt.timestep - (time.time() - step_start)
    if time_until_next > 0:
        time.sleep(time_until_next)

# Analyze body motion
positions = np.array(body_positions)
x_positions = positions[:, 0]
y_positions = positions[:, 1]

print(f"\n{'='*60}")
print("Analysis:")
print(f"  Initial Y position: {y_positions[0]:.6f} m")
print(f"  Final Y position:   {y_positions[-1]:.6f} m")
print(f"  Net Y displacement: {y_positions[-1] - y_positions[0]:+.6f} m")
print(f"  Average Y velocity: {(y_positions[-1] - y_positions[0]) / duration:.6f} m/s")
print(f"  Expected velocity:  {vel.vy:.6f} m/s")
print(f"\n  Y position range: {y_positions.min():.6f} to {y_positions.max():.6f} m")
print(f"  Y oscillation amplitude: ±{(y_positions.max() - y_positions.min()) / 2:.6f} m")

# Check if there's net translation
net_translation = y_positions[-1] - y_positions[0]
if abs(net_translation) > 0.01:
    print(f"\n✓ Robot IS translating laterally! Net: {net_translation:+.4f} m")
else:
    print(f"\n✗ Robot is NOT translating (just oscillating)")
print(f"{'='*60}")
