#!/usr/bin/env python3
"""Test to verify robot stays stationary with zero velocity."""

from gait_controller import DiagonalGaitController, GaitParameters
from joystick_input import VelocityCommand
import numpy as np

# Initialize gait controller
params = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)
controller = DiagonalGaitController(params, robot_width=0.1)
controller.reset()

print("=== Testing Stationary Mode ===\n")

# Set zero velocity
vel = VelocityCommand(vx=0.0, vy=0.0, omega=0.0)

# Update controller multiple times
for i in range(10):
    targets = controller.update(0.01, vel)

print(f"After 10 updates with zero velocity:")
print(f"  step_x = {controller.step_x:.6f} (should be 0.0)")
print(f"  step_y = {controller.step_y:.6f} (should be 0.0)")
print()

# Check that all leg targets are at the same position (no stepping)
print("Leg target positions:")
for leg, target in targets.items():
    print(f"  {leg}: x={target[0]:7.4f}, y={target[1]:7.4f}, z={target[2]:7.4f}")

print()

# Verify no motion in X direction (all legs should have x ≈ 0)
x_positions = [targets[leg][0] for leg in targets]
max_x_variation = max(abs(x) for x in x_positions)

print(f"Maximum X variation: {max_x_variation:.6f} m")

if max_x_variation < 0.001:
    print("✓ Robot is stationary (no X displacement)")
else:
    print(f"✗ Robot is moving! (X variation: {max_x_variation:.6f} m)")

# Check Y positions as well
y_positions = [targets[leg][1] for leg in targets]
y_variation = max(y_positions) - min(y_positions)
print(f"Y position variation: {y_variation:.6f} m (lateral offsets)")

print("\n" + "=" * 50)
if controller.step_x == 0.0 and controller.step_y == 0.0 and max_x_variation < 0.001:
    print("SUCCESS: Robot correctly stays stationary! ✓")
else:
    print("FAILURE: Robot is walking when it should be stationary! ✗")
print("=" * 50)
