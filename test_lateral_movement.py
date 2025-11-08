#!/usr/bin/env python3
"""Test to verify lateral movement (vy) works correctly."""

from gait_controller import DiagonalGaitController, GaitParameters
from joystick_input import VelocityCommand
import numpy as np

# Initialize gait controller
params = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)
controller = DiagonalGaitController(params, robot_width=0.1)
controller.reset()

print("=== Testing Lateral Movement (vy) ===\n")

# Test 1: Lateral movement to the right
print("Test 1: Lateral movement to the right (vy=0.1 m/s)")
vel = VelocityCommand(vx=0.0, vy=0.1, omega=0.0)

# Update controller
targets = controller.update(0.01, vel)

print(f"  step_y = {controller.step_y:.6f} (should be ~0.04 for half cycle)")
print(f"  Leg Y positions:")
for leg, target in sorted(targets.items()):
    print(f"    {leg}: y={target[1]:7.4f} m")

# Check that Y-displacement is present
y_values = [targets[leg][1] for leg in targets]
y_range = max(y_values) - min(y_values)

if y_range > 0.01:  # Swing legs should have different Y than stance
    print(f"  ✓ Lateral displacement detected (Y range: {y_range:.4f} m)\n")
else:
    print(f"  ✗ No lateral displacement! (Y range: {y_range:.4f} m)\n")

# Test 2: Lateral movement to the left
print("Test 2: Lateral movement to the left (vy=-0.1 m/s)")
vel = VelocityCommand(vx=0.0, vy=-0.1, omega=0.0)
targets = controller.update(0.01, vel)

print(f"  step_y = {controller.step_y:.6f} (should be ~-0.04)")
print(f"  Leg Y positions:")
for leg, target in sorted(targets.items()):
    print(f"    {leg}: y={target[1]:7.4f} m")

y_values = [targets[leg][1] for leg in targets]
y_range = max(y_values) - min(y_values)

if y_range > 0.01:
    print(f"  ✓ Lateral displacement detected (Y range: {y_range:.4f} m)\n")
else:
    print(f"  ✗ No lateral displacement! (Y range: {y_range:.4f} m)\n")

# Test 3: Combined forward + lateral
print("Test 3: Combined forward + lateral (vx=0.1, vy=0.05)")
vel = VelocityCommand(vx=0.1, vy=0.05, omega=0.0)
targets = controller.update(0.01, vel)

print(f"  step_x = {controller.step_x:.6f}, step_y = {controller.step_y:.6f}")
print(f"  Leg positions:")
for leg, target in sorted(targets.items()):
    print(f"    {leg}: x={target[0]:7.4f}, y={target[1]:7.4f}, z={target[2]:7.4f}")

x_range = max(targets[leg][0] for leg in targets) - min(targets[leg][0] for leg in targets)
y_range = max(targets[leg][1] for leg in targets) - min(targets[leg][1] for leg in targets)

print()
print("=" * 60)
if x_range > 0.01 and y_range > 0.01:
    print("SUCCESS: Both forward and lateral movement working! ✓")
    print(f"  X displacement range: {x_range:.4f} m")
    print(f"  Y displacement range: {y_range:.4f} m")
else:
    print("FAILURE: Movement not working correctly! ✗")
    print(f"  X displacement range: {x_range:.4f} m (should be > 0.01)")
    print(f"  Y displacement range: {y_range:.4f} m (should be > 0.01)")
print("=" * 60)
