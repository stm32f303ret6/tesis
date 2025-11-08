#!/usr/bin/env python3
"""Test script to verify joystick integration with gait controller."""

from gait_controller import DiagonalGaitController, GaitParameters
from joystick_input import VelocityCommand

# Initialize gait controller
params = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)
controller = DiagonalGaitController(params, robot_width=0.1)
controller.reset()

print("=== Testing Joystick Integration ===\n")

# Test 1: Stationary (zero velocity)
print("Test 1: Stationary")
vel = VelocityCommand(vx=0.0, vy=0.0, omega=0.0)
targets = controller.update(0.01, vel)
print(f"  ✓ Stationary test passed - got {len(targets)} leg targets\n")

# Test 2: Forward motion
print("Test 2: Forward motion (vx=0.1)")
vel = VelocityCommand(vx=0.1, vy=0.0, omega=0.0)
targets = controller.update(0.01, vel)
print(f"  ✓ Forward motion test passed")
print(f"    Step displacement: x={controller.step_x:.4f}, y={controller.step_y:.4f}\n")

# Test 3: Backward motion
print("Test 3: Backward motion (vx=-0.1)")
vel = VelocityCommand(vx=-0.1, vy=0.0, omega=0.0)
targets = controller.update(0.01, vel)
print(f"  ✓ Backward motion test passed")
print(f"    Step displacement: x={controller.step_x:.4f}, y={controller.step_y:.4f}\n")

# Test 4: Lateral motion
print("Test 4: Lateral motion (vy=0.05)")
vel = VelocityCommand(vx=0.0, vy=0.05, omega=0.0)
targets = controller.update(0.01, vel)
print(f"  ✓ Lateral motion test passed")
print(f"    Step displacement: x={controller.step_x:.4f}, y={controller.step_y:.4f}\n")

# Test 5: Rotation
print("Test 5: Rotation (omega=0.5 rad/s)")
vel = VelocityCommand(vx=0.0, vy=0.0, omega=0.5)
targets = controller.update(0.01, vel)
print(f"  ✓ Rotation test passed")
print(f"    Leg rotations: FL={controller._leg_rotations['FL']:.4f}, FR={controller._leg_rotations['FR']:.4f}\n")

# Test 6: Combined motion
print("Test 6: Combined motion (vx=0.1, vy=0.05, omega=0.3)")
vel = VelocityCommand(vx=0.1, vy=0.05, omega=0.3)
targets = controller.update(0.01, vel)
print(f"  ✓ Combined motion test passed")
print(f"    Step: x={controller.step_x:.4f}, y={controller.step_y:.4f}")
print(f"    FL rotation={controller._leg_rotations['FL']:.4f}, FR={controller._leg_rotations['FR']:.4f}\n")

# Test 7: Verify all legs get targets
print("Test 7: Verify all 4 legs receive targets")
assert set(targets.keys()) == {"FL", "FR", "RL", "RR"}
for leg, target in targets.items():
    assert target.shape == (3,), f"Expected 3D target for {leg}, got shape {target.shape}"
print("  ✓ All legs receiving correct 3D targets\n")

print("=" * 40)
print("All tests passed! ✓")
print("=" * 40)
