#!/usr/bin/env python3
"""Test IK with lateral displacement to ensure reachability."""

from gait_controller import DiagonalGaitController, GaitParameters
from joystick_input import VelocityCommand
from ik import solve_leg_ik_3dof
import numpy as np

IK_PARAMS = dict(L1=0.045, L2=0.06, base_dist=0.021, mode=2)

# Initialize gait controller
params = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)
controller = DiagonalGaitController(params, robot_width=0.1)
controller.reset()

print("=== Testing IK with Lateral Movement ===\n")

# Test different velocity combinations
test_cases = [
    ("Right lateral", VelocityCommand(vx=0.0, vy=0.1, omega=0.0)),
    ("Left lateral", VelocityCommand(vx=0.0, vy=-0.1, omega=0.0)),
    ("Forward + Right", VelocityCommand(vx=0.1, vy=0.05, omega=0.0)),
    ("Backward + Left", VelocityCommand(vx=-0.1, vy=-0.05, omega=0.0)),
    ("Forward + Right + Rotate", VelocityCommand(vx=0.1, vy=0.05, omega=0.3)),
]

all_passed = True

for test_name, vel in test_cases:
    print(f"Test: {test_name}")
    print(f"  Velocity: vx={vel.vx:.2f}, vy={vel.vy:.2f}, ω={vel.omega:.2f}")

    # Update gait controller
    targets = controller.update(0.01, vel)

    # Try to solve IK for all legs
    ik_results = {}
    failures = []

    for leg, target in targets.items():
        result = solve_leg_ik_3dof(target, **IK_PARAMS)
        ik_results[leg] = result

        if result is None:
            failures.append(leg)
            all_passed = False

    if failures:
        print(f"  ✗ IK FAILED for legs: {failures}")
        for leg in failures:
            print(f"    {leg} target: {targets[leg]}")
    else:
        print(f"  ✓ IK solved successfully for all legs")
        # Show tilt angles
        tilts = {leg: np.degrees(result[0]) for leg, result in ik_results.items()}
        print(f"    Tilt angles: {', '.join(f'{leg}={tilts[leg]:.1f}°' for leg in sorted(tilts.keys()))}")

    print()

print("=" * 60)
if all_passed:
    print("SUCCESS: All IK solutions valid with lateral movement! ✓")
else:
    print("WARNING: Some IK solutions failed - check reachability limits!")
print("=" * 60)
