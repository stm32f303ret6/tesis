#!/usr/bin/env python3
"""
Calculate the reachable workspace range (x, y, z) for each foot based on IK parameters.
This script determines the operational range from the leg frame origin.
"""

import numpy as np
import math
from ik import parallel_scara_ik_3dof

# Leg geometry parameters (from ik.py and height_control.py)
L1 = 0.045  # Upper link length (m)
L2 = 0.06   # Lower link length (m)
BASE_DIST = 0.021  # Distance between parallel arms (m)
IK_MODE = 2  # Default mode: left arm down, right arm up

# Physical constraints
MAX_REACH = L1 + L2  # 0.105m
MIN_REACH = abs(L1 - L2)  # 0.015m

def is_reachable(target_3d, mode=IK_MODE):
    """Check if a 3D target position is reachable by the leg."""
    result = parallel_scara_ik_3dof(
        target_3d,
        L1=L1,
        L2=L2,
        base_dist=BASE_DIST,
        mode=mode
    )
    return result is not None

def calculate_foot_range(resolution=50, mode=IK_MODE):
    """
    Calculate the reachable workspace range for a single foot.

    Args:
        resolution: Number of samples per axis
        mode: IK mode (1-4)

    Returns:
        dict with 'x', 'y', 'z' ranges and 'reachable_points'
    """
    # Define search ranges based on max reach
    # X: Forward/backward
    x_range = np.linspace(-MAX_REACH, MAX_REACH, resolution)
    # Y: Lateral (limited by tilt mechanism)
    y_range = np.linspace(-MAX_REACH * 0.5, MAX_REACH * 0.5, resolution)
    # Z: Vertical (negative is downward)
    z_range = np.linspace(-MAX_REACH, MAX_REACH * 0.2, resolution)

    reachable_points = []

    print(f"Sampling workspace with {resolution}^3 = {resolution**3} points...")
    print("This may take a moment...\n")

    # Sample the workspace
    for x in x_range:
        for y in y_range:
            for z in z_range:
                target = np.array([x, y, z])

                # Check if target is reachable
                if is_reachable(target, mode):
                    reachable_points.append(target)

    if not reachable_points:
        print("ERROR: No reachable points found!")
        return None

    reachable_points = np.array(reachable_points)

    # Calculate ranges
    x_min, x_max = reachable_points[:, 0].min(), reachable_points[:, 0].max()
    y_min, y_max = reachable_points[:, 1].min(), reachable_points[:, 1].max()
    z_min, z_max = reachable_points[:, 2].min(), reachable_points[:, 2].max()

    return {
        'x': (x_min, x_max),
        'y': (y_min, y_max),
        'z': (z_min, z_max),
        'reachable_points': reachable_points,
        'total_points': len(reachable_points),
        'sampled_points': resolution**3
    }

def calculate_practical_range(mode=IK_MODE):
    """
    Calculate practical working range based on common constraints.
    Uses analytical approach for faster computation.
    """
    print("Calculating practical range analytically...\n")

    # Test key positions
    test_positions = []

    # Generate test positions in a grid
    for x in np.linspace(-MAX_REACH, MAX_REACH, 20):
        for y in np.linspace(-MAX_REACH * 0.3, MAX_REACH * 0.3, 15):
            for z in np.linspace(-MAX_REACH, 0, 20):
                target = np.array([x, y, z])
                if is_reachable(target, mode):
                    test_positions.append(target)

    if not test_positions:
        return None

    test_positions = np.array(test_positions)

    x_min, x_max = test_positions[:, 0].min(), test_positions[:, 0].max()
    y_min, y_max = test_positions[:, 1].min(), test_positions[:, 1].max()
    z_min, z_max = test_positions[:, 2].min(), test_positions[:, 2].max()

    return {
        'x': (x_min, x_max),
        'y': (y_min, y_max),
        'z': (z_min, z_max),
        'tested_points': len(test_positions)
    }

def print_range_results(results, title="Foot Range"):
    """Pretty print the range results."""
    if results is None:
        print("No results to display.")
        return

    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

    x_min, x_max = results['x']
    y_min, y_max = results['y']
    z_min, z_max = results['z']

    print(f"\nX-axis (Forward/Backward) range:")
    print(f"  Min: {x_min:+.4f} m ({x_min*1000:+.1f} mm)")
    print(f"  Max: {x_max:+.4f} m ({x_max*1000:+.1f} mm)")
    print(f"  Range: {x_max - x_min:.4f} m ({(x_max - x_min)*1000:.1f} mm)")

    print(f"\nY-axis (Lateral/Sideways) range:")
    print(f"  Min: {y_min:+.4f} m ({y_min*1000:+.1f} mm)")
    print(f"  Max: {y_max:+.4f} m ({y_max*1000:+.1f} mm)")
    print(f"  Range: {y_max - y_min:.4f} m ({(y_max - y_min)*1000:.1f} mm)")

    print(f"\nZ-axis (Vertical, negative is down) range:")
    print(f"  Min: {z_min:+.4f} m ({z_min*1000:+.1f} mm)")
    print(f"  Max: {z_max:+.4f} m ({z_max*1000:+.1f} mm)")
    print(f"  Range: {z_max - z_min:.4f} m ({(z_max - z_min)*1000:.1f} mm)")

    print(f"\nPhysical parameters:")
    print(f"  L1 (upper link): {L1} m ({L1*1000:.1f} mm)")
    print(f"  L2 (lower link): {L2} m ({L2*1000:.1f} mm)")
    print(f"  Base distance: {BASE_DIST} m ({BASE_DIST*1000:.1f} mm)")
    print(f"  Max reach: {MAX_REACH} m ({MAX_REACH*1000:.1f} mm)")
    print(f"  Min reach: {MIN_REACH} m ({MIN_REACH*1000:.1f} mm)")
    print(f"  IK Mode: {IK_MODE}")

    if 'tested_points' in results:
        print(f"\nTested points: {results['tested_points']}")
    elif 'total_points' in results:
        print(f"\nReachable points: {results['total_points']} / {results['sampled_points']}")
        print(f"Workspace coverage: {results['total_points']/results['sampled_points']*100:.1f}%")

    print("=" * 60)

def test_safe_ranges():
    """Test the safe parameter ranges mentioned in CLAUDE.md"""
    print("\n" + "=" * 60)
    print("Testing Safe Parameter Ranges (from CLAUDE.md)".center(60))
    print("=" * 60)

    # Safe ranges from CLAUDE.md
    safe_stance_heights = [-0.04, -0.06, -0.08, -0.09]
    safe_step_lengths = [0.02, 0.04, 0.06]
    safe_step_heights = [0.01, 0.03, 0.05]

    print("\nTesting stance heights (vertical position):")
    for z in safe_stance_heights:
        target = np.array([0.0, 0.0, z])
        reachable = is_reachable(target)
        status = "✓ REACHABLE" if reachable else "✗ UNREACHABLE"
        print(f"  z = {z:+.3f} m ({z*1000:+.0f} mm): {status}")

    print("\nTesting step lengths (x-axis movement at stance height -0.08m):")
    stance_z = -0.08
    for x in safe_step_lengths:
        target = np.array([x, 0.0, stance_z])
        reachable = is_reachable(target)
        status = "✓ REACHABLE" if reachable else "✗ UNREACHABLE"
        print(f"  x = {x:+.3f} m ({x*1000:+.0f} mm): {status}")

    print("\nTesting step heights (z variation from stance at x=0.02m):")
    x_pos = 0.02
    stance_z = -0.08
    for lift in safe_step_heights:
        z = stance_z + lift  # Lift from stance height
        target = np.array([x_pos, 0.0, z])
        reachable = is_reachable(target)
        status = "✓ REACHABLE" if reachable else "✗ UNREACHABLE"
        print(f"  lift = {lift:.3f} m, z = {z:+.3f} m: {status}")

    print("\nTesting lateral movement (y-axis at stance height -0.08m):")
    lateral_offsets = [-0.02, -0.015, -0.01, 0.0, 0.01, 0.015, 0.02]
    for y in lateral_offsets:
        target = np.array([0.0, y, stance_z])
        reachable = is_reachable(target)
        status = "✓ REACHABLE" if reachable else "✗ UNREACHABLE"
        print(f"  y = {y:+.3f} m ({y*1000:+.1f} mm): {status}")


if __name__ == "__main__":
    print("Foot Range Calculator")
    print("Based on IK parameters and height_control.py values\n")

    # Method 1: Practical range (faster)
    practical_results = calculate_practical_range(mode=IK_MODE)
    if practical_results:
        print_range_results(practical_results, "Practical Working Range (Quick Analysis)")

    # Test safe parameter ranges
    test_safe_ranges()

    # Ask user if they want detailed analysis
    print("\n" + "=" * 60)
    response = input("\nPerform detailed workspace analysis? (slower, ~30-60s) [y/N]: ").strip().lower()

    if response == 'y':
        # Method 2: Detailed sampling (slower but more accurate)
        detailed_results = calculate_foot_range(resolution=40, mode=IK_MODE)
        if detailed_results:
            print_range_results(detailed_results, "Detailed Workspace Range")

            # Save results to file
            output_file = "foot_range_results.txt"
            with open(output_file, 'w') as f:
                import sys
                old_stdout = sys.stdout
                sys.stdout = f
                print_range_results(detailed_results, "Detailed Workspace Range")
                sys.stdout = old_stdout
            print(f"\nDetailed results saved to: {output_file}")
    else:
        print("\nSkipping detailed analysis.")

    print("\nAnalysis complete!")
