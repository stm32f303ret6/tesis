#!/usr/bin/env python3
import math
import numpy as np

def solve_2link_ik(target, base, L1, L2, elbow_up):
    """Solve 2-link planar arm IK."""
    dx, dy = target - base
    dist = np.linalg.norm([dx, dy])
    
    # Check reachability
    if not (abs(L1 - L2) <= dist <= L1 + L2):
        return None
    
    # Base to target angle
    alpha = math.atan2(dy, dx)
    
    # Elbow angle (law of cosines)
    cos_t2 = (dist**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_t2 = np.clip(cos_t2, -1, 1)
    theta2 = math.acos(cos_t2) * (1 if elbow_up else -1)
    
    # Shoulder angle
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta
    
    return theta1, theta2

def parallel_scara_ik(target, L1=None, L2=None, base_dist=None, mode=None):
    """
    Parallel SCARA IK with 4 working modes:
    1: A up, B down | 2: A down, B up | 3: both up | 4: both down
    """
    # Load default parameters from config if not provided
    if L1 is None or L2 is None or base_dist is None or mode is None:
        default_mode = 2
        L1 = L1 if L1 is not None else 0.045
        L2 = L2 if L2 is not None else 0.06
        base_dist = base_dist if base_dist is not None else 0.021
        mode = mode if mode is not None else default_mode
    
    # Define elbow configurations
    configs = {1: (True, False), 2: (False, True), 3: (True, True), 4: (False, False)}
    elbow_A, elbow_B = configs.get(mode, (True, False))
    
    # Base positions
    base_A = np.array([-base_dist/2, 0])
    base_B = np.array([base_dist/2, 0])
    
    # Solve for each arm
    sol_A = solve_2link_ik(target, base_A, L1, L2, elbow_A)
    sol_B = solve_2link_ik(target, base_B, L1, L2, elbow_B)
    
    if sol_A is None or sol_B is None:
        return None
    
    return (*sol_A, *sol_B)  # (θ1_A, θ2_A, θ1_B, θ2_B)

def parallel_scara_ik_3dof(target_3d, L1=None, L2=None, base_dist=None, mode=None, tilt_angle=None):
    """
    3DOF Parallel SCARA IK including tilt control.
    
    Args:
        target_3d: 3D target position [x, y, z] in leg frame
        L1: Upper link length
        L2: Lower link length  
        base_dist: Distance between parallel arms
        mode: Elbow configuration (1-4)
        tilt_angle: Desired tilt angle (radians). If None, computed automatically
    
    Returns:
        (tilt, θ1_A, θ1_B) or None if unreachable
    """
    # Load default parameters from config if not provided
    if L1 is None or L2 is None or base_dist is None or mode is None:
        default_mode = 2
        L1 = L1 if L1 is not None else 0.045
        L2 = L2 if L2 is not None else 0.06
        base_dist = base_dist if base_dist is not None else 0.021
        mode = mode if mode is not None else default_mode
    
    x, y, z = target_3d
    
    # If tilt angle not specified, compute it based on desired y-position
    if tilt_angle is None:
        # Simple approach: use tilt to achieve desired y-displacement
        # For small angles: y ≈ z * sin(tilt) ≈ z * tilt
        if abs(z) > 1e-6:
            tilt_angle = math.atan2(y, abs(z))
        else:
            tilt_angle = 0.0
    
    # Transform target to tilted leg frame
    # After tilt rotation, the effective target in the leg's XZ plane is:
    cos_tilt = math.cos(tilt_angle)
    sin_tilt = math.sin(tilt_angle)
    
    # Project target onto the tilted plane
    z_eff = z * cos_tilt + y * sin_tilt  # Effective Z in tilted frame
    y_eff = -z * sin_tilt + y * cos_tilt  # Effective Y (should be ~0 for pure planar motion)
    
    # Target in the 2D plane for SCARA IK
    target_2d = np.array([x, -z_eff])  # Note: negative z because leg extends downward
    
    # Solve 2DOF SCARA for the tilted target
    result_2d = parallel_scara_ik(target_2d, L1, L2, base_dist, mode)
    
    if result_2d is None:
        return None
    
    # Extract shoulder angles (ignore elbow angles as they're constrained)
    theta1_A, _, theta1_B, _ = result_2d
    
    return (tilt_angle, theta1_A, theta1_B)

def solve_leg_ik_3dof(target_position, tilt_angle=None, L1=None, L2=None, base_dist=None, mode=None):
    """
    Convenience function for solving 3DOF leg IK with robot-specific parameters.
    
    Args:
        target_position: [x, y, z] target in leg frame
        tilt_angle: Desired tilt angle (None for auto)
        L1, L2, base_dist: Leg geometry parameters (loaded from config if None)
        mode: SCARA configuration mode (loaded from config if None)
    
    Returns:
        (tilt, shoulder_L, shoulder_R) angles or None
    """
    return parallel_scara_ik_3dof(target_position, L1, L2, base_dist, mode, tilt_angle)

# Example usage
if __name__ == "__main__":
    # Test 2DOF version
    print("=== 2DOF IK Test ===")
    target = np.array([0, 0.4])  # Target above center
    
    modes = ["Up-Down", "Down-Up", "Up-Up", "Down-Down"]
    for i, name in enumerate(modes, 1):
        result = parallel_scara_ik(target, mode=i)
        if result:
            t1A, t2A, t1B, t2B = result
            print(f"{name}: A[{math.degrees(t1A):6.1f}°, {math.degrees(t2A):6.1f}°] "
                  f"B[{math.degrees(t1B):6.1f}°, {math.degrees(t2B):6.1f}°]")
        else:
            print(f"{name}: unreachable")
    
    # Test 3DOF version
    print("\n=== 3DOF IK Test ===")
    target_3d = np.array([0.0, 0.01, -0.04])  # x=0, y=1cm, z=-4cm
    
    for i, name in enumerate(modes, 1):
        result = solve_leg_ik_3dof(target_3d, mode=i)
        if result:
            tilt, t1A, t1B = result
            print(f"{name}: Tilt[{math.degrees(tilt):6.1f}°] A[{math.degrees(t1A):6.1f}°] "
                  f"B[{math.degrees(t1B):6.1f}°]")
        else:
            print(f"{name}: unreachable")
    
    # Test pure height control (no lateral displacement)
    print("\n=== Pure Height Control Test ===")
    heights = [0.02, 0.04, 0.06]
    for height in heights:
        target_height = np.array([0.0, 0.0, -height])
        result = solve_leg_ik_3dof(target_height, mode=2)
        if result:
            tilt, t1A, t1B = result
            print(f"Height {height*1000:2.0f}mm: Tilt[{math.degrees(tilt):6.1f}°] "
                  f"A[{math.degrees(t1A):6.1f}°] B[{math.degrees(t1B):6.1f}°]")
        else:
            print(f"Height {height*1000:2.0f}mm: unreachable")
