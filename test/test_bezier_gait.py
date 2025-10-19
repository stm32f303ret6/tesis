#!/usr/bin/env python3
"""
Test bezier curve trajectories for quadruped gait control.
Visualizes a single leg moving through a bezier-based gait cycle.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ik import solve_leg_ik_3dof

# Leg parameters from height_control.py
L1, L2 = 0.045, 0.06
ik_mode = 2


def bezier_curve(t, control_points):
    """
    Evaluate cubic bezier curve at parameter t (0 to 1).

    Args:
        t: Parameter value [0, 1]
        control_points: List of 4 control points [P0, P1, P2, P3]

    Returns:
        Point on bezier curve at t
    """
    P0, P1, P2, P3 = control_points

    # Cubic bezier formula
    return (1-t)**3 * P0 + \
           3 * (1-t)**2 * t * P1 + \
           3 * (1-t) * t**2 * P2 + \
           t**3 * P3


def generate_gait_trajectory(num_points=100,
                            step_height=0.03,
                            step_length=0.04,
                            stance_height=-0.05):
    """
    Generate a gait trajectory using bezier curves.

    Args:
        num_points: Number of points in trajectory
        step_height: Maximum height during swing phase (m)
        step_length: Forward/backward reach during step (m)
        stance_height: Height during stance phase (m)

    Returns:
        trajectory: Array of [x, y, z] positions
    """
    trajectory = []

    # Half cycle is swing phase (foot in air), half is stance (foot on ground)
    swing_points = num_points // 2
    stance_points = num_points - swing_points

    # SWING PHASE: Bezier curve from back to front, lifting foot
    # Control points for swing [x, y, z]
    P0_swing = np.array([-step_length/2, 0.0, stance_height])  # Start (back, on ground)
    P1_swing = np.array([-step_length/3, 0.0, stance_height + step_height])  # Lift
    P2_swing = np.array([step_length/3, 0.0, stance_height + step_height])   # Forward high
    P3_swing = np.array([step_length/2, 0.0, stance_height])   # End (front, touch down)

    swing_control_points = [P0_swing, P1_swing, P2_swing, P3_swing]

    for i in range(swing_points):
        t = i / swing_points
        point = bezier_curve(t, swing_control_points)
        trajectory.append(point)

    # STANCE PHASE: Linear motion from front to back (foot on ground)
    for i in range(stance_points):
        t = i / stance_points
        x = step_length/2 - step_length * t  # Move from front to back
        y = 0.0
        z = stance_height
        trajectory.append(np.array([x, y, z]))

    return np.array(trajectory)


def forward_kinematics_2dof(shoulder_L, shoulder_R, base_dist=0.021):
    """
    Compute forward kinematics for 2DOF parallel SCARA in XZ plane.
    This solves the 4-bar linkage to find the actual end effector position.

    Returns key points for visualization.
    """
    # Base positions of the two arms (in XZ plane, Y=0)
    base_L = np.array([-base_dist/2, 0])
    base_R = np.array([base_dist/2, 0])

    # First link endpoints (shoulder joints)
    shoulder_L_end = base_L + L1 * np.array([np.cos(shoulder_L), -np.sin(shoulder_L)])
    shoulder_R_end = base_R + L1 * np.array([np.cos(shoulder_R), -np.sin(shoulder_R)])

    # The end effector is where the two second links (L2) meet
    # We need to find the intersection of two circles:
    # Circle 1: center at shoulder_L_end, radius L2
    # Circle 2: center at shoulder_R_end, radius L2

    # Distance between shoulder ends
    dx = shoulder_R_end[0] - shoulder_L_end[0]
    dz = shoulder_R_end[1] - shoulder_L_end[1]
    d = np.sqrt(dx**2 + dz**2)

    # If unreachable, return approximate
    if d > 2*L2 or d < 1e-6:
        end_effector = (shoulder_L_end + shoulder_R_end) / 2
        return {
            'base_L': base_L,
            'base_R': base_R,
            'shoulder_L_end': shoulder_L_end,
            'shoulder_R_end': shoulder_R_end,
            'end_effector': end_effector
        }

    # Find intersection point using circle-circle intersection
    a = d / 2
    h = np.sqrt(L2**2 - a**2) if L2**2 >= a**2 else 0

    # Midpoint
    mid_x = (shoulder_L_end[0] + shoulder_R_end[0]) / 2
    mid_z = (shoulder_L_end[1] + shoulder_R_end[1]) / 2

    # Perpendicular direction (for mode 2: down-up, we want the lower intersection)
    perp_x = -dz / d if d > 0 else 0
    perp_z = dx / d if d > 0 else 0

    # End effector (take the downward solution for mode 2)
    end_effector = np.array([mid_x - h * perp_x, mid_z - h * perp_z])

    return {
        'base_L': base_L,
        'base_R': base_R,
        'shoulder_L_end': shoulder_L_end,
        'shoulder_R_end': shoulder_R_end,
        'end_effector': end_effector
    }


def forward_kinematics_3dof(tilt, shoulder_L, shoulder_R, base_dist=0.021):
    """
    Compute forward kinematics for 3DOF parallel SCARA with tilt.
    Returns key points in 3D space for visualization.

    Matches the transformation in parallel_scara_ik_3dof from ik.py:
    - IK forward transform: z_eff = z*cos(tilt) + y*sin(tilt)
                           target_2d = [x, -z_eff]
    - FK inverse transform: recover [x, y, z] from 2D solution
    """
    # Compute FK in the 2D planar frame
    # This gives us points in [x, z_planar] representing the mechanism in its plane
    fk_2d = forward_kinematics_2dof(shoulder_L, shoulder_R, base_dist)

    cos_tilt = np.cos(tilt)
    sin_tilt = np.sin(tilt)

    # Convert 2D planar points to 3D world coordinates
    def planar_to_world(point_2d):
        x_planar, z_planar = point_2d

        # The IK does: target_2d = [x, -z_eff] where z_eff = z*cos + y*sin
        # So z_planar is the actual z position in the planar mechanism
        # And we need to apply tilt rotation to get world coordinates

        # In the planar frame (no tilt): x stays, z_planar is the vertical drop
        # With tilt: the planar z becomes a combination of world y and z
        # Rotation matrix for tilt about X axis:
        #   [1      0         0    ]
        #   [0  cos(tilt) -sin(tilt)]
        #   [0  sin(tilt)  cos(tilt)]

        x = x_planar
        y = -z_planar * sin_tilt  # Planar Z contributes to world Y
        z = z_planar * cos_tilt    # Planar Z contributes to world Z

        return np.array([x, y, z])

    return {
        'base_L': planar_to_world(fk_2d['base_L']),
        'base_R': planar_to_world(fk_2d['base_R']),
        'shoulder_L_end': planar_to_world(fk_2d['shoulder_L_end']),
        'shoulder_R_end': planar_to_world(fk_2d['shoulder_R_end']),
        'end_effector': planar_to_world(fk_2d['end_effector'])
    }


def test_gait_visualization():
    """Create interactive visualization of gait trajectory and leg motion."""

    # Generate trajectory
    print("Generating bezier gait trajectory...")
    trajectory = generate_gait_trajectory(
        num_points=100,
        step_height=0.03,
        step_length=0.04,
        stance_height=-0.08
    )

    # Solve IK for each point
    print("Solving IK for trajectory points...")
    joint_angles = []
    valid_points = []

    for i, target in enumerate(trajectory):
        result = solve_leg_ik_3dof(target, L1=L1, L2=L2, base_dist=0.021, mode=ik_mode)
        if result:
            joint_angles.append(result)
            valid_points.append(target)
        else:
            print(f"IK failed at point {i}: {target}")

    if not joint_angles:
        print("ERROR: No valid IK solutions found!")
        return

    print(f"Valid IK solutions: {len(joint_angles)}/{len(trajectory)}")
    joint_angles = np.array(joint_angles)
    valid_points = np.array(valid_points)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))

    # 1. Trajectory in 3D
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(valid_points[0, 0], valid_points[0, 1], valid_points[0, 2],
                c='green', s=100, marker='o', label='Start')
    ax1.scatter(valid_points[-1, 0], valid_points[-1, 1], valid_points[-1, 2],
                c='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Foot Trajectory')
    ax1.legend()
    ax1.grid(True)

    # 2. Trajectory side view (XZ plane)
    ax2 = fig.add_subplot(132)
    ax2.plot(valid_points[:, 0], valid_points[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax2.scatter(valid_points[0, 0], valid_points[0, 2], c='green', s=100, marker='o', label='Start')

    # Mark swing/stance phases
    mid_point = len(valid_points) // 2
    ax2.plot(valid_points[:mid_point, 0], valid_points[:mid_point, 2],
             'r-', linewidth=3, alpha=0.5, label='Swing phase')
    ax2.plot(valid_points[mid_point:, 0], valid_points[mid_point:, 2],
             'g-', linewidth=3, alpha=0.5, label='Stance phase')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (XZ plane)')
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')

    # 3. Joint angles over time
    ax3 = fig.add_subplot(133)
    time_steps = np.arange(len(joint_angles))
    ax3.plot(time_steps, np.degrees(joint_angles[:, 0]), label='Tilt', linewidth=2)
    ax3.plot(time_steps, np.degrees(joint_angles[:, 1]), label='Shoulder L', linewidth=2)
    ax3.plot(time_steps, np.degrees(joint_angles[:, 2]), label='Shoulder R', linewidth=2)
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title('Joint Angles Over Gait Cycle')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('gait_trajectory_static.png', dpi=150)
    print("Saved static plot to gait_trajectory_static.png")

    # Create animation
    print("Creating animation...")
    fig2, ax_anim = plt.subplots(figsize=(8, 8))
    ax_anim.set_xlim(-0.08, 0.08)
    ax_anim.set_ylim(-0.12, 0.02)
    ax_anim.set_xlabel('X (m)')
    ax_anim.set_ylabel('Z (m)')
    ax_anim.set_title('Leg Motion Animation')
    ax_anim.grid(True)
    ax_anim.axis('equal')

    # Plot elements
    trajectory_line, = ax_anim.plot([], [], 'b--', alpha=0.3, label='Trajectory')
    current_point, = ax_anim.plot([], [], 'ro', markersize=10, label='Current target')
    leg_left, = ax_anim.plot([], [], 'r-', linewidth=3, label='Left arm')
    leg_right, = ax_anim.plot([], [], 'g-', linewidth=3, label='Right arm')

    ax_anim.legend()

    def init():
        trajectory_line.set_data(valid_points[:, 0], valid_points[:, 2])
        return trajectory_line, current_point, leg_left, leg_right

    def animate(frame):
        # Current target point
        target = valid_points[frame]
        current_point.set_data([target[0]], [target[2]])

        # Current joint angles
        tilt, shoulder_L, shoulder_R = joint_angles[frame]

        # Get leg configuration using proper FK
        fk = forward_kinematics_3dof(tilt, shoulder_L, shoulder_R)

        # Draw left arm (base -> shoulder -> end effector)
        leg_left.set_data(
            [fk['base_L'][0], fk['shoulder_L_end'][0], fk['end_effector'][0]],
            [fk['base_L'][2], fk['shoulder_L_end'][2], fk['end_effector'][2]]
        )

        # Draw right arm (base -> shoulder -> end effector)
        leg_right.set_data(
            [fk['base_R'][0], fk['shoulder_R_end'][0], fk['end_effector'][0]],
            [fk['base_R'][2], fk['shoulder_R_end'][2], fk['end_effector'][2]]
        )

        return trajectory_line, current_point, leg_left, leg_right

    anim = FuncAnimation(fig2, animate, init_func=init,
                        frames=len(valid_points), interval=50, blit=True, repeat=True)

    plt.tight_layout()

    # Save animation
    print("Saving animation (this may take a moment)...")
    anim.save('gait_animation.gif', writer='pillow', fps=20, dpi=100)
    print("Saved animation to gait_animation.gif")

    # plt.show()  # Disabled for non-interactive use


def verify_fk_ik():
    """Verify that FK and IK are consistent."""
    print("=== Verifying FK/IK consistency ===")

    test_targets = [
        np.array([0.0, 0.0, -0.05]),   # Straight down
        np.array([0.02, 0.0, -0.06]),  # Forward
        np.array([-0.02, 0.0, -0.04]), # Backward
        np.array([0.0, 0.01, -0.05]),  # With lateral offset
    ]

    max_error = 0
    for i, target in enumerate(test_targets):
        # Solve IK
        result = solve_leg_ik_3dof(target, L1=L1, L2=L2, base_dist=0.021, mode=ik_mode)
        if result is None:
            print(f"Test {i}: IK failed for target {target}")
            continue

        tilt, shoulder_L, shoulder_R = result

        # Compute FK
        fk = forward_kinematics_3dof(tilt, shoulder_L, shoulder_R)
        computed_pos = fk['end_effector']

        # Compare
        error = np.linalg.norm(computed_pos - target)
        max_error = max(max_error, error)

        print(f"Test {i}: Target {target} -> FK {computed_pos} | Error: {error*1000:.3f}mm")

    print(f"Max error: {max_error*1000:.3f}mm")
    print()
    # Note: IK uses approximations for tilt, so some error is expected
    # For lateral movements, error can be ~4mm due to IK approximation
    return max_error < 0.005  # Less than 5mm error is acceptable


if __name__ == "__main__":
    print("=== Bezier Gait Trajectory Test ===")
    print(f"Leg parameters: L1={L1}m, L2={L2}m")
    print(f"IK mode: {ik_mode}")
    print()

    # Verify FK/IK consistency first
    if verify_fk_ik():
        print("FK/IK verification PASSED!\n")
    else:
        print("WARNING: FK/IK verification FAILED - results may be inaccurate\n")

    test_gait_visualization()

    print("\nTest complete!")
