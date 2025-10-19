#!/usr/bin/env python3
"""
Diagonal gait controller using transitions state machine.
Implements trotting gait where diagonal leg pairs move together:
- State 1: FL and RR swing (in air), FR and RL stance (on ground)
- State 2: FL and RR stance (on ground), FR and RL swing (in air)
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from transitions import Machine
from ik import solve_leg_ik_3dof


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


# Gait parameters
GAIT_PARAMS = {
    'num_points': 600,
    'step_height': 0.05,
    'step_length': 0.1,
    'stance_height': 0.02
}

# IK parameters
IK_MODE = 2
L1, L2 = 0.045, 0.06
BASE_DIST = 0.021

# Leg indices mapping
# Motor indices: RL(0-2), RR(3-5), FL(6-8), FR(9-11)
# Each leg: [shoulder_L, shoulder_R, tilt]
LEG_INDICES = {
    'FL': (6, 7, 8),   # Front Left
    'FR': (9, 10, 11), # Front Right
    'RL': (0, 1, 2),   # Rear Left
    'RR': (3, 4, 5)    # Rear Right
}


class DiagonalGaitController:
    """
    State machine-based gait controller for diagonal (trotting) gait.

    States:
    - pair1_swing: FL and RR are swinging, FR and RL are in stance
    - pair2_swing: FR and RL are swinging, FL and RR are in stance
    """

    states = ['pair1_swing', 'pair2_swing']

    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Validate gait parameters
        max_reach = L1 + L2
        step_length = GAIT_PARAMS['step_length']
        stance_height = abs(GAIT_PARAMS['stance_height'])
        step_height = GAIT_PARAMS['step_height']

        # Check worst case: maximum horizontal + vertical reach
        worst_case_dist = np.sqrt((step_length/2)**2 + stance_height**2)
        if worst_case_dist > max_reach * 0.95:  # 95% safety margin
            print(f"⚠ WARNING: Gait parameters may be unreachable!")
            print(f"  Max reach: {max_reach:.3f}m")
            print(f"  Worst case distance: {worst_case_dist:.3f}m")
            print(f"  Recommend: step_length < {0.04:.3f}m, stance_height > -{0.09:.3f}m")

        # Generate gait trajectory
        self.trajectory = generate_gait_trajectory(**GAIT_PARAMS)
        self.num_points = len(self.trajectory)
        self.swing_points = self.num_points // 2
        self.stance_points = self.num_points - self.swing_points

        # Split trajectory into swing and stance phases
        self.swing_trajectory = self.trajectory[:self.swing_points]
        self.stance_trajectory = self.trajectory[self.swing_points:]

        # Current trajectory indices for each leg (0-based within their current phase)
        self.leg_indices = {
            'FL': 0,  # Start at beginning of swing
            'FR': 0,  # Start at beginning of stance
            'RL': 0,  # Start at beginning of stance
            'RR': 0   # Start at beginning of swing
        }

        # Initialize state machine
        self.machine = Machine(
            model=self,
            states=DiagonalGaitController.states,
            initial='pair1_swing'
        )

        # Define transitions
        self.machine.add_transition(
            trigger='step',
            source='pair1_swing',
            dest='pair2_swing',
            after='on_enter_pair2_swing'
        )
        self.machine.add_transition(
            trigger='step',
            source='pair2_swing',
            dest='pair1_swing',
            after='on_enter_pair1_swing'
        )

        # Get robot body for camera tracking
        self.robot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "robot"
        )

    def on_enter_pair1_swing(self):
        """Called when entering pair1_swing state (FL and RR swing)."""
        # Reset trajectory indices to start of swing for FL and RR
        self.leg_indices['FL'] = 0
        self.leg_indices['RR'] = 0
        # Reset to start of stance for FR and RL
        self.leg_indices['FR'] = 0
        self.leg_indices['RL'] = 0

    def on_enter_pair2_swing(self):
        """Called when entering pair2_swing state (FR and RL swing)."""
        # Reset trajectory indices to start of swing for FR and RL
        self.leg_indices['FR'] = 0
        self.leg_indices['RL'] = 0
        # Reset to start of stance for FL and RR
        self.leg_indices['FL'] = 0
        self.leg_indices['RR'] = 0

    def get_leg_target(self, leg_name):
        """Get current target position for a specific leg."""
        idx = self.leg_indices[leg_name]

        if self.state == 'pair1_swing':
            # FL and RR are swinging
            if leg_name in ['FL', 'RR']:
                if idx < self.swing_points:
                    return self.swing_trajectory[idx]
                else:
                    return self.swing_trajectory[-1]  # Hold at end
            else:  # FR and RL are in stance
                if idx < self.stance_points:
                    return self.stance_trajectory[idx]
                else:
                    return self.stance_trajectory[-1]  # Hold at end

        else:  # pair2_swing
            # FR and RL are swinging
            if leg_name in ['FR', 'RL']:
                if idx < self.swing_points:
                    return self.swing_trajectory[idx]
                else:
                    return self.swing_trajectory[-1]
            else:  # FL and RR are in stance
                if idx < self.stance_points:
                    return self.stance_trajectory[idx]
                else:
                    return self.stance_trajectory[-1]

    def set_leg_position(self, leg_name, target_3d):
        """Set a single leg to target position using IK."""
        result = solve_leg_ik_3dof(
            target_3d,
            L1=L1,
            L2=L2,
            base_dist=BASE_DIST,
            mode=IK_MODE
        )

        if result is None:
            print(f"IK failed for {leg_name} at target {target_3d}")
            return False

        tilt, ang1L, ang1R = result
        idx_L, idx_R, idx_tilt = LEG_INDICES[leg_name]

        # Apply angles based on leg position (front vs rear)
        if leg_name in ['FL', 'FR']:  # Front legs
            self.data.ctrl[idx_L] = ang1L
            self.data.ctrl[idx_R] = ang1R + np.pi
            self.data.ctrl[idx_tilt] = tilt
        else:  # Rear legs (RL, RR)
            self.data.ctrl[idx_L] = -ang1L
            self.data.ctrl[idx_R] = -ang1R - np.pi
            self.data.ctrl[idx_tilt] = tilt

        return True

    def update(self):
        """Update all leg positions and advance trajectory indices."""
        # Update each leg to its current target
        for leg_name in ['FL', 'FR', 'RL', 'RR']:
            target = self.get_leg_target(leg_name)
            self.set_leg_position(leg_name, target)

        # Increment trajectory indices
        for leg_name in ['FL', 'FR', 'RL', 'RR']:
            self.leg_indices[leg_name] += 1

        # Check if we need to transition to next state
        if self.state == 'pair1_swing':
            # Check if FL and RR have completed swing phase
            if (self.leg_indices['FL'] >= self.swing_points and
                self.leg_indices['RR'] >= self.swing_points):
                self.step()  # Transition to pair2_swing

        else:  # pair2_swing
            # Check if FR and RL have completed swing phase
            if (self.leg_indices['FR'] >= self.swing_points and
                self.leg_indices['RL'] >= self.swing_points):
                self.step()  # Transition to pair1_swing


def main():
    """Run the diagonal gait controller."""
    # Load model
    model = mujoco.MjModel.from_xml_path("model/world.xml")
    data = mujoco.MjData(model)

    # Create gait controller
    controller = DiagonalGaitController(model, data)

    print("=== Diagonal Gait Controller ===")
    print(f"Gait parameters: {GAIT_PARAMS}")
    print(f"Trajectory points: {controller.num_points} "
          f"(swing: {controller.swing_points}, stance: {controller.stance_points})")
    print("\nDiagonal pairs:")
    print("  Pair 1: FL (Front Left) and RR (Rear Right)")
    print("  Pair 2: FR (Front Right) and RL (Rear Left)")
    print("\nStarting simulation...")
    print("Press Ctrl+C or close window to exit\n")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        frame_count = 0

        while viewer.is_running():
            # Update gait controller
            controller.update()

            # Step simulation
            mujoco.mj_step(model, data)

            # Update camera to follow robot
            robot_pos = data.xpos[controller.robot_body_id]
            viewer.cam.lookat[:] = robot_pos

            # Sync viewer
            viewer.sync()

            # Print state info every 100 frames
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"State: {controller.state:15s} | "
                      f"FL: {controller.leg_indices['FL']:3d} | "
                      f"FR: {controller.leg_indices['FR']:3d} | "
                      f"RL: {controller.leg_indices['RL']:3d} | "
                      f"RR: {controller.leg_indices['RR']:3d} | "
                      f"Time: {elapsed:.1f}s")

    print("\nSimulation finished")


if __name__ == "__main__":
    main()
