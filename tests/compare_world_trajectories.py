#!/usr/bin/env python3
"""
Compare robot trajectories between world_train.xml and world.xml.
Records robot position every 0.1s for 20s in each world and plots the difference.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import List, Tuple

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from gait_controller import DiagonalGaitController, GaitParameters
from ik import solve_leg_ik_3dof


IK_PARAMS = dict(L1=0.045, L2=0.06, base_dist=0.021, mode=2)
FORWARD_SIGN = -1.0


class LegControl:
    def __init__(self, indices: Tuple[int, int, int], sign: float, offset: float):
        self.indices = indices
        self.sign = sign
        self.offset = offset


LEG_CONTROL = {
    "FL": LegControl(indices=(0, 1, 2), sign=-1.0, offset=-np.pi),
    "FR": LegControl(indices=(6, 7, 8), sign=1.0, offset=np.pi),
    "RL": LegControl(indices=(3, 4, 5), sign=-1.0, offset=-np.pi),
    "RR": LegControl(indices=(9, 10, 11), sign=1.0, offset=np.pi),
}

GAIT_PARAMS = GaitParameters(
    body_height=0.05,
    step_length=0.06,
    step_height=0.04,
    cycle_time=0.8
)


def apply_leg_angles(
    ctrl: mujoco.MjData,
    leg: str,
    angles: Tuple[float, float, float]
) -> None:
    """Map IK output angles into the actuator ordering."""
    tilt, ang_left, ang_right = angles
    config = LEG_CONTROL[leg]
    idx_left, idx_right, idx_tilt = config.indices
    sign = config.sign
    offset = config.offset

    ctrl.ctrl[idx_left] = sign * ang_left
    ctrl.ctrl[idx_right] = sign * ang_right + offset
    ctrl.ctrl[idx_tilt] = tilt


def apply_gait_targets(
    data: mujoco.MjData,
    controller: DiagonalGaitController,
    timestep: float
) -> None:
    """Evaluate the gait planner and push the resulting joint targets to MuJoCo."""
    leg_targets = controller.update(timestep)

    for leg in LEG_CONTROL:
        target = leg_targets.get(leg)
        if target is None:
            continue

        target_local = target.copy()
        target_local[0] *= FORWARD_SIGN

        result = solve_leg_ik_3dof(target_local, **IK_PARAMS)
        if result is None:
            print(f"[WARN] IK failed for leg {leg} with target {target}")
            continue

        apply_leg_angles(data, leg, result)


def run_simulation(
    world_file: str,
    duration: float = 20.0,
    record_interval: float = 0.1
) -> List[Tuple[float, float, float, float]]:
    """
    Run simulation and record robot position.

    Args:
        world_file: Path to MuJoCo XML world file
        duration: Total simulation time in seconds
        record_interval: Time between position recordings in seconds

    Returns:
        List of (time, x, y, z) tuples
    """
    print(f"Running simulation with {world_file}...")

    model = mujoco.MjModel.from_xml_path(world_file)
    data = mujoco.MjData(model)
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")

    controller = DiagonalGaitController(GAIT_PARAMS)
    controller.reset()

    positions = []
    sim_time = 0.0
    next_record_time = 0.0

    while sim_time < duration:
        # Apply gait control
        apply_gait_targets(data, controller, model.opt.timestep)

        # Step simulation
        mujoco.mj_step(model, data)
        sim_time = data.time

        # Record position at specified intervals
        if sim_time >= next_record_time:
            robot_pos = data.xpos[robot_body_id]
            positions.append((sim_time, robot_pos[0], robot_pos[1], robot_pos[2]))
            next_record_time += record_interval

    print(f"  Completed: {len(positions)} positions recorded")
    return positions


def plot_trajectories(
    positions_train: List[Tuple[float, float, float, float]],
    positions_normal: List[Tuple[float, float, float, float]]
) -> None:
    """Plot X-Y trajectories for both world files."""

    # Extract X and Y coordinates
    train_x = [p[1] for p in positions_train]
    train_y = [p[2] for p in positions_train]

    normal_x = [p[1] for p in positions_normal]
    normal_y = [p[2] for p in positions_normal]

    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(train_x, train_y, 'b-', linewidth=2, label='world_train.xml', alpha=0.7)
    plt.plot(normal_x, normal_y, 'r-', linewidth=2, label='world.xml', alpha=0.7)

    # Mark start and end points
    plt.plot(train_x[0], train_y[0], 'bo', markersize=10, label='Train start')
    plt.plot(train_x[-1], train_y[-1], 'bs', markersize=10, label='Train end')
    plt.plot(normal_x[0], normal_y[0], 'ro', markersize=10, label='Normal start')
    plt.plot(normal_x[-1], normal_y[-1], 'rs', markersize=10, label='Normal end')

    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title('Robot Trajectory Comparison: world_train.xml vs world.xml', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('tests/trajectory_comparison.png', dpi=150)
    print("Plot saved to tests/trajectory_comparison.png")
    plt.show()


def main() -> None:
    # Run simulation with world_train.xml
    positions_train = run_simulation("model/world_train.xml", duration=20.0, record_interval=0.1)

    # Run simulation with world.xml
    positions_normal = run_simulation("model/world.xml", duration=20.0, record_interval=0.1)

    # Plot comparison
    print("\nGenerating trajectory comparison plot...")
    plot_trajectories(positions_train, positions_normal)

    # Print summary statistics
    print("\nSummary:")
    print(f"  world_train.xml: traveled from ({positions_train[0][1]:.4f}, "
          f"{positions_train[0][2]:.4f}) to ({positions_train[-1][1]:.4f}, "
          f"{positions_train[-1][2]:.4f})")
    print(f"  world.xml:       traveled from ({positions_normal[0][1]:.4f}, "
          f"{positions_normal[0][2]:.4f}) to ({positions_normal[-1][1]:.4f}, "
          f"{positions_normal[-1][2]:.4f})")

    train_dist = np.sqrt((positions_train[-1][1] - positions_train[0][1])**2 +
                         (positions_train[-1][2] - positions_train[0][2])**2)
    normal_dist = np.sqrt((positions_normal[-1][1] - positions_normal[0][1])**2 +
                          (positions_normal[-1][2] - positions_normal[0][2])**2)

    print(f"\nTotal distance traveled:")
    print(f"  world_train.xml: {train_dist:.4f} m")
    print(f"  world.xml:       {normal_dist:.4f} m")


if __name__ == "__main__":
    main()
