#!/usr/bin/env python3
import argparse
import time
from typing import Dict, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from gait_controller import DiagonalGaitController, GaitParameters
from ik import solve_leg_ik_3dof
from utils.control_utils import apply_leg_angles, LEG_CONTROL

IK_PARAMS = dict(L1=0.045, L2=0.06, base_dist=0.021, mode=2)
FORWARD_SIGN = -1.0  # +1 keeps controller +X, -1 flips to match leg IK frame

GAIT_PARAMS = GaitParameters(body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8)

# Global variables for MuJoCo model and data (initialized in main)
model = None
data = None
robot_body_id = None


def apply_gait_targets(controller: DiagonalGaitController, timestep: float) -> None:
    """Evaluate the gait planner and push the resulting joint targets to MuJoCo."""
    leg_targets = controller.update(timestep)

    for leg in LEG_CONTROL:
        target = leg_targets.get(leg)
        if target is None:
            continue

        # Map controller forward direction to leg-local IK frame.
        # Current robot geometry yields opposite X sense; flip to move forward.
        target_local = target.copy()
        target_local[0] *= FORWARD_SIGN

        result = solve_leg_ik_3dof(target_local, **IK_PARAMS)
        if result is None:
            print(f"[WARN] IK failed for leg {leg} with target {target}")
            continue

        apply_leg_angles(data, leg, result)


def main() -> None:
    global model, data, robot_body_id

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Standalone quadruped robot simulation with MuJoCo')
    parser.add_argument('--terrain', type=str, choices=['flat', 'rough'], default='rough',
                        help='Terrain type: flat or rough (default)')
    args = parser.parse_args()

    # Select world file based on terrain argument
    world_file = "model/world.xml" if args.terrain == 'flat' else "model/world_train.xml"
    print(f"Loading world: {world_file} ({args.terrain} terrain)")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(world_file)
    data = mujoco.MjData(model)
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")

    controller = DiagonalGaitController(GAIT_PARAMS)
    controller.reset()

    # Calculate expected velocity from gait parameters
    expected_velocity_x = GAIT_PARAMS.step_length / GAIT_PARAMS.cycle_time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            apply_gait_targets(controller, model.opt.timestep)
            mujoco.mj_step(model, data)

            robot_pos = data.xpos[robot_body_id]

            viewer.cam.lookat[:] = robot_pos
            viewer.sync()

            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

    print("Demo finished")


if __name__ == "__main__":
    main()
