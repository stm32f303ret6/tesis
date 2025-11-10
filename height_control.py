#!/usr/bin/env python3
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

model = mujoco.MjModel.from_xml_path("model/world.xml")
data = mujoco.MjData(model)
robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")


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
    controller = DiagonalGaitController(GAIT_PARAMS)
    controller.reset()

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
