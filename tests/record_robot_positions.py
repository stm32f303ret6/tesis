#!/usr/bin/env python3
"""
Run the height-control gait for 10 seconds and record the robot body
position to a CSV at 10 Hz. This script mirrors the control path used
by `height_control.py` but runs headless without a viewer.

Output: `tests/robot_positions.csv` with columns: time,x,y,z
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Allow imports from repo root when executed from tests/
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mujoco  # type: ignore

from gait_controller import DiagonalGaitController, GaitParameters
from ik import solve_leg_ik_3dof


# Robot/IK configuration mirrors height_control.py
IK_PARAMS = dict(L1=0.045, L2=0.06, base_dist=0.021, mode=2)
FORWARD_SIGN = -1.0  # +1 keeps controller +X, -1 flips to match leg IK frame


@dataclass(frozen=True)
class LegControl:
    indices: Tuple[int, int, int]
    sign: float
    offset: float


LEG_CONTROL: Dict[str, LegControl] = {
    "FL": LegControl(indices=(0, 1, 2), sign=-1.0, offset=-np.pi),
    "FR": LegControl(indices=(6, 7, 8), sign=1.0, offset=np.pi),
    "RL": LegControl(indices=(3, 4, 5), sign=-1.0, offset=-np.pi),
    "RR": LegControl(indices=(9, 10, 11), sign=1.0, offset=np.pi),
}

GAIT_PARAMS = GaitParameters(
    body_height=0.05, step_length=0.06, step_height=0.04, cycle_time=0.8
)


def apply_leg_angles(ctrl: mujoco.MjData, leg: str, angles: Tuple[float, float, float]) -> None:
    """Map IK output angles into the actuator ordering."""
    tilt, ang_left, ang_right = angles
    config = LEG_CONTROL[leg]
    idx_left, idx_right, idx_tilt = config.indices
    sign = config.sign
    offset = config.offset

    ctrl.ctrl[idx_left] = sign * ang_left
    ctrl.ctrl[idx_right] = sign * ang_right + offset
    ctrl.ctrl[idx_tilt] = tilt


def apply_gait_targets(model: mujoco.MjModel, data: mujoco.MjData, controller: DiagonalGaitController) -> None:
    """Evaluate the gait planner and push the resulting joint targets to MuJoCo."""
    leg_targets = controller.update(model.opt.timestep)

    for leg in LEG_CONTROL:
        target = leg_targets.get(leg)
        if target is None:
            continue

        # Map controller forward direction to leg-local IK frame.
        target_local = target.copy()
        target_local[0] *= FORWARD_SIGN

        result = solve_leg_ik_3dof(target_local, **IK_PARAMS)
        if result is None:
            # Unreachable target; skip this leg for this step.
            continue

        apply_leg_angles(data, leg, result)


def main() -> None:
    # Load model and data
    model = mujoco.MjModel.from_xml_path("model/world_train.xml")
    data = mujoco.MjData(model)
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")

    controller = DiagonalGaitController(GAIT_PARAMS)
    controller.reset()

    # Logging setup
    out_path = Path(__file__).resolve().parent / "robot_positions.csv"
    sample_period = 0.1  # seconds (10 Hz)
    duration = 20.0  # seconds

    start_time = time.time()
    next_log_time = start_time

    with out_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time", "x", "y", "z"])  # header

        # Run the control loop in (approximately) real time
        while True:
            now = time.time()
            if now - start_time >= duration:
                break

            step_start = now

            apply_gait_targets(model, data, controller)
            mujoco.mj_step(model, data)

            # Log at 10 Hz
            if now >= next_log_time:
                pos = data.xpos[robot_body_id]
                writer.writerow([f"{now - start_time:.3f}", f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}"])
                next_log_time += sample_period

            # Sleep to match real-time with sim timestep
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

    print(f"Recorded robot positions to {out_path}")


if __name__ == "__main__":
    main()

