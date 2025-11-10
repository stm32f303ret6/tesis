#!/usr/bin/env python3
"""Phase 1 verification: residual wrapper + sensors.

Writes summary to phases_test/phase1_controller_summary.json.
"""

import json
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import mujoco

# Ensure repository root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controllers.bezier_gait_residual import BezierGaitResidualController
from gait_controller import GaitParameters, LEG_NAMES
from utils.sensor_utils import SensorReader


OUT_PATH = Path(__file__).with_name("phase1_controller_summary.json")


def verify_residual_wrapper() -> Dict[str, object]:
    ctrl = BezierGaitResidualController(GaitParameters())
    ctrl.reset()

    # Zero residuals should reproduce base targets if dt = 0.0 (no phase advance)
    base_targets = ctrl.base_controller.update(0.0)
    zeros = {leg: np.zeros(3) for leg in LEG_NAMES}
    final_zero = ctrl.update_with_residuals(0.0, zeros)

    zero_preserve = True
    for leg in LEG_NAMES:
        if not np.allclose(base_targets[leg], final_zero[leg], atol=1e-9):
            zero_preserve = False
            break

    # Large residuals should be clipped to +/- residual_scale per axis
    huge = {leg: np.array([100.0, -200.0, 300.0]) for leg in LEG_NAMES}
    final_huge = ctrl.update_with_residuals(0.0, huge)

    max_abs_residual_applied = 0.0
    per_leg_max_abs_delta: Dict[str, float] = {}
    for leg in LEG_NAMES:
        delta = final_huge[leg] - base_targets[leg]
        per_leg_max_abs_delta[leg] = float(np.max(np.abs(delta)))
        max_abs_residual_applied = max(max_abs_residual_applied, per_leg_max_abs_delta[leg])

    clipping_ok = max_abs_residual_applied <= (ctrl.residual_scale + 1e-6)

    return {
        "zero_residuals_preserve_base": bool(zero_preserve),
        "residual_scale": float(ctrl.residual_scale),
        "max_abs_residual_applied": float(max_abs_residual_applied),
        "per_leg_max_abs_delta": per_leg_max_abs_delta,
        "clipping_ok": bool(clipping_ok),
    }


def verify_sensors() -> Dict[str, object]:
    model = mujoco.MjModel.from_xml_path("model/world.xml")
    data = mujoco.MjData(model)

    # One step to populate sensordata
    mujoco.mj_step(model, data)
    reader = SensorReader(model, data)

    expected = [
        "body_pos",
        "body_quat",
        "body_linvel",
        "body_angvel",
        "FL_tilt_pos",
        "FL_shoulder_L_pos",
        "FL_shoulder_R_pos",
        "FR_tilt_pos",
        "FR_shoulder_L_pos",
        "FR_shoulder_R_pos",
        "RL_tilt_pos",
        "RL_shoulder_L_pos",
        "RL_shoulder_R_pos",
        "RR_tilt_pos",
        "RR_shoulder_L_pos",
        "RR_shoulder_R_pos",
        "FL_tilt_vel",
        "FL_shoulder_L_vel",
        "FL_shoulder_R_vel",
        "FR_tilt_vel",
        "FR_shoulder_L_vel",
        "FR_shoulder_R_vel",
        "RL_tilt_vel",
        "RL_shoulder_L_vel",
        "RL_shoulder_R_vel",
        "RR_tilt_vel",
        "RR_shoulder_L_vel",
        "RR_shoulder_R_vel",
        "FL_foot_pos",
        "FR_foot_pos",
        "RL_foot_pos",
        "RR_foot_pos",
        "FL_foot_vel",
        "FR_foot_vel",
        "RL_foot_vel",
        "RR_foot_vel",
    ]

    sensors_ok = True
    missing = []
    for name in expected:
        try:
            _ = reader.read_sensor(name)
        except Exception:
            sensors_ok = False
            missing.append(name)

    # Quick dimension sanity
    dims_ok = True
    try:
        assert reader.read_sensor("body_pos").shape == (3,)
        assert reader.read_sensor("body_quat").shape == (4,)
        assert reader.read_sensor("FL_foot_pos").shape == (3,)
    except AssertionError:
        dims_ok = False

    return {
        "sensors_ok": bool(sensors_ok),
        "dims_ok": bool(dims_ok),
        "missing": missing,
        "observed_sensors": list(reader.list_sensors()),
    }


def main() -> None:
    summary = {}
    summary.update(verify_residual_wrapper())
    summary.update(verify_sensors())

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
