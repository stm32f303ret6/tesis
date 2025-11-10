"""Control utilities shared by demos and environments.

Contains the leg actuator mapping and a helper to apply IK angles into
MuJoCo actuator control slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import mujoco


@dataclass(frozen=True)
class LegControl:
    indices: Tuple[int, int, int]
    sign: float
    offset: float


# Actuator mapping and sign/offsets matching the current robot.xml ordering.
LEG_CONTROL: Dict[str, LegControl] = {
    "FL": LegControl(indices=(0, 1, 2), sign=-1.0, offset=-np.pi),
    "FR": LegControl(indices=(6, 7, 8), sign=1.0, offset=np.pi),
    "RL": LegControl(indices=(3, 4, 5), sign=-1.0, offset=-np.pi),
    "RR": LegControl(indices=(9, 10, 11), sign=1.0, offset=np.pi),
}


def apply_leg_angles(ctrl: mujoco.MjData, leg: str, angles: Tuple[float, float, float]) -> None:
    """Map IK output angles into the actuator ordering.

    Args:
        ctrl: MuJoCo data; ``ctrl.ctrl`` is written with target actuator positions.
        leg: One of {"FL", "FR", "RL", "RR"}.
        angles: (tilt, shoulder_left, shoulder_right) in radians.
    """
    tilt, ang_left, ang_right = angles
    config = LEG_CONTROL[leg]
    idx_left, idx_right, idx_tilt = config.indices
    sign = config.sign
    offset = config.offset

    ctrl.ctrl[idx_left] = sign * float(ang_left)
    ctrl.ctrl[idx_right] = sign * float(ang_right) + float(offset)
    ctrl.ctrl[idx_tilt] = float(tilt)

