"""Utilities for reading and composing MuJoCo sensor data.

Frames/units follow MuJoCo defaults:
- Positions: meters in world frame (unless otherwise specified by sensor type)
- Quaternions: [w, x, y, z]
- Linear/angular velocities: m/s and rad/s in world frame
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import mujoco


class SensorReader:
    """Unified interface for reading robot sensors from MuJoCo.

    Build a mapping of sensor name to (address, dimension) and provide helpers
    to compose often-used state vectors.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.sensor_map: Dict[str, Tuple[int, int]] = {}
        self.sensor_objids: Dict[str, int] = {}
        self.sensor_types: Dict[str, int] = {}
        self._body_quat_body_id: Optional[int] = None
        self._build_sensor_map()

    def _build_sensor_map(self) -> None:
        self.sensor_map.clear()
        self.sensor_objids.clear()
        self.sensor_types.clear()
        self._body_quat_body_id = None
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr = int(self.model.sensor_adr[i])
            dim = int(self.model.sensor_dim[i])
            if name is not None:
                self.sensor_map[name] = (adr, dim)
                objid = int(self.model.sensor_objid[i])
                stype = int(self.model.sensor_type[i])
                self.sensor_objids[name] = objid
                self.sensor_types[name] = stype
                if (
                    name == "body_quat"
                    and stype == mujoco.mjtSensor.mjSENS_FRAMEQUAT
                    and objid >= 0
                ):
                    self._body_quat_body_id = objid

    def list_sensors(self) -> Tuple[str, ...]:
        return tuple(self.sensor_map.keys())

    def read_sensor(self, name: str) -> np.ndarray:
        if name not in self.sensor_map:
            raise ValueError(f"Sensor {name} not found in model")
        adr, dim = self.sensor_map[name]
        return self.data.sensordata[adr : adr + dim].copy()

    def get_body_state(self) -> np.ndarray:
        """Return [pos(3), quat(4), linvel(3), angvel(3)] = 13D."""
        pos = self.read_sensor("body_pos")
        quat = self.get_body_quaternion()
        linvel = self.read_sensor("body_linvel")
        angvel = self.read_sensor("body_angvel")
        return np.concatenate([pos, quat, linvel, angvel])

    def get_body_quaternion(self) -> np.ndarray:
        """Return body orientation quaternion [w, x, y, z] from model state."""
        if self._body_quat_body_id is not None:
            return self.data.xquat[self._body_quat_body_id].copy()
        return self.read_sensor("body_quat")

    def get_joint_states(self) -> np.ndarray:
        """Return all 12 joint positions + 12 velocities = 24D.

        Order per leg: tilt, shoulder_L, shoulder_R for {FL, FR, RL, RR}.
        """
        pos_names = [
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
        ]
        vel_names = [
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
        ]
        positions = [self.read_sensor(n) for n in pos_names]
        velocities = [self.read_sensor(n) for n in vel_names]
        return np.concatenate([np.concatenate(positions), np.concatenate(velocities)])

    def get_foot_positions(self) -> np.ndarray:
        """Return all 4 foot positions in world frame = 12D."""
        feet = [
            self.read_sensor("FL_foot_pos"),
            self.read_sensor("FR_foot_pos"),
            self.read_sensor("RL_foot_pos"),
            self.read_sensor("RR_foot_pos"),
        ]
        return np.concatenate(feet)

    def get_foot_velocities(self) -> np.ndarray:
        """Return all 4 foot linear velocities in world frame = 12D."""
        vels = [
            self.read_sensor("FL_foot_vel"),
            self.read_sensor("FR_foot_vel"),
            self.read_sensor("RL_foot_vel"),
            self.read_sensor("RR_foot_vel"),
        ]
        return np.concatenate(vels)
