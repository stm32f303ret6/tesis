"""Simplified Gymnasium environment for end-to-end quadruped locomotion learning.

The policy directly outputs 12 joint angles (no IK, no gait controller).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import mujoco

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

LEG_NAMES = ("FL", "FR", "RL", "RR")


class QuadrupedEnv(gym.Env):
    """End-to-end locomotion environment.

    Observation (49D):
        - body pos(3), quat(4), linvel(3), angvel(3) = 13D
        - joint pos(12), joint vel(12) = 24D
        - foot positions(12) = 12D

    Action (12D): joint position targets in [-1, 1], scaled to joint limits.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        model_path: str = "model/world_train.xml",
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # Build sensor map
        self._sensor_map: Dict[str, Tuple[int, int]] = {}
        self._build_sensor_map()

        # Action/observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        obs = self._get_observation()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        # Joint limits for scaling actions
        self.joint_range_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.joint_range_high = self.model.actuator_ctrlrange[:, 1].copy()

    def _build_sensor_map(self) -> None:
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                adr = int(self.model.sensor_adr[i])
                dim = int(self.model.sensor_dim[i])
                self._sensor_map[name] = (adr, dim)

    def _read_sensor(self, name: str) -> np.ndarray:
        adr, dim = self._sensor_map[name]
        return self.data.sensordata[adr:adr + dim].copy()

    def _get_observation(self) -> np.ndarray:
        # Body state (13D)
        pos = self._read_sensor("body_pos")
        quat = self._read_sensor("body_quat")
        quat = quat / max(1e-9, np.linalg.norm(quat))
        linvel = self._read_sensor("body_linvel")
        angvel = self._read_sensor("body_angvel")

        # Joint states (24D)
        joint_pos = np.array([
            self._read_sensor(f"{leg}_{j}_pos")[0]
            for leg in LEG_NAMES
            for j in ("tilt", "shoulder_L", "shoulder_R")
        ])
        joint_vel = np.array([
            self._read_sensor(f"{leg}_{j}_vel")[0]
            for leg in LEG_NAMES
            for j in ("tilt", "shoulder_L", "shoulder_R")
        ])

        # Foot positions (12D)
        foot_pos = np.concatenate([
            self._read_sensor(f"{leg}_foot_pos") for leg in LEG_NAMES
        ])

        obs = np.concatenate([
            pos, quat, linvel, angvel,
            joint_pos, joint_vel,
            foot_pos
        ]).astype(np.float32)

        return obs

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to joint limits."""
        action = np.clip(action, -1.0, 1.0)
        return 0.5 * (action + 1.0) * (self.joint_range_high - self.joint_range_low) + self.joint_range_low

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        rewards = {}

        # Forward velocity reward
        linvel = self._read_sensor("body_linvel")
        rewards["forward_vel"] = float(linvel[0])

        # Alive bonus
        rewards["alive"] = 1.0

        # Stability penalty (penalize excessive tilt)
        quat = self._read_sensor("body_quat")
        # Approximate tilt from quaternion
        w, x, y, z = quat
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(np.clip(2 * (w * y - z * x), -1, 1))
        rewards["stability"] = -0.5 * (roll ** 2 + pitch ** 2)

        # Energy penalty (penalize large control efforts)
        rewards["energy"] = -0.01 * float(np.sum(self.data.ctrl ** 2))

        total = sum(rewards.values())
        return float(total), rewards

    def _check_termination(self) -> Tuple[bool, bool]:
        quat = self._read_sensor("body_quat")
        w, x, y, z = quat
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(np.clip(2 * (w * y - z * x), -1, 1))

        terminated = abs(roll) > math.pi / 3 or abs(pitch) > math.pi / 3
        truncated = self.step_count >= self.max_episode_steps
        return bool(terminated), bool(truncated)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "initial")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        self.step_count = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray):
        # Apply scaled action to actuators
        self.data.ctrl[:] = self._scale_action(action)

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_observation()
        reward, info = self._compute_reward()
        terminated, truncated = self._check_termination()

        return obs, reward, terminated, truncated, {"reward_components": info}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
