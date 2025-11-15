"""Gymnasium environment for adaptive gait + residual learning.

This extends ResidualWalkEnv to support online gait parameter adaptation.
The policy learns to modulate both:
1. High-level gait parameters (step_height, step_length, cycle_time, body_height)
2. Low-level residual corrections (per-leg foot offsets)

This approach enables better adaptation to rough terrain by learning when to
take higher steps, adjust stride length, change gait frequency, etc.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import math
import numpy as np
import mujoco

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

from gait_controller import LEG_NAMES, GaitParameters
from controllers.adaptive_gait_controller import AdaptiveGaitController
from ik import solve_leg_ik_3dof
from utils.control_utils import apply_leg_angles
from utils.sensor_utils import SensorReader
from scipy.spatial.transform import Rotation as R

# Keep in sync with height_control.py
IK_PARAMS = dict(L1=0.045, L2=0.06, base_dist=0.021, mode=2)
FORWARD_SIGN = -1.0


def quat_to_euler(quat: np.ndarray, degrees: bool) -> Tuple[float, float, float]:
    """Convert MuJoCo quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw)."""
    quat_scipy = [quat[1], quat[2], quat[3], quat[0]]
    r = R.from_quat(quat_scipy)
    roll, pitch, yaw = r.as_euler('xyz', degrees=degrees)
    return roll, pitch, yaw


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to MuJoCo quaternion [w, x, y, z]."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=float)


class AdaptiveGaitEnv(gym.Env):  # type: ignore[misc]
    """Environment for learning adaptive gait parameters + residuals.

    Observation (69D):
    - body pos(3), quat(4), lin vel(3), ang vel(3)  [13D]
    - joint pos/vel(24)                              [24D]
    - foot pos(12), foot vel(12)                     [24D]
    - foot contacts(4)                               [4D]
    - current gait parameters(4)                     [4D]

    Action (16D):
    - gait parameter deltas(4): [d_step_height, d_step_length, d_cycle_time, d_body_height]
    - residuals per leg(12): 3D offset per leg in [-1, 1] scaled by residual_scale
    """

    metadata = {"render_modes": []}

    # Scale for parameter deltas (action in [-1, 1] scaled to reasonable delta)
    PARAM_DELTA_SCALES = {
        "step_height": 0.005,   # ±5mm per step
        "step_length": 0.005,   # ±5mm per step
        "cycle_time": 0.05,     # ±50ms per step
        "body_height": 0.003,   # ±3mm per step
    }

    def __init__(
        self,
        model_path: str = "model/world_train.xml",
        gait_params: Optional[GaitParameters] = None,
        residual_scale: float = 0.01,
        max_episode_steps: int = 6000,
        settle_steps: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Load MuJoCo model/data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Components
        self.controller = AdaptiveGaitController(
            base_params=gait_params,
            residual_scale=residual_scale,
        )
        self.sensor_reader = SensorReader(self.model, self.data)

        # Initial body height (set during reset)
        self.initial_body_height: Optional[float] = None

        # Scaling and episode config
        self.residual_scale = float(residual_scale)
        self.max_episode_steps = int(max_episode_steps)
        self.settle_steps = int(settle_steps)

        # RNG
        try:
            self.np_random, _ = gym.utils.seeding.np_random(seed)  # type: ignore[attr-defined]
        except Exception:
            self.np_random = np.random.default_rng(seed)

        # Spaces
        # Action: [param_deltas(4), residuals(12)]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )

        # Build observation space
        self.previous_action = np.zeros(16, dtype=float)
        self.step_count = 0
        obs = self._get_observation()
        obs_dim = int(obs.shape[0])
        high = np.ones(obs_dim, dtype=np.float32) * np.inf
        low = -high
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Precompute body ids for contact checks
        self._foot_contact_bodies = self._init_foot_contact_bodies()

    # ----------------------------- Helpers ------------------------------
    def _init_foot_contact_bodies(self) -> Dict[str, Tuple[int, int]]:
        """Return mapping leg->(elbow_left_bodyid, elbow_right_bodyid)."""
        mapping: Dict[str, Tuple[int, int]] = {}
        for leg in LEG_NAMES:
            bl = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_elbow_left"
            )
            br = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_elbow_right"
            )
            mapping[leg] = (int(bl) if bl != -1 else -1, int(br) if br != -1 else -1)
        return mapping

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector (69D).

        Returns:
            - body pos(3), quat(4), lin vel(3), ang vel(3)  [13D]
            - joint pos/vel(24)                              [24D]
            - foot pos(12), foot vel(12)                     [24D]
            - foot contacts(4)                               [4D]
            - current gait parameters(4)                     [4D]
        """
        obs_components = []

        # Body state
        body_state = self.sensor_reader.get_body_state()
        pos, quat, linvel, angvel = np.split(body_state, [3, 7, 10])
        quat = quat / max(1e-9, np.linalg.norm(quat))

        obs_components.append(pos)
        obs_components.append(quat)
        obs_components.append(linvel)
        obs_components.append(angvel)

        # Joint states
        joint_states = self.sensor_reader.get_joint_states()
        obs_components.append(joint_states)

        # Foot positions and velocities
        foot_positions = self.sensor_reader.get_foot_positions()
        obs_components.append(foot_positions)
        foot_velocities = self.sensor_reader.get_foot_velocities()
        obs_components.append(foot_velocities)

        # Foot contacts
        foot_contacts = self._get_foot_contact_forces()
        contact_array = np.array([foot_contacts[leg] for leg in LEG_NAMES], dtype=float)
        obs_components.append(contact_array)

        # Current gait parameters (normalized to [-1, 1] range)
        # Normalized as: (value - center) / scale
        # Where center = (min + max) / 2 and scale = (max - min) / 2
        # This maps [min, max] to [-1, 1] automatically based on controller ranges
        current_params = self.controller.get_current_parameters()
        ranges = self.controller.param_ranges

        param_obs = []
        for param_name in ["step_height", "step_length", "cycle_time", "body_height"]:
            min_val, max_val, _ = ranges[param_name]
            center = (min_val + max_val) / 2.0
            scale = (max_val - min_val) / 2.0
            normalized = (current_params[param_name] - center) / scale
            param_obs.append(normalized)

        obs_components.append(np.array(param_obs, dtype=float))

        obs = np.concatenate(obs_components).astype(np.float32)
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("Non-finite values in observation")
        return obs

    def _process_action(self, action: np.ndarray) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Convert normalized action to parameter deltas and residuals.

        Args:
            action: 16D array in [-1, 1]

        Returns:
            (param_deltas, residuals)
        """
        act = np.asarray(action, dtype=float).clip(-1.0, 1.0)

        # First 4 values are parameter deltas
        param_raw = act[:4]
        param_deltas = {
            "step_height": param_raw[0] * self.PARAM_DELTA_SCALES["step_height"],
            "step_length": param_raw[1] * self.PARAM_DELTA_SCALES["step_length"],
            "cycle_time": param_raw[2] * self.PARAM_DELTA_SCALES["cycle_time"],
            "body_height": param_raw[3] * self.PARAM_DELTA_SCALES["body_height"],
        }

        # Remaining 12 values are residuals
        residuals: Dict[str, np.ndarray] = {}
        for i, leg in enumerate(LEG_NAMES):
            residuals[leg] = act[4 + i * 3 : 4 + (i + 1) * 3] * self.residual_scale

        return param_deltas, residuals

    def _get_foot_contact_forces(self) -> Dict[str, float]:
        """Contact indicator per leg (0.0 or 1.0)."""
        forces = {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}
        if self.data.ncon <= 0:
            return forces

        geom_body = self.model.geom_bodyid

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1 = int(geom_body[g1]) if g1 >= 0 else -1
            b2 = int(geom_body[g2]) if g2 >= 0 else -1
            for leg, (bl, br) in self._foot_contact_bodies.items():
                if bl == -1 and br == -1:
                    continue
                if b1 in (bl, br) or b2 in (bl, br):
                    forces[leg] = 1.0
        return forces

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """Compute reward focused on forward velocity and gait quality."""
        rewards: Dict[str, float] = {}

        # 1. Forward velocity reward (simple progress reward)
        linvel = self.sensor_reader.read_sensor("body_linvel")
        forward_vel = float(linvel[0])
        lateral_vel = float(linvel[1])

        # Reward forward velocity
        rewards["forward_velocity"] = 5.0 * forward_vel

        # Penalize lateral velocity (drift)
        rewards["lateral_velocity_penalty"] = -2.0 * abs(lateral_vel)

        # 2. Foot contact pattern
        foot_contacts = self._get_foot_contact_forces()
        swing_flags = self.controller.get_swing_stance_flags()
        contact_reward = 0.0
        for leg in LEG_NAMES:
            is_swing = int(swing_flags[leg]) == 1
            has_contact = foot_contacts[leg] > 0.5
            if is_swing and has_contact:
                contact_reward -= 0.2
            elif is_swing and not has_contact:
                contact_reward += 0.05
            elif (not is_swing) and has_contact:
                contact_reward += 0.05
            else:
                contact_reward -= 0.2
        rewards["contact_pattern"] = float(contact_reward)

        # 3. Stability
        quat = self.sensor_reader.get_body_quaternion()
        roll, pitch, yaw = quat_to_euler(quat, True)
        tilt_penalty = roll * roll + pitch * pitch + yaw * yaw
        rewards["stability"] = -0.1 * float(tilt_penalty)



        total = float(sum(rewards.values()))
        return total, rewards

    def _check_termination(self) -> Tuple[bool, bool]:
        """Return (terminated, truncated)."""
        quat = self.sensor_reader.get_body_quaternion()
        roll, pitch, _ = quat_to_euler(quat, False)
        terminated = bool(
            (abs(roll) > math.pi / 3)
            or (abs(pitch) > math.pi / 3)
        )
        truncated = bool(self.step_count >= self.max_episode_steps)
        return terminated, truncated

    # ----------------------------- API ---------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            try:
                self.np_random, _ = gym.utils.seeding.np_random(seed)  # type: ignore[attr-defined]
            except Exception:
                self.np_random = np.random.default_rng(seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "initial")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Optional randomization
        if options and options.get("randomize", False):
            self.data.qpos[0] += float(self.np_random.uniform(-0.05, 0.05))  # type: ignore
            self.data.qpos[1] += float(self.np_random.uniform(-0.05, 0.05))  # type: ignore
            roll = float(self.np_random.uniform(-0.1, 0.1))  # type: ignore
            pitch = float(self.np_random.uniform(-0.1, 0.1))  # type: ignore
            yaw = float(self.np_random.uniform(-math.pi, math.pi))  # type: ignore
            self.data.qpos[3:7] = euler_to_quat(roll, pitch, yaw)

        # Reset controller
        self.controller.reset()
        self.step_count = 0
        self.previous_action = np.zeros(16, dtype=float)

        # Settle with base gait controller
        zero_residuals = {leg: np.zeros(3) for leg in LEG_NAMES}
        for _ in range(self.settle_steps):
            final_targets = self.controller.update_with_residuals(
                self.model.opt.timestep, zero_residuals
            )

            for leg in LEG_NAMES:
                target = np.asarray(final_targets[leg], dtype=float)
                target_local = target.copy()
                target_local[0] *= FORWARD_SIGN

                result = solve_leg_ik_3dof(target_local, **IK_PARAMS)
                if result is not None:
                    apply_leg_angles(self.data, leg, result)

            mujoco.mj_step(self.model, self.data)

        # Capture initial body height
        body_pos = self.sensor_reader.read_sensor("body_pos")
        self.initial_body_height = float(body_pos[2])

        obs = self._get_observation()
        info: Dict[str, float] = {}
        return obs, info

    def step(self, action: np.ndarray):
        # Process action into parameter deltas and residuals
        param_deltas, residuals = self._process_action(action)

        # Update controller with both parameter deltas and residuals
        final_targets = self.controller.update_with_residuals(
            self.model.opt.timestep, residuals, param_deltas
        )

        # IK and apply controls
        for leg in LEG_NAMES:
            target = np.asarray(final_targets[leg], dtype=float)
            target_local = target.copy()
            target_local[0] *= FORWARD_SIGN

            result = solve_leg_ik_3dof(target_local, **IK_PARAMS)
            if result is not None:
                apply_leg_angles(self.data, leg, result)

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Reward and termination
        reward, reward_info = self._compute_reward()
        terminated, truncated = self._check_termination()

        # Tracking
        self.step_count += 1
        self.previous_action = np.asarray(action, dtype=float).copy()

        # Observation and info
        obs = self._get_observation()
        info = {
            "reward_components": reward_info,
            "body_height": float(self.sensor_reader.read_sensor("body_pos")[2]),
            "gait_params": self.controller.get_current_parameters(),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info
