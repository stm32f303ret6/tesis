"""Gymnasium environment for residual-learning locomotion on rough terrain.

This environment wraps the diagonal Bézier gait controller with residual
corrections, runs IK to convert foot targets to joint angles, and steps
the MuJoCo simulation.

Frames/units
- Positions in meters, velocities in m/s, angles in radians.
- Quaternions follow MuJoCo order [w, x, y, z].
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import math
import os

import numpy as np
import mujoco

try:  # Gymnasium preferred
    import gymnasium as gym
except Exception:  # Fallback to classic gym if available
    import gym  # type: ignore

from gait_controller import LEG_NAMES, GaitParameters
from controllers.bezier_gait_residual import BezierGaitResidualController
from ik import solve_leg_ik_3dof
from utils.control_utils import apply_leg_angles
from utils.sensor_utils import SensorReader


# Keep in sync with height_control.py
IK_PARAMS = dict(L1=0.045, L2=0.06, base_dist=0.021, mode=2)
FORWARD_SIGN = -1.0


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = [float(q) for q in quat]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=float,
    )
    return R


def quat_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """Return roll, pitch, yaw from MuJoCo quaternion [w, x, y, z]."""
    w, x, y, z = [float(q) for q in quat]
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

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


class ResidualWalkEnv(gym.Env):  # type: ignore[misc]
    """Gymnasium-compatible env for PPO residual learning.

    Observation (~70–80D):
    - quat(4), lin vel(3), ang vel(3), projected gravity(3)
    - joint pos/vel(24), phase sin/cos(2), swing/stance(4)
    - prev action(12), command vel(1)

    Action (12D): residual per leg in [-1, 1] scaled by residual_scale.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str = "model/world_train.xml",
        gait_params: Optional[GaitParameters] = None,
        residual_scale: float = 0.02,
        target_velocity: float = 0.2,
        target_height: float = 0.05,
        max_episode_steps: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Load MuJoCo model/data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Components
        self.controller = BezierGaitResidualController(
            params=gait_params, residual_scale=residual_scale
        )
        self.sensor_reader = SensorReader(self.model, self.data)

        # Targets and scaling
        self.residual_scale = float(residual_scale)
        self.target_velocity = float(target_velocity)
        self.target_height = float(target_height)
        self.max_episode_steps = int(max_episode_steps)

        # RNG
        try:
            self.np_random, _ = gym.utils.seeding.np_random(seed)  # type: ignore[attr-defined]
        except Exception:
            self.np_random = np.random.default_rng(seed)

        # Spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Build a dummy observation to infer dimension
        self.previous_action = np.zeros(12, dtype=float)
        self.last_last_action: Optional[np.ndarray] = None
        self.step_count = 0
        obs = self._get_observation()
        obs_dim = int(obs.shape[0])
        high = np.ones(obs_dim, dtype=np.float32) * np.inf
        low = -high
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Precompute body ids for contact checks (from body names)
        self._foot_contact_bodies = self._init_foot_contact_bodies()

    # ----------------------------- Helpers ------------------------------
    def _init_foot_contact_bodies(self) -> Dict[str, Tuple[int, int]]:
        """Return mapping leg->(elbow_left_bodyid, elbow_right_bodyid)."""
        mapping: Dict[str, Tuple[int, int]] = {}
        for leg in ("FL", "FR", "RL", "RR"):
            bl = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_elbow_left"
            )
            br = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_elbow_right"
            )
            mapping[leg] = (int(bl) if bl != -1 else -1, int(br) if br != -1 else -1)
        return mapping

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs_components = []

        # Body state
        body_state = self.sensor_reader.get_body_state()
        pos, quat, linvel, angvel = np.split(body_state, [3, 7, 10])
        quat = quat / max(1e-9, np.linalg.norm(quat))  # Normalize defensively

        obs_components.append(quat)  # 4D
        obs_components.append(linvel)  # 3D
        obs_components.append(angvel)  # 3D

        # Projected gravity in body frame
        gravity_world = np.array([0.0, 0.0, -1.0])
        R = quat_to_rotation_matrix(quat)
        gravity_body = R.T @ gravity_world
        obs_components.append(gravity_body)  # 3D

        # Joint states (24D)
        joint_states = self.sensor_reader.get_joint_states()
        obs_components.append(joint_states)

        # Gait phase (sin/cos)
        phase_info = self.controller.get_phase_info()
        phase = float(phase_info.get("phase_normalized", 0.0))
        obs_components.append(
            np.array([math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)])
        )  # 2D

        # Swing/stance flags
        flags = self.controller.get_swing_stance_flags()
        flag_array = np.array([flags[leg] for leg in LEG_NAMES], dtype=float)
        obs_components.append(flag_array)  # 4D

        # Previous actions
        obs_components.append(self.previous_action.astype(float))  # 12D

        # Command velocity
        obs_components.append(np.array([self.target_velocity], dtype=float))  # 1D

        obs = np.concatenate(obs_components).astype(np.float32)
        # Ensure finite
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("Non-finite values in observation")
        return obs

    def _process_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert normalized action [-1, 1] to residual dict per leg."""
        act = np.asarray(action, dtype=float).clip(-1.0, 1.0)
        residuals: Dict[str, np.ndarray] = {}
        for i, leg in enumerate(LEG_NAMES):
            residuals[leg] = act[i * 3 : (i + 1) * 3] * self.residual_scale
        return residuals

    def _get_foot_contact_forces(self) -> Dict[str, float]:
        """Heuristic contact indicator per leg (0.0 or 1.0).

        Uses contact pairs: if any contact involves a geom whose body is one of
        the leg's elbow bodies, mark contact as present (1.0).
        """
        forces = {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}
        if self.data.ncon <= 0:
            return forces

        # Map geoms to bodies for quick lookup
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
        """Compute reward and component breakdown for logging."""
        rewards: Dict[str, float] = {}

        # 1. Forward velocity tracking (primary objective)
        linvel = self.sensor_reader.read_sensor("body_linvel")
        forward_vel = float(linvel[0])
        vel_error = abs(forward_vel - self.target_velocity)
        rewards["forward_velocity"] = 1.0 - vel_error  # [~ -inf, 1]

        # 2. Lateral stability (penalize sideways drift)
        lateral_vel = abs(float(linvel[1]))
        rewards["lateral_stability"] = -0.5 * lateral_vel

        # 3. Body height maintenance
        body_pos = self.sensor_reader.read_sensor("body_pos")
        height_error = abs(float(body_pos[2]) - self.target_height)
        rewards["height"] = -2.0 * height_error

        # 4. Body orientation (stay upright)
        quat = self.sensor_reader.read_sensor("body_quat")
        roll, pitch, _ = quat_to_euler(quat)
        tilt_penalty = roll * roll + pitch * pitch
        rewards["orientation"] = -3.0 * float(tilt_penalty)

        # 5. Energy efficiency (penalize large actions)
        action_magnitude = float(np.linalg.norm(self.previous_action))
        rewards["energy"] = -0.1 * action_magnitude

        # 6. Smooth control (penalize action changes)
        if self.last_last_action is not None:
            action_change = float(np.linalg.norm(self.previous_action - self.last_last_action))
            rewards["smoothness"] = -0.2 * action_change
        else:
            rewards["smoothness"] = 0.0

        # 7. Foot contact pattern (encourage proper gait)
        foot_contacts = self._get_foot_contact_forces()
        swing_flags = self.controller.get_swing_stance_flags()
        contact_reward = 0.0
        for leg in LEG_NAMES:
            is_swing = int(swing_flags[leg]) == 1
            has_contact = foot_contacts[leg] > 0.5
            if is_swing and has_contact:
                contact_reward -= 0.5
            elif (not is_swing) and has_contact:
                contact_reward += 0.1
        rewards["contact_pattern"] = float(contact_reward)

        # 8. Joint limits penalty (soft)
        joint_positions = self.sensor_reader.get_joint_states()[:12]
        limit_violations = float(np.sum(np.abs(joint_positions) > 2.5))
        rewards["joint_limits"] = -1.0 * limit_violations

        total = float(sum(rewards.values()))
        return total, rewards

    def _check_termination(self) -> Tuple[bool, bool]:
        """Return (terminated, truncated)."""
        body_pos = self.sensor_reader.read_sensor("body_pos")
        quat = self.sensor_reader.read_sensor("body_quat")
        roll, pitch, _ = quat_to_euler(quat)
        terminated = bool(
            (float(body_pos[2]) < 0.03)
            or (abs(roll) > math.pi / 3)
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

        # Reset MuJoCo state and apply keyframe if available
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "initial")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Optional randomization for robustness
        if options and options.get("randomize", False):
            # Random body XY
            self.data.qpos[0] += float(self.np_random.uniform(-0.05, 0.05))  # type: ignore[attr-defined]
            self.data.qpos[1] += float(self.np_random.uniform(-0.05, 0.05))  # type: ignore[attr-defined]
            # Random orientation
            roll = float(self.np_random.uniform(-0.1, 0.1))  # type: ignore[attr-defined]
            pitch = float(self.np_random.uniform(-0.1, 0.1))  # type: ignore[attr-defined]
            yaw = float(self.np_random.uniform(-math.pi, math.pi))  # type: ignore[attr-defined]
            self.data.qpos[3:7] = euler_to_quat(roll, pitch, yaw)

        # Reset controller and tracking vars
        self.controller.reset()
        self.step_count = 0
        self.previous_action = np.zeros(12, dtype=float)
        self.last_last_action = None

        # Forward a few steps to settle
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_observation()
        info: Dict[str, float] = {}
        return obs, info

    def step(self, action: np.ndarray):
        # Process action
        residuals = self._process_action(action)

        # Controller + residuals → final foot targets
        final_targets = self.controller.update_with_residuals(
            self.model.opt.timestep, residuals
        )

        # IK + apply controls per leg
        for leg in LEG_NAMES:
            target = np.asarray(final_targets[leg], dtype=float)
            target_local = target.copy()
            target_local[0] *= FORWARD_SIGN

            result = solve_leg_ik_3dof(target_local, **IK_PARAMS)
            if result is not None:
                apply_leg_angles(self.data, leg, result)

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Reward
        reward, reward_info = self._compute_reward()

        # Termination
        terminated, truncated = self._check_termination()

        # Tracking
        self.step_count += 1
        self.last_last_action = None if self.previous_action is None else self.previous_action.copy()
        self.previous_action = np.asarray(action, dtype=float).copy()

        # Observation and info
        obs = self._get_observation()
        info = {
            "reward_components": reward_info,
            "body_height": float(self.sensor_reader.read_sensor("body_pos")[2]),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ----------------------- Curriculum Support ------------------------
    def set_terrain_scale(self, scale: float) -> None:
        """Placeholder to scale terrain difficulty.

        For future use: would require updating heightfield data or reloading the
        model. Here we simply store the value for compatibility with callbacks.
        """
        self.terrain_scale = float(scale)
