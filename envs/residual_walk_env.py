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
from scipy.spatial.transform import Rotation as R

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


def quat_to_euler(quat: np.ndarray, degrees) -> Tuple[float, float, float]: # quat w,x,y,z ordering
    """
    Convert a MuJoCo quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw) in radians.
    """
    # SciPy expects [x, y, z, w], so reorder
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


class ResidualWalkEnv(gym.Env):  # type: ignore[misc]
    """Gymnasium-compatible env for PPO residual learning.

    Observation (65D):
    - body pos(3), quat(4), lin vel(3), ang vel(3)
    - joint pos/vel(24)
    - foot pos(12), foot vel(12)
    - foot contacts(4)

    Action (12D): residual per leg in [-1, 1] scaled by residual_scale.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str = "model/world_train.xml",
        gait_params: Optional[GaitParameters] = None,
        residual_scale: float = 0.02,
        max_episode_steps: int = 1000,
        settle_steps: int = 500,
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

        # Derive expected velocity from gait parameters
        params = self.controller.base_controller.params
        self.expected_velocity = float(params.step_length / params.cycle_time)

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
        """Construct observation vector (65D).

        Returns:
            - body pos(3), quat(4), lin vel(3), ang vel(3)
            - joint pos/vel(24)
            - foot pos(12), foot vel(12)
            - foot contacts(4)
        """
        obs_components = []

        # Body state: position, orientation, velocities
        body_state = self.sensor_reader.get_body_state()
        pos, quat, linvel, angvel = np.split(body_state, [3, 7, 10])
        quat = quat / max(1e-9, np.linalg.norm(quat))  # Normalize defensively

        obs_components.append(pos)     # 3D - body position
        obs_components.append(quat)    # 4D - body orientation
        obs_components.append(linvel)  # 3D - linear velocity
        obs_components.append(angvel)  # 3D - angular velocity

        # Joint states: positions and velocities (24D)
        joint_states = self.sensor_reader.get_joint_states()
        obs_components.append(joint_states)

        # Foot positions and velocities (12D + 12D = 24D)
        foot_positions = self.sensor_reader.get_foot_positions()
        obs_components.append(foot_positions)
        foot_velocities = self.sensor_reader.get_foot_velocities()
        obs_components.append(foot_velocities)

        # Foot contacts: binary flags for ground contact (4D)
        foot_contacts = self._get_foot_contact_forces()
        contact_array = np.array([foot_contacts[leg] for leg in LEG_NAMES], dtype=float)
        obs_components.append(contact_array)

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
        """Compute simplified reward focused on velocity and contact pattern.

        Velocity and height targets are derived from gait parameters, not arbitrary.
        Expected velocity = step_length / cycle_time from the base gait.
        """
        rewards: Dict[str, float] = {}

        # 1. Forward velocity tracking (primary objective)
        # Track the expected velocity from gait parameters
        linvel = self.sensor_reader.read_sensor("body_linvel")
        forward_vel = float(linvel[0])
        vel_error = abs(forward_vel - self.expected_velocity)
        rewards["forward_velocity"] = 5.0 * (1.0 - vel_error)

        # 2. Foot contact pattern (encourage proper gait)
        foot_contacts = self._get_foot_contact_forces()
        swing_flags = self.controller.get_swing_stance_flags()
        contact_reward = 0.0
        for leg in LEG_NAMES:
            is_swing = int(swing_flags[leg]) == 1
            has_contact = foot_contacts[leg] > 0.5
            if is_swing and has_contact:
                # Swing leg touching ground (BAD)
                contact_reward -= 0.2
            elif is_swing and not has_contact:
                # Swing leg in air (GOOD)
                contact_reward += 0.05
            elif (not is_swing) and has_contact:
                # Stance leg on ground (GOOD)
                contact_reward += 0.05
            else:  # (not is_swing) and (not has_contact)
                # Stance leg floating in air (BAD)
                contact_reward -= 0.2
        rewards["contact_pattern"] = float(contact_reward)

        # 3. Mild stability penalties (to avoid catastrophic failures)
        quat = self.sensor_reader.get_body_quaternion()
        roll, pitch, yaw = quat_to_euler(quat, True)
        tilt_penalty = roll * roll + pitch * pitch + yaw * yaw
        rewards["stability"] = -1 * float(tilt_penalty)

        # Penalize deviation from initial standing height (measured after settle)
        body_pos = self.sensor_reader.read_sensor("body_pos")
        if self.initial_body_height is not None:
            height_error = abs(float(body_pos[2]) - self.initial_body_height)
            rewards["height"] = -0.1 * height_error
        else:
            rewards["height"] = 0.0
            
        # Penalize lateral (y-axis) deviation from initial position (secondary lateral control)
        lateral_error = abs(float(body_pos[1]))
        rewards["lateral_stability"] = -5.0 * lateral_error

        total = float(sum(rewards.values()))
        return total, rewards

    def _check_termination(self) -> Tuple[bool, bool]:
        """Return (terminated, truncated)."""
        body_pos = self.sensor_reader.read_sensor("body_pos")
        quat = self.sensor_reader.get_body_quaternion()
        roll, pitch, _ = quat_to_euler(quat, False)
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

        # Settle on rough terrain: run gait controller with zero residuals
        # This allows the robot to stabilize its height and contact pattern
        zero_residuals = {leg: np.zeros(3) for leg in LEG_NAMES}
        for _ in range(self.settle_steps):
            # Update controller to advance gait phase
            final_targets = self.controller.update_with_residuals(
                self.model.opt.timestep, zero_residuals
            )

            # Apply IK and step simulation
            for leg in LEG_NAMES:
                target = np.asarray(final_targets[leg], dtype=float)
                target_local = target.copy()
                target_local[0] *= FORWARD_SIGN

                result = solve_leg_ik_3dof(target_local, **IK_PARAMS)
                if result is not None:
                    apply_leg_angles(self.data, leg, result)

            mujoco.mj_step(self.model, self.data)

        # Capture initial body height after settling
        # This becomes our reference for height tracking during the episode
        body_pos = self.sensor_reader.read_sensor("body_pos")
        self.initial_body_height = float(body_pos[2])

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
