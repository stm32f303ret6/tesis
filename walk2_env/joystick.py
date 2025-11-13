"""Joystick task: velocity tracking for Walk2 quadruped."""

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
import ml_collections

from mujoco_playground._src import mjx_env
from walk2_env import base


class Walk2Joystick(base.Walk2Env):
    """Velocity tracking task for Walk2 quadruped.

    The robot receives velocity commands (vx, vy, wz) and is rewarded for
    tracking them while maintaining stability and energy efficiency.
    """

    def __init__(
        self,
        xml_path: str,
        config: ml_collections.ConfigDict,
    ):
        """Initialize joystick task.

        Args:
            xml_path: Path to MuJoCo XML file
            config: Configuration dictionary
        """
        super().__init__(xml_path, config)

        # Extract command ranges
        self._vx_range = jnp.array(config.command_ranges.vx)
        self._vy_range = jnp.array(config.command_ranges.vy)
        self._wz_range = jnp.array(config.command_ranges.wz)

        print(f"Walk2Joystick task initialized:")
        print(f"  vx range: {self._vx_range}")
        print(f"  vy range: {self._vy_range}")
        print(f"  wz range: {self._wz_range}")

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset and sample new velocity command.

        Args:
            rng: JAX random key

        Returns:
            Initial state with velocity command
        """
        # Call parent reset
        state = super().reset(rng)

        # Sample velocity command
        rng = state.info['rng']
        rng, cmd_key = jax.random.split(rng)

        cmd_rng = jax.random.split(cmd_key, 3)
        vx_cmd = jax.random.uniform(
            cmd_rng[0],
            minval=self._vx_range[0],
            maxval=self._vx_range[1]
        )
        vy_cmd = jax.random.uniform(
            cmd_rng[1],
            minval=self._vy_range[0],
            maxval=self._vy_range[1]
        )
        wz_cmd = jax.random.uniform(
            cmd_rng[2],
            minval=self._wz_range[0],
            maxval=self._wz_range[1]
        )

        command = jnp.array([vx_cmd, vy_cmd, wz_cmd])

        # Add command to observation
        obs_with_cmd = jnp.concatenate([state.obs, command])

        # Store command in info
        info = state.info.copy()
        info['command'] = command
        info['rng'] = rng

        return state.replace(obs=obs_with_cmd, info=info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array
    ) -> mjx_env.State:
        """Execute step with velocity tracking reward.

        Args:
            state: Current state
            action: Control action

        Returns:
            Next state
        """
        # Store previous data for reward computation
        prev_data = state.data

        # Execute physics step
        state = super().step(state, action)

        # Extract command from state
        command = state.info['command']

        # Compute velocity tracking reward
        reward = self._compute_reward(state.data, action, prev_data, command)

        # Add command to observation
        obs_with_cmd = jnp.concatenate([state.obs, command])

        # Compute additional metrics
        metrics = state.metrics.copy()
        metrics.update(self._compute_tracking_metrics(state.data, command))

        return state.replace(
            reward=reward,
            obs=obs_with_cmd,
            metrics=metrics
        )

    def _compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        prev_data: mjx.Data,
        command: jax.Array
    ) -> jax.Array:
        """Compute multi-component reward for velocity tracking.

        Reward components:
        1. Velocity tracking (exponential reward for matching command)
        2. Stability (penalize vertical motion, orientation errors)
        3. Energy (penalize torques, power, jerky motions)
        4. Height (maintain target body height)

        Args:
            data: Current MuJoCo data
            action: Current action
            prev_data: Previous MuJoCo data
            command: Velocity command [vx, vy, wz]

        Returns:
            Total reward (clipped and scaled by dt)
        """
        cfg = self._config.reward_scales

        # Get body velocity and angular velocity
        body_vel = data.cvel[self._body_idx, :3]  # Linear velocity
        body_angvel = data.cvel[self._body_idx, 3:6]  # Angular velocity

        vx_cmd, vy_cmd, wz_cmd = command

        # === Velocity tracking rewards ===
        vx_error = jnp.abs(body_vel[0] - vx_cmd)
        vy_error = jnp.abs(body_vel[1] - vy_cmd)
        wz_error = jnp.abs(body_angvel[2] - wz_cmd)

        # Exponential rewards (peak at 1.0 when error is 0)
        r_vx = jnp.exp(-5.0 * vx_error) * cfg.tracking_vx
        r_vy = jnp.exp(-5.0 * vy_error) * cfg.tracking_vy
        r_wz = jnp.exp(-5.0 * wz_error) * cfg.tracking_wz

        # === Stability penalties ===
        # Penalize vertical velocity (jumping/hopping)
        vz = body_vel[2]
        p_z_vel = jnp.abs(vz) * cfg.z_velocity

        # Penalize orientation errors (roll/pitch)
        # Gravity vector should point down in body frame
        body_xmat = data.xmat[self._body_idx].reshape(3, 3)
        gravity_vec = body_xmat.T @ jnp.array([0., 0., -1.])
        # Penalize deviation from [0, 0, -1]
        orientation_error = jnp.sum((gravity_vec - jnp.array([0., 0., -1.]))**2)
        p_orientation = orientation_error * cfg.orientation

        # Height tracking
        body_height = data.xpos[self._body_idx, 2]
        height_error = jnp.abs(body_height - self._config.target_height)
        p_height = height_error * cfg.height

        # === Energy penalties ===
        # Torque penalty (L2 norm of control inputs)
        torques = data.ctrl
        p_torque = jnp.sum(torques**2) * cfg.torque

        # Power penalty (torque * velocity)
        # Get velocities for actuated joints
        qvel_ids = jnp.where(
            self._actuated_joint_qpos_ids < 7,
            self._actuated_joint_qpos_ids,
            self._actuated_joint_qpos_ids - 1
        )
        actuated_qvel = data.qvel[qvel_ids]
        power = jnp.sum(jnp.abs(torques * actuated_qvel))
        p_power = power * cfg.power

        # Action rate penalty (encourage smooth actions)
        # For now, simplified: penalize large actions
        p_action_rate = jnp.sum(action**2) * cfg.action_rate

        # === Joint limits penalty ===
        # Penalize approaching joint limits (only for actuated joints)
        actuated_qpos = data.qpos[self._actuated_joint_qpos_ids]
        home_actuated = self._home_qpos[self._actuated_joint_qpos_ids]
        joint_deviation = jnp.sum((actuated_qpos - home_actuated)**2)
        p_joint_limits = joint_deviation * cfg.joint_limits

        # === Total reward ===
        reward_tracking = r_vx + r_vy + r_wz
        reward_stability = -(p_z_vel + p_orientation + p_height)
        reward_energy = -(p_torque + p_power + p_action_rate)
        reward_safety = -p_joint_limits

        total_reward = (
            reward_tracking +
            reward_stability +
            reward_energy +
            reward_safety
        )

        # Clip reward to reasonable range and scale by timestep
        reward = jnp.clip(total_reward, -10.0, 10.0) * self._config.control_dt

        return reward

    def _compute_tracking_metrics(
        self,
        data: mjx.Data,
        command: jax.Array
    ) -> dict:
        """Compute velocity tracking metrics.

        Args:
            data: MuJoCo data
            command: Velocity command

        Returns:
            Dictionary of tracking metrics
        """
        body_vel = data.cvel[self._body_idx, :3]
        body_angvel = data.cvel[self._body_idx, 3:6]

        vx_cmd, vy_cmd, wz_cmd = command

        return {
            'vx_error': jnp.abs(body_vel[0] - vx_cmd),
            'vy_error': jnp.abs(body_vel[1] - vy_cmd),
            'wz_error': jnp.abs(body_angvel[2] - wz_cmd),
            'vx_cmd': vx_cmd,
            'vy_cmd': vy_cmd,
            'wz_cmd': wz_cmd,
        }

    @property
    def observation_size(self) -> int:
        """Size of observation space.

        Base observation (42) + command (3) = 45
        """
        return 42 + 3  # Base obs + velocity command
