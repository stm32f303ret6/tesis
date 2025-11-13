"""Base environment for Walk2 3DOF parallel SCARA quadruped."""

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import ml_collections
from typing import Any, Dict
import pathlib

from mujoco_playground._src import mjx_env

from walk2_env import constants


class Walk2Env(mjx_env.MjxEnv):
    """Base environment for Walk2 quadruped robot.

    This environment provides the core functionality for the 3DOF parallel
    SCARA quadruped. Specific tasks (joystick, getup, etc.) should inherit
    from this class and override reset() and step() as needed.
    """

    def __init__(
        self,
        xml_path: str,
        config: ml_collections.ConfigDict,
    ):
        """Initialize Walk2 environment.

        Args:
            xml_path: Path to MuJoCo XML file
            config: Configuration dictionary
        """
        self._xml_path = xml_path
        self._config = config

        # Load XML
        with open(xml_path, 'r') as f:
            xml_string = f.read()

        # Get assets for mesh files
        assets = self._get_assets()

        # Compile MuJoCo models
        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_string,
            assets=assets
        )
        self._mjx_model = mjx.put_model(self._mj_model)

        # Configure simulation timestep
        self._mjx_model = self._mjx_model.replace(
            opt=self._mjx_model.opt.replace(timestep=config.sim_dt)
        )

        # Calculate number of substeps per control step
        self._n_substeps = int(config.control_dt / config.sim_dt)

        # Store home position (use model's default qpos0)
        self._home_qpos = jnp.array(self._mj_model.qpos0)

        # Find body and joint indices
        self._body_idx = mujoco.mj_name2id(
            self._mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            constants.BODY
        )

        # Find indices for actuated joints in qpos
        # (needed for applying actions and getting observations)
        self._actuated_joint_qpos_ids = []
        for i in range(self._mj_model.nu):
            joint_id = self._mj_model.actuator_trnid[i, 0]
            qpos_adr = self._mj_model.jnt_qposadr[joint_id]
            self._actuated_joint_qpos_ids.append(qpos_adr)
        self._actuated_joint_qpos_ids = jnp.array(self._actuated_joint_qpos_ids)

        print(f"Walk2Env initialized:")
        print(f"  XML: {xml_path}")
        print(f"  Actuators: {self.action_size}")
        print(f"  Control dt: {config.control_dt}s ({1/config.control_dt:.0f}Hz)")
        print(f"  Sim dt: {config.sim_dt}s ({1/config.sim_dt:.0f}Hz)")
        print(f"  Substeps: {self._n_substeps}")

    @staticmethod
    def _get_assets() -> Dict[str, bytes]:
        """Load assets for robot (meshes and XML includes).

        Note: robot.xml has meshdir="assets", so all assets must be
        prefixed with "assets/" in the dictionary keys.

        Returns:
            Dictionary mapping asset names to file contents
        """
        assets = {}
        model_dir = pathlib.Path(__file__).parent.parent / "model"

        # Load STL mesh files (with assets/ prefix due to meshdir in robot.xml)
        asset_dir = model_dir / "assets"
        if asset_dir.exists():
            for stl_file in asset_dir.glob("*.stl"):
                with open(stl_file, 'rb') as f:
                    # Prefix with "assets/" due to meshdir="assets" in robot.xml
                    assets[f"assets/{stl_file.name}"] = f.read()

        # Load robot.xml (included by world files)
        robot_xml_path = model_dir / "robot.xml"
        if robot_xml_path.exists():
            with open(robot_xml_path, 'r') as f:
                assets['robot.xml'] = f.read().encode('utf-8')

        # Load heightfield for terrain (also needs assets/ prefix)
        hfield_path = model_dir / "hfield.png"
        if hfield_path.exists():
            with open(hfield_path, 'rb') as f:
                assets['assets/hfield.png'] = f.read()

        return assets

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset environment to initial state.

        Args:
            rng: JAX random key

        Returns:
            Initial state
        """
        # Initialize MJX data with default qpos
        data = mjx.make_data(self._mjx_model)

        # Start with home qpos
        qpos = self._home_qpos.copy()

        # Add noise to actuated joint positions
        rng, noise_key = jax.random.split(rng)
        joint_noise = jax.random.normal(
            noise_key,
            shape=(self.action_size,)
        ) * self._config.randomization.init_qpos_noise

        # Apply noise to actuated joints
        qpos = qpos.at[self._actuated_joint_qpos_ids].add(joint_noise)
        data = data.replace(qpos=qpos)

        # Set initial velocities (small noise on all DOFs)
        rng, vel_key = jax.random.split(rng)
        qvel = jax.random.normal(
            vel_key,
            shape=(self._mjx_model.nv,)
        ) * self._config.randomization.init_qvel_noise
        data = data.replace(qvel=qvel)

        # Forward kinematics to compute derived quantities
        data = mjx.forward(self._mjx_model, data)

        # Create initial observation
        obs = self._get_obs(data, jnp.zeros(self.action_size))

        # Return initial state
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(False),
            metrics={},
            info={'rng': rng}
        )

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array
    ) -> mjx_env.State:
        """Execute one control step.

        Args:
            state: Current state
            action: Control action (12D, normalized to [-1, 1])

        Returns:
            Next state
        """
        # Compute target joint positions for actuated joints
        # Get home positions for actuated joints only
        home_actuated = self._home_qpos[self._actuated_joint_qpos_ids]
        target_actuated = home_actuated + action * self._config.action_scale

        # Execute substeps with PD control
        data = state.data
        for _ in range(self._n_substeps):
            # Get current actuated joint positions and velocities
            current_actuated = data.qpos[self._actuated_joint_qpos_ids]

            # For velocities, we need to map from qvel (which has different indexing)
            # For simplicity, we'll extract velocities for actuated joints
            # The velocity indices correspond to qpos indices after the free joint
            # Free joint: qpos[0:7] -> qvel[0:6] (3 linear + 3 angular)
            # Other joints: qpos[7+i] -> qvel[6+i]
            qvel_ids = jnp.where(
                self._actuated_joint_qpos_ids < 7,
                self._actuated_joint_qpos_ids,
                self._actuated_joint_qpos_ids - 1
            )
            current_vel = data.qvel[qvel_ids]

            # PD control: tau = Kp * (target - current) - Kd * velocity
            tau = (
                self._config.kp * (target_actuated - current_actuated) -
                self._config.kd * current_vel
            )

            # Apply control and step physics
            data = data.replace(ctrl=tau)
            data = mjx.step(self._mjx_model, data)

        # Compute observation
        obs = self._get_obs(data, action)

        # Check termination
        done = self._check_termination(data)

        # Compute reward (base class returns 0, override in subclasses)
        reward = self._compute_reward(data, action, state.data)

        # Update metrics
        metrics = self._compute_metrics(data)

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state.info
        )

    def _get_obs(
        self,
        data: mjx.Data,
        last_action: jax.Array
    ) -> jax.Array:
        """Extract observation from MuJoCo data.

        Observation includes:
        - Joint positions for actuated joints (12)
        - Joint velocities for actuated joints (12)
        - Body orientation (gravity vector, 3)
        - Angular velocity (3)
        - Last action (12)

        Total: 42 dimensions

        Args:
            data: MuJoCo data
            last_action: Previous action

        Returns:
            Observation array
        """
        # Get actuated joint positions and velocities
        actuated_qpos = data.qpos[self._actuated_joint_qpos_ids]
        home_actuated = self._home_qpos[self._actuated_joint_qpos_ids]

        # Map to velocity indices (same as in step())
        qvel_ids = jnp.where(
            self._actuated_joint_qpos_ids < 7,
            self._actuated_joint_qpos_ids,
            self._actuated_joint_qpos_ids - 1
        )
        actuated_qvel = data.qvel[qvel_ids]

        # Body orientation (gravity vector in body frame)
        # xmat is rotation matrix, column 2 is Z-axis in world frame
        body_xmat = data.xmat[self._body_idx].reshape(3, 3)
        gravity_vec = body_xmat.T @ jnp.array([0., 0., -1.])

        # Angular velocity in body frame
        body_angvel = data.cvel[self._body_idx, 3:6]

        # Concatenate observation
        obs = jnp.concatenate([
            actuated_qpos - home_actuated,  # Joint angles relative to home (12)
            actuated_qvel,                   # Joint velocities (12)
            gravity_vec,                     # Gravity in body frame (3)
            body_angvel,                     # Angular velocity (3)
            last_action,                     # Previous action (12)
        ])

        return obs

    def _check_termination(self, data: mjx.Data) -> jax.Array:
        """Check if episode should terminate.

        Terminates if:
        - Body height too low (fell)
        - Body tilted too much (rolled/pitched)

        Args:
            data: MuJoCo data

        Returns:
            Boolean termination flag
        """
        # Get body height
        body_height = data.xpos[self._body_idx, 2]

        # Get body orientation (gravity vector)
        body_xmat = data.xmat[self._body_idx].reshape(3, 3)
        gravity_vec = body_xmat.T @ jnp.array([0., 0., -1.])

        # Check orientation: gravity vector should point down
        # If Z-component < cos(threshold), robot is tipped over
        min_z_gravity = jnp.cos(self._config.termination.roll_threshold)

        # Termination conditions
        height_fail = body_height < self._config.termination.height_threshold
        orientation_fail = gravity_vec[2] < min_z_gravity

        done = height_fail | orientation_fail
        return done

    def _compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        prev_data: mjx.Data
    ) -> jax.Array:
        """Compute reward (base implementation returns 0).

        Subclasses should override this method to implement task-specific
        reward functions.

        Args:
            data: Current MuJoCo data
            action: Action taken
            prev_data: Previous MuJoCo data

        Returns:
            Scalar reward
        """
        return jnp.array(0.0)

    def _compute_metrics(self, data: mjx.Data) -> Dict[str, jax.Array]:
        """Compute metrics for logging.

        Args:
            data: MuJoCo data

        Returns:
            Dictionary of metrics
        """
        # Body height
        body_height = data.xpos[self._body_idx, 2]

        # Body velocity
        body_vel = data.cvel[self._body_idx, :3]
        speed = jnp.linalg.norm(body_vel[:2])  # Horizontal speed

        return {
            'body_height': body_height,
            'speed': speed,
            'forward_velocity': body_vel[0],
        }

    # Required properties for MjxEnv interface

    @property
    def xml_path(self) -> str:
        """Path to MuJoCo XML file."""
        return self._xml_path

    @property
    def action_size(self) -> int:
        """Number of actuators (12 for 4 legs × 3 DOF)."""
        return constants.NUM_ACTUATORS

    @property
    def mj_model(self) -> mujoco.MjModel:
        """MuJoCo model."""
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        """JAX-compiled MuJoCo model."""
        return self._mjx_model
