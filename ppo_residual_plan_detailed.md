# PPO Residual Learning Implementation Plan
## Detailed Design for Quadruped Rough Terrain Locomotion

---

## Executive Summary

This document outlines a complete implementation plan for training a PPO-based residual learning system to enable your quadruped robot to traverse rough terrain. The approach augments your existing Bézier curve gait controller with learned residual corrections:

```
final_foot_position = bezier_trajectory_position + ppo_residual
```

**Key advantages of residual learning:**
- Preserves proven flat-ground locomotion from your Bézier controller
- Reduces learning complexity (RL only learns terrain-specific corrections)
- Enables smooth degradation (if policy fails, base controller still functions)
- Faster training convergence compared to learning from scratch

---

## Project Architecture Overview

### Current System (Flat Ground)
```
height_control.py
    ↓
DiagonalGaitController → Bézier swing/stance trajectories
    ↓
solve_leg_ik_3dof() → Joint angles (tilt, shoulder_L, shoulder_R)
    ↓
apply_leg_angles() → MuJoCo actuators → Robot motion
```

### Target System (Rough Terrain with Residuals)
```
ResidualWalkEnv (Gymnasium)
    ↓
Observation: IMU, joints, foot contacts, terrain → PPO Policy → Residual actions (12D)
    ↓
DiagonalGaitController → base_targets
    ↓
Residual addition: final_targets = base_targets + residuals
    ↓
solve_leg_ik_3dof() → Joint angles
    ↓
MuJoCo simulation → Reward computation → Policy update
```

---

## Phase 1: Code Refactoring & Modularization

### 1.1 Create Reusable Controller Module

**File:** `controllers/bezier_gait_residual.py`

**Purpose:** Wrap `DiagonalGaitController` to support residual injection and expose state queries needed by the RL policy.

**Key Changes:**
```python
class BezierGaitResidualController:
    """Wraps DiagonalGaitController with residual support."""

    def __init__(self, params: GaitParameters):
        self.base_controller = DiagonalGaitController(params)
        self.residual_scale = 0.02  # Max residual magnitude in meters

    def update_with_residuals(
        self,
        dt: float,
        residuals: Dict[str, np.ndarray]  # {"FL": [dx,dy,dz], ...}
    ) -> Dict[str, np.ndarray]:
        """
        Get base Bézier targets and add residual corrections.

        Args:
            dt: Timestep
            residuals: Per-leg residual offsets in meters [x, y, z]

        Returns:
            Dict of final foot targets per leg
        """
        # Get nominal Bézier targets
        base_targets = self.base_controller.update(dt)

        # Add clipped residuals
        final_targets = {}
        for leg in LEG_NAMES:
            base = base_targets[leg]
            res = residuals.get(leg, np.zeros(3))
            res_clipped = np.clip(res, -self.residual_scale, self.residual_scale)
            final_targets[leg] = base + res_clipped

        return final_targets

    def get_phase_info(self) -> Dict[str, float]:
        """Return gait phase information for observation."""
        return {
            "phase_elapsed": self.base_controller.phase_elapsed,
            "state_duration": self.base_controller.state_duration,
            "phase_normalized": self.base_controller.phase_elapsed /
                               self.base_controller.state_duration,
            "active_pair": 0 if self.base_controller.state == "pair_a_swing" else 1
        }

    def get_swing_stance_flags(self) -> Dict[str, int]:
        """Return 1 for swing legs, 0 for stance legs."""
        flags = {}
        for leg in LEG_NAMES:
            flags[leg] = 1 if leg in self.base_controller.active_swing_pair else 0
        return flags
```

**Implementation notes:**
- Keep the existing `DiagonalGaitController` unchanged for backward compatibility
- Add configurable `residual_scale` parameter (start with 2cm = 0.02m)
- Expose phase and swing/stance info needed for the RL observation space
- Implement workspace clamping to prevent IK failures

### 1.2 Create Sensor Utilities Module

**File:** `utils/sensor_utils.py`

**Purpose:** Extract and normalize sensor data from MuJoCo for RL observations.

```python
import numpy as np
import mujoco

class SensorReader:
    """Unified interface for reading robot sensors from MuJoCo."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._build_sensor_map()

    def _build_sensor_map(self):
        """Create name->index mapping for all sensors."""
        self.sensor_map = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, i
            )
            adr = self.model.sensor_adr[i]
            dim = self.model.sensor_dim[i]
            self.sensor_map[name] = (adr, dim)

    def read_sensor(self, name: str) -> np.ndarray:
        """Read sensor by name."""
        if name not in self.sensor_map:
            raise ValueError(f"Sensor {name} not found")
        adr, dim = self.sensor_map[name]
        return self.data.sensordata[adr:adr+dim].copy()

    def get_body_state(self) -> np.ndarray:
        """Return [pos(3), quat(4), linvel(3), angvel(3)] = 13D."""
        pos = self.read_sensor("body_pos")
        quat = self.read_sensor("body_quat")
        linvel = self.read_sensor("body_linvel")
        angvel = self.read_sensor("body_angvel")
        return np.concatenate([pos, quat, linvel, angvel])

    def get_joint_states(self) -> np.ndarray:
        """Return all 12 joint positions + 12 velocities = 24D."""
        joint_names = [
            "FL_shoulder_L", "FL_shoulder_R", "FL_tilt",
            "FR_shoulder_L", "FR_shoulder_R", "FR_tilt",
            "RL_shoulder_L", "RL_shoulder_R", "RL_tilt",
            "RR_shoulder_L", "RR_shoulder_R", "RR_tilt",
        ]
        positions = []
        velocities = []
        for name in joint_names:
            positions.append(self.read_sensor(f"{name}_pos"))
            velocities.append(self.read_sensor(f"{name}_vel"))
        return np.concatenate([np.concatenate(positions),
                              np.concatenate(velocities)])

    def get_foot_positions(self) -> np.ndarray:
        """Return all 4 foot positions in world frame = 12D."""
        feet = []
        for leg in ["FL", "FR", "RL", "RR"]:
            feet.append(self.read_sensor(f"{leg}_foot_pos"))
        return np.concatenate(feet)

    def get_foot_velocities(self) -> np.ndarray:
        """Return all 4 foot velocities = 12D."""
        vels = []
        for leg in ["FL", "FR", "RL", "RR"]:
            vels.append(self.read_sensor(f"{leg}_foot_vel"))
        return np.concatenate(vels)
```

**Why this matters:**
- Your `robot.xml` already has comprehensive sensors (lines 224-276)
- This module provides a clean interface for the RL environment
- Simplifies debugging (can easily log/visualize sensor values)

---

## Phase 2: Gymnasium Environment Implementation

### 2.1 Environment Core Structure

**File:** `envs/residual_walk_env.py`

**Class hierarchy:**
```python
gym.Env
    ↓
ResidualWalkEnv
    ↓
    - model: MjModel (from world_train.xml)
    - data: MjData
    - controller: BezierGaitResidualController
    - sensor_reader: SensorReader
```

### 2.2 Observation Space Design

**Total dimension: ~70-80D** (adjust based on what you need)

| Component | Dimension | Range | Description |
|-----------|-----------|-------|-------------|
| **Body orientation** | 4 | [-1, 1] | Quaternion (normalized) |
| **Body lin velocity** | 3 | [-5, 5] m/s | Body velocity in world frame |
| **Body ang velocity** | 3 | [-10, 10] rad/s | Roll/pitch/yaw rates |
| **Projected gravity** | 3 | [-1, 1] | Gravity in body frame (orientation info) |
| **Joint positions** | 12 | [-π, π] | All actuated joint angles |
| **Joint velocities** | 12 | [-20, 20] rad/s | Joint angular velocities |
| **Gait phase** | 2 | [0, 1] | [sin(phase), cos(phase)] for continuity |
| **Swing/stance flags** | 4 | {0, 1} | Binary flags per leg |
| **Previous actions** | 12 | [-1, 1] | Last residual commands (helps smoothness) |
| **Foot heights (optional)** | 4 | [0, 0.2] m | Current foot Z positions |
| **Command velocity (optional)** | 1 | [0, 1] m/s | Desired forward speed |

**Implementation:**
```python
def _get_observation(self) -> np.ndarray:
    """Construct observation vector."""
    obs_components = []

    # Body state
    body_state = self.sensor_reader.get_body_state()
    pos, quat, linvel, angvel = np.split(body_state, [3, 7, 10])

    obs_components.append(quat)  # 4D
    obs_components.append(linvel)  # 3D
    obs_components.append(angvel)  # 3D

    # Projected gravity (better than Euler angles for RL)
    gravity_world = np.array([0, 0, -1])
    rot_mat = quat_to_rotation_matrix(quat)
    gravity_body = rot_mat.T @ gravity_world
    obs_components.append(gravity_body)  # 3D

    # Joint states
    joint_states = self.sensor_reader.get_joint_states()
    obs_components.append(joint_states)  # 24D

    # Gait phase (encoded as sin/cos for continuity)
    phase_info = self.controller.get_phase_info()
    phase = phase_info["phase_normalized"]
    obs_components.append(np.array([
        np.sin(2 * np.pi * phase),
        np.cos(2 * np.pi * phase)
    ]))  # 2D

    # Swing/stance flags
    flags = self.controller.get_swing_stance_flags()
    flag_array = np.array([flags[leg] for leg in LEG_NAMES])
    obs_components.append(flag_array)  # 4D

    # Previous actions (initialize to zeros at reset)
    obs_components.append(self.previous_action)  # 12D

    # Command velocity
    obs_components.append(np.array([self.target_velocity]))  # 1D

    return np.concatenate(obs_components)
```

### 2.3 Action Space Design

**Dimension: 12D** (3D residual per foot × 4 legs)

```python
self.action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(12,),
    dtype=np.float32
)
```

**Action interpretation:**
- Action is in [-1, 1] range (normalized)
- Scaled by `residual_scale` (default 0.02m = 2cm)
- Order: `[FL_x, FL_y, FL_z, FR_x, FR_y, FR_z, RL_x, RL_y, RL_z, RR_x, RR_y, RR_z]`
- Applied in leg-local coordinate frame (same as Bézier targets)

**Action processing:**
```python
def _process_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert normalized action to residual dict."""
    residuals = {}
    for i, leg in enumerate(LEG_NAMES):
        residuals[leg] = action[i*3:(i+1)*3] * self.residual_scale
    return residuals
```

### 2.4 Reward Function Design

**Critical for training success!** Use a weighted combination:

```python
def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
    """Compute reward and component breakdown for logging."""
    rewards = {}

    # 1. Forward velocity tracking (primary objective)
    linvel = self.sensor_reader.read_sensor("body_linvel")
    forward_vel = linvel[0]  # X-axis is forward
    vel_error = abs(forward_vel - self.target_velocity)
    rewards["forward_velocity"] = 1.0 - vel_error  # [0, 1]

    # 2. Lateral stability (penalize sideways drift)
    lateral_vel = abs(linvel[1])
    rewards["lateral_stability"] = -0.5 * lateral_vel

    # 3. Body height maintenance
    body_pos = self.sensor_reader.read_sensor("body_pos")
    height_error = abs(body_pos[2] - self.target_height)
    rewards["height"] = -2.0 * height_error

    # 4. Body orientation (stay upright)
    quat = self.sensor_reader.read_sensor("body_quat")
    roll, pitch, yaw = quat_to_euler(quat)
    tilt_penalty = roll**2 + pitch**2
    rewards["orientation"] = -3.0 * tilt_penalty

    # 5. Energy efficiency (penalize large actions)
    action_magnitude = np.linalg.norm(self.previous_action)
    rewards["energy"] = -0.1 * action_magnitude

    # 6. Smooth control (penalize action changes)
    if self.last_last_action is not None:
        action_change = np.linalg.norm(
            self.previous_action - self.last_last_action
        )
        rewards["smoothness"] = -0.2 * action_change
    else:
        rewards["smoothness"] = 0.0

    # 7. Foot contact pattern (encourage proper gait)
    foot_contacts = self._get_foot_contact_forces()
    swing_flags = self.controller.get_swing_stance_flags()
    contact_reward = 0.0
    for leg in LEG_NAMES:
        is_swing = swing_flags[leg]
        has_contact = foot_contacts[leg] > 1.0  # Threshold in Newtons
        # Penalize contacts during swing, reward during stance
        if is_swing and has_contact:
            contact_reward -= 0.5
        elif not is_swing and has_contact:
            contact_reward += 0.1
    rewards["contact_pattern"] = contact_reward

    # 8. Joint limits penalty
    joint_positions = self.sensor_reader.get_joint_states()[:12]
    limit_violations = np.sum(np.abs(joint_positions) > 2.5)  # Near limits
    rewards["joint_limits"] = -1.0 * limit_violations

    # Total weighted reward
    total = sum(rewards.values())

    return total, rewards

def _get_foot_contact_forces(self) -> Dict[str, float]:
    """Estimate contact forces from MuJoCo contact data."""
    forces = {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}

    # Get relevant geom IDs for foot tips
    foot_geom_ids = {
        "FL": [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "FL_elbow_left"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "FL_elbow_right")
        ],
        # ... similar for other legs
    }

    # Iterate through contacts
    for i in range(self.data.ncon):
        contact = self.data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2

        # Check if either geom is a foot
        for leg, geom_ids in foot_geom_ids.items():
            if geom1 in geom_ids or geom2 in geom_ids:
                # Sum normal force
                force = np.linalg.norm(contact.frame[:3])
                forces[leg] += force

    return forces
```

**Reward tuning tips:**
- Start with sparse rewards (forward velocity only)
- Gradually add terms as training stabilizes
- Monitor per-component rewards in TensorBoard
- Adjust weights based on priority (e.g., safety > efficiency)

### 2.5 Termination Conditions

```python
def _check_termination(self) -> Tuple[bool, bool]:
    """
    Returns:
        (terminated, truncated) following Gymnasium convention
    """
    # Termination (failure)
    body_pos = self.sensor_reader.read_sensor("body_pos")
    quat = self.sensor_reader.read_sensor("body_quat")
    roll, pitch, _ = quat_to_euler(quat)

    terminated = (
        body_pos[2] < 0.03 or  # Body too low (crashed)
        abs(roll) > np.pi/3 or  # Excessive roll
        abs(pitch) > np.pi/3    # Excessive pitch
    )

    # Truncation (time limit)
    truncated = self.step_count >= self.max_episode_steps

    return terminated, truncated
```

### 2.6 Reset Implementation

```python
def reset(self, seed=None, options=None):
    """Reset environment to initial state."""
    super().reset(seed=seed)

    # Reset MuJoCo state
    mujoco.mj_resetData(self.model, self.data)

    # Optionally add randomization for robustness
    if options and options.get("randomize", False):
        # Random initial body position offset
        self.data.qpos[0] += self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[1] += self.np_random.uniform(-0.05, 0.05)
        # Random initial orientation
        roll = self.np_random.uniform(-0.1, 0.1)
        pitch = self.np_random.uniform(-0.1, 0.1)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[3:7] = euler_to_quat(roll, pitch, yaw)

    # Reset controller
    self.controller.reset()

    # Reset tracking variables
    self.step_count = 0
    self.previous_action = np.zeros(12)
    self.last_last_action = None

    # Forward simulation to settle
    for _ in range(10):
        mujoco.mj_step(self.model, self.data)

    obs = self._get_observation()
    info = {}

    return obs, info
```

### 2.7 Step Function

```python
def step(self, action: np.ndarray):
    """Execute one environment step."""
    # Process action to residuals
    residuals = self._process_action(action)

    # Get final foot targets from controller + residuals
    final_targets = self.controller.update_with_residuals(
        self.model.opt.timestep,
        residuals
    )

    # Apply IK and control
    for leg in LEG_NAMES:
        target = final_targets[leg]

        # Apply forward sign flip (from height_control.py)
        target_local = target.copy()
        target_local[0] *= FORWARD_SIGN

        # Solve IK
        result = solve_leg_ik_3dof(target_local, **IK_PARAMS)
        if result is not None:
            apply_leg_angles(self.data, leg, result)
        # else: keep previous angles (graceful degradation)

    # Step simulation
    mujoco.mj_step(self.model, self.data)

    # Compute reward
    reward, reward_info = self._compute_reward()

    # Check termination
    terminated, truncated = self._check_termination()

    # Update tracking
    self.step_count += 1
    self.last_last_action = self.previous_action.copy()
    self.previous_action = action.copy()

    # Get next observation
    obs = self._get_observation()

    # Info dict for logging
    info = {
        "reward_components": reward_info,
        "body_height": self.sensor_reader.read_sensor("body_pos")[2],
        "forward_velocity": self.sensor_reader.read_sensor("body_linvel")[0],
    }

    return obs, reward, terminated, truncated, info
```

---

## Phase 3: Training Pipeline

### 3.1 Dependencies

Add to `requirements.txt`:
```
gymnasium>=0.29.0
stable-baselines3>=2.0.0
tensorboard>=2.14.0
shimmy>=1.3.0  # For Gymnasium-Gym compatibility
```

Install:
```bash
pip install gymnasium stable-baselines3 tensorboard shimmy
```

### 3.2 Training Script

**File:** `train_residual_ppo.py`

```python
#!/usr/bin/env python3
"""Train PPO policy with residual actions for rough terrain locomotion."""

import os
from datetime import datetime
from typing import Dict

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.residual_walk_env import ResidualWalkEnv


def make_env(rank: int, seed: int = 0):
    """Create environment with unique seed."""
    def _init():
        env = ResidualWalkEnv()
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def main():
    # Configuration
    config = {
        "total_timesteps": 5_000_000,
        "n_envs": 4,  # Parallel environments
        "learning_rate": 3e-4,
        "n_steps": 2048,  # Steps per environment per update
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # Entropy bonus for exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": 42,
    }

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/residual_ppo_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Training output: {log_dir}")

    # Create vectorized environments
    env = DummyVecEnv([make_env(i, config["seed"]) for i in range(config["n_envs"])])

    # Wrap with normalization (critical for stable training!)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config["gamma"],
    )

    # Create evaluation environment (single, normalized separately)
    eval_env = DummyVecEnv([make_env(0, config["seed"] + 1000)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize eval rewards
        clip_obs=10.0,
        training=False,  # Fixed statistics from training
    )

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        tensorboard_log=log_dir,
        policy_kwargs={
            "net_arch": [256, 256, 128],  # 3-layer MLP
            "activation_fn": torch.nn.ReLU,
        },
        verbose=1,
        seed=config["seed"],
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,  # Save every 50k steps
        save_path=f"{log_dir}/checkpoints",
        name_prefix="ppo_residual",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/eval",
        eval_freq=10_000,  # Evaluate every 10k steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # Train!
    print("Starting training...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback_list,
        progress_bar=True,
    )

    # Save final model
    model.save(f"{log_dir}/final_model")
    env.save(f"{log_dir}/vec_normalize.pkl")

    print(f"Training complete! Model saved to {log_dir}")


if __name__ == "__main__":
    main()
```

**Training tips:**
- Start with 4 parallel environments, scale up to 8-16 if training is slow
- Use `tensorboard --logdir runs/` to monitor training
- Watch for reward plateaus (may need curriculum learning)
- First 500k steps are often unstable - don't panic if robot falls frequently

### 3.3 Curriculum Learning (Optional but Recommended)

**File:** `callbacks/curriculum_callback.py`

```python
from stable_baselines3.common.callbacks import BaseCallback

class TerrainCurriculumCallback(BaseCallback):
    """Gradually increase terrain difficulty during training."""

    def __init__(
        self,
        start_scale: float = 0.1,
        end_scale: float = 1.0,
        total_steps: int = 2_000_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        # Linearly interpolate terrain difficulty
        progress = min(1.0, self.num_timesteps / self.total_steps)
        current_scale = (
            self.start_scale +
            (self.end_scale - self.start_scale) * progress
        )

        # Update all environments
        for env in self.training_env.envs:
            env.env_method("set_terrain_scale", current_scale)

        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"Terrain scale: {current_scale:.2f}")

        return True
```

Add method to `ResidualWalkEnv`:
```python
def set_terrain_scale(self, scale: float):
    """Scale terrain height for curriculum learning."""
    self.terrain_scale = scale
    # Modify heightfield in MuJoCo model
    # (requires reloading or dynamic modification)
```

---

## Phase 4: Evaluation & Visualization

### 4.1 Playback Script

**File:** `play_residual_policy.py`

```python
#!/usr/bin/env python3
"""Load trained policy and visualize in MuJoCo."""

import argparse
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.residual_walk_env import ResidualWalkEnv


def play_policy(
    model_path: str,
    normalize_path: Optional[str] = None,
    num_episodes: int = 3,
    render: bool = True,
):
    """Load and run trained policy."""

    # Load model
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Create environment
    env = ResidualWalkEnv()

    # Wrap with normalization if provided
    if normalize_path:
        print(f"Loading normalization stats from {normalize_path}")
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(normalize_path, vec_env)
        vec_env.training = False  # Disable updates
        vec_env.norm_reward = False
    else:
        vec_env = DummyVecEnv([lambda: env])

    # Run episodes
    for episode in range(num_episodes):
        obs = vec_env.reset()
        episode_reward = 0
        step = 0

        print(f"\n=== Episode {episode + 1} ===")

        while True:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            step += 1

            # Render
            if render:
                time.sleep(0.001)  # Match simulation speed

            if done[0]:
                break

        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Steps: {step}")
        print(f"Final height: {info[0].get('body_height', 0):.3f}m")
        print(f"Avg velocity: {info[0].get('forward_velocity', 0):.3f}m/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="Path to normalization stats (.pkl)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )

    args = parser.parse_args()

    play_policy(
        model_path=args.model,
        normalize_path=args.normalize,
        num_episodes=args.episodes,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Play best model with normalization
python play_residual_policy.py \
    --model runs/residual_ppo_20231110_143022/best_model/best_model.zip \
    --normalize runs/residual_ppo_20231110_143022/vec_normalize.pkl \
    --episodes 5

# Quick test without normalization
python play_residual_policy.py --model runs/.../final_model.zip --no-render
```

### 4.2 Comparison Script

**File:** `tests/compare_residual_vs_baseline.py`

```python
#!/usr/bin/env python3
"""Compare residual policy vs. baseline Bézier controller."""

import matplotlib.pyplot as plt
import numpy as np

from envs.residual_walk_env import ResidualWalkEnv
from stable_baselines3 import PPO


def run_comparison(model_path: str, num_episodes: int = 10):
    """Run baseline and residual policies, plot results."""

    env = ResidualWalkEnv()
    model = PPO.load(model_path)

    results = {
        "baseline": {"distances": [], "falls": 0},
        "residual": {"distances": [], "falls": 0},
    }

    for mode in ["baseline", "residual"]:
        print(f"\n=== Testing {mode} ===")
        env.use_residuals = (mode == "residual")

        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            initial_pos = env.sensor_reader.read_sensor("body_pos")[:2]

            while not done:
                if mode == "residual":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = np.zeros(12)  # No residuals

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Calculate distance traveled
            final_pos = env.sensor_reader.read_sensor("body_pos")[:2]
            distance = np.linalg.norm(final_pos - initial_pos)

            results[mode]["distances"].append(distance)
            if terminated:  # Fall
                results[mode]["falls"] += 1

            print(f"  Ep {ep+1}: {distance:.2f}m")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Distance comparison
    ax1.bar(
        ["Baseline", "Residual"],
        [
            np.mean(results["baseline"]["distances"]),
            np.mean(results["residual"]["distances"]),
        ],
        yerr=[
            np.std(results["baseline"]["distances"]),
            np.std(results["residual"]["distances"]),
        ],
    )
    ax1.set_ylabel("Distance traveled (m)")
    ax1.set_title("Average Distance Traveled")

    # Fall rate comparison
    ax2.bar(
        ["Baseline", "Residual"],
        [
            results["baseline"]["falls"] / num_episodes,
            results["residual"]["falls"] / num_episodes,
        ],
    )
    ax2.set_ylabel("Fall rate")
    ax2.set_title("Fall Rate")
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig("residual_vs_baseline.png", dpi=150)
    print("\nPlot saved to residual_vs_baseline.png")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python compare_residual_vs_baseline.py <model_path>")
        sys.exit(1)

    run_comparison(sys.argv[1])
```

---

## Phase 5: Hyperparameter Tuning & Optimization

### 5.1 Key Hyperparameters to Tune

| Parameter | Default | Tuning Range | Impact |
|-----------|---------|--------------|--------|
| `learning_rate` | 3e-4 | [1e-5, 1e-3] | Too high: unstable; too low: slow |
| `residual_scale` | 0.02 | [0.01, 0.05] | Larger = more correction authority |
| `n_steps` | 2048 | [1024, 4096] | Buffer size per update |
| `batch_size` | 64 | [32, 256] | Larger = more stable but slower |
| `ent_coef` | 0.01 | [0.0, 0.1] | Higher = more exploration |
| `gamma` | 0.99 | [0.95, 0.995] | Discount factor for long-term rewards |
| `clip_range` | 0.2 | [0.1, 0.3] | PPO clipping threshold |

### 5.2 Optuna Integration (Optional)

**File:** `tune_hyperparameters.py`

```python
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def objective(trial):
    """Optuna objective function."""

    # Sample hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-3, 0.1)
    gamma = trial.suggest_uniform("gamma", 0.95, 0.995)

    # Create environment
    env = DummyVecEnv([lambda: ResidualWalkEnv()])

    # Train model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        ent_coef=ent_coef,
        gamma=gamma,
        verbose=0,
    )
    model.learn(total_timesteps=100_000)

    # Evaluate
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
```

---

## Phase 6: Testing & Validation

### 6.1 Unit Tests

**File:** `tests/test_residual_env.py`

```python
import pytest
import numpy as np
from envs.residual_walk_env import ResidualWalkEnv

def test_env_creation():
    """Test environment can be created."""
    env = ResidualWalkEnv()
    assert env.observation_space is not None
    assert env.action_space is not None

def test_reset():
    """Test reset returns valid observation."""
    env = ResidualWalkEnv()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)

def test_step():
    """Test step with random action."""
    env = ResidualWalkEnv()
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_zero_residuals():
    """Test that zero residuals match baseline controller."""
    env = ResidualWalkEnv()
    env.reset()

    # Run with zero residuals (should be stable on flat ground)
    for _ in range(100):
        action = np.zeros(12)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            pytest.fail("Robot fell with zero residuals on flat ground")

def test_residual_clipping():
    """Test that large residuals are properly clipped."""
    env = ResidualWalkEnv()
    env.reset()

    # Try extreme actions
    action = np.ones(12) * 100  # Very large
    obs, reward, terminated, truncated, info = env.step(action)
    # Should not crash or produce NaN
    assert np.all(np.isfinite(obs))
```

Run tests:
```bash
pytest tests/test_residual_env.py -v
```

### 6.2 Integration Tests

**File:** `tests/test_training_smoke.py`

```python
def test_training_runs():
    """Smoke test: verify training runs without errors."""
    env = DummyVecEnv([lambda: ResidualWalkEnv()])
    model = PPO("MlpPolicy", env, verbose=0)

    # Train for small number of steps
    model.learn(total_timesteps=1000)

    # Should complete without errors
    assert True

def test_model_save_load():
    """Test model can be saved and loaded."""
    import tempfile

    env = DummyVecEnv([lambda: ResidualWalkEnv()])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(f"{tmpdir}/test_model")

        # Load
        loaded_model = PPO.load(f"{tmpdir}/test_model")

        # Test prediction
        obs = env.reset()
        action1, _ = model.predict(obs)
        action2, _ = loaded_model.predict(obs)

        np.testing.assert_array_almost_equal(action1, action2)
```

---

## Phase 7: Deployment Checklist

### 7.1 Pre-Deployment Validation

- [ ] Model achieves target success rate (>80%) on rough terrain
- [ ] No joint limit violations during 100 test episodes
- [ ] Average episode length > 500 steps (5+ seconds)
- [ ] Body height maintained within ±2cm of target
- [ ] Forward velocity within ±0.1 m/s of target
- [ ] No NaN or Inf values in observations/actions
- [ ] Unit tests pass
- [ ] Integration tests pass

### 7.2 Monitoring Dashboard

Create a simple monitoring script to track key metrics:

**File:** `utils/monitor_training.py`

```python
#!/usr/bin/env python3
"""Monitor training progress from TensorBoard logs."""

import argparse
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def plot_training_curves(log_dir: str):
    """Plot reward curves from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Get available tags
    scalar_tags = ea.Tags()["scalars"]

    # Plot key metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("rollout/ep_rew_mean", "Episode Reward"),
        ("rollout/ep_len_mean", "Episode Length"),
        ("train/learning_rate", "Learning Rate"),
        ("train/entropy_loss", "Entropy Loss"),
    ]

    for ax, (tag, title) in zip(axes.flat, metrics):
        if tag in scalar_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values)
            ax.set_title(title)
            ax.set_xlabel("Steps")
            ax.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Path to TensorBoard log directory")
    args = parser.parse_args()
    plot_training_curves(args.log_dir)
```

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Phase 1 | 1-2 days | Refactoring and modularization |
| Phase 2 | 3-4 days | Gymnasium environment implementation |
| Phase 3 | 1 day | Training pipeline setup |
| Phase 4 | 2-3 days | Initial training runs (5M steps) |
| Phase 5 | 2-3 days | Hyperparameter tuning |
| Phase 6 | 1-2 days | Evaluation and testing |
| Phase 7 | 1 day | Documentation and deployment prep |
| **Total** | **11-16 days** | **Full implementation** |

---

## Troubleshooting Guide

### Common Issues

#### 1. Training is unstable / Robot falls immediately
**Symptoms:** Reward stays negative, episode length < 50 steps

**Solutions:**
- Reduce `residual_scale` (try 0.01m instead of 0.02m)
- Increase `ent_coef` for more exploration
- Check reward function weights (body orientation should be high)
- Verify IK is not failing (add logging)
- Start with flat ground (`world.xml`) to verify environment

#### 2. Reward plateaus early
**Symptoms:** Reward stops improving after 500k steps

**Solutions:**
- Implement curriculum learning (gradually increase terrain difficulty)
- Add domain randomization (vary robot mass, friction)
- Increase network capacity (try [512, 512, 256])
- Adjust reward shaping (may be overfitting to local optimum)

#### 3. Policy is jittery / high-frequency oscillations
**Symptoms:** Robot vibrates, actuators oscillate rapidly

**Solutions:**
- Add action smoothness penalty (higher weight on `smoothness` reward)
- Reduce learning rate
- Add temporal smoothing to actions (low-pass filter)
- Increase `clip_range` to limit policy changes

#### 4. IK failures during training
**Symptoms:** Warnings about unreachable targets

**Solutions:**
- Reduce `residual_scale`
- Add workspace clamping before IK call
- Verify foot workspace with `foot_range_calculator.py`
- Check for numerical issues (NaN/Inf in actions)

#### 5. Training is too slow
**Symptoms:** < 500 steps/second

**Solutions:**
- Increase number of parallel environments (up to 16)
- Use faster integration (`implicitfast` in MuJoCo)
- Disable rendering during training
- Profile code to find bottlenecks (likely IK or sensor reading)

---

## Advanced Extensions

### 1. Terrain Randomization
- Modify `world_train.xml` heightfield procedurally
- Add different terrain types (stairs, slopes, gaps)
- Randomize friction coefficients

### 2. Command Conditioning
- Add commanded velocity to observation
- Train policy to follow velocity commands
- Enable user control via joystick/keyboard

### 3. Multi-Objective Optimization
- Add secondary objectives (minimize energy, maximize stability margin)
- Use Pareto-optimal policies
- Trade-off speed vs. robustness

### 4. Sim-to-Real Transfer
- Domain randomization (mass, inertia, actuator delays)
- System identification on real robot
- Safety constraints for deployment

### 5. Hierarchical RL
- High-level policy selects gait parameters
- Low-level policy generates residuals
- Better for diverse terrain types

---

## References & Resources

### Papers
1. **"Learning Quadrupedal Locomotion over Challenging Terrain"** (Hwangbo et al., 2019)
2. **"Deep RL for Legged Robots"** (Lee et al., 2020)
3. **"Residual RL for Manipulation"** (Silver et al., 2018)

### Repositories
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
- Gymnasium docs: https://gymnasium.farama.org/
- MuJoCo documentation: https://mujoco.readthedocs.io/

### Tutorials
- PPO explained: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Gym custom environments: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

---

## Conclusion

This plan provides a comprehensive roadmap for implementing PPO-based residual learning for your quadruped robot. The key advantages of this approach:

1. **Modularity:** Preserves your working Bézier controller as a fallback
2. **Sample efficiency:** RL only learns terrain-specific corrections
3. **Robustness:** Graceful degradation if policy fails
4. **Extensibility:** Easy to add new features (velocity commands, terrain types)

**Next steps:**
1. Start with Phase 1 (refactoring)
2. Implement core environment (Phase 2)
3. Run initial training on flat ground to validate setup
4. Progressively add terrain difficulty

**Success criteria:**
- Robot walks 10+ meters on rough terrain without falling
- Maintains upright posture (roll/pitch < 15°)
- Follows commanded velocities (0.2-0.5 m/s)

Good luck with the implementation! Feel free to iterate on the reward function and network architecture based on your specific terrain and robot dynamics.
