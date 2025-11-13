# MuJoCo Playground: Custom Quadruped Environment Creation Guide

## Repository Overview

**Project**: google-deepmind/mujoco_playground
**URL**: https://github.com/google-deepmind/mujoco_playground
**Purpose**: GPU-accelerated environment suite for robot learning research and sim-to-real transfer
**License**: Apache 2.0
**Python**: 3.10+

### Key Features
- JAX-based physics simulation with MuJoCo MJX
- Pre-built quadruped environments (Go1, Spot, Apollo, H1, etc.)
- Modular environment framework for custom robots
- Training scripts with PPO and RSL-RL support
- Domain randomization and vision-based capabilities

---

## Part 1: Directory Structure for Environments

### Main Directory Tree
```
mujoco_playground/
├── _src/                           # Core implementation
│   ├── mjx_env.py                  # Base environment class
│   ├── registry.py                 # Environment registry
│   ├── wrapper.py                  # Wrapper classes
│   ├── reward.py                   # Reward utility functions
│   ├── gait.py                     # Gait control utilities
│   ├── dm_control_suite/           # Classic control environments
│   ├── locomotion/                 # Quadruped/biped environments
│   │   ├── go1/                    # Unitree Go1 robot
│   │   │   ├── __init__.py         # Module exports
│   │   │   ├── base.py             # Base Go1 environment
│   │   │   ├── joystick.py         # Velocity tracking task
│   │   │   ├── getup.py            # Recovery from falls task
│   │   │   ├── handstand.py        # Dynamic skill task
│   │   │   ├── go1_constants.py    # Robot constants/paths
│   │   │   ├── randomize.py        # Domain randomization
│   │   │   └── xmls/               # MuJoCo model files
│   │   │       ├── go1_mjx.xml
│   │   │       ├── go1_mjx_feetonly.xml
│   │   │       ├── scene_mjx_flat_terrain.xml
│   │   │       ├── scene_mjx_rough_terrain.xml
│   │   │       ├── sensor_feet.xml
│   │   │       └── assets/         # Mesh files
│   │   ├── spot/                   # Boston Dynamics Spot
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── joystick.py
│   │   │   ├── getup.py
│   │   │   ├── spot_constants.py
│   │   │   └── xmls/
│   │   ├── apollo/                 # Sanctuary AI Apollo
│   │   ├── h1/                     # Unitree H1 (bipedal)
│   │   └── [other robots...]
│   ├── manipulation/               # Manipulation environments
│   └── locomotion_test.py
├── config/                         # Training configurations
│   ├── __init__.py
│   ├── locomotion_params.py        # Locomotion hyperparameters
│   ├── dm_control_suite_params.py
│   └── manipulation_params.py
├── learning/                       # Training scripts
│   ├── train_jax_ppo.py            # Main training script
│   └── train_notebooks/            # Jupyter tutorials
├── pyproject.toml                  # Dependencies
└── README.md                       # Documentation
```

### Where Quadruped Environments Are Defined
**Primary Location**: `/mujoco_playground/_src/locomotion/[robot_name]/`

Each robot has:
1. **base.py** - Extends `MjxEnv`, provides base sensors and properties
2. **Task files** (joystick.py, getup.py, etc.) - Extend base with specific objectives
3. **constants.py** - Robot-specific paths, joint names, foot indices
4. **xmls/** - MuJoCo model files (scenes, models, sensors)
5. **__init__.py** - Exports tasks and registers them

---

## Part 2: Base Class/API for Creating Custom Environments

### Base Class: `MjxEnv`
**Location**: `mujoco_playground/_src/mjx_env.py`

```python
from mujoco_playground._src import mjx_env

class MjxEnv(abc.ABC):
    """Abstract base class for MuJoCo JAX environments."""
    
    def __init__(self, config: ml_collections.ConfigDict):
        """Initialize with configuration dictionary."""
        self._cfg = config
```

### Required Abstract Methods to Implement

#### 1. `reset(rng: jax.Array) -> State`
```python
def reset(self, rng: jax.Array) -> 'State':
    """Reset environment to initial state.
    
    Args:
        rng: JAX random number generator state
        
    Returns:
        State: Initial state containing obs, reward, done, metrics, info
    """
    # Your reset logic here
    pass
```

#### 2. `step(state: State, action: jax.Array) -> State`
```python
def step(self, state: 'State', action: jax.Array) -> 'State':
    """Execute one environment step.
    
    Args:
        state: Current state
        action: Control action (size = action_size)
        
    Returns:
        State: Next state after physics simulation
    """
    # Your step logic here
    pass
```

### Required Properties

#### 1. `xml_path` Property
```python
@property
def xml_path(self) -> str:
    """Path to main MuJoCo XML model file."""
    return "/path/to/model.xml"
```

#### 2. `action_size` Property
```python
@property
def action_size(self) -> int:
    """Number of action dimensions (actuators)."""
    return 12  # For 4-legged robot with 3 DOF each
```

#### 3. `mj_model` Property
```python
@property
def mj_model(self) -> mujoco.MjModel:
    """Return compiled MuJoCo model."""
    return self._mj_model
```

#### 4. `mjx_model` Property
```python
@property
def mjx_model(self) -> mujoco.mjx.Model:
    """Return JAX-compiled MuJoCo model."""
    return self._mjx_model
```

### Utility Properties/Methods Available

```python
# From MjxEnv base:
self.observation_size  # Computed from observation shape
self.dt                # Control timestep (config.control_dt)
self.sim_dt            # Simulation timestep (config.sim_dt)
self.n_substeps        # Derived: ceil(dt / sim_dt)
self.model_assets      # Asset registry for rendering
self.render()          # Render observation
self.tree_replace()    # Utility for state manipulation
```

### State Dataclass Structure

**Location**: `mujoco_playground/_src/mjx_env.py` (line ~175)

```python
@struct.dataclass
class State:
    """Environment state container."""
    data: mjx.Data                      # Physics simulation state
    obs: Union[jax.Array, Mapping]      # Observation (array or dict)
    reward: jax.Array                   # Scalar reward
    done: jax.Array                     # Episode termination flag
    metrics: Dict[str, jax.Array]       # Tracked metrics (e.g., distance)
    info: Dict[str, Any]                # Additional metadata
    
    def tree_replace(self, **kwargs):
        """Update nested fields using dot notation (e.g., 'data.qpos')."""
        pass
```

---

## Part 3: Observations, Actions, and Rewards

### Example: Spot Joystick Environment (Velocity Tracking Task)

#### **Observations** (`_get_obs()`)

The joystick environments create observations in two variants:

**State Observation** (policy input):
```python
obs = {
    'linear_vel': global_linvel + noise,      # 3D global velocity
    'gyro': gyroscope + noise,                 # 3D angular velocity
    'gravity': gravity + noise,                # 3D gravity vector
    'command': (vx_cmd, vy_cmd, wz_cmd),      # Velocity command (3,)
    'joint_angles': qpos - qpos_default,      # Joint angles relative to home (12,)
    'joint_vels': qvel,                       # Joint velocities (12,)
    'last_action': prev_action,               # Previous action (12,)
}
# obs shape: typically 50-60 elements total
```

**Privileged State** (optional for value function training):
```python
privileged_obs = {
    'gyro': gyroscope_clean,
    'accelerometer': accelerometer,
    'gravity': gravity_clean,
    'linvel_local': local_linvel,
    'angvel': angvel,
    'actuator_forces': torques,
    'contact': contact_states,
    'feet_vel': foot_velocities,
}
```

#### **Actions**

Motor control via PD tracking:
```
motor_targets = default_pose + action * action_scale

Where:
  - action: jax.Array of shape (12,) from [-1, 1]
  - action_scale: scalar (typically 0.5 rad)
  - default_pose: home configuration (12,)
  - Motor command sent to PD controller with Kp, Kd gains
```

#### **Rewards**

Multi-component reward function with 14+ terms, weighted and clipped:

```python
reward = clip(sum([
    # Velocity tracking rewards
    w_vx * exp(-|vx - vx_cmd| / 0.1),
    w_vy * exp(-|vy - vy_cmd| / 0.1),
    w_wz * exp(-|wz - wz_cmd| / 0.1),
    
    # Base stability (negative rewards for unwanted motion)
    -w_z_vel * |z_vel|,                    # Penalize vertical drift
    -w_z_ang * |roll, pitch|,              # Penalize tilt
    -w_ang_xy * |yaw_rate_unwanted|,
    
    # Energy efficiency
    -w_torque * sum(actuator_forces^2),
    -w_power * sum(torque * joint_vel),
    -w_action_smooth * sum((action[t] - action[t-1])^2),
    
    # Feet dynamics
    -w_slip * foot_slip_magnitude,
    w_foot_clear * foot_clearance,
    w_air_time * swing_phase_duration,
    
    # Safety
    -w_joint_limits * violation_penalty,
    -w_termination * is_done,
    
]), min=0, max=10000) * dt
```

Key configuration parameters:
- `obs_noise`: [gyro, gravity, joint_angle] noise scales
- `reward_scales`: Dictionary of component weights
- `action_scale`: Range of motor commands
- `implicit_bias`: Normalize reward by timestep

---

## Part 4: Example Quadruped Environment Files

### Example 1: Go1 Environment Structure

**File**: `mujoco_playground/_src/locomotion/go1/base.py`

```python
import mujoco
import mujoco.mjx as mjx
from mujoco_playground._src import mjx_env

class Go1Env(mjx_env.MjxEnv):
    """Base environment for Unitree Go1 robot."""
    
    def __init__(self, xml_path: str, config: ml_collections.ConfigDict):
        self._xml_path = xml_path
        self._config = config
        
        # Load assets
        assets = self.get_assets()
        
        # Compile models
        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_str, assets=assets
        )
        self._mjx_model = mjx.put_model(self._mj_model)
        
        # Configure physics
        self._mjx_model.opt.timestep = config.sim_dt
        # ... PD gain setup, rendering config, etc.
    
    @staticmethod
    def get_assets():
        """Load XML assets from local and menagerie paths."""
        # Returns dict of asset_name -> asset_path
        pass
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Initialize robot at origin with random perturbations."""
        # Sample initial state
        # Return State with obs, reward=0, done=False
        pass
    
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Execute one control timestep."""
        # Apply PD control
        # Simulate physics
        # Compute observations, rewards
        # Return new State
        pass
    
    @property
    def xml_path(self) -> str:
        return self._xml_path
    
    @property
    def action_size(self) -> int:
        return self._mjx_model.nu  # Number of actuators
    
    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model
    
    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
    
    # Sensor accessor methods
    def get_feet_pos(self, state: mjx_env.State) -> jax.Array:
        """Return positions of all 4 feet."""
        return mjx_env.get_sensor_data(
            state.data, 'FR_foot_pos', 'FL_foot_pos', ...
        )
```

**File**: `mujoco_playground/_src/locomotion/go1/joystick.py`

```python
class Joystick(go1_base.Go1Env):
    """Go1 velocity tracking task."""
    
    def __init__(self, xml_path: str, config: ml_collections.ConfigDict):
        super().__init__(xml_path, config)
        # Task-specific setup
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset with random position and sample velocity command."""
        state = super().reset(rng)
        # Sample cmd: (vx, vy, wz) from configured ranges
        # Embed cmd in obs
        return state
    
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Execute one step: simulate -> compute obs/reward."""
        # Call parent step (handles physics)
        state = super().step(state, action)
        
        # Compute task-specific reward
        reward = self._compute_reward(state)
        
        # Update metrics
        metrics = {...}
        
        return state.tree_replace(reward=reward, metrics=metrics)
    
    def _compute_reward(self, state: mjx_env.State) -> jax.Array:
        """Weighted sum of reward components."""
        pass
    
    def _get_obs(self, state: mjx_env.State) -> jax.Array:
        """Extract observation from state."""
        pass
    
    @classmethod
    def create(cls, *args, **kwargs):
        """Factory method for environment creation."""
        return cls(*args, **kwargs)
```

**File**: `mujoco_playground/_src/locomotion/go1/go1_constants.py`

```python
import pathlib

# XML file paths
_ROOT = pathlib.Path(__file__).parent
FEET_ONLY_FLAT_TERRAIN_XML = str(_ROOT / 'xmls/scene_mjx_feetonly_flat_terrain.xml')
FULL_FLAT_TERRAIN_XML = str(_ROOT / 'xmls/scene_mjx_flat_terrain.xml')
FULL_COLLISIONS_FLAT_TERRAIN_XML = str(_ROOT / 'xmls/scene_mjx_fullcollisions_flat_terrain.xml')
FEET_ONLY_ROUGH_TERRAIN_XML = str(_ROOT / 'xmls/scene_mjx_feetonly_rough_terrain.xml')

# Robot structure
FEET = ['FL', 'FR', 'RL', 'RR']  # Leg indices
BODY = 'trunk'

# Sensor names
SENSORS = {
    'upvector': ('upvector',),
    'global_linvel': ('global_linvel',),
    'global_angvel': ('global_angvel',),
    'local_linvel': ('local_linvel',),
    'gyro': ('gyro',),
    'accelerometer': ('accelerometer',),
    'feet_pos': tuple(f'{foot}_foot_pos' for foot in FEET),
    'feet_contact': tuple(f'{foot}_contact' for foot in FEET),
}

def task_to_xml(task: str) -> str:
    """Map task name to XML path."""
    mapping = {
        'flat_terrain': FULL_FLAT_TERRAIN_XML,
        'rough_terrain': FEET_ONLY_ROUGH_TERRAIN_XML,
    }
    return mapping[task]
```

**File**: `mujoco_playground/_src/locomotion/go1/__init__.py`

```python
from mujoco_playground._src.locomotion.go1 import base, joystick, getup

def get_config() -> ml_collections.ConfigDict:
    """Default config for Go1 environments."""
    return ml_collections.ConfigDict({
        'control_dt': 0.02,           # 50Hz control
        'sim_dt': 0.001,              # 1000Hz simulation
        'action_scale': 0.5,
        'obs_noise': [0.01, 0.01, 0.01],  # [gyro, gravity, jpos]
        'reward_scales': {...},
    })

# Register environments
ALL_ENVS = (
    'Go1JoystickFlatTerrain',
    'Go1JoystickRoughTerrain',
    'Go1GetupFlatTerrain',
    'Go1HandstandFlatTerrain',
)

_ENVS = {
    'Go1JoystickFlatTerrain': joystick.Joystick,
    'Go1GetupFlatTerrain': getup.Getup,
    'Go1HandstandFlatTerrain': handstand.Handstand,
}

_CFGS = {
    env_name: get_config for env_name in ALL_ENVS
}

def load(env_name: str, **kwargs) -> base.Go1Env:
    """Create environment instance."""
    cls = _ENVS[env_name]
    config = _CFGS[env_name](**kwargs)
    xml_path = go1_constants.task_to_xml('flat_terrain')
    return cls(xml_path=xml_path, config=config)
```

### Example 2: Spot Getup (Recovery) Environment

**File**: `mujoco_playground/_src/locomotion/spot/getup.py`

```python
class Getup(spot_base.SpotEnv):
    """Recover from fall task."""
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Random initialization: drop from height or start at home."""
        rng, subrng = jax.random.split(rng)
        
        # 60% probability: drop from height
        # 40% probability: start at home
        dropped = jax.random.uniform(subrng) < 0.6
        
        if dropped:
            qpos = self._get_random_qpos(subrng)
        else:
            qpos = self._home_qpos
        
        # Initialize data
        state = mjx_env.State(...)
        return state
    
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Simulate and compute recovery-specific reward."""
        state = super().step(state, action)
        
        # Recovery rewards
        upright_reward = self._upright_reward(state)
        height_reward = self._height_reward(state)
        posture_reward = self._posture_regularization(state)
        
        # Gate rewards: only count when upright AND at right height
        gate = (upright_error < 0.01) & (height_error < 0.005)
        reward = jnp.where(
            gate,
            upright_reward + height_reward + posture_reward,
            0.0
        )
        
        return state.tree_replace(reward=reward)
```

---

## Part 5: Key Imports and Dependencies

### Essential Imports

```python
# JAX/Flax ecosystem
import jax
import jax.numpy as jnp
from flax import struct
import ml_collections

# MuJoCo
import mujoco
import mujoco.mjx as mjx

# Playground library
from mujoco_playground._src import mjx_env
from mujoco_playground._src import registry
from mujoco_playground._src import wrapper
from mujoco_playground._src import reward

# Standard library
import abc
import pathlib
from typing import Dict, Any, Optional, Tuple, Union, Callable
```

### Required Dependencies (from pyproject.toml)

**Core**:
- mujoco >= 3.3.6.dev
- mujoco-mjx >= 3.3.6.dev
- jax
- flax
- brax >= 0.12.5

**Optional for training**:
- rsl-rl-lib >= 3.0.0
- wandb

**Optional for vision**:
- madrona-mjx (see Madrona-MJX repo)

---

## Part 6: How Environments Load MuJoCo XML Models

### Pattern 1: Direct Path

```python
class MyEnv(mjx_env.MjxEnv):
    def __init__(self, xml_path: str, config: ml_collections.ConfigDict):
        with open(xml_path) as f:
            xml_string = f.read()
        
        assets = self.get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_string,
            assets=assets
        )
        self._mjx_model = mjx.put_model(self._mj_model)
```

### Pattern 2: Asset Loading with Menagerie

```python
@staticmethod
def get_assets() -> Dict[str, str]:
    """Load assets from local and menagerie repos."""
    local_assets = {}
    
    # Load local assets
    local_dir = pathlib.Path(__file__).parent / 'xmls/assets'
    for file in local_dir.glob('**/*.stl'):
        local_assets[file.name] = str(file)
    
    # Load from menagerie (if available)
    menagerie_assets = {}
    # ... fetch from google-deepmind/mujoco_menagerie
    
    return {**local_assets, **menagerie_assets}
```

### Pattern 3: XML with Includes

**Main scene file** (scene_mjx_flat_terrain.xml):
```xml
<?xml version="1.0"?>
<mujoco model="scene">
  <include file="go1_mjx.xml"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1" material="ground"/>
  </worldbody>
  <include file="sensor_feet.xml"/>
</mujoco>
```

**Robot model** (go1_mjx.xml):
```xml
<?xml version="1.0"?>
<mujoco model="go1">
  <worldbody>
    <body name="trunk" pos="0 0 0.3">
      <inertial mass="5.5" diaginv="..."/>
      <geom type="box" size="0.1 0.05 0.05"/>
      <!-- 4 legs with joints -->
      <body name="FL_hip" pos="0.1 0.1 0">
        <joint name="FL_hip" type="hinge" axis="0 0 1"/>
        <!-- ... more joints and bodies -->
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- PD actuators for each joint -->
    <motor name="FL_hip" joint="FL_hip"/>
  </actuator>
</mujoco>
```

### Key Compilation Steps

```python
# 1. Load XML string (with includes resolved)
xml_string = open(xml_path).read()

# 2. Compile to MuJoCo model
mj_model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)

# 3. Compile to JAX-compatible model
mjx_model = mjx.put_model(mj_model)

# 4. Configure physics parameters
mjx_model.opt.timestep = config.sim_dt

# 5. Create initial physics state
mjx_data = mjx.make_data(mjx_model)
```

---

## Part 7: Complete Minimal Environment Template

Here's a minimal working template for a custom quadruped:

```python
# my_quadruped_env.py
import jax
import jax.numpy as jnp
import ml_collections
import mujoco
import mujoco.mjx as mjx
from mujoco_playground._src import mjx_env

class MyQuadrupedEnv(mjx_env.MjxEnv):
    """Custom 3DOF parallel SCARA quadruped environment."""
    
    def __init__(
        self,
        xml_path: str,
        config: ml_collections.ConfigDict,
        **kwargs
    ):
        """Initialize environment.
        
        Args:
            xml_path: Path to MuJoCo XML model
            config: ConfigDict with dt, sim_dt, action_scale, etc.
        """
        self._xml_path = xml_path
        self._config = config
        
        # Load and compile model
        with open(xml_path) as f:
            xml_string = f.read()
        
        self._mj_model = mujoco.MjModel.from_xml_string(xml_string)
        self._mjx_model = mjx.put_model(self._mj_model)
        
        # Physics config
        self._mjx_model.opt.timestep = config.sim_dt
        
        # Home position (default pose)
        self._home_qpos = jnp.zeros(self._mjx_model.nq)
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset to initial state."""
        rng, subrng = jax.random.split(rng)
        
        # Create physics state
        data = mjx.make_data(self._mjx_model)
        
        # Set initial configuration
        data = data.replace(qpos=self._home_qpos)
        
        # Step to settle
        data = mjx.step(self._mjx_model, data)
        
        # Get initial observation
        obs = self._get_obs(data)
        
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(False),
            metrics={},
            info={}
        )
    
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Execute one control timestep."""
        # PD control: target = home + action * scale
        target = self._home_qpos + action * self._config.action_scale
        
        # Simulate for n_substeps
        data = state.data
        for _ in range(self.n_substeps):
            # Simple PD: tau = Kp * (target - qpos) - Kd * qvel
            error = target - data.qpos
            tau = (
                self._config.kp * error - 
                self._config.kd * data.qvel
            )
            data = data.replace(ctrl=tau)
            data = mjx.step(self._mjx_model, data)
        
        # Compute observation and reward
        obs = self._get_obs(data)
        reward = self._compute_reward(data, action)
        done = self._is_done(data)
        
        metrics = {'reward': reward}
        
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info={}
        )
    
    def _get_obs(self, data: mjx.Data) -> jax.Array:
        """Extract observation from state."""
        return jnp.concatenate([
            data.qpos,      # Joint angles (12,)
            data.qvel,      # Joint velocities (12,)
        ])  # Shape: (24,)
    
    def _compute_reward(self, data: mjx.Data, action: jax.Array) -> jax.Array:
        """Compute reward (simple example: penalize action magnitude)."""
        action_cost = jnp.sum(action ** 2)
        return -0.1 * action_cost
    
    def _is_done(self, data: mjx.Data) -> jax.Array:
        """Check episode termination (e.g., robot fell)."""
        torso_z = data.xpos[self._mj_model.body('trunk').id, 2]
        return torso_z < 0.1  # Fell below 10cm
    
    # Required properties
    @property
    def xml_path(self) -> str:
        return self._xml_path
    
    @property
    def action_size(self) -> int:
        return self._mjx_model.nu
    
    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model
    
    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

# Default config
def get_config() -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'control_dt': 0.02,
        'sim_dt': 0.001,
        'action_scale': 0.5,
        'kp': 100.0,
        'kd': 2.0,
    })
```

### Registration in Locomotion Module

Add to `mujoco_playground/_src/locomotion/my_quadruped/__init__.py`:

```python
from mujoco_playground._src.locomotion.my_quadruped import base

ALL_ENVS = ('MyQuadrupedEnv',)

_ENVS = {
    'MyQuadrupedEnv': base.MyQuadrupedEnv,
}

_CFGS = {
    'MyQuadrupedEnv': base.get_config,
}

def load(env_name: str, config=None, **kwargs):
    cls = _ENVS[env_name]
    cfg = _CFGS[env_name]()
    if config:
        cfg.update(config)
    return cls(xml_path='path/to/model.xml', config=cfg, **kwargs)
```

---

## Part 8: Training and Using Custom Environments

### Loading from Registry

```python
from mujoco_playground._src import registry

# Load environment
env = registry.load(
    'MyQuadrupedEnv',
    config=ml_collections.ConfigDict({
        'action_scale': 0.5,
        'control_dt': 0.02,
    })
)

# Reset and step
rng = jax.random.PRNGKey(0)
state = env.reset(rng)

action = jnp.zeros(env.action_size)
state = env.step(state, action)

print(f"Obs shape: {state.obs.shape}")
print(f"Reward: {state.reward}")
```

### Training with Brax PPO

```python
from mujoco_playground._src import registry, wrapper
import brax

# Create environment
env = registry.load('MyQuadrupedEnv')

# Wrap for training
train_env = wrapper.wrap_for_brax_training(
    env,
    episode_length=1000,
    action_repeat=1,
)

# Get training function
from brax.training import ppo

make_inference_fn, params, _ = ppo.train(
    environment=train_env,
    num_timesteps=10_000_000,
    episode_length=1000,
    batch_size=256,
    num_envs=4096,
)
```

---

## Summary: Key Files to Create for Custom Environment

```
mujoco_playground/_src/locomotion/my_scara_quadruped/
├── __init__.py                    # Module exports, registration
├── base.py                        # Base class (extends MjxEnv)
├── [task_name].py                 # Task-specific (extends base)
├── constants.py                   # Robot-specific constants
├── xmls/                          # MuJoCo model files
│   ├── my_robot.xml              # Main model
│   ├── scene_flat.xml            # Scene with includes
│   ├── sensors.xml               # Sensor definitions
│   └── assets/                   # Mesh files
└── README.md                      # Documentation
```

---

## Quick Reference: Critical Class Names and Patterns

| Component | Location | Class/Function |
|-----------|----------|-----------------|
| Base Environment | `mjx_env.py` | `MjxEnv` (ABC) |
| State Container | `mjx_env.py` | `State` (dataclass) |
| Environment Loading | `registry.py` | `load()`, `get_default_config()` |
| Wrapper Utilities | `wrapper.py` | `wrap_for_brax_training()`, `BraxAutoResetWrapper` |
| Reward Functions | `reward.py` | `tolerance()` |
| Example: Go1 Base | `go1/base.py` | `Go1Env` |
| Example: Go1 Task | `go1/joystick.py` | `Joystick` |
| Example: Spot Base | `spot/base.py` | `SpotEnv` |
| Training Entry | `train_jax_ppo.py` | `train()` with registry env |

