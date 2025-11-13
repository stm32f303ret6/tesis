# MuJoCo Playground: Architecture Quick Reference

## Core Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Training Script                         │
│                 (learning/train_jax_ppo.py)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │ registry.load('Go1JoystickFlatTerrain')
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Environment Registry                            │
│              (_src/registry.py)                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ _envs: {env_name: EnvClass}                             │   │
│  │ _cfgs: {env_name: get_config_fn}                        │   │
│  │ _randomizer: {env_name: randomize_fn}                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │ delegates to locomotion/go1/__init__.py               │
└─────────────────────────────────────────────────────────────────┘
                         │ returns Go1Env instance
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Task Environment (Joystick)                         │
│        (_src/locomotion/go1/joystick.py)                        │
│  Extends: Go1Env                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ reset() -> State (initial obs, reward=0, done=False)    │   │
│  │ step(state, action) -> State (new obs, reward, done)    │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │ calls super().step() for physics
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                Base Robot Environment                            │
│           (_src/locomotion/go1/base.py)                         │
│  Extends: MjxEnv                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ reset(rng) -> State                                      │   │
│  │ step(state, action) -> State                             │   │
│  │ get_feet_pos(state) -> foot positions                   │   │
│  │ get_gyro(state) -> angular velocity                     │   │
│  │ ... more sensor accessors ...                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│  - Loads MuJoCo model from XML                                 │
│  - Configures PD control gains                                │
│  - Manages MJX data (physics state)                           │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               Abstract Base Class: MjxEnv                        │
│                  (_src/mjx_env.py)                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ @property xml_path -> str                               │   │
│  │ @property action_size -> int                            │   │
│  │ @property mj_model -> mujoco.MjModel                   │   │
│  │ @property mjx_model -> mjx.Model                       │   │
│  │ @property observation_size -> int                       │   │
│  │ @property dt, sim_dt, n_substeps                        │   │
│  │                                                         │   │
│  │ @abstractmethod reset(rng) -> State                     │   │
│  │ @abstractmethod step(state, action) -> State            │   │
│  │                                                         │   │
│  │ render() -> observation                                │   │
│  │ tree_replace(state, **kwargs) -> modified_state        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    State Dataclass                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ data: mjx.Data          # Physics simulation state       │   │
│  │ obs: jax.Array | Dict   # Observation (policy input)    │   │
│  │ reward: jax.Array       # Scalar reward (float32)       │   │
│  │ done: jax.Array         # Boolean episode flag          │   │
│  │ metrics: Dict[str, Array] # Tracked metrics             │   │
│  │ info: Dict[str, Any]    # Additional metadata           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              MuJoCo Physics (JAX Backend)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ mjx.Model: Compiled physics model (differentiable)      │   │
│  │ mjx.Data: Simulation state (qpos, qvel, xpos, ...)     │   │
│  │ mjx.step(): Single physics timestep (takes microsecs)  │   │
│  │ mjx.make_data(): Initialize physics state              │   │
│  └──────────────────────────────────────────────────────────┘   │
│  - XML -> mujoco.MjModel -> mjx.Model (compilation)             │
│  - Supports batched operations (vmap)                           │
│  - Fully differentiable physics                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Class Hierarchy for Custom Quadruped

```
MjxEnv (ABC)
  │
  ├── Go1Env (base.py)
  │     ├── Joystick (joystick.py)
  │     ├── Getup (getup.py)
  │     ├── Handstand (handstand.py)
  │     └── ...
  │
  ├── SpotEnv (base.py)
  │     ├── Joystick (joystick.py)
  │     ├── Getup (getup.py)
  │     └── ...
  │
  ├── ApolloEnv
  │     └── ...
  │
  └── CustomQuadrupedEnv  <-- YOUR IMPLEMENTATION
        ├── CustomVelocityTracking
        ├── CustomGetup
        └── ...
```

## Configuration Flow

```
ml_collections.ConfigDict
│
├── Control Timing
│   ├── control_dt: 0.02 (50Hz)
│   ├── sim_dt: 0.001 (1000Hz)
│   └── n_substeps = ceil(control_dt / sim_dt)
│
├── Motor Control
│   ├── action_scale: 0.5 (radians)
│   ├── kp: 100.0 (proportional gain)
│   └── kd: 2.0 (derivative gain)
│
├── Observation
│   ├── obs_noise: [0.01, 0.01, 0.01]  # [gyro, gravity, jpos]
│   └── obs_shape: (50-60,) or {key: shape}
│
└── Rewards
    ├── reward_scales: {...}  # 14+ components
    ├── implicit_bias: dt     # Scale by timestep
    └── clip: [0, 10000]      # Value range
```

## Key File Organization Pattern

```
mujoco_playground/_src/locomotion/[robot_name]/

1. __init__.py
   ├── ALL_ENVS = ('TaskA', 'TaskB', ...)
   ├── _ENVS = {task_name: EnvClass}
   ├── _CFGS = {task_name: get_config_fn}
   ├── load(env_name, config, **kwargs)
   └── get_default_config()

2. constants.py
   ├── XML_PATH = '/path/to/xmls'
   ├── FEET = ['FL', 'FR', 'RL', 'RR']
   ├── BODY = 'trunk'
   ├── SENSORS = {name: (sensor_names,)}
   └── task_to_xml(task: str) -> str

3. base.py
   ├── class [Robot]Env(MjxEnv)
   │   ├── __init__(xml_path, config)
   │   ├── get_assets() -> Dict
   │   ├── reset(rng) -> State
   │   ├── step(state, action) -> State
   │   ├── get_feet_pos(state)
   │   ├── get_gyro(state)
   │   └── ... sensor accessors ...
   │
   └── get_config() -> ConfigDict

4. [task].py (e.g., joystick.py)
   ├── class [Task]([Robot]Env)
   │   ├── __init__(xml_path, config)
   │   ├── reset(rng) -> State
   │   ├── step(state, action) -> State
   │   ├── _get_obs(state) -> obs
   │   ├── _compute_reward(state, action)
   │   └── _is_done(state) -> bool
   │
   └── @classmethod create(...)

5. randomize.py (optional)
   ├── def randomize_config(rng, config)
   └── Domain randomization functions

6. xmls/
   ├── [robot].xml           # Model definition
   ├── scene_[terrain].xml   # Scene with includes
   ├── sensors_[type].xml    # Sensor definitions
   └── assets/               # Mesh files (.stl)
```

## Data Flow: Single Step

```
┌─────────────────────┐
│  User Action (12,)  │  agent_action from policy
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  step(state, action) -> State               │
│                                             │
│  1. Compute motor targets:                  │
│     target = home_qpos + action * scale     │
│                                             │
│  2. Loop n_substeps times:                  │
│     ├── Compute PD error                    │
│     ├── tau = Kp * error - Kd * qvel        │
│     ├── state.data.ctrl = tau               │
│     └── state.data = mjx.step(state.data)   │
│                                             │
│  3. Extract observation:                    │
│     obs = [qpos, qvel, gravity, ...]        │
│                                             │
│  4. Compute reward:                         │
│     reward = w1*tracking + w2*stable - ... │
│                                             │
│  5. Check termination:                      │
│     done = torso_z < 0.1 or episode_len > 1000
│                                             │
│  6. Return State(data, obs, reward, done)   │
└──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Updated State      │  ready for next step
│  obs: (50,)        │
│  reward: 1.23      │
│  done: False       │
└─────────────────────┘
```

## Observation Structure Examples

### Joystick Task (Velocity Tracking)
```python
obs = jnp.concatenate([
    linvel (3),           # Global linear velocity [vx, vy, vz]
    gyro (3),            # Angular velocity [wx, wy, wz]
    gravity (3),         # Gravity vector [gx, gy, gz]
    cmd (3),             # Velocity command [vx_cmd, vy_cmd, wz_cmd]
    qpos_offset (12),    # Joint angles relative to home
    qvel (12),           # Joint velocities
    action_prev (12),    # Previous action
])
# Total: 3+3+3+3+12+12+12 = 48 elements
```

### Getup Task (Recovery)
```python
obs = jnp.concatenate([
    linvel (3),
    gyro (3),
    gravity (3),
    qpos_offset (12),
    qvel (12),
    action_prev (12),
])
# Total: 45 elements (no command)
```

## Action to Motor Command Mapping

```
Raw Policy Action (normalized [-1, 1])
       │
       ├─ Multiply by action_scale (0.5 rad)
       │
       ├─ Add to home configuration
       │     target_qpos = qpos_home + scaled_action
       │
       └─ Send to PD Controller
           tau = Kp * (target - current_qpos) - Kd * qvel
           Apply tau to motors via mjx.Data.ctrl
```

## Reward Components (Joystick Example)

```
Total Reward = sum([
    +exp(-||v_actual - v_cmd|| / scale) * w_tracking,
    -|z_vel| * w_z_stability,
    -|roll, pitch| * w_tilt,
    -sum(tau^2) * w_energy,
    -sum(action_diff^2) * w_smoothness,
    +air_time * w_swing,
    -slip * w_slip,
    ...
]) * timestep_scale

Clipped to [0, 10000] for numerical stability
```

## Critical Implementation Points

### 1. State Management
- Always use `state.tree_replace()` to modify state safely
- Never directly mutate `state` objects (JAX principle)
- JAX arrays are immutable; use `.replace()` on Data objects

### 2. Physics Loop
```python
# Template for n_substeps
data = state.data
for _ in range(self.n_substeps):
    # 1. Set control input
    data = data.replace(ctrl=tau)
    # 2. Step physics once
    data = mjx.step(self._mjx_model, data)
# Return updated state
```

### 3. Sensor Access Pattern
```python
# Use mjx_env helper to get sensor values
sensor_value = mjx_env.get_sensor_data(state.data, 'sensor_name')

# Or directly via data indexing
qpos = state.data.qpos  # All joint angles
xpos = state.data.xpos  # All body positions
xmat = state.data.xmat  # All body rotations
```

### 4. Config Dictionary Usage
```python
config = ml_collections.ConfigDict({
    'control_dt': 0.02,
    'sim_dt': 0.001,
})

# Access as attributes
self._config.control_dt      # Not dict-like
self._config.get('key')      # Also works for safety
```

### 5. Random Number Generation
```python
# Always split RNG before use
rng, subrng = jax.random.split(rng)
value = jax.random.uniform(subrng, shape=(12,))

# In reset, include in loop
rng, *subrngs = jax.random.split(rng, num=4)
qpos_init = subrngs[0]
vel_init = subrngs[1]
```

## Common Pitfalls

1. **Forgetting to call mjx.step()** - Physics won't advance
2. **Not splitting RNG** - RNG state gets reused across episodes
3. **Mixing up control vs sim timesteps** - n_substeps = ceil(dt/sim_dt)
4. **Mutating state directly** - Use `state.tree_replace()` instead
5. **Observation includes NaNs** - Check for divide-by-zero in reward computation
6. **Action not rescaled** - Remember action_scale when converting to motor commands
7. **Forgetting to implement required properties** - xml_path, action_size, mj_model, mjx_model

