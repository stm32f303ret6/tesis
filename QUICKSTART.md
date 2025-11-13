# MuJoCo Playground Integration: Quick Start (5 Minutes)

## The Essentials

### What You Need to Know
Your 3DOF parallel SCARA quadruped can be integrated into MuJoCo Playground by following a standard pattern used by Go1, Spot, and other quadrupeds.

### Environment Hierarchy
```
MjxEnv (abstract base)
  └── Walk2Env (your base)
      └── Walk2Joystick (velocity tracking task)
```

### Core Requirements

**4 Properties to Implement:**
```python
@property
def xml_path(self) -> str:
    return "/path/to/walk2.xml"

@property
def action_size(self) -> int:
    return 12  # 3 DOF × 4 legs

@property
def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

@property
def mjx_model(self) -> mjx.Model:
    return self._mjx_model
```

**2 Methods to Implement:**
```python
def reset(self, rng: jax.Array) -> State:
    # Initialize environment, return State
    pass

def step(self, state: State, action: jax.Array) -> State:
    # Apply action, simulate, compute reward
    # Return new State
    pass
```

### The State Object
```python
@dataclass
class State:
    data: mjx.Data              # Physics state
    obs: jax.Array | Dict       # Observation (50-60 elements)
    reward: jax.Array           # Scalar reward
    done: jax.Array             # Episode termination flag
    metrics: Dict[str, Array]   # Tracked metrics
    info: Dict[str, Any]        # Extra info
```

### Minimal Step Implementation
```python
def step(self, state, action):
    # 1. Compute motor targets
    target = self._home_qpos + action * self._config.action_scale
    
    # 2. Simulate physics
    data = state.data
    for _ in range(self.n_substeps):
        tau = self._config.kp * (target - data.qpos) - \
              self._config.kd * data.qvel
        data = data.replace(ctrl=tau)
        data = mjx.step(self._mjx_model, data)
    
    # 3. Compute observation
    obs = jnp.concatenate([data.qpos, data.qvel])
    
    # 4. Compute reward
    reward = jnp.sum(action**2) * -0.1  # Simple penalty
    
    # 5. Check termination
    done = data.xpos[trunk_id, 2] < 0.1  # Fell
    
    # 6. Return State
    return State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics={},
        info={}
    )
```

## File Structure
```
walk2/
├── mujoco_playground_src/locomotion/walk2/
│   ├── __init__.py              # Register: 'Walk2Joystick'
│   ├── base.py                  # Walk2Env class
│   ├── joystick.py              # Walk2Joystick class
│   ├── constants.py             # XML paths, sensor names
│   └── xmls/
│       ├── walk2.xml           # Your robot model
│       └── scene_flat.xml      # Scene with includes
│
└── Config: locomotion_params.py
    Add: 'Walk2Joystick': config_override
```

## Key Points for Your Robot

**Action Interpretation:**
```
Policy Action (12,) in [-1, 1]
    ↓
× action_scale (0.5)
    ↓
+ home_qpos (default leg angles)
    ↓
Use as IK target OR direct motor target
    ↓
PD Control: tau = Kp * error - Kd * vel
    ↓
Motor command
```

**Observation Design:**
```python
obs = [
    # 4 tilt angles
    qpos[0:4],
    # 4 shoulder_L angles
    qpos[4:8],
    # 4 shoulder_R angles
    qpos[8:12],
    # 12 velocities
    qvel[0:12],
    # Maybe: velocity command, IMU, etc.
]
# Shape: (28,) or larger
```

**Reward Example:**
```python
# Velocity tracking
v_error = jnp.linalg.norm(actual_vel - target_vel)
tracking_reward = jnp.exp(-5.0 * v_error)

# Energy penalty
energy_penalty = jnp.sum(tau**2) * 0.01

# Stability
is_tipped = torso_angle > 0.3
tip_penalty = 10.0 * is_tipped

reward = tracking_reward - energy_penalty - tip_penalty
# Clip to [0, 10000] and scale by dt
```

## Configuration Example
```python
def get_config():
    return ml_collections.ConfigDict({
        'control_dt': 0.02,      # 50Hz control
        'sim_dt': 0.001,         # 1000Hz sim (20 substeps)
        'action_scale': 0.5,     # Radians per action unit
        'kp': 100.0,             # PD proportional
        'kd': 2.0,               # PD derivative
        'obs_noise': [0.01, 0.01, 0.01],
        'reward_scales': {
            'tracking': 1.0,
            'energy': 0.01,
            'stability': 0.1,
        },
    })
```

## Registration Pattern
```python
# In __init__.py:
ALL_ENVS = ('Walk2Joystick',)

_ENVS = {
    'Walk2Joystick': joystick.Walk2Joystick,
}

_CFGS = {
    'Walk2Joystick': get_config,
}

def load(env_name, config=None, **kwargs):
    cls = _ENVS[env_name]
    cfg = _CFGS[env_name]()
    if config:
        cfg.update(config)
    xml_path = constants.SCENE_XML
    return cls(xml_path=xml_path, config=cfg, **kwargs)
```

## Testing Your Environment
```python
from mujoco_playground._src import registry

# Load
env = registry.load('Walk2Joystick')

# Test reset
rng = jax.random.PRNGKey(0)
state = env.reset(rng)
print(f"Obs shape: {state.obs.shape}")

# Test step
action = jnp.zeros(12)
state = env.step(state, action)
print(f"Reward: {state.reward}")
print(f"Done: {state.done}")
```

## Training
```bash
python learning/train_jax_ppo.py --env_name Walk2Joystick
```

## Integration Checklist

Quick checklist (order of implementation):

1. **Setup**
   - [ ] Create `mujoco_playground/_src/locomotion/walk2/` directory
   - [ ] Copy your XML to `xmls/walk2.xml`

2. **Constants**
   - [ ] Create `constants.py` with paths and sensor names
   - [ ] Define XML path, feet names, body name

3. **Base Environment**
   - [ ] Create `base.py` with `Walk2Env(MjxEnv)`
   - [ ] Implement 4 required properties
   - [ ] Load XML and compile models
   - [ ] Add sensor accessor methods

4. **Physics Loop**
   - [ ] Implement `reset()` - initialize data, return State
   - [ ] Implement `step()` - physics, reward, done
   - [ ] Test with simple forward dynamics

5. **Task**
   - [ ] Create `joystick.py` with `Walk2Joystick(Walk2Env)`
   - [ ] Override `reset()` to sample velocity command
   - [ ] Override `step()` with reward logic
   - [ ] Implement `_compute_reward()` and `_get_obs()`

6. **Registration**
   - [ ] Create `__init__.py`
   - [ ] Register environment in ALL_ENVS, _ENVS, _CFGS
   - [ ] Test: `registry.load('Walk2Joystick')`

7. **Configuration**
   - [ ] Add `locomotion_params.py` entry
   - [ ] Set hyperparameters

8. **Training**
   - [ ] Run `train_jax_ppo.py --env_name Walk2Joystick`
   - [ ] Monitor with Weights & Biases or TensorBoard

## Connecting to Your Existing Code

**Your IK Solver:**
```python
# In step():
# Instead of direct motor commands,
target_positions = ... # x,y,z for each foot
tilt_angles = action[0:4]
# Call your IK:
from walk2.ik import solve_leg_ik_3dof
for leg_idx in range(4):
    angles = solve_leg_ik_3dof(target_positions[leg_idx], 
                                tilt_angles[leg_idx])
    # Apply angles via PD control
```

**Your Gait Controller:**
```python
# Initialize in __init__():
from walk2.gait_controller import DiagonalGaitController
self._gait = DiagonalGaitController(
    GaitParameters(body_height=0.3, step_length=0.1, ...)
)

# In step():
foot_targets = self._gait.evaluate()  # Leg-local frame targets
# Convert to world frame and apply IK
```

## Performance Tips

1. **Action Scaling**: Adjust `action_scale` based on joint ranges
2. **Timesteps**: Use `sim_dt=0.001` and `control_dt=0.02` (20 substeps)
3. **Rewards**: Keep total reward in [0, 10000] range
4. **Observation**: Normalize or add noise for robustness
5. **Network Size**: Start with 256x256 hidden layers for quadrupeds

## Common Issues

**"AttributeError: 'NoneType' has no attribute 'step'"**
- Check that `mjx_model` property is correctly implemented
- Verify XML file exists at specified path

**"Shape mismatch in observation"**
- Ensure all concatenated arrays in `_get_obs()` are flattened
- Check action_size matches number of actuators

**Physics unstable or robot falls immediately**
- Adjust PD gains (Kp, Kd)
- Check joint limits in XML
- Verify home_qpos is valid standing position

**Training not converging**
- Increase observation noise
- Check reward scaling (should be 0-10000)
- Try larger network or longer training

## Next: Full Documentation

After understanding the quick start:
1. Read `MUJOCO_PLAYGROUND_ARCHITECTURE.md` (15 min)
2. Reference `MUJOCO_PLAYGROUND_GUIDE.md` for details
3. Study Go1/Spot examples in the actual repo

---

**Time to Implementation**: 2-4 hours for basic environment
**Time to Training**: Additional 1-2 hours for config and hyperparameter tuning

