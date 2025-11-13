# MuJoCo Playground: Custom Environment Integration Guide

This package contains comprehensive documentation for integrating your 3DOF parallel SCARA quadruped with the MuJoCo Playground framework.

## Quick Start

If you want to quickly understand how to create a custom quadruped environment:

1. **Start with architecture**: Read `MUJOCO_PLAYGROUND_ARCHITECTURE.md` (15 min read)
   - Visual diagrams showing the class hierarchy
   - Data flow for a single step
   - Configuration structure
   - Quick reference for key patterns

2. **Then read the full guide**: `MUJOCO_PLAYGROUND_GUIDE.md` (comprehensive reference)
   - All 8 sections with detailed explanations
   - Complete code examples
   - Import statements and dependencies
   - XML model loading patterns

## Document Overview

### MUJOCO_PLAYGROUND_ARCHITECTURE.md (368 lines)
**Purpose**: High-level architectural patterns and reference material

Contains:
- Core architecture diagram showing class relationships
- Class hierarchy for quadruped environments
- Configuration flow and timing
- Key file organization patterns
- Data flow for a single environment step
- Observation structure examples
- Action to motor command mapping
- Reward structure
- Critical implementation points
- Common pitfalls

Best for: Quick lookups, understanding the big picture, debugging

### MUJOCO_PLAYGROUND_GUIDE.md (938 lines)
**Purpose**: Comprehensive reference guide with code examples

Contains 8 major sections:

**Part 1**: Directory structure for environments
- Complete directory tree with descriptions
- Where quadruped environments live
- How different robot types are organized

**Part 2**: Base class and API for custom environments
- `MjxEnv` abstract base class details
- Required abstract methods (reset, step)
- Required properties (xml_path, action_size, mj_model, mjx_model)
- Utility properties and methods
- State dataclass structure

**Part 3**: Observations, actions, and rewards
- Example observation structures (Spot Joystick)
- Action representation and motor control
- Reward computation with 14+ components
- Example: Velocity tracking task

**Part 4**: Example quadruped environment files
- Go1 environment structure (base.py, joystick.py, constants.py)
- File organization patterns
- Spot getup (recovery) environment
- Task-specific reward gating

**Part 5**: Key imports and dependencies
- Essential imports (JAX, MuJoCo, Flax)
- Required dependencies from pyproject.toml
- Optional packages for training/vision

**Part 6**: How environments load MuJoCo XML models
- Pattern 1: Direct path loading
- Pattern 2: Asset loading with Menagerie
- Pattern 3: XML with includes
- Key compilation steps

**Part 7**: Complete minimal environment template
- Full working example for custom quadruped
- Registration pattern in locomotion module

**Part 8**: Training and using custom environments
- Loading from registry
- Training with Brax PPO

Best for: Detailed reference, learning by example, implementation guide

## Your 3DOF Parallel SCARA Quadruped

To integrate your walk2 robot with MuJoCo Playground:

### Step 1: Create Environment Module Structure
```
mujoco_playground/_src/locomotion/walk2/
├── __init__.py                # Module registration
├── base.py                    # Extends MjxEnv
├── joystick.py               # Velocity tracking task
├── constants.py              # Robot-specific constants
└── xmls/                     # MuJoCo models
    ├── walk2.xml            # Your robot model
    ├── scene_flat.xml       # Scene definition
    └── sensors.xml          # Sensor definitions
```

### Step 2: Adapt Your Existing Code

Your current codebase (`height_control.py`, `ik.py`, `gait_controller.py`) provides:
- **IK solver** (`parallel_scara_ik_3dof`) - Essential for leg control
- **Gait controller** - State machine for coordination
- **Existing XML models** - Ready to adapt

The MuJoCo Playground approach:
- Wraps these components in the standard environment API
- Provides JAX-based physics simulation
- Enables training with Brax PPO
- Allows domain randomization and benchmarking

### Step 3: Key Integration Points

**Observation Design** (adapt to your robot):
```python
# Based on your sensor suite
obs = jnp.concatenate([
    # Your IK inputs
    tilt_angles (4),           # One per leg
    shoulder_L_angles (4),
    shoulder_R_angles (4),
    
    # IMU feedback
    gyro (3),
    accelerometer (3),
    
    # Proprioception
    qpos_offset (12),          # Relative to home
    qvel (12),
    
    # Task command
    velocity_cmd (3),          # vx, vy, wz
])
```

**Action to IK Mapping**:
```python
# Your action space maps to IK target positions
action (12,) -> action_scale -> x, y, z targets per leg
           -> parallel_scara_ik_3dof()
           -> motor commands
```

**Reward Design** (example):
```python
reward = (
    velocity_tracking_reward(v_actual, v_cmd) +
    gait_stability_reward() +
    energy_efficiency_reward() +
    foot_contact_reward()
)
```

## Key Concepts to Master

### 1. State Management (JAX-specific)
- All state updates via `state.tree_replace()`
- Never mutate JAX arrays directly
- Use `data.replace()` for mjx.Data objects

### 2. Timing Configuration
- `control_dt`: Your control loop frequency (typically 0.02 = 50Hz)
- `sim_dt`: Physics simulation timestep (typically 0.001 = 1000Hz)
- `n_substeps`: Automatically computed (ceil(control_dt / sim_dt) = 20)

### 3. PD Control Pattern
```python
target_qpos = home_qpos + action * action_scale
tau = Kp * (target_qpos - current_qpos) - Kd * current_qvel
```

For your SCARA legs:
- Use IK to convert desired (x,y,z,tilt) to joint angles
- Apply PD control to track desired angles
- Adjust Kp/Kd for your link lengths and masses

### 4. Registry System
```python
# From anywhere:
env = registry.load('Walk2VelocityTracking', config={...})

# Automatically finds your environment:
# mujoco_playground/_src/locomotion/walk2/__init__.py
```

## Integration Checklist

- [ ] Create `mujoco_playground/_src/locomotion/walk2/` directory
- [ ] Copy your robot XML model to `xmls/walk2.xml`
- [ ] Create `base.py` extending `MjxEnv`
- [ ] Implement `reset()`, `step()`, required properties
- [ ] Integrate IK solver from your `ik.py`
- [ ] Create `joystick.py` for velocity tracking task
- [ ] Create `constants.py` with robot configuration
- [ ] Register environment in `__init__.py`
- [ ] Create `configs/walk2_params.py` for hyperparameters
- [ ] Test loading: `registry.load('Walk2VelocityTracking')`
- [ ] Run training: `python learning/train_jax_ppo.py --env_name Walk2VelocityTracking`

## Reference Files in MuJoCo Playground

Key files to study (in order of importance for your use case):

1. **Base class**: `mujoco_playground/_src/mjx_env.py`
   - Understand the environment API
   
2. **Go1 Example**: `mujoco_playground/_src/locomotion/go1/`
   - Most complete example
   - Similar complexity to your robot
   
3. **Spot Example**: `mujoco_playground/_src/locomotion/spot/`
   - Shows recovery task (Getup)
   - Different terrain variations
   
4. **Registry**: `mujoco_playground/_src/registry.py`
   - Understand how environments are loaded
   
5. **Training**: `learning/train_jax_ppo.py`
   - See how environments are used for training
   
6. **Configurations**: `mujoco_playground/config/locomotion_params.py`
   - Example hyperparameters for training

## Dependencies

Core requirements:
```
mujoco >= 3.3.6.dev
mujoco-mjx >= 3.3.6.dev
jax
flax
brax >= 0.12.5
ml_collections
```

Training:
```
rsl-rl-lib >= 3.0.0  (alternative RL algorithm)
wandb                (experiment tracking)
```

Vision (optional):
```
madrona-mjx          (GPU rendering)
```

## Next Steps

1. **Immediate**: Clone MuJoCo Playground and familiarize yourself with Go1/Spot examples
2. **Short-term**: Create minimal base environment for your robot
3. **Medium-term**: Implement velocity tracking task
4. **Long-term**: Add domain randomization, multiple tasks, terrain variations

## Troubleshooting

**Environment fails to load**
- Check that all required properties are implemented
- Verify XML path is correct
- Ensure action_size matches your actuators

**Physics simulation is unstable**
- Check joint limits in XML
- Adjust timestep (increase sim_dt)
- Verify PD gains (Kp, Kd)

**Observations contain NaNs**
- Check for divide-by-zero in reward computation
- Verify sensor names match XML definitions
- Look for uninitialized variables

**Training is slow or diverges**
- Check observation normalization
- Verify reward scaling (should be 0-10000 range)
- Adjust network sizes in config
- Try domain randomization

## Resources

- **MuJoCo Documentation**: https://mujoco.readthedocs.io/
- **Brax Documentation**: https://github.com/google/brax
- **JAX Documentation**: https://jax.readthedocs.io/
- **MuJoCo Playground Repo**: https://github.com/google-deepmind/mujoco_playground

---

**Last Updated**: November 2024
**MuJoCo Playground Version**: 0.0.4+
**Guide Completeness**: Very Thorough (1300+ lines across 2 documents)

