# PPO Training Summary: Go2 Quadruped Locomotion

This document provides a comprehensive summary of the Reinforcement Learning problem formulation for training the Unitree Go2 quadruped robot using Proximal Policy Optimization (PPO).

---

## Table of Contents
1. [Action Space](#action-space)
2. [Observation Space](#observation-space)
3. [Reward Function](#reward-function)
4. [Episode Termination Conditions](#episode-termination-conditions)
5. [Reset Logic](#reset-logic)
6. [Command Generation](#command-generation)
7. [PPO Hyperparameters](#ppo-hyperparameters)
8. [Environment Configuration](#environment-configuration)

---

## Action Space

**Dimensionality**: 12 (continuous)

**Description**: Residual position adjustments for each of the 12 robot joints (3 per leg: hip, thigh, calf)

**Joint Order**:
```python
dof_names = [
    "FR_hip_joint",   "FR_thigh_joint",   "FR_calf_joint",    # Front Right
    "FL_hip_joint",   "FL_thigh_joint",   "FL_calf_joint",    # Front Left
    "RR_hip_joint",   "RR_thigh_joint",   "RR_calf_joint",    # Rear Right
    "RL_hip_joint",   "RL_thigh_joint",   "RL_calf_joint",    # Rear Left
]
```

**Default Joint Angles** (homing position in radians):
```python
{
    "FL_hip_joint": 0.0,    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,  "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,  "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,  "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,  "RR_calf_joint": -1.5,
}
```

**Action Processing**:
```python
# Actions are clipped
actions = clip(actions, -clip_actions, clip_actions)  # clip_actions = 100.0

# 1-step latency simulation (real robot behavior)
executed_actions = last_actions if simulate_action_latency else actions

# Residual formulation: actions modify default pose
target_dof_pos = executed_actions * action_scale + default_dof_pos
# action_scale = 0.25
```

**Control Type**: PD position control
- kp = 20.0
- kd = 0.5

---

## Observation Space

**Dimensionality**: 48 (continuous)

**Components** (in order):

| Component | Dimensions | Range/Scale | Description |
|-----------|-----------|-------------|-------------|
| Base angular velocity | 3 | ×0.25 | ωx, ωy, ωz in base frame (rad/s) |
| Projected gravity | 3 | unscaled | Gravity vector in base frame |
| Commands | 5 | scaled* | [vx, vy, ωz, height, jump] |
| Joint positions (relative) | 12 | ×1.0 | (q - q_default) |
| Joint velocities | 12 | ×0.05 | q̇ (rad/s) |
| Previous actions | 12 | unscaled | actions at t-1 |
| Jump toggle state | 1 | [0, 1] | Normalized counter |

**Total**: 3 + 3 + 5 + 12 + 12 + 12 + 1 = **48 dimensions**

**Command Scaling**:
```python
commands_scale = [
    lin_vel_scale,   # 2.0 for vx
    lin_vel_scale,   # 2.0 for vy
    ang_vel_scale,   # 0.25 for ωz
    lin_vel_scale,   # 2.0 for height
    lin_vel_scale,   # 2.0 for jump
]
```

**Observation Computation** (from `go2_env.py:237-248`):
```python
obs = torch.cat([
    base_ang_vel * 0.25,                              # 3
    projected_gravity,                                # 3
    commands * commands_scale,                        # 5
    (dof_pos - default_dof_pos) * 1.0,               # 12
    dof_vel * 0.05,                                   # 12
    actions,                                          # 12
    (jump_toggled_buf / jump_reward_steps).unsqueeze(-1),  # 1
], axis=-1)
```

---

## Reward Function

**Total Reward**: Sum of weighted individual reward components (weights include dt factor)

### Reward Components

| Reward Name | Weight | Formula | Active Condition |
|-------------|--------|---------|------------------|
| **tracking_lin_vel** | 1.0 × dt | exp(-‖v_cmd - v_xy‖² / 0.25) | Always |
| **tracking_ang_vel** | 0.2 × dt | exp(-(ω_cmd - ω_z)² / 0.25) | Always |
| **lin_vel_z** | -1.0 × dt | v_z² | When NOT jumping |
| **base_height** | -50.0 × dt | (z - z_cmd)² | When NOT jumping |
| **action_rate** | -0.005 × dt | ‖a_t - a_{t-1}‖² | When NOT jumping |
| **similar_to_default** | -0.1 × dt | ‖q - q_default‖ | When NOT jumping |
| **jump_height_tracking** | 0.5 × dt | exp(-(z - z_target)²) | Peak phase only |
| **jump_height_achievement** | 10.0 × dt | 1.0 if \|z - z_target\| < 0.2 else 0.0 | Peak phase only |
| **jump_speed** | 1.0 × dt | exp(v_z) × 0.2 | Peak phase only |
| **jump_landing** | 0.08 × dt | -(z - z_default)² | Landing phase only |

**Note**: dt = 0.02 (control timestep)

### Jump Phases

The jump toggle buffer counts down from `jump_reward_steps` (50 steps) when a jump is commanded:

- **Preparation** (steps 50-30): Regular rewards apply, robot prepares to jump
- **Peak** (steps 30-20): Jump height and speed rewards active
- **Landing** (steps 20-0): Landing penalty active to encourage proper landing

### Active Masks

Many rewards use `active_mask = (jump_toggled_buf < 0.01).float()` to disable during jumps:
- Vertical velocity penalty
- Action rate penalty
- Pose similarity penalty
- Base height tracking

---

## Episode Termination Conditions

Episodes terminate when **any** of the following conditions are met:

```python
# From go2_env.py:219-221
terminate = (
    (episode_length_buf > max_episode_length) OR
    (abs(roll) > termination_if_roll_greater_than) OR
    (abs(pitch) > termination_if_pitch_greater_than)
)
```

### Termination Parameters

| Parameter | Training Value | Evaluation Value | Description |
|-----------|---------------|------------------|-------------|
| max_episode_length | 1000 steps | 1000 steps | 20 seconds @ 50Hz |
| termination_if_roll_greater_than | 10° | 50° | Roll angle threshold |
| termination_if_pitch_greater_than | 10° | 50° | Pitch angle threshold |

**Note**: Evaluation uses relaxed thresholds (50° vs 10°) to allow testing of recovery behaviors.

### Time-out Tracking

```python
# Time-outs are tracked separately for proper advantage computation
time_out_idx = (episode_length_buf > max_episode_length).nonzero()
extras["time_outs"][time_out_idx] = 1.0
```

---

## Reset Logic

### Reset Triggers

1. Episode termination conditions met
2. Explicit reset requested

### Reset Procedure (from `go2_env.py:264-307`)

When environments are reset:

**1. Joint State Reset**:
```python
dof_pos[envs_idx] = default_dof_pos
dof_vel[envs_idx] = 0.0
robot.set_dofs_position(dof_pos[envs_idx], zero_velocity=True)
```

**2. Base State Reset**:
```python
base_pos[envs_idx] = base_init_pos  # [0.0, 0.0, 0.42]
base_quat[envs_idx] = base_init_quat  # [1.0, 0.0, 0.0, 0.0]
base_lin_vel[envs_idx] = 0
base_ang_vel[envs_idx] = 0
robot.zero_all_dofs_velocity(envs_idx)
```

**3. Buffer Reset**:
```python
last_actions[envs_idx] = 0.0
last_dof_vel[envs_idx] = 0.0
episode_length_buf[envs_idx] = 0
reset_buf[envs_idx] = True
jump_toggled_buf[envs_idx] = 0.0
jump_target_height[envs_idx] = 0.0
```

**4. Command Initialization**:
```python
# Sample new random commands
_sample_commands(envs_idx)

# Override height to default
commands[envs_idx, 3] = base_height_target  # 0.3 m
```

**5. Episode Statistics Logging**:
```python
# Average reward per second for each component
for key in episode_sums.keys():
    extras["episode"]["rew_" + key] = mean(episode_sums[key][envs_idx]) / episode_length_s
    episode_sums[key][envs_idx] = 0.0
```

**Note**: No randomization of initial states during training (unlike some RL implementations).

---

## Command Generation

Commands specify target velocities and height for the robot to track.

### Command Space

**Dimensionality**: 5
- `commands[0]`: Linear velocity x (forward/backward)
- `commands[1]`: Linear velocity y (left/right)
- `commands[2]`: Angular velocity z (yaw rotation)
- `commands[3]`: Base height target
- `commands[4]`: Jump height (transient, reset to 0 after processing)

### Command Ranges (Training)

```python
lin_vel_x_range = [-1.0, 2.0]   # m/s
lin_vel_y_range = [-0.5, 0.5]   # m/s
ang_vel_range = [-0.6, 0.6]     # rad/s
height_range = [0.2, 0.4]       # m (default: 0.3)
jump_range = [0.5, 1.5]         # m
```

### Command Sampling Strategy (from `go2_env.py:156-168`)

**Primary Resampling** (every 4 seconds = 200 steps):
```python
if episode_length_buf % (resampling_time_s / dt) == 0:
    # Sample new commands
    commands[:, 0] = uniform(lin_vel_x_range)
    commands[:, 1] = uniform(lin_vel_y_range)
    commands[:, 2] = uniform(ang_vel_range)
    commands[:, 3] = uniform(height_range)
    commands[:, 4] = 0.0
```

**Additional Random Sampling** (each step during training):
```python
# 5% of environments get completely random commands
random_idxs_1 = random_permutation(num_envs)[:int(num_envs * 0.05)]
_sample_commands(random_idxs_1)

# Another 5% get random jump commands
random_idxs_2 = random_permutation(num_envs)[:int(num_envs * 0.05)]
commands[random_idxs_2, 4] = uniform(jump_range)
```

### Adaptive Command Scaling

Commands are scaled based on height deviation to maintain stability:

```python
# Scale velocities proportionally to height difference from default
height_diff_scale = 0.5 + abs(commands[3] - base_height_target) /
                          (height_range[1] - base_height_target) * 0.5

commands[0] *= height_diff_scale  # Scale vx
commands[1] *= height_diff_scale  # Scale vy
commands[2] *= height_diff_scale  # Scale ωz
```

**Effect**: When robot is at extreme heights, velocity commands are reduced (minimum 50% of original).

### Jump Command Processing

```python
# Detect jump toggle (command goes from 0 → non-zero)
jump_cmd_now = (commands[:, 4] > 0.0).float()
toggle_mask = ((jump_toggled_buf == 0.0) & (jump_cmd_now > 0.0)).float()

# Activate jump toggle buffer for N steps
jump_toggled_buf += toggle_mask * jump_reward_steps  # 50 steps

# Store target jump height
jump_target_height = where(jump_cmd_now > 0.0, commands[:, 4], jump_target_height)

# Decrement buffer each step
jump_toggled_buf = clamp(jump_toggled_buf - 1.0, min=0.0)

# Reset jump command to 0 after processing
commands[:, 4] = 0.0
```

---

## PPO Hyperparameters

**Algorithm Configuration** (from `go2_train.py:14-28`):

```python
algorithm = {
    "clip_param": 0.2,              # PPO clipping parameter ε
    "desired_kl": 0.01,             # Target KL divergence for adaptive learning rate
    "entropy_coef": 0.01,           # Entropy bonus coefficient
    "gamma": 0.99,                  # Discount factor
    "lam": 0.95,                    # GAE lambda
    "learning_rate": 0.001,         # Adam learning rate
    "max_grad_norm": 1.0,           # Gradient clipping threshold
    "num_learning_epochs": 5,       # Epochs per PPO update
    "num_mini_batches": 4,          # Mini-batches per epoch
    "schedule": "adaptive",         # Learning rate schedule
    "use_clipped_value_loss": True, # Use clipped value function loss
    "value_loss_coef": 1.0,         # Value function loss coefficient
}
```

**Policy Network Architecture** (from `go2_train.py:30-35`):

```python
policy = {
    "activation": "elu",                    # Activation function
    "actor_hidden_dims": [512, 256, 128],   # Actor MLP layers
    "critic_hidden_dims": [512, 256, 128],  # Critic MLP layers
    "init_noise_std": 1.0,                  # Initial action noise std dev
}
```

**Network Structure**:
- **Actor**: 48 (obs) → 512 → 256 → 128 → 12 (actions)
- **Critic**: 48 (obs) → 512 → 256 → 128 → 1 (value)

**Training Configuration** (from `go2_train.py:36-51`):

```python
runner = {
    "algorithm_class_name": "PPO",
    "policy_class_name": "ActorCritic",
    "num_steps_per_env": 24,        # Rollout length per environment
    "max_iterations": 10000,        # Total training iterations (default)
    "save_interval": 100,           # Save checkpoint every N iterations
    "log_interval": 1,              # Log metrics every N iterations
}
```

**Training Loop**:
- **Environments**: 4096 (default, parallelized)
- **Steps per iteration**: 4096 envs × 24 steps = 98,304 steps
- **Updates per iteration**: 5 epochs × 4 mini-batches = 20 gradient updates
- **Total training steps**: 10,000 iterations × 98,304 = ~983M steps

---

## Environment Configuration

**Simulation Parameters** (from `go2_env.py:32-45`):

```python
dt = 0.02                    # Control timestep: 50 Hz
substeps = 2                 # Physics substeps per control step
episode_length_s = 20.0      # Episode duration: 20 seconds
max_episode_length = 1000    # 20s / 0.02s = 1000 steps
resampling_time_s = 4.0      # Command resampling interval
```

**Robot Configuration**:

```python
base_init_pos = [0.0, 0.0, 0.42]     # Initial base position (m)
base_init_quat = [1.0, 0.0, 0.0, 0.0] # Initial orientation (w, x, y, z)
```

**Control Parameters**:

```python
kp = 20.0                    # Position gain
kd = 0.5                     # Derivative gain
action_scale = 0.25          # Action → position mapping scale
clip_actions = 100.0         # Action clipping threshold
simulate_action_latency = True  # 1-step delay (real robot behavior)
```

**Physics Solver**:

```python
constraint_solver = "Newton"
enable_collision = True
enable_joint_limit = True
```

**Reward Configuration**:

```python
tracking_sigma = 0.25              # Gaussian kernel width for tracking rewards
base_height_target = 0.3           # Target base height (m)
feet_height_target = 0.075         # Target foot clearance (m)
jump_upward_velocity = 1.2         # Target upward velocity for jumps (m/s)
jump_reward_steps = 50             # Jump phase duration (steps)
```

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Observation Space** | 48 dimensions (continuous) |
| **Action Space** | 12 dimensions (continuous, residual) |
| **Number of Rewards** | 10 components |
| **Episode Length** | 20 seconds (1000 steps @ 50Hz) |
| **Parallel Environments** | 4096 (default) |
| **Steps per Iteration** | 98,304 |
| **Network Parameters** | ~600k (approximate, for [512,256,128] MLP) |
| **Training Duration** | 10,000 iterations (default) |
| **Total Training Steps** | ~983M environment steps |

---

## References

**Source Files**:
- Environment: `src/go2_env.py`
- Training: `src/go2_train.py`
- Evaluation: `src/go2_eval.py`, `src/go2_eval_teleop.py`

**Framework**:
- PPO Implementation: `rsl_rl/` (RSL-RL framework)
- Physics Simulator: `genesis/` (Genesis)

---

*Generated for the quadruped locomotion training project*
