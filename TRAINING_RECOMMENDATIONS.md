# Training Recommendations: Adapting Go2 Success to Your Robot

This document provides actionable recommendations to fix your PPO training based on successful go2 quadruped training.

## Executive Summary

Your training converges but performs worse than baseline because:
1. **Observation space is too large and poorly scaled** (~80D vs optimal 48D)
2. **Rewards lack proper dt scaling and are too weak** (height penalty 50x weaker)
3. **Training scale is insufficient** (12 envs vs 4096, 10M vs 983M steps)
4. **Action scale might be too small** (0.02 vs 0.25)
5. **Termination is too lenient** (60° vs 10° roll/pitch)

## Priority 1: Fix Observation Space (CRITICAL)

### Current Issues
- ~80 dimensions with unscaled, redundant information
- Foot positions/velocities (24D) are redundant with joint states
- No observation scaling applied
- Missing multi-dimensional command space

### Recommended Observation (48D)
```python
obs = [
    base_ang_vel * 0.25,                    # 3D - scaled!
    projected_gravity,                      # 3D
    commands * commands_scale,              # 5D: [vx, vy, ωz, height, step_height]
    (joint_pos - default_joint_pos) * 1.0, # 12D - relative to default!
    joint_vel * 0.05,                       # 12D - scaled!
    actions,                                # 12D - previous action
    gait_phase_encoding,                    # 1D - normalized phase counter
]
```

### Implementation
```python
# In _get_observation():

# 1. Scale angular velocity
obs_components.append(angvel * 0.25)  # 3D

# 2. Projected gravity (already correct)
obs_components.append(gravity_body)  # 3D

# 3. Multi-dimensional commands (scaled)
commands = np.array([
    self.target_velocity,      # vx
    0.0,                        # vy (lateral)
    0.0,                        # ωz (yaw rate)
    self.target_height,         # height command
    self.controller.params.step_height,  # step height
])
commands_scale = np.array([2.0, 2.0, 0.25, 2.0, 2.0])
obs_components.append(commands * commands_scale)  # 5D

# 4. Joint positions RELATIVE to default (not absolute!)
joint_states = self.sensor_reader.get_joint_states()[:12]  # positions only
default_joint_pos = np.zeros(12)  # Define your default pose
obs_components.append((joint_states - default_joint_pos) * 1.0)  # 12D

# 5. Joint velocities (scaled)
joint_vels = self.sensor_reader.get_joint_states()[12:24]
obs_components.append(joint_vels * 0.05)  # 12D

# 6. Previous actions
obs_components.append(self.previous_action)  # 12D

# 7. Gait phase (normalized counter, not sin/cos)
phase = self.controller.get_phase_info()["phase_normalized"]
obs_components.append(np.array([phase]))  # 1D

# REMOVE:
# - Quaternion (use projected gravity instead)
# - Foot positions/velocities (redundant)
# - Swing/stance flags (use phase instead)
# - Linear velocity (network should learn from body motion)
```

**Expected observation shape: 48D**

## Priority 2: Fix Reward Function (CRITICAL)

### Current Issues
- No dt scaling (rewards scale incorrectly with timestep)
- Linear tracking instead of exponential
- Height penalty is 50x weaker than go2
- Contact pattern reward is complex and noisy

### Recommended Rewards
```python
def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
    rewards = {}
    dt = self.model.opt.timestep  # Typically 0.002

    # 1. Forward velocity tracking (exponential, not linear!)
    linvel = self.sensor_reader.read_sensor("body_linvel")
    forward_vel = float(linvel[0])
    vel_error_sq = (forward_vel - self.target_velocity) ** 2
    rewards["tracking_lin_vel"] = 1.0 * dt * np.exp(-vel_error_sq / 0.25)

    # 2. Lateral velocity penalty
    lateral_vel = float(linvel[1])
    rewards["tracking_lat_vel"] = 0.5 * dt * np.exp(-lateral_vel**2 / 0.25)

    # 3. Vertical velocity penalty (robot should move horizontally)
    vertical_vel = float(linvel[2])
    rewards["lin_vel_z"] = -1.0 * dt * vertical_vel**2

    # 4. Base height tracking (MUCH STRONGER than current!)
    body_pos = self.sensor_reader.read_sensor("body_pos")
    height_error = float(body_pos[2]) - self.target_height
    rewards["base_height"] = -50.0 * dt * height_error**2  # -50.0, not -1.0!

    # 5. Action rate penalty (smoothness)
    if self.last_last_action is not None:
        action_diff = self.previous_action - self.last_last_action
        rewards["action_rate"] = -0.005 * dt * np.sum(action_diff**2)
    else:
        rewards["action_rate"] = 0.0

    # 6. Orientation penalty (stay upright)
    quat = self.sensor_reader.get_body_quaternion()
    roll, pitch, _ = quat_to_euler(quat)
    rewards["orientation"] = -1.0 * dt * (roll**2 + pitch**2)

    # 7. Joint position deviation from default (encourage natural poses)
    joint_pos = self.sensor_reader.get_joint_states()[:12]
    default_joint_pos = np.zeros(12)  # Define your default
    joint_dev = joint_pos - default_joint_pos
    rewards["similar_to_default"] = -0.1 * dt * np.sum(joint_dev**2)

    # REMOVE:
    # - Complex contact pattern reward (too noisy, let network figure it out)
    # - Energy penalty on action magnitude (action_rate is sufficient)
    # - Joint limits penalty (use proper joint limit enforcement instead)

    total = sum(rewards.values())
    return total, rewards
```

### Key Changes
- ✅ All rewards scaled by `dt`
- ✅ Exponential tracking rewards with σ² = 0.25
- ✅ Height penalty increased from -1.0 to **-50.0**
- ✅ Removed noisy contact pattern reward
- ✅ Added similarity to default pose

## Priority 3: Fix Termination Conditions (HIGH)

### Current
```python
terminated = (
    (body_pos[2] < 0.03) or
    (abs(roll) > π/3) or      # 60° - TOO LENIENT!
    (abs(pitch) > π/3)
)
```

### Recommended (Training Mode)
```python
# Strict termination during training forces good behavior
terminated = (
    (body_pos[2] < 0.03) or
    (abs(roll) > 0.175) or    # 10° like go2
    (abs(pitch) > 0.175)
)
```

### Recommended (Evaluation Mode)
```python
# Relaxed for testing recovery
terminated = (
    (body_pos[2] < 0.03) or
    (abs(roll) > 0.873) or    # 50°
    (abs(pitch) > 0.873)
)
```

**Implementation:** Add `training_mode: bool` parameter to environment constructor.

## Priority 4: Increase Action Scale (MEDIUM)

### Current
```python
residual_scale = 0.02  # Might be too small!
```

### Recommended
```python
# Try progressively larger scales
residual_scale = 0.05  # 2.5x increase (conservative)
# or
residual_scale = 0.10  # 5x increase (moderate)
# or
residual_scale = 0.25  # Match go2 (aggressive, but you're modifying feet not joints)
```

**Rationale:** Your actions modify foot positions (indirect), while go2 modifies joint positions (direct). You might need larger corrections to have meaningful effect after IK projection.

**Recommendation:** Start with 0.05, monitor training. If policy learns to saturate actions at ±1.0 without improving, increase to 0.10.

## Priority 5: Scale Up Training (HIGH)

### Current
- 12 parallel environments
- 10M total timesteps
- Default network [64, 64]

### Minimum Recommended
```python
# train_residual_ppo_v2.py arguments
--n-envs 64              # 5x increase (manageable on single machine)
--total-timesteps 50000000  # 5x increase (50M steps)
--n-steps 2048           # Keep (good value)
--batch-size 512         # Keep (good value)
```

### Optimal (if you have compute)
```python
--n-envs 256             # 21x increase
--total-timesteps 200000000  # 20x increase (200M steps)
--n-steps 2048
--batch-size 1024
```

### Network Size
```python
# In stable_baselines3 PPO:
policy_kwargs = dict(
    net_arch=dict(
        pi=[512, 256, 128],  # Actor (go2 architecture)
        vf=[512, 256, 128],  # Critic
    ),
    activation_fn=torch.nn.ELU,  # go2 uses ELU
)

model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    policy_kwargs=policy_kwargs,
    ...
)
```

## Priority 6: Improve PPO Hyperparameters (MEDIUM)

### Current (v2)
```python
learning_rate=3e-4
gamma=0.99
gae_lambda=0.95
n_epochs=10
ent_coef=0.0
clip_range=0.2
```

### Recommended (Match go2)
```python
learning_rate=1e-3        # Higher LR (go2 uses 0.001)
gamma=0.99                # Keep
gae_lambda=0.95           # Keep
n_epochs=5                # Reduce (go2 uses 5, not 10)
ent_coef=0.01             # Add exploration bonus (go2 uses 0.01)
clip_range=0.2            # Keep
max_grad_norm=1.0         # Add gradient clipping
```

**Rationale:**
- Higher LR speeds up learning (go2 uses 1e-3 vs your 3e-4)
- Fewer epochs (5 vs 10) reduces overfitting per update
- Entropy bonus encourages exploration (go2 uses 0.01, you use 0.0)

## Priority 7: Remove Settle Steps (LOW)

### Current
```python
# In reset():
for _ in range(500):  # 500 settle steps
    # Run baseline controller...
```

### Recommended
```python
# Remove settle steps entirely (like go2)
# Reset to default pose immediately
mujoco.mj_resetData(self.model, self.data)
if key_id != -1:
    mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
# Start episode immediately
```

**Rationale:**
- go2 doesn't use settle steps
- Forces policy to learn from unstable initial conditions
- Faster training (no wasted steps)

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Fix observation space (48D, scaled)
2. ✅ Fix reward function (dt scaling, exponential tracking, -50.0 height)
3. ✅ Fix termination (10° roll/pitch during training)
4. ✅ Increase action scale (0.05 → 0.10)
5. ✅ Remove settle steps

**Expected:** 2-3x improvement in learned performance

### Phase 2: Scale Up (requires compute)
6. ✅ Increase parallel envs (64 or 256)
7. ✅ Increase training duration (50M → 200M steps)
8. ✅ Use larger network [512, 256, 128]
9. ✅ Adjust PPO hyperparameters (LR=1e-3, entropy=0.01, epochs=5)

**Expected:** Approach or exceed baseline performance

### Phase 3: Advanced (optional)
10. Add command curriculum (start with easy commands, increase difficulty)
11. Add domain randomization (physics params, terrain difficulty)
12. Implement adaptive learning rate (go2 uses "adaptive" schedule)
13. Add action latency simulation (1-step delay like go2)

## Quick Start: Minimal Code Changes

To get started immediately, here's the minimal change set for `residual_walk_env.py`:

```python
# 1. Update _get_observation() - reduce to 48D
# 2. Update _compute_reward() - add dt scaling, exponential tracking, -50.0 height
# 3. Update _check_termination() - use 0.175 rad (10°) for roll/pitch
# 4. Update __init__() - increase residual_scale to 0.10
# 5. Update reset() - remove settle_steps loop

# Then train with more envs:
python3 train_residual_ppo_v2.py \
    --n-envs 64 \
    --total-timesteps 50000000 \
    --learning-rate 0.001 \
    --ent-coef 0.01 \
    --n-epochs 5 \
    --run-name fixed_v1
```

## Debugging Checklist

After implementing changes, verify:
- [ ] Observation shape is exactly 48
- [ ] Observation values are reasonably scaled (check with `print(obs.min(), obs.max())`)
- [ ] Rewards contain `dt` factor in all components
- [ ] Height reward dominates early training (should be largest magnitude)
- [ ] Episode length is reasonable (not terminating too early)
- [ ] Action saturation is < 50% (check `abs(action) > 0.9`)
- [ ] TensorBoard shows improving `ep_rew_mean` over time

## Expected Training Metrics (After Fixes)

### Early Training (0-10M steps)
- ep_rew_mean: -50 to -10 (negative due to learning)
- ep_len_mean: 200-500 steps (frequent falls)
- Dominant reward: base_height (most negative)

### Mid Training (10-50M steps)
- ep_rew_mean: -10 to +5
- ep_len_mean: 500-800 steps
- tracking_lin_vel becomes positive

### Late Training (50M+ steps)
- ep_rew_mean: +5 to +15
- ep_len_mean: 800-1000 steps (hitting max)
- All rewards balanced, positive total

## Summary

The go2 training succeeds because:
1. **Compact, well-scaled observations** (48D with proper normalization)
2. **Strong, properly scaled rewards** (dt factor, exponential tracking, -50.0 height)
3. **Massive training scale** (4096 envs, 983M steps, large network)
4. **Strict termination** (10° roll/pitch forces good behavior)

Your training struggles because:
1. ❌ Observation too large and unscaled (~80D)
2. ❌ Rewards too weak and lacking dt scaling
3. ❌ Insufficient training (12 envs, 10M steps)
4. ❌ Lenient termination (60° allows bad behavior)

**Priority:** Implement Phase 1 changes first. These are quick wins that should show immediate improvement. Then scale up compute for Phase 2.

Good luck! 🚀
