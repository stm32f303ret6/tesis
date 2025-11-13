# Adaptive Gait Training Guide

## Overview

This guide explains how to train policies that learn **adaptive gait parameters** in addition to residual corrections. This approach is more powerful than residual-only learning for rough terrain locomotion.

### Residual-Only vs Adaptive Gait

| Approach | Action Space | What It Learns | Best For |
|----------|--------------|----------------|----------|
| **Residual-Only** | 12D (3D per leg) | Fine-grained foot position corrections | Mild terrain variations, smooth surfaces |
| **Adaptive Gait** | 16D (4 gait params + 12 residuals) | High-level gait modulation + fine corrections | Rough terrain, varied conditions |

### Why Adaptive Gait?

On rough terrain, sometimes you need more than small corrections:
- **Higher steps** to clear obstacles
- **Shorter/longer strides** for stability vs speed
- **Faster/slower cadence** based on terrain difficulty
- **Different body height** for ground clearance

Residual corrections alone may not be enough to make these large adaptations.

## Architecture

### Key Components

1. **`AdaptiveGaitController`** (`controllers/adaptive_gait_controller.py`)
   - Wraps `DiagonalGaitController` with parameter adaptation
   - Accepts 4 parameter deltas: `step_height`, `step_length`, `cycle_time`, `body_height`
   - Applies clipping to keep parameters in safe ranges
   - Rebuilds Bézier curves when parameters change

2. **`AdaptiveGaitEnv`** (`envs/adaptive_gait_env.py`)
   - Observation: 69D (includes current gait params)
   - Action: 16D (4 param deltas + 12 residuals)
   - Reward includes parameter smoothness term to discourage wild changes

3. **Training Script** (`train_adaptive_gait_ppo.py`)
   - Similar to `train_residual_ppo_v3.py` but uses `AdaptiveGaitEnv`
   - Recommended: 10M+ timesteps (more complex policy)

## Quick Start

### 1. Train Adaptive Gait Policy

```bash
# Train for 10M steps (takes ~2-4 hours on 80 parallel envs)
python3 train_adaptive_gait_ppo.py
```

The script will save to `runs/adaptive_gait_YYYYMMDD_HHMMSS/`.

### 2. Evaluate and Visualize

```bash
# Play the trained policy and watch gait parameters adapt
python3 play_adaptive_policy.py \
  --model runs/adaptive_gait_XXX/final_model.zip \
  --normalize runs/adaptive_gait_XXX/vec_normalize.pkl \
  --seconds 30 \
  --deterministic
```

Watch the console output - it will print real-time gait parameter values:
```
[t=5.2s] Gait params: step_h=0.0450m, step_l=0.0620m, cycle_t=0.785s, body_h=0.0515m
[t=6.3s] Gait params: step_h=0.0485m, step_l=0.0590m, cycle_t=0.820s, body_h=0.0495m
...
```

### 3. Compare with Residual-Only

To understand the benefit, train both approaches and compare:

```bash
# Train residual-only baseline
python3 train_residual_ppo_v3.py

# Train adaptive gait
python3 train_adaptive_gait_ppo.py

# Compare performance in TensorBoard
tensorboard --logdir runs/
```

Look for:
- Higher `ep_rew_mean` in adaptive gait
- Longer `ep_len_mean` (fewer falls)
- Better forward velocity tracking

## Tuning Guide

### If the policy doesn't adapt parameters much

**Symptoms:**
- Gait parameters stay close to base values
- Little variance in parameter values during evaluation
- Similar performance to residual-only

**Solutions:**
1. **Increase parameter delta scales** in `AdaptiveGaitEnv.PARAM_DELTA_SCALES`:
   ```python
   PARAM_DELTA_SCALES = {
       "step_height": 0.01,   # Increase from 0.005 to 0.01
       "step_length": 0.01,   # Increase from 0.005 to 0.01
       "cycle_time": 0.1,     # Increase from 0.05 to 0.1
       "body_height": 0.006,  # Increase from 0.003 to 0.006
   }
   ```

2. **Reduce parameter smoothness penalty** in `_compute_reward()`:
   ```python
   rewards["parameter_smoothness"] = -2.0 * float(param_penalty)  # Reduce from -5.0
   ```

3. **Add parameter exploration bonus** to encourage trying different values:
   ```python
   # In _compute_reward()
   param_variance = np.var([current_params[k] for k in ["step_height", "step_length"]])
   rewards["exploration"] = 10.0 * float(param_variance)
   ```

### If the policy is unstable

**Symptoms:**
- Robot falls frequently
- Gait parameters oscillate wildly
- Training diverges or plateaus early

**Solutions:**
1. **Reduce parameter delta scales** (make changes slower)
2. **Increase parameter smoothness penalty** (penalize large changes)
3. **Reduce learning rate**: Try `1e-5` instead of `1e-4`
4. **Add temporal smoothing**: Filter parameter changes over multiple steps
5. **Reduce residual scale**: Try `0.005` instead of `0.01`

### If training is too slow

**Solutions:**
1. **Increase n_envs**: Use 128 or 160 parallel environments
2. **Use smaller network**: Set `network_size="medium"` instead of `"large"`
3. **Reduce n_epochs**: Try 5 instead of 10
4. **Train on flat terrain first**, then fine-tune on rough terrain

## Advanced: Hierarchical Control

For even better performance, consider a **two-level hierarchy**:

1. **High-level policy** (slow, 10 Hz):
   - Outputs: gait parameter targets
   - Inputs: terrain features, body state, contact pattern
   - Updates every 0.1s

2. **Low-level policy** (fast, 500 Hz):
   - Outputs: residual corrections
   - Inputs: detailed joint/foot state, current gait params
   - Updates every timestep

This is more complex to implement but can be more stable and sample-efficient.

## Parameter Ranges

Default safe ranges for gait parameters:

| Parameter | Min | Max | Default | Notes |
|-----------|-----|-----|---------|-------|
| `step_height` | 0.015m | 0.06m | 0.04m | Limited by leg reach ~0.105m |
| `step_length` | 0.03m | 0.08m | 0.06m | Longer = faster but less stable |
| `cycle_time` | 0.6s | 1.2s | 0.8s | Shorter = faster cadence |
| `body_height` | 0.04m | 0.08m | 0.05m | Limited by leg geometry |

These ranges are defined in `AdaptiveGaitController.DEFAULT_RANGES` and can be customized.

## Debugging Tips

### 1. Check parameter adaptation

Add logging in `AdaptiveGaitEnv.step()`:
```python
if self.step_count % 100 == 0:
    params = self.controller.get_current_parameters()
    print(f"Step {self.step_count}: {params}")
```

### 2. Visualize parameter trajectories

Modify `play_adaptive_policy.py` to save parameter history and plot:
```python
import matplotlib.pyplot as plt

# After playback loop
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, values) in zip(axes.flat, gait_param_history.items()):
    ax.plot(values)
    ax.set_title(name)
    ax.set_xlabel("Time step")
plt.tight_layout()
plt.savefig("gait_param_adaptation.png")
```

### 3. Compare with baseline gait

Log the distance from base parameters:
```python
param_deviation = sum([
    abs(current_params[k] - getattr(self.controller.base_params, k))
    for k in ["step_height", "step_length", "cycle_time", "body_height"]
])
info["param_deviation"] = param_deviation
```

## Expected Results

After 10M timesteps of training, you should see:

1. **Terrain-aware adaptation**:
   - Higher `step_height` when encountering obstacles
   - Shorter `step_length` on difficult terrain
   - Slower `cycle_time` for stability

2. **Better performance metrics**:
   - 10-20% higher `ep_rew_mean` vs residual-only
   - Longer episode lengths (fewer falls)
   - More consistent forward velocity

3. **Interpretable behavior**:
   - Gait parameters should correlate with terrain difficulty
   - Parameters should be relatively smooth (not oscillating wildly)

## Common Issues

### Issue: "Non-finite values in observation"

**Cause:** Controller rebuild during parameter update can cause discontinuities.

**Fix:** Add observation clipping or smooth parameter transitions:
```python
def _rebuild_controller_if_needed(self):
    if self._params_dirty:
        # Smooth transition: interpolate parameters over N steps
        for _ in range(5):
            self.base_controller.update(self.model.opt.timestep * 0.2)
        self._params_dirty = False
```

### Issue: "Policy learns to set extreme parameters"

**Cause:** Reward doesn't penalize extreme values enough.

**Fix:** Add explicit penalty for being far from base:
```python
extreme_penalty = 0.0
for param_name in ["step_height", "step_length", "cycle_time", "body_height"]:
    current_val = current_params[param_name]
    base_val = getattr(self.controller.base_params, param_name)
    min_val, max_val, _ = self.controller.param_ranges[param_name]
    # Penalize being in outer 20% of range
    if current_val < min_val + 0.2 * (max_val - min_val):
        extreme_penalty += 1.0
    if current_val > max_val - 0.2 * (max_val - min_val):
        extreme_penalty += 1.0
rewards["extreme_penalty"] = -10.0 * extreme_penalty
```

## Further Reading

- **[Walk These Ways](https://arxiv.org/abs/2403.15693)**: Combines model-based control with RL
- **[Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251)**: Anymal adaptive gait
- **[RMA: Rapid Motor Adaptation](https://arxiv.org/abs/2107.04034)**: Adaptation module learns to predict terrain parameters

## Questions?

Check the main `CLAUDE.md` for general project info, or examine the code:
- Controller: `controllers/adaptive_gait_controller.py`
- Environment: `envs/adaptive_gait_env.py`
- Training: `train_adaptive_gait_ppo.py`
