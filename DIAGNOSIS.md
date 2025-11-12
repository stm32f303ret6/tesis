# Training Diagnosis: prod_5m_20251111_173611

## Summary

The trained model **is not broken** - it learned exactly what the reward function told it to learn. However, the learned behavior is **not useful** for robust walking due to:

1. **Reward function misalignment**
2. **Train/eval gait parameter mismatch**
3. **Insufficient forward velocity incentive**

---

## Issue #1: Reward Function Prioritizes Wrong Behaviors

### Current Reward Components (residual_walk_env.py:261-324)

```
Component              Weight    Impact
--------------------------------------------
forward_velocity       1.0       Primary objective
lateral_stability      0.5       Drift penalty
height                 1.0       Height tracking
orientation            1.0       Stay upright
energy                 0.1       Penalize large actions
smoothness             0.2       Penalize action changes
contact_pattern        ±2.0      Gait timing
joint_limits           1.0       Safety
```

### Problem

The model discovered it can achieve **high reward without moving fast** by:
- Minimizing residuals → energy reward
- Keeping residuals constant → smoothness reward
- Maintaining contact patterns → contact reward
- Staying upright and at correct height → easy when static

**Result:** Model moves at only **0.056 m/s** (28% of target 0.2 m/s)

### Evidence

- Baseline (no residuals): 0.157 m/s, ~1.3 reward/step
- Trained model: 0.056 m/s, ~0.9 reward/step
- Training curves show "improvement" to +660 reward over 1000-step episodes
- But this reward comes from stability, not locomotion!

---

## Issue #2: Train/Eval Gait Mismatch

### Training Gait (train_residual_ppo.py:29)
```python
ResidualWalkEnv()  # Uses defaults:
  body_height = 0.07
  step_length = 0.05
  step_height = 0.015
  cycle_time = 0.8
```

### Evaluation Gait (phase4_viewer_play_policy.py:47)
```python
GaitParameters(
  body_height = 0.05,   # ← DIFFERENT!
  step_length = 0.06,   # ← DIFFERENT!
  step_height = 0.04,   # ← DIFFERENT!
  cycle_time = 0.8
)
```

**Impact:** The residuals are optimized for one foot trajectory pattern, then applied to a completely different trajectory. This is like training a steering correction for a bicycle and testing it on a motorcycle.

---

## Issue #3: Model Learns Conservative Strategy

The model learned:
- **Minimal residuals** (near-zero actions)
- **No disturbance** to baseline controller
- **Just enough motion** to avoid termination

This is a **local optimum** where the model avoids risk by barely moving, rather than learning useful residual corrections for robust terrain adaptation.

---

## Observed Behavior

### Direct Test (100 steps)
- ✓ Model runs without errors
- ✓ Maintains stable height (0.123 m)
- ✓ Positive forward motion (0.0075 m with training gait, 0.0374 m with eval gait)
- ✗ Very slow velocity (much slower than baseline)

### Your Playback Command
```bash
python3 phase4_viewer_play_policy.py \
  --model runs/prod_5m_20251111_173611/final_model.zip \
  --normalize runs/prod_5m_20251111_173611/vec_normalize.pkl \
  --seconds 20 --deterministic
```

**Result:** Robot walks but performs worse than baseline because:
1. Gait mismatch makes residuals counterproductive
2. Model never learned aggressive terrain adaptation
3. Conservative strategy fails on rough terrain

---

## Solutions

### Quick Fix: Test with Correct Gait
```bash
# Edit phase4_viewer_play_policy.py line 47 to match training:
gait = GaitParameters(
    body_height=0.07,    # Match training
    step_length=0.05,    # Match training
    step_height=0.015,   # Match training
    cycle_time=0.8
)
```

### Proper Fix: Retrain with Better Reward Function

**Recommended changes to residual_walk_env.py:**

```python
def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
    rewards: Dict[str, float] = {}

    # Make forward velocity MUCH more important
    linvel = self.sensor_reader.read_sensor("body_linvel")
    forward_vel = float(linvel[0])
    vel_error = abs(forward_vel - self.target_velocity)
    rewards["forward_velocity"] = 5.0 * (1.0 - vel_error)  # 1.0 → 5.0

    # Keep other penalties light
    rewards["lateral_stability"] = -0.3 * abs(float(linvel[1]))  # 0.5 → 0.3
    rewards["energy"] = -0.05 * float(np.linalg.norm(self.previous_action))  # 0.1 → 0.05

    # Remove smoothness penalty entirely for residual learning
    # (residuals SHOULD vary with terrain)
    # rewards["smoothness"] = 0.0

    # ... keep height, orientation, contact_pattern, joint_limits
```

### Additional Training Recommendations

1. **Match gaits everywhere:**
   - Fix `train_residual_ppo.py` to explicitly set gait params
   - Use same params in all eval scripts

2. **Increase training steps:**
   - Current: 5M timesteps
   - Recommended: 10-20M for complex terrain

3. **Use curriculum learning:**
   - Start on flat terrain
   - Gradually increase terrain difficulty
   - This prevents the "freeze and survive" strategy

4. **Add domain randomization:**
   - Randomize gait parameters during training
   - Makes policy robust to parameter changes

5. **Consider different observation space:**
   - Add terrain heightmap sensing
   - Add foot contact force magnitudes (not just binary)
   - Remove command velocity (it's constant anyway)

---

## Root Cause

**The command is correct. The code works. But the model learned the wrong thing.**

This is a classic **reward hacking** problem in RL: the agent found a way to get high reward (stay still and stable) without achieving the actual goal (robust terrain walking).

---

## ✅ FIXES IMPLEMENTED

All fixes have been applied. See `TRAINING_FIXES.md` for details.

### Quick Start
```bash
# Train with fixed parameters (10M steps, ~1.5 hours)
bash train_quick.sh

# Or use the detailed script
python3 train_residual_ppo_v2.py --total-timesteps 10000000 --n-envs 12

# Evaluate the result
python3 debug_model.py runs/prod_fixed_TIMESTAMP
```

### What Changed
1. ✅ Fixed gait parameters in `train_residual_ppo.py`
2. ✅ Improved reward function (5x velocity weight)
3. ✅ Created `train_residual_ppo_v2.py` with best practices
4. ✅ Updated all evaluation scripts for consistency

See `TRAINING_FIXES.md` for complete documentation.
