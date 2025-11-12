# Training Fixes - November 2025

## What Was Fixed

### Problem Summary
The trained model (`prod_5m_20251111_173611`) walked worse than the baseline controller due to:

1. **Gait parameter mismatch**: Training used broken default gait (baseline moved backwards at -0.0044 m/s)
2. **Weak reward signal**: Forward velocity weight too low (1.0x), allowing model to "cheat" by staying still
3. **Excessive penalties**: Energy and smoothness penalties discouraged useful residual corrections

### Changes Made

#### 1. Fixed `train_residual_ppo.py` (line 28-47)
- **Added explicit gait parameters** matching the working baseline:
  ```python
  gait = GaitParameters(
      body_height=0.05,
      step_length=0.06,
      step_height=0.04,
      cycle_time=0.8
  )
  ```
- These parameters give baseline ~0.09 m/s forward velocity (previously: -0.004 m/s)

#### 2. Improved `envs/residual_walk_env.py` reward function (line 261-297)
- **Forward velocity weight**: 1.0 → **5.0** (now dominates the reward)
- **Lateral stability**: 0.5 → **0.3** (reduced penalty)
- **Energy penalty**: 0.1 → **0.05** (allow larger residuals)
- **Smoothness penalty**: 0.2 → **0.05** (residuals should vary with terrain)

**Expected impact**: Model should now learn aggressive terrain adaptation instead of conservative "stay still" strategy.

#### 3. Updated evaluation scripts
- Fixed `tests/compare_residual_vs_baseline.py` to use same gait as training
- `phase4_viewer_play_policy.py` and `play_residual_policy.py` already had correct gait

#### 4. Created `train_residual_ppo_v2.py`
New training script with best practices:
- Fixed gait parameters
- Better hyperparameters (larger n_steps=2048)
- Checkpoint saving every 100k steps
- Comprehensive logging and config saving
- Support for 12+ parallel environments

---

## How to Use

### Quick Test (10 minutes, 1M steps)
```bash
python3 train_residual_ppo_v2.py \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --run-name quick_test
```

### Production Training (Recommended: 10M steps)
```bash
python3 train_residual_ppo_v2.py \
    --total-timesteps 10000000 \
    --n-envs 12 \
    --run-name prod_fixed
```

### Long Training (20M steps for best results)
```bash
python3 train_residual_ppo_v2.py \
    --total-timesteps 20000000 \
    --n-envs 16 \
    --n-steps 2048 \
    --batch-size 512 \
    --run-name prod_long
```

### Monitor Training Progress
```bash
# Watch tensorboard logs
tensorboard --logdir runs/

# Or check raw monitor files
tail -f runs/prod_fixed_TIMESTAMP/monitor_0.csv.monitor.csv
```

### Evaluate Trained Model
```bash
# After training completes, use the printed command:
python3 phases_test/phase4_viewer_play_policy.py \
    --model runs/prod_fixed_TIMESTAMP/final_model.zip \
    --normalize runs/prod_fixed_TIMESTAMP/vec_normalize.pkl \
    --seconds 20 --deterministic

# Or debug with the analysis script:
python3 debug_model.py runs/prod_fixed_TIMESTAMP
```

---

## Expected Results

### Before Fixes (old model)
```
Baseline:     0.095 m/s forward
Trained:      0.010 m/s forward  ← WORSE!
Improvement:  -89% (model learned to stay still)
```

### After Fixes (expected)
```
Baseline:     0.095 m/s forward
Trained:      0.15-0.20 m/s forward  ← BETTER!
Improvement:  +50-100% (useful residual corrections)
```

The model should now:
- ✓ Move faster than baseline
- ✓ Maintain stable height and orientation
- ✓ Adapt foot placements to rough terrain
- ✓ Use residuals when needed (not stay at zero)

---

## Training Tips

### Compute Requirements
- **Minimum**: 4 CPU cores, 8GB RAM (for `--n-envs 4`)
- **Recommended**: 12 CPU cores, 16GB RAM (for `--n-envs 12`)
- **GPU**: Not required (MuJoCo CPU-bound, small MLP policy)

### Training Time Estimates
- 1M steps, 4 envs: ~10 minutes
- 10M steps, 12 envs: ~1.5 hours
- 20M steps, 16 envs: ~2.5 hours

### Hyperparameter Notes

**If training is unstable:**
- Reduce `--learning-rate` to `1e-4`
- Increase `--n-steps` to `4096`
- Reduce `--n-envs` to avoid outlier environments

**If learning is too slow:**
- Increase `--learning-rate` to `5e-4`
- Reduce `--n-steps` to `1024` (more frequent updates)
- Increase `--n-envs` for more diverse experience

**For better exploration:**
- Add `--ent-coef 0.01` (entropy bonus)
- Use `--randomize` flag (domain randomization, future feature)

---

## Debugging Failed Runs

### Model still walks worse than baseline?

1. **Check training progress:**
   ```bash
   python3 << 'EOF'
   import pandas as pd
   df = pd.read_csv("runs/YOUR_RUN/monitor_0.csv.monitor.csv", skiprows=1)
   print(f"First 10 episodes: {df['r'].head(10).mean():.2f}")
   print(f"Last 100 episodes: {df['r'].tail(100).mean():.2f}")
   print(f"Best episode: {df['r'].max():.2f}")
   EOF
   ```

   **Expected**: Last 100 should be >> First 10 (e.g., +2.0 or more)

2. **Run debug script:**
   ```bash
   python3 debug_model.py runs/YOUR_RUN
   ```

   **Look for**:
   - Mean action magnitude > 0.5 (model is using residuals)
   - Trained velocity > Baseline velocity
   - No early termination

3. **Check gait parameters:**
   ```bash
   grep -A5 "GaitParameters" train_residual_ppo_v2.py
   ```

   **Should see**:
   ```python
   gait = GaitParameters(
       body_height=0.05,  # NOT 0.07
       step_length=0.06,  # NOT 0.05
       step_height=0.04,  # NOT 0.015
   ```

### Model learns but then collapses?
- **Symptom**: Reward increases, then suddenly drops
- **Cause**: Exploration found an exploit or instability
- **Fix**: Reduce learning rate, increase PPO clip range to 0.3

### Model never improves?
- **Symptom**: Reward stays flat for 1M+ steps
- **Cause**: Baseline is too good, residuals can't help
- **Fix**: Try harder terrain, or increase residual_scale to 0.04

---

## Files Modified

1. `train_residual_ppo.py` - Fixed gait parameters
2. `envs/residual_walk_env.py` - Improved reward function
3. `tests/compare_residual_vs_baseline.py` - Fixed gait parameters
4. `train_residual_ppo_v2.py` - **NEW**: Production training script
5. `debug_model.py` - **NEW**: Analysis script
6. `DIAGNOSIS.md` - **NEW**: Detailed problem analysis
7. `TRAINING_FIXES.md` - **NEW**: This file

---

## Next Steps

1. **Run new training:**
   ```bash
   python3 train_residual_ppo_v2.py --total-timesteps 10000000 --n-envs 12 --run-name fixed_v1
   ```

2. **Monitor progress** (optional):
   ```bash
   tensorboard --logdir runs/
   ```

3. **Evaluate result:**
   ```bash
   python3 debug_model.py runs/fixed_v1_TIMESTAMP
   ```

4. **If successful**, try harder challenges:
   - Increase terrain difficulty (edit `hfield.png`)
   - Add velocity variations during training
   - Test on unseen terrain

5. **If still not working**, see DIAGNOSIS.md for deeper analysis

---

## Questions?

Check these files for more details:
- `DIAGNOSIS.md` - Root cause analysis
- `debug_model.py` - Performance testing script
- `train_residual_ppo_v2.py` - Training script with comments
- `envs/residual_walk_env.py` - Environment and reward logic
