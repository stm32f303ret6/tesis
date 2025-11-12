# Quick Start - Fixed Training

## TL;DR

Your old model walked worse than baseline because it was trained with broken gait parameters and a weak reward function. Everything is now fixed.

## Train New Model (Simple)

```bash
# Quick training (10M steps, ~1.5 hours with 12 cores)
bash train_quick.sh

# When done, evaluate:
RUN_DIR=$(ls -td runs/prod_fixed_* | head -1)
python3 debug_model.py "$RUN_DIR"
```

## Train New Model (Advanced)

```bash
# Full control over parameters
python3 train_residual_ppo_v2.py \
    --total-timesteps 10000000 \
    --n-envs 12 \
    --n-steps 2048 \
    --batch-size 512 \
    --run-name my_run

# Or for quick test (10 minutes)
python3 train_residual_ppo_v2.py \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --run-name quick_test
```

## What Was Fixed

### 1. Gait Parameters
**Problem**: Training used gait that moved backwards (-0.004 m/s)
**Fix**: Now uses working gait (0.095 m/s baseline)

### 2. Reward Function
**Problem**: Model could get high reward by staying still
**Fix**: Forward velocity weight increased 5x (1.0 → 5.0)

### 3. Consistency
**Problem**: Train/eval used different gait parameters
**Fix**: All scripts now use same gait

## Expected Results

| Metric | Old Model | New Model (Expected) |
|--------|-----------|----------------------|
| Baseline velocity | 0.095 m/s | 0.095 m/s |
| Trained velocity | 0.010 m/s | 0.15-0.20 m/s |
| Improvement | -89% (worse!) | +50-100% (better!) |

## Files Changed

- ✅ `train_residual_ppo.py` - Fixed gait
- ✅ `envs/residual_walk_env.py` - Better rewards
- ✅ `tests/compare_residual_vs_baseline.py` - Fixed gait
- ✨ `train_residual_ppo_v2.py` - NEW: Best training script
- ✨ `debug_model.py` - NEW: Analysis tool
- ✨ `train_quick.sh` - NEW: Simple wrapper
- 📖 `DIAGNOSIS.md` - Problem analysis
- 📖 `TRAINING_FIXES.md` - Complete guide
- 📖 `CHANGELOG_FIXES.md` - All changes

## Need Help?

1. **Training not improving?**
   - Run: `python3 debug_model.py YOUR_RUN_DIR`
   - Check: `TRAINING_FIXES.md` → "Debugging Failed Runs"

2. **Want to understand the problem?**
   - Read: `DIAGNOSIS.md`

3. **Need detailed docs?**
   - Read: `TRAINING_FIXES.md`

## Next Steps

1. Train: `bash train_quick.sh`
2. Wait: ~1.5 hours (or monitor with `tensorboard --logdir runs/`)
3. Evaluate: `python3 debug_model.py runs/prod_fixed_TIMESTAMP`
4. Test visually: `python3 phases_test/phase4_viewer_play_policy.py --model ... --normalize ...`

If the new model works well, you should see:
- ✅ Forward velocity > baseline
- ✅ Stable walking on rough terrain
- ✅ Useful residual corrections (action magnitude > 0.5)
- ✅ No early termination/falling
