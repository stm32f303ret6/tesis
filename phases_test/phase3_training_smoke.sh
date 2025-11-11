#!/usr/bin/env bash
set -euo pipefail

# Phase 3 training smoke: 10k timesteps, single env.
export MUJOCO_GL=${MUJOCO_GL:-egl}

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Run training
python3 train_residual_ppo.py --total-timesteps 10000 --n-envs 1 --n-steps 512 --run-name smoke

# Find latest smoke run dir
RUN_DIR=$(ls -dt runs/smoke_* | head -n1)

# Write artifact list
ARTIFACTS_FILE=phases_test/phase3_artifacts.txt
{
  echo "$RUN_DIR/final_model.zip"
  echo "$RUN_DIR/vec_normalize.pkl"
} > "$ARTIFACTS_FILE"

# Metrics JSON
TB_DIR_EXISTS=false
EP_REW_MEAN_PRESENT=false

# TensorBoard event files (directly in run dir or within PPO_*/ subdir)
if ls "$RUN_DIR"/events.out.tfevents.* >/dev/null 2>&1; then
  TB_DIR_EXISTS=true
elif ls "$RUN_DIR"/PPO_*/events.out.tfevents.* >/dev/null 2>&1; then
  TB_DIR_EXISTS=true
fi

# Basic heuristic: if monitor files exist and are non-empty, assume ep_rew_mean present
if ls "$RUN_DIR"/monitor_*.csv >/dev/null 2>&1; then
  if [ -s "$(ls "$RUN_DIR"/monitor_*.csv | head -n1)" ]; then
    EP_REW_MEAN_PRESENT=true
  fi
fi

cat > phases_test/phase3_training_smoke_metrics.json <<JSON
{
  "timesteps": 10000,
  "tensorboard_logdir_exists": ${TB_DIR_EXISTS},
  "episode_reward_mean_present": ${EP_REW_MEAN_PRESENT}
}
JSON

echo "Phase 3 smoke complete. See $ARTIFACTS_FILE and phases_test/phase3_training_smoke_metrics.json"
