#!/usr/bin/env bash
# Quick training script with fixed parameters
# Usage: bash train_quick.sh [total_timesteps] [n_envs] [run_name]

set -euo pipefail

TIMESTEPS="${1:-10000000}"  # Default: 10M
N_ENVS="${2:-64}"           # Default: 12
RUN_NAME="${3:-prod_fixed}" # Default: prod_fixed

echo "========================================================================"
echo "Training Residual PPO (Fixed Version)"
echo "========================================================================"
echo "Total timesteps: $TIMESTEPS"
echo "Parallel envs:   $N_ENVS"
echo "Run name:        $RUN_NAME"
echo "========================================================================"
echo ""

python3 train_residual_ppo_v3.py \
    --total-timesteps "$TIMESTEPS" \
    --n-envs "$N_ENVS" \
    --run-name "$RUN_NAME" \
    --checkpoint-freq 100000

echo ""
echo "========================================================================"
echo "Training complete!"
echo "========================================================================"
echo ""
echo "To evaluate, find your run directory with:"
echo "  ls -ltr runs/ | tail -1"
echo ""
echo "Then run:"
echo "  python3 debug_model.py runs/YOUR_RUN_DIR"
