#!/usr/bin/env bash
set -euo pipefail

# Phase 4 — Evaluation & Visualization
#
# Usage:
#   bash phases_test/phase4_playback.sh [RUN_DIR]
#
# RUN_DIR should contain final_model.zip and vec_normalize.pkl. If omitted, the
# script tries a few patterns and falls back to the most recent directory under runs/.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  # Prefer the path styles used in this repo's smoke runs
  if compgen -G "runs/smoke4_*" > /dev/null; then
    RUN_DIR=$(ls -1dt runs/smoke4_* | head -n1)
  elif compgen -G "runs/smoke_*" > /dev/null; then
    RUN_DIR=$(ls -1dt runs/smoke_* | head -n1)
  else
    # Fallback to the newest under runs/
    RUN_DIR=$(ls -1dt runs/* | head -n1)
  fi
fi

MODEL_ZIP="$RUN_DIR/final_model.zip"
NORMALIZE_PKL="$RUN_DIR/vec_normalize.pkl"

if [[ ! -f "$MODEL_ZIP" ]]; then
  echo "Model not found: $MODEL_ZIP" >&2
  exit 1
fi
if [[ ! -f "$NORMALIZE_PKL" ]]; then
  echo "VecNormalize file not found: $NORMALIZE_PKL" >&2
  exit 1
fi

OUT_TXT="phases_test/phase4_playback.txt"
OUT_BASE="phases_test/phase4_playback_baseline.txt"
OUT_RES="phases_test/phase4_playback_residual.txt"
mkdir -p "phases_test"
rm -f "$OUT_TXT" "$OUT_BASE" "$OUT_RES"

echo "Running baseline (zero residuals) episodes..."
python3 play_residual_policy.py \
  --episodes 2 \
  --baseline \
  --normalize "$NORMALIZE_PKL" \
  --summary-out "$OUT_BASE"

echo "Running residual policy episodes..."
python3 play_residual_policy.py \
  --model "$MODEL_ZIP" \
  --normalize "$NORMALIZE_PKL" \
  --episodes 2 \
  --deterministic \
  --summary-out "$OUT_RES"

# Merge into a single summary file
cat "$OUT_BASE" "$OUT_RES" > "$OUT_TXT"

echo "Generating comparison plot..."
python3 tests/compare_residual_vs_baseline.py \
  --model "$MODEL_ZIP" \
  --normalize "$NORMALIZE_PKL" \
  --episodes 3 \
  --out phases_test/residual_vs_baseline.png

echo "Phase 4 complete. Artifacts:"
echo "- $OUT_TXT"
echo "- phases_test/residual_vs_baseline.png"
