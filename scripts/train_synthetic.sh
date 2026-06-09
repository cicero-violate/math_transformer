#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x ".venv-cuda/bin/python" ]]; then
  PYTHON=".venv-cuda/bin/python"
elif [[ -x "../../.venv-torch/bin/python" ]]; then
  PYTHON="../../.venv-torch/bin/python"
else
  PYTHON="python"
fi

OUT_DIR="${OUT_DIR:-data/synthetic}"
TRAIN="${TRAIN:-10000}"
VAL="${VAL:-1000}"
TEST="${TEST:-1000}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/tiny.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/synthetic_tiny.pt}"
QUALITY_K="${QUALITY_K:-16,32,64,128}"
ROUTE_FRACTION="${ROUTE_FRACTION:-0.6}"
MAX_STEPS="${MAX_STEPS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"

echo "==> generating synthetic data"
"$PYTHON" -m src.synthetic_data \
  --out-dir "$OUT_DIR" \
  --train "$TRAIN" \
  --val "$VAL" \
  --test "$TEST" \
  --seed "$SEED" \
  --route-fraction "$ROUTE_FRACTION"

echo "==> training"
"$PYTHON" -m src.train \
  --config "$CONFIG" \
  --data "$OUT_DIR/train.jsonl" \
  --max-steps "$MAX_STEPS" \
  --eval-interval "$EVAL_INTERVAL" \
  --save-checkpoint "$CHECKPOINT"

echo "==> quality eval"
"$PYTHON" -m src.eval \
  --quality \
  --quality-k "$QUALITY_K" \
  --examples "$OUT_DIR/val.jsonl" \
  --checkpoint "$CHECKPOINT"
