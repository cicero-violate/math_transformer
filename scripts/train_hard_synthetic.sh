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

OUT_DIR="${OUT_DIR:-data/synthetic_hard}"
CONFIG="${CONFIG:-configs/synthetic_hard.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/synthetic_hard_dense.pt}"
LOSS_CSV="${LOSS_CSV:-runs/train_curves/synthetic_hard_dense.csv}"
MAX_STEPS="${MAX_STEPS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
DEVICE="${DEVICE:-auto}"

if [[ ! -f "$OUT_DIR/train.jsonl" ]]; then
  echo "Missing $OUT_DIR/train.jsonl; generating hard synthetic data first."
  scripts/generate_hard_synthetic.sh
fi

"$PYTHON" -m src.train \
  --config "$CONFIG" \
  --data "$OUT_DIR/train.jsonl" \
  --max-steps "$MAX_STEPS" \
  --eval-interval "$EVAL_INTERVAL" \
  --attention-mode full \
  --device "$DEVICE" \
  --save-checkpoint "$CHECKPOINT" \
  --save-loss-csv "$LOSS_CSV"