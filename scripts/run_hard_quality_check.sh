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
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/synthetic_hard_dense.pt}"
QUALITY_K="${QUALITY_K:-4,8,16}"
DEVICE="${DEVICE:-auto}"

if [[ ! -f "$OUT_DIR/val.jsonl" ]]; then
  echo "Missing $OUT_DIR/val.jsonl; generating hard synthetic data first."
  scripts/generate_hard_synthetic.sh
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint $CHECKPOINT" >&2
  echo "Run scripts/train_hard_synthetic.sh first, or set CHECKPOINT=..." >&2
  exit 2
fi

"$PYTHON" -m src.eval \
  --quality \
  --examples "$OUT_DIR/val.jsonl" \
  --checkpoint "$CHECKPOINT" \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --middle-bridge-width 1 \
  --quality-k "$QUALITY_K" \
  --quality-device "$DEVICE"