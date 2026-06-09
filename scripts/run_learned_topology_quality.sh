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
EXAMPLES="${EXAMPLES:-$OUT_DIR/val.jsonl}"
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/synthetic_hard_dense.pt}"
SCORER="${SCORER:-runs/checkpoints/learned_topology_scorer.best.pt}"
LEARNED_K="${LEARNED_K:-8}"
QUALITY_K="${QUALITY_K:-16}"
DEVICE="${DEVICE:-auto}"

"$PYTHON" -m src.eval \
  --quality \
  --examples "$EXAMPLES" \
  --checkpoint "$CHECKPOINT" \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --middle-bridge-width 1 \
  --quality-k "$QUALITY_K" \
  --quality-device "$DEVICE" \
  --learned-scorer-checkpoint "$SCORER" \
  --learned-k "$LEARNED_K"
