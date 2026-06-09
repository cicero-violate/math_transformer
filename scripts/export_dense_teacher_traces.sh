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
SPLIT="${SPLIT:-train}"
OUT="${OUT:-runs/teacher_traces/synthetic_hard_${SPLIT}.jsonl}"
DEVICE="${DEVICE:-auto}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
EXAMPLES="$OUT_DIR/${SPLIT}.jsonl"

if [[ ! -f "$EXAMPLES" ]]; then
  echo "Missing $EXAMPLES; generating hard synthetic data first."
  OUT_DIR="$OUT_DIR" scripts/generate_hard_synthetic.sh
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint $CHECKPOINT" >&2
  echo "Run scripts/train_hard_synthetic.sh first, or set CHECKPOINT=..." >&2
  exit 2
fi

"$PYTHON" -m src.teacher_traces \
  --examples "$EXAMPLES" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT" \
  --device "$DEVICE" \
  --max-examples "$MAX_EXAMPLES"