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
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/topology_scorer.champion.pt}"
DEVICE="${DEVICE:-auto}"
EVAL_K="${EVAL_K:-8}"
TARGET_K="${TARGET_K:-16}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
TRACE_OUTPUT="${TRACE_OUTPUT:-}"

CMD=("$PYTHON" -m src.eval_topology_scorer
  --examples "$EXAMPLES"
  --checkpoint "$CHECKPOINT"
  --device "$DEVICE"
  --eval-k "$EVAL_K"
  --target-k "$TARGET_K"
  --max-examples "$MAX_EXAMPLES")

if [[ -n "$TRACE_OUTPUT" ]]; then
  CMD+=(--trace-output "$TRACE_OUTPUT")
fi

"${CMD[@]}"
