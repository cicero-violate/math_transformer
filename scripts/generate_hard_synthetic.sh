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
TRAIN="${TRAIN:-10000}"
VAL="${VAL:-2000}"
TEST="${TEST:-2000}"
SEED="${SEED:-77}"
ROUTE_FRACTION="${ROUTE_FRACTION:-0.9}"
MAX_DEPTH="${MAX_DEPTH:-5}"

"$PYTHON" -m src.synthetic_data \
  --out-dir "$OUT_DIR" \
  --train "$TRAIN" \
  --val "$VAL" \
  --test "$TEST" \
  --seed "$SEED" \
  --route-fraction "$ROUTE_FRACTION" \
  --hard \
  --max-depth "$MAX_DEPTH"