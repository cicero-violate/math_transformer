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

OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-runs/checkpoints/scorer_runtime_j_best.pt}"
SUMMARY_CSV="${SUMMARY_CSV:-runs/checkpoints/scorer_runtime_j_selection.csv}"
LOG_DIR="${LOG_DIR:-runs/runtime_scorer_selection}"
DEVICE="${DEVICE:-auto}"

if [[ "$#" -eq 0 ]]; then
  set -- \
    runs/checkpoints/scorer_dense_mix_0p25_finetune.runtime_best.pt \
    runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e4.runtime_best.pt \
    runs/checkpoints/scorer_dense_mix_0p25_finetune_lr3e5.runtime_best.pt \
    runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e5.runtime_best.pt
fi

mkdir -p "$LOG_DIR" "$(dirname "$SUMMARY_CSV")" "$(dirname "$OUTPUT_CHECKPOINT")"
logs=()
mix_args=()

for scorer in "$@"; do
  if [[ ! -f "$scorer" ]]; then
    echo "missing scorer checkpoint: $scorer" >&2
    exit 1
  fi

  safe="$(printf '%s' "$scorer" | tr -c 'A-Za-z0-9_.-' '_')"
  log="$LOG_DIR/${safe}.quality.log"

  echo "evaluating $scorer"
  SCORER="$scorer" DEVICE="$DEVICE" scripts/run_learned_topology_quality.sh | tee "$log"

  logs+=("$log")
  mix_args+=(--mix "$scorer")
done

"$PYTHON" -m src.dense_mix_sweep "${logs[@]}" "${mix_args[@]}" --summary-csv "$SUMMARY_CSV"

best="$("$PYTHON" - "$SUMMARY_CSV" <<'PYSEL'
from __future__ import annotations

import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
with summary.open(newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"summary has no candidates: {summary}")
print(rows[0]["mix"])
PYSEL
)"

cp "$best" "$OUTPUT_CHECKPOINT"

echo "selected $best"
echo "copied to $OUTPUT_CHECKPOINT"
echo "summary $SUMMARY_CSV"
