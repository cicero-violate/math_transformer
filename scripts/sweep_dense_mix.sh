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

MIXES="${MIXES:-0.05 0.10 0.15 0.20 0.25 0.35 0.50}"
SWEEP_DIR="${SWEEP_DIR:-runs/dense_mix_sweep}"
MAX_STEPS="${MAX_STEPS:-5000}"
MAX_EXAMPLES="${MAX_EXAMPLES:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-512}"
DEVICE="${DEVICE:-auto}"
DENSE_CHECKPOINT="${DENSE_CHECKPOINT:-runs/checkpoints/synthetic_hard_dense.pt}"
QUALITY_K="${QUALITY_K:-16}"
LEARNED_K="${LEARNED_K:-8}"

mkdir -p "$SWEEP_DIR/checkpoints" "$SWEEP_DIR/logs"
logs_file="$SWEEP_DIR/logs.list"
mix_file="$SWEEP_DIR/mixes.list"
: > "$logs_file"
: > "$mix_file"

for mix in $MIXES; do
  safe_mix="${mix//./p}"
  ckpt="$SWEEP_DIR/checkpoints/scorer_mix_${safe_mix}.pt"
  best="$SWEEP_DIR/checkpoints/scorer_mix_${safe_mix}.best.pt"
  train_log="$SWEEP_DIR/logs/train_mix_${safe_mix}.log"
  quality_log="$SWEEP_DIR/logs/quality_mix_${safe_mix}.log"

  echo "=== train dense_mix=$mix ==="
  MAX_STEPS="$MAX_STEPS" MAX_EXAMPLES="$MAX_EXAMPLES" \
  EVAL_INTERVAL="$EVAL_INTERVAL" EVAL_MAX_EXAMPLES="$EVAL_MAX_EXAMPLES" \
  DEVICE="$DEVICE" DENSE_CHECKPOINT="$DENSE_CHECKPOINT" DENSE_MIX="$mix" \
  CHECKPOINT="$ckpt" BEST_CHECKPOINT="$best" \
  scripts/train_topology_scorer.sh 2>&1 | tee "$train_log"

  echo "=== quality dense_mix=$mix ==="
  SCORER="$best" QUALITY_K="$QUALITY_K" LEARNED_K="$LEARNED_K" DEVICE="$DEVICE" \
  scripts/run_learned_topology_quality.sh 2>&1 | tee "$quality_log"

  echo "$quality_log" >> "$logs_file"
  echo "$mix" >> "$mix_file"
done

args=()
while IFS= read -r mix; do args+=(--mix "$mix"); done < "$mix_file"
"$PYTHON" -m src.dense_mix_sweep $(cat "$logs_file") "${args[@]}" \
  --summary-csv "$SWEEP_DIR/summary.csv" | tee "$SWEEP_DIR/selection.txt"

best_mix="$(head -n 1 "$SWEEP_DIR/selection.txt" | sed -E 's/^mix=([^ ]+).*/\1/')"
if [[ -n "$best_mix" ]]; then
  safe_best="${best_mix//./p}"
  best_ckpt="$SWEEP_DIR/checkpoints/scorer_mix_${safe_best}.best.pt"
  if [[ -f "$best_ckpt" ]]; then
    cp "$best_ckpt" "$SWEEP_DIR/checkpoints/scorer_best_by_quality.pt"
    echo "$best_ckpt" > "$SWEEP_DIR/best_checkpoint.txt"
    echo "best_mix=$best_mix best_checkpoint=$best_ckpt"
  fi
fi
