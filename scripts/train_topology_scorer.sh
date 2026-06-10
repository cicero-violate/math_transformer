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
EXAMPLES="${EXAMPLES:-$OUT_DIR/train.jsonl}"
VAL_EXAMPLES="${VAL_EXAMPLES:-$OUT_DIR/val.jsonl}"
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/learned_topology_scorer.pt}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:-runs/checkpoints/learned_topology_scorer.best.pt}"
BEST_SELECTION="${BEST_SELECTION:-edge_recall}"
MAX_STEPS="${MAX_STEPS:-1000}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-auto}"
TARGET_K="${TARGET_K:-16}"
EVAL_K="${EVAL_K:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-512}"
DENSE_CHECKPOINT="${DENSE_CHECKPOINT:-}"
DENSE_MIX="${DENSE_MIX:-0.0}"
RESUME_SCORER_CHECKPOINT="${RESUME_SCORER_CHECKPOINT:-}"
RUNTIME_QUALITY_EXAMPLES="${RUNTIME_QUALITY_EXAMPLES:-}"
RUNTIME_QUALITY_CHECKPOINT="${RUNTIME_QUALITY_CHECKPOINT:-}"
RUNTIME_QUALITY_INTERVAL="${RUNTIME_QUALITY_INTERVAL:-0}"
RUNTIME_QUALITY_MAX_EXAMPLES="${RUNTIME_QUALITY_MAX_EXAMPLES:-0}"
RUNTIME_QUALITY_BEST_CHECKPOINT="${RUNTIME_QUALITY_BEST_CHECKPOINT:-}"
RUNTIME_QUALITY_PATIENCE="${RUNTIME_QUALITY_PATIENCE:-0}"
RUNTIME_QUALITY_MIN_DELTA="${RUNTIME_QUALITY_MIN_DELTA:-1e-5}"
RUNTIME_QUALITY_STOP_ON_DEGRADE="${RUNTIME_QUALITY_STOP_ON_DEGRADE:-0}"
RUNTIME_KL_LOSS="${RUNTIME_KL_LOSS:-0.0}"
RUNTIME_COS_LOSS="${RUNTIME_COS_LOSS:-0.0}"
RUNTIME_HIDDEN_L1_LOSS="${RUNTIME_HIDDEN_L1_LOSS:-0.0}"
REPLAY_CANDIDATES="${REPLAY_CANDIDATES:-}"
REPLAY_WEIGHT_SCALE="${REPLAY_WEIGHT_SCALE:-0.1}"
REPLAY_MAX_WEIGHT="${REPLAY_MAX_WEIGHT:-8.0}"
REPLAY_SAMPLE_RATIO="${REPLAY_SAMPLE_RATIO:-0.0}"

EXTRA_RUNTIME_STOP_FLAG=()
if [[ "$RUNTIME_QUALITY_STOP_ON_DEGRADE" == "1" ]]; then
  EXTRA_RUNTIME_STOP_FLAG=(--runtime-quality-stop-on-degrade)
fi

if [[ ! -f "$EXAMPLES" ]]; then
  echo "Missing $EXAMPLES; generating hard synthetic data first."
  OUT_DIR="$OUT_DIR" scripts/generate_hard_synthetic.sh
fi

"$PYTHON" -m src.train_topology_scorer \
  --examples "$EXAMPLES" \
  --save-checkpoint "$CHECKPOINT" \
  --max-steps "$MAX_STEPS" \
  --max-examples "$MAX_EXAMPLES" \
  --lr "$LR" \
  --device "$DEVICE" \
  --target-k "$TARGET_K" \
  --eval-k "$EVAL_K" \
  --val-examples "$VAL_EXAMPLES" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-max-examples "$EVAL_MAX_EXAMPLES" \
  --best-checkpoint "$BEST_CHECKPOINT" \
  --best-selection "$BEST_SELECTION" \
  --dense-checkpoint "$DENSE_CHECKPOINT" \
  --dense-mix "$DENSE_MIX" \
  --resume-scorer-checkpoint "$RESUME_SCORER_CHECKPOINT" \
  --runtime-quality-examples "$RUNTIME_QUALITY_EXAMPLES" \
  --runtime-quality-checkpoint "$RUNTIME_QUALITY_CHECKPOINT" \
  --runtime-quality-interval "$RUNTIME_QUALITY_INTERVAL" \
  --runtime-quality-max-examples "$RUNTIME_QUALITY_MAX_EXAMPLES" \
  --runtime-quality-best-checkpoint "$RUNTIME_QUALITY_BEST_CHECKPOINT" \
  --runtime-quality-patience "$RUNTIME_QUALITY_PATIENCE" \
  --runtime-quality-min-delta "$RUNTIME_QUALITY_MIN_DELTA" \
  --runtime-kl-loss "$RUNTIME_KL_LOSS" \
  --runtime-cos-loss "$RUNTIME_COS_LOSS" \
  --runtime-hidden-l1-loss "$RUNTIME_HIDDEN_L1_LOSS" \
  --replay-candidates "$REPLAY_CANDIDATES" \
  --replay-weight-scale "$REPLAY_WEIGHT_SCALE" \
  --replay-max-weight "$REPLAY_MAX_WEIGHT" \
  --replay-sample-ratio "$REPLAY_SAMPLE_RATIO" \
  "${EXTRA_RUNTIME_STOP_FLAG[@]}"
