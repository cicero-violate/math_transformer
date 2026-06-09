#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# v11 runtime-J-aligned fine-tune from the selected runtime-J checkpoint.
export MAX_STEPS="${MAX_STEPS:-1000}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-2000}"
export LR="${LR:-1e-5}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-512}"
export DEVICE="${DEVICE:-auto}"
export DENSE_CHECKPOINT="${DENSE_CHECKPOINT:-runs/checkpoints/synthetic_hard_dense.pt}"
export DENSE_MIX="${DENSE_MIX:-0.25}"
export RESUME_SCORER_CHECKPOINT="${RESUME_SCORER_CHECKPOINT:-runs/checkpoints/scorer_runtime_j_best.pt}"
export CHECKPOINT="${CHECKPOINT:-runs/checkpoints/scorer_runtime_aligned.pt}"
export BEST_CHECKPOINT="${BEST_CHECKPOINT:-runs/checkpoints/scorer_runtime_aligned.topology_best.pt}"
export RUNTIME_QUALITY_EXAMPLES="${RUNTIME_QUALITY_EXAMPLES:-data/synthetic_hard/val.jsonl}"
export RUNTIME_QUALITY_CHECKPOINT="${RUNTIME_QUALITY_CHECKPOINT:-runs/checkpoints/synthetic_hard_dense.pt}"
export RUNTIME_QUALITY_INTERVAL="${RUNTIME_QUALITY_INTERVAL:-100}"
export RUNTIME_QUALITY_MAX_EXAMPLES="${RUNTIME_QUALITY_MAX_EXAMPLES:-1024}"
export RUNTIME_QUALITY_BEST_CHECKPOINT="${RUNTIME_QUALITY_BEST_CHECKPOINT:-runs/checkpoints/scorer_runtime_aligned.runtime_best.pt}"
export RUNTIME_QUALITY_PATIENCE="${RUNTIME_QUALITY_PATIENCE:-2}"
export RUNTIME_QUALITY_MIN_DELTA="${RUNTIME_QUALITY_MIN_DELTA:-1e-5}"
export RUNTIME_QUALITY_STOP_ON_DEGRADE="${RUNTIME_QUALITY_STOP_ON_DEGRADE:-1}"
export RUNTIME_KL_LOSS="${RUNTIME_KL_LOSS:-0.50}"
export RUNTIME_COS_LOSS="${RUNTIME_COS_LOSS:-0.25}"
export RUNTIME_HIDDEN_L1_LOSS="${RUNTIME_HIDDEN_L1_LOSS:-0.10}"

echo "==> v11 runtime-aligned topology scorer training"
echo "    resume=$RESUME_SCORER_CHECKPOINT"
echo "    checkpoint=$CHECKPOINT"
echo "    runtime_best=$RUNTIME_QUALITY_BEST_CHECKPOINT"
echo "    lr=$LR runtime_patience=$RUNTIME_QUALITY_PATIENCE stop_on_degrade=$RUNTIME_QUALITY_STOP_ON_DEGRADE"
echo "    runtime_losses kl=$RUNTIME_KL_LOSS cos=$RUNTIME_COS_LOSS hidden_l1=$RUNTIME_HIDDEN_L1_LOSS"

exec scripts/train_topology_scorer.sh
