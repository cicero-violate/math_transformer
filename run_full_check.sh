#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer

echo "=== QUALITY CHECK ==="
./run_quality_check.sh

echo
echo "=== SPEED CHECK ==="
scripts/benchmark_attention.sh \
  --sizes 1024 \
  --node-mode trees \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1 \
  --profile-prepared-block \
  --warmup 3 \
  --iters 10
