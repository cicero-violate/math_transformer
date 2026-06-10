#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SCORER="${SCORER:-runs/checkpoints/topology_scorer.champion.pt}"
HAND_K="${HAND_K:-16}"
DEVICE="${DEVICE:-auto}"
BENCH_STEPS="${BENCH_STEPS:-100}"
BENCH_N="${BENCH_N:-1024}"
BENCH_NODE_MODE="${BENCH_NODE_MODE:-trees}"
ALLOW_FAIL="${ALLOW_FAIL:-1}"
LEARNED_K_VALUES="${LEARNED_K_VALUES:-6 4}"

for k in $LEARNED_K_VALUES; do
  echo "==> learned topology sweep K=$k vs hand K=$HAND_K"
  SCORER="$SCORER" \
  LEARNED_K="$k" \
  HAND_K="$HAND_K" \
  DEVICE="$DEVICE" \
  BENCH_STEPS="$BENCH_STEPS" \
  BENCH_N="$BENCH_N" \
  BENCH_NODE_MODE="$BENCH_NODE_MODE" \
  ALLOW_FAIL="$ALLOW_FAIL" \
  TMP_DIR="runs/benchmarks/learned_topology_k${k}_tmp" \
  scripts/benchmark_learned_topology.sh
  echo
 done
