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

PROTOCOL_CONFIG="${PROTOCOL_CONFIG:-configs/learned_topology_locked_protocol.json}"
# shellcheck disable=SC1090
eval "$($PYTHON -m src.topology_protocol --config "$PROTOCOL_CONFIG" --emit-shell-env)"

mkdir -p "$TMP_DIR" "$(dirname "$TRACE_OUTPUT")" "$(dirname "$ARTIFACT_JSON")" "$(dirname "$ARTIFACT_JSONL")"

echo "locked_protocol_name=$LOCKED_PROTOCOL_NAME"
echo "locked_protocol_config=$LOCKED_PROTOCOL_CONFIG"
echo "locked_protocol_hash=$LOCKED_PROTOCOL_HASH"
echo "locked_protocol_scorer=$SCORER"
echo "locked_protocol_examples=$EXAMPLES"
echo "locked_protocol_checkpoint=$CHECKPOINT"
echo "locked_protocol_benchmark n=$BENCH_N steps=$BENCH_STEPS seed=$BENCH_SEED node_mode=$BENCH_NODE_MODE hand_k=$HAND_K learned_k=$LEARNED_K"

scripts/benchmark_learned_topology.sh

"$PYTHON" -m src.check_topology_champion_regression \
  --artifact "$ARTIFACT_JSON" \
  --champion-metadata "$CHAMPION_METADATA" \
  --route-min-delta "$ROUTE_MIN_DELTA" \
  --generic-min-delta "$GENERIC_MIN_DELTA"
