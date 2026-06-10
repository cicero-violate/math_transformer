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

CANDIDATE="${CANDIDATE:-${SCORER:-}}"
CHAMPION_CHECKPOINT="${CHAMPION_CHECKPOINT:-runs/checkpoints/topology_scorer.champion.pt}"
CHAMPION_METADATA="${CHAMPION_METADATA:-runs/checkpoints/topology_scorer.champion.json}"
EXAMPLES="${EXAMPLES:-data/synthetic_hard/val.jsonl}"
DENSE_CHECKPOINT="${DENSE_CHECKPOINT:-runs/checkpoints/synthetic_hard_dense.pt}"
LEARNED_K="${LEARNED_K:-8}"
HAND_K="${HAND_K:-16}"
DEVICE="${DEVICE:-auto}"
BENCHMARK_LOG="${BENCHMARK_LOG:-}"
REQUIRE_BENCHMARK="${REQUIRE_BENCHMARK:-1}"
ROUTE_MIN_DELTA="${ROUTE_MIN_DELTA:-0.0}"
GENERIC_MIN_DELTA="${GENERIC_MIN_DELTA:-0.0}"
FORCE="${FORCE:-0}"

if [[ -z "$CANDIDATE" ]]; then
  echo "Set CANDIDATE=/path/to/scorer.pt or SCORER=/path/to/scorer.pt" >&2
  exit 2
fi

CMD=("$PYTHON" -m src.promote_topology_scorer
  --candidate "$CANDIDATE"
  --champion-checkpoint "$CHAMPION_CHECKPOINT"
  --champion-metadata "$CHAMPION_METADATA"
  --examples "$EXAMPLES"
  --dense-checkpoint "$DENSE_CHECKPOINT"
  --learned-k "$LEARNED_K"
  --hand-k "$HAND_K"
  --device "$DEVICE"
  --route-min-delta "$ROUTE_MIN_DELTA"
  --generic-min-delta "$GENERIC_MIN_DELTA")

if [[ -n "$BENCHMARK_LOG" ]]; then
  CMD+=(--benchmark-log "$BENCHMARK_LOG")
fi
if [[ "$REQUIRE_BENCHMARK" == "0" ]]; then
  CMD+=(--no-require-benchmark)
fi
if [[ "$FORCE" == "1" ]]; then
  CMD+=(--force)
fi

"${CMD[@]}"
