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

TMP_DIR="${TMP_DIR:-runs/benchmarks/champion_regression_tmp}"
ARTIFACT_JSON="${ARTIFACT_JSON:-$TMP_DIR/champion_regression_artifact.json}"
ARTIFACT_JSONL="${ARTIFACT_JSONL:-runs/benchmarks/champion_regression_artifacts.jsonl}"
CHAMPION_CHECKPOINT="${CHAMPION_CHECKPOINT:-runs/checkpoints/topology_scorer.champion.pt}"
CHAMPION_METADATA="${CHAMPION_METADATA:-runs/checkpoints/topology_scorer.champion.json}"
mkdir -p "$TMP_DIR"

SCORER="$CHAMPION_CHECKPOINT" \
LEARNED_K="${LEARNED_K:-8}" \
HAND_K="${HAND_K:-16}" \
TMP_DIR="$TMP_DIR" \
ARTIFACT_JSON="$ARTIFACT_JSON" \
ARTIFACT_JSONL="$ARTIFACT_JSONL" \
scripts/benchmark_learned_topology.sh

"$PYTHON" -m src.check_topology_champion_regression \
  --artifact "$ARTIFACT_JSON" \
  --champion-metadata "$CHAMPION_METADATA" \
  --route-min-delta "${ROUTE_MIN_DELTA:-0.0}" \
  --generic-min-delta "${GENERIC_MIN_DELTA:-0.0}"
