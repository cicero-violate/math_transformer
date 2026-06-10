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

OUTPUT="${OUTPUT:-runs/replay/topology_replay_candidates.jsonl}"
MAX_RECORDS="${MAX_RECORDS:-100}"
MIN_SCORE="${MIN_SCORE:-0}"
SUMMARY="${SUMMARY:-}"
INCLUDE_SUCCESSES="${INCLUDE_SUCCESSES:-0}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 TRACE.jsonl [TRACE2.jsonl ...]" >&2
  exit 2
fi

CMD=("$PYTHON" -m src.topology_trace_replay
  --output "$OUTPUT"
  --max-records "$MAX_RECORDS"
  --min-score "$MIN_SCORE")

if [[ -n "$SUMMARY" ]]; then
  CMD+=(--summary "$SUMMARY")
fi
if [[ "$INCLUDE_SUCCESSES" == "1" ]]; then
  CMD+=(--include-successes)
fi
CMD+=("$@")

"${CMD[@]}"
