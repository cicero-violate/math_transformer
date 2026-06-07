#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.eval --benchmark --examples data/examples.jsonl "$@"
