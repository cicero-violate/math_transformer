#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-Simplify 2x + 12x.}"

.venv312/bin/python scripts/run_v26_graph_decoder_generate.py \
  --checkpoint runs/v26_graph_decoder/qwen_style_tiny_clean \
  --prompt "$PROMPT" \
  --temperature 0.3 \
  --compact
