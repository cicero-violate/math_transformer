#!/usr/bin/env bash
set -euo pipefail

.venv312/bin/python scripts/run_v26_graph_decoder_train.py \
  --adjacency-path runs/qwen_distillation_harness/qwen_style_tiny/v25_measured_cli/graph_prior_eval/selected_adjacencies/qwen_topk_k2.json \
  --teacher-artifacts runs/sparse_student_distill/qwen25-3b-smart/sparse_student/v25_01_qwen25_3b_smart_sparse_smoke \
  --output-dir runs/v26_graph_decoder/qwen_style_tiny \
  --block-size 128 \
  --hidden-dim 128 \
  --n-layers 4 \
  --n-heads 4 \
  --epochs 100 \
  --device cuda
