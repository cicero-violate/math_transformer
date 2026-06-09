cd /workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer

.venv-cuda/bin/python -m src.eval \
  --quality \
  --examples data/synthetic/val.jsonl \
  --checkpoint runs/checkpoints/v7_k16.pt \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --middle-bridge-width 1 \
  --quality-k 16 \
  --quality-device cuda

.venv-cuda/bin/python -m src.eval \
  --quality \
  --examples data/synthetic/val.jsonl \
  --checkpoint runs/checkpoints/v7_k16.pt \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --middle-bridge-width 1 \
  --quality-k 8,16,32,64 \
  --quality-device cuda
