#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x ".venv-cuda/bin/python" ]]; then
  PYTHON=".venv-cuda/bin/python"
elif [[ -x "../../.venv-torch/bin/python" ]]; then
  PYTHON="../../.venv-torch/bin/python"
else
  PYTHON="python"
fi

has_cuda() {
  "$PYTHON" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

echo "python=$($PYTHON - <<'PY'
import sys
print(sys.executable)
PY
)"
"$PYTHON" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
print(f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
PY

echo "== pytest: full regression suite =="
"$PYTHON" -m pytest -q

echo "== pytest: v7 focused regression suite =="
"$PYTHON" -m pytest -q \
  tests/test_middle_preserving_topology.py \
  tests/test_scored_topk.py \
  tests/test_sparse_attention.py \
  tests/test_topology_cache.py \
  tests/test_train_paths.py

if has_cuda; then
  echo "== CUDA benchmark: v7 fixed-K smoke =="
  scripts/benchmark_attention.sh \
    --sizes "${V7_SIZES:-1024}" \
    --node-mode "${V7_NODE_MODE:-trees}" \
    --topology-mode middle_preserving_topk \
    --fixed-k "${V7_FIXED_K:-16}" \
    --max-neighbors "${V7_MAX_NEIGHBORS:-16}" \
    --middle-bridge-width "${V7_MIDDLE_BRIDGE_WIDTH:-1}" \
    --warmup "${V7_WARMUP:-1}" \
    --iters "${V7_ITERS:-3}"

  if [[ "${FULL_BENCH:-0}" == "1" ]]; then
    echo "== CUDA benchmark: legacy K sweep =="
    for K in 16 32 64 128; do
      scripts/benchmark_attention.sh \
        --sizes 1024 \
        --node-mode roots,trees \
        --max-neighbors "$K"
    done
  fi
else
  echo "== CUDA benchmark skipped: torch.cuda.is_available() is false for $PYTHON =="
fi
