#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/neurosymbolic/encoder}"
RAW="${RAW:-$OUT_DIR/raw.jsonl}"
EXAMPLES="${EXAMPLES:-$OUT_DIR/examples.jsonl}"
VOCAB="${VOCAB:-$OUT_DIR/vocab.json}"
WEIGHTS="${WEIGHTS:-$OUT_DIR/weights.safetensors}"
LOSS_TSV="${LOSS_TSV:-$OUT_DIR/self_play_loss.tsv}"
VERIFY_CACHE="${VERIFY_CACHE:-$OUT_DIR/value_cache.jsonl}"
CRATES="${CRATES:-neurosymbolic/search neurosymbolic/encoder neurosymbolic/structural-editor algorithms ai score}"
ITERATIONS="${ITERATIONS:-5}"
SIMULATIONS="${SIMULATIONS:-50}"
MAX_CANDIDATES="${MAX_CANDIDATES:-64}"
VALUE="${VALUE:-depth}"
MAX_LEN="${MAX_LEN:-512}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-5}"
DEVICE="${DEVICE:-auto}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

ENCODER_FEATURES=()
if [[ "${DEVICE:-auto}" == "cuda" ]] || { [[ "${DEVICE:-auto}" == "auto" ]] && command -v nvidia-smi &>/dev/null; }; then
  ENCODER_FEATURES=(--features cuda)
  # CUDA 11.8 does not parse Arch's current GCC/libstdc++ 15 headers.
  # Pin both nvcc host-compiler knobs so Candle's CUDA kernel build uses GCC 11.
  CUDA_HOST_COMPILER="${CUDA_HOST_COMPILER:-/usr/bin/g++-11}"
  if [[ ! -x "$CUDA_HOST_COMPILER" ]]; then
    printf 'error: CUDA build requested but CUDA_HOST_COMPILER is not executable: %s\n' "$CUDA_HOST_COMPILER" >&2
    printf '       install a CUDA 11.8-compatible g++ or set CUDA_HOST_COMPILER/NVCC_CCBIN/CUDAHOSTCXX.\n' >&2
    exit 1
  fi
  export NVCC_CCBIN="$CUDA_HOST_COMPILER"
  export CUDAHOSTCXX="$CUDA_HOST_COMPILER"
fi

cargo build -p search -p encoder -p structural-editor --release "${ENCODER_FEATURES[@]}"
export PATH="$ROOT/target/release:$PATH"

if [[ ! -f "$VOCAB" ]]; then
  cargo run -p encoder --release -- build-vocab \
    --src-dir "$ROOT" \
    --out "$VOCAB"
fi

if [[ ! -f "$LOSS_TSV" ]]; then
  printf "iteration\texamples\tloss\n" > "$LOSS_TSV"
fi

for iter in $(seq 1 "$ITERATIONS"); do
  for crate in $CRATES; do
    dir="$ROOT/$crate"
    [[ -f "$dir/Cargo.toml" ]] || continue

    search_args=(
      --root "$dir"
      --dynamic
      --max-candidates "$MAX_CANDIDATES"
      --value "$VALUE"
      --value-cache "$VERIFY_CACHE"
      --simulations "$SIMULATIONS"
      --dump "$RAW"
    )

    if [[ -f "$WEIGHTS" ]]; then
      search_args+=(
        --encoder-weights "$WEIGHTS"
        --encoder-vocab "$VOCAB"
        --encoder-small
        --encoder-n-ops "$MAX_CANDIDATES"
      )
    fi

    cargo run -p search --release -- "${search_args[@]}"
  done

  cargo run -p encoder --release "${ENCODER_FEATURES[@]}" -- collect \
    --raw "$RAW" \
    --vocab "$VOCAB" \
    --out "$EXAMPLES" \
    --max-len "$MAX_LEN"

  train_log="$(mktemp)"
  train_args=(
    --data "$EXAMPLES"
    --out "$WEIGHTS"
    --n-ops "$MAX_CANDIDATES"
    --small
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --device "$DEVICE"
  )
  if [[ -f "$WEIGHTS" ]]; then
    train_args+=(--resume "$WEIGHTS")
  fi

  cargo run -p encoder --release "${ENCODER_FEATURES[@]}" -- train "${train_args[@]}" 2>&1 | tee "$train_log"
  loss="$(awk -F'loss=' '/loss=/{value=$2} END{print value+0}' "$train_log")"
  examples="$(wc -l < "$EXAMPLES")"
  printf "%s\t%s\t%s\n" "$iter" "$examples" "$loss" >> "$LOSS_TSV"
  rm -f "$train_log"
done
