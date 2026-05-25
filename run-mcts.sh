#!/usr/bin/env bash
set -euo pipefail

# MCTS search with judgement-delta value function.
# Run from anywhere — script resolves paths relative to prototype/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTOTYPE="$SCRIPT_DIR/.."

# Pre-build required binaries if not present.
if [[ ! -f "$PROTOTYPE/target/debug/canon-rustc-v3" || ! -f "$PROTOTYPE/target/debug/judgement" ]]; then
    echo "[run-mcts] building canon-rustc-v3 and judgement..."
    cargo build --manifest-path "$PROTOTYPE/Cargo.toml" -p canon-rustc-v3 -p judgement
fi

cd "$PROTOTYPE"

# Pre-warm the sandbox dep cache so MCTS simulations don't have to compile
# axum/tokio/etc from scratch on the first simulation.
SANDBOX_TARGET="target/sandbox-target"
if [[ ! -d "$SANDBOX_TARGET/debug/deps" ]]; then
    echo "[run-mcts] pre-warming sandbox dep cache (first run — compiling ai deps, this takes a few minutes)..."
    cargo check -p ai --target-dir "$SANDBOX_TARGET" 2>&1 | grep -E "^(error|warning:|Compiling|Finished|error\[)" || true
    echo "[run-mcts] dep cache ready at $SANDBOX_TARGET"
fi

cargo run -p search -- \
    --root ai \
    --value judgement-delta \
    --baseline-judgement state/judgement.jsonl \
    --canon-bin target/debug/canon-rustc-v3 \
    --judgement-bin target/debug/judgement \
    --crate-name ai \
    --dynamic \
    --simulations "${SIMULATIONS:-50}" \
    --dump state/node_records.jsonl \
    --value-cache state/value_cache.jsonl \
    "$@"
