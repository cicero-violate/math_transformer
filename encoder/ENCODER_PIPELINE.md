# Encoder Training Pipeline

End-to-end instructions for collecting training data from the workspace and
training the dual-head encoder (policy + value) used by MCTS code search.

```
workspace .rs files
       │
       ▼
 encoder build-vocab          → vocab.json
       │
       ▼
 search --dump  (per crate)   → raw.jsonl       (root path + MCTS policy + value score)
       │
       ▼
 encoder collect              → examples.jsonl  (tokenised TrainExample JSONL)
       │
       ▼
 encoder train                → weights.safetensors
       │
       ▼
 EncoderInfer::load()         ← wired into search as Policy + Value
```

---

## Prerequisites

Build the two binaries used at runtime:

```bash
# From the prototype workspace root:
cargo build -p structural-editor --release
cargo build -p search --release
cargo build -p encoder --release

# Make structural-editor available on PATH for this shell session:
export PATH="$(pwd)/target/release:$PATH"
```

---

## Step 1 — Build vocabulary

Scan all `.rs` files in the prototype workspace and build a token vocabulary.

```bash
cargo run -p encoder --release -- build-vocab \
  --src-dir . \
  --out vocab.json \
  --vocab-size 8192
```

`vocab.json` is a `{ token: id }` JSON map. Keep it alongside your weights.

---

## Step 2 — Collect training data with MCTS

Run MCTS once per crate. Each run appends one record to `raw.jsonl`:

```json
{"root": "/abs/path/to/crate", "policy": [0.071, ...], "value": 0.92}
```

### Quick run (stub value — tests pipeline only, no real signal)

```bash
for crate in neurosymbolic/search neurosymbolic/encoder neurosymbolic/structural-editor algorithms; do
  cargo run -p search --release -- \
    --root "$(pwd)/$crate" \
    --dynamic --max-candidates 32 \
    --value stub \
    --simulations 10 \
    --dump raw.jsonl
done
```

### Full run (real signal — requires structural-editor on PATH)

`--value check` runs `cargo check` in a sandbox for each leaf (~0.5–2 s per sim).
`--dynamic` scans the target crate and generates ops automatically — no `ops.jsonl`
needed. `--dump` writes one record per MCTS node visited (interior nodes included).

```bash
export PATH="$(pwd)/target/release:$PATH"
rm -f raw.jsonl

for crate in neurosymbolic/search neurosymbolic/encoder neurosymbolic/structural-editor algorithms ai score; do
  dir="$(pwd)/$crate"
  [ -f "$dir/Cargo.toml" ] || continue
  echo "=== $crate ==="
  cargo run -p search --release -- \
    --root "$dir" \
    --dynamic --max-candidates 64 \
    --value check \
    --simulations 50 \
    --dump raw.jsonl 2>&1 | grep -E "Root value|Dumping|tree nodes"
done

echo "Total records: $(wc -l < raw.jsonl)"
```

### Parameters

| Flag | Meaning |
|------|---------|
| `--simulations` | Tree walks per run. 50–200 is good for dev; 1000+ for real training. |
| `--max-depth` | How deep the tree grows. Default 8. |
| `--c-puct` | Exploration constant. Default 1.5 (AlphaZero uses 1–2). |
| `--value check` | Fast binary score (pass/fail). Good for early training. |
| `--value composite` | Slower but fractional (test pass ratio). Better signal. |

---

## Step 3 — Tokenise raw records

Converts each raw record into a `TrainExample` with padded token IDs:

```bash
cargo run -p encoder --release -- collect \
  --raw raw.jsonl \
  --vocab vocab.json \
  --out examples.jsonl \
  --max-len 2048
```

Each output line:
```json
{"tokens": [1, 45, 312, ...], "policy_target": [0.071, ...], "value_target": 0.92}
```

---

## Step 4 — Train

```bash
# Small model (fast, good for dev — 4 layers, d_model=256):
cargo run -p encoder --release -- train \
  --data examples.jsonl \
  --out weights.safetensors \
  --n-ops auto \
  --device auto \
  --small \
  --epochs 20 \
  --lr 1e-4

# Full model (6 layers, d_model=512) — use once you have 10k+ examples:
cargo run -p encoder --release -- train \
  --data examples.jsonl \
  --out weights.safetensors \
  --n-ops auto \
  --device auto \
  --epochs 50 \
  --lr 3e-4 \
  --batch-size 64
```

`--n-ops auto` infers the maximum `policy` length in your `examples.jsonl`.
When using `--dynamic --max-candidates 64`, set `--n-ops 64`.
When using a static `--templates ops.jsonl` with 14 lines, set `--n-ops 14`.
Use `--resume weights.safetensors` to continue from an existing checkpoint.

---

## Step 5 — Wire trained model into search

Pass encoder weights and vocab to `search` to use encoder-backed policy priors:

```rust
search \
  --root ./neurosymbolic/search \
  --dynamic \
  --value check \
  --encoder-weights weights.safetensors \
  --encoder-vocab vocab.json \
  --encoder-small
```

During early training, keep `--value check` as the verifier and use the encoder
only for the **policy** priors (hybrid: neural policy + symbolic value). Once
the value head is good enough, switch to `--value encoder`.

---

## Growing the dataset

One run per crate = one example. To scale up:

- **More crates**: point `--root` at every crate in the workspace.
- **More roots**: run on external Rust repos you clone locally.
- **Interior states**: modify `Mcts::run()` to emit a record for every node
  visited, not just the root (each visited node has a visit-count distribution
  and a backed-up value).
- **Self-play loop**: run search → apply best sequence → run search again on
  the mutated repo. Each iteration produces a new example and a better codebase.

---

## Quick sanity-check (end to end in ~30 s)

```bash
cargo build -p encoder -p search --release

cargo run -p encoder --release -- build-vocab \
  --src-dir . --out vocab.json --vocab-size 4096

cargo run -p search --release -- \
  --root "$(pwd)/neurosymbolic/search" \
  --templates neurosymbolic/search/ops.jsonl \
  --value stub --simulations 10 --dump raw.jsonl

cargo run -p encoder --release -- collect \
  --raw raw.jsonl --vocab vocab.json --out examples.jsonl

cargo run -p encoder --release -- train \
  --data examples.jsonl --out weights.safetensors \
  --n-ops auto --small --epochs 3 --batch-size 1
```
