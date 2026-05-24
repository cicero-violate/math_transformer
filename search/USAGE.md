# search — MCTS-based neurosymbolic code search

AlphaZero-style tree search over structural code edits. A policy proposes ops,
MCTS explores sequences of those ops, and a value function scores the resulting
repo state. The best sequence is extracted by following the most-visited edges.

```
parse repo
    └─ policy proposes ops (priors)
         └─ MCTS explores op sequences (PUCT selection)
              └─ sandbox applies ops to a temp copy
                   └─ value fn scores the result (cargo check / test)
                        └─ backup propagates value up the tree
                             └─ best-sequence extracted by visit count
```

The structural-editor binary executes the actual file edits. The search crate
treats ops as opaque JSON and delegates all edit logic to that subprocess.

---

## Build

```bash
cargo build -p search --release
# structural-editor must be on PATH for --value check/test/composite:
export PATH="$(pwd)/target/release:$PATH"
```

---

## CLI

```
search --root <PATH> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--root <PATH>` | required | Crate root to search from |
| `--dynamic` | off | Scan the crate and generate ops dynamically (replaces `--templates`) |
| `--max-candidates <N>` | `64` | Max ops when `--dynamic` is used |
| `--templates <FILE>` | none | Static JSONL op template list (ignored when `--dynamic` is set) |
| `--value <MODE>` | `stub` | Value function (see below) |
| `--simulations <N>` | `50` | MCTS simulations per run |
| `--max-depth <N>` | `8` | Max tree depth before forced evaluation |
| `--c-puct <F>` | `1.5` | Exploration constant in PUCT formula |
| `--seq-len <N>` | `5` | Max steps in best-sequence extraction |
| `--editor-bin <PATH>` | `structural-editor` | Path to the structural-editor binary |
| `--dump <FILE>` | none | Append all MCTS node records to a JSONL file (see below) |

### Value functions

| `--value` | Speed | Score |
|-----------|-------|-------|
| `stub` | instant | constant 0.5 — tests MCTS plumbing, no real signal |
| `depth` | instant | `depth / max_depth` — deeper = better, no compilation |
| `check` | ~1 s | 1.0 if `cargo check` passes, 0.0 otherwise |
| `test` | ~5–30 s | fraction of tests that pass |
| `composite` | ~5–30 s | `check` gate then test fraction |

`check` and `composite` copy the crate to a tempdir for each sandbox evaluation.
`structural-editor` must be on PATH (or specified via `--editor-bin`).

---

## Policy modes

### Dynamic (recommended)

`--dynamic` uses `CandidateGen`: scans the crate's `.rs` files for every `pub fn`,
`pub struct`, `pub enum`, `pub trait`, `pub type`, and `impl` block, and generates
structural ops for each item. No `ops.jsonl` file needed — the action space adapts
to whichever crate you point `--root` at, and to edits already applied mid-search.

```bash
search --root ./search --dynamic --value check --simulations 100
```

### Static templates

`--templates` loads a JSONL file where each line is one op. Blank lines and lines
starting with `//` are ignored. Useful when you want a fixed, curated action space.

```bash
search --root ./search --templates ops.jsonl --value check --simulations 100
```

See `ops.jsonl` for a worked example against this project.

---

## Op format

Ops are structural-editor JSON. Full reference: `../structural-editor/USAGE.md`.
The most common shapes:

### rename_symbol

```json
{
  "op": "rename_symbol",
  "loc": { "loc": "selector", "path": "src/lib.rs", "selector": "fn old_name" },
  "new_name": "new_name",
  "scope_files": ["src/lib.rs", "src/main.rs"]
}
```

### set_attr — add a derive, attribute, or doc comment

```json
{ "op": "set_attr", "loc": { "loc": "selector", "path": "src/lib.rs", "selector": "struct Foo" }, "key": "derive", "value": "Clone" }
{ "op": "set_attr", "loc": { "loc": "selector", "path": "src/lib.rs", "selector": "fn bar" }, "key": "must_use", "value": "" }
{ "op": "set_attr", "loc": { "loc": "selector", "path": "src/lib.rs", "selector": "struct Foo" }, "key": "doc", "value": "A helpful description." }
```

`key` values (lowercase): `derive`, `cfg`, `allow`, `must_use`, `inline`,
`deprecated`, `doc`, `repr`, `visibility`, `custom`.

### create_node — insert a function or create a file

```json
{
  "op": "create_node",
  "kind": "Function",
  "loc": { "loc": "selector", "path": "src/lib.rs", "selector": "impl MyStruct" },
  "text": "    pub fn new() -> Self { Self::default() }"
}
```

```json
{
  "op": "create_node",
  "kind": "File",
  "loc": { "loc": "anchor", "path": "src/new_module.rs", "byte_from": 0, "byte_to": 0 },
  "text": "pub struct NewType;\n"
}
```

### replace_node — rewrite a function body or whole item

```json
{
  "op": "replace_node",
  "loc": { "loc": "selector", "path": "src/lib.rs", "selector": "fn score" },
  "target": "Body",
  "text": "{ 1.0 }"
}
```

`target` values: `Whole`, `Body`, `Signature`, `Type`, `Value`.

### delete_node

```json
{
  "op": "delete_node",
  "loc": { "loc": "selector", "path": "src/lib.rs", "selector": "fn unused" },
  "compiler_proven_unused": true
}
```

### verify — gate op (asserts a predicate, 0 edit cost)

```json
{ "op": "verify", "predicate": { "predicate": "file_exists", "path": "src/lib.rs" } }
{ "op": "verify", "predicate": { "predicate": "symbol_exists", "path": "src/lib.rs", "symbol": "MyStruct" } }
{ "op": "verify", "predicate": { "predicate": "contains_text", "path": "src/lib.rs", "text": "impl Default" } }
```

### NodeLocator forms

| Form | When to use |
|------|-------------|
| `"loc": "selector"` with `"selector": "fn foo"` | Named item — resilient to reformatting |
| `"loc": "anchor"` with `"byte_from"` / `"byte_to"` | Exact byte range — precise but fragile |

---

## Output

`stderr` — action probabilities and search stats:
```
Action probabilities (64 actions):
  0.0156  {"op":"verify", ...}
  0.0156  {"op":"set_attr", ...}
  ...
Root value: 1.0000  |  tree nodes: 12
```

`stdout` — best op sequence by visit count:
```
Best sequence (3 steps):
  0: {"op":"set_attr", ...}
  1: {"op":"create_node", ...}
  2: {"op":"verify", ...}
```

---

## Training data collection (`--dump`)

`--dump <FILE>` appends one JSON record **per MCTS node visited** (not just the
root) to the file. Each record:

```json
{
  "root":       "/abs/path/to/crate",
  "op_history": ["...", "..."],
  "policy":     [0.083, 0.021, ...],
  "value":      0.92
}
```

`policy` is the visit-count distribution over the node's children (AlphaZero
improved policy). `value` is the MCTS-backed-up value at that node.

### Full data collection across all crates

```bash
export PATH="$(pwd)/target/release:$PATH"
rm -f raw.jsonl

for crate in search encoder structural-editor algorithms ai score; do
  dir="$(pwd)/$crate"
  [ -f "$dir/Cargo.toml" ] || continue
  echo "=== $crate ==="
  search \
    --root "$dir" \
    --dynamic --max-candidates 64 \
    --value check \
    --simulations 50 \
    --dump raw.jsonl 2>&1 | grep -E "Root value|Dumping|tree nodes"
done

echo "Total records: $(wc -l < raw.jsonl)"
```

This feeds directly into `encoder collect` (see `ENCODER_PIPELINE.md`).

---

## Wiring in a trained encoder

After training, implement the adapter structs sketched in `encoder/src/infer.rs`
and wire them into `main.rs`:

```rust
use encoder::infer::EncoderInfer;
use encoder::config::EncoderConfig;
use std::sync::Arc;

let infer = Arc::new(EncoderInfer::load(
    Path::new("weights.safetensors"),
    Path::new("vocab.json"),
    EncoderConfig::small(),
)?);

let policy = EncPolicy { inner: infer.clone(), templates };
let value  = EncValue  { inner: infer.clone() };
let mut mcts = Mcts::new(config, policy, value);
```

During early training keep `--value check` as the ground-truth verifier and use
the encoder for **policy priors only** (hybrid: neural policy + symbolic value).
Once the value head converges, replace `VerifyValue` with `EncValue` too.

See `ENCODER_PIPELINE.md` for the full train → deploy loop.
