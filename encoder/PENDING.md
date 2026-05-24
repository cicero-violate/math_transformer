# encoder — pending work

Status: complete.

Implementation note: the runtime adapter structs live in
`neurosymbolic/search/src/main.rs` instead of `encoder/src/infer.rs`. Keeping the
trait impls in `encoder` would require `encoder -> search` while the search CLI
also needs `search -> encoder`, which creates an invalid Cargo package cycle.
The shipped wiring keeps the same behavior with `search` depending on `encoder`.

This file lists what needs to be implemented before the encoder is fully wired
into the MCTS search loop. Items are ordered by dependency.

---

## 1. Wire `EncoderInfer` into `search` as real `Policy` and `Value` adapters — done

**File**: `src/infer.rs` (adapter structs) + `../search/src/main.rs` (CLI wiring)

The adapter pattern is already sketched as a comment block at the bottom of
`encoder/src/infer.rs`. Codex task: turn the comments into real structs.

### What to implement

In `encoder/src/infer.rs`, add:

```rust
use search::policy::{Candidate, Policy};
use search::state::RepoState;
use search::value::Value;
use std::sync::Arc;

pub struct EncPolicy {
    pub inner:     Arc<EncoderInfer>,
    pub templates: Vec<serde_json::Value>,
}

impl Policy for EncPolicy {
    fn propose(&self, state: &RepoState) -> Vec<Candidate> {
        let files = read_state_files(&state.root);
        let history: Vec<String> = state.ops.iter().map(|o| o.to_string()).collect();
        let refs: Vec<&str> = history.iter().map(|s| s.as_str()).collect();
        let (priors, _) = self.inner.run(&files, &refs, &self.templates)
            .unwrap_or_else(|_| (vec![1.0 / self.templates.len() as f32; self.templates.len()], 0.5));
        self.templates.iter().zip(priors)
            .map(|(op, p)| Candidate { op: op.clone(), prior: p })
            .collect()
    }
}

pub struct EncValue {
    pub inner: Arc<EncoderInfer>,
}

impl Value for EncValue {
    fn score(&self, state: &RepoState) -> f32 {
        let files = read_state_files(&state.root);
        let history: Vec<String> = state.ops.iter().map(|o| o.to_string()).collect();
        let refs: Vec<&str> = history.iter().map(|s| s.as_str()).collect();
        self.inner.run(&files, &refs, &[])
            .map(|(_, v)| v)
            .unwrap_or(0.5)
    }
}

/// Read all .rs files under `root`, returning (rel_path, content) pairs.
fn read_state_files(root: &std::path::Path) -> Vec<(String, String)> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root).min_depth(1) {
        let Ok(e) = entry else { continue };
        let name = e.file_name().to_string_lossy();
        if name == "target" || name.starts_with('.') { continue; }
        if e.path().extension().and_then(|x| x.to_str()) != Some("rs") { continue; }
        let rel = e.path().strip_prefix(root).unwrap_or(e.path())
            .to_string_lossy().replace('\\', "/");
        if let Ok(src) = std::fs::read_to_string(e.path()) {
            files.push((rel.into_owned(), src));
        }
    }
    files
}
```

Note: `read_state_files` returns owned `String` pairs; the caller converts to
`&str` slices before passing to `EncoderInfer::run`. The borrow dance:

```rust
let files_owned = read_state_files(&state.root);
let files: Vec<(&str, &str)> = files_owned.iter()
    .map(|(p, c)| (p.as_str(), c.as_str()))
    .collect();
```

### CLI wiring in `../search/src/main.rs`

Add two new flags to `Cli`:

```rust
/// Path to trained encoder weights (safetensors). Enables neural policy + value.
#[arg(long)]
encoder_weights: Option<PathBuf>,

/// Path to encoder vocab (JSON). Required when --encoder-weights is set.
#[arg(long)]
encoder_vocab: Option<PathBuf>,
```

Then in `main()`, if both flags are present, build `EncPolicy` + `EncValue` and
pass them to `Mcts::new` instead of `NoisyTemplatePolicy` / `ConstantValue`.

For a hybrid run (neural policy, symbolic value), use `EncPolicy` + `VerifyValue`.
This is the recommended first-pass because the value head needs more training data
before it can replace the symbolic oracle.

---

## 2. `encoder collect` — include file content in training examples — done

**File**: `src/main.rs`, subcommand `collect`

The current `collect` implementation reads `op_history` from each raw record but
only uses it to construct the token sequence. It does **not** read the actual `.rs`
files from `state.root` at collection time.

Fix: for each raw record, read the `.rs` files under `record["root"]` and call
`tokenizer.encode_state(&files, &op_history_refs, max_len)` with real file content.
The current stub passes empty file content which degrades the encoder's ability to
learn from code structure.

The `read_rs_files(root)` helper already exists in `main.rs` — use it here.

---

## 3. GPU support in `train.rs` — done

**File**: `src/train.rs`

`TrainConfig::device` is `Device::Cpu`. To enable GPU:

```rust
// In TrainConfig or as a CLI flag:
let device = if candle_core::utils::cuda_is_available() {
    Device::new_cuda(0)?
} else if candle_core::utils::metal_is_available() {
    Device::new_metal(0)?
} else {
    Device::Cpu
};
```

Wire as a `--device cpu|cuda|metal` CLI flag in the `train` subcommand.
Tensors created in `data::collate` must be moved to the same device before
passing to the model.

---

## 4. Self-play loop script — done

**File**: `scripts/self_play.sh` (new file, shell or Python)

The loop:
1. Run `search --dynamic --value check --dump raw.jsonl` across all crates
2. Run `encoder collect` to tokenise raw records → `examples.jsonl`
3. Run `encoder train` to produce updated `weights.safetensors`
4. On next iteration, `search` uses `--encoder-weights weights.safetensors`
   (neural policy) instead of `--dynamic` alone
5. Repeat

Each iteration should append to `raw.jsonl` (not overwrite) so the dataset grows.
Track per-iteration loss in a TSV for plotting.

---

## 5. `--n-ops auto` in `encoder train` — done

**File**: `src/main.rs`, subcommand `train`

Currently `--n-ops` must be specified manually. When `--dynamic` is used for data
collection the action space size varies per crate. Fix: infer `n_ops` from the
maximum `policy_target` length across all examples in `examples.jsonl` at load
time, and use that as `EncoderConfig::n_ops`. Emit a warning if examples have
inconsistent policy lengths.

---

## 6. Checkpoint resumption in `train.rs` — done

**File**: `src/train.rs`

Add `--resume <weights>` flag: load existing weights into `VarMap` before
training begins so training can be interrupted and continued without losing
progress. Use `varmap.load(path)?` to restore.

---

## Dependencies note

Items 1 → 2 → 4 are the critical path. Items 3, 5, 6 are quality-of-life
improvements that can follow. Start with item 1 (adapter structs) since a working
`EncPolicy` is what closes the MCTS → encoder feedback loop.
