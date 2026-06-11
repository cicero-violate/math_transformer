# Plan v17 — Learned Topology Transformer Handoff

## Current Status

```text
Project = neurosymbolic/math_transformer
Status = promising learned-topology prototype, not yet scalable proof
Default scorer = runs/checkpoints/topology_scorer.champion.pt
Accepted baseline = K=8 learned topology vs hand K=16 passed current quality/speed proof
Main bottleneck = stable evidence at larger T + trace/evaluator loop + execution speed
```

Core objective:

```text
Q_sparse / Q_dense_or_hand >= 0.95
T_sparse < T_dense_or_hand
J = Q - λT - γM
```

The architecture direction is:

```text
snapshot / IR
→ candidate topology
→ learned edge scores
→ TopK sparse topology
→ neighbor table / sparse IR
→ sparse transformer execution
→ output
→ feedback trace
→ better routing
```

## Key Conceptual Decisions

| Decision | Current Position |
| --- | --- |
| Learned topology | Treat as optimized sparse IR / learned routing policy. |
| Topology vs features | Features describe nodes/edges; topology selects communication/execution edges. |
| Runtime topology | At inference, topology is score-only + TopK; scorer was trained with backprop offline. |
| Baked IR | Useful later; bake neighbor tables per graph/snapshot/config, not one universal graph. |
| Layers | Do not assume many layers. Test `L ∈ {1,2,4,8}` with synced topology. |
| Synced topology | First multi-layer design should reuse the same learned topology across all layers. |
| Current capability | Fixed nodes + learned edge routing. Node selection/editing is a future upgrade. |
| Feature groups | Use multiple modular feature matrices/groups, then concatenate into one versioned scorer tensor. |
| Feature scope | Start with one shared feature tensor/topology across blocks; only add per-block feature/topology variants after depth sweep proves value. |
| Coding model thesis | A 4-layer code-IR model can be powerful if compiler-structured, tool-backed, and evaluator-driven. |

## Priority Order

| Priority | Work item | Status | Acceptance |
| ---: | --- | --- | --- |
| P0.1 | Lock evaluator / benchmark protocol | Partial | Same command configs produce stable, comparable quality/speed reports. |
| P0.2 | Preserve K=8 learned-topology baseline | Done | Keep `scorer_runtime_aligned.runtime_best.pt` as baseline. |
| P0.3 | Add topology trace logging | Partial | `eval_topology_scorer` and quality eval emit compact JSONL traces; wrappers accept `TRACE_OUTPUT`. Full benchmark artifact tracing still pending. |
| P0.4 | Re-run K=6 benchmark after timing-noise investigation | Pending | Quality and speed pass at `BENCH_N=1024`, `BENCH_STEPS=100`. |
| P1.1 | Export learned-topology failure diagnostics | Done | Failure JSONL uses the standard topology trace schema with generic misses, missing/extra edges, hidden/logit drift. |
| P1.2 | Expand hard validation set | Partial | Deeper/longer/held-out/generic-stress examples exist. |
| P1.3 | Select/document default scorer checkpoint | Partial | Scripts/docs consistently reference one default scorer. |
| P1.4 | Profile execution buckets | Partial | Stable timing for topology prepare, QKV, attention, outproj, FFN, router, total. |
| P2.1 | Bake prepared topology cache | Partial | Reused graphs bypass scorer/topology rebuild and load neighbor tables. |
| P2.2 | Large-`T` scaling proof | Pending | Show `T² → T*K` advantage at `T >= 4096`. |
| P2.3 | Execution/kernel specialization pass | Partial | Prepared sparse/block path wins without changing topology behavior. |
| P2.4 | Depth sweep with synced topology | Pending | Compare `L ∈ {1,2,4,8}` using `J=Q-λT-γM`. |
| P3.1 | Feature registry/gating | Pending | Multiple feature groups compile into one versioned tensor; features can be added/deleted by utility without breaking checkpoint schema. |
| P3.2 | Node selection | Pending | Active node set improves quality/runtime objective over fixed-node baseline. |
| P3.3 | Code/AST dataset path | Pending | JSONL supports AST/CFG/SSA/call/type nodes, edges, labels, feedback. |
| P3.4 | Edit/action head | Pending | Model proposes graph/code edits verified by tests/typechecker/evaluator. |

## Feature Matrix / Feature Group Plan

Current topology scorer input:

```text
F ∈ R^{N × N × 10}
S_ij = MLPθ(F_ij)
A = TopK(S)
```

Current 10 edge features:

```text
identity
symbolic_dependency
composition
shape_compat
middle_bridge
local_window
same_operator
embedding_cos
relative_abs_position
relative_signed_position
```

Next feature design:

```text
F_ij = concat(
  F_struct_ij,
  F_embed_ij,
  F_type_shape_ij,
  F_runtime_ij,
  F_trace_ij,
  F_task_ij
)
```

Recommended next schema:

```text
feature_schema = topology_edge_features.v2
d_e ≈ 20 to 32
start with one scorer over the concatenated feature tensor
add feature gates z_k before growing feature count aggressively
```

Important checkpoint rule:

```text
Current checkpoints expect d_e = 10.
Changing d_e requires a new scorer checkpoint or an adapter layer.
Do not silently reuse old checkpoints with a new feature schema.
```

Transformer-block scope:

```text
Default: build F once per graph/snapshot, score once, compile A/neighbors once, reuse A across all L transformer blocks.

H^(l+1) = Block_l(H^(l), A)
A^(0) = A^(1) = ... = A

Only test per-layer features/topologies later:
A^(l) = TopK(Scorer_l(F, H^(l)))
```

Interpretation:

```text
Features describe candidate edges.
Topology selects which edges execute.
Transformer blocks update hidden states over the selected topology.
```

## Immediate Next Session Goals

1. Start from `TODO.md`; it is now the source-of-truth roadmap.
2. Extend trace logging into full benchmark artifacts and richer prediction/error traces before adding new architecture.
3. Treat feature groups as a schema/versioning problem before changing `d_e`.
4. Keep the scorer and topology behavior fixed while profiling execution.
5. Re-run current quality/speed baseline with stable commands.
6. Use `src.topology_trace_replay` / `scripts/select_topology_replay.sh` to select replay candidates for scorer retraining and promotion-loop inputs.

## Files To Inspect First

| Area | Files |
| --- | --- |
| Roadmap | `TODO.md`, `plan.v17.md` |
| Topology | `src/topology.py`, `src/learned_topology.py`, `src/learned_topology_runtime.py` |
| Cache / baked IR | `src/topology_cache.py` |
| Training | `src/train_topology_scorer.py`, `scripts/train_topology_scorer.sh` |
| Eval / benchmark | `src/eval.py`, `src/eval_topology_scorer.py`, `scripts/benchmark_learned_topology.sh`, `scripts/run_learned_topology_quality.sh`, `scripts/eval_topology_scorer.sh` |
| Diagnostics | `src/export_learned_topology_failures.py` |
| Attention execution | `src/attention.py`, `src/sparse_attention.py`, `src/block_sparse_attention.py` |
| Triton kernels | `src/triton_attention.py`, `src/triton_block_sparse_attention.py` |
| Tests | `tests/test_learned_topology.py`, `tests/test_learned_topology_runtime.py`, `tests/test_train_topology_scorer.py`, `tests/test_topology_cache.py`, `tests/test_sparse_attention.py`, `tests/test_block_sparse_attention.py` |

## Current Baseline Commands

```bash
# Main accepted proof, K=8 vs hand K=16
SCORER=runs/checkpoints/topology_scorer.champion.pt \
LEARNED_K=8 \
HAND_K=16 \
scripts/benchmark_learned_topology.sh

# K sweep, allowed to report failures without aborting
SCORER=runs/checkpoints/topology_scorer.champion.pt \
BENCH_STEPS=100 \
BENCH_N=1024 \
scripts/sweep_learned_k.sh

# Learned topology quality evaluation
SCORER=runs/checkpoints/topology_scorer.champion.pt \
scripts/run_learned_topology_quality.sh

# Topology scorer eval
SCORER=runs/checkpoints/topology_scorer.champion.pt \
scripts/eval_topology_scorer.sh

# Runtime-aligned training attempt
./run_training.sh
```

## Trace Logging Target Schema

Add a standard trace object like:

```json
{
  "sample_id": "...",
  "domain": "math",
  "nodes_hash": "...",
  "scorer_checkpoint": "runs/checkpoints/topology_scorer.champion.pt",
  "topology_config": {
    "fixed_k": 8,
    "local_window": 1,
    "middle_bridge_width": 1
  },
  "edge_feature_schema": ["identity", "symbolic_dependency", "composition", "shape_compat", "middle_bridge", "local_window", "same_operator", "embedding_cos", "relative_abs_position", "relative_signed_position"],
  "edge_scores_summary": {
    "min": 0.0,
    "max": 0.0,
    "mean": 0.0
  },
  "active_edges_summary": {
    "n": 0,
    "k": 0,
    "active_edges": 0,
    "sparsity_ratio": 0.0
  },
  "prediction": {},
  "target": {},
  "loss": 0.0,
  "diagnostics": {}
}
```

Full dense `F` and full dense `S` can be optional or compressed. Do not make trace logging explode file sizes by default.

## Non-Negotiable Proof Gates

```text
1. Do not add new architecture before locked evaluator + trace logging exist.
2. Do not change feature schema without checkpoint/version handling.
3. Do not claim sparse is faster unless wall-clock beats dense/hand on stable commands.
4. Do not promote topology/model changes without held-out validation.
5. Prefer better topology over more layers until depth sweep proves otherwise.
```

## Trace Replay / Failure-Driven Selection

Implemented utility:

```bash
scripts/select_topology_replay.sh TRACE.jsonl
```

Direct module:

```bash
python -m src.topology_trace_replay \
  --output runs/replay/topology_replay_candidates.jsonl \
  TRACE.jsonl
```

Purpose:

```text
standardized traces
→ rank route failures / high-divergence examples
→ export replay candidates
→ replay-weighted scorer fine-tuning with appended replay records and replay ratio oversampling
→ later promotion-loop data
```

Current status:

```text
Replay candidate selection exists.
Replay-weighted topology scorer training exists via `--replay-candidates`; missing replay expressions are appended/weighted, `--replay-sample-ratio` can oversample replay steps, and runtime checkpoint selection is route-first/generic-aware.
Actual promotion-loop automation is not implemented yet.
```

## Next Implementation Bias

```text
Do less theory.
Measure more.
Log every routing decision.
Preserve the K=8 baseline.
Only promote changes that improve J = Q - λT - γM.
```
