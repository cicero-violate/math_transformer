# Plan v24 — Qwen Graph Prior Validation + Sparse Student Gate

## Official Name

```text
Checkpoint Graph Prior Proof
```

Runtime intent:

```text
compiled teacher-weight graph G_0 -> controlled sparse adjacency A_0 -> baseline comparisons -> proof that checkpoint-derived topology carries useful signal
```

## Why v24 Exists

v23 implemented the checkpoint-weight graph compiler and sparse-student bridge stub.

v24 moves the project from:

```text
we can compile a large teacher checkpoint into a derived graph artifact
```

to:

```text
the derived graph artifact improves sparse student topology or is rejected as non-useful
```

The key correction is:

```text
Do not assume G_0 is useful because it came from Qwen.
Prove G_0 beats random, hand, learned, and dense baselines under locked gates.
```

## v23 Completion State

Implemented before v24:

```text
src/qwen_weight_graph.py:
  Tensor manifest compiler.
  Typed Qwen/MoE tensor parser.
  Block-energy edge compiler.
  Head/expert structural graph builder.
  Artifact writer/reader.
  WorldGraph schema adapter.
  Sparse student runtime stub.

tests/test_qwen_weight_graph.py:
  104 tests for fixtures, deterministic TopK, graph artifacts, no raw payloads, and gate behavior.

Total test state:
  533 passed.
```

Important distinction:

```text
v23 proves checkpoint graph compilation.
v24 must prove checkpoint graph usefulness.
```

## v24 Decision

```text
Use G_0 as a topology prior only after it survives explicit baseline comparisons.
Do not claim sparse-student quality from artifact generation alone.
Do not claim memory savings until end-to-end student memory is measured.
Do not claim speed proof until repeated locked distributional artifacts pass.
```

Correct pipeline:

```text
runs/qwen_weight_graph/<model_id_or_hash>/
  -> manifest.json / nodes.jsonl / edges.jsonl / stats.json
  -> load_weight_graph_as_world_graph(...)
  -> graph-prior adjacency extraction
  -> controlled sparse student/evaluator path
  -> baseline comparison matrix
  -> acceptance/rejection report
```

Rejected pipeline:

```text
G_0 exists -> assume better student topology
```

## Core Objects

```text
G_0 = compiled checkpoint-derived candidate graph.
A_qwen = sparse adjacency selected from G_0.
A_random = randomized topology with matched node/edge budget.
A_hand = existing middle_preserving_topk baseline.
A_learned = existing champion learned-topology baseline.
A_dense = dense/full baseline.
S(A) = student/evaluator run under adjacency A.
Q = quality metric for task/evaluator.
M = measured memory.
T = measured runtime.
J = Q - λT - γM.
```

Topology comparison set:

| adjacency | meaning |
|---|---|
| dense/full | no sparse-prior restriction |
| hand K=4 | formal hand topology baseline |
| learned K=4 | current fragile quality leader |
| random K=4 | matched random sparse topology |
| qwen K=4 | TopK graph prior from G_0 |
| qwen K=8/16 | wider graph prior stress variants |

## Validation Formula

Initial sparse prior:

```text
A_qwen = TopK(ScoreEdges(G_0), K)
```

Primary signal gate:

```text
Δ_prior = Q(S(A_qwen)) - Q(S(A_random))
```

Baseline gate:

```text
Q(S(A_qwen)) >= Q(S(A_hand_K4)) - ε_Q
```

End-to-end objective:

```text
J(A) = Q(S(A)) - λ T(S(A)) - γ M(S(A))
```

Promotion-style comparison:

```text
A_qwen promotes only if:
  quality_ok = true
  memory_ok = true
  repeated_speed_distribution_ok = true or explicitly marked as speed_pending
  old champion scorer behavior unchanged = true
```

## Artifact Contract

Output directory:

```text
runs/qwen_graph_prior/<model_id_or_hash>/<experiment_id>/
```

Required artifacts:

```text
prior_config.json
baseline_matrix.json
baseline_matrix.csv
adjacency_summary.json
quality_report.json
memory_report.json
runtime_report.json
paired_regression_report.jsonl
promotion_decision.json
```

`prior_config.json` must record:

```text
source_weight_graph_dir
source_manifest_hash
graph_scope
block_size
topk
edge_score_name
selection_policy
random_seed
baseline_set
quality_dataset
runtime_protocol
memory_protocol
```

`baseline_matrix.json` rows must include:

```json
{
  "adjacency_name": "qwen_topk_k4",
  "source": "G_0",
  "k": 4,
  "edge_count": 1234,
  "node_count": 567,
  "route_acc": 0.0,
  "generic_acc": 0.0,
  "affine_acc": 0.0,
  "memory_mb": 0.0,
  "block_ms_median": 0.0,
  "quality_ok": false,
  "memory_ok": false,
  "speed_ok": false
}
```

Hard rule:

```text
No v24 artifact may treat graph generation as quality evidence.
Every quality claim must cite a baseline row.
```

## Implementation Plan

### P0.1 — Graph Prior Loader CLI

Add:

```text
src/qwen_graph_prior_eval.py
```

Acceptance:

```text
Reads a v23 weight-graph artifact directory.
Loads G_0 through load_weight_graph_as_world_graph(...).
Builds qwen_topk adjacency candidates for K in configured set.
Does not load teacher checkpoint weights.
Does not alter champion scorer defaults.
```

### P0.2 — Matched Random Baseline

Add matched random adjacency generator:

```text
A_random ~ Match(node_types, relation_types, in_degree, out_degree, |E_qwen|)
```

Acceptance:

```text
Random baseline preserves edge budget and coarse relation/node-type distribution.
Multiple seeds are supported.
Random baseline artifacts are deterministic under seed.
```

### P1.1 — Baseline Matrix Runner

Compare:

```text
dense/full
hand K=4
learned K=4
random K=4
qwen K={4,8,16}
```

Acceptance:

```text
Writes JSON/CSV baseline matrix.
Reports route/generic/affine quality.
Reports paired wins/losses against hand K=4 and random K=4.
Reports memory and runtime separately.
```

### P1.2 — Graph-Prior Paired Regression Export

For every qwen-vs-baseline disagreement:

```text
sample_id
target
baseline_pred
qwen_pred
qwen_extra_edges
qwen_missing_edges
edge_sources
source_tensor_provenance
```

Acceptance:

```text
Disagreements are explainable down to graph edge provenance.
No raw tensor values are exported.
```

### P1.3 — Sparse Student Stub Upgrade

Upgrade `run_sparse_student_stub` from manifest-only proof to topology-input proof:

```text
G_0 + task labels -> sparse topology/student eval/training stub
```

Acceptance:

```text
Q_student / baseline is reported.
M_student is measured or marked unavailable.
T_student reports cold/cache/static regimes separately.
The result cannot be promoted if required metrics are unavailable.
```

### P2.1 — Prior Selection Ablation

Ablate score policies:

```text
normalized_frobenius
per_tensor_zscore
per_layer_percentile
relation_weighted_score
closure_preserving_score
```

Acceptance:

```text
At least one score policy beats matched random or the result is marked v24_negative.
Policy changes do not affect v23 graph artifact schema.
```

## First Valid v24 Experiment

Use v23 fixture graph first:

```text
graph_fixture ∈ {tiny_safetensors_fixture, small_qwen_style_checkpoint}
K ∈ {4,8,16}
random_seeds ∈ {0,1,2,3,4}
graph_scope ∈ {mlp_only, attention_mlp, attention_mlp_moe}
baselines ∈ {dense, hand_K4, learned_K4, random_matched}
```

Acceptance:

```text
Compiler artifacts load without teacher checkpoint.
A_qwen is built deterministically.
A_random is budget-matched.
Baseline matrix is emitted.
Qwen-prior result is explicitly positive, neutral, or negative.
Existing test suite remains green.
```

First quality acceptance remains conservative:

```text
No claim that Qwen topology is useful unless Δ_prior > 0 against matched random.
No claim that Qwen topology beats current champion unless it passes hand/learned/dense comparison gates.
No speed claim until repeated locked distributional gate passes.
```

## Non-Negotiable Gates

1. Do not run or query the teacher at student runtime.
2. Do not load raw teacher checkpoint weights in v24 evaluation; use G_0 artifacts only.
3. Do not store raw transformer weights in graph-prior artifacts.
4. Do not compare qwen topology against an unmatched random graph.
5. Do not claim usefulness without matched-random improvement.
6. Do not claim promotion without hand K=4 and learned K=4 comparison.
7. Do not claim memory savings without measured student memory.
8. Do not claim speed proof unless repeated locked distributional artifacts pass.
9. Do not mutate champion scorer behavior.
10. Keep full-`G_world` closure forbidden at runtime; closure is bounded/local only.

## Decision

```text
v24 is the graph-prior validation plan.
The next artifact is not a larger compiler.
The next artifact is a baseline matrix proving whether G_0-derived topology helps a sparse student.
```

Summary formula:

```text
G_0 -> A_qwen -> matched baselines -> quality/runtime/memory gates -> prior accepted or rejected
```
