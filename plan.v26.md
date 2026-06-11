# Plan v26 — Adaptive Sparse Student Rewiring Loop

## Official Name

```text
Error-Driven Sparse Graph Rewiring Student
```

Runtime intent:

```text
fixed-topology distilled sparse student -> edge utility traces -> bounded topology proposals -> accept/reject rewiring -> recursively improve sparse adjacency and student weights
```

## Why v26 Exists

v25 proves whether a sparse graph-native student can absorb teacher behavior under fixed or controlled topology.

v26 moves the project from:

```text
θ learns while A is fixed
```

to:

```text
θ and A learn together under bounded, evidence-gated rewiring
```

The key correction is:

```text
Adaptive rewiring is not part of the first distillation proof.
Adaptive rewiring begins only after v25 records enough edge utility and error-trace evidence.
```

## v25 Expected Completion State

Required before v26 positive claims:

```text
Fixed-topology sparse student distillation runner exists.
KD delta and prior delta reports exist.
Edge trace and gradient edge statistics are emitted.
Teacher is not required at student runtime.
Runtime/memory/quality reports exist.
```

Important distinction:

```text
v25 teaches student weights.
v26 teaches student weights and graph structure.
```

## v26 Decision

```text
Allow topology updates only through bounded proposals and proof gates.
Use v25 traces as evidence, not as automatic deletion commands.
Prefer archive/tombstone over destructive edge deletion.
Keep runtime adjacency bounded at all times.
```

Correct pipeline:

```text
S_t(A_t, θ_t)
  -> evaluate / distill / trace
  -> compute edge utility and error signals
  -> propose ΔA_t
  -> bounded closure and regression checks
  -> accept/reject proposals
  -> S_{t+1}(A_{t+1}, θ_{t+1})
```

Rejected pipeline:

```text
low edge score -> delete edge permanently without paired proof
```

## Core Objects

```text
A_t = active sparse adjacency at step t.
θ_t = student parameters at step t.
ΔA_t = proposed edge additions/removals/weight updates.
G_0 = original checkpoint-derived graph prior.
G_pool = optional candidate edge pool from prior graphs and traces.
u_e = measured edge utility.
a_e = edge activation frequency.
g_e = edge gradient signal.
r_e = edge residual/error contribution.
c_e = edge compute/memory cost.
H = bounded closure horizon.
B_A = active edge budget.
```

Rewiring operation types:

| operation | meaning |
|---|---|
| keep | preserve active edge |
| add | activate candidate edge from G_0/G_pool |
| downweight | reduce edge score but keep available |
| archive | remove from active A but preserve provenance |
| tombstone | mark rejected edge with reason and rollback data |

## Rewiring Formula

Student update:

```text
θ_{t+1} = θ_t - η ∇_θ L
```

Topology proposal:

```text
ΔA_t = f(g_e, a_e, u_e, r_e, c_e, G_0, closure_critical_e)
```

Bounded topology update:

```text
A_{t+1} = KeepTopB(Prune(A_t ∪ Add(ΔA_t)), B_A)
```

Constrained objective:

```text
A* = argmax_A [
  Q(S(A))
  - λ |A|
  - γ M(A)
  - β T(A)
  - δ R_old(A)
]
```

Closure-preservation gate:

```text
ImportantPaths(A^{<=H}) ≈ ImportantPaths((A \ e)^{<=H})
```

or deletion must show accepted Pareto tradeoff:

```text
ΔQ >= -ε_Q and ΔM <= 0 and ΔT <= 0
```

## Artifact Contract

Output directory:

```text
runs/adaptive_rewire/<student_id>/<experiment_id>/
```

Required artifacts:

```text
rewire_config.json
initial_adjacency.json
edge_utility.jsonl
proposal_batch.jsonl
accepted_rewrites.jsonl
rejected_rewrites.jsonl
closure_preservation_report.jsonl
old_domain_regression_report.json
rewire_iteration_metrics.jsonl
final_adjacency.json
runtime_report.json
memory_report.json
quality_report.json
```

Optional artifacts:

```text
rollback_manifest.json
candidate_edge_pool.jsonl
edge_tombstones.jsonl
visual_edge_deltas.jsonl
```

`rewire_config.json` must record:

```text
source_distill_run
source_weight_graph_dir
initial_adjacency
candidate_pool
edge_budget
proposal_budget
closure_horizon
acceptance_policy
rollback_enabled
old_domain_regression_budget
```

Hard rule:

```text
No edge is permanently deleted without provenance, paired metrics, and rollback/tombstone data.
```

## Implementation Plan

### P0.1 — Edge Utility Aggregator

Add:

```text
src/adaptive_rewire.py
```

Aggregate from v25 traces:

```text
activation_frequency
gradient_norm
loss_contribution
error_correlation
compute_cost
source_prior_score
closure_critical_flag
```

Acceptance:

```text
Reads edge_trace.jsonl and gradient_edge_stats.jsonl.
Emits normalized edge_utility.jsonl.
Handles missing traces without crashing.
Does not mutate A.
```

### P0.2 — Rewire Proposal Generator

Generate bounded proposals:

```text
remove_candidates = low utility, high cost, non-critical edges
add_candidates = high prior score or high residual-neighbor edges from G_0/G_pool
weight_updates = score/confidence adjustments
```

Acceptance:

```text
Proposal count is bounded.
Proposals include reason codes.
Proposals include source provenance.
No proposal is auto-accepted.
```

### P1.1 — Acceptance Gate Runner

For every proposal batch:

```text
Evaluate A_t vs A_t + ΔA_t.
Compare quality, memory, runtime, paired regressions, closure preservation.
Accept only if gates pass.
```

Acceptance:

```text
accepted_rewrites.jsonl and rejected_rewrites.jsonl are emitted.
Every accepted rewrite has metric evidence.
Every rejected rewrite has reason code.
```

### P1.2 — Bounded Active Graph Enforcement

Acceptance:

```text
|A_t| <= B_A for every iteration.
Budget violations fail closed.
KeepTopB preserves anchors, required self edges, verifier-required nodes, and closure-critical paths.
```

### P1.3 — Old-Domain Regression Harness

Acceptance:

```text
Runs old-domain validation after every accepted batch.
Rejects rewiring batch if Q_old(S_{t+1}) < Q_old(S_t) - ε.
Reports generic/affine and route-level regressions separately.
```

### P2.1 — Recurrent Rewiring Loop

Run:

```text
for t in 0..T_rewire:
  train/eval student
  collect trace
  propose ΔA_t
  gate ΔA_t
  accept/reject
```

Acceptance:

```text
Each iteration emits metrics.
A_t sequence is reproducible.
Final result includes best_by_Q, best_by_J, and best_by_memory variants.
```

### P2.2 — Closure-Critical Edge Preservation

Use bounded quantale closure:

```text
A^{<=H} = I ∨ A ∨ A^2 ∨ ... ∨ A^H
```

Acceptance:

```text
Edges on important bounded paths are marked before pruning.
Deleting/archive of closure-critical edges requires explicit accepted Pareto tradeoff.
No full-G_world closure is computed.
```

## First Valid v26 Experiment

```text
source_run = accepted or neutral v25 distillation run
A_0 ∈ {qwen_K4, qwen_K8}
B_A ∈ {|A_0|, 1.25|A_0|, 1.5|A_0|}
proposal_budget ∈ {8,16,32}
closure_horizon H ∈ {1,2,3}
T_rewire ∈ {1,2,3}
```

Acceptance:

```text
At least one accepted proposal batch improves J or result is marked v26_negative.
Quality regressions are paired and reported.
Runtime and memory are measured.
Old-domain regression gate runs.
Final active graph remains bounded.
Existing test suite remains green.
```

## Non-Negotiable Gates

1. Do not begin adaptive rewiring without v25 trace artifacts or an explicit synthetic trace fixture.
2. Do not mutate topology outside the proposal/acceptance pipeline.
3. Do not permanently delete edges without provenance, rollback, and paired quality/runtime/memory evidence.
4. Do not exceed configured active edge budget.
5. Do not use raw teacher weights at student runtime.
6. Do not claim speed proof unless repeated locked distributional artifacts pass.
7. Do not accept rewiring that breaks old-domain regression budget.
8. Do not prune closure-critical paths without bounded closure comparison or accepted Pareto tradeoff.
9. Keep quantale closure bounded/local; no full-`G_world` closure at runtime.
10. Keep champion scorer behavior unchanged unless passing promotion gates.

## Decision

```text
v26 is the adaptive rewiring plan.
The next artifact is not another teacher.
The next artifact is a bounded evidence-gated loop where student errors and traces improve A over time.
```

Summary formula:

```text
A_t, θ_t -> traces -> ΔA_t proposals -> gates -> A_{t+1}, θ_{t+1}
```
