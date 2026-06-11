# Plan v27 — Continual Multi-Teacher Graph Ingestion

## Official Name

```text
Bounded Multi-Teacher Graph Pool Student
```

Runtime intent:

```text
multiple teacher checkpoints -> multiple compiled graph priors -> offline graph pool -> evidence-gated edge selection -> bounded active sparse student graph -> continual distillation without unbounded runtime growth
```

## Why v27 Exists

v26 adds adaptive rewiring for one sparse student.

v27 moves the project from:

```text
one teacher-derived graph prior can initialize and improve one sparse student
```

to:

```text
many teacher-derived graph priors can be ingested over time while the runtime student remains bounded
```

The key correction is:

```text
Continual ingestion is not continual edge accumulation.
The library may grow; the active student adjacency must stay bounded.
```

## v26 Expected Completion State

Required before v27 positive claims:

```text
Adaptive rewiring loop exists.
Edge utility aggregator exists.
Proposal/acceptance gates exist.
Old-domain regression gate exists.
Runtime active graph budget is enforced.
Accepted/rejected rewrite artifacts exist.
```

Important distinction:

```text
v26 makes one student recursive over its own topology.
v27 makes the teacher-source library continual while keeping runtime sparse.
```

## v27 Decision

```text
Every teacher contributes candidate structure, not mandatory runtime structure.
Use a teacher registry and offline graph pool.
Accept edges by evidence, conflict handling, and old-domain regression gates.
Do not treat all teachers as equally reliable across all domains.
```

Correct pipeline:

```text
teacher checkpoint_i
  -> v23 compiler -> G_0_i
  -> teacher registry entry
  -> graph pool manifest update
  -> conflict and provenance analysis
  -> bounded candidate selection
  -> v26-style acceptance gates
  -> active adjacency A*_t
```

Rejected pipeline:

```text
for every new teacher:
  append all edges to active student graph
```

## Core Objects

```text
Θ_T^{(i)} = checkpoint weights of teacher i.
G_0^{(i)} = graph compiled from teacher i.
TeacherRegistry = metadata and reliability records for all teacher sources.
G_pool = offline graph library of all candidate teacher-derived graphs.
A_t = active runtime sparse adjacency.
S_t = sparse student at ingestion step t.
ρ_i(x) = domain/input-conditioned teacher reliability weight.
C = conflict cost across teachers/domains.
R = robustness score.
B_A = active adjacency edge budget.
```

Teacher source types:

| source type | allowed use |
|---|---|
| checkpoint graph | candidate topology prior from v23 artifacts |
| top-r logits | behavior distillation artifact from v25 |
| hidden summary | optional behavior artifact, gated |
| symbolic verifier | task/evaluator label source |
| domain-specific model | teacher source with domain reliability weights |

## Multi-Teacher Formula

Graph pool update:

```text
G_pool,t+1 = G_pool,t ∪ G_0^{(i)}
```

Active adjacency selection:

```text
A* = argmax_{A ⊆ G_pool} [
  Q(A)
  - λ |A|
  - γ M(A)
  - β T(A)
  - δ C(A)
  + κ R(A)
]
```

Multi-teacher distillation:

```text
L_multiKD = Σ_i ρ_i(x) KL(p_{T_i}^τ(.|x) || p_S^τ(.|x))
```

Total loss:

```text
L_v27 = L_multiKD + α L_task + λ |A| + γ M + β T + δ C - κ R
```

Conflict-aware mixture target:

```text
p_mix(y|x) = Σ_i ρ_i(x) p_{T_i}(y|x)
```

Runtime budget invariant:

```text
|A_t| <= B_A for all t
```

Old-domain regression gate:

```text
Q_old(S_{t+1}) >= Q_old(S_t) - ε
```

## Artifact Contract

Output directory:

```text
runs/multi_teacher_ingestion/<student_id>/<ingestion_id>/
```

Required artifacts:

```text
teacher_registry.json
graph_pool_manifest.json
ingestion_config.json
teacher_ingestion_report.json
candidate_edges.jsonl
accepted_edges.jsonl
rejected_edges.jsonl
conflict_report.jsonl
teacher_reliability_report.json
old_domain_regression_report.json
active_adjacency_budget_report.json
multi_teacher_kd_report.json
quality_report.json
memory_report.json
runtime_report.json
```

Optional artifacts:

```text
domain_router_report.json
teacher_disagreement_examples.jsonl
graph_pool_index.sqlite
edge_lineage.jsonl
rollback_manifest.json
```

`teacher_registry.json` must record:

```json
{
  "teacher_id": "hash-or-name",
  "source_model": "model-id-or-local-path",
  "source_config_hash": "hash",
  "source_index_hash": "hash",
  "compiler_version": "version",
  "graph_artifact_dir": "runs/qwen_weight_graph/...",
  "domain_tags": ["math", "code"],
  "teacher_type": "dense_or_moe_or_verifier",
  "reliability_score": 0.0,
  "accepted_edge_count": 0,
  "rejected_edge_count": 0,
  "last_ingested_at": "timestamp"
}
```

Hard rule:

```text
G_pool can grow offline.
A_t cannot grow beyond configured runtime budget.
```

## Implementation Plan

### P0.1 — Teacher Registry Schema

Add:

```text
src/multi_teacher_registry.py
```

Acceptance:

```text
Registers teacher graph artifacts by hash.
Records config/index/compiler hashes.
Records domain tags and reliability score.
Rejects duplicate teacher entries unless explicitly versioned.
Does not store raw checkpoint weights in registry records.
```

### P0.2 — Graph Pool Manifest

Add offline graph-pool index:

```text
G_pool = {G_0^{(1)}, G_0^{(2)}, ..., G_0^{(n)}}
```

Acceptance:

```text
Graph pool stores references to graph artifacts, not copied raw weights.
Pool manifest records node/edge counts, relation counts, source teachers, and hash lineage.
Runtime student loader can request bounded slices only.
```

### P0.3 — Active Budget Gate

Acceptance:

```text
Every ingestion run reports |A|, B_A, and budget_ok.
If |A| > B_A, ingestion fails closed before evaluation claims.
```

### P1.1 — Cross-Teacher Edge Canonicalization

Canonical key:

```text
canonical_edge_key = hash(src_type, src_role, dst_type, dst_role, rel, layer_delta, block_range_signature)
```

Acceptance:

```text
Equivalent edges across teachers can be merged or compared.
Teacher-specific provenance is retained.
Conflicting scores are preserved, not overwritten.
```

### P1.2 — Teacher Conflict Report

Detect:

```text
logit disagreement
edge-prior disagreement
domain mismatch
old-domain regression
quality tradeoff
```

Acceptance:

```text
conflict_report.jsonl identifies teacher pair, domain, sample_id/edge_id, conflict score, and proposed resolution.
No teacher is globally trusted by default.
```

### P1.3 — Evidence-Gated Edge Ingestion

For each candidate edge from new teacher:

```text
accept if ΔQ > 0 or ΔM < 0 or ΔT < 0 or ΔR > 0
and old-domain regression is within ε
and active graph remains within B_A
```

Acceptance:

```text
accepted_edges.jsonl and rejected_edges.jsonl are emitted.
Every edge has reason code and teacher provenance.
Accepted edges can be rolled back by ingestion_id.
```

### P2.1 — Multi-Teacher KD Mixture

Implement teacher mixture weights:

```text
ρ_i(x) = DomainRouter(x, teacher_i)
```

Start with static weights:

```text
ρ_i = normalized reliability score for matching domain tag
```

Acceptance:

```text
Supports top-r logit artifacts from multiple teachers.
Reports teacher contribution per domain.
Reports disagreement examples.
Does not require live teacher inference at student runtime.
```

### P2.2 — Continual Ingestion Harness

Run sequence:

```text
for teacher_i in ingestion_order:
  register teacher_i
  add G_0_i to graph pool
  propose candidate edges
  evaluate/gate candidate batch
  update A_t if accepted
  run old-domain regression
  emit ingestion report
```

Acceptance:

```text
Sequential ingestion is reproducible.
Each step reports ΔQ, ΔM, ΔT, ΔR, C, |A|, and old-domain regressions.
A negative teacher is archived but not active.
```

### P3.1 — Domain Router

Later add learned router:

```text
ρ_i(x) = softmax(r_i(x))
```

Acceptance:

```text
Domain router must beat static domain weights before use.
Router cannot bypass teacher conflict gates.
```

## First Valid v27 Experiment

```text
teachers ∈ {tiny_fixture_teacher_A, tiny_fixture_teacher_B, small_qwen_style_teacher}
ingestion_order ∈ fixed deterministic order
B_A ∈ {|A_v26|, 1.25|A_v26|}
top_r ∈ {16,32}
conflict_policy ∈ {prefer_domain_match, weighted_average, reject_conflict}
old_domain_epsilon ∈ {0, 0.005, 0.01}
```

Acceptance:

```text
teacher_registry.json is emitted.
graph_pool_manifest.json is emitted.
At least one teacher contributes accepted edges or result is marked v27_negative.
A negative/conflicting teacher is rejected without breaking the run.
Runtime active graph remains bounded.
Old-domain regression gate runs after every accepted batch.
Existing test suite remains green.
```

## Non-Negotiable Gates

1. Do not append every teacher edge into active A.
2. Do not allow runtime adjacency to grow without budget.
3. Do not store raw teacher checkpoint weights in registry, graph pool, or active graph records.
4. Do not require live teacher inference at student runtime.
5. Do not accept a teacher graph without quality/memory/runtime/conflict evidence.
6. Do not overwrite existing student behavior without old-domain regression gates.
7. Do not treat all teachers as equally reliable across all domains.
8. Do not merge conflicting teacher edges by overwriting provenance.
9. Do not claim continual learning unless sequential ingestion beats or preserves prior gates.
10. Keep quantale closure bounded/local; no full-`G_world` closure at runtime.

## Decision

```text
v27 is the continual multi-teacher graph ingestion plan.
The next artifact is not an infinitely growing runtime graph.
The next artifact is a teacher registry + graph pool + bounded active adjacency selection system.
```

Summary formula:

```text
{teacher_i} -> {G_0_i} -> G_pool -> evidence-gated A* -> sparse student S -> regression-safe continual improvement
```
