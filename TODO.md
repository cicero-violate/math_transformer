# TODO — Learned Topology Transformer

## Current State

```text
Project = neurosymbolic/math_transformer
Status = promoted learned-topology scorer exists; scalable proof still pending
Champion scorer = runs/checkpoints/topology_scorer.champion.pt
Champion metadata = runs/checkpoints/topology_scorer.champion.json
Current accepted proof = learned topology K=8 beats hand topology K=16 on route quality and passes strict speed gate at current benchmark scale
Main bottleneck = stable large-T/runtime proof + execution speed + broader validation
```

Canonical objective:

```text
Q_sparse / Q_dense_or_hand >= 0.95
T_sparse < T_dense_or_hand
J = Q - λT - γM
```

Promotion rule:

```text
Candidate promotes only if:
route_acc_candidate >= route_acc_champion
generic_acc_candidate >= generic_acc_champion
benchmark quality_ok=True
benchmark speed_ok=True
benchmark strict_speed_ok=True
```

Current champion proof:

```text
route_acc = 0.9871220604703248
generic_expert = 236/259 = 0.9111969111969112
benchmark speedup ≈ 1.0146x on current benchmark
strict_speed_ok=True
```

## Status Meanings

| Status  | Meaning                                                                     |
| ---     | ---                                                                         |
| Done    | Implemented and accepted into the current baseline.                         |
| Partial | Exists, but still needs proof, cleanup, integration, or broader validation. |
| Pending | Not implemented or not yet demonstrated.                                    |
| Later   | Useful, but below current proof/engineering path.                           |

## Priority Roadmap

| Rank | Work item                                   | Status  | Evidence / Current State                                                                                                              | Next Action                                                                                                           |
| ---: | ---                                         | ---     | ---                                                                                                                                   | ---                                                                                                                   |
|    1 | Champion scorer promotion path              | Done    | `src/promote_topology_scorer.py`, `scripts/promote_topology_scorer.sh`, and `runs/checkpoints/topology_scorer.champion.*` exist.      | Keep champion immutable except through promotion gate.                                                                |
|    2 | Wire champion as default scorer             | Done    | Quality, benchmark, sweep, topology eval, and failure export defaults now use `runs/checkpoints/topology_scorer.champion.pt`.         | Keep training outputs separate so experiments cannot overwrite champion.                                              |
|    3 | Preserve K=8 learned-topology baseline      | Done    | Champion K=8 learned topology passes current quality/speed proof vs hand K=16.                                                        | Use champion as default runtime scorer and historical baselines only for comparison.                                  |
|    4 | Route-first/generic-aware runtime selection | Done    | Runtime checkpoint selection now prioritizes route accuracy, then generic expert accuracy, before dense/hidden/logit proxy metrics.   | Keep proxy metrics as tie-breakers only.                                                                              |
|    5 | Replay/failure learning loop                | Partial | Trace replay, appended replay records, replay weighting, replay oversampling, and route-first runtime selection exist.                | Run replay loops only through promotion gate; tune replay ratios only if they beat champion.                          |
|    6 | Lock evaluator and benchmark protocol       | Partial | Quality/benchmark scripts exist; benchmark artifacts now record config, seeds, hardware, hashes, quality, speed, and gates.            | Run repeated locked benchmark artifacts on target hardware to quantify timing variance.                                |
|    7 | Topology trace logging                      | Done    | `src/topology_trace.py`, scorer eval traces, quality traces, standardized failure traces, and benchmark JSON/JSONL artifacts exist.    | Keep artifacts as promotion/regression audit evidence.                                                                |
|    8 | Learned-topology failure diagnostics        | Done    | `src/export_learned_topology_failures.py` emits standard trace schema with route misses, missing/extra edges, and hidden/logit drift. | Keep diagnostics as replay/training input and promotion audit evidence.                                               |
|    9 | Hard validation expansion                   | Partial | Synthetic hard validation exists; generic expert remains the main failure mode.                                                       | Add deeper trees, longer dependencies, ambiguous operators, held-out templates, larger `T`, and generic-stress cases. |
|   10 | Prepared topology cache / baked sparse IR   | Partial | `src/topology_cache.py` has prepared topology and neighbor-table caching; baked artifact workflow is not canonical.                   | Cache compiled neighbor tables keyed by nodes, feature schema, scorer checkpoint, topology config, and window.        |
|   11 | Execution bottleneck profiling              | Partial | Profiling hooks and benchmark bucket reports exist.                                                                                   | Produce stable bucket reports for topology prepare, QKV, attention kernel, outproj, FFN, router, total.               |
|   12 | Sparse attention/runtime stabilization      | Partial | Token sparse, block sparse, Triton, and prepared paths exist; speed win is config-dependent.                                          | Optimize execution without changing topology behavior.                                                                |
|   13 | Large-`T` scaling proof                     | Pending | Current accepted proof is at current benchmark scale; large `T` sweep is incomplete.                                                  | Benchmark `T=2048,4096,8192,16384` with prepared sparse/block paths.                                                  |
|   14 | Block sparse path / memory locality         | Partial | Block sparse modules/tests exist.                                                                                                     | Benchmark block sparse vs token sparse at `T >= 4096`; prefer block path if wall-clock wins.                          |
|   15 | Depth sweep with synced topology            | Pending | Multi-layer model path exists, but no canonical `L ∈ {1,2,4,8}` report.                                                               | Measure `J(L)=Q(L)-λT(L)-γM(L)` using the same topology across layers.                                                |
|   16 | Shared-topology multi-layer default         | Partial | `MathRoutedTransformer` can stack layers with same topology config/cache.                                                             | Start with shared `A` across layers; promote only if quality gain beats runtime/memory cost.                          |
|   17 | Feature registry and feature gates          | Pending | Current scorer feature schema is fixed at 10 edge features.                                                                           | Add feature schema registry, learned gates `z_j`, and ablation reports before changing checkpoint shape.              |
|   18 | Node selection / active-node routing        | Pending | Current system uses fixed nodes + learned edges.                                                                                      | Add node scorer `q_i=g(x_i)` and test `V_active=TopM(q)`.                                                             |
|   19 | Dynamic-K controller                        | Pending | Fixed-K learned topology path exists.                                                                                                 | Learn per-node `K_i=controller(node_i, uncertainty_i, importance_i)` after fixed-K proof is stable.                   |
|   20 | Shape/type validity model                   | Partial | Shape/env handling exists; replay env canonicalization fixed JSON list vs tuple issue.                                                | Train `expression + env → valid/invalid + output shape`.                                                              |
|   21 | Symbolic math embedding model               | Partial | `MathEmbedder` exists without dedicated contrastive/equivalence objective.                                                            | Train equivalent expressions close and different expressions far.                                                     |
|   22 | Algebraic canonicalizer                     | Partial | Parser/normalizer/canonical-ish flow exists.                                                                                          | Expand rewrite/canonicalization coverage and tests.                                                                   |
|   23 | Code/AST dataset path                       | Pending | No code-IR dataset path in this module.                                                                                               | Build JSONL with AST/CFG/SSA/call/type nodes, edges, labels, and evaluator feedback.                                  |
|   24 | Program dependency graph                    | Pending | Not implemented in this module.                                                                                                       | Model files/functions/types/variables plus call/import/read/write/def-use edges.                                      |
|   25 | Edit/action head                            | Pending | Current system selects/reroutes edges; it does not mutate graph/code nodes as actions.                                                | Add actions: insert/delete/replace node, add/remove edge, rewrite subtree, call tool, run tests.                      |

## Immediate Pending Tasks

| Priority | Task                                   | Status  | Target / Acceptance                                                                                           |
| ---:     | ---                                    | ---     | ---                                                                                                           |
| P0       | Keep champion default wired            | Done    | Inference/eval scripts default to `runs/checkpoints/topology_scorer.champion.pt`.                             |
| P0       | Lock evaluator command suite           | Partial | One command set produces stable quality/speed reports with committed settings and seeds.                      |
| P0       | Full benchmark artifact trace          | Done    | `scripts/benchmark_learned_topology.sh` emits JSON/JSONL artifacts with quality, speed buckets, hashes, hardware, config, and gates. |
| P0       | Champion regression check script       | Done    | `scripts/check_topology_champion_regression.sh` benchmarks the champion and verifies artifact gates against champion metadata.        |
| P1       | Expand hard validation set             | Partial | Add deeper/longer/held-out/generic-stress examples and report route/expert breakdown.                         |
| P1       | Profile execution buckets              | Partial | Produce stable timing table for topology prepare, QKV, attention, outproj, FFN, router, total.                |
| P1       | Bake prepared topology cache           | Partial | Reused graphs/snapshots bypass scorer/topology rebuild and load neighbor tables directly.                     |
| P1       | Large-`T` scaling benchmark            | Pending | Show `T² → T*K` advantage at `T >= 4096`.                                                                     |
| P2       | Kernel/execution specialization pass   | Partial | Stable prepared-block speed win beyond timing noise; do not change topology behavior.                         |
| P2       | Depth sweep with synced topology       | Pending | Compare `L ∈ {1,2,4,8}` and select by `J=Q-λT-γM`.                                                            |
| P2       | Replay-loop sweep under promotion gate | Partial | Replay variants run through route-first runtime selection and champion promotion only.                        |
| P3       | Feature registry/gating                | Pending | Features can be added/deleted by measured utility without silently breaking checkpoints.                      |
| P3       | Node selection                         | Pending | Active node set improves quality/runtime objective over fixed-node baseline.                                  |
| P3       | Code/AST dataset path                  | Pending | Code sample JSONL supports nodes, node features, known edges, labels, evaluator feedback.                     |
| P3       | Edit/action head                       | Pending | Model proposes graph/code edits verified by tests/typechecker/evaluator.                                      |

## Useful Commands

```bash
# Main accepted proof, champion K=8 vs hand K=16
LEARNED_K=8 \
HAND_K=16 \
scripts/benchmark_learned_topology.sh

# Champion regression proof with structured artifact audit
scripts/check_topology_champion_regression.sh

# Learned topology quality evaluation, champion default
scripts/run_learned_topology_quality.sh

# Export failures from champion default
python -m src.export_learned_topology_failures \
  --output runs/diagnostics/learned_topology_failures.champion.jsonl

# Select replay candidates from failure traces
scripts/select_topology_replay.sh runs/diagnostics/learned_topology_failures.champion.jsonl

# Promote a candidate only after benchmark log passes
CANDIDATE=runs/checkpoints/<candidate>.pt \
BENCHMARK_LOG=runs/benchmarks/<candidate>.log \
scripts/promote_topology_scorer.sh
```

## Non-Negotiable Proof Gates

```text
1. Do not promote outside `scripts/promote_topology_scorer.sh` / `src.promote_topology_scorer`.
2. Do not overwrite `runs/checkpoints/topology_scorer.champion.pt` during training.
3. Do not change feature schema without checkpoint/version handling.
4. Do not claim sparse is faster unless wall-clock beats dense/hand on stable commands.
5. Do not add new architecture before locked evaluator, trace logging, and promotion gates stay green.
6. Prefer better topology over more layers until depth sweep proves otherwise.
```

## Later Domain Graphs

| Rank | Task | Status | Next Action |
| ---: | --- | --- | --- |
| 1 | Market value graph | Later | Build daily market snapshot JSONL only after topology trace/evaluator loop is stable. |
| 2 | Agent memory graph | Later | Track tasks, decisions, failures, patches, files, tests, and dependencies. |
| 3 | Knowledge provenance graph | Later | Model claims, sources, evidence, contradictions, supports/refutes edges. |
| 4 | Supply chain graph | Later | Acquire data and model companies/products/regions/dependencies. |
| 5 | Pain / manual work graph | Later | Model workflows, friction, repetition, handoffs, and automation value. |
