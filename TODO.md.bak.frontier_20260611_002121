# TODO — Learned Topology Transformer

## Current State

```text
Project = neurosymbolic/math_transformer
Status = pruning-first topology result is now quantified; learned K=4 is the current quality leader by a fragile +1 sample over hand K=4
Champion scorer = runs/checkpoints/topology_scorer.champion.pt
Champion metadata = runs/checkpoints/topology_scorer.champion.json
Current accepted quality proof = corrected fixed-K sweep with dense, hand K={2,3,4,5,6,8,12,16}, learned K={2,3,4,5,6,8,12,16}
Current quality leader = learned topology K=4, 1774/1786 route correct
Formal baseline = hand middle_preserving_topk K=4, 1773/1786 route correct
Main bottleneck = convert the +1 learned K=4 quality edge into defensible mechanism evidence and repeated-run speed/runtime proof
```

Canonical objective:

```text
Q_sparse / Q_dense_or_hand >= 0.95
T_sparse < T_dense_or_hand
J = Q - λT - γM
```

Promotion rule:

```text
Candidate promotes only through the promotion gate and only if:
route_acc_candidate >= route_acc_champion_or_required_baseline
generic_acc_candidate >= generic_acc_champion_or_required_baseline
affine_acc_candidate does not create an unacceptable Pareto regression
benchmark quality_ok=True
benchmark speed_ok=True
single-run strict_speed_ok is diagnostic only
repeated locked speed distribution passes median/p25/p75/pass-rate gate
```

Current quality evidence:

```text
Validation set = data/synthetic_hard/val.jsonl
n = 1786

Dense full                         = 1752/1786, route_acc=0.980963, generic=225/259, affine=296/296
Hand middle_preserving_topk K=4    = 1773/1786, route_acc=0.992721, generic=252/259, affine=290/296
Hand middle_preserving_topk K=5    = 1773/1786, route_acc=0.992721, generic=247/259, affine=295/296
Hand middle_preserving_topk K=8    = 1766/1786, route_acc=0.988802, generic=239/259, affine=296/296
Learned topology K=4               = 1774/1786, route_acc=0.993281, generic=253/259, affine=290/296
Learned topology K=5               = 1772/1786, route_acc=0.992161, generic=247/259, affine=294/296
Learned topology K=8               = 1767/1786, route_acc=0.989362, generic=240/259, affine=296/296

Pruning effect: hand K=4 - dense = +21 samples.
Learned selection effect at best K: learned K=4 - hand K=4 = +1 sample.
Conclusion: pruning / structural denoising remains the dominant effect; learned edge selection has a real but fragile +1 net sample at K=4.
```

Current edge-level evidence:

```text
Artifact = runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl
Records = 9 paired disagreements
learned_win = 5
learned_loss = 4
net = +1
First observed learned win: sample_id=60, target=affine_expert, hand_pred=matmul_expert, learned_pred=affine_expert
```

## Status Meanings

| Status  | Meaning                                                                     |
| ---     | ---                                                                         |
| Done    | Implemented and accepted into the current baseline.                         |
| Partial | Exists, but still needs proof, cleanup, integration, or broader validation. |
| Pending | Not implemented or not yet demonstrated.                                    |
| Later   | Useful, but below current proof/engineering path.                           |

## Priority Roadmap

| Rank | Work item                                   | Status  | Evidence / Current State                                                                                                                        | Next Action                                                                                          |
| ---: | ---                                         | ---     | ---                                                                                                                                             | ---                                                                                                  |
|    1 | Champion scorer promotion path              | Done    | `src/promote_topology_scorer.py`, `scripts/promote_topology_scorer.sh`, and champion metadata exist.                                            | Keep champion immutable except through promotion gate.                                               |
|    2 | Wire champion as default scorer             | Done    | Quality, benchmark, sweep, topology eval, and failure export defaults use champion checkpoint.                                                  | Keep training outputs separate so experiments cannot overwrite champion.                             |
|    3 | Fixed-K sweep summarizer                    | Done    | `src/topology_k_sweep.py` emits `runs/diagnostics/topology_k_sweep_summary.{json,csv}` with dense, hand, learned rows and paired wins/losses.   | Treat this as the first command before topology promotion discussion.                                |
|    4 | Preserve hand K=4 formal baseline           | Done    | Corrected sweep uses `fixed_k=k` for hand rows; hand K=4 = 1773/1786.                                                                           | Keep hand K=4 in every quality report until a policy dominates it on quality/runtime Pareto.         |
|    5 | Learned same-K edge-selection proof         | Partial | Learned K=4 beats hand K=4 by +1 net sample; paired flips are 5 wins and 4 losses.                                                              | Analyze edge deltas; do not overclaim beyond tiny K=4 selection effect.                              |
|    6 | Edge-level topology evidence                | Partial | `src/export_topology_edge_deltas.py` writes learned K=4 vs hand K=4 paired edge deltas with node labels, removed/extra edges, and scores.       | Add aggregate edge-delta summaries by outcome, relation, node type, and expert class.                |
|    7 | Lock evaluator and benchmark protocol       | Done    | `configs/learned_topology_locked_protocol.json`, `scripts/run_locked_topology_protocol.sh`, structured artifacts, and regression checker exist. | Add repeated-run aggregation; stop treating single-run strict speed as final proof.                  |
|    8 | Route-first/generic-aware runtime selection | Done    | Runtime checkpoint selection prioritizes route accuracy, then generic expert accuracy.                                                          | Add affine-regression awareness after K=4 result.                                                    |
|    9 | Replay/failure learning loop                | Partial | Trace replay and replay weighting exist.                                                                                                        | Replay should target 4 learned K=4 losses without erasing 5 wins; compare to hand K=4.               |
|   10 | Hard validation expansion                   | Partial | Synthetic hard validation exists; generic expert remains the main moving class.                                                                 | Add deeper/generic-stress/held-out composites and report by-expert breakdown.                        |
|   11 | Prepared topology cache / baked sparse IR   | Partial | Cache exists; cold prepare is real but not static-loop cost.                                                                                    | Split cold/cache-hit/static/growing regimes and bake topology artifacts for fixed hashes.            |
|   12 | Execution bottleneck profiling              | Partial | Profiling hooks and benchmark bucket reports exist.                                                                                             | Add repeated locked benchmark aggregation with median/p25/p75/pass-rate.                             |
|   13 | Sparse attention/runtime stabilization      | Partial | Token sparse, block sparse, Triton, and prepared paths exist; speed win is config-dependent and noisy.                                          | Optimize only after K/Pareto baseline is settled.                                                    |
|   14 | Large-`T` scaling proof                     | Pending | Current quality evidence is at validation scale; large-T sweep is incomplete.                                                                   | Benchmark `T=2048,4096,8192,16384` with static/preloaded and cold regimes reported separately.       |
|   15 | Block sparse path / memory locality         | Partial | Block sparse modules/tests exist.                                                                                                               | Compare block sparse vs token sparse at `T >= 4096`, using K=4 and learned candidates.               |
|   16 | Dynamic-K controller                        | Pending | Fixed-K sweep suggests best K and affine/generic tradeoff are class-dependent.                                                                  | Learn or script per-expression/per-node K to retain generic gains while avoiding affine regressions. |
|   17 | Depth sweep with synced topology            | Pending | Multi-layer path exists; no canonical `L ∈ {1,2,4,8}` report.                                                                                   | Measure `J(L)=Q-λT-γM` only after K policy is settled.                                               |
|   18 | Feature registry and feature gates          | Pending | Feature schema fixed at 10 edge features.                                                                                                       | Add registry/gates before scorer shape changes.                                                      |
|   19 | Node selection / active-node routing        | Pending | Current system uses fixed nodes plus edges.                                                                                                     | Defer until K/pruning and edge-level evidence are stable.                                            |
|   20 | Shape/type validity model                   | Partial | Shape/env handling exists.                                                                                                                      | Continue after topology policy proof.                                                                |
|   21 | Symbolic math embedding model               | Partial | `MathEmbedder` exists without contrastive/equivalence objective.                                                                                | Train after topology evidence clarifies useful structure.                                            |
|   22 | Algebraic canonicalizer                     | Partial | Parser/normalizer flow exists.                                                                                                                  | Expand coverage after generic-stress validation.                                                     |
|   23 | Code/AST dataset path                       | Pending | No code-IR dataset path in this module.                                                                                                         | Later.                                                                                               |
|   24 | Program dependency graph                    | Pending | Not implemented in this module.                                                                                                                 | Later.                                                                                               |
|   25 | Edit/action head                            | Pending | Current system selects/reroutes edges; it does not mutate graph/code nodes as actions.                                                          | Later.                                                                                               |

## Immediate Pending Tasks

| Priority | Task                                       | Status  | Target / Acceptance                                                                                                                                              |
| ---:     | ---                                        | ---     | ---                                                                                                                                                              |
| P0       | Fixed-K sweep summarizer                   | Done    | `python -m src.topology_k_sweep` emits CSV/JSON for dense, hand K sweep, learned K sweep, by-expert accuracy, correct counts, and paired wins/losses.            |
| P0       | Correct hand K semantics                   | Done    | Hand rows run as `k_values=[k]` and `fixed_k=k`; no shared `fixed_k=max(K)` contamination.                                                                       |
| P0       | Paired wins/losses semantics               | Done    | `wins_vs_*` and `losses_vs_*` count per-sample flips, while `correct_delta_vs_*` remains net difference.                                                         |
| P0       | Learned K=4 vs hand K=4 edge-delta export  | Done    | `runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl` contains 9 paired flip records with node labels, hand/learned edges, removed/extra edges, and scores. |
| P0       | Edge-delta aggregate analyzer              | Pending | Summarize the 5 wins and 4 losses by target expert, prediction flip, edge relation, removed/extra edge type, and recurring node pattern.                         |
| P0       | Repeated locked speed aggregator           | Pending | Run N locked artifacts and gate on median/p25/p75/pass-rate instead of single-run strict speed.                                                                  |
| P0       | Promote/benchmark reports include hand K=4 | Pending | Promotion and benchmark summaries include dense, hand K=4, learned K=4, champion policy, generic/affine accuracy, and paired wins/losses.                        |
| P1       | Replay/selection from K=4 losses           | Pending | Target the 4 learned K=4 losses while preserving the 5 wins; compare against hand K=4.                                                                           |
| P1       | Expand generic-stress validation set       | Partial | Add deeper/longer/held-out/generic-stress examples and report route/expert breakdown.                                                                            |
| P1       | Profile timing regimes                     | Partial | Separate cold prepare, cache hit, static preloaded block, and growing-node miss regimes.                                                                         |
| P1       | Bake prepared topology cache               | Partial | Reused graphs/snapshots bypass scorer/topology rebuild and load neighbor tables directly.                                                                        |
| P2       | Large-`T` scaling benchmark                | Pending | Show static/preloaded sparse advantage at `T >= 4096`, with cold/dynamic costs separately reported.                                                              |
| P2       | Kernel/execution specialization pass       | Partial | Stable prepared-block speed win beyond timing noise; do not change topology behavior.                                                                            |
| P2       | Dynamic-K controller                       | Pending | Per-node/expression K preserves generic gains while avoiding affine regressions.                                                                                 |
| P3       | Feature registry/gating                    | Pending | Features can be added/deleted by measured utility without silently breaking checkpoints.                                                                         |
| P3       | Node selection                             | Pending | Active node set improves quality/runtime objective over fixed-node baseline.                                                                                     |
| P3       | Code/AST dataset path                      | Pending | Code sample JSONL supports nodes, node features, known edges, labels, evaluator feedback.                                                                        |
| P3       | Edit/action head                           | Pending | Model proposes graph/code edits verified by tests/typechecker/evaluator.                                                                                         |

## Useful Commands

```bash
# Corrected fixed-K topology quality sweep
python -m src.topology_k_sweep \
  --examples data/synthetic_hard/val.jsonl \
  --checkpoint runs/checkpoints/synthetic_hard_dense.pt \
  --k-values 2,3,4,5,6,8,12,16 \
  --learned-scorer-checkpoint runs/checkpoints/topology_scorer.champion.pt \
  --quality-device auto \
  --json-out runs/diagnostics/topology_k_sweep_summary.json \
  --csv-out runs/diagnostics/topology_k_sweep_summary.csv

# Edge-level paired flips: learned K=4 vs hand K=4
python -m src.export_topology_edge_deltas \
  --examples data/synthetic_hard/val.jsonl \
  --checkpoint runs/checkpoints/synthetic_hard_dense.pt \
  --learned-scorer-checkpoint runs/checkpoints/topology_scorer.champion.pt \
  --k 4 \
  --quality-device auto \
  --output runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl

# Main historical benchmark, champion K=8 vs hand K=16
LEARNED_K=8 \
HAND_K=16 \
scripts/benchmark_learned_topology.sh

# Champion regression proof with structured artifact audit
scripts/check_topology_champion_regression.sh

# Locked evaluator/benchmark protocol using committed settings
scripts/run_locked_topology_protocol.sh

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
4. Do not claim sparse is faster unless repeated locked wall-clock artifacts beat dense/hand on distributional metrics.
5. Do not add new architecture before locked evaluator, trace logging, promotion gates, K-sweep baselines, and edge-delta artifacts stay green.
6. Prefer K/pruning evidence over more layers until K policy is settled.
7. Every topology claim must compare against hand K=4 and report generic/affine tradeoff.
8. Every learned-topology mechanism claim must cite edge-level deltas, not just aggregate route accuracy.
9. Learned K=4 may be described as quality leader, but only as a fragile +1 sample improvement until repeated validation confirms it.
```

## Later Domain Graphs

| Rank | Task                       | Status | Next Action                                                                           |
| ---: | ---                        | ---    | ---                                                                                   |
|    1 | Market value graph         | Later  | Build daily market snapshot JSONL only after topology trace/evaluator loop is stable. |
|    2 | Agent memory graph         | Later  | Track tasks, decisions, failures, patches, files, tests, and dependencies.            |
|    3 | Knowledge provenance graph | Later  | Model claims, sources, evidence, contradictions, supports/refutes edges.              |
|    4 | Supply chain graph         | Later  | Acquire data and model companies/products/regions/dependencies.                       |
|    5 | Pain / manual work graph   | Later  | Model workflows, friction, repetition, handoffs, and automation value.                |
