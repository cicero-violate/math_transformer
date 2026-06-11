# TODO — Recurrent Frontier Graph Transformer

## Current State

```text
Project = neurosymbolic/math_transformer
Name = Recurrent Frontier Graph Transformer
Former subsystem name = Learned Topology Transformer / Math-Routed Sparse Transformer
Status = v22 QuantaleSpec / QuantaleFrontierPlanner is implemented; pruning-first topology result remains quantified; learned K=4 is the current quality leader by a fragile +1 sample over hand K=4
Champion scorer = runs/checkpoints/topology_scorer.champion.pt
Champion metadata = runs/checkpoints/topology_scorer.champion.json
Current accepted quality proof = corrected fixed-K sweep with dense, hand K={2,3,4,5,6,8,12,16}, learned K={2,3,4,5,6,8,12,16}
Current quality leader = learned topology K=4, 1774/1786 route correct
Formal baseline = hand middle_preserving_topk K=4, 1773/1786 route correct
Main bottleneck = convert CUDA/locked-protocol wins into repeated-run speed proof, then compile large teacher checkpoint weights into a derived graph-native sparse-student candidate graph
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
repeated locked speed distribution passes median/p25/p75/pass-rate gate; current first K4-vs-K4 repeated run fails speed despite quality win
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

## Architecture Delta — Recurrent Frontier Graph Attention

```text
New target = bounded active compute over an unbounded persistent graph.
Current system = fixed nodes + selected sparse edges inside one loaded graph snapshot.
New delta = recurrent frontier expansion over a stored world graph, with optional persistent graph writeback.

G_world = persistent mega graph on disk/RAM.
G_t = active graph slice loaded for step t.
∂G_t = frontier/boundary of active graph with unloaded neighbors.
F_θ = sparse attention + MLP reasoning kernel.
ω_G = persistent graph memory weights: edge strength, node utility, confidence, staleness.

H_t = F_θ^L(X_t, G_t)
C_t = BoundaryCandidates(∂G_t, G_world)
G_{t+1} = KeepTopB(G_t ∪ TopK(Score(H_t, C_t)))
G_world^{t+1} = Accept(G_world^t ⊕ ΔG_t)

Effective hops ≈ L * T_outer.
Active compute ≈ O(T_outer * L * |G_t| * K * d), with |G_t| bounded.
```

Interpretation:

```text
K = fanout width per sparse attention layer.
L = local sparse-attention depth inside the active graph.
T_outer = recurrent graph-walk / frontier expansion steps.
B = active graph node/edge budget.

The project should not jump directly from K-sweeps to larger dense context.
The next architectural unlock is recurrent sparse graph traversal:
load small graph → reason → score frontier → lazy-load expansion → prune to budget → repeat.
```


## Latest Evidence — 2026-06-11 CUDA / Locked Protocol

```text
CUDA focused graph/frontier suite: 116 passed before v22 quantale landing.
CUDA full suite after v22 quantale landing: 429 passed.
CUDA smoke quality on synthetic_tiny: dense/full 8/15 = 0.533333; topology K=4 9/15 = 0.600000.
CUDA K-sweep artifacts written to runs/cuda_smoke/topology_k_sweep.{json,csv}.
Locked protocol champion_k8_vs_hand_k16_locked passed quality and strict speed.
Repeated locked protocol first K4-vs-K4 run failed speed: learned quality wins but learned block_ms is slower.
Repeated protocol harness now records per-run failures, prints total_runs/failed_runs/succeeded, and writes JSON/CSV summaries even when runs fail.
Depth sweep CLI repair is complete: dense k=None prints as full, expert keys use generic_expert / affine_expert, and single-checkpoint L-sweep limitation is recorded.
```

Locked protocol pass:

```text
dense/full:        route_acc=0.980963, correct=1752/1786
hand_topology K16: route_acc=0.982083, correct=1754/1786, block_ms=0.929171
learned_topology K8: route_acc=0.987122, correct=1763/1786, block_ms=0.916944
speedup=1.013335
quality_ok=True speed_ok=True strict_speed_ok=True
```

Repeated-run blocker:

```text
hand_topology K4:    route_acc=0.985442, block_ms=0.985052
learned_topology K4: route_acc=0.993281, block_ms=1.054465
speedup=0.934172
quality_ok=True speed_ok=False strict_speed_ok=False
diagnosis=learned_attention_not_faster_than_hand
```

Interpretation:

```text
Single locked protocol is green.
Repeated speed proof is not green.
Repeated aggregation harness is green, but distributional speed acceptance is still not green.
Do not claim final speed proof until repeated locked distribution metrics pass.
```


## Architecture Direction — Checkpoint-to-Sparse-Student Distillation

```text
New direction = use a large pretrained model as source structure, then train a new graph-native sparse student.
Non-goal = query/run the large model at runtime or assume active-parameter count makes the original model fit in limited VRAM.
Goal = convert teacher structure/evidence into a sparse model with more efficient connections and deleted redundant edges.

Teacher = large dense/MoE checkpoint or teacher behavior artifact.
Student = Recurrent Frontier Graph Transformer with learned sparse topology.
G_0 = candidate graph extracted from teacher tensors, teacher traces, distillation labels, or structural priors.
A = sparse student adjacency after edge scoring/pruning.
W = student edge/node weights.
φ = trainable sparse student parameters.

Y_teacher = f_teacher(X)
Y_student = f_graph(X; A, W, φ)

Train objective:
L = KL(p_teacher || p_student) + α L_task + λ |A| + γ M + β T

Runtime objective:
Q_student ≈ Q_teacher_or_baseline
M_student <= local memory budget
T_student < dense/hand/runtime baseline on repeated locked artifacts
```

Interpretation:

```text
This is graph-native distillation, not sparse execution of the original teacher.
The teacher supplies candidate structure and supervision; the deployed artifact is a new sparse student.
Edges are deleted only by measured utility, not by assumption.
The same proof gates still apply: quality, runtime distribution, memory, paired regressions, and repeatability.
```

Checkpoint-weight path:

```text
New concrete v23 direction = compile a large Qwen / dense / MoE checkpoint into a derived graph artifact.
Non-goal = prompt/query extraction from Qwen.
Non-goal = store raw transformer weights as graph records.
Goal = safetensors checkpoint -> tensor manifest -> block/head/expert graph -> G_0 -> sparse student candidate topology.

Teacher checkpoint tensors are source evidence.
Graph records store derived structure/statistics only: node IDs, edge IDs, tensor provenance, block ranges, scores, confidence, utility, staleness, and optional cached embeddings.

Parameter-level graph is rejected.
First target = block-level + head-level + expert-level graph compiler.
```

Weight-derived edge score:

```text
Given W ∈ R^{d_out × d_in}, split W into blocks W_ab.

s_ab = ||W_ab||_F / sqrt(|W_ab|)
E_K = {(a,b): b ∈ TopK_b(s_ab)}
```


## Architecture Direction — Graph-Native Bounded Kleene Closure

```text
New direction = implement bounded Kleene closure inside this math_transformer / graph-transformer system.
Non-goal = use any external orchestration module or compute full A* over G_world at runtime.
Goal = use bounded closure over the active graph plus frontier candidates to plan multi-hop expansion, preserve closure-critical edges, and support safe pruning.

A = one-hop active/candidate adjacency or transition matrix.
I = zero-hop identity.
H = maximum closure horizon.
A^{<=H} = I ∨ A ∨ A^2 ∨ ... ∨ A^H.

Boolean closure = reachability / verifier feasibility.
Cost closure = lowest-cost bounded path.
Utility closure = highest-utility bounded path.

Input = G_t, C_t, H_t, edge scores, verifier mask, budget B, horizon H.
Output = closure-aware frontier choices, keep/archive edge decisions, and stop/continue signal.
```

Interpretation:

```text
Sparse graph transformer = learned local reasoning kernel.
Bounded Kleene closure = internal graph planning primitive over G_t ∪ C_t.
Use A^{<=H}, not literal infinite A*.
Use closure to reason about paths and redundant edges, but keep the deployed implementation in this repository.
```


## Architecture Direction — Quantale-Native Closure Planner

```text
New direction = replace ad-hoc boolean/cost/utility closure branches with one internal QuantaleSpec abstraction.
Non-goal = MCTS-first search or external orchestration module wiring.
Goal = deterministic compositional path planning over G_t ∪ C_t with explicit join/compose/better/valid semantics.

QuantaleSpec fields:
  name
  zero
  one
  join(a,b)
  compose(a,b)
  better(a,b)
  valid(x)

Boolean = OR / AND reachability.
Cost = min / + cheapest bounded path.
Utility = max / + highest-utility bounded path.
```

Interpretation:

```text
Quantale planner is the next core abstraction.
Beam/MCTS can come later as approximate search over quantale traces.
Cost and utility bugs should be fixed by algebra-specific better() and valid() definitions.
v22 implemented this abstraction in `src/graph_closure.py` and `src/frontier_planner.py`.
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
|    7 | Lock evaluator and benchmark protocol       | Done    | CUDA locked protocol `champion_k8_vs_hand_k16_locked` passes quality and strict speed: learned K8 route_acc=0.987122, speedup=1.013335 vs hand K16. | Keep locked artifact reproducible; do not treat it as repeated-run proof.                            |
|    8 | Route-first/generic-aware runtime selection | Done    | Runtime checkpoint selection prioritizes route accuracy, then generic expert accuracy.                                                          | Add affine-regression awareness after K=4 result.                                                    |
|    9 | Replay/failure learning loop                | Partial | Trace replay and replay weighting exist.                                                                                                        | Replay should target 4 learned K=4 losses without erasing 5 wins; compare to hand K=4.               |
|   10 | Hard validation expansion                   | Partial | Synthetic hard validation exists; generic expert remains the main moving class.                                                                 | Add deeper/generic-stress/held-out composites and report by-expert breakdown.                        |
|   11 | Prepared topology cache / baked sparse IR   | Partial | Cache exists; cold prepare is real but not static-loop cost.                                                                                    | Split cold/cache-hit/static/growing regimes and bake topology artifacts for fixed hashes.            |
|   12 | Execution bottleneck profiling              | Partial | Bucket reports exist; repeated K4-vs-K4 run shows learned quality win but speed failure (`speedup=0.934172`, diagnosis learned_attention_not_faster_than_hand); repeated harness now emits summaries even on failures. | Prioritize distributional speed fixes and bucket-level timing attribution.                            |
|   13 | Sparse attention/runtime stabilization      | Partial | Token sparse, block sparse, Triton, and prepared paths exist; speed win is config-dependent and noisy.                                          | Optimize only after K/Pareto baseline is settled.                                                    |
|   14 | Large-`T` scaling proof                     | Pending | Current quality evidence is at validation scale; large-T sweep is incomplete.                                                                   | Benchmark `T=2048,4096,8192,16384` with static/preloaded and cold regimes reported separately.       |
|   15 | Block sparse path / memory locality         | Partial | Block sparse modules/tests exist.                                                                                                               | Compare block sparse vs token sparse at `T >= 4096`, using K=4 and learned candidates.               |
|   16 | Dynamic-K controller                        | Pending | Fixed-K sweep suggests best K and affine/generic tradeoff are class-dependent.                                                                  | Learn or script per-expression/per-node K to retain generic gains while avoiding affine regressions. |
|   17 | Depth sweep with synced topology            | Done    | Formatter handles dense `k=None` as `full`; expert keys use `generic_expert` / `affine_expert`; multi-L sweeps record `single_checkpoint_l_sweep` limitation. | Do not use one strict L=2 checkpoint as proof for L={1,2,4,8}; require per-L checkpoints or explicit partial-load experiment. |
|   18 | Feature registry and feature gates          | Pending | Feature schema fixed at 10 edge features.                                                                                                       | Add registry/gates before scorer shape changes.                                                      |
|   19 | Recurrent frontier graph attention          | Pending | Current system uses fixed nodes plus selected edges; no `G_world`, `G_t`, frontier, or recurrent expansion loop exists.                         | Implement `G_world → G_t → F_θ^L → ∂G_t → G_{t+1}` with bounded `KeepTopB` active graph budget.       |
|   20 | Graph-native bounded Kleene closure         | Done    | `quantale_closure` and `quantale_bounded_closure` provide one generic bounded closure path over Boolean, cost, and utility specs.                | Keep no-full-`G_world` runtime gate and compare planner outputs under identical K,L,B,H.              |
|   21 | Quantale-native frontier planner            | Done    | `QuantaleFrontierPlanner` and `score_candidates_quantale(...)` exist; planner reports `closure_path_count` and `closure_compute_cost`.             | Compare vs one-hop TopK / existing closure under same K,L,B,H,T_outer.                                |
|   22 | Closure-preserving edge deletion gate       | Pending | No closure-preservation test exists before deleting/archive candidate edges.                                                                     | Before deleting an edge, compare important bounded closure under `A^{<=H}` vs `(A\e)^{<=H}`.       |
|   23 | QuantaleSpec closure algebra                | Done    | `QuantaleSpec` defines name, zero, one, join, compose, reduce, better, and valid; Boolean/cost/utility are built-in specs via `_SPEC_MAP`.       | Add more regression/property tests before changing planner semantics.                                 |
|   23 | Shape/type validity model                   | Partial | Shape/env handling exists.                                                                                                                      | Continue after topology policy proof.                                                                |
|   24 | Symbolic math embedding model               | Partial | `MathEmbedder` exists without contrastive/equivalence objective.                                                                                | Train after topology evidence clarifies useful structure.                                            |
|   25 | Algebraic canonicalizer                     | Partial | Parser/normalizer flow exists.                                                                                                                  | Expand coverage after generic-stress validation.                                                     |
|   26 | Code/AST/IR dataset path                    | Pending | No code-IR dataset path in this module.                                                                                                         | Add JSONL for code nodes, AST/CFG/SSA/call/type edges, node features, labels, traces, and feedback.  |
|   27 | Program dependency / world graph compiler   | Pending | Not implemented in this module.                                                                                                                 | Compile code into `G_world = AST ∪ CFG ∪ SSA ∪ CALL ∪ TYPE ∪ TEST`, then retrieve active slices.      |
|   28 | Graph writeback / edit-action head          | Pending | Current system selects/reroutes edges; it does not persistently mutate nodes/edges or write accepted `ΔG` back to a world graph.                 | Add propose/verify/accept loop for `ΔV+`, `ΔV-`, `ΔE+`, `ΔE-`, and `Δω`; prefer archive over delete. |
|   29 | Checkpoint-to-sparse-student distillation   | Pending | No path yet converts a large teacher checkpoint into a graph-native sparse student; current proof is fixed-K topology plus quantale-frontier infrastructure. | Implement converter/training plan: teacher checkpoint weights → derived candidate graph `G_0` → edge scoring → TopK pruning → sparse student distillation. |
|   30 | Qwen checkpoint weight graph compiler       | Pending | TODO now explicitly selects the weight path, not prompt/query extraction; no `safetensors -> tensor/block/head/expert graph` compiler exists yet. | Add compiler for tensor manifest, typed Qwen/MoE tensor parsing, block-energy edges, and graph export artifacts. |
|   31 | Redundant-edge deletion proof for student   | Pending | Pruning effect is proven in current topology sweep, but not yet for teacher-informed sparse-student distillation.                                | Add measured edge utility, sparsity penalty, paired regressions, and quality/runtime/memory acceptance gates before deleting edges permanently. |
|   32 | Continual multi-teacher graph ingestion     | Pending | Single-teacher checkpoint graph compilation exists; no evidence-gated teacher registry, graph pool, conflict handling, or bounded multi-teacher student graph exists. | Add teacher registry + graph pool; accept edges only through quality/memory/runtime/conflict gates while keeping runtime adjacency bounded. |

## Immediate Pending Tasks

| Priority | Task                                       | Status  | Target / Acceptance                                                                                                                                              |
| ---:     | ---                                        | ---     | ---                                                                                                                                                              |
| P0       | Fixed-K sweep summarizer                   | Done    | `python -m src.topology_k_sweep` emits CSV/JSON for dense, hand K sweep, learned K sweep, by-expert accuracy, correct counts, and paired wins/losses.            |
| P0       | CUDA full-suite proof                      | Done    | `.venv-cuda/bin/python -m pytest -q` passes after v22 quantale landing: 429 passed. CUDA smoke K-sweep artifacts exist in `runs/cuda_smoke/`.                    |
| P0       | Correct hand K semantics                   | Done    | Hand rows run as `k_values=[k]` and `fixed_k=k`; no shared `fixed_k=max(K)` contamination.                                                                       |
| P0       | Paired wins/losses semantics               | Done    | `wins_vs_*` and `losses_vs_*` count per-sample flips, while `correct_delta_vs_*` remains net difference.                                                         |
| P0       | Learned K=4 vs hand K=4 edge-delta export  | Done    | `runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl` contains 9 paired flip records with node labels, hand/learned edges, removed/extra edges, and scores. |
| P0       | Edge-delta aggregate analyzer              | Pending | Summarize the 5 wins and 4 losses by target expert, prediction flip, edge relation, removed/extra edge type, and recurring node pattern.                         |
| P0       | Repeated locked protocol aggregator        | Done    | Per-run failures no longer abort the loop; total_runs/failed_runs/succeeded are printed; JSON/CSV summary artifacts are written even when runs fail.              |
| P0       | Repeated locked speed proof                | Pending | First repeated K4-vs-K4 run failed speed (`speedup=0.934172`) despite quality win; distributional speed gate remains open.                                         |
| P0       | Promote/benchmark reports include hand K=4 | Pending | Promotion and benchmark summaries include dense, hand K=4, learned K=4, champion policy, generic/affine accuracy, and paired wins/losses.                        |
| P1       | Replay/selection from K=4 losses           | Pending | Target the 4 learned K=4 losses while preserving the 5 wins; compare against hand K=4.                                                                           |
| P1       | Expand generic-stress validation set       | Partial | Add deeper/longer/held-out/generic-stress examples and report route/expert breakdown.                                                                            |
| P1       | Profile timing regimes                     | Partial | Separate cold prepare, cache hit, static preloaded block, and growing-node miss regimes.                                                                         |
| P1       | Bake prepared topology cache               | Partial | Reused graphs/snapshots bypass scorer/topology rebuild and load neighbor tables directly.                                                                        |
| P1       | Graph-native closure design / API          | Done    | `src/graph_closure.py` exposes generic quantale bounded closure over active/candidate slices; no external orchestration module dependency is required.             |
| P1       | QuantaleSpec closure prototype             | Done    | Boolean/cost/utility closure are unified through one internal `QuantaleSpec` kernel with explicit `better()` and `valid()` semantics.                             |
| P2       | Large-`T` scaling benchmark                | Pending | Show static/preloaded sparse advantage at `T >= 4096`, with cold/dynamic costs separately reported.                                                              |
| P2       | Kernel/execution specialization pass       | Partial | Stable prepared-block speed win beyond timing noise; do not change topology behavior.                                                                            |
| P2       | Dynamic-K controller                       | Pending | Per-node/expression K preserves generic gains while avoiding affine regressions.                                                                                 |
| P3       | Feature registry/gating                    | Pending | Features can be added/deleted by measured utility without silently breaking checkpoints.                                                                         |
| P3       | Recurrent frontier prototype               | Pending | `T_outer ∈ {1,2,3}` improves `J=Q-λT-γM` over single fixed-graph pass while keeping `|G_t| ≤ B`.                                              |
| P3       | Code/AST/IR dataset path                   | Pending | Code sample JSONL supports nodes, node features, AST/CFG/SSA/call/type edges, labels, traces, and evaluator feedback.                                           |
| P3       | Graph writeback / edit-action head         | Pending | Model proposes `ΔG` graph/code edits; verifier accepts only changes supported by tests/typechecker/evaluator/trace evidence.                                     |
| P3       | Qwen checkpoint weight graph compiler      | Pending | Define and implement the non-prompt path: Qwen safetensors/checkpoint weights → tensor manifest → block/head/expert graph → derived `G_0` artifacts.                |
| P3       | Sparse-student distillation design note    | Pending | Define the non-runtime-query path: teacher checkpoint weights / behavior artifact → candidate sparse graph → edge scoring/pruning → student training/eval under memory budget. |
| P3       | Student redundant-edge gate                | Pending | No edge is deleted from the student unless paired quality, repeated speed, and memory metrics improve or remain within accepted Pareto bounds.                    |
| P3       | Continual multi-teacher ingestion design   | Pending | Add teacher registry, graph pool, per-teacher provenance, conflict reports, accepted/rejected edge logs, and bounded active adjacency gates.                    |


## Recurrent Frontier Graph Attention Tasks

| Priority | Task                                      | Status  | Target / Acceptance |
| ---:     | ---                                       | ---     | --- |
| P0       | Preserve current K=4 proof gates          | Pending | Recurrent/frontier experiments must still report dense, hand K=4, learned K=4, generic/affine, paired wins/losses, and repeated speed distribution. |
| P0       | Depth sweep CLI repair                    | Done    | Dense `k=None` formatter renders `full`; expert keys match `generic_expert` / `affine_expert`; single-checkpoint L-sweep limitation is recorded. |
| P1       | Define graph identity schema              | Pending | Nodes and edges use hash IDs; labels are metadata; raw facts are append-only; derived graph weights are mutable/versioned. |
| P1       | Split `G_world` from `G_active`           | Pending | Persistent graph storage and active GPU graph slice are represented separately; active slice has explicit node/edge budget `B`. |
| P1       | Frontier extraction                       | Pending | Compute `∂G_t` boundary nodes/edges from active graph and expose unloaded candidate neighbors from `G_world`. |
| P1       | Boundary candidate scorer                 | Pending | Score candidate expansions using hidden states `H_t`, node/edge features, relation type, and persistent graph weights `ω_G`. |
| P1       | Quantale frontier planner                 | Done    | `QuantaleFrontierPlanner` exists over `G_t ∪ C_t`; next step is comparison against one-hop TopK and current closure planner under same K,L,B,H,T_outer. |
| P1       | Closure-critical edge preservation        | Pending | Mark edges/nodes required by important bounded reachability or high-utility paths so `KeepTopB` does not prune them accidentally. |
| P1       | KeepTopB active graph budget              | Pending | After expansion, prune active graph to budget by utility while preserving required anchors, self edges, verifier-required nodes, and closure-critical paths. |
| P1       | Recurrent sparse pass benchmark           | Pending | Compare `T_outer={1,2,3}` with fixed `L=2`, `K=4/8`, one-hop TopK vs bounded-closure planner, and report `Q`, runtime, memory, expansion count, and stale/noisy edge rate. |
| P2       | Persistent graph memory weights           | Pending | Store `ω_ij`, node utility, confidence, staleness, provenance, and last-updated version in graph records; do not store transformer weights in graph. |
| P2       | Graph writeback verifier                  | Pending | Accept/reject `ΔV+`, `ΔV-`, `ΔE+`, `ΔE-`, `Δω`; archive/tombstone instead of destructive delete; support rollback. |
| P3       | Code world graph compiler                 | Pending | Compile code into AST/CFG/SSA/CALL/TYPE/TEST graph and run recurrent frontier prototype over code tasks. |
| P3       | Market event graph compiler               | Later   | Compile market streams into time/asset/factor/event/regime/risk graph after code/math frontier loop is stable. |

## Checkpoint Weight Graph Compiler Tasks

| Priority | Task                                      | Status  | Target / Acceptance |
| ---:     | ---                                       | ---     | --- |
| P0       | Keep raw-weight storage forbidden         | Done    | `raw_weight_payload_in_graph=false` is enforced in `WeightGraphManifest.as_dict()`; Gate 2 stub detects violations at load time; 104 tests verify no raw tensor payloads in any graph record. |
| P0       | Tensor manifest compiler                  | Done    | `build_tensor_manifest_from_directory` reads safetensors headers only (no data loaded); emits `TensorSpec` with name, shard, shape, dtype, layer, module, projection, expert/head metadata; supports single-file and sharded index-based checkpoints. |
| P0       | Typed Qwen/MoE tensor parser              | Done    | `parse_qwen_tensor_name` identifies self_attn q/k/v/o projections, mlp gate/up/down, MoE router (gate), MoE expert projections, shared_expert, embed_tokens, lm_head, final_norm, layernorm variants; unknown tensors logged with parse_ok=False. |
| P1       | Block-energy graph builder                | Done    | `compute_block_scores` partitions 2D tensors into blocks, scores by normalized Frobenius norm, returns TopK (block_row, block_col, score) with deterministic tie-break; block nodes and block-flow edges emitted by `_build_block_graph_for_tensor`. |
| P1       | Head/expert graph builder                 | Done    | `_build_structural_graph` emits model/layer/projection/router/expert nodes; residual_next, qk_affinity_prior, value_flow_prior, mlp_gate_flow, mlp_up_flow, router_to_expert_prior, expert_flow_prior, tensor_contains_block edges. |
| P1       | Candidate graph artifact schema           | Done    | `write_weight_graph_artifacts` writes `manifest.json`, `nodes.jsonl`, `edges.jsonl`, `stats.json`, optional `tensor_parse_failures.jsonl`; `read_weight_graph_artifacts` round-trips all fields. |
| P1       | Weight-graph smoke test                   | Done    | `tests/test_qwen_weight_graph.py` — 104 tests on in-memory safetensors fixtures verify deterministic node/edge counts, no raw payloads, valid 16-char hex hash IDs, Gate 2/3/4/5 compliance. |
| P2       | Sparse-student adapter                    | Done    | `load_weight_graph_as_world_graph` bridges G_0 artifact to WorldGraph for planner/scorer use (P1.3); `run_sparse_student_stub` is the P2.1 entry point stub; champion scorer behavior is unchanged. |
| P2       | Student deletion gate                     | Pending | Delete/archive edges only after paired quality, repeated speed, and memory metrics improve or remain within accepted Pareto bounds. |

## Continual Multi-Teacher Graph Ingestion Tasks

```text
Goal = ingest multiple teacher checkpoints over time without allowing runtime graph growth to become unbounded.
Core invariant = the offline graph pool may grow; the active student adjacency must remain bounded.
```

Pipeline:

```text
teacher checkpoint_i
  -> compiled graph G_0_i
  -> teacher registry
  -> graph pool
  -> evidence-gated edge selection
  -> bounded runtime student graph A*
```

Selection objective:

```text
G_pool,t+1 = G_pool,t ∪ G_0_i

A* = argmax_{A ⊆ G_pool} [
  Q(A)
  - λ |A|
  - γ M(A)
  - β T(A)
  - δ C(A)
]
```

Where:

```text
Q = quality
M = memory cost
T = runtime cost
C = teacher/domain conflict cost
```

| Priority | Task                                | Status  | Target / Acceptance |
| ---:     | ---                                 | ---     | --- |
| P0       | Teacher registry schema             | Pending | Record teacher ID, model/config/index hashes, domain tags, source path, compiler version, graph artifact path, reliability score, and accepted/rejected edge counts. |
| P0       | Graph pool manifest                 | Pending | Maintain `G_pool` as an offline library of teacher-derived candidate graphs; runtime student does not load every pool edge. |
| P0       | Bounded active adjacency gate        | Pending | Enforce `|A| <= configured_edge_budget` for every multi-teacher run; fail closed if the active graph exceeds budget. |
| P1       | Evidence-gated edge acceptance       | Pending | Accept teacher edges only when paired quality, memory, runtime, robustness, or closure-preservation metrics justify them. |
| P1       | Teacher conflict report             | Pending | Detect and report disagreements across teachers/domains; route by teacher/domain weights instead of treating every teacher as equally reliable. |
| P1       | Old-domain regression gate           | Pending | Require `Q_old(S_t+1) >= Q_old(S_t) - ε` before accepting a new teacher ingestion batch. |
| P2       | Accepted/rejected edge logs          | Pending | Emit `accepted_edges.jsonl`, `rejected_edges.jsonl`, and reason codes for every edge considered from a new teacher graph. |
| P2       | Multi-teacher KD mixture             | Pending | Support `sum_i ρ_i KL(p_T_i^τ || p_S^τ)` with per-domain teacher weights and top-r logit artifacts. |
| P3       | Continual ingestion harness          | Pending | Run teacher ingestion batches sequentially and report ΔQ, ΔM, ΔT, ΔR, conflict cost, active edge count, and old-domain regressions. |

Required artifacts:

```text
teacher_registry.json
graph_pool_manifest.json
teacher_ingestion_report.json
accepted_edges.jsonl
rejected_edges.jsonl
conflict_report.jsonl
old_domain_regression_report.json
```

Acceptance gates:

```text
New teacher graph improves at least one of:
  ΔQ > 0
  ΔM < 0
  ΔT < 0
  ΔR > 0

Old-domain regression remains bounded:
  Q_old(S_t+1) >= Q_old(S_t) - ε

Runtime graph remains bounded:
  |A| <= configured_edge_budget

Raw teacher weights are never stored in graph records.
Only metadata, hashes, ranges, scores, and provenance are allowed.
```

Non-negotiable rules:

```text
Do not append every teacher edge into the active student graph.
Do not allow runtime adjacency to grow without budget.
Do not accept a teacher graph without quality/memory/runtime/conflict evidence.
Do not overwrite existing student behavior without regression gates.
Do not treat all teachers as equally reliable across all domains.
```

Summary:

```text
Grow the knowledge pool.
Bound the active graph.
Distill by evidence.
```

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


## v22 CUDA / Locked Commands

```bash
# Full CUDA suite
.venv-cuda/bin/python -m pytest -q

# CUDA quality smoke
.venv-cuda/bin/python -m src.eval --quality   --examples data/examples.jsonl   --checkpoint runs/checkpoints/synthetic_tiny.pt   --quality-k 4   --quality-device cuda   --d-model 64 --n-heads 4 --n-layers 2 --d-ff 128   --topology-mode middle_preserving_topk   --fixed-k 4

# CUDA K sweep
.venv-cuda/bin/python -m src.topology_k_sweep   --examples data/examples.jsonl   --checkpoint runs/checkpoints/synthetic_tiny.pt   --k-values 4,8,16   --quality-device cuda   --d-model 64 --n-heads 4 --n-layers 2 --d-ff 128   --json-out runs/cuda_smoke/topology_k_sweep.json   --csv-out runs/cuda_smoke/topology_k_sweep.csv

# Locked protocol
scripts/run_locked_topology_protocol.sh

# Repeated locked protocol: currently exposes K4-vs-K4 speed blocker
scripts/run_repeated_locked_topology_protocol.sh
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
10. Recurrent frontier claims must compare `T_outer=1` against `T_outer>1` under the same `K`, `L`, budget `B`, and evaluator.
11. Do not claim unbounded/infinite memory without reporting active graph budget, retrieval/frontier cost, and stop condition.
12. Do not persist graph mutations without verifier evidence, provenance, versioning, and rollback path.
13. Do not store transformer weights in graph records; graph may store node/edge memory weights `ω_G`, cached embeddings, confidence, utility, and staleness.
14. Hash IDs are identity; generated labels are metadata; raw facts should be append-only and derived structures should be mutable/versioned.
15. Do not compute full Kleene closure over `G_world` at runtime; only bounded/local `A^{<=H}` over active/candidate slices is allowed.
16. Kleene/closure planner work must stay inside this `math_transformer` graph-transformer code path; do not wire to any external orchestration module.
17. Closure-aware frontier claims must compare against one-hop/heuristic TopK under the same `K`, `L`, `T_outer`, `B`, horizon `H`, and evaluator.
18. Edge deletion must preserve important bounded closure or show an accepted Pareto tradeoff in quality/runtime/memory.
19. Do not claim repeated speed proof from the single locked pass; repeated K4-vs-K4 currently fails speed.
20. Quantale planner work must be internal to `math_transformer`; do not wire to external orchestration modules.
21. MCTS/search must wait until one-hop, closure, and quantale baselines are stable under identical K,L,B,H,T_outer.
22. Depth sweep cannot claim L={1,2,4,8} from one strict L=2 checkpoint.
```

## Later Domain Graphs

| Rank | Task                       | Status | Next Action                                                                           |
| ---: | ---                        | ---    | ---                                                                                   |
|    0 | Code world graph           | Pending | Build AST/CFG/SSA/CALL/TYPE/TEST graph compiler after recurrent frontier schema lands. |
|    1 | Market value graph         | Later  | Build daily market snapshot JSONL only after topology trace/evaluator loop is stable. |
|    2 | Agent memory graph         | Later  | Track tasks, decisions, failures, patches, files, tests, and dependencies.            |
|    3 | Knowledge provenance graph | Later  | Model claims, sources, evidence, contradictions, supports/refutes edges.              |
|    4 | Supply chain graph         | Later  | Acquire data and model companies/products/regions/dependencies.                       |
|    5 | Pain / manual work graph   | Later  | Model workflows, friction, repetition, handoffs, and automation value.                |
