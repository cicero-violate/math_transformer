# Plan v21 — Recurrent Frontier Graph Transformer

## Official Name

```text
Project name: Recurrent Frontier Graph Transformer
Short name: RFGT
Core mechanism: Recurrent Frontier Graph Attention
Former name: Math-Routed Sparse Transformer / Learned Topology Transformer
```

The previous names were accurate for the implemented math-expression sparse-topology subsystem, but they are too narrow for the actual architecture direction.

The correct system is no longer only:

```text
math expression graph -> TopK topology -> sparse attention
```

The target system is:

```text
persistent world graph -> bounded active graph -> sparse graph attention -> frontier expansion -> verified writeback
```

## Thesis

```text
Sparse attention is not just a cheaper attention kernel.
Sparse attention can be a bounded graph-walking engine over a persistent graph memory.
```

Current implementation proves useful components:

```text
1. Fixed-K topology can denoise dense attention.
2. Learned K=4 is a fragile +1 validation sample over hand K=4.
3. Sparse attention kernels and cached/prepared topology paths exist.
4. Multi-layer transformer support exists, but depth over a synced topology is not yet fully benchmarked.
```

New architectural target:

```text
Use small active GPU graphs to traverse a much larger persistent graph over multiple recurrent passes.
```

## Core Formula

```text
G_world = persistent graph on disk/RAM
G_t     = active graph slice loaded at recurrent step t
∂G_t    = frontier / boundary of G_t
F_θ     = sparse attention + MLP reasoning kernel
ω_G     = persistent graph memory weights
B       = active node/edge budget
K       = sparse fanout per layer
L       = local transformer depth
T_outer = recurrent frontier expansion steps
H       = bounded closure horizon over active/candidate graph only
A^{<=H} = I ∨ A ∨ A^2 ∨ ... ∨ A^H
```

Forward / expansion loop:

```text
H_t = F_θ^L(X_t, G_t)
C_t = BoundaryCandidates(∂G_t, G_world)
G_{t+1} = KeepTopB(G_t ∪ TopK(Score(H_t, C_t)))
G_world^{t+1} = Accept(G_world^t ⊕ ΔG_t)
```

Effective reach:

```text
D_effective ≈ L * T_outer
```

Active compute:

```text
O(T_outer * L * |G_t| * K * d)
```

with:

```text
|G_t| <= B
G_t ⊂ G_world
```

## Graph-Native Bounded Kleene Closure

Kleene closure is useful here, but only in bounded/local form:

```text
A^{<=H} = I ∨ A ∨ A^2 ∨ ... ∨ A^H
```

Meaning:

```text
Use repeated hops over G_t ∪ C_t to reason about reachable, cheap, or high-utility paths.
Do not compute full A* over G_world at runtime.
Do not depend on any external orchestration module; this belongs inside math_transformer / graph transformer code.
```

Internal planner modes:

| Mode | Meaning | First use |
|---|---|---|
| Boolean closure | reachable / allowed path exists | verifier feasibility, dependency reachability |
| Cost closure | lowest-cost bounded path | cheap frontier expansion |
| Utility closure | highest-utility bounded path | best reasoning trace / edge retention |

Implementation target:

```text
src/graph_closure.py      # bounded closure primitives over active graph slices
src/frontier_planner.py   # one-hop TopK vs bounded-closure planner comparison
```

The closure planner is not the neural model. It is the graph-transformer-native control primitive for frontier expansion and safe pruning.

## What We Have Now

| Component | Status | Notes |
|---|---:|---|
| Dense masked attention baseline | Done | Correctness baseline exists. |
| Hand TopK topology | Done | Hand K=4 is formal baseline. |
| Learned TopK topology scorer | Partial | Learned K=4 is +1 over hand K=4, fragile. |
| Edge-level delta export | Partial | JSONL exists; aggregate analyzer pending. |
| Sparse attention kernels | Partial | Token sparse, block sparse, Triton paths exist. |
| Prepared topology cache / baked sparse IR | Partial | Exists, but cold/cache/static/growing regimes need proof. |
| Multi-layer local depth | Partial | Model supports `n_layers`; canonical depth sweep pending. |
| Active node routing | Pending | Current system mostly assumes fixed nodes. |
| Graph-native bounded Kleene closure | Pending | No internal `A^{<=H}` closure primitive over active/candidate graph slices yet. |
| Frontier expansion | Pending | No `G_world -> G_t -> ∂G_t -> G_{t+1}` loop yet. |
| Persistent graph writeback | Pending | No verifier-backed `ΔG` commit path yet. |
| Code/AST/IR graph compiler | Pending | No code world graph path in this module yet. |

## Correct Current Interpretation

The accepted quality result is still pruning-first:

```text
Dense full                         = 1752/1786
Hand middle_preserving_topk K=4    = 1773/1786
Learned topology K=4               = 1774/1786
```

Effect decomposition:

```text
Pruning effect: hand K=4 - dense = +21 samples
Learned selection effect: learned K=4 - hand K=4 = +1 sample
```

Therefore:

```text
Sparse structural denoising is proven stronger than learned edge selection so far.
```

Allowed claim:

```text
Learned K=4 is the current fragile quality leader.
```

Disallowed claim:

```text
Learned topology strongly dominates hand topology.
```

## New Delta From v21

The project should now distinguish three different depths:

| Symbol | Meaning | Current status |
|---|---|---|
| `K` | Width per sparse attention layer | Implemented / swept. |
| `L` | Local depth inside active graph | Implemented, not fully benchmarked. |
| `T_outer` | Recurrent frontier graph-walk steps | New architecture; not implemented. |
| `H` | Bounded Kleene closure horizon over `G_t ∪ C_t` | New internal planner primitive; not implemented. |

The main unlock is not simply bigger `L` or bigger `K`.

The main unlock is:

```text
T_outer > 1 under a fixed active graph budget B.
```

The new planning primitive is:

```text
A^{<=H} over G_t ∪ C_t
```

That changes the system from fixed sparse attention into recurrent graph traversal with bounded path-level planning.

## Updated Priority Order

| Priority | Work item | Status | Acceptance |
|---:|---|---:|---|
| P0.1 | Preserve K=4 proof gates | Pending | Every new experiment still reports dense, hand K=4, learned K=4, generic/affine, paired wins/losses. |
| P0.2 | Edge-delta aggregate analyzer | Pending | Summarize 5 learned wins and 4 losses by expert, relation, node pattern, edge score, removed/extra edge type. |
| P0.3 | Repeated locked speed aggregator | Pending | Median/p25/p75/pass-rate gate replaces single-run strict speed. |
| P0.4 | Depth sweep with synced topology | Pending | Measure `L ∈ {1,2,4,8}` at fixed K before claiming multi-hop value. |
| P1.1 | Graph identity schema | Pending | Hash IDs for nodes/edges; labels are metadata; raw facts append-only; derived graph state mutable/versioned. |
| P1.2 | Split world graph from active graph | Pending | Represent `G_world` and `G_t` separately; active graph has explicit node/edge budget `B`. |
| P1.3 | Frontier extraction | Pending | Compute `∂G_t` and unloaded candidate neighbors. |
| P1.4 | Boundary candidate scorer | Pending | Score candidate frontier expansions using `H_t`, node/edge features, relation type, and `ω_G`. |
| P1.5 | Bounded Kleene closure API | Pending | Implement internal `A^{<=H}` over `G_t ∪ C_t`; Boolean first, then cost/utility. No external orchestration dependency. |
| P1.6 | Closure-aware frontier planner | Pending | Compare one-hop TopK expansion vs bounded-closure traces under same K, L, B, H, evaluator. |
| P1.7 | KeepTopB pruning | Pending | After expansion, prune active graph to budget while preserving anchors, verifier-required nodes, and closure-critical paths. |
| P1.8 | Recurrent frontier benchmark | Pending | Compare `T_outer={1,2,3}` under same K, L, B, H, evaluator; include one-hop TopK vs closure planner. |
| P2.1 | Persistent graph memory weights | Pending | Store `ω_ij`, utility, confidence, staleness, provenance, version. Do not store transformer weights in graph. |
| P2.2 | Closure-preserving edge deletion gate | Pending | Before archive/delete, compare important bounded closure under `A^{<=H}` vs `(A\e)^{<=H}` and require accepted Pareto result. |
| P2.3 | Graph writeback verifier | Pending | Accept/reject `ΔV+`, `ΔV-`, `ΔE+`, `ΔE-`, `Δω`; archive/tombstone instead of destructive delete. |
| P3.1 | Code world graph compiler | Pending | Compile AST/CFG/SSA/CALL/TYPE/TEST into `G_world`; run active frontier experiments on code tasks. |
| P3.2 | Market event graph compiler | Later | Compile time/asset/factor/event/regime/risk graph only after math/code frontier loop is stable. |

## First Valid v21 Experiment

Do not build the whole memory system first.

Build the smallest proof:

```text
K = 4 or 8
L = 2
T_outer ∈ {1,2,3}
H ∈ {1,2,3,4}
|G_t| <= B
same evaluator
same validation split
same dense/hand/learned K=4 baselines
one-hop TopK frontier baseline vs bounded-closure frontier planner
```

Acceptance:

```text
J(T_outer=3, closure) > J(T_outer=1, one-hop TopK)
```

where:

```text
J = Q - λT - γM - ρC_frontier - κC_closure
```

and the report includes:

```text
route_acc
generic_acc
affine_acc
runtime median/p25/p75
active node count
active edge count
frontier expansions
closure horizon H
closure path count
closure compute cost
pruned nodes/edges
closure-critical preserved edges
stale/noisy edge rate
```

## Naming Rules

Use these names going forward:

```text
Project/system: Recurrent Frontier Graph Transformer
Mechanism: Recurrent Frontier Graph Attention
Memory substrate: Persistent world graph
Runtime graph: Active frontier graph
Graph update: Verified graph writeback
Kernel: Sparse graph attention
Internal graph planner: Bounded Kleene closure planner
```

Avoid using the old names as the primary system name:

```text
Math-Routed Sparse Transformer
Learned Topology Transformer
```

Those names can remain as historical subsystem names.

## Non-Negotiable Gates

```text
1. Do not promote topology scorers outside the promotion gate.
2. Hand K=4 remains a required quality baseline.
3. Learned K=4 remains a fragile quality leader until repeated validation confirms it.
4. Every topology quality report must include generic and affine accuracy.
5. Every learned-topology mechanism claim must cite edge-level deltas.
6. Sparse speed claims require repeated locked wall-clock distribution, not one run.
7. Recurrent frontier claims must compare T_outer=1 against T_outer>1 under the same K, L, B, and evaluator.
8. Do not claim unlimited memory without reporting active graph budget, retrieval/frontier cost, and stop condition.
9. Do not persist graph mutations without verifier evidence, provenance, versioning, and rollback path.
10. Do not store transformer weights in graph records; graph records may store node/edge memory weights and cached features.
11. Hash IDs are identity; generated labels are metadata.
12. Raw facts should be append-only; derived structures should be mutable and versioned.
13. Do not compute full Kleene closure over `G_world` at runtime; only bounded/local `A^{<=H}` over active/candidate slices is allowed.
14. Kleene/closure planner work stays inside `math_transformer`; do not wire to any external orchestration module.
15. Closure-aware frontier claims must compare against one-hop/heuristic TopK under the same `K`, `L`, `T_outer`, `B`, `H`, and evaluator.
16. Edge deletion must preserve important bounded closure or show an accepted Pareto tradeoff in quality/runtime/memory.
```

## Decision

```text
v21 shifts the project from learned fixed sparse topology to recurrent frontier graph traversal.
```

This does not invalidate the current K=4 topology work. It reframes it as the local sparse routing kernel inside a larger graph memory system.
