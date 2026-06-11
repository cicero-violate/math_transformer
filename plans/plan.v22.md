# Plan v22 — CUDA-Proven Sparse Topology + Quantale-Native Frontier Planner

## Official Name

```text
Sparse Quantale Graph Transformer
```

Runtime graph:

```text
persistent world graph -> bounded active graph -> sparse graph attention -> quantale path closure -> frontier expansion -> verified writeback
```

## Why v22 Exists

v21 moved the project from fixed sparse topology toward recurrent frontier graph traversal.

v22 records the first CUDA proof pass and tightens the next direction:

```text
Do not jump to MCTS yet.
First make bounded closure algebra generic and quantale-native inside math_transformer.
Then use beam/MCTS later only as approximations when the exact bounded planner becomes too expensive.
```

The core change is conceptual:

```text
closure modes are not separate hacks.
closure modes are instances of one path algebra.
```

## Latest Evidence — 2026-06-11 CUDA Run

Environment:

```text
CUDA Python: .venv-cuda/bin/python
Torch: 2.5.1+cu121
GPU: NVIDIA GeForce GTX 1050
```

Full CUDA test suite:

```text
.venv-cuda/bin/python -m pytest -q
402 passed, 1 warning in 22.79s
```

Focused graph/frontier CUDA test suite:

```text
116 passed
```

CUDA quality smoke on `runs/checkpoints/synthetic_tiny.pt`:

| mode | k | examples | route_acc | correct |
|---|---:|---:|---:|---:|
| dense/full | full | 15 | 0.533333 | 8/15 |
| topology_only | 4 | 15 | 0.600000 | 9/15 |

CUDA K-sweep artifacts:

```text
runs/cuda_smoke/topology_k_sweep.json
runs/cuda_smoke/topology_k_sweep.csv
```

Locked protocol pass:

```text
scripts/run_locked_topology_protocol.sh
```

Protocol:

```text
champion_k8_vs_hand_k16_locked
hash=6673abb961f69721c04dcb5746b3e70f22d046143b4eef8ea559d6c89ef827f9
examples=data/synthetic_hard/val.jsonl
checkpoint=runs/checkpoints/synthetic_hard_dense.pt
scorer=runs/checkpoints/topology_scorer.champion.pt
```

Quality:

| mode | k | examples | route_acc | correct |
|---|---:|---:|---:|---:|
| dense/full | full | 1786 | 0.980963 | 1752/1786 |
| hand_topology | 16 | 1786 | 0.982083 | 1754/1786 |
| learned_topology | 8 | 1786 | 0.987122 | 1763/1786 |

Speed:

| metric | hand K=16 | learned K=8 |
|---|---:|---:|
| block_ms | 0.929171 | 0.916944 |
| attention_ms | 0.320675 | 0.309453 |
| non_attention_ms | 0.608496 | 0.607490 |
| speedup | — | 1.013335 |

Acceptance:

```text
quality_ok=True
speed_ok=True
strict_speed_ok=True
acceptance_passed=True
```

Repeated locked protocol status:

```text
scripts/run_repeated_locked_topology_protocol.sh
```

First run exposed a blocker:

| comparison   |     hand |  learned |
|--------------+----------+----------|
| K            |        4 |        4 |
| route_acc    | 0.985442 | 0.993281 |
| block_ms     | 0.985052 | 1.054465 |
| attention_ms | 0.329192 | 0.334689 |
| speedup      |        — | 0.934172 |

Status:

```text
quality_ok=True
speed_ok=False
strict_speed_ok=False
diagnosis=learned_attention_not_faster_than_hand
```

Interpretation:

```text
Single locked K8-vs-hand-K16 protocol passes.
Repeated K4-vs-K4 protocol exposes speed regression despite learned quality win.
Do not claim repeated speed proof yet.
```

## Core Formula

```text
G_world = persistent graph memory
G_t     = active graph slice at outer step t
∂G_t    = frontier / boundary of G_t
C_t     = candidate expansion nodes/edges
A_t     = weighted adjacency / transition relation over G_t ∪ C_t
K       = sparse fanout per layer
L       = local transformer depth
B       = active graph budget
H       = bounded closure horizon
T_outer = recurrent frontier expansion steps
```

Learned local pass:

```text
H_t = F_θ^L(X_t, G_t)
```

Quantale path closure:

```text
A_Q^{<=H} = I_Q ∨ A_t ∨ A_t^2 ∨ ... ∨ A_t^H
```

Frontier update:

```text
G_{t+1} = KeepTopB(G_t ∪ TopK(PathScore(A_Q^{<=H})))
```

Score:

```text
J = Q - λT - γM - ρC_frontier - κC_closure
```

## Quantale-Native Planner

The current `src/graph_closure.py` exposes separate modes:

```text
boolean
cost
utility
```

v22 target:

```text
turn these into instances of a single internal QuantaleSpec.
```

Proposed internal interface:

```text
QuantaleSpec:
  name
  zero              # unreachable / impossible value
  one               # identity path value
  join(a,b)         # choose/merge alternative paths
  compose(a,b)      # concatenate path steps
  better(a,b)       # ranking comparator
  valid(x)          # reachable/usable value predicate
```

Built-in specs:

| spec    | join | compose | meaning                      |
|---------+------+---------+------------------------------|
| Boolean | OR   | AND     | reachability / permission    |
| Cost    | min  | +       | cheapest bounded path        |
| Utility | max  | +       | highest utility bounded path |

Why this is better:

```text
Cost sorting bugs disappear because the algebra defines better().
Utility reachability bugs disappear because the algebra defines valid().
Closure code becomes generic and testable once.
```

## Search / MCTS Position

MCTS is not rejected. It is delayed.

Correct stack:

```text
one-hop TopK baseline
→ bounded closure baseline
→ quantale-native bounded planner
→ beam search over quantale traces
→ MCTS only if branching makes exact/beam planner too expensive
```

MCTS state later:

```text
state  = ActiveGraph G_t
action = add node / add edge / preserve / prune / stop
prior  = sparse model score + quantale path score
value  = evaluator quality - runtime - memory - closure cost
```

But v22 does not implement MCTS first.

## Required Fixes Before New Claims

| priority | fix                                                  | reason                                                                                                          |
|----------+------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------|
| P0       | depth sweep table formatter handles `k=None`         | Dense/full row currently crashes table print.                                                                   |
| P0       | depth sweep uses correct expert keys                 | `generic_expert` / `affine_expert`, not `generic` / `affine`.                                                   |
| P0       | depth sweep protocol clarified                       | One strict L=2 checkpoint cannot prove L={1,2,4,8}. Need per-L checkpoints or explicit partial-load experiment. |
| P0       | repeated locked protocol summarizes partial failures | A failed first run should still emit a clear summary artifact.                                                  |
| P0       | repeated speed proof remains open                    | K4 learned wins quality but loses speed in first repeated run.                                                  |
| P1       | quantale-native closure API                          | Replace mode-specific logic with one algebraic closure kernel.                                                  |
| P1       | cost/utility planner semantics                       | Cost lower-is-better; utility unreachable pairs not counted as reachable.                                       |
| P1       | operational tombstone filtering                      | Deactivated records must be invisible to all graph queries by default.                                          |

## Updated Priority Order

| Priority | Work item                             | Status  | Acceptance                                                                                |
|----------+---------------------------------------+---------+-------------------------------------------------------------------------------------------|
| P0.0     | Preserve CUDA green suite             | Done    | `.venv-cuda/bin/python -m pytest -q` remains green.                                       |
| P0.1     | Locked protocol pass                  | Done    | `champion_k8_vs_hand_k16_locked` passes quality and strict speed.                         |
| P0.2     | Repeated locked speed proof           | Blocked | Needs median/p25/p75/pass-rate artifact; current first K4-vs-K4 repeated run fails speed. |
| P0.3     | Depth sweep repair                    | Pending | CLI prints dense/full rows, reports expert keys, and defines valid L-sweep protocol.      |
| P0.4     | K4/K8/K16 CUDA smoke artifacts        | Done    | `runs/cuda_smoke/topology_k_sweep.{json,csv}` exist and are reproducible.                 |
| P1.1     | QuantaleSpec internal API             | Pending | Boolean/cost/utility closure implemented through one generic bounded closure path.        |
| P1.2     | QuantaleFrontierPlanner               | Pending | Beats or matches ClosureAwarePlanner and one-hop TopK under same K,L,B,H,T_outer.         |
| P1.3     | Closure-preserving deletion semantics | Partial | Deactivation/tombstone works and is query-invisible; rollback reactivates.                |
| P2.1     | Beam search over quantale traces      | Later   | Compare beam vs quantale TopK before MCTS.                                                |
| P2.2     | MCTS frontier planner                 | Later   | Only after quantale planner establishes deterministic baseline and cost model.            |
| P3.1     | Sparse-student distillation           | Later   | Teacher structure/behavior → sparse quantale-native student.                              |

## First Valid v22 Experiment

```text
K ∈ {4,8}
L = 2
B ∈ {16,32,64}
H ∈ {1,2,3,4}
T_outer ∈ {1,2,3}
planner ∈ {one_hop_topk, bounded_closure, quantale_frontier}
same evaluator
same seeds
same CUDA environment
```

Acceptance:

```text
J(quantale_frontier) > J(one_hop_topk)
```

and:

```text
quality does not regress
active graph budget B is respected
closure cost is reported
wall-clock distribution is reported across repeated runs
```

## Non-Negotiable Gates

1. Do not claim repeated speed proof until repeated locked protocol passes distributional metrics.
2. Do not claim depth-sweep proof from a single strict L=2 checkpoint across L={1,2,4,8}.
3. Do not compute full closure over `G_world` at runtime.
4. Do not wire to any external orchestration module; planner stays inside `math_transformer`.
5. Quantale mode semantics must define `join`, `compose`, `better`, and `valid` explicitly.
6. Edge deletion must preserve important bounded closure or show accepted Pareto tradeoff.
7. Search/MCTS must compare against quantale baseline under identical K,L,B,H,T_outer.

## Decision

```text
v22 locks in CUDA viability and locked protocol pass.
The next architectural move is quantale-native bounded planning, not MCTS-first.
Repeated locked speed remains the main proof blocker.
Depth sweep must be repaired before it can be used as a P0 gate.
```
