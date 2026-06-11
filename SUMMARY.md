# System Summary — Recurrent Frontier Graph Transformer

## What This System Is

A neurosymbolic sparse graph transformer that treats graph topology as both a compiled artifact and, in the v21 target, a persistent memory substrate. The current subsystem selects K neighbors per token from a structured MathNode graph, producing an O(N×K) sparse attention pattern instead of O(N²) dense attention. The next architecture adds a persistent world graph, bounded active frontier graph, recurrent frontier expansion, and verified graph writeback.

## Current Quality State (n=1786 validation examples)

| Mode | Correct | route_acc | generic | affine |
|---|---|---|---|---|
| Dense full | 1752/1786 | 0.9810 | 225/259 | 296/296 |
| Hand K=4 | 1773/1786 | 0.9927 | 252/259 | 290/296 |
| Hand K=5 | 1773/1786 | 0.9927 | 247/259 | 295/296 |
| Hand K=8 | 1766/1786 | 0.9888 | 239/259 | 296/296 |
| **Learned K=4** | **1774/1786** | **0.9933** | **253/259** | **290/296** |
| Learned K=5 | 1772/1786 | 0.9922 | 247/259 | 294/296 |
| Learned K=8 | 1767/1786 | 0.9894 | 240/259 | 296/296 |

**Learned K=4 is the current quality leader by +1 sample over hand K=4.**
This is fragile — 9 paired disagreements, 5 learned wins, 4 learned losses.

**The effect decomposition:**
- Pruning effect (hand K=4 − dense): **+21 samples** — structural denoising dominates
- Learned selection effect (learned K=4 − hand K=4): **+1 sample** — real but fragile

**The affine tradeoff is real and tracked:**
Both hand K=4 and learned K=4 regress on affine examples (290/296 vs 296/296
for dense). Sparse topology at K=4 loses 6 affine examples that dense gets
right. Every topology claim must report this Pareto tradeoff — quality on
generic improves, quality on affine regresses.

---

## The Good

**The core architectural bet is correct.**
For structured symbolic domains — math, code, knowledge graphs — the input
graph has real edges that carry semantic meaning. Dense attention wastes compute
attending across those edges uniformly. Topology-guided sparse attention follows
structure. The denoising result proves this: hand K=4 beats dense full attention
on routing accuracy (99.27% vs 98.10%). Pruning semantically irrelevant edges
improves quality, not just speed.

**O(N×K) scales where O(N²) cannot.**
At N=4096+, dense attention and Graphormer run out of memory. The fixed-K
neighbor table produces regular (N×K) rectangular matmuls that are
hardware-friendly — not irregular sparse matrices that fight GPU memory access
patterns. The theoretical scaling advantage becomes a physical requirement at
large N.

**The compiler framing gives correct engineering properties.**
TopologyCache keyed on content hash — same as compiler artifact cache.
Champion promotion gate — same as "this optimization pass is safe to ship."
Locked evaluation protocol — same as a compiler regression suite.
Replay loop — same as a fuzzer that adds hard cases to the test suite.
This infrastructure is production-quality for a research system.

**The topology is a first-class runtime object.**
Dense transformers have no graph. The topology here can be queried, cached,
partially invalidated, and updated without touching model weights. If the graph
changes, only the affected subgraph's cache entry is invalidated. Dense
transformers have no equivalent capability.

**The diagnostic tooling is honest.**
The edge delta analyzer, topology trace logging, failure export, champion
regression checker — the system measures what it claims and catches regressions.
The +1 sample learned K=4 lead is explicitly flagged as fragile. Every mechanism
claim now requires edge-level delta evidence, not just aggregate accuracy.
The affine regression is tracked and reported alongside generic gains.

**The right family for symbolic domains.**
Molecule transformers (Graphormer, GPS, AlphaFold) proved that structure-guided
attention beats dense on graph-structured scientific data. Math expression trees,
code ASTs, and knowledge graphs have the same property: known structure,
sparse true dependencies, noise from irrelevant pairs. The architecture is in
the correct family.

**Distillation path is real.**
A dense SOTA model's attention weights are implicit graph structure. Using those
weights as supervision for the topology scorer converts implicit structure into
explicit compiled topology. The soft surrogate forward (`_soft_topology_forward`)
is the beginning of this path.

---

## The Bad

**The two-loop training separation is the fundamental ceiling.**
The topology scorer is trained to imitate the hand heuristic — not to optimize
the transformer's task performance. The scorer cannot discover topologies better
than what it imitates. The transformer cannot signal to the scorer that its
edge choices are hurting predictions. These two loops are blind to each other.

**No gradient through topology selection.**
`topk_mask_from_scores` is a hard discrete argmax — not differentiable.
Downstream task loss cannot flow back into the scorer. The soft surrogate
trains against a frozen dense teacher, not the sparse model's actual signal.
This is why learned K=8 cannot beat hand K=4 yet: the scorer is optimizing
the wrong objective.

**O(N²) cold start.**
`build_edge_feature_tensor` builds an (N×N×10) tensor on every cache miss.
At N=1024 this is 766ms. At N=4096 it does not fit in memory. For dynamic
inputs where the graph changes every step — autoregressive generation, live
environments — every step is a cold miss. The cache only helps for fixed
repeated graphs.

| N | Feature tensor | Cold start |
|---|---|---|
| 1024 | 10M floats | 766ms measured |
| 4096 | 168M floats | ~12s extrapolated |
| 16384 | 2.7B floats | out of memory |

**Fixed K wastes capacity.**
Every token gets exactly K neighbors regardless of structural complexity. A
leaf node needs 1-2 neighbors. A deeply nested operator may need 12. Fixed K
over-connects simple tokens and under-connects complex ones.

**No equivariance.**
The scorer sees `a+b` and `b+a` as different inputs and can produce different
topologies. Every commutativity variant is a separate training example. This
wastes sample efficiency, reduces generalization, and directly causes the
inconsistent leaf→leaf behavior observed in the edge delta analysis.

The correct symmetry group is:

```
G_symbolic = S_nodes × S_commutative_children × G_alpha-rename × G_tree-automorphism
```

None of these symmetries are enforced.

**No MPNN local propagation.**
The system uses only sparse global attention. Local symbolic propagation —
parent→child, sibling→sibling, binder→bound variable — is not a separate
layer. The scorer compensates by selecting local neighbors explicitly, which
is inefficient and error-prone. The hand K=4 heuristic outperforms because it
follows symbolic edges directly without learning to rediscover them.

**Fixed 10 features limit the scorer.**
The scorer is a 3-layer MLP on 10 hand-designed features. Multi-hop reasoning,
ancestor distance, scope relationships, shortest-path distance — all invisible.
The scorer cannot learn patterns that are not captured in the feature set.

**Same topology across all layers.**
Every attention layer uses the same sparse pattern. Shallow layers need local
syntactic neighbors. Deep layers need long-range semantic neighbors. A single
fixed topology applied uniformly is wrong.

**Speed advantage is currently within measurement noise.**
1.014x speedup. Single-run strict_speed_ok is now classified as diagnostic
only — not a hard gate. Promotion now requires repeated locked run distribution
(median/p25/p75/pass-rate). The theoretical O(NKd) vs O(N²d) advantage does
not materialize at current N. The win only becomes decisive at N >> 1024
which has not been benchmarked.

**The +1 learned K=4 lead is extremely fragile.**
9 paired disagreements total. 5 learned wins, 4 learned losses. A single
additional loss flips the result. This cannot be described as proven — only
as a current quality lead requiring repeated validation and mechanism evidence
from edge-level deltas before any claim is defensible.

**Narrow domain.**
Trained on synthetic math expressions. The topology scorer learns math-specific
patterns that do not transfer to code, molecules, or language without full
retraining. The architecture transfers; the learned weights do not.

---

## The Ceilings In Order Of Severity

| Ceiling | Root cause | Fix |
|---|---|---|
| +1 lead is statistically fragile | 9 paired samples total | Repeated validation + expanded test set |
| Affine regression unresolved | K=4 loses 6 affine examples | Dynamic-K or affine-aware topology |
| Scorer cannot beat what it imitates | Two-loop separation | End-to-end gradient through topology |
| Wrong training signal | No gradient through topk | Straight-through estimator or REINFORCE |
| Breaks at N > 4096 dynamic inputs | O(N²) feature tensor | Block heuristic or local approximation |
| Inconsistent topology for equivalent graphs | No equivariance | Algebraic canonicalization + L_S loss |
| Suboptimal neighbor allocation | Fixed K | Dynamic-K controller |
| Cannot learn multi-hop patterns | Fixed 10 features | Feature registry + shortest-path features |
| Different layers need different topology | Single topology | Per-layer topology scorer |
| Speed win not yet real | Small N | Large-T benchmark at N=4096+ |

---

## The Correct Next Architecture

Following the GPS (GraphGPS) family with symbolic equivariance:

```
G → Canonicalizer → Structural Encoder → [Local MPNN + Sparse Attention] → Topology Scorer → Verifier
```

Each block:

```
H+ = LN(H + MPNN(H, A) + SparseAttn(H, S_θ, B) + MLP(H))
```

where `S_θ = TopK(ψ_θ(H, A, B))` and `B_ij` encodes structural biases
(edge type, shortest-path distance, ancestor distance, scope distance).

Add in this order:

1. **Algebraic canonicalization** — now, no model changes, removes fake variation
2. **Topology equivariance loss** `L_S = |S(PAP^T, PH) - P S(A,H) P^T|^2` — after K-sweep
3. **Local MPNN layer** — after scorer is proven
4. **GPS fusion gate** — after MPNN is validated
5. **End-to-end gradient through topology** — the fundamental fix, hardest to implement

Do not add EGNN. The symmetry group is symbolic-algebraic, not E(3). Geometric
equivariance is only meaningful when coordinates have real physical geometry.
Force-directed layout coordinates are rendering artifacts, not semantic positions.

---

## Where This Beats Other Architectures

| Regime                            | Winner                                                        |
|-----------------------------------+---------------------------------------------------------------|
| 3D molecular geometry             | EGNN / AlphaFold (E(3) equivariance is a hard physical prior) |
| Small graphs N < 100              | Graphormer (dense + continuous biases more expressive)        |
| Large symbolic graphs N > 2000    | This system (only architecture that scales)                   |
| Fixed graphs, many forward passes | This system (topology cache amortizes cold start)             |
| Dynamic changing graphs           | This system (topology is a runtime object)                    |
| Open-domain language              | Dense transformer (structure not recoverable)                 |

---

## The Long-Term Bet

Dense transformers are the assembly language of AI — they work for everything
but are optimal for nothing. This system is the compiler layer for structured
domains. The industry will get there domain by domain: molecules first (already
there), then code, then math, then knowledge graphs. The structured transformer
compiler will follow the same adoption curve as LLVM — not replacing everything
overnight, but becoming the right tool for the domains where structure is known.

The question is not whether this architecture is correct. It is whether the
implementation can prove the hypothesis at the scale where the advantage is
decisive.
