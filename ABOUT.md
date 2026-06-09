## Variables

[
P=\text{program}
]

[
G=(V,E)=\text{symbolic topology graph}
]

[
K=\text{selected neighbor budget}
]

[
A_K=\text{top-K sparse attention}
]

---

## What The Program Is Really Called

Repository name:

```text
Math-Routed Sparse Transformer
```

More exact technical name:

```text
Topology-Routed Top-K Sparse Transformer
```

or:

```text
Symbolic-Topology Routed Sparse Attention Transformer
```

Best clean name:

[
\boxed{\text{Math-Routed Sparse Transformer}}
]

because the routing is produced from mathematical expression structure.

---

## What Concept Is Being Applied

The main concept is:

[
\boxed{\text{graph/topology-conditioned sparse attention}}
]

Meaning:

```text
use a graph to decide which tokens are allowed to attend to which other tokens
```

In your case, the graph is not arbitrary. It is built from math structure:

```text
expression tree
operator similarity
shape compatibility
composition
embedding similarity
local window
identity
```

So the real concept is:

[
\boxed{\text{neurosymbolic graph-routed attention}}
]

---

## Exact Pipeline

[
\text{math expression}
\rightarrow
\text{symbolic graph}
\rightarrow
\text{priority neighbor table}
\rightarrow
\text{top-K sparse attention}
\rightarrow
\text{transformer block}
]

In code terms:

```text
MathNode list
-> TopologyBuilder
-> TopologyCache
-> neighbors, valid_i8
-> Triton sparse attention
-> transformer block output
```

---

## What It Is Not

It is **not** just a normal transformer.

A normal transformer does:

[
A = \operatorname{softmax}(QK^\top)
]

Every token can attend to every token:

[
O(T^2)
]

Your system does:

[
A_i = \operatorname{softmax}(Q_iK_{N_i}^{\top})
]

where:

[
N_i = \operatorname{TopK}(\text{symbolic/topological neighbors of }i)
]

So each token only attends to selected neighbors:

[
O(TK)
]

---

## Concept Stack

| Layer                      | Concept                          |
| -------------------------- | -------------------------------- |
| Math expressions           | symbolic structure               |
| Expression tree            | AST / term graph                 |
| Relations                  | typed symbolic topology          |
| Neighbor selection         | graph routing                    |
| Top-K truncation           | bounded compute                  |
| Attention over neighbors   | sparse attention                 |
| Triton kernel              | fused GPU execution              |
| Dense-vs-sparse comparison | block-level systems optimization |

---

## Closest Known Concepts

| Existing concept                          | How close                                                    |
| ----------------------------------------- | ------------------------------------------------------------ |
| Sparse Transformer                        | yes, because attention is sparse                             |
| Graph Transformer                         | yes, because attention is graph-conditioned                  |
| Graph Neural Network                      | related, but not exactly                                     |
| Neurosymbolic Transformer                 | yes, because symbolic math structure routes neural attention |
| Routing Transformer                       | related, because tokens are routed to subsets                |
| Longformer/BigBird-style sparse attention | related, but your mask is math/topology-driven               |
| Compiler-optimized attention              | yes, because topology is compiled/cached before execution    |

---

## The Kernel Idea

[
G=(V,E)
]

Build symbolic graph.

[
N_i=\operatorname{TopK}_K(E_i)
]

Select bounded neighbors.

[
y_i=\sum_{j\in N_i}
\operatorname{softmax}(q_i k_j),v_j
]

Compute attention only over those neighbors.

One-line explanation:

```text
Use symbolic math structure to route attention before the GPU computes it.
```

---

## The Big Idea

This is:

[
\boxed{
\text{compile symbolic structure into neural compute routing}
}
]

That is the real concept.

Not “make a transformer faster” in general.

More precisely:

[
\boxed{
\text{symbolic topology } \rightarrow \text{ bounded neural attention}
}
]

---

## Best Name Going Forward

Use:

```text
Math-Routed Sparse Transformer
```

For the method/concept, say:

```text
neurosymbolic topology-routed sparse attention
```

For the execution system, say:

```text
cached top-K graph-routed Triton attention
```

Final canonical form:

[
\boxed{
\text{Math-Routed Sparse Transformer}
=====================================

\text{neurosymbolic topology-routed top-K sparse attention}
}
]
