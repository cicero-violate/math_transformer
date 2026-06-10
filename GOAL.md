# Goal — Math-Routed Sparse Transformer

## Objective

Build a math-routed transformer block that preserves dense-model quality while reducing full block latency through symbolic top-K sparse attention.

The target is not to process every symbolic edge faster. The target is to compile a large symbolic topology into a bounded neighbor table, run atteention over that selected subset, and beat the dense transformer block at useful quality.

---

## Variables

```text
T     = number of tokens / math nodes
K     = selected neighbors per node
E     = total allowed symbolic edges
D_b   = dense full block latency
S_c   = cached sparse block latency
D_a   = dense attention latency
S_a   = Triton sparse attention latency
Q(K)  = task quality at neighbor budget K
Q_d   = dense baseline quality
```

---

## Core Target

```text
S_c(T, K) < D_b(T)
```

while preserving quality:

```text
Q(K) >= 0.95 * Q_d
```

One-line target:

```text
same useful quality, lower full-block latency, bounded top-K compute
```

---

## Correct Scaling Claim

Sparse runtime should scale with selected neighbor slots:

```text
O(T * K * D)
```

not with the full symbolic topology:

```text
O(E * D)
```

and not with dense attention:

```text
O(T^2 * D)
```

The symbolic topology is used for routing and selection. It is not the set of edges processed by the Triton kernel after top-K truncation.

---

## Results We Want

### 1. Kernel-Level Win

```text
S_a << D_a
```

Acceptance target:

```text
D_a / S_a >= 3x
```

Current status: achieved at large T. The Triton sparse attention kernel is not the main remaining bottleneck.

### 2. Block-Level Win

```text
S_c < D_b
```

This is the primary performance result.

Target cases:

```text
512 roots
512 trees
1024 roots
1024 trees
```

Minimum viable result:

```text
1024 trees: S_c < D_b at a useful K
```

Strong result:

```text
512 trees and 1024 trees: S_c < D_b with acceptable quality
```

Great result:

```text
512 roots, 512 trees, 1024 roots, 1024 trees: S_c < D_b with acceptable quality
```

### 3. Quality Retention

For each tested K:

```text
K in {16, 32, 64, 128}
```

measure:

```text
Q(K) / Q_d
```

Acceptance target:

```text
Q(K) >= 0.95 * Q_d
```

Speed without quality is not a valid architecture win.

---

## Required Evaluation Table

For each topology mode and sequence size, produce:

| K | S_a | S_c | D_a | D_b | sparse block wins? | Q(K) | Q(K)/Q_d |
|---:|---:|---:|---:|---:|---|---:|---:|
| 16 | | | | | | | |
| 32 | | | | | | | |
| 64 | | | | | | | |
| 128 | | | | | | | |

The chosen operating point is:

```text
K* = argmax_K Q(K) / S_c(K)
```

subject to:

```text
S_c(K) < D_b
Q(K) >= 0.95 * Q_d
```

---

## Current Working Hypothesis

```text
K = 16
```

is the fastest large-tree operating point and is therefore the current source default.

```text
K = 128
```

may be preferable if quality requires a larger symbolic neighborhood, because the latest large-tree run also crossed block-level parity at K=128.

Roots mode still fails block-level parity in current runs, which means roots are dominated by block overhead rather than sparse kernel time.

---

## Non-Goals

Do not claim that runtime scales with all allowed symbolic edges.

Do not optimize for:

```text
processing every allowed edge
```

The intended architecture is:

```text
symbolic topology -> priority routing -> top-K neighbor table -> bounded Triton sparse attention
```

not:

```text
symbolic topology -> attention over all allowed edges
```

---

## Architecture Acceptance Statement

The architecture is successful when we can state:

```text
We compile a large symbolic math topology into a bounded top-K attention route, preserve dense-model quality within tolerance, and run the cached sparse transformer block faster than the dense block.
```

Formally:

```text
Q(K) >= 0.95 * Q_d
S_c(T, K) < D_b(T)
S_a(T, K) = O(T * K)
K << T
```
