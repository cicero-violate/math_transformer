# Math-Routed Sparse Transformer Plan v7

## Goal

Apply the lost-in-the-middle topology lesson to this sparse math transformer.

The paper's useful abstraction is:

\[
A = \alpha I + \beta C + \gamma G
\]

Where:

- \(I\) is the residual/self path.
- \(C\) is the existing causal/local/topology route.
- \(G\) is an explicit graph-routing correction that prevents middle-context starvation.

For this project, v7 adds \(G\) as an opt-in sparse topology relation named `middle_bridge`.

---

## Problem

Current v6 routing solved the wrong half of the sparse-attention problem first:

\[
E_i = \operatorname{TopK}_j(\text{semantic score}(i,j), K)
\]

This keeps compute bounded, but it does not guarantee that structurally weak middle positions survive truncation.

The paper implies that the sparse mask itself is part of the optimization geometry. If middle positions are rarely selected, no amount of downstream attention kernel work repairs the missing path.

---

## v7 Design

Add a middle-preserving bridge matrix:

\[
G \in \{0,1\}^{n \times n}
\]

Each row receives a small deterministic candidate set:

\[
M(i)=\{\lfloor n/4\rfloor, \lfloor n/2\rfloor, \lfloor 3n/4\rfloor\}
\]

With optional width \(w\):

\[
M_w(i)=\{a-r,\dots,a+r : a\in M(i), 0\le a<n, 0\le r\le w\}
\]

Then scored routing becomes:

\[
\text{score}(i,j)=
\sum_r w_r f_r(i,j)+w_{mid}G_{ij}
\]

And final sparse neighbors remain bounded:

\[
|N(i)|\le K
\]

No dense all-pairs attention is introduced at runtime beyond the existing topology-scoring path.

---

## Implementation Scope

### Add

- `src/middle_preserving_topology.py`
  - `middle_anchor_indices(n, width)`
  - `middle_bridge_matrix(n, width)`
  - `middle_bridge_matrix_torch(n, width, device)`
  - `middle_coverage_score(...)`

### Modify

- `src/topology.py`
  - Add relation weight: `middle_bridge`.
  - Add relation priority for opted-in bridge edges.
  - Add topology mode: `middle_preserving_topk`.
  - Include bridge scoring and diagnostics only for that mode.

- `src/topology_cache.py`
  - Include topology mode and v7 parameters in cache key.
  - Use bridge-aware priority matrices when active.

- `src/sparse_attention.py`
  - Treat priority `0` as lowest priority during prioritized truncation.

- `src/model.py`
  - Thread `middle_bridge_width` through model/block constructors.

- `src/eval.py`
  - Expose `middle_preserving_topk` in `--topology-mode`.
  - Add `--middle-bridge-width`.

### Add tests

- Middle anchor generation and matrix shape.
- Middle coverage increases under `middle_preserving_topk` relative to `scored_topk`.
- Model forward works with the new topology mode.
- Priority `0` is sorted last, not first.

---

## Acceptance Gates

```text
pytest tests/test_middle_preserving_topology.py tests/test_sparse_attention.py tests/test_scored_topk.py
```

Required properties:

```text
middle_preserving_topk avg_k <= fixed_k
middle coverage > scored_topk middle coverage on the same sequence
self-loops are preserved
model forward returns the expected shape
```

---

## Benchmark Command

```bash
scripts/benchmark_attention.sh \
  --sizes 1024 \
  --node-mode trees \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1
```

Compare against:

```bash
scripts/benchmark_attention.sh \
  --sizes 1024 \
  --node-mode trees \
  --topology-mode scored_topk \
  --fixed-k 16 \
  --max-neighbors 16
```

---

## Interpretation

v7 is not a kernel optimization. It is a routing-topology correction.

The intended result is:

\[
\min_{x \in \text{middle band}} \rho(x)
\]

increases while:

\[
O(nK)
\]

is preserved.
