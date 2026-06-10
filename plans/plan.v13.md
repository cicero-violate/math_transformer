# plan.v13.md — Turn Learned K=6 Into a Material Full-Block Speed Win

## Context

v12 completed the learned-topology K=6 benchmark plumbing and produced a valid paired benchmark pass.

Final accepted command:

```bash
SCORER=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
LEARNED_K=6 \
HAND_K=16 \
BENCH_STEPS=100 \
BENCH_N=1024 \
HAND_TRITON_BLOCK_K=16 \
LEARNED_TRITON_BLOCK_K=8 \
BENCH_SEED=0 \
scripts/benchmark_learned_topology.sh
```

Final result:

```text
hand K=16 route_acc=0.9821 block_ms=1.094273 attention_ms=0.361384 non_attention_ms=0.732890
learned K=6 route_acc=0.9871 block_ms=1.088686 attention_ms=0.352508 non_attention_ms=0.736179
speedup=1.005132
quality_ok=True speed_ok=True
```

The paired benchmark fixed the prior false-negative issue by comparing:

```text
same MathRoutedTransformerBlock weights
same x input
same benchmark seed
different prepared topology buffers
different Triton BLOCK_K settings
```

So v12 proves:

```text
learned K=6 quality > hand K=16 quality
learned K=6 edges = 6144 vs hand K=16 edges = 16384
learned K=6 attention kernel is faster
learned K=6 full prepared block is slightly faster
```

But the full-block speedup is only:

```text
1.005x
```

The bottleneck is now non-attention cost:

```text
T = A + N
A = attention kernel latency
N = norm/qkv/out_proj/residual/norm2/ffn overhead
```

For learned K=6:

```text
attention_ms     = 0.352508  (~32.4% of block)
non_attention_ms = 0.736179  (~67.6% of block)
```

Therefore topology improvement alone cannot produce the target speedup unless more of the block is fused or removed from the dominant non-attention path.

---

## v13 Objective

Turn learned K=6 from a small paired benchmark pass into a material speed win.

Primary target:

```text
quality_ok=True
speed_ok=True
speedup >= 1.10x
```

Minimum acceptable result:

```text
learned K=6 route_acc >= hand K=16 route_acc
learned K=6 attention_ms < hand K=16 attention_ms
learned K=6 block_ms < hand K=16 block_ms
clear timing-bucket explanation if speedup < 1.10x
```

---

## Task 1 — Preserve v12 Paired Benchmark as Baseline

### Goal

Make the v12 accepted paired benchmark the official baseline for v13.

### Validation command

```bash
SCORER=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
LEARNED_K=6 \
HAND_K=16 \
BENCH_STEPS=100 \
BENCH_N=1024 \
HAND_TRITON_BLOCK_K=16 \
LEARNED_TRITON_BLOCK_K=8 \
BENCH_SEED=0 \
scripts/benchmark_learned_topology.sh
```

Expected baseline:

```text
acceptance_passed quality_ok=True speed_ok=True
speedup ~= 1.005x
```

### Required audit fields

The JSON reports must include:

```text
prepared_static_sparse_block_ms
prepared_static_sparse_attention_ms
prepared_static_sparse_non_attention_ms
triton_block_k
effective_triton_block_k
selector_results.paired_prepared_shared_block
```

`selector_results.paired_prepared_shared_block` must include:

```text
topology_prepare_ms
norm1_ms
qkv_ms
attention_kernel_ms
out_proj_ms
residual1_ms
norm2_ms
ffn_ms
residual2_ms
total_block_ms
same_block_weights
same_input
```

---

## Task 2 — Add Bucket Delta Summary to Console Output

### Problem

The wrapper reports total attention and non-attention time, but v13 needs to identify which non-attention bucket dominates.

### Implementation

Extend `scripts/benchmark_learned_topology.sh` summary parsing to read:

```python
hand_b["selector_results"]["paired_prepared_shared_block"]
learned_b["selector_results"]["paired_prepared_shared_block"]
```

Print a compact bucket table:

```text
bucket                hand_ms    learned_ms    delta_ms
-------------------------------------------------------
norm1_ms              ...        ...           ...
qkv_ms                ...        ...           ...
attention_kernel_ms   ...        ...           ...
out_proj_ms           ...        ...           ...
norm2_ms              ...        ...           ...
ffn_ms                ...        ...           ...
total_block_ms        ...        ...           ...
```

### Acceptance

Every failed speed run must identify whether the bottleneck is:

```text
attention kernel
qkv projection
out projection
layernorm
ffn
residual/measurement overhead
```

---

## Task 3 — Implement Fused Norm + QKV in Prepared Fast Path

### Motivation

Current block timing includes separate:

```text
norm1_ms
qkv_ms
```

These are K-independent and dominate enough to hide attention wins.

There is already partial infrastructure in `NeighborSparseMathAttention.forward_fused_norm_qkv(...)` and `triton_layernorm_qkv(...)`.

### Work

Inspect:

```text
src/attention.py
src/triton_attention.py
src/model.py
```

Confirm whether prepared profiling uses separate norm and QKV even when fused kernels exist.

Implement one of:

1. Use existing fused norm+qkv path in `profile_cached_sparse_block`, or
2. Add a profiling mode that times fused norm+qkv as one bucket, or
3. Add a new explicit prepared fast path:

```python
profile_cached_sparse_block_fused_norm_qkv(...)
```

Expected new buckets:

```text
norm_qkv_ms
attention_kernel_ms
out_proj_ms
residual1_ms
norm2_ms
ffn_ms
residual2_ms
total_block_ms
```

### Acceptance

```text
norm_qkv_ms < norm1_ms + qkv_ms
learned K=6 block_ms improves
```

---

## Task 4 — Implement Fused Attention + Output Projection

### Motivation

Current block timing includes:

```text
attention_kernel_ms
out_proj_ms
```

There is already infrastructure for:

```text
triton_neighbor_attention_outproj
NeighborSparseMathAttention.enable_fused_attn_outproj
```

But prepared profiling currently calls:

```text
triton_neighbor_attention_flat
out_project_for_profile
```

separately.

### Work

Add a prepared profiling path that uses fused attention+out projection when enabled.

Candidate API:

```python
profile_cached_sparse_block(
    ..., 
    fused_norm_qkv: bool = False,
    fused_attn_outproj: bool = False,
)
```

or a separate method:

```python
profile_cached_sparse_block_fused(...)
```

Expected buckets:

```text
attention_outproj_ms
```

instead of:

```text
attention_kernel_ms
out_proj_ms
```

### Acceptance

```text
attention_outproj_ms < attention_kernel_ms + out_proj_ms
learned K=6 remains faster than hand K=16
full block speedup improves toward 1.10x
```

---

## Task 5 — Add Fused Mode Flags to Benchmark CLI

### Goal

Make fusion experiments auditable and reproducible from the benchmark wrapper.

### Proposed flags

In `src.eval`:

```bash
--profile-fused-norm-qkv
--profile-fused-attn-outproj
```

In `scripts/benchmark_learned_topology.sh`:

```bash
FUSED_NORM_QKV=1
FUSED_ATTN_OUTPROJ=1
```

Forward to paired benchmark:

```bash
--profile-fused-norm-qkv
--profile-fused-attn-outproj
```

JSON metadata should include:

```text
profile_fused_norm_qkv
profile_fused_attn_outproj
```

Console output should include:

```text
fused_norm_qkv=True fused_attn_outproj=True
```

---

## Task 6 — Larger-N Scaling Check

### Motivation

At `N=1024`, attention is only about one third of total block time. Learned topology should matter more as sequence length grows.

### Commands

Run paired benchmark at larger N:

```bash
for N in 1024 2048 4096; do
  SCORER=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
  LEARNED_K=6 \
  HAND_K=16 \
  BENCH_STEPS=100 \
  BENCH_N=$N \
  HAND_TRITON_BLOCK_K=16 \
  LEARNED_TRITON_BLOCK_K=8 \
  BENCH_SEED=0 \
  TMP_DIR=runs/benchmarks/v13_n${N} \
  scripts/benchmark_learned_topology.sh
done
```

Record:

```text
N
hand block_ms
learned block_ms
speedup
hand attention_ms
learned attention_ms
attention_share
```

### Acceptance

Learned K=6 speedup should improve as N increases if attention dominates more.

If speedup does not improve, inspect memory locality and block layout.

---

## Task 7 — Locality Ordering Experiment

### Motivation

Arbitrary learned edges can be GPU-hostile even with fewer edges. v12 showed kernel speed improvement, but v13 should test whether learned neighbor ordering can improve memory locality.

### Options

1. Sort learned neighbors ascending after selection:

```text
N_i = sort(N_i)
```

2. Stable order by local block id:

```text
block_id = j // block_size
order by (block_id, score_rank)
```

3. Locality-biased learned score:

```text
score'(i,j) = score(i,j) - lambda * abs(i-j)
```

### Acceptance

```text
route_acc does not drop below hand K=16
attention_kernel_ms decreases
block_ms decreases
```

---

## Task 8 — Decision Criteria

### Promote v13 if

```text
quality_ok=True
speed_ok=True
speedup >= 1.10x
```

### Keep v12 but document limitation if

```text
quality_ok=True
speed_ok=True
1.00x <= speedup < 1.10x
```

and the reason is clearly:

```text
non-attention overhead dominates
```

### Pivot to block-sparse topology if

```text
learned token-edge topology improves quality but does not deliver material speed
```

Then v14 should switch from token-edge learned topology to block learned topology:

```text
score block pairs instead of token pairs
select top-K blocks instead of top-K token neighbors
use GPU-friendly block-sparse layout
```

---

## Expected v13 Deliverables

1. `src.eval` paired benchmark supports fused profiling flags.
2. `scripts/benchmark_learned_topology.sh` reports bucket deltas.
3. JSON reports include fusion metadata and bucket timings.
4. Learned K=6 remains quality-superior.
5. Learned K=6 achieves either:

```text
speedup >= 1.10x
```

or a precise explanation showing which bucket prevents it.

---

## Formula Summary

```text
T = A + N
A = sparse attention latency
N = norm/qkv/out_proj/residual/norm2/ffn latency
```

```text
speedup = hand_block_ms / learned_block_ms
```

```text
quality_ok = learned_route_acc >= hand_route_acc
speed_ok = learned_block_ms < hand_block_ms
```

```text
edge_reduction = (hand_edges - learned_edges) / hand_edges
```

v13 success requires:

```text
quality_ok and speed_ok and speedup >= 1.10
```
