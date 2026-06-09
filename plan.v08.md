# Math-Routed Sparse Transformer Plan v8

## Goal

Turn v7 fixed-`K` sparse topology from a benchmark feature into a prepared hot-path execution strategy.

v7 proved:

```text
allowed = n * K
K = 16
middle_bridge > 0
Q_v7,K16 = Q_full on the current synthetic route validation split
```

v8 answers the next question:

```text
If sparse attention kernel time is low but sparse block time is still high, where is the overhead?
```

---

## Current Status

### Correctness

```text
full pytest suite: 197 passed, 1 warning
focused v7 suite: 59 passed, 1 warning
prepared topology focused suite: passing
```

### v7 smoke benchmark

```text
n = 1024
topology_mode = middle_preserving_topk
fixed_k = 16
max_neighbors = 16
middle_bridge_width = 1

allowed = 16,384
avg_k = 16.0
max_k = 16
rel_red = 0.9844
middle_bridge = 2,907
```

### Sparse attention vs dense attention

Typical smoke result:

```text
d_attn ~= 1.99 ms
s_tri  ~= 0.31 ms
```

So the raw Triton sparse attention kernel is about:

```text
1.99 / 0.31 ~= 6.4x faster
```

### Sparse block vs dense block

Typical smoke result after cache correction:

```text
d_blk ~= 2.46 ms
s_c   ~= 2.63 ms
```

The sparse block is close to parity but not consistently below dense block in the benchmark path.

---

## v8 Implemented Changes

### 1. Cache mode correctness

`TopologyCache` now treats both scored modes as scored top-`K`:

```python
mode in ("scored_topk", "middle_preserving_topk")
```

This prevents `middle_preserving_topk` from falling through to the legacy detailed/union topology path inside model cache execution.

---

### 2. Prepared hot-path topology

Added:

```python
PreparedTopology
TopologyCache.get_or_prepare(...)
```

Prepared topology stores only:

```text
neighbors: (T, K) long
valid_i8:  (T, K) int8
```

It intentionally does not expose dense hot-path structures:

```text
mask:     (T, T)
priority: (T, T)
```

The target hot path is:

```text
x -> norm -> qkv -> Triton(neighbors, valid_i8) -> out_proj -> FFN
```

not:

```text
x -> rebuild topology -> dense mask/priority -> neighbor conversion -> attention
```

---

### 3. Static topology buffers

Added to `MathRoutedTransformerBlock`:

```python
static_neighbors
static_valid_i8
prepare_static_topology(...)
```

Added to `MathRoutedTransformer`:

```python
prepare_static_topology(...)
```

This allows a fixed symbolic graph to be compiled once and reused repeatedly.

For fixed graphs, the intended lifecycle is:

```text
parse/normalize graph
compile fixed-K topology once
store static_neighbors/static_valid_i8
run many forwards
```

---

### 4. Block-level profiler

Added:

```python
profile_cached_sparse_block(...)
```

And CLI:

```bash
.venv-cuda/bin/python scripts/profile_sparse_block.py \
  --n 1024 \
  --node-mode trees \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1 \
  --warmup 3 \
  --iters 10
```

Profiler buckets:

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
```

---

### 5. Triton tile override hooks

Added optional controls:

```python
triton_block_d
triton_block_k
```

Threaded into:

```text
triton_neighbor_attention
triton_neighbor_attention_flat
NeighborSparseMathAttention
MathRoutedTransformerBlock
MathRoutedTransformer
```

This enables experiments like:

```bash
--triton-block-d 16 --triton-block-k 16
```

---

## Profiler Result

Default profiler result at `n=1024`, `trees`, `K=16`:

```text
prepared length=1024 k=16 memory_bytes=147456 device=cuda:0
middle_bridge=4585

bucket                         mean_ms    median_ms
----------------------------------------------------
topology_prepare_ms               0.015        0.014
norm1_ms                          0.114        0.114
qkv_ms                            0.113        0.091
attention_kernel_ms               0.338        0.335
out_proj_ms                       0.086        0.082
residual1_ms                      0.044        0.043
norm2_ms                          0.115        0.112
ffn_ms                            0.177        0.173
residual2_ms                      0.042        0.042
total_block_ms                    1.044        1.008
```

Second default run:

```text
bucket                         mean_ms    median_ms
----------------------------------------------------
topology_prepare_ms               0.018        0.018
norm1_ms                          0.129        0.122
qkv_ms                            0.104        0.102
attention_kernel_ms               0.366        0.353
out_proj_ms                       0.106        0.097
residual1_ms                      0.058        0.052
norm2_ms                          0.121        0.118
ffn_ms                            0.200        0.187
residual2_ms                      0.055        0.050
total_block_ms                    1.158        1.128
```

---

## Key Finding

The original suspicion was:

```text
sparse block overhead may be dominated by topology/index/LUT movement
```

The profiler shows:

```text
topology_prepare_ms ~= 0.014-0.018 ms
```

Therefore:

```text
Topology/index preparation is no longer the bottleneck in prepared mode.
```

The remaining cost is normal transformer block work:

```text
LayerNorm
QKV projection
Triton sparse attention
out projection
FFN
residual/dropout/module dispatch
```

Approximate decomposition:

```text
total_block_ms      ~= 1.0-1.1 ms
attention_kernel_ms ~= 0.34 ms
non-attention work  ~= 0.7 ms
```

---

## Triton Tile Experiment

Forced tile run:

```bash
--triton-block-d 16 --triton-block-k 16
```

Result:

```text
attention_kernel_ms median ~= 0.358 ms
total_block_ms median      ~= 1.147 ms
```

Default was slightly better or equal:

```text
attention_kernel_ms median ~= 0.335-0.353 ms
total_block_ms median      ~= 1.008-1.128 ms
```

Conclusion:

```text
Do not force 16x16 Triton tiles as the default.
Keep automatic tile choice unless more profiling proves otherwise.
```

---

## What v8 Proves

```text
✓ v7 cache path now uses true scored top-K for middle_preserving_topk
✓ prepared topology reduces topology_prepare_ms to ~0.02 ms
✓ static topology buffers work
✓ block profiler works
✓ tile override hooks work
✓ default Triton tile choice is at least as good as forced 16x16 on GTX 1050
✓ remaining overhead is transformer block compute, not topology preparation
```

---

## What v8 Does Not Prove

```text
✗ consistent sparse block speedup over dense block in benchmark path
✗ better prediction quality than dense attention
✗ long-sequence paged Triton execution
✗ fused LayerNorm/QKV/sparse-attention/out-projection kernel
✗ generalization to harder held-out graph/template families
```

---

## Updated Engineering Diagnosis

Old diagnosis:

```text
Sparse block is slow because topology/indexing may be thrashing.
```

New diagnosis:

```text
Prepared topology makes topology/indexing effectively free.
The next bottleneck is standard transformer block overhead around the sparse kernel.
```

Current bottleneck order from profiler:

```text
1. sparse attention kernel: ~0.34 ms
2. FFN: ~0.17-0.20 ms
3. LayerNorm pair: ~0.23-0.24 ms combined
4. QKV + out projection: ~0.19-0.20 ms combined
5. residual/dropout/module dispatch: remaining overhead
```

---

## Plan v8.1: Next Optimization

### Priority 1: benchmark prepared fast path directly

Current `scripts/benchmark_attention.sh` still reports the broader cached block path.

Add a benchmark row for:

```text
prepared_static_sparse_block_ms
```

Acceptance:

```text
prepared_static_sparse_block_ms ~= profiler total_block_ms
```

Target:

```text
prepared_static_sparse_block_ms < d_blk
```

---

### Priority 2: reduce transformer block overhead

Candidate optimizations:

```text
fuse/drop dropout calls in eval path
avoid duplicate module dispatch around residuals
specialize inference-only block path
combine norm/QKV scheduling where possible
reduce FFN width for sparse route workloads if quality is preserved
```

Acceptance:

```text
total_block_ms <= 0.9 ms at n=1024 trees K=16 on GTX 1050
```

---

### Priority 3: add prepared topology benchmark mode

Add CLI option:

```bash
scripts/benchmark_attention.sh \
  --profile-prepared-block
```

or Python option:

```bash
python -m src.eval --prepared-block-benchmark
```

Metrics:

```text
prepared_block_ms
prepared_attention_kernel_ms
prepared_non_attention_ms
```

---

### Priority 4: long-sequence paging execution

Existing:

```text
PagedNeighborTable storage exists
```

Missing:

```text
page-wise Triton execution
```

Target design:

```text
for page in neighbor_pages:
    run sparse attention for token rows in page
```

Acceptance:

```text
n > 4096 runs without dense (T,T) hot-path allocation
paged memory grows as O(T*K)
```

---

## Regression Commands

### Correctness

```bash
./run_tests.sh
```

### Focused v8 tests

```bash
.venv-cuda/bin/python -m pytest -q \
  tests/test_prepared_topology.py \
  tests/test_topology_cache.py \
  tests/test_middle_preserving_topology.py \
  tests/test_sparse_attention.py \
  tests/test_scored_topk.py \
  tests/test_train_paths.py
```

### Profiler

```bash
.venv-cuda/bin/python scripts/profile_sparse_block.py \
  --n 1024 \
  --node-mode trees \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1 \
  --warmup 3 \
  --iters 10
```

### Tile experiment

```bash
.venv-cuda/bin/python scripts/profile_sparse_block.py \
  --n 1024 \
  --node-mode trees \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1 \
  --triton-block-d 16 \
  --triton-block-k 16 \
  --warmup 3 \
  --iters 10
```

---

## Bottom Line

v8 changes the project state from:

```text
fixed-K topology exists, but block overhead is unclear
```

to:

```text
fixed-K topology is prepared/static, topology overhead is negligible, and the remaining bottleneck is measured transformer block compute
```

The topology problem is now largely solved for the current path.

The next optimization problem is block fusion and inference specialization.
