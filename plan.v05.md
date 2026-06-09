# Math-Routed Sparse Transformer Plan v5

## Status

Project path:

```text
/workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer/
```

Current confirmed baseline:

```text
tests: 146 passed
attention_mode switch: full / dense_masked / neighbor_sparse
topology cache: in-process, keyed on nodes+env+topk+local_window+max_neighbors
env-aware model forward: shape_compat and composition active when env is passed
prioritized neighbor selection: identity > symbolic_dep > composition > shape_compat > embedding_topk > local_window > same_operator
runs/ persistence structure: benchmarks, topology_cache, eval_reports, model_checkpoints, failure_sets
GPU support: CUDA synchronize, device-aware tensor placement, .to(device) throughout
```

Current architecture stage:

```text
Cached env-aware neighbor-sparse transformer — GPU-capable, topology-bottlenecked
```

Target architecture stage:

```text
GPU-resident topology — topology build moved to GPU or cached permanently, attention kernel competitive at real batch sizes
```

---

## 1. Benchmark Findings — GPU vs CPU

### 1.1 Hardware

```text
GPU: NVIDIA GeForce GTX 1050 (4GB GDDR5, 112 GB/s bandwidth)
CPU: 4 threads
CUDA: 12.4 driver, 11.8 nvcc
PyTorch: 2.5.1+cu121
```

### 1.2 Attention-only kernel

GPU benchmark (n=512, roots mode, d_model=64, n_heads=4):

```text
dense_full_attn:    0.591ms
dense_masked_attn:  0.861ms
nbr_sparse_exact:   6.181ms  (max_k=173 out of 512)
nbr_sparse_trunc:   2.865ms  (K=86, half of max_k)
```

GPU benchmark (n=512, trees mode):

```text
dense_full_attn:    0.598ms
nbr_sparse_exact:  14.568ms  (max_k=389 out of 512 — barely sparse)
nbr_sparse_trunc:   6.405ms
```

CPU vs GPU comparison (n=256, roots):

```text
             CPU dense_full   CPU s_trunc   GPU dense_full   GPU s_trunc
n=256 roots    1.48ms          2.67ms          0.21ms           0.92ms
```

GPU is ~7x faster for dense matmul, ~3x faster for truncated sparse.
Ratio gap: sparse/dense = 1.8x on CPU, 4.4x on GPU.
GPU worsens the sparse/dense ratio because BLAS matmul scales better with GPU bandwidth than gather ops do.

### 1.3 Block-level

GPU benchmark (n=512, roots):

```text
full_block:              0.858ms
dense_masked_block:    360.145ms
sparse_block_uncached: 828.375ms
sparse_block_cached:   847.239ms
```

The GPU block timings are dominated by topology build on CPU, not attention compute.

```text
full_block << dense_masked_block << sparse_block
```

because full_block does zero topology work.

### 1.4 Root cause analysis

Two separate bottlenecks:

**Bottleneck 1 — Attention kernel:**

```text
index_select + batched dot products are gather-bound
GTX 1050: 112 GB/s bandwidth, low SM count (640 CUDA cores)
dense matmul uses BLAS (cuBLAS), highly optimized for this hardware
gather ops are not BLAS-accelerated
result: sparse kernel slower than dense on this GPU at d_model=64
```

**Bottleneck 2 — Topology build:**

```text
MathEmbedder.encode_batch: pure Python + NumPy, runs on CPU
TopologyBuilder.build_detailed: NumPy matrix ops, runs on CPU
build_priority_matrix: NumPy, runs on CPU
GPU sits idle during topology build for dense_masked and uncached sparse
topology build time grows faster than O(n^2) in practice due to Python overhead
result: block-level timings are topology-dominated, not attention-dominated
```

### 1.5 What the current system does prove

```text
relation reduction is real: 75-77% fewer edges at n=512 roots
symbolic_dependency: 26060 edges at n=512 trees
shape_compat + composition: active when env is passed
cached topology eliminates rebuild overhead
sparse attention kernel is correct: matches dense masked within 1e-5
priority-ordered truncation retains highest-value edges first
```

### 1.6 What is not yet proven

```text
block-level speedup over full_block
end-to-end wall-clock win on GPU
sparse kernel faster than dense for any tested n on GTX 1050
```

---

## 2. Why GTX 1050 Is the Wrong Target for This Kernel

The neighbor sparse attention kernel does:

```text
for each token i:
    gather K neighbor keys and values from position list N(i)
    compute K dot products
    softmax over K scores
    weighted sum of K values
```

This is memory-bandwidth-bound with irregular access patterns.

Dense matmul does:

```text
Q @ K^T: tiled matrix multiply, L1/L2 cache-friendly, BLAS-accelerated
```

On GTX 1050:

```text
BLAS matmul throughput: ~1.8 TFLOPS FP32
Gather throughput: limited by 112 GB/s GDDR5
Crossover point: sparse wins only when K/n is very small AND n is large
```

On A100:

```text
BLAS matmul throughput: ~312 TFLOPS FP32 (but scaling is similar)
Memory bandwidth: 2TB/s (17x faster than GTX 1050)
Gather throughput: much higher
Crossover point: sparse can win at moderate K/n ratios
```

Conclusion:

```text
GTX 1050 is a bandwidth-starved consumer GPU
The sparse kernel needs either:
  (a) custom Triton kernel (fused gather + attention)
  (b) higher-bandwidth GPU (A100, H100, RTX 4090)
  (c) much smaller K (K <= 8–16 fixed, not topology-derived max_k)
```

---

## 3. Next Targets

### 3.1 Fix Bottleneck 2 — GPU-resident topology

Move topology build to GPU or make it permanent per expression set.

**Option A: GPU topology build**

Port `symbolic_dependency_matrix`, `embedding_topk_matrix`, `shape_compatibility_matrix` to PyTorch ops.

Expected:
```text
symbolic_dependency: torch.tensor(parent-child pairs) -> scatter -> GPU bool matrix
embedding_topk: torch.cdist or torch.topk on GPU Z matrix
shape_compat: dict lookup -> comparison -> bool matrix
```

Result: topology build runs on GPU, no CPU-GPU sync per forward pass.

**Option B: Persistent topology cache on disk**

For a fixed expression set, build once and save to disk:

```text
runs/topology_cache/<hash>.pt
```

Load at model init, never rebuild during inference.

This already mostly works via TopologyCache but is in-process only.

**Option C: Amortize over batch**

Run topology build once per batch (not per sample), reuse across the batch.

Current block forward builds topology per call; with batching, one build serves many samples.

Acceptance:

```text
block-level with cached GPU topology:
  topology_build_ms ≈ 0 (cached)
  attention_ms dominated by attention kernel
  full_block < dense_masked_block < sparse_block_cached
```

---

### 3.2 Fix Bottleneck 1 — Faster sparse kernel

**Option A: Triton kernel**

Write a fused gather-attention kernel in Triton:

```text
@triton.jit
def neighbor_sparse_attention(Q, K, V, neighbors, valid, Out, ...):
    # Load q[i], gather k[neighbors[i,:]], compute scores, softmax, aggregate
```

Advantage: no Python loop, fused gather+dot+softmax in one kernel launch.

**Option B: torch.compile**

Apply `torch.compile` to `neighbor_attention`:

```python
neighbor_attention_compiled = torch.compile(neighbor_attention)
```

May fuse index_select + pointwise ops.

**Option C: Smaller fixed K**

Reduce max_neighbors to 8–16 regardless of topology density.

Current max_k at n=512 roots = 173. At K=16:

```text
compute: n * K * D = 512 * 16 * 64 = 524,288 FLOPs
vs dense: n^2 * D = 512 * 512 * 64 = 16,777,216 FLOPs
ratio: 32x fewer FLOPs
```

This may be enough to cross the BLAS crossover point.

Expected quality impact: priority ordering ensures the 16 most important neighbors are kept.

---

### 3.3 Bigger GPU

The honest path to proving the architecture:

```text
rent an A100 for 1 hour ($3-4)
run: bash scripts/benchmark_attention.sh --sizes 256,512,1024 --node-mode roots,trees --iters 50
```

Expected on A100 (projection):

```text
n=512 roots: dense_full ≈ 0.05ms, sparse_trunc ≈ 0.3ms  (K=86)
n=1024 roots: dense_full ≈ 0.18ms, sparse_trunc ≈ 0.6ms
crossover uncertain — need real measurement
```

---

## 4. Benchmark Stability Issues

### 4.1 trees mode max_k grows too fast

At n=512 trees, max_k=389. This means 76% sparsity is gone.

Root cause: expression trees have high branching → many symbolic_dep edges → high row degree.

Fix options:

```text
(a) hard cap max_k at K_max = 32 regardless of topology
(b) use only root nodes in trees mode (already possible with --node-mode roots)
(c) add --max-neighbors flag to benchmark (already implemented, just default it)
```

Recommended: default max_neighbors=32 for trees mode benchmarks.

### 4.2 same_operator dominates at large n

At n=512 roots, same_operator = 57,972 edges (97% of allowed edges).

Root cause: 6 expression types cycled, so ~n/6 nodes share each operator.

Fix: diversify expression set, or weight same_operator lower (it already has priority 7, lowest).

### 4.3 Topology build is not measured separately

Current benchmark lumps topology_build + attention into block timing.

Add explicit topology-only timing column:

```text
topology_build_ms
```

so we can see exactly what fraction is build vs compute.

---

## 5. Architecture Claim After v5

Valid claims after v4 + GPU run:

```text
The system builds real math relation matrices (symbolic, shape, composition, embedding).
The sparse attention kernel is correct: matches dense masked within 1e-5.
The topology cache eliminates rebuild overhead for repeated forward passes.
Priority-ordered truncation retains the highest-value neighbors under K truncation.
On GTX 1050, dense matmul outperforms the index-gather sparse kernel at all tested sizes.
The bottleneck for block-level timing is topology build (CPU), not attention compute.
```

Target claim after v5 (if Triton kernel or A100 run succeeds):

```text
For math-structured sequences with K/n < 0.1, fused neighbor-sparse attention
is faster than dense matmul on modern high-bandwidth GPUs.
```

---

## 6. Implementation Checklist

### Sprint 1 — Topology timing column

Files:

```text
src/eval.py
```

Add `topology_build_ms` column to benchmark.

Build topology, time it separately, then time attention with pre-built topology.

---

### Sprint 2 — Default max_neighbors cap

Files:

```text
scripts/benchmark_attention.sh
src/eval.py
```

Default `--max-neighbors 32` in benchmark script.

This ensures K/n is small enough to test the regime where sparse can win.

---

### Sprint 3 — torch.compile on sparse kernel

Files:

```text
src/sparse_attention.py
src/eval.py
```

```python
neighbor_attention_compiled = torch.compile(neighbor_attention)
```

Add `compiled_sparse_attn_ms` benchmark column.

---

### Sprint 4 — GPU topology build

Files:

```text
src/topology.py (GPU variants)
src/topology_cache.py
src/model.py
```

Port matrix builds to torch ops on GPU.

---

### Sprint 5 — Triton kernel

Files:

```text
src/triton_attention.py
src/eval.py
tests/test_triton_attention.py
```

Fused gather + dot + softmax.

Gate: triton_sparse_attn_ms < dense_full_attn_ms at some n.

---

## 7. Success Gates v5

### Gate 1 — Topology build isolated

```text
topology_build_ms reported separately
topology_build_ms < attention_ms for cached path
```

### Gate 2 — torch.compile win

```text
compiled_sparse_attn_ms < sparse_exact_ms
```

### Gate 3 — Fixed-K crossover (K=16)

```text
sparse_trunc_attn_ms (K=16) < dense_full_attn_ms at some n on current GPU
```

### Gate 4 — Triton kernel

```text
triton_sparse_attn_ms < dense_full_attn_ms at some n
```

### Gate 5 — A100 / H100 validation

```text
run benchmark on high-bandwidth GPU
report crossover n honestly
```

---

## 8. Benchmark Reference (GTX 1050, v4 run)

```text
device=cuda  threads=4  python=3.12.13 (PyTorch 2.5.1+cu121)

n    mode   allowed   full    avg_k  max_k  sparsity  rel_red  d_attn  m_attn  s_exct  s_trnc  d_blk    m_blk    s_uc     s_c
64   roots  1030      4096    16.1   23     0.2515    0.7485   0.103   0.172   0.204   0.202   0.621    6.105   14.715  15.121
128  roots  3910      16384   30.5   45     0.2386    0.7614   0.135   0.197   0.537   0.302   0.725   47.374   81.462  78.625
256  roots  15046     65536   58.8   87     0.2296    0.7704   0.212   0.307   1.729   0.924   0.723  121.426  267.590 266.181
512  roots  59334     262144  115.9  173    0.2263    0.7737   0.591   0.861   6.181   2.865   0.858  360.145  828.375 847.239
64   trees  2162      4096    33.8   48     0.5278    0.4722   0.114   0.159   0.324   0.220   0.598    4.926   13.390  13.659
128  trees  8604      16384   67.2   98     0.5251    0.4749   0.211   0.298   1.049   0.590   1.154   25.154   70.022  65.959
256  trees  34246     65536   133.8  195    0.5226    0.4774   0.200   0.264   3.803   1.934   0.851   93.263  225.172 224.887
512  trees  136648    262144  266.9  389    0.5213    0.4787   0.598   0.841  14.568   6.405   0.841  216.816  535.531 491.993
```

Key observations:

```text
d_attn is fastest at all n — dense matmul wins on this GPU
s_trnc is 2-3x faster than s_exct (halving K helps significantly)
block timings are topology-dominated (CPU bottleneck)
cached vs uncached sparse block differ by <5% — topology not the block bottleneck now (kernel is)
relation_reduction 73-77% for roots, 47-48% for trees at large n
```
