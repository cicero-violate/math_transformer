# Conclusions — Math-Routed Sparse Transformer

**Hardware / runtime**: CUDA GPU, 4 CPU threads, Python 3.12.13, Arch Linux  
**Benchmark path**: CUDA/Triton neighbor-sparse attention through `scripts/benchmark_attention.sh`  
**Kernel**: `src/triton_attention.py::_nbr_sparse_attn_kernel` and flat-output variant  
**Current sparse block mode**: cached topology + top-`K` neighbor truncation

---

## Headline Finding

The CUDA/Triton sparse attention kernel is validated, but the full sparse block only wins in selected cases.

The important correction is that the benchmark is **not** measuring attention over every allowed topology edge. It is measuring a routed top-`K` sparse attention path:

```text
symbolic topology -> priority neighbors -> top-K selected neighbors -> Triton attention
```

Therefore sparse attention runtime scales primarily with:

```text
T * K * D
```

not with:

```text
allowed_edges * D
```

The `allowed` column describes the full symbolic topology. The Triton kernel consumes the truncated neighbor table determined by `--max-neighbors`.

---

## Current Implementation State

The hot path now includes these optimizations:

1. `TopologyCache` stores both `valid` and precompiled `valid_i8`.
2. The Triton wrapper no longer performs live `valid.to(torch.int8).contiguous()`.
3. The cached sparse block has a direct `forward_cached_fast_path` for `return_metadata=False`.
4. Sparse attention uses a fused QKV projection cache for inference.
5. A flat-output Triton sparse attention wrapper was added so sparse output can feed `out_proj` directly.
6. A CUDA correctness test was added for the flat-output Triton wrapper.
7. The default sparse neighbor budget is now `DEFAULT_MAX_NEIGHBORS = 16`, matching the best large-tree benchmark operating point. Passing `max_neighbors=None` still requests exact/unbounded `diag.max_k` behavior.
8. Experimental selector modes were added:
   - `topology_only`
   - `kmip_only`
   - `symbolic_kmip`
   - `symbolic_candidate_kmip`
9. Benchmark reporting now includes selector-level attention time, cached block time, and dense-output proxy metrics.
10. `stk_atn` now reports valid scored-topK attention timing instead of `0.000`.

CUDA correctness status:

```text
.venv-cuda/bin/python -m pytest -q tests/test_triton_attention.py
9 passed
```

---

## Full Sweep Result: n = 64..1024, K = 32

Run:

```bash
scripts/benchmark_attention.sh --sizes 64,128,256,512,1024 --node-mode roots,trees
```

At `K=32`, the kernel-level result is strong:

|    n | mode  | d_attn | s_tri | dense / sparse kernel |
|------+-------+--------+-------+-----------------------|
| 1024 | roots |  1.968 | 0.312 |                 6.31x |
| 1024 | trees |  2.044 | 0.276 |                 7.41x |

The block-level result is mixed:

|    n | mode  | d_blk |   s_c | block result         |
|------+-------+-------+-------+----------------------|
|  128 | roots | 1.095 | 0.908 | sparse wins          |
|  128 | trees | 0.975 | 0.875 | sparse wins          |
| 1024 | trees | 2.498 | 2.432 | sparse wins slightly |
| 1024 | roots | 2.470 | 4.408 | sparse loses         |

The strongest system-level positive signal is:

```text
1024 trees: s_c = 2.432 ms < d_blk = 2.498 ms
```

This is only a small win, but it proves cached sparse block parity is reachable.

---

## K-Scaling Result at n = 1024

Runs:

```bash
scripts/benchmark_attention.sh --sizes 1024 --node-mode roots,trees --max-neighbors 16
scripts/benchmark_attention.sh --sizes 1024 --node-mode roots,trees --max-neighbors 32
scripts/benchmark_attention.sh --sizes 1024 --node-mode roots,trees --max-neighbors 64
scripts/benchmark_attention.sh --sizes 1024 --node-mode roots,trees --max-neighbors 128
```

### Roots

Topology:

```text
allowed = 234,950
full    = 1,048,576
avg_k   = 229.4
max_k   = 343
```

| max_neighbors | effective slots T*K | d_attn | s_tri | d_blk |   s_c | block result |
|---------------+---------------------+--------+-------+-------+-------+--------------|
|            16 | 16,384              |  2.051 | 0.292 | 2.417 | 4.274 | sparse loses |
|            32 | 32,768              |  1.973 | 0.326 | 2.691 | 4.306 | sparse loses |
|            64 | 65,536              |  1.979 | 0.375 | 2.461 | 4.796 | sparse loses |
|           128 | 131,072             |  1.973 | 0.651 | 2.423 | 4.229 | sparse loses |

Roots interpretation:

- Kernel time increases with `K`, as expected.
- Block time remains dominated by non-kernel overhead.
- The roots cached sparse block does not beat dense at n=1024 for any tested `K`.
- Lowering `K` helps, but not enough to overcome block overhead.

### Trees

Topology:

```text
allowed = 545,422
full    = 1,048,576
avg_k   = 532.6
max_k   = 778
```

| max_neighbors | effective slots T*K | d_attn | s_tri | d_blk |   s_c | block result |
|---------------+---------------------+--------+-------+-------+-------+--------------|
|            16 | 16,384              |  1.973 | 0.289 | 2.467 | 2.241 | sparse wins  |
|            32 | 32,768              |  1.961 | 0.294 | 2.438 | 2.789 | sparse loses |
|            64 | 65,536              |  2.003 | 0.432 | 2.403 | 3.296 | sparse loses |
|           128 | 131,072             |  1.954 | 0.475 | 2.466 | 2.418 | sparse wins  |

Trees interpretation:

- The best tree result is `K=16`, where cached sparse block wins:

```text
s_c = 2.241 ms < d_blk = 2.467 ms
```

- Increasing `K` increases kernel work, but block latency is noisy because the surrounding block overhead dominates.
- `K=128` now also crosses tree block parity in the latest rerun: `s_c = 2.418 ms < d_blk = 2.466 ms`.
- `K=16` remains the fastest observed large-tree operating point, while `K=128` may be preferable if quality requires a larger neighbor budget.

---

## Selector Comparison at n = 1024, K = 16

Run:

```bash
scripts/benchmark_attention.sh \
  --sizes 1024 \
  --node-mode roots,trees \
  --max-neighbors 16 \
  --selector-candidate-neighbors 64
```

This run compares four selector modes:

```text
topology_only             = cached symbolic priority table -> top-K neighbors
kmip_only                 = full q_i^T k_j top-K over all tokens
symbolic_kmip             = full alpha*q_i^T k_j + beta*R_symbolic over all tokens
symbolic_candidate_kmip   = score only 64 cached symbolic candidates, then select K=16
```

### Roots

| selector                | attention | cached block | dense L1 proxy | dense cosine proxy |
|-------------------------+-----------+--------------+----------------+--------------------|
| topology_only           | 0.264 ms  | 4.319 ms     |       0.275798 |           0.092203 |
| kmip_only               | 2.084 ms  | 5.907 ms     |       0.260597 |           0.350401 |
| symbolic_kmip           | 2.300 ms  | 6.567 ms     |       0.268183 |           0.239802 |
| symbolic_candidate_kmip | 3.591 ms  | 8.307 ms     |       0.266916 |           0.190543 |

Roots interpretation:

- `topology_only` remains fastest by a large margin.
- `kmip_only` has the best dense-output proxy similarity, but it is much slower than topology-only.
- `symbolic_kmip` is neither fastest nor best by dense proxy.
- `symbolic_candidate_kmip` is unexpectedly slower than full `symbolic_kmip` in this implementation, despite scoring only 64 symbolic candidates per row.

### Trees

| selector                | attention | cached block | dense L1 proxy | dense cosine proxy |
|-------------------------+-----------+--------------+----------------+--------------------|
| topology_only           | 0.299 ms  | 2.090 ms     |       0.286512 |           0.110257 |
| kmip_only               | 2.129 ms  | 3.912 ms     |       0.262491 |           0.340259 |
| symbolic_kmip           | 2.307 ms  | 4.366 ms     |       0.270157 |           0.195079 |
| symbolic_candidate_kmip | 3.598 ms  | 5.784 ms     |       0.271243 |           0.175268 |

Trees interpretation:

- `topology_only` is the only selector that clearly beats dense block time in this run:

```text
topology_only: s_c = 2.090 ms < d_blk = 2.335 ms
```

- This is now the strongest large-tree block-level result observed so far.
- `kmip_only` again has the best dense-output proxy but loses the block-level speed objective.
- `symbolic_candidate_kmip` does not recover the expected performance benefit from candidate restriction. The likely cause is that its current PyTorch gather/scoring/topK path has more overhead than the full matrix score path at `T=1024, C=64`.
- Dense-output proxy metrics are not task quality. They are useful smoke signals only; the architecture decision still requires `Q(K)` on actual tasks.

Selector verdict:

```text
Do not promote k-MIP or symbolic candidate k-MIP yet.
Keep topology_only as the default.
Treat k-MIP selectors as experimental quality probes until selection is fused or otherwise made cheaper.
```

---

## Correct Scaling Interpretation

The table should not be read as:

```text
runtime scales with allowed edges
```

It should be read as:

```text
runtime scales with selected top-K neighbor slots
```

For `n=1024`:

| mode  | allowed edges |  K | kernel slots T*K |
|-------+---------------+----+------------------|
| roots | 234,950       | 32 | 32,768           |
| trees | 545,422       | 32 | 32,768           |

Even though trees have about 2.32x more allowed edges than roots, the Triton kernel sees the same number of slots when `K` is fixed. That is why `s_tri` is similar for roots and trees at the same `K`.

The topology edge count matters for:

1. topology construction cost,
2. neighbor priority selection,
3. model quality,
4. whether a small `K` preserves enough symbolic context.

It does not directly determine Triton sparse attention runtime once top-`K` truncation is applied.

---

## Topology Build Remains Offline-Only

At `n=1024`, topology construction remains far too expensive for a live forward path:

| mode  | topo_ms range in latest runs           |
|-------+----------------------------------------|
| roots | about 2.00s in the latest run          |
| trees | about 1.08s to 1.10s in the latest run |

The architecture must remain two-stage:

```text
Stage 1: compile/cache topology once
Stage 2: run repeated model forwards using cached CUDA neighbors + valid_i8
```

Any benchmark that includes topology construction in every forward is measuring topology generation, not model inference.

---

## What Is Proven

```text
✓ Triton sparse attention is correct on CUDA: 9 tests passed
✓ Sparse attention kernel is much faster than dense attention at n=1024
✓ Runtime scales with T*K, not total allowed edges
✓ Cached sparse block can beat dense block in selected large-tree cases
✓ K is now an explicit accuracy/speed control knob
✓ Topology construction must be cached/offline
✓ scored_topK attention timing is now reported
✓ topology_only at K=16 can beat dense block at n=1024 trees
```

---

## What Is Not Yet Proven

```text
✗ Universal cached sparse block speedup
✗ Roots-mode block-level speedup at n=1024
✗ End-to-end training or inference speedup including all model infrastructure
✗ Quality retention as K is reduced to 16 or 32
✗ k-MIP or symbolic k-MIP selector speedup
✗ Task-quality improvement from k-MIP selectors
```

---

## Engineering Verdict

The right claim is now:

> Math routing can compile a large symbolic topology into a bounded top-K neighbor table, and the resulting Triton sparse attention path can beat dense attention. Cached sparse blocks can cross dense-block parity when K is small enough and topology structure is favorable.

The wrong claim would be:

> Sparse attention runtime scales with all allowed symbolic edges.

It does not. With top-K truncation, runtime scales with `T*K`.

---

## Recommended Next Step

The k-MIP selector comparison has now been run. The result is clear: topology-only is still the only selector that satisfies the speed objective, while k-MIP variants are currently too expensive.

Current routing is topology-prior top-`K`:

```text
N_i = TopK_j R_symbolic(i, j)
```

k-MIP-style routing is learned inner-product top-`K`:

```text
N_i = TopK_j q_i^T k_j
```

The proposed next architecture is symbolic k-MIP:

```text
score(i, j) = alpha * q_i^T k_j + beta * R_symbolic(i, j)
N_i = TopK_j score(i, j)
```

This keeps the math topology as a structural prior while allowing learned Q/K geometry to recover useful neighbors that static symbolic rules miss.

Current selector decision:

```text
Keep topology_only as the source default.
Do not promote symbolic_kmip.
Do not promote symbolic_candidate_kmip.
```

Updated priority order:

1. Measure real task quality for `topology_only` at `K=16`, `K=32`, `K=64`, and `K=128`.
2. Use `trees` as the primary block-parity target and `roots` as the overhead stress test.
3. Profile why the cached topology-only block still varies between winning and losing around dense parity.
4. If k-MIP is revisited, make selection cheaper before benchmarking again:
   - fused candidate scoring/topK kernel,
   - approximate topK/indexed retrieval,
   - or precomputed learned candidate tables.
5. Continue to measure both speed and quality:
   - `S_a`: sparse Triton attention latency
   - `S_c`: cached sparse block latency
   - `D_b`: dense block latency
   - `Q(K)`: task quality
   - `Q(K) / Q_dense`
6. Keep `K=16` as the current source default until quality data proves a larger `K` is necessary.
7. Do not spend more time optimizing topology build for live inference; topology build belongs in the cache/compiler layer.

Target acceptance condition:

```text
For n=1024 trees:
  find selector mode and K such that:
    S_c < D_b
    Q(K) >= 0.95 * Q_dense

For k-MIP variants:
  do not promote unless they beat or match topology_only quality
  and preserve cached sparse block parity on trees

For roots:
  use roots as an overhead stress test, not the primary architecture target
  identify why s_c remains above d_blk even when K is small
```

Decision rule:

```text
If topology_only at K=16 preserves quality:
  keep the current source default and avoid k-MIP complexity.

If topology_only loses quality but a k-MIP selector recovers it:
  first make selector runtime competitive with topology_only.
  only then consider promotion.

If kmip_only matches symbolic_kmip:
  the symbolic topology is not adding enough value and should be simplified.
```
