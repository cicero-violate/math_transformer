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

| n | mode | d_attn | s_tri | dense / sparse kernel |
|---:|------|-------:|------:|----------------------:|
| 1024 | roots | 1.968 | 0.312 | 6.31x |
| 1024 | trees | 2.044 | 0.276 | 7.41x |

The block-level result is mixed:

| n | mode | d_blk | s_c | block result |
|---:|------|------:|----:|--------------|
| 128 | roots | 1.095 | 0.908 | sparse wins |
| 128 | trees | 0.975 | 0.875 | sparse wins |
| 1024 | trees | 2.498 | 2.432 | sparse wins slightly |
| 1024 | roots | 2.470 | 4.408 | sparse loses |

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

| max_neighbors | effective slots T*K | d_attn | s_tri | d_blk | s_c | block result |
|--------------:|--------------------:|-------:|------:|------:|----:|--------------|
| 16  | 16,384  | 2.051 | 0.292 | 2.417 | 4.274 | sparse loses |
| 32  | 32,768  | 1.973 | 0.326 | 2.691 | 4.306 | sparse loses |
| 64  | 65,536  | 1.979 | 0.375 | 2.461 | 4.796 | sparse loses |
| 128 | 131,072 | 1.992 | 0.608 | 2.533 | 5.308 | sparse loses |

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

| max_neighbors | effective slots T*K | d_attn | s_tri | d_blk | s_c | block result |
|--------------:|--------------------:|-------:|------:|------:|----:|--------------|
| 16  | 16,384  | 1.973 | 0.289 | 2.467 | 2.241 | sparse wins |
| 32  | 32,768  | 1.961 | 0.294 | 2.438 | 2.789 | sparse loses |
| 64  | 65,536  | 2.003 | 0.432 | 2.403 | 3.296 | sparse loses |
| 128 | 131,072 | 2.009 | 0.475 | 2.393 | 2.581 | sparse loses |

Trees interpretation:

- The best tree result is `K=16`, where cached sparse block wins:

```text
s_c = 2.241 ms < d_blk = 2.467 ms
```

- Increasing `K` increases kernel work and usually worsens block-level latency.
- `K=32` has shown near-parity or slight wins in other runs, but the K-sweep shows it is not stable enough to claim a general block win.
- For the current implementation, the practical operating point is likely `K=16` for large tree-shaped inputs.

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

| mode | allowed edges | K | kernel slots T*K |
|------|--------------:|--:|-----------------:|
| roots | 234,950 | 32 | 32,768 |
| trees | 545,422 | 32 | 32,768 |

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

| mode | topo_ms range in latest runs |
|------|-----------------------------:|
| roots | about 2.47s to 2.92s |
| trees | about 1.26s to 1.67s |

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
```

---

## What Is Not Yet Proven

```text
✗ Universal cached sparse block speedup
✗ Roots-mode block-level speedup at n=1024
✗ End-to-end training or inference speedup including all model infrastructure
✗ Quality retention as K is reduced to 16 or 32
✗ Valid scored-topK attention timing; stk_atn is still reported as 0.000
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

The next work item is quality/performance co-design around `K`:

1. Evaluate task quality at `K=16`, `K=32`, `K=64`, and `K=128`.
2. Keep `K=16` as the current large-tree performance baseline and source default.
3. Profile `s_c` into projection, Triton attention, output projection, FFN, LayerNorm, and Python dispatch.
4. Do not spend more time optimizing topology build for live inference; topology build belongs in the cache/compiler layer.

Target acceptance condition:

```text
For n=1024 trees, K=16:
  maintain quality while keeping s_c < d_blk

For roots:
  identify and remove the block overhead that keeps s_c > d_blk even when K is small
```
