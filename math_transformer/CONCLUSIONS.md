# Conclusions — Math-Routed Sparse Transformer

**Hardware / runtime**: CUDA GPU, 4 CPU threads, Python 3.12.13, Arch Linux  
**Run**: `scripts/benchmark_attention.sh --sizes 64,128,256,512 --node-mode roots,trees`  
**Benchmark path**: CUDA/Triton-only neighbor-sparse attention. CPU fallback removed from execution.  
**Kernel**: `src/triton_attention.py::_nbr_sparse_attn_kernel`

---

## Headline Finding

**The sparse attention kernel now beats dense attention on CUDA.**

The neighbor-sparse path is routed through the fused Triton kernel. In the benchmark table,
`s_trnc == s_tri`, which means the truncated sparse attention measurement is the Triton
kernel measurement. `s_comp` is intentionally zero because the old `torch.compile` sparse
path is no longer used in the benchmark hot path.

At `n=512`:

- **roots**: Triton sparse attention is **3.08× faster** than dense attention.
- **trees**: Triton sparse attention is **2.38× faster** than dense attention.

The attention bottleneck has moved: the kernel is no longer the main problem. Topology
construction and block-level Python/symbolic overhead now dominate wall-clock time.

---

## Attention-Only Timings

All timings are milliseconds. `d_attn` is dense full attention. `s_tri` is the fused Triton
neighbor-sparse attention kernel.

| n | mode | allowed | full | rel_red | d_attn | s_tri | dense / Triton |
|---:|------|--------:|-----:|--------:|-------:|------:|---------------:|
| 64  | roots | 1,030 | 4,096 | 74.85% | 0.131 | 0.084 | **1.56×** |
| 128 | roots | 3,910 | 16,384 | 76.14% | 0.158 | 0.101 | **1.56×** |
| 256 | roots | 15,046 | 65,536 | 77.04% | 0.209 | 0.129 | **1.62×** |
| 512 | roots | 59,334 | 262,144 | 77.37% | 0.589 | 0.191 | **3.08×** |
| 64  | trees | 2,162 | 4,096 | 47.22% | 0.121 | 0.083 | **1.46×** |
| 128 | trees | 8,604 | 16,384 | 47.49% | 0.178 | 0.157 | **1.13×** |
| 256 | trees | 34,246 | 65,536 | 47.74% | 0.196 | 0.144 | **1.36×** |
| 512 | trees | 136,648 | 262,144 | 47.87% | 0.593 | 0.249 | **2.38×** |

### Interpretation

Roots mode is faster because it is much sparser:

- `n=512 roots`: 59,334 allowed edges, **77.37%** relation reduction.
- `n=512 trees`: 136,648 allowed edges, **47.87%** relation reduction.

Trees mode is more semantically connected but more expensive:

- `n=512 roots`: `symbolic_dependency = 0`
- `n=512 trees`: `symbolic_dependency = 26,060`

So the tradeoff is clear:

```text
roots = higher sparsity, faster kernel
 trees = richer symbolic dependency routing, more edges
```

---

## Topology Build Cost Is Now the Bottleneck

The Triton attention kernel is fast. The symbolic topology builder is not.

| n | mode | topo_ms | s_tri | topo / Triton |
|---:|------|--------:|------:|--------------:|
| 64  | roots | 15.897 | 0.084 | **189×** |
| 128 | roots | 77.640 | 0.101 | **769×** |
| 256 | roots | 256.576 | 0.129 | **1,989×** |
| 512 | roots | 586.338 | 0.191 | **3,070×** |
| 64  | trees | 10.742 | 0.083 | **129×** |
| 128 | trees | 49.680 | 0.157 | **316×** |
| 256 | trees | 164.201 | 0.144 | **1,140×** |
| 512 | trees | 445.693 | 0.249 | **1,790×** |

The current runtime shape is therefore:

```text
symbolic topology construction >> model/block Python overhead >> Triton attention
```

The attention kernel win is real, but it is hidden at the block/system level unless topology
is compiled once and reused.

---

## Block-Level Timings Expose Hot-Path Overhead

At `n=512`:

| mode | d_blk | sparse uncached | sparse cached |
|------|------:|----------------:|--------------:|
| roots | 0.894 | 420.010 | 415.785 |
| trees | 0.930 | 256.026 | 255.805 |

These numbers are not explained by the Triton kernel:

- `n=512 roots s_tri = 0.191ms`
- `n=512 roots sparse cached block = 415.785ms`

So the block path is still doing expensive non-kernel work. The cached sparse block is not yet
a clean hot path.

Likely remaining costs:

1. Recomputing or revalidating symbolic topology inputs.
2. Python-side node embedding / route metadata work.
3. Cache lookup and tensor movement overhead.
4. Unconditional router/diagnostic work inside model forward.

The target block path should look like this:

```text
precompiled (neighbors, valid) on CUDA + projected QKV -> Triton kernel -> output
```

not this:

```text
forward -> symbolic topology/cache/router work -> neighbor tensors -> Triton kernel -> output
```

---

## Scored Top-K Timing Is Currently Invalid

The benchmark reports:

```text
stk_atn = 0.000
```

for every row. That means scored-topK attention is not being measured successfully. It is
likely failing inside the guarded scored-topK timing block and being swallowed by an exception
handler.

Do not use `stk_atn` as evidence until the exception is surfaced and fixed.

---

## What Is Now Proven

```text
✓ CUDA/Triton neighbor-sparse attention is wired into the benchmark path
✓ CPU fallback is removed from benchmark execution
✓ Triton sparse attention beats dense attention for n=64..512 on this CUDA run
✓ At n=512, sparse attention is 3.08× faster than dense in roots mode
✓ At n=512, sparse attention is 2.38× faster than dense in trees mode
✓ Roots preserves ~75–77% relation reduction across n=64..512
✓ Trees preserves ~47–48% relation reduction while adding symbolic dependency edges
✓ The bottleneck has moved from attention compute to topology/block infrastructure
```

---

## What Is Not Yet Proven

```text
✗ Block-level wall-clock speedup
✗ End-to-end model speedup
✗ A clean cached hot path with topology fully removed from forward
✗ Valid scored-topK attention timings
✗ Training-speed improvement
```

---

## Architecture Verdict

The central kernel-level claim is now stronger than before:

> Math-structured sparse routing can produce a neighbor-sparse attention pattern that runs
> faster than dense attention when executed by a fused CUDA/Triton kernel.

This is now shown directly on CUDA, not just CPU.

However, the system-level claim is still blocked by infrastructure overhead:

> The model cannot show wall-clock wins until symbolic topology, embedding, routing, and
> diagnostics are removed from the repeated forward hot path.

The right abstraction is a two-stage pipeline:

```text
Stage 1: compile symbolic graph once
         nodes + shapes + relations -> neighbors, valid, diagnostics

Stage 2: run model repeatedly
         Q, K, V + cached CUDA neighbors/valid -> Triton attention
```

The current benchmark proves Stage 2 can win. The next engineering step is to make the model
actually use Stage 2 without re-entering Stage 1.

---

## Next Step

Make cached sparse block timing approximate the kernel-level result.

Concrete target:

```text
sparse_block_cached ~= dense_block + Triton_sparse_attention_overhead
```

Immediate work:

1. Precompute node embeddings and topology outside `forward`.
2. Store `neighbors` and `valid` as CUDA tensors in the cache.
3. Add a no-diagnostics/no-router fast path for benchmark and training.
4. Surface scored-topK exceptions instead of swallowing them.
5. Re-run the benchmark with block-level timing after the hot path is cleaned.

Success criterion:

```text
s_c at n=512 should fall from hundreds of ms to low single-digit ms.
```
