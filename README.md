# Math-Routed Sparse Transformer

A neurosymbolic transformer prototype that converts mathematical expression structure into sparse attention neighborhoods, then executes the sparse attention path with a fused CUDA/Triton kernel.

The project tests this architecture:

```text
math expression graph
    -> symbolic / semantic topology
    -> cached neighbor table
    -> Triton neighbor-sparse attention
    -> transformer block
```

## Current Status

The current CUDA benchmark shows that the Triton neighbor-sparse attention kernel is faster than dense attention at all tested sizes from `n=64` to `n=512`.

At `n=512`:

| mode | dense attention | Triton sparse attention | speedup |
|---|---:|---:|---:|
| roots | 0.590 ms | 0.214 ms | 2.76× |
| trees | 0.590 ms | 0.206 ms | 2.86× |

The cached sparse block hot path is also fixed. The shared topology cache now works correctly and avoids repeated topology construction.

At `n=512`:

| mode | sparse uncached block | sparse cached block | improvement |
|---|---:|---:|---:|
| roots | 503.333 ms | 2.808 ms | 179× |
| trees | 254.872 ms | 2.012 ms | 127× |

The remaining gap is block-level overhead: cached sparse block is still slower than the dense full block, but now by low single-digit factors rather than hundreds of times.

## Repository Layout

```text
src/
  attention.py          Dense and neighbor-sparse attention modules
  triton_attention.py   Fused Triton neighbor-sparse attention kernel
  sparse_attention.py   PyTorch sparse reference implementation and neighbor helpers
  model.py              Math-routed transformer block/model
  topology.py           Symbolic/semantic topology construction
  topology_cache.py     Cached topology and frozen node embeddings
  eval.py               Benchmark CLI and reporting

scripts/
  benchmark_attention.sh  CUDA/Triton benchmark entrypoint

tests/
  test_triton_attention.py
  test_sparse_attention.py
  test_attention_mask.py
```

## Requirements

The benchmark path is CUDA/Triton-only. CPU fallback for sparse benchmark execution has been removed.

Use the CUDA virtual environment when present:

```bash
.venv-cuda/bin/python
```

The benchmark script selects the CUDA environment automatically if available:

```bash
scripts/benchmark_attention.sh
```

Expected runtime header:

```text
device=cuda  threads=4  python=3.12.13
```

If CUDA is unavailable, the benchmark fails fast instead of silently running the wrong CPU path.

## Quick Start

From the repository root:

```bash
cd /workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer
scripts/benchmark_attention.sh --sizes 64,128,256,512 --node-mode roots,trees
```

Run Triton correctness tests:

```bash
.venv-cuda/bin/python -m pytest -q tests/test_triton_attention.py
```

Run sparse attention reference tests:

```bash
.venv-cuda/bin/python -m pytest -q tests/test_sparse_attention.py tests/test_attention_mask.py
```

## Benchmark Columns

| column     | meaning                                                         |
|------------+-----------------------------------------------------------------|
| `n`        | number of math nodes / tokens                                   |
| `mode`     | node collection mode: `roots` or `trees`                        |
| `allowed`  | allowed sparse attention edges                                  |
| `full`     | dense full-attention edges, equal to `n²`                       |
| `avg_k`    | average allowed neighbors per token                             |
| `max_k`    | maximum allowed neighbors before truncation                     |
| `rel_red`  | relation reduction versus dense full attention                  |
| `topo_ms`  | topology build time                                             |
| `d_attn`   | dense full attention time                                       |
| `s_trnc`   | truncated sparse attention time                                 |
| `s_comp`   | old `torch.compile` sparse path; now unused in CUDA/Triton path |
| `s_tri`    | fused Triton sparse attention time                              |
| `stk_bld`  | scored-topK topology build time                                 |
| `stk_atn`  | scored-topK attention time; currently invalid/zero until fixed  |
| `amrt_10`  | topology amortized over 10 forwards                             |
| `amrt_100` | topology amortized over 100 forwards                            |
| `d_blk`    | dense/full transformer block time                               |
| `s_uc`     | sparse block with uncached topology                             |
| `s_c`      | sparse block with cached topology and metadata fast path        |

## Architecture Modes

### `roots`

Uses repeated normalized expression roots as nodes. It is more sparse and faster.

At `n=512`:

```text
allowed edges: 59,334 / 262,144
relation reduction: 77.37%
symbolic_dependency: 0
```

### `trees`

Uses full expression-tree nodes. It is less sparse but contains real symbolic dependency edges.

At `n=512`:

```text
allowed edges: 136,648 / 262,144
relation reduction: 47.87%
symbolic_dependency: 26,060
```

## Key Implementation Notes

### Triton Sparse Attention

The main kernel is:

```python
src/triton_attention.py::_nbr_sparse_attn_kernel
```

It computes:

```text
for each token t:
  gather neighbor keys/values
  score q_t against each neighbor key
  softmax over valid neighbors
  weighted sum neighbor values
```

The wrapper is:

```python
triton_neighbor_attention(q, k, v, neighbors, valid)
```

It requires CUDA tensors and fails fast otherwise.

### Cached Sparse Block Hot Path

The cache bug fixed in the current version was this pattern:

```python
cache = self._topology_cache or TopologyCache(maxsize=1)
```

Because `TopologyCache` implements `__len__`, an empty cache was falsy and got discarded before warming.

The fixed pattern is:

```python
cache = (
    self._topology_cache
    if self._topology_cache is not None
    else TopologyCache(maxsize=1)
)
```

The benchmark uses:

```python
return_metadata=False
```

for cached sparse block timing. This skips router/diagnostic return work on the hot path while preserving the default API behavior.

## Current Conclusions

What is proven:

```text
- CUDA/Triton neighbor-sparse attention is active.
- Triton sparse attention beats dense attention for n=64..512 on the current CUDA run.
- Cached sparse block no longer rebuilds topology every forward.
- Topology construction is the dominant cost when uncached.
```

What remains:

```text
- Make cached sparse block faster than dense full block.
- Fix scored-topK attention timing; stk_atn currently reports 0.000.
- Reduce sparse block overhead around the Triton call.
- Prove end-to-end model/training speedup.
```

## Development Priorities

1. Precompute `valid_i8` so the Triton wrapper does not convert the boolean valid mask every call.
2. Avoid unnecessary `.contiguous()` calls in the hot path.
3. Reduce projection/collect overhead around sparse attention.
4. Add a direct precompiled block path: `x + cached CUDA neighbors/valid -> Triton -> output`.
5. Surface scored-topK timing exceptions instead of swallowing them.
6. Benchmark at `n=1024+` where sparse advantage should widen.

## License

No license file is currently defined in this repository.
