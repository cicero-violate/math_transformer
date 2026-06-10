# Math-Routed Sparse Transformer

A neurosymbolic transformer prototype that converts mathematical expression structure into sparse attention neighborhoods, then executes the sparse attention path with a fused CUDA/Triton kernel.

The project has gone through fourteen planning iterations. The current architecture is:

```text
math expression graph
    -> symbolic / semantic / learned topology
    -> cached neighbor table (offline; O(T*K) hot path)
    -> Triton block-token sparse attention
    -> transformer block
```

## Proven Results

### Prepared static sparse block beats dense block

At `n=1024, K=16, trees, middle_preserving_topk`:

```text
d_blk  = 2.398 ms   (dense full block)
s_c    = 2.017 ms   (cached sparse block)
p_blk  = 0.941 ms   (prepared static sparse block)
```

Speedups:

```text
prepared vs dense block  = 2.55x
prepared vs cached block = 2.14x
cached vs dense block    = 1.19x
```

The prepared fast path uses pre-installed static topology buffers and avoids topology lookup overhead entirely.

### Triton sparse attention vs dense attention

At `n=1024, K=16, trees`:

```text
d_attn = ~1.96 ms
s_tri  = ~0.30 ms
kernel speedup ≈ 6.5x
```

### Block-token Triton kernel

The default CUDA path uses one Triton program per `(batch, head, token-block)` with `block_t=2`:

```text
block_t=2 median_ms = 0.605
one-token median_ms = 0.617
speedup             = 1.02x
```

### Learned topology quality vs hand topology

Learned scorer (`K=6`) vs hand topology (`K=16`) on `synthetic_hard/val.jsonl`:

```text
hand K=16    route_acc = 0.9821
learned K=6  route_acc = 0.9871
```

Learned topology achieves better route accuracy at a smaller neighbor budget.

### v7 middle-preserving topology

Fixed-K topology with explicit middle-context bridge edges:

```text
allowed = 16,384 (exactly T*K)
middle_bridge = 2,907 edges surviving top-K selection
route accuracy with K=16 = 1.0000
dense_agree = 1.0000
```

The graph itself is now fixed-K, not a large union mask with runtime truncation.

### Topology build is offline-only

At `n=1024`, topology construction takes 1–2 seconds. It cannot be amortized into a live forward path. The architecture is strictly two-stage:

```text
Stage 1 (offline): compile/cache topology once
Stage 2 (inference): cached CUDA neighbors + valid_i8 -> Triton kernel
```

## What Has Not Yet Been Proven

```text
- Universal cached sparse block speedup (roots mode at n=1024 still loses)
- Material full-block speedup including learned scorer construction
- Token-pair learned topology at N=4096 (OOMs; block topology pivot addresses this)
- Stable tree block-level speedup across repeated benchmark runs
- Quality retention on harder held-out template families
- End-to-end training or inference speedup for the full model
```

## Repository Layout

```text
src/
  ir.py                         MathNode IR, op_class
  parser.py                     Expression parser
  embedder.py                   MathEmbedder, frozen node embeddings
  topology.py                   Symbolic/semantic topology builder (union, scored_topk)
  middle_preserving_topology.py Middle-bridge relation and fixed-K builder
  topology_cache.py             TopologyCache, CachedTopology, PreparedTopology, PagedNeighborTable
  topology_search.py            Offline relation-weight search loop
  learned_topology.py           Edge feature tensors and learned topology scorer model
  learned_topology_runtime.py   Runtime wrapper for learned scorer inference
  block_topology.py             BlockTopologyConfig, PreparedBlockTopology dataclasses
  block_learned_topology.py     HeuristicBlockTopologyBuilder (O(B^2), no N^2 scoring)
  attention.py                  DenseMaskedMathAttention, NeighborSparseMathAttention
  triton_attention.py           Triton sparse attention kernels (token and block-token)
  sparse_attention.py           PyTorch sparse reference implementation
  model.py                      MathRoutedTransformerBlock, MathRoutedTransformer
  router.py                     OperatorRouter
  normalize.py                  Expression normalization
  shape.py                      Shape/type compatibility
  verifier.py                   Expression verifier
  tasks.py                      Task definitions
  synthetic_data.py             Deterministic route/shape synthetic data generator
  dense_mix_sweep.py            Dense-sparse mix sweep utility
  train.py                      Training loop
  train_topology_scorer.py      Topology scorer training
  eval.py                       Benchmark CLI, quality eval, paired learned benchmark
  eval_topology_scorer.py       Scorer evaluation utilities
  export_learned_topology_failures.py  Failure export for scorer debugging

scripts/
  benchmark_attention.sh            CUDA/Triton attention benchmark
  benchmark_learned_topology.sh     Paired hand vs learned topology benchmark
  train_synthetic.sh                Generate data + train + quality eval
  train_topology_scorer.sh          Train topology scorer from dense teacher traces
  eval_topology_scorer.sh           Evaluate scored topology quality
  export_dense_teacher_traces.sh    Export teacher attention traces for scorer training
  self_play.sh                      Self-play topology improvement loop
  sweep_dense_mix.sh                Dense/sparse mix ratio sweep
  sweep_learned_k.sh                Learned K sweep
  train_hard_synthetic.sh           Train on synthetic_hard dataset
  run_hard_quality_check.sh         Quality gate for synthetic_hard
  run_learned_topology_quality.sh   Learned topology quality check
  select_best_runtime_scorer.sh     Select best scorer checkpoint by runtime metric
  profile_sparse_block.py           Sparse block profiler
  generate_hard_synthetic.sh        Generate synthetic_hard dataset

configs/
  tiny.yaml               Small model for quick iteration
  debug.yaml              Minimal config for CI/plumbing
  benchmark.yaml          Standard benchmark config
  synthetic_hard.yaml     Harder synthetic training config
  synthetic_overnight.yaml  Long overnight training config

tests/
  test_triton_attention.py
  test_sparse_attention.py
  test_attention_mask.py
  test_topology.py
  test_topology_cache.py
  test_priority_neighbors.py
  test_prepared_topology.py
  test_model_env.py
  test_learned_topology.py
  test_learned_topology_runtime.py
  test_middle_preserving_topology.py
  test_scored_topk.py
  test_truncated_sparse.py
  test_router.py
  test_ir.py
  test_parser.py
  test_shape.py
  test_verifier.py
  test_synthetic_data.py
  test_train_paths.py
  test_quality_eval.py
  test_dense_mix_sweep.py
  test_eval_topology_scorer.py
  test_topology_scorer_dense_teacher.py
  test_topology_scorer_resume.py
  test_topology_scorer_runtime_quality.py
  test_topology_search.py
  test_train_topology_scorer.py

data/
  examples.jsonl
  synthetic/           10k/1k/1k split (train/val/test)
  synthetic_hard/      Harder distribution split

configs/
  (see above)
```

## Requirements

The benchmark path is CUDA/Triton-only. Use the CUDA virtual environment:

```bash
.venv-cuda/bin/python
```

The benchmark scripts select the CUDA environment automatically:

```bash
scripts/benchmark_attention.sh
scripts/benchmark_learned_topology.sh
```

Expected runtime header:

```text
device=cuda  threads=4  python=3.12.13
```

If CUDA is unavailable the benchmark fails fast rather than silently running the wrong CPU path.

## Quick Start

### Attention benchmark

```bash
scripts/benchmark_attention.sh --sizes 64,128,256,512,1024 --node-mode roots,trees
```

With fixed-K middle-preserving topology:

```bash
scripts/benchmark_attention.sh \
  --sizes 1024 \
  --node-mode trees \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1 \
  --profile-prepared-block
```

### Learned topology benchmark

```bash
SCORER=runs/checkpoints/scorer_runtime_j_best.pt \
CHECKPOINT=runs/checkpoints/synthetic_hard_dense.pt \
BENCH_N=1024 \
scripts/benchmark_learned_topology.sh
```

### Train on synthetic data

```bash
scripts/train_synthetic.sh
```

With explicit sizes and checkpoint:

```bash
TRAIN=50000 VAL=5000 TEST=5000 SEED=1 \
MAX_STEPS=5000 EVAL_INTERVAL=500 \
CHECKPOINT=runs/checkpoints/my_run.pt \
scripts/train_synthetic.sh
```

### Quality evaluation

```bash
.venv-cuda/bin/python -m src.eval \
  --quality \
  --quality-k 16,32,64,128 \
  --examples data/synthetic/val.jsonl \
  --checkpoint runs/checkpoints/synthetic_big.pt
```

### Run tests

```bash
.venv-cuda/bin/python -m pytest -q tests/
```

Triton correctness tests only:

```bash
.venv-cuda/bin/python -m pytest -q tests/test_triton_attention.py
```

## Topology Modes

### `union`

Default. Computes symbolic relations and takes their union. Produces a large sparse mask, truncated to `max_neighbors` at inference time. Not recommended for large `n`.

### `scored_topk`

Weights symbolic relations by learned/hand-tuned coefficients and selects the top-K neighbors per token at build time. The graph itself is fixed-K.

### `middle_preserving_topk`

Extends `scored_topk` with an explicit `middle_bridge` relation that preserves tokens far from endpoints. Default weight `0.7`, priority `4`. This is the current recommended topology for quality+speed.

```bash
--topology-mode middle_preserving_topk --fixed-k 16 --middle-bridge-width 1
```

### `learned_block_topk`

Block-pair topology. Scores `B×B` block pairs instead of `N×N` token pairs. Avoids the `O(N²)` scoring OOM that token-pair learned topology hits at `N=4096`.

```bash
--topology-mode learned_block_topk --block-size 64 --topk-blocks 4 --block-token-cap 16
```

## Node Modes

### `roots`

Repeated normalized expression roots as tokens. More sparse, faster topology build.

At `n=512`:

```text
allowed edges: 59,334 / 262,144   (rel_red = 77.37%)
symbolic_dependency: 0
```

### `trees`

Full expression-tree nodes. Less sparse but contains real symbolic dependency edges. The primary target for architecture validation.

At `n=512`:

```text
allowed edges: 136,648 / 262,144  (rel_red = 47.87%)
symbolic_dependency: 26,060
```

## Benchmark Columns

| column            | meaning                                                              |
|-------------------|----------------------------------------------------------------------|
| `n`               | number of math nodes / tokens                                        |
| `mode`            | node collection mode: `roots` or `trees`                             |
| `allowed`         | allowed sparse attention edges                                       |
| `full`            | dense full-attention edges, equal to `n²`                            |
| `avg_k`           | average allowed neighbors per token                                  |
| `max_k`           | maximum allowed neighbors before truncation                          |
| `rel_red`         | relation reduction versus dense full attention                       |
| `topo_ms`         | topology build time                                                  |
| `d_attn`          | dense full attention time                                            |
| `s_trnc`          | truncated sparse attention time                                      |
| `s_tri`           | fused Triton sparse attention time                                   |
| `stk_bld`         | scored-topK topology build time                                      |
| `stk_atn`         | scored-topK attention time                                           |
| `amrt_10`         | topology amortized over 10 forwards                                  |
| `amrt_100`        | topology amortized over 100 forwards                                 |
| `d_blk`           | dense/full transformer block time                                    |
| `s_uc`            | sparse block with uncached topology                                  |
| `s_c`             | sparse block with cached topology                                    |
| `p_blk`           | prepared static sparse block (no topology lookup at all)             |
| `p_attn`          | attention kernel time within the prepared block                      |
| `p_non`           | non-attention overhead within the prepared block                     |

## Selector Modes

| mode                     | description                                                        |
|--------------------------|--------------------------------------------------------------------|
| `topology_only`          | cached symbolic priority table → top-K; default                   |
| `kmip_only`              | live `q_i^T k_j` top-K over all tokens; much slower               |
| `symbolic_kmip`          | `alpha * q_i^T k_j + beta * R_symbolic`; slower, not yet worth it |
| `symbolic_candidate_kmip`| score only symbolic candidates then select K; overhead-heavy in PyTorch |

`topology_only` is the only selector that satisfies the speed objective at the current development stage.

## Key Implementation Notes

### Prepared static fast path

The primary hot path:

```python
block.prepare_static_topology(nodes, env, device=device)
out = block.forward_static_fast_path(x)
```

Or at model level:

```python
model.prepare_static_topology(nodes, env, device=device)
out = model.forward_static_fast_path(x)
```

This requires topology to be compiled once before inference. It stores `(neighbors, valid_i8)` as registered buffers and bypasses all topology lookup on every forward.

### Block-token Triton kernel

The default CUDA sparse attention path:

```python
NeighborSparseMathAttention.enable_block_token_attention = True
NeighborSparseMathAttention.block_token_attention_t = 2
```

One Triton program handles `block_t` consecutive tokens. This increases work per Triton program and reduces launch overhead.

### Topology cache key

`TopologyCache` keys include topology mode, `fixed_k`, and `middle_bridge_width`, preventing stale reuse across routing modes.

### TopologyCache falsy-cache fix

```python
# wrong: an empty TopologyCache is falsy (implements __len__)
cache = self._topology_cache or TopologyCache(maxsize=1)

# correct:
cache = self._topology_cache if self._topology_cache is not None else TopologyCache(maxsize=1)
```

### Paged neighbor table

`PagedNeighborTable` splits long-sequence neighbor tables into fixed-size row pages for storage efficiency:

```python
PagedNeighborTable(neighbor_pages, valid_pages, valid_i8_pages, length, page_size, k)
```

### Block topology for N≥4096

Token-pair learned topology OOMs at `N=4096` because `N²=16.7M` pairs are scored.

Block topology scores `B×B` pairs where `B = ceil(N / block_size)`:

```text
N=4096, block_size=64 -> B=64, B^2=4096 (4096x fewer score entries)
```

`HeuristicBlockTopologyBuilder` implements this without any `O(N²)` allocation.

## Learned Topology Pipeline

```text
dense teacher model
    -> export_dense_teacher_traces.sh (attention trace JSONL)
    -> train_topology_scorer.sh (trains LearnedTopologyScorer)
    -> scorer checkpoint

scorer checkpoint + test nodes
    -> learned_topology_runtime.py (token-pair scores)
    -> topK neighbor selection
    -> PreparedTopology (same interface as symbolic topology)
```

Train:

```bash
scripts/export_dense_teacher_traces.sh
scripts/train_topology_scorer.sh
```

Evaluate:

```bash
SCORER=runs/checkpoints/scorer_runtime_j_best.pt \
scripts/eval_topology_scorer.sh
```

Select best runtime-aligned checkpoint:

```bash
scripts/select_best_runtime_scorer.sh
```

## Training

```bash
.venv-cuda/bin/python -m src.train \
  --config configs/tiny.yaml \
  --data data/synthetic/train.jsonl \
  --attention-mode neighbor_sparse \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1 \
  --device cuda \
  --max-steps 5000 \
  --eval-interval 500 \
  --save-checkpoint runs/checkpoints/run.pt
```

## Offline Topology Search

Mutates scored-topK relation weights to improve quality without retraining:

```bash
.venv-cuda/bin/python -m src.topology_search \
  --checkpoint runs/checkpoints/run.pt \
  --k 16 \
  --iterations 1000 \
  --output runs/topology_search/best_weights.json
```

Evaluate the found weights:

```bash
.venv-cuda/bin/python -m src.eval \
  --quality \
  --topology-mode scored_topk \
  --fixed-k 16 \
  --quality-k 16 \
  --relation-weights-json runs/topology_search/best_weights.json \
  --checkpoint runs/checkpoints/run.pt
```

## Current v14 Focus

v14 pivots from token-pair learned topology to block-pair learned topology to resolve the `N=4096` OOM:

```text
S_token in R^(N x N)   -> OOM at N=4096
S_block in R^(B x B)   -> O(block_size^2) reduction
```

The `learned_block_topk` topology mode and `HeuristicBlockTopologyBuilder` implement the first (heuristic, non-trained) version.

Target:

```text
quality_ok=True
speed_ok=True
speedup >= 1.10x
no OOM for N in {1024, 2048, 4096}
```

## License

MIT. See `LICENSE`.
