# Math-Routed Sparse Transformer Plan v4

## Status

Project path:

```text
/workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer/
```

Current confirmed baseline:

```text
tests: 118 passed
neighbor sparse attention: implemented
attention_mode switch: implemented
roots/trees benchmark modes: implemented
relation diagnostics: implemented
shape_compat/composition: active when --examples data/examples.jsonl is passed
```

Current honest state:

```text
Gate 5 is real when benchmark receives examples/env
neighbor sparse correctness is real
full runtime win is not proven yet
block-level sparse remains slower at small n due to topology + neighbor conversion overhead
```

Current architecture stage:

```text
Math-Routed Transformer with real neighbor-sparse attention path, but incomplete env/caching integration
```

Target architecture stage:

```text
Cached, env-aware, benchmark-stable Math-Routed Neighbor-Sparse Transformer
```

---

## 1. Core Objective

Turn the current implementation from a correct sparse prototype into a credible performance system.

The next boundary is not adding more math ideas.

The next boundary is engineering correctness:

```text
pass shape/type env into the model
use the same relation matrices in benchmark and model
cache topology and neighbor lists
benchmark cached vs uncached sparse blocks
run larger-n timing
separate CPU overhead from attention-kernel savings
```

The core architecture remains:

```text
math object
  -> symbolic IR
  -> normalized expression
  -> math-function vectors Z
  -> relation matrices G, S, C, R, L, I
  -> allowed attention mask A
  -> priority matrix P
  -> neighbor list N(A,P)
  -> neighbor sparse attention
  -> route/expert head
  -> verifier
```

Where:

```text
G = symbolic dependency
S = shape compatibility
C = composition/type compatibility
R = embedding top-k
L = local window
I = identity/self attention
P = relation priority matrix
```

---

## 2. Verified Facts From Review

### 2.1 Tests pass

Current test baseline:

```text
118 passed
```

This is a strong correctness baseline.

Do not break it.

---

### 2.2 Gate 5 is real with examples/env

Command that reproduces real topology proof:

```text
python -m src.eval --benchmark --sizes 32 --node-mode roots,trees --examples data/examples.jsonl
```

Observed:

```text
roots n=32:
  symbolic_dependency: 0
  shape_compat: 70
  composition: 50
  embedding_topk: 120
  identity: 32

trees n=32:
  symbolic_dependency: 118
  shape_compat: 82
  composition: 98
  embedding_topk: 130
  identity: 32
```

Conclusion:

```text
relation matrices are implemented and can produce real math topology
```

But only when shape env is actually passed.

---

### 2.3 Default benchmark script hides shape/composition

Current wrapper issue:

```bash
python -m src.eval --benchmark
```

It does not pass:

```text
--examples data/examples.jsonl
```

and it does not forward user args.

Therefore default script can report:

```text
shape_compat: 0
composition: 0
```

This is a script issue, not a topology implementation issue.

---

### 2.4 Model forward does not receive env

Current model path builds topology like:

```python
np_mask = self.topology.build(nodes, z)
```

Missing:

```python
np_mask = self.topology.build(nodes, z, env)
```

Therefore actual model forward currently does not use shape/composition relations unless env support is added.

This is the most important correctness/integration gap.

---

### 2.5 Prioritized neighbor selection is benchmark-only

Benchmark uses:

```python
neighbors_from_mask_prioritized(mask, priority, max_k)
```

Model uses:

```python
neighbors_from_mask(mask, K)
```

With `K=max_k`, this does not change correctness because no truncation happens.

But once max neighbors are capped, model and benchmark will diverge.

Model must support prioritized neighbors before any fixed-K/truncated sparse attention benchmark is trusted.

---

### 2.6 Sparse kernel correctness is real

Tests show:

```text
neighbor_attention matches dense masked attention for the same mask
all-ones sparse matches dense full
self-only sparse returns v
NeighborSparseMathAttention module has shape/equivalence tests
neighbor_sparse model block runs
```

This is good.

Keep dense masked attention as the reference implementation.

---

### 2.7 Runtime speed is not proven end-to-end

Current truth:

```text
neighbor sparse computes fewer relation scores
CPU timing at small n is noisy
block-level sparse can be slower due to topology overhead
end-to-end speedup is not proven
```

Correct claim:

```text
computed relation reduction exists; runtime speedup requires caching/larger n/GPU-friendly execution
```

Incorrect claim:

```text
the full model is already faster
```

---

## 3. Immediate Fixes

## Fix 1 — Benchmark script forwarding

File:

```text
scripts/benchmark_attention.sh
```

Current:

```bash
python -m src.eval --benchmark
```

Replace with:

```bash
python -m src.eval --benchmark --examples data/examples.jsonl "$@"
```

Acceptance:

```text
scripts/benchmark_attention.sh --sizes 32 --node-mode roots,trees
```

must produce nonzero:

```text
shape_compat
composition
```

for the relevant cases.

---

## Fix 2 — Pass env into model forward

Files:

```text
src/model.py
src/train.py
src/eval.py
tests/test_model_env.py
```

Change block API:

```python
def forward(self, x, nodes=None, env=None):
    ...
```

Change topology build:

```python
np_mask = self.topology.build(nodes, z, env)
```

Change transformer API:

```python
def forward(self, x, nodes=None, env=None):
    for layer in self.layers:
        x, mask, route_info = layer(x, nodes, env=env)
```

Acceptance test:

```python
out, masks, _ = model(x, nodes, env=env)
assert masks[0] includes shape_compat/composition-derived edges
```

Need diagnostic access to confirm this.

Recommended return extension:

```python
return x, mask, route_info, diagnostics
```

or optional debug mode:

```python
model.forward(..., return_diagnostics=True)
```

---

## Fix 3 — Use priority matrix in model neighbor_sparse mode

Files:

```text
src/model.py
src/topology.py
src/sparse_attention.py
```

Current model:

```python
nb, valid = neighbors_from_mask(mask, K)
```

Target:

```python
priority = build_priority_matrix(nodes, z=z, env=env, topk=topk, local_window=local_window)
nb, valid = neighbors_from_mask_prioritized(mask, priority, K)
```

Acceptance:

```text
model neighbor order matches benchmark neighbor order
identity/self edges are retained first
symbolic edges outrank same_operator edges
```

---

## Fix 4 — Add topology cache

This is the key for block-level speed.

Create:

```text
src/topology_cache.py
```

Cache object:

```python
@dataclass
class CachedTopology:
    mask: torch.Tensor
    priority: np.ndarray
    neighbors: torch.Tensor
    valid: torch.Tensor
    diagnostics: MaskDiagnostics
```

Cache key:

```text
nodes_hash + env_hash + topk + local_window + max_neighbors + relation_flags
```

Recommended functions:

```python
def stable_nodes_hash(nodes: list[MathNode]) -> str:
    ...

def stable_env_hash(env: dict[str, tuple[int, ...]] | None) -> str:
    ...

class TopologyCache:
    def get_or_build(self, nodes, z, env, builder, max_neighbors=None) -> CachedTopology:
        ...
```

Acceptance:

```text
first forward builds topology
second forward reuses topology
cached sparse block is faster than uncached sparse block
```

Benchmark columns:

```text
s_blk_uncached
s_blk_cached
```

---

## Fix 5 — Add fixed-K sparse mode

Current neighbor sparse uses:

```text
K = max_k_from_mask(mask)
```

This preserves exact equivalence but may waste compute due to padding.

Add optional cap:

```text
max_neighbors
```

Modes:

```text
exact_sparse: K=max_k_from_mask(mask), matches dense masked exactly
truncated_sparse: K=max_neighbors, approximate but faster
```

Use priority order for truncation:

```text
1 identity
2 symbolic_dependency
3 composition
4 shape_compat
5 embedding_topk
6 local_window
7 same_operator
```

Acceptance:

```text
exact_sparse matches dense masked
truncated_sparse keeps all identity edges
truncated_sparse keeps symbolic_dependency before same_operator
quality impact is measured, not assumed
```

---

## 4. Benchmark v4 Requirements

The benchmark must report three layers:

```text
attention-only
block-level
end-to-end
```

It must also distinguish:

```text
uncached sparse
cached sparse
exact sparse
truncated sparse
```

---

## 4.1 Attention-only benchmark

Purpose:

```text
measure pure dense vs sparse attention kernel cost
```

Inputs:

```text
same q,k,v
same mask
same neighbor list
```

Columns:

```text
n
mode
allowed
full
avg_k
max_k
K_used
padding_ratio
sparsity
rel_reduce
dense_full_attn_ms
dense_masked_attn_ms
neighbor_sparse_exact_ms
neighbor_sparse_trunc_ms
```

Gate:

```text
sparse computes fewer dot products
runtime may or may not be faster on CPU at small n
```

---

## 4.2 Block-level benchmark

Purpose:

```text
measure actual transformer block path
```

Columns:

```text
full_block_ms
dense_masked_block_ms
sparse_block_uncached_ms
sparse_block_cached_ms
```

Gate:

```text
sparse_block_cached_ms < sparse_block_uncached_ms
```

The stronger gate:

```text
sparse_block_cached_ms < full_block_ms
```

Only claim full block speedup after this passes.

---

## 4.3 End-to-end benchmark

Purpose:

```text
measure whole system cost
```

Includes:

```text
parse
normalize
embed
topology
neighbor conversion
attention
router
verifier
```

Columns:

```text
full_e2e_ms
masked_e2e_ms
sparse_e2e_uncached_ms
sparse_e2e_cached_ms
```

Gate:

```text
if sparse_e2e_cached_ms is not faster, report that honestly
```

---

## 4.4 Timing hygiene

Small CPU timing is noisy.

Add:

```text
median_ms
p50_ms
p95_ms
min_ms
std_ms
```

At minimum:

```text
median and min
```

Set or report:

```text
torch.get_num_threads()
device
python version
```

For CUDA later:

```python
torch.cuda.synchronize()
```

before and after timed blocks.

---

## 5. Required Tests

### 5.1 Script forwarding test

Test command:

```text
scripts/benchmark_attention.sh --sizes 32 --node-mode roots --iters 1 --warmup 1
```

Expected:

```text
shape_compat > 0
composition > 0
```

because script passes examples by default.

---

### 5.2 Env-aware model test

Test:

```python
model = MathRoutedTransformer(attention_mode="dense_masked")
out, masks, routes, diags = model(x, nodes, env=env, return_diagnostics=True)
assert diags[0].by_relation["shape_compat"] > 0
assert diags[0].by_relation["composition"] > 0
```

---

### 5.3 Priority parity test

Benchmark and model should create the same neighbors for the same:

```text
nodes
env
topk
local_window
max_neighbors
```

Test:

```python
nb_model == nb_benchmark
valid_model == valid_benchmark
```

---

### 5.4 Cache correctness test

Test:

```text
cached mask equals uncached mask
cached neighbors equal uncached neighbors
cached diagnostics equal uncached diagnostics
```

---

### 5.5 Cache speed smoke test

Not a strict unit test, but benchmark should show:

```text
cached topology builds fewer times than uncached
```

Track:

```text
cache_hits
cache_misses
```

---

### 5.6 Truncated sparse safety test

For `max_neighbors < max_k`:

```text
identity edges are always retained
symbolic edges are retained before same_operator
no row has zero valid neighbors
```

---

## 6. Persistence Plan

Persist durable state, not temporary tensors.

Create runtime artifact structure:

```text
runs/
  benchmarks/
  topology_cache/
  eval_reports/
  model_checkpoints/
  failure_sets/
```

Persist benchmark reports:

```json
{
  "n": 32,
  "mode": "trees",
  "device": "cpu",
  "full_edges": 1024,
  "allowed_edges": 562,
  "avg_k": 17.6,
  "max_k": 24,
  "padding_ratio": 0.268,
  "by_relation": {
    "symbolic_dependency": 118,
    "shape_compat": 82,
    "composition": 98
  },
  "timings": {
    "dense_full_attn_ms": 0.057,
    "dense_masked_attn_ms": 0.079,
    "neighbor_sparse_attn_ms": 0.274,
    "sparse_block_cached_ms": null
  }
}
```

Persist topology cache only when useful:

```text
mask
priority
neighbors
valid
diagnostics
cache key
```

Do not persist:

```text
Q/K/V activations
full dense attention matrices
temporary forward tensors
```

---

## 7. Development Sequence

### Sprint 1 — CLI and env correctness

Files:

```text
scripts/benchmark_attention.sh
src/model.py
src/eval.py
tests/test_model_env.py
```

Deliverables:

```text
script forwards args
script passes examples by default
model receives env
model diagnostics prove shape/composition active
```

Acceptance:

```text
118+ tests pass
benchmark script shows shape_compat/composition nonzero by default
```

---

### Sprint 2 — Priority parity

Files:

```text
src/model.py
src/topology.py
src/sparse_attention.py
tests/test_priority_neighbors.py
```

Deliverables:

```text
model uses priority matrix
benchmark/model neighbor generation matches
```

---

### Sprint 3 — Topology cache

Files:

```text
src/topology_cache.py
src/model.py
src/eval.py
tests/test_topology_cache.py
```

Deliverables:

```text
cache key
cached topology object
cache hits/misses
cached sparse block benchmark
```

Acceptance:

```text
s_blk_cached < s_blk_uncached
```

---

### Sprint 4 — Larger-n benchmark

Run:

```text
scripts/benchmark_attention.sh --sizes 32,64,128,256 --node-mode roots,trees --iters 30 --warmup 10
```

If too slow, separate:

```text
attention-only large n
block-level medium n
end-to-end small/medium n
```

Deliverable:

```text
runs/benchmarks/v4_large_n.json
```

---

### Sprint 5 — Fixed-K truncated sparse

Files:

```text
src/sparse_attention.py
src/model.py
src/eval.py
tests/test_truncated_sparse.py
```

Deliverables:

```text
max_neighbors option
exact sparse mode
truncated sparse mode
priority-safe truncation
quality/runtime comparison
```

---

## 8. Success Gates v4

### Gate A — Script correctness

```text
scripts/benchmark_attention.sh --sizes 32 --node-mode roots,trees
```

must show:

```text
shape_compat > 0
composition > 0
```

for env-backed benchmarks.

---

### Gate B — Env-aware model

Actual model forward must use:

```text
shape_compat
composition
```

not only benchmark diagnostics.

---

### Gate C — Cache effectiveness

```text
sparse_block_cached_ms < sparse_block_uncached_ms
```

This is the immediate performance gate.

---

### Gate D — Block speedup

```text
sparse_block_cached_ms < dense_full_block_ms
```

Only after this passes can we claim block-level speedup.

---

### Gate E — End-to-end honesty

Report one of:

```text
sparse_e2e_cached faster than dense_e2e
```

or:

```text
sparse attention is faster but end-to-end remains overhead-bound
```

Both are acceptable if reported honestly.

---

### Gate F — Quality preserved

Sparse exact mode must preserve:

```text
same output as dense masked within tolerance
same route outputs where expected
same verifier behavior
```

Truncated mode must report quality delta.

---

## 9. Current Recommended Claims

Valid claims now:

```text
The system builds real math relation matrices.
The system can convert relation masks into neighbor sparse attention.
The sparse attention implementation matches dense masked attention for the same mask.
The model has an attention_mode switch and can execute neighbor_sparse blocks.
Shape/composition relations are real when env is passed.
```

Invalid or premature claims:

```text
The whole transformer is faster.
The benchmark proves end-to-end speedup.
Shape/composition are active in model forward by default.
Sparse attention is always faster on CPU at small n.
```

Target claim after v4:

```text
The cached env-aware neighbor-sparse transformer block can reduce computed attention relations and can be benchmarked fairly against dense full and dense masked blocks.
```

Stronger target claim after v4 if Gate D passes:

```text
For math-structured sequences at sufficient n, cached neighbor-sparse attention produces block-level runtime speedup while preserving dense-masked correctness.
```

---

## 10. Final v4 Architecture

```text
Math-Routed Sparse Transformer v4 =
  symbolic IR
  + normalized math expressions
  + env-aware shape/type metadata
  + relation matrices G/S/C/R/L/I
  + priority matrix P
  + cached topology object
  + exact or truncated neighbor list
  + neighbor sparse attention in real model forward
  + dense masked reference path
  + route head
  + verifier
  + benchmark/persistence reports
```

Core thesis:

```text
Do not use dense attention to rediscover math topology.
Build topology from symbolic and learned math relations.
Cache it.
Compute attention only over meaningful neighbors.
Verify correctness separately from neural routing.
```

