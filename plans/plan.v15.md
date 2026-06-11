# plan.v15.md — Native Block-Sparse Attention for Material Hot-Path Speedup

## Context

v14 proved that block-level learned topology solves the scaling and preparation problem.

Observed v14 result at `N=4096`, `block_size=64`, `topk_blocks=4`, `HAND_K=16`, `LEARNED_K=16`:

```text
mode=topology_only       k=16 route_acc=0.9821
mode=learned_block_topk  k=16 route_acc=0.9832
quality_ok=True
```

Prepare time improved massively:

```text
hand topology_prepare_ms  = 18825.108081
block topology_prepare_ms =   139.708112
prepare_speedup           ≈ 134.75x
```

But prepared hot-path speed did not materially improve:

```text
hand prepared_block_ms       = 2.796351
block prepared_block_ms      = 2.802018
prepared_speedup             = 0.997977
hand prepared_attention_ms   = 1.089718
block prepared_attention_ms  = 1.097860
```

v14 therefore validates the representation but not the kernel path. The remaining bottleneck is that selected blocks are expanded back into token-neighbor tables and executed by the existing token sparse attention kernel.

---

## v15 Objective

Implement native block-sparse attention that consumes block topology directly.

Target:

```text
quality_ok=True
no OOM at N=4096
prepared_attention_ms < hand K=16 prepared_attention_ms
prepared_block_ms < hand K=16 prepared_block_ms
prepared_speedup >= 1.10x
```

Minimum acceptable:

```text
route_acc >= hand K=16 route_acc
native block attention faster than token-neighbor block expansion
no OOM at N=4096
clear timing explanation if prepared_speedup < 1.10x
```

---

## Core Pivot

Current v14 runtime:

```text
block_neighbors(B, K_B)
  -> token_neighbors(N, K)
  -> token-neighbor sparse attention
```

Desired v15 runtime:

```text
block_neighbors(B, K_B)
  -> native block-sparse attention
```

Definitions:

```text
B = ceil(N / block_size)
S_block in R^(B x B)
```

For `N=4096`, `block_size=64`:

```text
B = 64
B^2 = 4096
```

Native block-sparse attention computes:

```text
for query block i:
    selected key blocks = block_neighbors[i]
    attend Q_i over K/V tokens in selected key blocks
```

The kernel should exploit:

```text
contiguous block loads
fixed block_size tiles
small K_B
block-level reuse
```

---

## Task 1 — Freeze v14 Decision Evidence

Add a v14 conclusion to benchmark output when `BLOCK_TOPOLOGY=1`:

```text
block_topology_quality_ok=True
block_topology_no_oom=True
block_topology_prepare_speedup=...
block_topology_prepared_speedup=...
native_block_sparse_required=True
v14_decision=pivot_native_block_sparse
```

Acceptance:

```text
benchmark output clearly says v14 solved prepare/scaling but not hot-path speed
```

---

## Task 2 — Preserve Block Fields Through PreparedTopology

v14 currently prepares block topology but then primarily exposes token-neighbor fields.

Extend prepared topology with optional block fields:

```python
block_neighbors: torch.Tensor | None
block_valid_i8: torch.Tensor | None
block_size: int | None
is_block_topology: bool
```

Acceptance for `learned_block_topk`:

```text
prepared.neighbors.shape == (N, K)
prepared.block_neighbors.shape == (B, K_B)
prepared.block_valid_i8.shape == prepared.block_neighbors.shape
prepared.block_size == BLOCK_SIZE
prepared.is_block_topology is True
```

Existing token-neighbor tests must still pass.

---

## Task 3 — Add Reference Block-Sparse Attention

Create:

```text
src/block_sparse_attention.py
```

Add:

```python
def block_sparse_attention_reference(q, k, v, block_neighbors, block_valid_i8, block_size):
    ...
```

Expected shapes:

```text
q,k,v:              (batch, heads, N, d_head)
block_neighbors:   (B, K_B)
block_valid_i8:    (B, K_B)
output:            (batch, heads, N, d_head)
```

Correctness reference:

```text
build equivalent dense block mask
compare reference block sparse output to dense masked attention output
```

Acceptance:

```text
works for N in {127, 128, 129, 1024, 4096}
no NaNs
matches dense masked block attention within tolerance
```

---

## Task 4 — Add Native Block Attention Mode

Expose a new mode:

```text
attention_mode="block_sparse"
```

or CLI flag:

```bash
--native-block-sparse-attn
```

Wrapper env var:

```bash
NATIVE_BLOCK_SPARSE=1
```

Acceptance command:

```bash
BLOCK_TOPOLOGY=1 \
NATIVE_BLOCK_SPARSE=1 \
BENCH_N=4096 \
scripts/benchmark_learned_topology.sh
```

must complete without OOM.

---

## Task 5 — Implement Fast Native Block-Sparse Kernel

After the reference path is correct, add an optimized implementation.

Proposed file:

```text
src/triton_block_sparse_attention.py
```

Kernel inputs:

```text
q,k,v
block_neighbors
block_valid_i8
block_size
```

Work unit:

```text
(batch, head, query_block)
```

Kernel algorithm:

```text
load Q query block contiguously
for each selected key block:
    load K/V block contiguously
    update online softmax accumulator
write output block contiguously
```

Acceptance:

```text
native_block_attention_ms < token_neighbor_attention_ms
native_block_prepared_block_ms < hand_prepared_block_ms
no OOM at N=4096
correctness close to reference
```

---

## Task 6 — Benchmark Matrix

Run:

```bash
for N in 1024 2048 4096; do
  for MODE in hand token_neighbor_block native_block_reference native_block_fast; do
    ...
  done
done
```

Required columns:

```text
N
mode
route_acc
block_count
block_size
topk_blocks
prepare_ms
learned_scorer_ms
neighbor_table_build_ms
prepared_block_ms
prepared_attention_ms
prepared_non_attention_ms
total_with_prepare_ms
speedup_vs_hand_prepared
speedup_vs_token_neighbor_block
oom
quality_ok
speed_ok
```

---

## Task 7 — Tests

Add:

```text
tests/test_block_sparse_attention.py
tests/test_native_block_sparse_benchmark.py
```

Required cases:

```text
N < block_size
N == block_size
N not divisible by block_size
N = 4096
K_B = 1
K_B > 1
invalid block entries are masked
self block included
local blocks included
```

Acceptance:

```bash
python -m pytest tests/test_block_sparse_attention.py -q
```

passes on CPU and CUDA where available.

---

## Task 8 — Decision Rules

Promote v15 if:

```text
quality_ok=True
no_oom_at_4096=True
native_block_attention_ms < token_neighbor_attention_ms
native_block_prepared_block_ms < hand_prepared_block_ms
prepared_speedup >= 1.10
```

Tune kernel if:

```text
quality_ok=True
no_oom_at_4096=True
native_block_attention_ms < token_neighbor_attention_ms
prepared_speedup < 1.10
```

Reject native path if:

```text
correctness fails
or native_block_attention_ms >= token_neighbor_attention_ms
```

---

## Expected Deliverables

1. `plan.v15.md` documents the native block-sparse pivot.
2. Prepared topology preserves block-neighbor fields.
3. Reference block-sparse attention exists and is tested.
4. Native fast block-sparse attention exists behind a flag.
5. Benchmark wrapper supports:

```bash
BLOCK_TOPOLOGY=1
NATIVE_BLOCK_SPARSE=1
```

6. Benchmark table covers `N in {1024, 2048, 4096}`.
7. v15 either achieves `prepared_speedup >= 1.10x` or identifies the next kernel bottleneck.

---

## Formula Summary

```text
B = ceil(N / block_size)
```

```text
S_token in R^(N x N)
S_block in R^(B x B)
```

```text
T_total = T_prepare + T_prepared_block
```

```text
T_prepared_block = T_attention + T_non_attention
```

```text
speedup_prepared = hand_prepared_block_ms / native_block_prepared_block_ms
```

```text
speedup_attention = token_neighbor_attention_ms / native_block_attention_ms
```

```text
quality_ok = native_block_route_acc >= hand_route_acc
```

v15 success requires:

```text
quality_ok and speedup_prepared >= 1.10 and no_oom_at_4096
```

---

## Agent Loop for v15

```text
1. Run v14 benchmark and parse metrics.
2. Confirm quality_ok and no_oom.
3. Confirm prepared_speedup < 1.10.
4. Preserve block fields through cache/model.
5. Implement reference block attention.
6. Prove correctness against dense block mask.
7. Implement fast native block kernel.
8. Benchmark native path.
9. Promote, tune, or reject based on decision rules.
```

Invariant:

```text
Never optimize without bucketed timings and correctness parity tests.
```
