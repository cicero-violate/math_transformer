# Plan v16 — Native Block Sparse Runtime Hardening

## Status

v15 achieved strict acceptance:

```text
quality_ok=True
strict_speed_ok=True
native_block_backend_triton=True
BLOCK_TOKEN_CAP=2
native_effective_token_k=14
```

Core result:

```text
hand_k16_block_ms=2.891905
learned_k8_block_ms=2.843211
speedup=1.017126
```

Therefore v16 is no longer about proving the architecture can win once.

v16 goal:

```text
turn the v15 win into a repeatable, cleaner, scalable runtime path
```

---

## Main Objective

```text
Native learned block sparse attention should be a first-class runtime path.
```

Formula:

```text
block topology -> capped native token candidates -> fused Triton attention -> output projection
```

Target:

```text
Q_learned >= Q_hand
T_learned < T_hand
repeatably
```

---

## Acceptance Criteria

### 1. Repeatable Strict Speed Pass

Run:

```bash
ACCEPTANCE_TOL_MS=0 \
BLOCK_TOPOLOGY=1 \
NATIVE_BLOCK_SPARSE=1 \
BLOCK_TOKEN_CAP=2 \
FUSED_NORM_QKV=1 \
FUSED_ATTN_OUTPROJ=1 \
BENCH_N=4096 \
scripts/benchmark_learned_topology.sh
```

Pass condition:

```text
acceptance_passed quality_ok=True speed_ok=True strict_speed_ok=True
```

Repeatability condition:

```text
strict pass >= 3 / 5 runs
```

Required metrics:

```text
learned_token_quality_win=True
learned_token_attention_win=True
learned_token_material_speed_win=True
native_block_backend_triton=True
```

---

### 2. Remove Token Neighbor Table Cost From Native Path

Current native path still reports:

```text
neighbor_table_build_ms > 0
```

But native block sparse does not need the full token neighbor table.

v16 target:

```text
native_block_sparse=True
=> do not build token_neighbors unless diagnostics require it
```

Add mode:

```text
prepare_mode = "native_block_only"
```

Required prepared fields:

```text
block_neighbors
block_valid_i8
block_token_indices
block_token_valid_i8
block_size
```

Optional diagnostic fields:

```text
token_neighbors
token_valid_i8
```

Acceptance:

```text
neighbor_table_build_ms <= 1.0 ms
```

or:

```text
neighbor_table_build_ms=0
```

for native benchmark mode.

---

### 3. Real Native Block Attention + Out Projection Fusion

Current learned block path supports fused profiling, but v16 should make it a real fused runtime primitive.

Implement:

```text
triton_block_sparse_attention_outproj(...)
```

Input:

```text
q, k, v
block_token_indices
block_token_valid_i8
W_out
b_out
```

Output:

```text
B x T x d_model
```

Target:

```text
attention_outproj_ms_learned < attention_outproj_ms_hand
```

Acceptance:

```text
native_block_attention_ms <= hand_k16_attention_ms
```

and:

```text
out_proj_ms = 0
attention_outproj_ms > 0
```

---

### 4. Make Intra-Block Token Selection Smarter

Current token cap uses sampled tokens per selected block.

Current formula:

```text
K_eff = K_blocks * BLOCK_TOKEN_CAP
```

For v15 pass:

```text
K_blocks ≈ 7
BLOCK_TOKEN_CAP = 2
K_eff = 14
```

v16 should replace fixed sampling with learned intra-block token selection.

Implement:

```text
block_i, token_j, query_summary -> token_score
top_c(token_score within selected block)
```

Target:

```text
same speed as cap=2
higher hidden_cos
lower hidden_l1
lower logit_kl
```

Acceptance:

```text
route_acc >= hand_route_acc
hidden_cos >= 0.94
logit_kl <= 0.02
strict_speed_ok=True
```

Stretch:

```text
hidden_cos >= 0.97
```

---

### 5. Scale Curve Benchmark

Benchmark sizes:

```text
N = 1024, 2048, 4096, 8192, 16384
```

For each size record:

```text
hand_block_ms
learned_block_ms
hand_attention_ms
learned_attention_ms
topology_prepare_ms
neighbor_table_build_ms
native_effective_token_k
route_acc
hidden_cos
logit_kl
```

Acceptance:

```text
learned speedup improves as N grows
```

Expected:

```text
speedup(N) should increase with N
```

because learned native block sparse should avoid full token-table materialization and reduce attention work.

---

### 6. First-Class Config Object

Replace scattered env/script flags with a canonical runtime config:

```python
NativeBlockSparseConfig(
    block_size=64,
    topk_blocks=4,
    block_local_window=1,
    block_token_cap=2,
    native_block_sparse=True,
    fused_norm_qkv=True,
    fused_attn_outproj=True,
)
```

Acceptance:

```text
scripts still work
tests use config directly
model accepts config directly
```

---

### 7. Benchmark Hygiene

Add benchmark report fields:

```text
strict_speed_ok
speed_gap_ms
acceptance_tolerance_ms
native_block_backend
native_effective_token_k
prepare_mode
token_table_built
```

Add output summary:

```text
quality: pass/fail
attention: pass/fail
prepared block: pass/fail
with prepare: pass/fail
strict: pass/fail
```

No hidden metadata should be counted as timing.

Invariant:

```text
total_block_ms = sum(duration_buckets only)
```

---

### 8. CUDA Correctness Tests

Add CUDA tests when available:

```text
test_triton_block_sparse_matches_vectorized
test_triton_block_sparse_token_cap_matches_reference_mask
test_fused_block_attention_outproj_matches_unfused
test_native_block_sparse_no_nan_n4096
```

Tolerance:

```text
atol=1e-4
rtol=1e-4
```

CPU fallback remains required.

---

## v16 Deliverables

### Files likely touched

```text
src/block_sparse_attention.py
src/triton_block_sparse_attention.py
src/block_learned_topology.py
src/block_topology.py
src/topology_cache.py
src/attention.py
src/model.py
src/eval.py
scripts/benchmark_learned_topology.sh
tests/test_block_sparse_attention.py
tests/test_native_block_sparse_benchmark.py
```

### New optional files

```text
src/native_block_sparse_config.py
tests/test_triton_block_sparse_cuda.py
scripts/sweep_native_block_sparse.sh
```

---

## Priority Order

### P0 — Lock v15 Baseline

```text
repeat strict pass 3 / 5
save benchmark JSON
document canonical command
```

### P1 — Remove Native Token Table Build

```text
native_block_only prepare mode
no token_neighbors unless requested
```

### P2 — Real Fused Block Attention OutProj

```text
triton_block_sparse_attention_outproj
```

### P3 — Learned Intra-Block Token Selector

```text
replace fixed sampling with learned top-c inside selected blocks
```

### P4 — Scale Curve

```text
N = 1024..16384
prove asymptotic advantage
```

### P5 — Clean Runtime API

```text
NativeBlockSparseConfig
```

---

## Stop Conditions

Do not continue optimizing if:

```text
strict pass is not repeatable
```

First fix repeatability.

Do not claim architecture win if:

```text
route_acc wins but hidden_cos collapses
```

Track both:

```text
task quality
representation quality
```

Do not optimize token cap below useful representation quality.

---

## Canonical v16 Command

```bash
ACCEPTANCE_TOL_MS=0 \
BLOCK_TOPOLOGY=1 \
NATIVE_BLOCK_SPARSE=1 \
BLOCK_TOKEN_CAP=2 \
FUSED_NORM_QKV=1 \
FUSED_ATTN_OUTPROJ=1 \
BENCH_N=4096 \
scripts/benchmark_learned_topology.sh
```

---

## v16 Success Definition

```text
Native learned block sparse is not a benchmark trick.
It is a repeatable runtime path.
```

Final gate:

```text
quality_ok=True
strict_speed_ok=True
native_block_backend_triton=True
token_table_built=False
repeat_pass_rate >= 3/5
```

---

## Summary Variables

```text
N = sequence length
B = ceil(N / S)
S = block size
K_b = selected blocks per query block
c = tokens selected per block
K_eff = K_b * c
```

Pass conditions:

```text
Q_pass = route_acc_learned >= route_acc_hand
T_pass = T_learned < T_hand
```

v16 pass:

```text
Q_pass
and T_pass
and native_block_backend_triton
and not token_table_built
```
