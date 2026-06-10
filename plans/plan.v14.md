# plan.v14.md — Block-Learned Sparse Topology for Scalable Material Speedup

## Context

v13 implemented the paired prepared-block benchmark improvements and fused profiling path:

```text
--profile-fused-norm-qkv
--profile-fused-attn-outproj
```

The benchmark wrapper now supports:

```bash
FUSED_NORM_QKV=1
FUSED_ATTN_OUTPROJ=1
```

v13 also added timing-bucket deltas for the paired hand-vs-learned prepared block.

The strongest v13 fused baseline at `N=1024` showed:

```text
hand K=16 route_acc=0.9821 block_ms=1.068913 attention_ms=0.343203 non_attention_ms=0.725710
learned K=6 route_acc=0.9871 block_ms=1.035754 attention_ms=0.309477 non_attention_ms=0.726277
speedup=1.032014
quality_ok=True speed_ok=True
```

The larger-N scaling check showed:

```text
N=1024 speedup=1.039081 quality_ok=True speed_ok=True
N=2048 speedup=0.991479 quality_ok=True speed_ok=False
N=4096 CUDA OOM during learned topology scoring
```

At `N=2048`, learned sparse attention remained faster:

```text
hand attention_outproj_ms    = 0.618555
learned attention_outproj_ms = 0.572913
delta                        = -0.045642 ms
```

But non-attention overhead erased the gain:

```text
hand non_attention_ms    = 1.339727
learned non_attention_ms = 1.402199
delta                   = +0.062472 ms
```

At `N=4096`, the current learned topology path attempted dense token-pair scoring and OOMed inside:

```text
learned_topology_runtime.py -> _scores
learned_topology.py -> scorer(features)
torch.nn.Linear
```

The root cause is that current learned topology scoring is token-pair dense:

```text
S_token in R^(N x N)
cost = O(N^2)
```

For `N=4096`:

```text
N^2 = 16,777,216 token pairs
```

So v13 establishes:

```text
learned K=6 quality > hand K=16 quality
learned K=6 prepared attention is faster
full-block speedup is not material
learned token-pair topology construction does not scale
```

---

## v14 Objective

Replace token-pair learned topology with block-pair learned topology so learned sparsity remains quality-superior while becoming GPU-friendly and scalable.

Primary target:

```text
quality_ok=True
speed_ok=True
speedup >= 1.10x
no OOM for N in {1024, 2048, 4096}
```

Minimum acceptable result:

```text
block-learned route_acc >= hand K=16 route_acc
block-learned prepared attention_ms < hand K=16 attention_ms
block-learned full prepared block_ms < hand K=16 block_ms
block topology prepare does not OOM at N=4096
clear prepare/block timing explanation if speedup < 1.10x
```

---

## Core Design

Move from token graph scoring:

```text
S_token in R^(N x N)
```

to block graph scoring:

```text
S_block in R^(B x B)
B = ceil(N / block_size)
```

For example:

```text
N = 4096
block_size = 64
B = 64
B^2 = 4096 block pairs
```

Reduction versus token-pair scoring:

```text
N^2 / B^2 = block_size^2 = 4096x fewer score entries for block_size=64
```

The block scorer selects block neighbors:

```text
N_B(i) = topK_blocks_j score(block_i, block_j)
```

Then the selected block pairs define contiguous token ranges for sparse attention.

Final attention neighborhood:

```text
N_final(t) = local_token_neighbors(t)
           union tokens_in_selected_blocks(block(t))
           union required_symbolic_bridge_tokens(t)
```

---

## Task 1 — Freeze v13 Evidence and Diagnostics

### Goal

Document that v13 succeeded on quality and attention speed but failed on material/scalable full-block speed.

### Work

Create a concise v13 result note in benchmark output and reports:

```text
learned_token_quality_win=True
learned_token_attention_win=True
learned_token_material_speed_win=False
learned_token_scaling_ok=False
```

### Diagnostic patch

Current console output prints:

```text
dominant_bucket=...
```

This is ambiguous because it reports the largest learned-vs-hand regression, not the largest absolute bucket.

Replace with:

```text
dominant_regression_bucket=...
dominant_absolute_bucket=...
```

Definitions:

```text
dominant_regression_bucket = argmax_i(learned_bucket_i - hand_bucket_i)
dominant_absolute_bucket   = argmax_i(learned_bucket_i)
```

### Acceptance

For every paired benchmark, console output identifies both:

```text
largest learned-vs-hand regression
largest absolute learned cost bucket
```

---

## Task 2 — Split Prepare Time From Prepared Block Time

### Motivation

v13 showed two separate costs:

```text
topology preparation/scoring
prepared block execution
```

The learned token topology can make prepared attention faster while still failing end-to-end or OOMing during topology construction.

### Work

Report both timing classes explicitly:

```text
topology_prepare_ms
learned_scorer_ms
neighbor_table_build_ms
prepared_block_ms
prepared_attention_ms
prepared_non_attention_ms
total_with_prepare_ms
```

For paired prepared benchmarks:

```text
prepared_block_ms = measured block forward over already-installed topology
total_with_prepare_ms = topology_prepare_ms + prepared_block_ms
```

### Acceptance

JSON and console output must make clear whether a failure is caused by:

```text
learned scorer construction
neighbor table conversion
attention kernel
norm/qkv/outproj/ffn block overhead
```

---

## Task 3 — Add Block Topology Data Structures

### Goal

Represent block-level neighborhoods in a way that can be expanded into existing token-neighbor tables first, and later replaced by native block-sparse kernels.

### Proposed file

```text
src/block_topology.py
```

### Proposed dataclasses

```python
@dataclass
class BlockTopologyConfig:
    block_size: int = 64
    topk_blocks: int = 4
    include_local_blocks: int = 1
    include_self_block: bool = True
    include_symbolic_bridge: bool = True

@dataclass
class PreparedBlockTopology:
    block_neighbors: torch.Tensor      # (B, topk_blocks_eff), int64
    block_valid_i8: torch.Tensor       # (B, topk_blocks_eff), int8
    token_neighbors: torch.Tensor      # existing-compatible token neighbors
    token_valid_i8: torch.Tensor       # existing-compatible token valid mask
    diagnostics: MaskDiagnostics
```

### Acceptance

For any `N`, `block_size`, `topk_blocks`:

```text
block_neighbors.shape[0] == ceil(N / block_size)
token_neighbors.shape[0] == N
token_valid_i8.shape == token_neighbors.shape
```

---

## Task 4 — Implement Heuristic Block Topology Builder First

### Motivation

Before training a block scorer, prove the block representation and benchmark path work.

### Proposed file

```text
src/block_learned_topology.py
```

### Initial heuristic scoring

Score block pairs using deterministic features:

```text
score(i, j) = local_bias(i, j)
            + symbolic_bridge_bias(i, j)
            + middle_preserving_bias(i, j)
            - lambda_distance * abs(i - j)
```

Block features:

```text
block_id
start_token
end_token
operator histogram
node kind histogram
shape/env summary
middle token coverage
```

### Expansion rule

For each token `t` in block `i`, selected block `j` contributes candidate token neighbors from:

```text
[j * block_size, min((j + 1) * block_size, N))
```

The first implementation may cap token expansion to a fixed token K for compatibility with the current neighbor table:

```text
max_token_neighbors = topk_blocks * tokens_per_selected_block_cap + local_window + symbolic_bridge_cap
```

### Acceptance

The heuristic block builder can produce a valid prepared topology for:

```text
N = 1024
N = 2048
N = 4096
```

without OOM.

---

## Task 5 — Add Block-Learned Topology Mode

### Goal

Expose block topology through the existing eval and benchmark interfaces.

### Proposed CLI flags

```bash
--topology-mode learned_block_topk
--block-size 64
--topk-blocks 4
--block-local-window 1
--block-token-cap 16
```

### Proposed wrapper env vars

```bash
BLOCK_TOPOLOGY=1
BLOCK_SIZE=64
TOPK_BLOCKS=4
BLOCK_LOCAL_WINDOW=1
BLOCK_TOKEN_CAP=16
```

### Acceptance

Benchmark command supports:

```bash
SCORER=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
LEARNED_K=6 \
HAND_K=16 \
BENCH_N=4096 \
BLOCK_TOPOLOGY=1 \
BLOCK_SIZE=64 \
TOPK_BLOCKS=4 \
scripts/benchmark_learned_topology.sh
```

and completes without CUDA OOM.

---

## Task 6 — Train or Distill a Block Scorer

### Motivation

The heuristic block builder may be fast but may lose route quality. If so, train a block scorer.

### Teacher

Use the current learned token scorer/topology as teacher at smaller sizes where it does not OOM:

```text
N in {512, 1024, 2048}
```

Aggregate token-pair teacher scores into block-pair labels:

```text
teacher_block_score(i, j)
  = max or mean score over token pairs (t in block_i, u in block_j)
```

### Student

Train:

```text
block_scorer(block_i_features, block_j_features) -> score_ij
```

### Loss

```text
L = L_route
  + lambda_teacher * L_teacher_agree
  + lambda_sparse * L_sparsity
  + lambda_latency * L_latency_proxy
```

### Acceptance

Block scorer achieves:

```text
block_learned route_acc >= hand K=16 route_acc
```

while maintaining:

```text
N=4096 no OOM
```

---

## Task 7 — Benchmark Matrix

Run the fused prepared benchmark matrix:

```bash
for N in 1024 2048 4096; do
  for MODE in hand token_learned block_heuristic block_learned; do
    ...
  done
done
```

Required table:

```text
N
mode
route_acc
prepare_ms
learned_scorer_ms
prepared_block_ms
prepared_attention_ms
prepared_non_attention_ms
total_with_prepare_ms
speedup_vs_hand_prepared
speedup_vs_hand_total
oom
```

### Acceptance

v14 can make a decision from one table:

```text
promote block heuristic
train block scorer
pivot to native block-sparse kernel
```

---

## Task 8 — Native Block-Sparse Kernel Decision

### Motivation

The first block topology version may expand selected blocks back into token-neighbor tables. This avoids OOM and validates quality, but it may not fully exploit GPU locality.

### Decision rule

If block topology improves prepare/scoring scalability but prepared block speedup remains below `1.10x`, implement native block-sparse attention:

```text
input layout: selected block ranges
kernel work: block query x selected key blocks
memory: contiguous block loads
```

### Acceptance for native block-sparse pivot

```text
block topology quality_ok=True
prepare no-OOM at N=4096
prepared token-neighbor expansion speedup < 1.10x
```

Then v15 should implement native block-sparse attention kernels.

---

## Decision Criteria

### Promote v14 if

```text
quality_ok=True
speed_ok=True
speedup >= 1.10x
no OOM at N=4096
```

### Keep v13 but document limitation if

```text
quality_ok=True
attention speedup exists
block topology does not preserve quality
```

and root cause is clearly documented.

### Pivot to native block-sparse kernels if

```text
block topology avoids OOM and preserves quality
but token-neighbor expansion prevents material speedup
```

---

## Expected v14 Deliverables

1. `plan.v14.md` documents the block topology pivot.
2. Benchmark diagnostics split regression and absolute bottlenecks.
3. Benchmark reports split prepare/scoring time from prepared block time.
4. Heuristic block topology builder runs at `N=4096` without OOM.
5. CLI supports block topology benchmark flags.
6. One comparison table covers `N in {1024, 2048, 4096}`.
7. v14 either achieves:

```text
speedup >= 1.10x
```

or proves the next pivot should be native block-sparse kernels.

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
score_reduction = N^2 / B^2
```

When `N` is divisible by `block_size`:

```text
score_reduction = block_size^2
```

```text
T_total = T_prepare + T_prepared_block
```

```text
T_prepared_block = T_attention + T_non_attention
```

```text
speedup_prepared = hand_prepared_block_ms / learned_prepared_block_ms
```

```text
speedup_total = hand_total_with_prepare_ms / learned_total_with_prepare_ms
```

```text
quality_ok = learned_route_acc >= hand_route_acc
```

```text
speed_ok = learned_block_ms < hand_block_ms
```

v14 success requires:

```text
quality_ok and speed_ok and speedup_prepared >= 1.10 and no_oom_at_4096
```
