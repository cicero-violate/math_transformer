# Math-Routed Sparse Transformer Plan v3

## Status

Project path:

```text
/workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer/
```

Current benchmark checkpoint:

```text
n   allowed   full   sparsity   rel_reduce   dense_full_ms   dense_masked_ms   nbr_sparse_ms
8       40      64     0.6250      0.3750          48.60            25.24           0.16
16     104     256     0.4062      0.5938           1.37             3.01           0.20
32     294    1024     0.2871      0.7129           1.53             4.96           0.33
```

Interpretation:

```text
relation reduction works
neighbor sparse attention kernel exists
neighbor sparse attention is fast in isolation
dense masked attention remains slow because it still computes dense QK^T
full end-to-end sparse transformer speedup is not proven yet
```

Current architecture stage:

```text
Math-Routed Dense-Masked Transformer + Isolated Neighbor Sparse Attention Kernel
```

Target architecture stage:

```text
End-to-End Math-Routed Neighbor-Sparse Transformer
```

---

## 1. Core Objective

Move from logical sparsity to actual model-level computed sparsity.

Current working idea:

```text
math topology reduces allowed attention edges
```

Current missing piece:

```text
the main transformer block still uses dense masked attention
```

Target:

```text
math topology -> neighbor list -> neighbor sparse attention inside the real transformer block
```

Final objective:

```text
same mathematical relation mask
same output shape
same verifier behavior
lower computed attention work
lower wall-clock time at sufficient n
```

---

## 2. Correct Benchmark Interpretation

The current benchmark has three timing columns:

```text
dense_full_ms
```

This times the full model path without math mask.

```text
dense_masked_ms
```

This times the full model path with topology mask, but attention still computes dense QK^T.

```text
nbr_sparse_ms
```

This times isolated neighbor sparse attention using random q/k/v and the topology-derived neighbor list.

Therefore:

```text
nbr_sparse_ms proves sparse attention kernel speed
nbr_sparse_ms does not yet prove full model speed
```

Do not claim full transformer speedup until the sparse kernel is wired into the actual model block.

---

## 3. Current Real Wins

### 3.1 Relation reduction improves with n

Observed:

```text
n=8   rel_reduce=0.3750
n=16  rel_reduce=0.5938
n=32  rel_reduce=0.7129
```

This is the desired trend.

As sequence size grows:

```text
full_edges = n^2
allowed_edges grows slower than n^2
relation_reduction increases
```

### 3.2 Neighbor sparse kernel exists

Current sparse kernel:

```text
neighbor_attention(q, k, v, neighbors, valid)
```

Expected complexity:

```text
O(n k d)
```

instead of:

```text
O(n^2 d)
```

### 3.3 Benchmark now separates relation reduction from runtime

This is correct.

Relation reduction and runtime speedup must remain separate metrics.

---

## 4. Current Problems

### 4.1 Neighbor sparse is not integrated into the model

Current issue:

```text
MathRoutedAttention -> dense QK^T -> mask
```

Needed:

```text
MathNeighborSparseAttention -> gather neighbors -> compute only allowed q_i dot k_j
```

### 4.2 Benchmark is not apples-to-apples yet

Current comparison:

```text
dense_full_ms      = full model path
 dense_masked_ms   = full masked model path
 nbr_sparse_ms     = isolated attention kernel only
```

Needed comparison:

```text
dense_full_attention_only
 dense_masked_attention_only
 neighbor_sparse_attention_only

full_transformer_block
 dense_masked_transformer_block
 neighbor_sparse_transformer_block

full_end_to_end_pipeline
 masked_end_to_end_pipeline
 sparse_end_to_end_pipeline
```

### 4.3 symbolic_dependency is zero

Observed:

```text
symbolic_dependency: 0
```

Likely cause:

```text
benchmark sequence uses only root expressions, not collected expression-tree nodes
```

Current style:

```python
all_nodes = [normalize(parse(expr)) for expr in exprs]
```

Needed style:

```python
all_nodes = []
for expr in exprs:
    root = normalize(parse(expr))
    all_nodes.extend(root.collect_nodes())
```

### 4.4 shape_compat is zero

Observed:

```text
shape_compat: 0
```

Cause:

```text
benchmark does not pass shape/type environment into topology diagnostics
```

Needed:

```text
shape env from examples.jsonl -> shape_compatibility_matrix(nodes, env)
```

### 4.5 Dense masked path is expected to be slow

Dense masked still computes:

```text
QK^T in R^{n x n}
```

Then masks after the expensive operation.

This is only a correctness baseline.

It is not the final system.

---

## 5. Target Architecture

Final block pipeline:

```text
nodes
  -> deterministic/learned math embeddings Z
  -> relation matrices G, C, R, L, I
  -> allowed mask A_allowed
  -> neighbor list N
  -> projected Q,K,V
  -> neighbor sparse attention
  -> output projection
  -> feedforward/expert routing
  -> verifier
```

Where:

```text
G = symbolic dependency matrix
C = composition/type compatibility matrix
R = embedding similarity / topk matrix
L = local fallback matrix
I = identity/self-attention matrix
```

Final attention:

```text
out_i = sum_{j in N(i)} softmax(q_i dot k_j / sqrt(d)) v_j
```

Not:

```text
out = softmax(QK^T / sqrt(d) + mask) V
```

The former computes only neighbor edges.

The latter computes all edges first.

---

## 6. Phase 1 — Integrate Neighbor Sparse Attention Into Model

### 6.1 Add attention mode

Modify `src/attention.py` or create a wrapper module:

```text
DenseFullAttention
DenseMaskedMathAttention
NeighborSparseMathAttention
```

Recommended API:

```python
class NeighborSparseMathAttention(nn.Module):
    def forward(self, x, neighbors, valid):
        ...
```

Inputs:

```text
x:         [B,T,d_model]
neighbors: [T,K]
valid:     [T,K]
```

Output:

```text
out: [B,T,d_model]
```

Internals:

```text
q_proj, k_proj, v_proj
reshape to [B,H,T,Dh]
neighbor_attention(q,k,v,neighbors,valid)
out_proj
```

Acceptance:

```text
output shape equals dense attention output shape
same mask converted to neighbors produces numerically close output to dense masked attention
```

---

### 6.2 Add model attention mode switch

Modify `src/model.py`:

```python
attention_mode: Literal["full", "dense_masked", "neighbor_sparse"]
```

Behavior:

```text
full            -> no mask, dense full attention
dense_masked    -> topology mask + dense masked attention
neighbor_sparse -> topology mask -> neighbors -> neighbor sparse attention
```

Acceptance:

```python
model = MathRoutedTransformer(attention_mode="neighbor_sparse")
out, masks, routes = model(x, nodes)
assert out.shape == x.shape
```

---

### 6.3 Preserve correctness baseline

Do not delete dense masked attention.

It remains the reference implementation.

Tests must compare:

```text
dense_masked attention
neighbor_sparse attention
```

on identical q/k/v and identical allowed edges.

---

## 7. Phase 2 — Apples-to-Apples Benchmarks

### 7.1 Attention-only benchmark

Same q/k/v for all variants:

```text
dense_full_attention_only_ms
dense_masked_attention_only_ms
neighbor_sparse_attention_only_ms
```

Inputs:

```text
q,k,v: [B,H,T,Dh]
mask: [T,T]
neighbors: [T,K]
valid: [T,K]
```

Required output columns:

```text
n
max_k
avg_k
full_edges
allowed_edges
sparsity_ratio
relation_reduction
dense_full_attention_ms
dense_masked_attention_ms
neighbor_sparse_attention_ms
```

---

### 7.2 Block-level benchmark

Same `x` and same topology for all variants:

```text
full_block_ms
dense_masked_block_ms
neighbor_sparse_block_ms
```

This proves whether the transformer block is faster.

---

### 7.3 End-to-end benchmark

Include full system overhead:

```text
parse
normalize
embed
topology build
neighbor conversion
attention
router
verifier
```

Output:

```text
full_e2e_ms
masked_e2e_ms
sparse_e2e_ms
```

This proves whether the whole system is faster.

---

### 7.4 Benchmark truth labels

Use these labels:

```text
relation_reduction = fewer allowed edges than full attention
computed_relation_reduction = fewer computed dot products
runtime_speedup = actual wall-clock improvement
memory_reduction = lower peak memory
```

Do not use `speedup` without saying which layer it refers to.

---

## 8. Phase 3 — Activate Symbolic Dependency Edges

### 8.1 Change benchmark node construction

Current problem:

```text
symbolic_dependency: 0
```

Fix:

```python
all_nodes = []
for expr in exprs:
    root = normalize(parse(expr))
    all_nodes.extend(root.collect_nodes())
```

Then trim or repeat to target n.

Required benchmark result:

```text
symbolic_dependency > 0
```

### 8.2 Keep root-only benchmark as separate case

Root-only sequence is still useful.

Create two benchmark modes:

```text
--node-mode roots
--node-mode trees
```

Expected:

```text
roots -> tests cross-expression relation routing
trees -> tests symbolic dependency topology
```

---

## 9. Phase 4 — Activate Shape Compatibility Edges

### 9.1 Add shape environment to benchmark

Use `data/examples.jsonl` shape fields.

Example:

```json
{"shape":{"A":[32,64],"x":[64],"b":[32],"out":[32]}}
```

Load into:

```python
env = {
  "A": (32,64),
  "x": (64,),
  "b": (32,)
}
```

### 9.2 Add shape compatibility matrix

Implement or strengthen:

```python
shape_compatibility_matrix(nodes, env) -> bool[n,n]
```

Should connect nodes when:

```text
output shape of one can feed input shape of another
same output shape participates in add/elementwise relation
matmul output shape matches add bias shape
```

Acceptance:

```text
shape_compat > 0
```

in at least one benchmark.

---

## 10. Phase 5 — Neighbor List Quality

### 10.1 Track max_k and avg_k

Allowed edges alone is not enough.

Add:

```text
max_k = max allowed neighbors per row
avg_k = allowed_edges / n
```

Important because neighbor sparse cost is closer to:

```text
O(n * max_k * d)
```

if padded neighbor lists are used.

Add report columns:

```text
avg_k
max_k
padding_ratio
```

Where:

```text
padding_ratio = 1 - (allowed_edges / (n * max_k))
```

A high padding ratio means wasted sparse compute.

### 10.2 Sort/prioritize neighbors

Current neighbor conversion takes first `row_idx[:k]`.

Needed:

```text
rank neighbors by relation priority or relation weight
```

Priority order:

```text
1. identity/self
2. symbolic dependency
3. verifier-required edge
4. composition/type edge
5. shape compatibility
6. embedding topk
7. local window
8. same operator
```

This matters when max_k truncation happens.

---

## 11. Phase 6 — Sparse Attention Correctness Tests

Create or strengthen:

```text
tests/test_sparse_attention.py
```

Required tests:

```text
neighbors_from_mask returns valid padded neighbors
neighbor_attention shape matches dense attention
neighbor_attention equals dense masked attention for same mask
all-masked rows do not produce NaN
self-only mask works
variable row lengths work
neighbor sparse module works inside model block
```

Core equivalence test:

```python
out_dense = math_attention(q, k, v, mask)
neighbors, valid = neighbors_from_mask(mask, max_k_from_mask(mask))
out_sparse = neighbor_attention(q, k, v, neighbors, valid)
assert torch.allclose(out_dense, out_sparse, atol=1e-5)
```

---

## 12. Phase 7 — Training Objective Upgrade

Current training loss is still not the real objective unless already changed.

Target losses:

```text
route prediction loss
shape validity loss
similarity/retrieval loss
mask edge prediction loss
```

### 12.1 Route prediction

Input:

```text
math expression
```

Target:

```text
expert id
```

Metric:

```text
route_accuracy
```

### 12.2 Shape validity prediction

Input:

```text
math expression + shape env
```

Target:

```text
valid / invalid
output shape class or tuple
```

Metric:

```text
invalid_accept_rate
invalid_reject_rate
shape_accuracy
```

### 12.3 Relation edge prediction

Input:

```text
node pair
```

Target:

```text
should attend / should not attend
relation type
```

Metric:

```text
edge_precision
edge_recall
edge_f1
```

---

## 13. Phase 8 — Persistence and Continual Training

Persist durable state, not temporary tensors.

Save:

```text
raw expression
normalized IR
stable hash
shape/type metadata
embedding vector Z or cache key
relation edge lists
neighbor lists when useful
routing decisions
verifier failures
benchmark reports
model checkpoints
candidate promotion reports
```

Do not persist by default:

```text
Q/K/V activations
full dense attention matrices
every intermediate tensor
```

Continual loop:

```text
stable model
candidate model
failure collector
synthetic generator
replay buffer
frozen eval set
promotion gate
rollback
```

Promotion rule:

```text
candidate must improve or preserve task quality
candidate must not increase invalid accept rate
candidate must preserve verifier pass/fail correctness
candidate should reduce computed relation count or runtime
```

---

## 14. Success Gates

### Gate 1 — Current baseline preserved

Required:

```text
all tests pass
benchmark script runs
training script runs
```

Command:

```text
bash scripts/run_tests.sh -q
bash scripts/run_tiny.sh
bash scripts/benchmark_attention.sh
```

---

### Gate 2 — Attention-only sparse proof

Required:

```text
neighbor_sparse_attention_ms < dense_full_attention_ms
```

for at least one nontrivial n.

Already likely satisfied by current checkpoint.

Need apples-to-apples timing confirmation.

---

### Gate 3 — Block-level sparse proof

Required:

```text
neighbor_sparse_block_ms < dense_full_block_ms
```

for sufficiently large n.

This is the next decisive gate.

---

### Gate 4 — End-to-end sparse proof

Required:

```text
sparse_e2e_ms < dense_e2e_ms
```

or, if topology overhead dominates:

```text
attention compute reduced but end-to-end not yet faster
```

State the result honestly.

---

### Gate 5 — Real math topology proof

Required benchmark diagnostics:

```text
symbolic_dependency > 0
shape_compat > 0
embedding_topk > 0
identity = n
```

This proves the system is using actual math relations, not only embedding/local/same-op edges.

---

### Gate 6 — Quality preserved

Required:

```text
same output shape
same verifier behavior
route accuracy preserved or improved
shape validity preserved or improved
```

No speed gain matters if correctness collapses.

---

## 15. Immediate Implementation Checklist

### Step 1

Integrate `neighbor_attention` into the actual model block.

Files:

```text
src/attention.py
src/sparse_attention.py
src/model.py
```

### Step 2

Add attention mode switch.

```text
full
dense_masked
neighbor_sparse
```

### Step 3

Add apples-to-apples benchmark layers.

```text
attention-only
block-level
end-to-end
```

### Step 4

Fix benchmark node construction.

```text
roots mode
trees mode
```

### Step 5

Add shape environment into topology diagnostics.

```text
shape_compat > 0
```

### Step 6

Add max_k, avg_k, and padding ratio to benchmark.

```text
avg_k = allowed_edges / n
padding_ratio = 1 - allowed_edges / (n * max_k)
```

---

## 16. Expected Next Benchmark Format

Target output:

```text
n  mode   allowed  full  avg_k  max_k  pad  sparsity  rel_reduce  dense_attn  masked_attn  nbr_attn  full_block  masked_block  nbr_block
```

Example:

```text
32 trees  294      1024  9.18   15     .39  .2871     .7129       1.20        1.35         .33       2.10        4.90          1.80
```

Relation diagnostics:

```text
symbolic_dependency: >0
same_operator: >0
embedding_topk: >0
local_window: >0
shape_compat: >0
identity: n
```

---

## 17. Architecture Claim After This Plan

If this plan succeeds, the valid claim becomes:

```text
We have an end-to-end transformer block whose attention computation is constrained by math-derived neighbor lists and computed sparsely.
```

Do not claim:

```text
We solved general transformer efficiency.
```

Do claim:

```text
For math-structured sequences, we can reduce computed attention relations using symbolic and embedding-derived topology.
```

---

## 18. Final Target

The final system should be:

```text
math object
  -> symbolic IR
  -> normalized expression
  -> math-function vector matrix Z
  -> symbolic dependency matrix G
  -> composition/type matrix C
  -> embedding relation matrix R
  -> allowed edge set
  -> neighbor sparse attention
  -> expert/kernel route
  -> verifier
  -> persisted failures and benchmark reports
```

Core thesis:

```text
Attention should not discover all math structure by brute force.
Math structure should define the attention topology first.
The model should only learn weights over meaningful relations.
```

