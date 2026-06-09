# Math-Routed Transformer Plan v2

## Status

Project path:

```text
/workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer/
```

Source path:

```text
/workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer/src/
```

Current prototype status:

```text
source compiles
unit tests pass
training script runs from project directory
benchmark script runs
math-routed dense-masked attention works
real sparse speedup not implemented yet
```

Current test baseline:

```text
56 passed
```

Architecture name:

```text
Math-Routed Transformer
```

More precise current name:

```text
Math-Routed Dense-Masked Transformer Prototype
```

Target final name:

```text
Math-Routed Sparse Transformer
```

---

## 1. Core Objective

Build a transformer architecture where mathematical structure controls attention, routing, and verification.

The system should not force a transformer to rediscover obvious math topology from raw sequence tokens.

Instead:

```text
math object
  -> symbolic IR
  -> normalized expression
  -> math-function embedding
  -> topology/relation matrix
  -> sparse attention
  -> expert/kernel route
  -> verifier gate
```

The embedding model should represent math functions/operators/expression nodes as vectors.

Attention should remain matrix-based, but the matrix should become a math-relation matrix instead of unconstrained all-to-all token attention.

---

## 2. Correct Mental Model

The embedding vector is not the math function itself.

```text
z_f != f
```

The embedding vector is a routing coordinate for the function.

```text
z_f = E_theta(f)
```

The function remains symbolic/executable/verifiable.

```text
f = symbolic IR + semantics + shape/type rules + execution rule
```

The matrix stack should be:

```text
Z = matrix of math-function vectors
G = symbolic dependency matrix
C = legal composition/type-compatibility matrix
R = learned/similarity relation matrix
A = final attention matrix
```

Final attention form:

```text
A_math = softmax(score(Q,K) + mask(G) + mask(C) + relation_bias(R))
```

In v0/v1, this can be dense masked attention.

In v2, it must become real sparse/block-sparse/index-gather attention.

---

## 3. Current Reality Check

What currently works:

```text
IR exists
parser exists
normalizer exists
deterministic embedder exists
topology mask exists
dense masked attention exists
rule-based router exists
L0-L2 verifier exists
training loop runs
benchmark loop runs
unit tests pass
```

What is not yet true:

```text
runtime sparse speedup is not present
attention still computes QK^T densely
training objective is placeholder
small training expressions saturate attention masks
numeric constants are not parsed correctly
config-relative paths are fragile
embedding fingerprint uses Python hash behavior
shape propagation is shallow
verifier is permissive in several cases
```

Do not claim efficiency until wall-clock or memory improves.

Current correct claim:

```text
The prototype reduces allowed attention relations logically.
```

Incorrect claim:

```text
The prototype is already faster.
```

---

## 4. Priority Order

Implement in this order:

1. Fix parser constants and canonical normalization.
2. Fix config-relative data loading.
3. Add shape propagation and better verifier coverage.
4. Make topology masks non-saturated in tiny training.
5. Add relation-reduction benchmark metrics.
6. Add real sparse/index-gather attention.
7. Replace placeholder training loss with supervised routing/mask/shape tasks.
8. Add learned math-function embeddings.
9. Add continual training loop with candidate/eval/promote gates.

---

## 5. Phase A — Correctness Fixes

### A1. Parse numeric constants correctly

Current issue:

```text
add(x,0) parses 0 as a variable-like token
const(0) parses as op const with no value
```

Required behavior:

```text
0 -> const(0)
1 -> const(1)
3.14 -> const(3.14)
-2 -> const(-2)
const(0) -> const(0)
const(1.5) -> const(1.5)
```

Acceptance tests:

```python
def test_parse_int_constant():
    assert parse("0").op == "const"
    assert parse("0").value == 0


def test_parse_float_constant():
    assert parse("3.14").op == "const"
    assert parse("3.14").value == 3.14


def test_parse_const_call():
    node = parse("const(0)")
    assert node.op == "const"
    assert node.value == 0
```

---

### A2. Make identity normalization actually work

Required rewrites:

```text
add(x,0) -> x
add(0,x) -> x
mul(x,1) -> x
mul(1,x) -> x
```

Do not apply this unsafe rewrite:

```text
matmul(A,1) -> A
```

Matrix multiplication identity is not scalar `1`.

Only allow matmul identity removal if there is an explicit identity-matrix node later:

```text
matmul(A,I) -> A only if I is known identity matrix with compatible shape
```

Acceptance tests:

```python
def test_add_zero_normalizes():
    assert repr(normalize(parse("add(x,0)"))) == "x"


def test_mul_one_normalizes():
    assert repr(normalize(parse("mul(x,1)"))) == "x"


def test_matmul_scalar_one_does_not_normalize():
    assert repr(normalize(parse("matmul(A,1)"))) == "matmul(A, 1)"
```

---

### A3. Stable deterministic fingerprints

Current issue:

```text
hash(node)
```

Python hash behavior can vary across processes for strings.

Required:

Use stable hash:

```text
sha256(repr(node).encode()).digest()
```

Acceptance:

```text
same expression -> same fingerprint across processes
```

Add test that invokes a subprocess twice and checks equality.

---

### A4. Fix config-relative path resolution

Current issue:

```text
python -m neurosymbolic.math_transformer.src.train --config neurosymbolic/math_transformer/configs/tiny.yaml
```

can fail because:

```text
data/examples.jsonl
```

is resolved relative to current working directory.

Required:

Resolve data paths relative to the project root or config file.

Recommended rule:

```python
config_path = Path(config_path).resolve()
project_root = config_path.parent.parent
data_path = project_root / cfg["data"]["path"]
```

Acceptance:

Both commands work:

```text
cd neurosymbolic/math_transformer && python -m src.train --config configs/tiny.yaml
python -m neurosymbolic.math_transformer.src.train --config neurosymbolic/math_transformer/configs/tiny.yaml
```

---

## 6. Phase B — Shape and Type Semantics

### B1. Add shape inference

Create:

```text
src/shape.py
```

Core API:

```python
def infer_shape(node: MathNode, env: dict[str, tuple[int, ...]]) -> tuple[int, ...] | None:
    ...
```

Required support:

```text
var
const
add
mul
matmul
affine
sum
mean
transpose
```

Rules:

```text
add(x,y): shapes must match or be broadcast-compatible
mul(x,y): shapes must match or be broadcast-compatible
matmul(A,x): A[-1] must equal x[0]
affine(A,x,b): shape(matmul(A,x)) must equal shape(b)
sum(i,x_i): reduction shape, v0 can return (1,) or configured target
```

Acceptance:

```python
infer_shape(parse("matmul(A,x)"), {"A": (32,64), "x": (64,)}) == (32,)
infer_shape(parse("add(matmul(A,x),b)"), {"A": (32,64), "x": (64,), "b": (32,)}) == (32,)
```

Invalid shapes should raise or return a structured failure.

---

### B2. Strengthen verifier

Current verifier checks only the root node with shallow input shape maps.

Required:

Verifier should recursively validate subtrees.

Add:

```python
Verifier.check_tree(root, env)
```

It should check:

```text
all matmul inner dimensions
all add shape compatibility
expert compatibility
optional dtype compatibility
root output shape if supplied
```

Acceptance:

```python
check_tree(add(matmul(A,x),b), {A:(32,64), x:(32,), b:(32,)}) fails
check_tree(add(matmul(A,x),b), {A:(32,64), x:(64,), b:(32,)}) passes
```

---

## 7. Phase C — Better Topology Matrices

### C1. Separate matrix types

Create explicit functions for each relation matrix:

```python
symbolic_dependency_matrix(nodes) -> bool[n,n]
same_operator_matrix(nodes) -> bool[n,n]
shape_compatibility_matrix(nodes, env) -> bool[n,n]
composition_matrix(nodes, env) -> bool[n,n]
embedding_topk_matrix(Z, k) -> bool[n,n]
local_window_matrix(n, w) -> bool[n,n]
```

Then combine:

```python
A_allowed = (
    symbolic_dependency
    | same_operator
    | shape_compatibility
    | composition
    | embedding_topk
    | local_window
    | identity
)
```

Do not hide all matrix logic inside one builder.

The point of this project is to inspect the math matrices.

---

### C2. Add relation diagnostics

For every mask, return:

```json
{
  "n": 64,
  "full_edges": 4096,
  "allowed_edges": 1754,
  "sparsity_ratio": 0.4282,
  "relation_reduction": 0.5718,
  "by_relation": {
    "symbolic_dependency": 120,
    "same_operator": 300,
    "shape_compatibility": 420,
    "embedding_topk": 512,
    "local_window": 256,
    "identity": 64
  }
}
```

Acceptance:

Benchmark must print relation reduction separately from runtime speedup.

---

### C3. Prevent tiny-mask saturation

Problem:

```text
topk=4 local_window=2
```

saturates masks for tiny sequences of 3–5 nodes.

Fix config:

```yaml
debug:
  topk: 1
  local_window: 0

tiny:
  topk: 1
  local_window: 1
```

Also add a generated long sequence dataset for sparsity tests.

Acceptance:

Tiny training should report at least one example with:

```text
sparsity < 1.0
```

---

## 8. Phase D — Real Sparse Attention

Current attention still computes:

```python
scores = q @ k.transpose(-2, -1)
```

That is dense:

```text
O(n^2 d)
```

Masking afterward does not remove the expensive operation.

### D1. Add neighbor-index attention

Create:

```text
src/sparse_attention.py
```

Core API:

```python
def neighbor_attention(q, k, v, neighbors):
    ...
```

Where:

```text
q:         [B,H,T,D]
k:         [B,H,T,D]
v:         [B,H,T,D]
neighbors: [T,K]
```

Compute only:

```text
q_i dot k_j for j in neighbors[i]
```

Target complexity:

```text
O(n k d)
```

Acceptance:

For a mask converted to neighbors, neighbor attention should approximately match dense masked attention on the same allowed entries.

Test:

```python
out_dense = math_attention(q,k,v,mask)
out_sparse = neighbor_attention(q,k,v,neighbors_from_mask(mask))
assert torch.allclose(out_dense, out_sparse, atol=1e-5)
```

For fixed-size neighbor lists, pad missing neighbors with self index and mask them properly.

---

### D2. Benchmark dense masked vs neighbor sparse

Add benchmark columns:

```text
n
k
full_edges
allowed_edges
sparsity_ratio
relation_reduction
dense_full_ms
dense_masked_ms
neighbor_sparse_ms
wall_clock_speedup_vs_full
```

Acceptance:

Even if neighbor sparse is not faster on CPU for small n, benchmark must distinguish:

```text
logical edge reduction
actual runtime
actual memory
```

Do not conflate them.

---

### D3. Later GPU path

After neighbor attention works:

```text
block-sparse attention
Triton kernel
FlashAttention-style block routing
custom CUDA only if needed
```

Do not start here.

First implement correct index-gather sparse attention in PyTorch.

---

## 9. Phase E — Training Objective Upgrade

Current loss:

```python
loss = out.pow(2).mean()
```

This is only a placeholder.

Replace with supervised tasks.

### E1. Route prediction

Input:

```text
expression
```

Target:

```text
expert id
```

Loss:

```text
cross_entropy(route_logits, expert_target)
```

Metrics:

```text
route_accuracy
router_confusion_matrix
```

---

### E2. Shape prediction

Input:

```text
expression + variable shapes
```

Target:

```text
output shape
valid/invalid flag
```

Loss:

```text
shape_classification_loss + validity_loss
```

Metrics:

```text
shape_accuracy
invalid_accept_rate
invalid_reject_rate
```

---

### E3. Similarity retrieval

Input:

```text
expression pair
```

Target:

```text
same_structure / different_structure
```

Loss:

```text
contrastive loss or binary classification loss
```

Metrics:

```text
recall_at_k
pair_accuracy
```

---

### E4. Mask prediction / missing edge prediction

Input:

```text
nodes + known useful edges
```

Target:

```text
edge labels
```

Loss:

```text
binary cross entropy over candidate edges
```

Metrics:

```text
edge_precision
edge_recall
edge_f1
```

---

## 10. Phase F — Learned Math Function Embeddings

Current embedder is deterministic.

Keep it as baseline.

Add learned embedder later:

```text
src/learned_embedder.py
```

Architecture options:

```text
TreeLSTM
recursive MLP over IR
small graph neural network
operator-token transformer over expression tree
```

Recommended first learned version:

```text
recursive MLP over MathNode tree
```

Input features:

```text
operator id
arity
shape vector
dtype id
linearity flag
commutativity flag
child embeddings
```

Output:

```text
z_f in R^d
```

Training signals:

```text
same structure pairs close
wrong structures far
correct route predictable
correct topology edges recoverable
```

Do not remove deterministic embedder.

Use:

```text
z = concat(z_deterministic, z_learned)
```

or:

```text
z = learned_projection(z_deterministic, tree_features)
```

---

## 11. Phase G — Continual Training Loop

The system can train continuously, but the live model should not mutate blindly.

Correct architecture:

```text
stable model
candidate model
failure collector
synthetic generator
replay buffer
evaluation gate
promotion gate
rollback
```

Loop:

```text
run model
collect verifier failures
generate synthetic math tasks
train candidate
run frozen eval
compare against stable
promote only if better
```

Promotion condition:

```text
candidate_route_accuracy >= stable_route_accuracy
candidate_invalid_accept_rate <= stable_invalid_accept_rate
candidate_sparsity_ratio <= stable_sparsity_ratio or quality improves
candidate_runtime <= stable_runtime or relation reduction improves
no regression on frozen tests
```

Required directories:

```text
runs/
  candidates/
  stable/
  eval_reports/
  failure_sets/
  replay_buffers/
```

Do not implement this before correctness and benchmark instrumentation are solid.

---

## 12. File-Level TODOs

### `src/parser.py`

Add:

```text
numeric literal parsing
const(...) special case
negative number parsing
better error messages
```

---

### `src/normalize.py`

Fix:

```text
remove matmul scalar-one identity rewrite
add mul identity only for scalar multiplication
preserve shape/dtype when safe
canonicalize commutative op arg ordering
```

---

### `src/embedder.py`

Fix:

```text
replace hash(node) with stable sha256 fingerprint
add domain/codomain/linearity slots
separate function-vector embedding from expression-node embedding
```

---

### `src/topology.py`

Refactor:

```text
separate relation matrices
return diagnostics
support composition/type-compatibility matrix
support neighbors_from_mask(mask, max_k)
```

---

### `src/attention.py`

Keep:

```text
dense masked attention as correctness baseline
```

Add:

```text
clear naming: DenseMaskedMathAttention
```

---

### `src/sparse_attention.py`

Create:

```text
neighbor_attention
neighbors_from_mask
padded neighbor masks
comparison tests against dense masked attention
```

---

### `src/router.py`

Improve:

```text
return ranked routes, not just one route
lower confidence for heuristic pattern matches
add route logits later
add route balancing diagnostics
```

---

### `src/verifier.py`

Improve:

```text
recursive tree checks
shape inference integration
invalid route rejection
output shape validation
optionally strict dtype mode
```

---

### `src/train.py`

Fix:

```text
config-relative path resolution
real supervised losses
training metrics
checkpoint output
```

---

### `src/eval.py`

Fix:

```text
relation reduction vs runtime speedup labeling
include dense masked baseline
include neighbor sparse baseline after implemented
export JSON report
```

---

## 13. Acceptance Gates

### Gate 1 — Correctness Gate

Required:

```text
all existing tests pass
new parser constant tests pass
new normalization tests pass
new config path tests pass
```

Command:

```text
bash scripts/run_tests.sh
```

---

### Gate 2 — Matrix Inspection Gate

Required benchmark output includes:

```text
full_edges
allowed_edges
sparsity_ratio
relation_reduction
by_relation counts
```

Required:

```text
at least one nontrivial example has sparsity_ratio < 1.0
```

---

### Gate 3 — Sparse Attention Equivalence Gate

Required:

```text
neighbor sparse attention matches dense masked attention within tolerance
```

Test:

```text
pytest tests/test_sparse_attention.py
```

---

### Gate 4 — Real Efficiency Gate

Required:

At larger n, one of the following must improve:

```text
wall-clock runtime
peak memory
computed score count
```

If only computed score count improves, call it relation efficiency, not runtime efficiency.

---

### Gate 5 — Learning Gate

Required:

Replace placeholder loss with at least one real supervised objective:

```text
route prediction
shape prediction
similarity classification
mask edge prediction
```

---

## 14. Immediate Next Commands

Run from project root:

```text
cd /workspace/ai_sandbox/canon-mini-agent/prototype
```

Then:

```text
bash neurosymbolic/math_transformer/scripts/run_tests.sh -q
bash neurosymbolic/math_transformer/scripts/run_tiny.sh
bash neurosymbolic/math_transformer/scripts/benchmark_attention.sh
```

After fixes, this should also work:

```text
python -m neurosymbolic.math_transformer.src.train --config neurosymbolic/math_transformer/configs/tiny.yaml
```

---

## 15. Near-Term Implementation Sprint

### Sprint 1 — Fix correctness

Files:

```text
src/parser.py
src/normalize.py
src/embedder.py
src/train.py
tests/test_parser.py
tests/test_ir.py
tests/test_train_paths.py
```

Deliverable:

```text
constants parse correctly
normalization works safely
train path works from any cwd
stable fingerprints exist
```

---

### Sprint 2 — Matrix diagnostics

Files:

```text
src/topology.py
src/eval.py
tests/test_topology.py
```

Deliverable:

```text
separate relation matrices
diagnostics by relation
relation reduction printed clearly
```

---

### Sprint 3 — Shape verifier

Files:

```text
src/shape.py
src/verifier.py
tests/test_shape.py
tests/test_verifier.py
```

Deliverable:

```text
recursive shape checks
invalid nested matmul rejected
valid affine shape accepted
```

---

### Sprint 4 — Sparse attention

Files:

```text
src/sparse_attention.py
src/attention.py
src/model.py
tests/test_sparse_attention.py
```

Deliverable:

```text
neighbor attention implemented
dense masked equivalence test passes
benchmark compares all variants
```

---

### Sprint 5 — Real learning

Files:

```text
src/tasks.py
src/train.py
src/eval.py
data/examples.jsonl
```

Deliverable:

```text
route prediction objective
shape validity objective
real metrics
checkpoint output
```

---

## 16. Final Target Architecture

```text
Math-Routed Sparse Transformer =
  symbolic IR
  + normalized math expressions
  + math-function vector matrix Z
  + symbolic dependency matrix G
  + composition/type matrix C
  + embedding relation matrix R
  + sparse attention over allowed math relations
  + expert/kernel router
  + recursive verifier
  + continual candidate training loop
```

The central thesis remains:

```text
Do not use attention to discover math topology from scratch.
Represent the topology explicitly, then use attention only over mathematically meaningful relations.
```

