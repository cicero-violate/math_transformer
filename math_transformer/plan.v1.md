# Math-Routed Sparse Transformer Plan

## Status

Prototype target:

```text
/workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer/src/
```

Primary artifact:

```text
plan.md
```

Architecture name:

```text
Math-Routed Sparse Transformer
```

Alternate names:

```text
Symbolic-Topology Transformer
Neurosymbolic Math Transformer
Math-Indexed Transformer
```

---

## 1. Objective

Build a transformer variant that uses explicit mathematical structure before attention runs.

The core move is:

```text
symbolic math object
  -> math embedding
  -> topology mask
  -> sparse attention
  -> operator/kernel routing
  -> compiled execution
  -> verification gate
```

The goal is not to make the embedding model replace symbolic math.

The goal is to use math embeddings as a routing layer that reduces unnecessary attention, improves operator selection, and preserves correctness through symbolic validation.

---

## 2. Core Hypothesis

Standard transformer attention is wasteful because it learns relation structure from raw sequence positions.

For mathematical data, relation structure is often already available or inferable from symbolic form.

Instead of this:

```text
sequence tokens -> dense all-to-all attention -> learned structure
```

Use this:

```text
symbolic objects -> explicit topology -> sparse/routed attention
```

The expected efficiency gain comes from replacing full attention:

```text
O(n^2 d)
```

with topology-limited attention:

```text
O(n k d)
```

where:

```text
k << n
```

---

## 3. Non-Goals

This prototype should not attempt to do everything.

Out of scope for the first prototype:

- Full theorem proving.
- Large-scale language model pretraining.
- General-purpose symbolic algebra replacement.
- Production compiler backend.
- Perfect semantic equivalence detection.
- Replacing exact symbolic verification with vector similarity.

The first prototype should prove the routing idea, not solve all mathematical reasoning.

---

## 4. Core Principle

The embedding is not truth.

```text
E_theta(m) != m
```

The embedding is navigation.

```text
E_theta(m) = routing coordinate
```

Correctness must come from symbolic checks, type checks, shape checks, executable tests, or verifier gates.

The architecture must keep three layers separate:

| Layer           | Role                                       |
|-----------------+--------------------------------------------|
| Symbolic layer  | Truth, identity, exact structure           |
| Embedding layer | Similarity, routing, clustering, retrieval |
| Tensor layer    | Fast execution on GPU/CPU                  |

---

## 5. System Overview

High-level pipeline:

```text
m_i
  -> parse/normalize
  -> symbolic IR node
  -> math embedding z_i
  -> topology graph G
  -> sparse attention mask A
  -> transformer block
  -> expert/kernel router
  -> execution plan
  -> verifier
```

Mathematical form:

```text
m_i in M
z_i = E_theta(m_i)
A_ij = 1[j in TopK(z_i) or edge(i, j) in G]
h_i' = MathAttn(h_i, H, A)
e_i = Router(z_i, h_i')
T = Compile(m_i, e_i)
accept = Verify(T, m_i)
```

---

## 6. Repository Layout

Target structure:

```text
neurosymbolic/math_transformer/
  plan.md
  README.md
  src/
    __init__.py
    ir.py
    parser.py
    normalize.py
    embedder.py
    topology.py
    attention.py
    router.py
    verifier.py
    model.py
    train.py
    eval.py
  tests/
    test_ir.py
    test_parser.py
    test_topology.py
    test_attention_mask.py
    test_router.py
    test_verifier.py
  configs/
    tiny.yaml
    debug.yaml
    benchmark.yaml
  data/
    examples.jsonl
  scripts/
    run_tiny.sh
    run_tests.sh
    benchmark_attention.sh
  notes/
    design.md
    experiments.md
```

For the first pass, keep implementation small and inspectable.

---

## 7. Mathematical Objects

The prototype should support a small but useful set of math objects.

Initial object classes:

| Class                 | Example                 | Why it matters                  |
|-----------------------+-------------------------+---------------------------------|
| Scalar expression     | `x + y`                 | Basic symbolic composition      |
| Affine map            | `Ax + b`                | Common neural layer form        |
| Matrix multiplication | `AB`                    | Core tensor operation           |
| Reduction             | `sum_i x_i`             | Common aggregation primitive    |
| Gradient              | `grad f(x)`             | Autodiff routing                |
| Graph expression      | `G = (V, E)`            | Topology-aware attention        |
| Recurrence            | `h_t = f(h_{t-1}, x_t)` | State-space/RNN routing         |
| Constraint            | `Ax <= b`               | Optimization and verifier hooks |

The first parser can use a constrained syntax rather than full LaTeX.

Recommended initial syntax:

```text
affine(A, x, b)
matmul(A, B)
add(x, y)
sum(i, x_i)
grad(f, x)
constraint(leq(matmul(A, x), b))
```

This avoids spending the first iteration on full symbolic parsing.

---

## 8. Symbolic IR

Define a compact intermediate representation.

Minimal node:

```python
@dataclass(frozen=True)
class MathNode:
    op: str
    args: tuple["MathNode", ...]
    value: str | float | int | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
```

Required invariants:

- Nodes are immutable.
- Nodes have stable hashes.
- Equivalent normalized forms should hash consistently.
- Shape and dtype are explicit when known.
- Operator names are canonical.

Example:

```text
Ax + b
```

Normalized IR:

```text
add(matmul(A, x), b)
```

---

## 9. Normalization Rules

Normalize before embedding.

Initial rules:

| Input      | Normalized                 |
|------------+----------------------------|
| `x + y`    | `add(x, y)`                |
| `y + x`    | `add(x, y)` if commutative |
| `A x`      | `matmul(A, x)`             |
| `A(x + y)` | `matmul(A, add(x, y))`     |
| `Wx + b`   | `add(matmul(W, x), b)`     |
| `x + 0`    | `x`                        |
| `x * 1`    | `x`                        |

Do not over-normalize in v0.

Avoid unsafe transformations unless exact.

---

## 10. Math Embedding Model

The embedding model maps symbolic IR into vector space.

```text
E_theta: M -> R^d
```

Output:

```text
z_i = E_theta(m_i)
```

Initial implementation options:

### Option A: Hand-built structural embedding

Use operator IDs, shape features, depth, arity, and subtree hashes.

Pros:

- Fast.
- Deterministic.
- Easy to debug.
- Good for proving the routing layer.

Cons:

- Limited semantic generalization.

### Option B: Small learned tree encoder

Use a recursive encoder over IR nodes.

Pros:

- Learns similarity beyond exact syntax.
- Better long-term architecture.

Cons:

- Requires dataset and training loop.

### Recommended v0

Start with Option A, then add Option B.

The key prototype question is not whether the embedding is perfect.

The key question is whether math-derived topology can reduce attention cost while preserving task quality.

---

## 11. Topology Builder

The topology graph combines symbolic edges and embedding-neighbor edges.

```text
G = (V, E_symbolic union E_embedding union E_local)
```

Edge types:

| Edge type         | Meaning                       |
|-------------------+-------------------------------|
| parent_child      | IR dependency edge            |
| same_operator     | Same operator class           |
| same_shape        | Compatible tensor shape       |
| topk_embedding    | Nearby math embedding         |
| local_sequence    | Nearby sequence position      |
| verifier_required | Must preserve for correctness |

The attention mask is built from this graph.

```text
A_ij = 1 if edge(i, j) exists or j in TopK(z_i)
```

Keep local sequence windows as a safety fallback:

```text
A = symbolic_edges OR embedding_topk OR local_window
```

---

## 12. Math-Routed Attention

Standard attention:

```text
Attn(Q,K,V) = softmax(QK^T / sqrt(d)) V
```

Math-routed sparse attention:

```text
MathAttn(Q,K,V,A) = softmax((QK^T / sqrt(d)) + mask(A)) V
```

where:

```text
mask(A_ij) = 0 if A_ij = 1
mask(A_ij) = -inf if A_ij = 0
```

Implementation target:

```python
def math_attention(q, k, v, mask):
    scores = q @ k.transpose(-2, -1) / sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, -float("inf"))
    probs = softmax(scores, dim=-1)
    return probs @ v
```

v0 can use dense masked attention for correctness.

v1 should use actual sparse/block-sparse kernels.

---

## 13. Complexity Target

Full attention cost:

```text
O(n^2 d)
```

Sparse math attention cost:

```text
O(n k d)
```

Expected gain:

```text
gain ~= n / k
```

Example:

```text
n = 4096
k = 64
gain ~= 64x relation reduction
```

This is a theoretical relation-count reduction.

Actual wall-clock speedup depends on kernel implementation, batching, sparsity layout, and memory traffic.

---

## 14. Expert and Kernel Router

Use the math embedding and hidden state to route tokens/subgraphs to specialized modules.

```text
e_i = Router(z_i, h_i)
```

Initial experts:

| Expert            | Handles                           |
|-------------------+-----------------------------------|
| affine_expert     | `Ax + b`                          |
| matmul_expert     | `AB`, `Ax`                        |
| reduction_expert  | `sum`, `mean`, `norm`             |
| graph_expert      | graph neighborhoods               |
| grad_expert       | gradient/autodiff forms           |
| constraint_expert | inequalities/equality constraints |
| generic_expert    | fallback                          |

Routing rule for v0:

```text
e_i = table_lookup(op_class(m_i))
```

Routing rule for v1:

```text
e_i = argmax_e RouterMLP([z_i, h_i])
```

Correctness requirement:

The generic expert must always be available.

No token should be dropped because routing is uncertain.

---

## 15. Verifier Gate

The verifier checks whether produced transformations or execution plans preserve constraints.

Verifier levels:

| Level | Check                                  |
|-------+----------------------------------------|
| L0    | Shape check                            |
| L1    | Dtype check                            |
| L2    | Operator compatibility                 |
| L3    | Symbolic equivalence for safe rewrites |
| L4    | Numerical test on sampled inputs       |
| L5    | Formal proof or external solver        |

v0 should implement L0-L2.

v1 should add L3-L4.

L5 is not required for this prototype.

---

## 16. Training Tasks

Start with synthetic supervised tasks.

Task families:

| Task                    | Input                   | Target                   |
|-------------------------+-------------------------+--------------------------|
| Operator classification | expression              | op class                 |
| Shape inference         | expression + symbols    | output shape             |
| Similarity retrieval    | pair of expressions     | same/different structure |
| Mask prediction         | expression sequence     | useful attention edges   |
| Rewrite validation      | before/after expression | valid/invalid            |
| Kernel routing          | expression              | expert/kernel ID         |

Do not start with open-ended theorem proving.

Use constrained tasks that directly evaluate the architecture.

---

## 17. Dataset Format

Use JSONL.

Example:

```json
{"id":"ex_0001","expr":"affine(A,x,b)","normalized":"add(matmul(A,x),b)","op_class":"affine","shape":{"A":[32,64],"x":[64],"b":[32],"out":[32]},"expert":"affine_expert"}
```

Pairwise similarity example:

```json
{"id":"pair_0001","a":"add(matmul(W,x),b)","b":"add(matmul(A,u),c)","label":"same_structure"}
```

Mask example:

```json
{"id":"mask_0001","nodes":["A","x","matmul(A,x)","b","add(matmul(A,x),b)"],"edges":[[0,2],[1,2],[2,4],[3,4]]}
```

---

## 18. Evaluation Metrics

Measure both quality and efficiency.

Quality metrics:

| Metric                    | Meaning                                |
|---------------------------+----------------------------------------|
| op_accuracy               | Correct operator classification        |
| shape_accuracy            | Correct shape inference                |
| route_accuracy            | Correct expert/kernel route            |
| retrieval_recall_at_k     | Similar structure retrieved            |
| rewrite_validity_accuracy | Correct valid/invalid rewrite judgment |
| task_loss                 | Downstream task loss                   |

Efficiency metrics:

| Metric             | Meaning                           |
|--------------------+-----------------------------------|
| attention_edges    | Number of allowed attention pairs |
| sparsity_ratio     | `allowed_edges / n^2`             |
| peak_memory        | Runtime memory usage              |
| tokens_per_second  | Throughput                        |
| wall_time          | End-to-end runtime                |
| verifier_fail_rate | Invalid outputs caught            |

Core success metric:

```text
same or better task quality with fewer attention edges
```

---

## 19. Prototype Phases

### Phase 0: Scaffolding

Deliverables:

- Create module directory.
- Add README.
- Add basic tests.
- Add tiny config.
- Add synthetic example data.

Acceptance:

```text
pytest tests/
```

passes.

---

### Phase 1: Symbolic IR

Deliverables:

- `ir.py`
- `parser.py`
- `normalize.py`
- tests for canonical forms

Supported forms:

```text
add(x,y)
matmul(A,x)
affine(A,x,b)
sum(i,x_i)
grad(f,x)
```

Acceptance:

```text
affine(A,x,b) -> add(matmul(A,x),b)
```

---

### Phase 2: Deterministic Math Embeddings

Deliverables:

- `embedder.py`
- structural feature vector
- deterministic embedding hash
- cosine similarity utility

Features:

- operator ID
- arity
- depth
- subtree count
- shape signature
- dtype ID
- commutativity flag

Acceptance:

Structurally similar expressions have higher similarity than unrelated expressions.

Example:

```text
sim(add(matmul(W,x),b), add(matmul(A,u),c))
  > sim(add(matmul(W,x),b), grad(f,x))
```

---

### Phase 3: Topology Mask

Deliverables:

- `topology.py`
- symbolic graph builder
- TopK embedding neighbor builder
- local window fallback
- boolean attention mask

Acceptance:

For an expression tree, parent-child nodes can attend to each other.

For structurally similar nodes, TopK edges are added.

---

### Phase 4: Math-Routed Attention Block

Deliverables:

- `attention.py`
- dense masked attention v0
- math mask integration
- baseline full attention comparison

Acceptance:

Masked attention returns same shape as full attention.

Mask density is lower than full attention.

---

### Phase 5: Expert Router

Deliverables:

- `router.py`
- rule-based op-class router
- generic fallback expert
- route diagnostics

Acceptance:

Expressions route correctly:

```text
affine(A,x,b) -> affine_expert
matmul(A,B) -> matmul_expert
sum(i,x_i) -> reduction_expert
grad(f,x) -> grad_expert
```

---

### Phase 6: Verifier

Deliverables:

- `verifier.py`
- shape checker
- dtype checker
- operator compatibility checker
- basic sampled numerical checker where possible

Acceptance:

Invalid matrix multiplication shapes fail.

Valid affine forms pass.

---

### Phase 7: Tiny Model

Deliverables:

- `model.py`
- end-to-end math-routed transformer block
- tiny training loop
- synthetic dataset

Acceptance:

The model runs one training step and one eval step.

---

### Phase 8: Benchmark

Deliverables:

- full attention baseline
- math mask attention variant
- sparsity report
- runtime report
- quality report

Acceptance:

Report includes:

```text
n
k
allowed_edges
sparsity_ratio
baseline_time
math_routed_time
quality_metric
```

---

## 20. Minimal v0 Implementation Order

Implement in this order:

1. `ir.py`
2. `parser.py`
3. `normalize.py`
4. `embedder.py`
5. `topology.py`
6. `attention.py`
7. `router.py`
8. `verifier.py`
9. `model.py`
10. `eval.py`

Do not start with the neural model.

Start with the symbolic substrate.

---

## 21. Design Rules

### Rule 1: Exact before learned

Use exact symbolic facts wherever available.

Use learned embeddings only where exact structure is unavailable, expensive, or too rigid.

### Rule 2: Routing is allowed to be approximate

Routing may be probabilistic.

Correctness may not be probabilistic.

### Rule 3: Keep fallback paths

Every sparse/routed mechanism needs a generic fallback.

### Rule 4: Measure sparsity explicitly

Never claim efficiency without counting edges and measuring runtime.

### Rule 5: Separate model quality from verifier quality

A bad model caught by a verifier is still a bad model.

A good verifier does not prove the model is strong.

Track both.

---

## 22. Initial Classes and APIs

### Math IR

```python
class MathNode:
    op: str
    args: tuple[MathNode, ...]
    value: object | None
    shape: tuple[int, ...] | None
    dtype: str | None
```

### Embedder

```python
class MathEmbedder:
    def encode(self, node: MathNode) -> Tensor:
        ...
```

### Topology Builder

```python
class TopologyBuilder:
    def build(self, nodes: list[MathNode], z: Tensor) -> BoolTensor:
        ...
```

### Attention

```python
class MathRoutedAttention(nn.Module):
    def forward(self, x: Tensor, mask: BoolTensor) -> Tensor:
        ...
```

### Router

```python
class OperatorRouter:
    def route(self, node: MathNode, z: Tensor, h: Tensor) -> str:
        ...
```

### Verifier

```python
class Verifier:
    def check(self, node: MathNode, plan: ExecutionPlan) -> VerificationResult:
        ...
```

---

## 23. First Toy Example

Input expression:

```text
affine(A,x,b)
```

Normalize:

```text
add(matmul(A,x),b)
```

Create nodes:

```text
A
x
matmul(A,x)
b
add(matmul(A,x),b)
```

Build symbolic edges:

```text
A -> matmul(A,x)
x -> matmul(A,x)
matmul(A,x) -> add(matmul(A,x),b)
b -> add(matmul(A,x),b)
```

Build attention mask:

```text
A_ij = parent_child OR child_parent OR local_window OR topk_embedding
```

Route:

```text
matmul(A,x) -> matmul_expert
add(matmul(A,x),b) -> affine_expert
```

Verify:

```text
shape(A) = [m,n]
shape(x) = [n]
shape(b) = [m]
shape(out) = [m]
```

---

## 24. Risk Register

| Risk                   | Failure mode                         | Mitigation                                      |
|------------------------+--------------------------------------+-------------------------------------------------|
| Bad embeddings         | Wrong topology edges                 | Keep symbolic and local fallback edges          |
| Sparse mask too strict | Model misses needed context          | Add global tokens and generic fallback          |
| Sparse kernels slow    | Wall-clock worse despite fewer edges | Start dense-masked, then benchmark block sparse |
| Parser complexity      | Prototype stalls                     | Use constrained S-expression syntax first       |
| Verifier too weak      | Invalid plans pass                   | Start with shape/dtype, expand slowly           |
| Router collapse        | Everything goes to generic expert    | Add route balancing metrics                     |
| Synthetic-only success | Does not generalize                  | Add progressively harder generated data         |

---

## 25. Key Experiments

### Experiment A: Structure Retrieval

Question:

```text
Do structurally equivalent expressions land near each other?
```

Metric:

```text
Recall@K
```

### Experiment B: Sparse Mask Quality

Question:

```text
Can topology masks preserve useful relations with fewer edges?
```

Metrics:

```text
edge_count
sparsity_ratio
task_accuracy
```

### Experiment C: Routing Accuracy

Question:

```text
Can math embeddings select the correct operator expert?
```

Metric:

```text
route_accuracy
```

### Experiment D: Attention Runtime

Question:

```text
Does sparse topology reduce runtime or only theoretical edge count?
```

Metrics:

```text
wall_time
peak_memory
tokens_per_second
```

### Experiment E: Verifier Catch Rate

Question:

```text
Does verifier catch invalid shapes, rewrites, and routes?
```

Metrics:

```text
fail_rate_on_invalid
pass_rate_on_valid
```

---

## 26. Success Criteria

The prototype is successful if it demonstrates all of the following:

1. Math objects can be parsed into stable symbolic IR.
2. Similar math structures get similar embeddings.
3. A topology mask can be built from symbolic and embedding edges.
4. Attention can run with that mask.
5. The mask uses fewer edges than full attention.
6. Routing can select operator-specific experts.
7. Basic verifier checks catch invalid execution plans.
8. A benchmark compares full attention vs math-routed attention.

Minimum acceptable result:

```text
same output shape
lower attention density
working route diagnostics
passing verifier tests
```

Strong result:

```text
similar task quality
meaningfully lower attention density
measurable runtime or memory improvement
```

---

## 27. Immediate Next Steps

Execute these next:

```text
mkdir -p neurosymbolic/math_transformer/src
mkdir -p neurosymbolic/math_transformer/tests
mkdir -p neurosymbolic/math_transformer/configs
mkdir -p neurosymbolic/math_transformer/data
mkdir -p neurosymbolic/math_transformer/scripts
```

Then create:

```text
src/ir.py
src/parser.py
src/normalize.py
tests/test_ir.py
tests/test_parser.py
```

First implementation target:

```text
affine(A,x,b) parses and normalizes to add(matmul(A,x),b)
```

First test target:

```text
pytest neurosymbolic/math_transformer/tests/test_parser.py
```

---

## 28. Final Architecture Statement

The intended architecture is:

```text
Math-Routed Sparse Transformer =
  symbolic IR
  + math embedding geometry
  + topology-constrained attention
  + operator/expert routing
  + execution planning
  + verifier gate
```

Core thesis:

```text
Do not force the transformer to rediscover math topology from raw tokens.
Give it the topology first, then use attention only where attention is useful.
```

