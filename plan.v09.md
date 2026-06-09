# plan.v9.md — Hard Validation + Learned Topology Scorer Gate

## Diagnosis

The current v7 result is a real systems milestone, but not yet a research proof.

Current proof:

```text
hand-scored topology TopK sparse attention preserves easy route labels
```

Needed proof:

```text
learned topology sparse attention preserves dense quality on hard reasoning graphs
and beats dense block runtime end-to-end
```

## Non-Negotiable Gates

### Quality Gate

```text
hard_route_acc(K=8) >= 0.98 * dense_route_acc
hard_route_acc(K=4) >= 0.95 * dense_route_acc
dense_agree(K=8) >= 0.98
hidden_cos(K=8) >= 0.98
logit_kl(K=8) <= 0.05
```

### Speed Gate

```text
prepared_sparse_block_ms < dense_block_ms
topology_build_ms reported separately
```

Sparse speed claims must separate:

```text
topology_build_ms
prepared_sparse_attention_ms
prepared_sparse_block_ms
dense_block_ms
```

## Implementation Order

### 1. Strengthen Quality Evaluation

Add sparse-vs-dense metrics beyond class agreement:

```text
hidden_l1
hidden_cos
logit_l1
logit_kl_dense_to_sparse
```

Reason: `dense_agree=1.0` only proves same route argmax. It does not prove comparable hidden computation.

### 2. Add Hard Synthetic Validation

Add a hard-data mode with:

```text
nested expressions
deeper expression trees
mixed symbolic families
held-out templates
irrelevant distractor branches
longer dependency chains
more unique normalized expressions
```

Minimum target:

```text
unique_route_expr >= 1000
```

### 3. Keep Current Topology Name Honest

Current topology is:

```text
hand-scored topology TopK
```

Do not call it learned topology until:

```text
s_ij = f_theta(node_i, node_j, relation_features)
```

and `f_theta` is trainable and checkpointed.

### 4. Implement Learned Edge Scorer

Add:

```text
src/learned_topology.py
```

Minimal scorer:

```text
node_i embedding
node_j embedding
relation feature vector
relative position features
shape compatibility features
→ edge_score
```

### 5. Generate Dense Teacher Traces

For each hard example, save:

```text
expr
env
dense logits
dense hidden states
optional dense attention proxy / selected dense top edges
```

### 6. Train Sparse Topology Scorer

Training objective:

```text
L = CE(y_sparse, y_true)
  + lambda_kl * KL(y_dense || y_sparse)
  + lambda_h * ||h_sparse - h_dense||^2
  + lambda_edge * edge_target_loss
```

### 7. Verify K Compression

Run:

```text
K=16 hand-scored topology
K=8 learned topology
K=4 learned topology
dense full attention
```

Claim only if:

```text
K=8 learned quality >= K=16 hand quality
K=8 learned prepared block < dense block
```

## Current Slice Implemented First

This plan starts with:

```text
quality metrics + hard validation scaffolding
```

because learned scorer training without a hard validation gate risks optimizing against the wrong target.

## Implemented v9 Slices

### Slice 1: Evaluation Gate

```text
QualityReport now includes hidden_l1, hidden_cos, logit_l1, logit_kl.
Hard synthetic generation is available via --hard.
```

### Slice 2: Hard Pipeline + Teacher Traces

```text
scripts/generate_hard_synthetic.sh
scripts/train_hard_synthetic.sh
scripts/run_hard_quality_check.sh
scripts/export_dense_teacher_traces.sh
src/teacher_traces.py
```

### Slice 3: Learned Scorer Bootstrap

```text
src/learned_topology.py
src/train_topology_scorer.py
scripts/train_topology_scorer.sh
```

Current scorer training target is deliberately conservative:

```text
learned scorer imitates hand-scored K=16 topology
```

This is not the final dense-attention teacher target. It is a bootstrap stage to verify:

```text
f_theta(phi_ij) can recover the current symbolic topology before replacing it.
```
