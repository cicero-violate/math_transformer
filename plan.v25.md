# Plan v25 — Sparse Student Distillation from Validated Graph Prior

## Official Name

```text
Graph-Prior Knowledge Distillation Student
```

Runtime intent:

```text
validated checkpoint graph prior A_qwen -> fixed/controlled sparse student topology -> teacher behavior artifact -> sparse student distillation -> measured quality/runtime/memory gates
```

## Why v25 Exists

v24 validates whether the compiled checkpoint graph prior has signal.

v25 moves the project from:

```text
G_0-derived topology helps or fails under task/evaluator baselines
```

to:

```text
a sparse graph-native student can absorb teacher behavior under fixed or controlled topology
```

The key correction is:

```text
Do not mix distillation and adaptive rewiring in the first behavioral transfer proof.
First prove the student can learn teacher behavior with topology fixed or tightly controlled.
```

## v24 Expected Completion State

Required before v25 positive claims:

```text
Graph-prior loader CLI exists.
Matched random baselines exist.
Baseline matrix compares dense, hand K=4, learned K=4, random, and qwen-prior adjacency.
Qwen-prior adjacency has a clear positive/neutral/negative decision.
No teacher checkpoint is needed at student runtime.
```

Important distinction:

```text
v24 proves topology signal.
v25 proves behavior transfer.
v26 will prove adaptive topology learning.
```

## v25 Decision

```text
Use teacher behavior artifacts for training, not live teacher inference at student runtime.
Start with top-r logits and task labels.
Keep A fixed or controlled during the first distillation proof.
Record edge/activation/gradient traces for v26, but do not use them to mutate topology yet.
```

Correct pipeline:

```text
G_0 / A_qwen
  -> sparse student initialization
  -> teacher behavior artifact generation or ingestion
  -> top-r logit distillation
  -> task-label auxiliary loss
  -> quality/runtime/memory evaluation
  -> edge trace artifacts for v26
```

Rejected pipeline:

```text
distillation + adaptive rewiring + graph growth all in one experiment
```

## Core Objects

```text
T = teacher model or teacher behavior artifact.
S = sparse graph-native student.
G_0 = compiled checkpoint-derived graph prior.
A_0 = fixed sparse adjacency selected from G_0.
θ = trainable student parameters.
z_T = teacher logits.
z_S = student logits.
τ = distillation temperature.
R = number of stored top logits.
Q = task/evaluator quality.
M = measured memory.
T_runtime = measured runtime.
```

Distillation artifact types:

| artifact | meaning |
|---|---|
| top_r_logits.jsonl | compressed teacher probability targets |
| task_labels.jsonl | existing labels / verifier outputs |
| teacher_metadata.json | teacher identity, tokenizer, config hash, generation settings |
| distill_examples.jsonl | prompt/input IDs and target metadata |
| edge_trace.jsonl | per-edge usage stats for v26 |
| gradient_edge_stats.jsonl | per-edge gradient/utility stats for v26 |

## Distillation Formula

Teacher softened distribution:

```text
p_T^τ(y|x) = softmax(z_T(x) / τ)
```

Student softened distribution:

```text
p_S^τ(y|x) = softmax(z_S(x) / τ)
```

Distillation loss:

```text
L_KD = τ^2 KL(p_T^τ || p_S^τ)
```

v25 loss:

```text
L_v25 = L_KD + α L_task + λ |A_0| + γ M + β T_runtime
```

With fixed topology:

```text
θ_{t+1} = θ_t - η ∇_θ L_v25
A_{t+1} = A_t
```

Top-r compressed teacher target:

```text
TopR(z_T) = {(token_j, logit_j): j ∈ top R tokens}
```

Storage reduction:

```text
O(NR) instead of O(N|V|)
```

## Artifact Contract

Output directory:

```text
runs/sparse_student_distill/<teacher_id>/<student_id>/<experiment_id>/
```

Required artifacts:

```text
distill_config.json
teacher_metadata.json
student_config.json
train_metrics.jsonl
eval_metrics.json
quality_report.json
memory_report.json
runtime_report.json
kd_delta_report.json
prior_delta_report.json
edge_trace.jsonl
gradient_edge_stats.jsonl
```

Optional artifacts:

```text
top_r_logits.jsonl
hidden_summary.jsonl
calibration_report.json
failure_cases.jsonl
```

`distill_config.json` must record:

```text
source_weight_graph_dir
source_prior_experiment_dir
teacher_id
teacher_behavior_artifact
student_topology
fixed_adjacency = true
allow_rewiring = false
top_r
temperature
task_loss_weight
kd_loss_weight
memory_budget
runtime_protocol
```

Hard rule:

```text
v25 may record rewiring signals.
v25 must not use those signals to mutate A during the main distillation proof.
```

## Implementation Plan

### P0.1 — Teacher Behavior Artifact Schema

Add compact top-r logit target format:

```json
{
  "sample_id": "hash",
  "input_hash": "hash",
  "teacher_id": "qwen-small-or-fixture",
  "position": 0,
  "temperature": 2.0,
  "top_r": [
    {"token_id": 123, "logit": 4.25},
    {"token_id": 456, "logit": 3.91}
  ],
  "tail_mass_estimate": 0.0
}
```

Acceptance:

```text
Schema stores top-r logits only, not full-vocab logits by default.
Teacher metadata includes tokenizer/config hashes.
Artifacts are reproducible by sample_id and input_hash.
```

### P0.2 — Fixed-Topology Distillation Runner

Add:

```text
src/sparse_student_distill.py
```

Acceptance:

```text
Loads A_0 from v24 accepted qwen-prior artifact.
Loads task labels and optional top-r teacher logits.
Trains/evaluates student parameters θ with fixed adjacency.
Does not call the teacher during student runtime.
Does not mutate A in the main proof.
```

### P1.1 — KD Delta Gate

Compare:

```text
S(A_qwen + task labels)
S(A_qwen + task labels + KD)
S(A_random + task labels + KD)
S(A_hand_K4 + task labels + KD)
S(A_learned_K4 + task labels + KD)
```

Acceptance:

```text
Δ_KD = Q(S(A_qwen + KD)) - Q(S(A_qwen + labels)) > 0
Δ_prior = Q(S(A_qwen + KD)) - Q(S(A_random + KD)) > 0
Regression by expert class is reported.
```

### P1.2 — Runtime/Memory Distillation Gate

Acceptance:

```text
M_student <= configured memory budget.
Runtime reports cold/cache/static regimes separately.
No speed proof is claimed unless repeated locked distributional metrics pass.
```

### P1.3 — Edge Trace Collection for v26

Record but do not apply:

```text
edge_activation_frequency
edge_gradient_norm
edge_loss_contribution
edge_confidence
edge_compute_cost
edge_source_teacher
edge_source_tensor
```

Acceptance:

```text
edge_trace.jsonl and gradient_edge_stats.jsonl are emitted.
Trace collection can be disabled.
Trace collection does not change A during v25.
```

### P2.1 — Teacher Ladder

Use staged teachers:

```text
tiny fixture -> small local Qwen-style teacher -> Qwen 1.5B/3B -> Qwen 7B -> Qwen 30B/A3B behavior artifact
```

Acceptance:

```text
Each teacher stage has a separate artifact directory and report.
No large-teacher claim is made from a small-teacher run.
No full-30B runtime dependency is introduced into the student.
```

### P2.2 — Optional Hidden Summary Distillation

Later add:

```text
L_hidden = Σ_i || P_i h_S^i - h_T^{m(i)} ||_2^2
```

Acceptance:

```text
Hidden summaries are optional and gated.
Logit KD remains the first proof path.
No dense-transformer internal geometry is forced unless it improves measured gates.
```

## First Valid v25 Experiment

```text
teacher ∈ {tiny_fixture_teacher, small_qwen_style_teacher}
A ∈ {qwen_K4, random_K4, hand_K4, learned_K4}
top_r ∈ {16,32,64}
temperature ∈ {1,2,4}
loss_mix ∈ {KD_only, task_plus_KD}
```

Acceptance:

```text
KD artifacts validate.
Fixed A is enforced.
Δ_KD > 0 or explicitly marked v25_negative.
Δ_prior > 0 or explicitly marked prior_not_helping_distillation.
Quality/runtime/memory reports are emitted.
Existing test suite remains green.
```

## Non-Negotiable Gates

1. Do not use live teacher inference at student runtime.
2. Do not require the large teacher checkpoint for deployed student inference.
3. Do not store full-vocab logits unless explicitly configured and budgeted.
4. Do not store raw transformer weights in distillation or graph records.
5. Do not adapt/rewire A during the primary v25 proof.
6. Do not claim G_0 helps distillation unless qwen+KD beats random+KD.
7. Do not claim distillation helps unless KD beats labels-only under same A.
8. Do not claim speed proof unless repeated locked distributional artifacts pass.
9. Do not delete/archive student edges without paired quality, speed, and memory gates.
10. Keep quantale planner work inside `math_transformer`.

## Decision

```text
v25 is the fixed-topology sparse student distillation plan.
The next artifact is not adaptive rewiring.
The next artifact is proof that a sparse graph-native student can absorb teacher behavior under controlled topology.
```

Summary formula:

```text
G_0 -> A_qwen -> fixed sparse student -> top-r KD + task loss -> Δ_KD and Δ_prior gates -> traces for v26
```
