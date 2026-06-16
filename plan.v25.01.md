# Plan v25.01 — Local Ollama Qwen 3B Teacher to Fixed Sparse Student

## Official Name

```text
Qwen25-3B-Smart Sparse Student Distillation Run
```

Runtime intent:

```text
local Ollama teacher qwen25-3b-smart:latest
  -> teacher response artifacts
  -> fixed/controlled sparse graph-native student
  -> supervised teacher-response distillation
  -> quality/runtime/memory gates
  -> edge/gradient traces for v26
```

## Purpose

`plan.v25.md` defines the general sparse-student distillation proof.

`plan.v25.01.md` makes that plan executable for the local teacher now available on this machine:

```text
teacher_id = qwen25-3b-smart:latest
teacher_backend = ollama
student_family = sparse graph-native math_transformer student
student_topology = fixed A_qwen / controlled sparse adjacency
```

This is the first practical local run:

```text
3B local teacher -> sparse student behavior transfer
```

## Critical Correction from v25

The v25 plan starts with top-r logit distillation:

```text
TopR(z_T) -> KL / KD loss
```

But the selected local teacher is served through Ollama, and Ollama's standard API exposes generated text, not full logits or top-r logits.

Therefore v25.01 must not claim logit KD.

Correct v25.01 path:

```text
teacher generated answer y_T
student distribution p_S(. | x, y_<t)
L_text = - Σ_t log p_S(y_T,t | x, y_T,<t)
```

Optional future path if a logit-capable teacher backend is added:

```text
L_KD = τ^2 KL(p_T^τ || p_S^τ)
```

v25.01 is therefore:

```text
teacher-response distillation first,
logit distillation optional later.
```

## Core Decision

```text
Use qwen25-3b-smart:latest as the local text teacher.
Do not use ZSE as the teacher runtime for this run.
Use Ollama because it is fast and stable on the GTX 1050 machine.
Keep the sparse topology fixed during the main proof.
Record topology/edge traces, but do not mutate A during v25.01.
```

## Teacher

Primary teacher:

```text
qwen25-3b-smart:latest
```

Fallback teacher:

```text
qwen2.5:3b
```

Fast smoke teacher:

```text
qwen25-1.5b-q8-smart:latest
```

Critic/verifier only:

```text
gemma3:4b
gemma3-4b-lowvram:latest
```

Rules:

```text
Use Qwen as the main teacher for Qwen-style sparse student behavior.
Use Gemma only as a disagreement critic or failure-case filter.
Do not mix teacher families in the main positive claim.
```

## Student

Student target:

```text
S = sparse graph-native student
A = fixed adjacency from accepted v24/v25 graph-prior artifact
θ = trainable student parameters
```

Allowed topology modes:

| mode | meaning | allowed in v25.01 main proof |
|---|---|---|
| `A_qwen_fixed` | accepted graph-prior sparse adjacency | yes |
| `A_random_fixed` | matched random K baseline | yes |
| `A_hand_K4_fixed` | hand sparse baseline | yes |
| `A_learned_K4_fixed` | learned sparse baseline | yes |
| adaptive rewiring | mutate edges during training | no |
| graph growth/pruning | add/delete topology during proof | no |

Hard rule:

```text
A_{t+1} = A_t
```

Only student weights update:

```text
θ_{t+1} = θ_t - η ∇_θ L_v25.01
```

## Distillation Objective

Teacher-response supervised loss:

```text
L_text = - Σ_t log p_S(y_T,t | x, y_T,<t)
```

Task-label auxiliary loss:

```text
L_task = supervised loss on canonical task labels / verifier labels
```

Sparse/runtime measurements:

```text
L_sparse = λ_edge |A| + λ_act activation_cost + λ_mem memory_cost
L_runtime = β T_runtime
```

v25.01 training loss:

```text
L_v25.01 = w_text L_text + w_task L_task + L_sparse + L_runtime
```

Default first-run weights:

```text
w_text = 1.0
w_task = 0.25
λ_edge = 0.0 if A is fixed and edge count is constant
λ_act = measured-only for first run
λ_mem = measured-only for first run
β = measured-only for first run
```

Reason:

```text
The first proof should show behavior transfer, not optimize every budget simultaneously.
```

## Dataset Design

Dataset families:

| family | purpose |
|---|---|
| arithmetic_short | exact small reasoning |
| symbolic_short | algebraic manipulation |
| logic_short | discrete reasoning |
| code_short | optional code behavior |
| refusal_boundary_safe | harmless instruction discipline |
| project_specific_math_transformer | sparse topology / graph-prior reasoning prompts |

Scale ladder:

```text
N_smoke = 128 examples
N_gate = 1,024 examples
N_train = 8,192 examples
N_full_local = 32,768 examples
```

Do not jump to `N_full_local` until `N_smoke` and `N_gate` pass.

## Teacher Generation Protocol

Teacher backend:

```text
ollama
```

Teacher endpoint:

```text
http://127.0.0.1:11434/api/generate
```

Teacher model:

```text
qwen25-3b-smart:latest
```

Default generation settings:

```json
{
  "temperature": 0.2,
  "top_p": 0.9,
  "repeat_penalty": 1.05,
  "num_predict": 256,
  "seed": 25
}
```

For exact arithmetic labels:

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "num_predict": 128
}
```

Teacher prompt format:

```text
You are a precise teacher generating distillation targets for a small sparse student.
Answer compactly. Show only necessary reasoning. End with a final answer field.

Problem:
{problem}

Return:
reasoning: ...
answer: ...
```

## Artifact Contract

Output directory:

```text
runs/sparse_student_distill/qwen25-3b-smart/sparse_student/<experiment_id>/
```

Required artifacts:

```text
distill_config.json
teacher_metadata.json
teacher_prompts.jsonl
teacher_responses.jsonl
teacher_response_quality.jsonl
student_config.json
train_metrics.jsonl
eval_metrics.json
quality_report.json
runtime_report.json
memory_report.json
kd_delta_report.json
prior_delta_report.json
edge_trace.jsonl
gradient_edge_stats.jsonl
failure_cases.jsonl
```

Optional artifacts:

```text
top_r_logits.jsonl
teacher_pairwise_preferences.jsonl
critic_disagreements.jsonl
hidden_summary.jsonl
calibration_report.json
```

`teacher_metadata.json` must record:

```json
{
  "teacher_id": "qwen25-3b-smart:latest",
  "teacher_backend": "ollama",
  "ollama_model_digest": "record_if_available",
  "ollama_show": "raw ollama show output or path to it",
  "generation_settings": {},
  "created_at": "ISO-8601",
  "logits_available": false
}
```

`distill_config.json` must record:

```json
{
  "plan": "v25.01",
  "teacher_id": "qwen25-3b-smart:latest",
  "teacher_backend": "ollama",
  "teacher_artifact_type": "text_response",
  "source_weight_graph_dir": "...",
  "source_prior_experiment_dir": "...",
  "student_topology": "A_qwen_fixed",
  "fixed_adjacency": true,
  "allow_rewiring": false,
  "loss": "teacher_text_sft_plus_task",
  "top_r": null,
  "temperature": null,
  "task_loss_weight": 0.25,
  "teacher_text_loss_weight": 1.0,
  "memory_budget": "recorded",
  "runtime_protocol": "locked_eval"
}
```

## JSONL Schemas

### `teacher_prompts.jsonl`

```json
{
  "sample_id": "sha256",
  "family": "arithmetic_short",
  "split": "train",
  "prompt": "Problem text",
  "expected_answer": "optional verifier answer",
  "source": "synthetic_v25_01"
}
```

### `teacher_responses.jsonl`

```json
{
  "sample_id": "sha256",
  "teacher_id": "qwen25-3b-smart:latest",
  "prompt_hash": "sha256",
  "response": "reasoning: ...\nanswer: ...",
  "answer_extracted": "...",
  "latency_ms": 0,
  "eval_options": {
    "temperature": 0.2,
    "top_p": 0.9,
    "num_predict": 256
  }
}
```

### `teacher_response_quality.jsonl`

```json
{
  "sample_id": "sha256",
  "answer_match": true,
  "critic_pass": true,
  "parse_pass": true,
  "kept_for_training": true,
  "drop_reason": null
}
```

### Optional `top_r_logits.jsonl`

Only valid if a non-Ollama or modified-Ollama backend provides logits:

```json
{
  "sample_id": "sha256",
  "teacher_id": "qwen25-3b-smart:latest",
  "backend": "logit_capable_backend",
  "position": 0,
  "temperature": 2.0,
  "top_r": [
    {"token_id": 123, "logit": 4.25}
  ]
}
```

Hard rule:

```text
Do not create fake top-r logits from generated text.
```

## Implementation Plan

### P0.1 — Capture Teacher Metadata

Add or use:

```text
scripts/capture_ollama_teacher_metadata.py
```

Acceptance:

```text
Teacher identity is pinned before generating examples.
Model tag, digest, quantization, context length, and generation settings are recorded.
Run fails if teacher is not available locally.
```

### P0.2 — Generate Prompt Set

Add or use:

```text
scripts/generate_v25_01_prompts.py
```

Acceptance:

```text
N_smoke=128 prompt set is deterministic.
Each sample has family, split, prompt, and optional expected_answer.
No teacher response is mixed into prompt generation.
```

### P0.3 — Query Ollama Teacher

Add or use:

```text
scripts/query_ollama_teacher.py
```

Acceptance:

```text
Can generate N_smoke without manual intervention.
Handles Ollama server unavailable errors clearly.
Does not overwrite existing samples unless --force is passed.
```

### P0.4 — Filter / Verify Teacher Outputs

Add or use:

```text
scripts/filter_teacher_responses.py
```

Acceptance:

```text
Each teacher response receives kept_for_training=true/false.
Failure reasons are explicit.
At least 90% of arithmetic_short smoke examples pass exact verification.
```

### P1.1 — Build Student Training Examples

Add or use:

```text
scripts/build_v25_01_distill_examples.py
```

Training target format:

```text
input = prompt / problem statement
target = teacher response with answer field
loss_mask = target tokens only unless configured otherwise
```

Acceptance:

```text
Token counts are recorded.
Examples above target length budget are dropped or truncated with report.
No teacher runtime is required after this artifact is built.
```

### P1.2 — Fixed Sparse Student Runner

Add or use:

```text
src/sparse_student_distill.py
```

Required CLI:

```bash
python -m src.sparse_student_distill \
  --plan v25.01 \
  --teacher-artifacts runs/sparse_student_distill/qwen25-3b-smart/sparse_student/<experiment_id>/ \
  --adjacency artifacts/v24/<accepted_A_qwen>.json \
  --student-config configs/sparse_student_v25_01.json \
  --fixed-adjacency true \
  --allow-rewiring false
```

Acceptance:

```text
Loads fixed A.
Trains θ only.
Does not mutate A.
Emits train_metrics.jsonl and student checkpoint.
Can run on N_smoke without crashing.
```

### P1.3 — Baseline Matrix

Run:

```text
S(A_qwen_fixed + task only)
S(A_qwen_fixed + teacher text)
S(A_random_fixed + teacher text)
S(A_hand_K4_fixed + teacher text)
S(A_learned_K4_fixed + teacher text)
```

Acceptance:

```text
Δ_teacher_text = Q(S(A_qwen + teacher_text)) - Q(S(A_qwen + task_only))
Δ_prior = Q(S(A_qwen + teacher_text)) - Q(S(A_random + teacher_text))
```

Pass conditions:

```text
Δ_teacher_text > 0 for at least one locked eval family.
Δ_prior >= 0 for smoke.
No positive graph-prior claim unless Δ_prior > 0 on gate/full run.
```

### P1.4 — Evaluation Protocol

Evaluate on:

```text
heldout arithmetic_short
heldout symbolic_short
heldout logic_short
project_specific_math_transformer eval prompts
runtime/memory microbench
```

Acceptance:

```text
Teacher artifacts are never used as evaluation answers unless split=eval and marked heldout.
All eval prompts are locked before training.
Runtime metrics are measured with fixed batch/sequence settings.
```

### P1.5 — Edge and Gradient Traces for v26

Record:

```text
edge_activation_frequency
edge_gradient_norm
edge_loss_contribution
edge_confidence
edge_compute_cost
edge_source_prior
edge_source_tensor
```

Acceptance:

```text
edge_trace.jsonl exists.
gradient_edge_stats.jsonl exists.
Trace collection can be disabled with --no-edge-trace.
Trace collection does not alter topology.
```

## First v25.01 Experiment

Experiment id:

```text
v25_01_qwen25_3b_smart_sparse_smoke
```

Teacher:

```text
qwen25-3b-smart:latest
```

Student topology:

```text
A_qwen_fixed
A_random_fixed
```

Dataset:

```text
N_smoke = 128
families = arithmetic_short, symbolic_short, logic_short, project_specific_math_transformer
```

Training:

```text
loss = teacher_text_sft_plus_task
w_text = 1.0
w_task = 0.25
fixed_adjacency = true
allow_rewiring = false
```

Pass/fail:

```text
PASS if teacher generation succeeds, filtering succeeds, training runs, reports emit, and Δ_teacher_text is non-negative on smoke.
FAIL-CLEAN if artifacts emit and Δ_teacher_text <= 0.
FAIL-BUG if scripts crash, topology mutates, or artifacts are missing.
```

## Scale-Up Ladder

```text
v25_01_smoke_128
  -> v25_01_gate_1024
  -> v25_01_train_8192
  -> v25_01_full_local_32768
```

Only advance if:

```text
artifact schema validates
teacher response quality is acceptable
training is stable
runtime/memory reports exist
quality delta is non-negative or failure is explained
```

## Non-Negotiable Gates

1. Do not use ZSE teacher inference for v25.01.
2. Do not use live teacher inference at student runtime.
3. Do not claim logit KD from Ollama outputs.
4. Do not fabricate top-r logits.
5. Do not mutate sparse topology during the main proof.
6. Do not claim graph-prior benefit unless A_qwen beats matched random under the same teacher artifacts.
7. Do not claim teacher distillation benefit unless teacher-text training beats task-only under the same A.
8. Do not mix Qwen and Gemma outputs in the main positive claim.
9. Do not scale beyond N_smoke until artifacts and filtering pass.
10. Do not archive/delete edges based on v25.01 traces; edge mutation belongs to v26.

## Expected Outputs

Minimum successful smoke run:

```text
runs/sparse_student_distill/qwen25-3b-smart/sparse_student/v25_01_qwen25_3b_smart_sparse_smoke/
  distill_config.json
  teacher_metadata.json
  teacher_prompts.jsonl
  teacher_responses.jsonl
  teacher_response_quality.jsonl
  student_config.json
  train_metrics.jsonl
  eval_metrics.json
  quality_report.json
  runtime_report.json
  memory_report.json
  kd_delta_report.json
  prior_delta_report.json
  edge_trace.jsonl
  gradient_edge_stats.jsonl
  failure_cases.jsonl
```

## Decision

```text
v25.01 is the local executable sparse-student distillation plan using qwen25-3b-smart:latest as an Ollama text teacher.
It is not logit KD unless a logit-capable teacher backend is added.
The proof target is behavior transfer into a fixed-topology sparse student.
```

Summary formula:

```text
qwen25-3b-smart:latest via Ollama
  -> verified teacher text artifacts
  -> fixed A_qwen sparse student
  -> supervised teacher-response distillation + task labels
  -> Δ_teacher_text and Δ_prior gates
  -> edge/gradient traces for v26
```