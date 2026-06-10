# Plan v11 — Speed/Quality Proof and Runtime-J-Aligned Training

## Current State

v10 completed the low-learning-rate runtime-J selection sweep.

Selected runtime-J checkpoint:

```text
runs/checkpoints/scorer_runtime_j_best.pt
```

This is copied from:

```text
runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e4.runtime_best.pt
```

Full validation ranking:

```text
rank  checkpoint                                                     runtime_J
1     scorer_dense_mix_0p25_finetune_lr1e4.runtime_best.pt           1.6820821
2     scorer_dense_mix_0p25_finetune_lr3e5.runtime_best.pt           1.6820783
3     scorer_dense_mix_0p25_finetune.runtime_best.pt                 1.6820765
4     scorer_dense_mix_0p25_finetune_lr1e5.runtime_best.pt           1.6820755
```

Current selected learned K=8 full validation:

```text
route_acc   = 0.9866
dense_agree = 0.9944
hidden_l1   = 0.162904
hidden_cos  = 0.978373
logit_l1    = 0.100872
logit_kl    = 0.013007
```

Hand topology K=16 full validation:

```text
route_acc   = 0.9821
dense_agree = 0.9989
hidden_l1   = 0.079942
hidden_cos  = 0.992537
logit_l1    = 0.046326
logit_kl    = 0.002046
```

Quality comparison:

```text
learned K=8 route_acc > hand K=16 route_acc
0.9866 > 0.9821
```

Dense-equivalence comparison:

```text
learned K=8 still trails hand K=16 on hidden/logit equivalence
```

## v11 Main Objective

Turn the learned-topology result into an end-to-end speed/quality proof.

Required proof:

```text
learned K=8 route_acc >= hand K=16 route_acc
learned K=8 prepared sparse block faster than hand K=16 prepared sparse block
```

Secondary objective:

```text
replace topology-only fine-tuning with runtime-J-aligned training signals
```

## Runtime-J Definition

```text
J = Q + 0.5*cos(h) + 0.25*dense_agree - 2*KL - 0.1*hidden_l1
```

Where:

```text
Q           = route accuracy
cos(h)      = dense/sparse hidden cosine
KL          = dense-to-sparse logit KL
dense_agree = dense/sparse prediction agreement
hidden_l1   = dense/sparse hidden L1
```

## v11 Priority Order

```text
1. Prove learned K=8 speed vs hand K=16.
2. Add learned-topology benchmark support if missing.
3. Export failure diagnostics, especially generic_expert misses.
4. Add runtime-J early stopping / anti-drift guard.
5. Implement runtime-J-aligned dense-equivalence training.
6. Sweep learned K=6 and K=4 only after K=8 speed proof passes.
```

Do not prioritize more blind low-LR topology fine-tuning. v10 showed that it only gives tiny improvements and tends to drift away from runtime-J.

---

# Implementation Step 1 — Final Selected Quality Check

Run:

```bash
SCORER=runs/checkpoints/scorer_runtime_j_best.pt \
scripts/run_learned_topology_quality.sh
```

Acceptance:

```text
mode=learned_topology k=8 route_acc >= 0.9866
hidden_cos >= 0.978373 approximately
hidden_l1 <= 0.162904 approximately
logit_kl <= 0.013007 approximately
```

Expected:

```text
learned K=8 remains above hand K=16 route_acc
```

---

# Implementation Step 2 — Benchmark Learned K=8 vs Hand K=16

## Goal

Measure prepared sparse block runtime for:

```text
hand topology K=16
learned topology K=8
```

Minimum report:

```text
hand_k16_block_ms
learned_k8_block_ms
speedup = hand_k16_block_ms / learned_k8_block_ms
hand_k16_route_acc
learned_k8_route_acc
learned_hidden_cos
learned_logit_kl
learned_hidden_l1
```

Acceptance:

```text
learned_k8_route_acc >= hand_k16_route_acc
learned_k8_block_ms < hand_k16_block_ms
```

Stretch:

```text
speedup >= 1.25x
```

## If benchmark path already supports learned topology

Run existing benchmark with:

```bash
SCORER=runs/checkpoints/scorer_runtime_j_best.pt \
LEARNED_K=8 \
HAND_K=16 \
scripts/benchmark_learned_topology.sh
```

## If benchmark path does not support learned topology

Add:

```text
scripts/benchmark_learned_topology.sh
```

Responsibilities:

```text
1. Resolve python executable like other scripts.
2. Run quality evaluation for hand K=16 and learned K=8.
3. Run prepared sparse block benchmark for hand K=16.
4. Run prepared sparse block benchmark for learned K=8.
5. Print one compact comparison table.
6. Exit nonzero if acceptance gates fail unless ALLOW_FAIL=1.
```

Suggested environment variables:

```bash
SCORER="${SCORER:-runs/checkpoints/scorer_runtime_j_best.pt}"
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/synthetic_hard_dense.pt}"
EXAMPLES="${EXAMPLES:-data/synthetic_hard/val.jsonl}"
HAND_K="${HAND_K:-16}"
LEARNED_K="${LEARNED_K:-8}"
DEVICE="${DEVICE:-auto}"
BENCH_STEPS="${BENCH_STEPS:-100}"
ALLOW_FAIL="${ALLOW_FAIL:-0}"
```

---

# Implementation Step 3 — Add Learned Topology to Prepared Block Benchmark

If the benchmark cannot currently instantiate learned topology inside the prepared sparse runtime, patch it.

Required CLI support:

```text
--topology-mode learned_topology
--learned-scorer-checkpoint PATH
--learned-k 8
--profile-prepared-block
```

The learned benchmark path should use:

```text
src.learned_topology_runtime
src.learned_topology
```

and the selected scorer:

```text
runs/checkpoints/scorer_runtime_j_best.pt
```

Acceptance:

```bash
SCORER=runs/checkpoints/scorer_runtime_j_best.pt \
LEARNED_K=8 \
HAND_K=16 \
scripts/benchmark_learned_topology.sh
```

prints:

```text
mode                 k    route_acc    block_ms    hidden_cos    logit_kl
hand_topology        16   ...          ...         ...           ...
learned_topology     8    ...          ...         ...           ...
```

---

# Implementation Step 4 — Failure Diagnostics

## Motivation

Most experts are solved. The remaining route errors are concentrated in generic_expert.

Observed learned K=8:

```text
generic_expert = 235/259 = 0.9073
other experts mostly 1.0000
```

## Add script

```text
scripts/export_learned_topology_failures.sh
```

Default output:

```text
runs/diagnostics/learned_topology_failures.jsonl
```

Required fields per failure:

```text
example_id
expression
true_expert
dense_pred
learned_pred
hand_pred
is_generic_expert
learned_top_edges
hand_top_edges
missing_edges
extra_edges
hidden_l1
hidden_cos
logit_l1
logit_kl
```

Acceptance:

```bash
SCORER=runs/checkpoints/scorer_runtime_j_best.pt \
scripts/export_learned_topology_failures.sh
```

prints:

```text
wrote runs/diagnostics/learned_topology_failures.jsonl
failures_total=...
generic_expert_failures=...
```

and the JSONL file is nonempty if learned K=8 has route misses.

---

# Implementation Step 5 — Runtime-J Early Stopping / Anti-Drift Guard

## Motivation

v10 observed:

```text
runtime-J often peaks at step 0 or step 100, then degrades
```

Topology validation recall can improve while runtime-J worsens.

Add trainer options:

```text
--runtime-quality-patience INT
--runtime-quality-min-delta FLOAT
--runtime-quality-stop-on-degrade
```

Wrapper environment variables:

```bash
RUNTIME_QUALITY_PATIENCE="${RUNTIME_QUALITY_PATIENCE:-0}"
RUNTIME_QUALITY_MIN_DELTA="${RUNTIME_QUALITY_MIN_DELTA:-1e-5}"
RUNTIME_QUALITY_STOP_ON_DEGRADE="${RUNTIME_QUALITY_STOP_ON_DEGRADE:-0}"
```

Behavior:

```text
if runtime quality evaluation is enabled:
  update runtime-best checkpoint when J improves by min_delta
  increment stale counter otherwise
  if stop_on_degrade and stale counter >= patience:
    stop training early
```

Acceptance:

```bash
MAX_STEPS=1000 \
MAX_EXAMPLES=2000 \
LR=1e-5 \
RUNTIME_QUALITY_INTERVAL=100 \
RUNTIME_QUALITY_PATIENCE=2 \
RUNTIME_QUALITY_STOP_ON_DEGRADE=1 \
RUNTIME_QUALITY_BEST_CHECKPOINT=/tmp/runtime_best.pt \
scripts/train_topology_scorer.sh
```

Expected:

```text
training stops before step 999 when runtime-J keeps degrading
runtime-best checkpoint remains the best observed runtime-J point
```

---

# Implementation Step 6 — Runtime-J-Aligned Dense-Equivalence Training

## Motivation

v10 result:

```text
low-LR topology fine-tuning gave only tiny gains
```

The current objective optimizes topology targets, not dense-equivalence directly.

Target dense-equivalence:

```text
hidden_cos >= 0.985
logit_kl <= 0.005
hidden_l1 <= 0.120
route_acc >= 0.9866
```

## Add differentiable dense-equivalence loss

Proposed training objective:

```text
L = L_edge
  + lambda_kl * KL(logits_dense || logits_sparse)
  + lambda_cos * (1 - cos(hidden_dense, hidden_sparse))
  + lambda_l1 * L1(hidden_dense, hidden_sparse)
```

Initial weights:

```text
lambda_kl  = 0.50
lambda_cos = 0.25
lambda_l1  = 0.10
```

Add trainer args:

```text
--runtime-kl-loss FLOAT
--runtime-cos-loss FLOAT
--runtime-hidden-l1-loss FLOAT
```

Wrapper env vars:

```bash
RUNTIME_KL_LOSS="${RUNTIME_KL_LOSS:-0.0}"
RUNTIME_COS_LOSS="${RUNTIME_COS_LOSS:-0.0}"
RUNTIME_HIDDEN_L1_LOSS="${RUNTIME_HIDDEN_L1_LOSS:-0.0}"
```

## First experiment

```bash
MAX_STEPS=1000 \
MAX_EXAMPLES=2000 \
LR=1e-5 \
EVAL_INTERVAL=100 \
EVAL_MAX_EXAMPLES=512 \
DEVICE=auto \
DENSE_CHECKPOINT=runs/checkpoints/synthetic_hard_dense.pt \
DENSE_MIX=0.25 \
RUNTIME_KL_LOSS=0.50 \
RUNTIME_COS_LOSS=0.25 \
RUNTIME_HIDDEN_L1_LOSS=0.10 \
RESUME_SCORER_CHECKPOINT=runs/checkpoints/scorer_runtime_j_best.pt \
CHECKPOINT=runs/checkpoints/scorer_runtime_aligned.pt \
BEST_CHECKPOINT=runs/checkpoints/scorer_runtime_aligned.topology_best.pt \
RUNTIME_QUALITY_EXAMPLES=data/synthetic_hard/val.jsonl \
RUNTIME_QUALITY_CHECKPOINT=runs/checkpoints/synthetic_hard_dense.pt \
RUNTIME_QUALITY_INTERVAL=100 \
RUNTIME_QUALITY_MAX_EXAMPLES=1024 \
RUNTIME_QUALITY_BEST_CHECKPOINT=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
RUNTIME_QUALITY_PATIENCE=2 \
RUNTIME_QUALITY_STOP_ON_DEGRADE=1 \
scripts/train_topology_scorer.sh
```

Acceptance:

```text
full-validation runtime-J improves over 1.6820821
route_acc >= 0.9866
hidden_cos > 0.978373
hidden_l1 < 0.162904
logit_kl <= 0.013007
```

Stretch:

```text
hidden_cos >= 0.985
logit_kl <= 0.005
hidden_l1 <= 0.120
```

---

# Implementation Step 7 — Learned K Sweep After Speed Proof

Only run after learned K=8 speed/quality proof passes.

Candidates:

```text
learned K=6
learned K=4
```

Commands:

```bash
SCORER=runs/checkpoints/scorer_runtime_j_best.pt \
LEARNED_K=6 \
scripts/run_learned_topology_quality.sh

SCORER=runs/checkpoints/scorer_runtime_j_best.pt \
LEARNED_K=4 \
scripts/run_learned_topology_quality.sh
```

Acceptance for K=6:

```text
route_acc >= hand K=16 route_acc
block_ms < learned K=8 block_ms
```

Acceptance for K=4:

```text
route_acc >= dense full route_acc - 0.002
block_ms < learned K=8 block_ms
```

---

# v11 Acceptance Gates

## Gate 1 — Selected checkpoint quality

```text
SCORER=runs/checkpoints/scorer_runtime_j_best.pt scripts/run_learned_topology_quality.sh
```

passes:

```text
learned K=8 route_acc >= hand K=16 route_acc
```

## Gate 2 — Speed proof

```text
learned K=8 block_ms < hand K=16 block_ms
```

## Gate 3 — Benchmark report

A reproducible benchmark report exists with:

```text
hand K=16 block_ms
learned K=8 block_ms
speedup
route_acc values
hidden/logit equivalence values
```

## Gate 4 — Failure diagnostics

```text
runs/diagnostics/learned_topology_failures.jsonl
```

exists and identifies expert-level failure concentration.

## Gate 5 — Anti-drift guard

Runtime-J early stopping prevents long runs that degrade after the first few runtime evaluations.

## Gate 6 — Runtime-aligned training

At least one runtime-aligned training run improves full-validation runtime-J over:

```text
1.6820821
```

---

# Risks and Mitigations

## Risk 1 — Benchmark noise hides speedup

Mitigation:

```text
warmup iterations
multiple repetitions
median and p95 latency
pin DEVICE
report batch/sequence sizes
```

## Risk 2 — Learned topology preparation overhead cancels K reduction

Mitigation:

```text
benchmark prepared topology separately from scorer inference
cache learned topology when possible
report scorer_ms, prepare_ms, block_ms independently
```

## Risk 3 — Runtime-aligned dense losses are expensive

Mitigation:

```text
start with subset runtime training
cache dense teacher traces if needed
run early stopping aggressively
```

## Risk 4 — KL improves but route accuracy drops

Mitigation:

```text
keep route accuracy as hard acceptance gate
rank by runtime-J
save runtime-best only when J improves
```

---

# Canonical v11 Session Order

```text
1. Run final quality check for scorer_runtime_j_best.pt.
2. Inspect existing benchmark support for learned topology.
3. Add scripts/benchmark_learned_topology.sh if missing.
4. Patch prepared block benchmark to load learned topology if needed.
5. Run learned K=8 vs hand K=16 speed/quality benchmark.
6. Add failure diagnostics export script.
7. Add runtime-J early stopping options.
8. Add runtime-aligned dense-equivalence loss options.
9. Run first runtime-aligned training experiment.
10. Only after K=8 speed proof passes, sweep learned K=6 and K=4.
```

---

# Definition of Done

v11 succeeds if:

```text
learned K=8 is both more accurate and faster than hand K=16
```

and the result is reproducible with:

```text
runs/checkpoints/scorer_runtime_j_best.pt
scripts/benchmark_learned_topology.sh
runs/checkpoints/scorer_runtime_j_selection.csv
```

Stretch success:

```text
runtime-aligned training improves dense-equivalence beyond v10 best
hidden_cos >= 0.985
logit_kl <= 0.005
```
