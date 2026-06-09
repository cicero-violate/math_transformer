# Plan v10 — Runtime-J Fine-Tuning and Dense-Equivalence Push

## Current State

We have crossed the important learned-topology routing gate:

```text
learned_topology K=8 route_acc > hand_topology K=16 route_acc
```

Current best learned runtime checkpoint:

```text
runs/checkpoints/scorer_dense_mix_0p25_finetune.runtime_best.pt
```

Full validation quality:

```text
dense full route_acc                  = 0.9810
hand topology K=16 route_acc          = 0.9821
learned topology K=8 route_acc        = 0.9866

hand K=16 hidden_cos                  = 0.992537
learned K=8 hidden_cos                = 0.978368

hand K=16 logit_kl                    = 0.002046
learned K=8 logit_kl                  = 0.013007

hand K=16 hidden_l1                   = 0.079942
learned K=8 hidden_l1                 = 0.162935
```

Interpretation:

```text
route-quality gate: passed
dense-equivalence gate: not passed
runtime learned-topology integration: working
resume fine-tuning: working
runtime-J checkpointing: working
```

## Main v10 Objective

Improve dense-equivalence while preserving the learned routing win.

Target:

```text
Q_learned_K8 >= 0.9866
hidden_cos_learned_K8 >= 0.985
logit_kl_learned_K8 <= 0.005
hidden_l1_learned_K8 <= 0.120
```

Hard replacement target against hand K=16:

```text
Q_learned_K8 >= Q_hand_K16
hidden_cos_learned_K8 close to hidden_cos_hand_K16
logit_kl_learned_K8 close to logit_kl_hand_K16
```

## v10 Hypothesis

The last fine-tune used the default learning rate:

```text
lr = 1e-3
```

Runtime-J peaked immediately and then degraded, which implies:

```text
1e-3 oversteps the local optimum
```

v10 should use smaller learning rates:

```text
1e-4
3e-5
1e-5
```

Expected behavior:

```text
small LR preserves routing
small LR gradually improves hidden/logit agreement
runtime-J checkpointing captures best point before drift
```

## Implementation Step 1 — Expose LR in Wrapper

The trainer already supports:

```text
--lr
```

Patch:

```text
scripts/train_topology_scorer.sh
```

Add:

```bash
LR="${LR:-1e-3}"
```

and pass:

```bash
--lr "$LR"
```

Acceptance:

```bash
LR=1e-4 MAX_STEPS=1 MAX_EXAMPLES=4 scripts/train_topology_scorer.sh
```

should invoke training with no CLI error.

## Implementation Step 2 — Low-LR Resume Fine-Tune Runs

Run from current runtime-best:

```text
runs/checkpoints/scorer_dense_mix_0p25_finetune.runtime_best.pt
```

### Run A: lr=1e-4

```bash
MAX_STEPS=1000 \
MAX_EXAMPLES=2000 \
LR=1e-4 \
EVAL_INTERVAL=100 \
EVAL_MAX_EXAMPLES=512 \
DEVICE=auto \
DENSE_CHECKPOINT=runs/checkpoints/synthetic_hard_dense.pt \
DENSE_MIX=0.25 \
RESUME_SCORER_CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune.runtime_best.pt \
CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e4.pt \
BEST_CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e4.topology_best.pt \
RUNTIME_QUALITY_EXAMPLES=data/synthetic_hard/val.jsonl \
RUNTIME_QUALITY_CHECKPOINT=runs/checkpoints/synthetic_hard_dense.pt \
RUNTIME_QUALITY_INTERVAL=100 \
RUNTIME_QUALITY_MAX_EXAMPLES=1024 \
RUNTIME_QUALITY_BEST_CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e4.runtime_best.pt \
scripts/train_topology_scorer.sh
```

### Run B: lr=3e-5

```bash
MAX_STEPS=1000 \
MAX_EXAMPLES=2000 \
LR=3e-5 \
EVAL_INTERVAL=100 \
EVAL_MAX_EXAMPLES=512 \
DEVICE=auto \
DENSE_CHECKPOINT=runs/checkpoints/synthetic_hard_dense.pt \
DENSE_MIX=0.25 \
RESUME_SCORER_CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune.runtime_best.pt \
CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr3e5.pt \
BEST_CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr3e5.topology_best.pt \
RUNTIME_QUALITY_EXAMPLES=data/synthetic_hard/val.jsonl \
RUNTIME_QUALITY_CHECKPOINT=runs/checkpoints/synthetic_hard_dense.pt \
RUNTIME_QUALITY_INTERVAL=100 \
RUNTIME_QUALITY_MAX_EXAMPLES=1024 \
RUNTIME_QUALITY_BEST_CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr3e5.runtime_best.pt \
scripts/train_topology_scorer.sh
```

### Run C: lr=1e-5

```bash
MAX_STEPS=1000 \
MAX_EXAMPLES=2000 \
LR=1e-5 \
EVAL_INTERVAL=100 \
EVAL_MAX_EXAMPLES=512 \
DEVICE=auto \
DENSE_CHECKPOINT=runs/checkpoints/synthetic_hard_dense.pt \
DENSE_MIX=0.25 \
RESUME_SCORER_CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune.runtime_best.pt \
CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e5.pt \
BEST_CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e5.topology_best.pt \
RUNTIME_QUALITY_EXAMPLES=data/synthetic_hard/val.jsonl \
RUNTIME_QUALITY_CHECKPOINT=runs/checkpoints/synthetic_hard_dense.pt \
RUNTIME_QUALITY_INTERVAL=100 \
RUNTIME_QUALITY_MAX_EXAMPLES=1024 \
RUNTIME_QUALITY_BEST_CHECKPOINT=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e5.runtime_best.pt \
scripts/train_topology_scorer.sh
```

## Implementation Step 3 — Full Validation Evaluation

Evaluate all runtime-best candidates:

```bash
SCORER=runs/checkpoints/scorer_dense_mix_0p25_finetune.runtime_best.pt \
scripts/run_learned_topology_quality.sh

SCORER=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e4.runtime_best.pt \
scripts/run_learned_topology_quality.sh

SCORER=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr3e5.runtime_best.pt \
scripts/run_learned_topology_quality.sh

SCORER=runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e5.runtime_best.pt \
scripts/run_learned_topology_quality.sh
```

Rank by runtime-J:

```text
J = Q + 0.5*cos(h) + 0.25*dense_agree - 2*KL - 0.1*hidden_l1
```

Best checkpoint should be copied to:

```text
runs/checkpoints/scorer_runtime_j_best.pt
```

## Implementation Step 4 — Add Candidate Selection Script

Add:

```text
scripts/select_best_runtime_scorer.sh
```

Responsibilities:

```text
accept list of scorer checkpoints
run scripts/run_learned_topology_quality.sh for each
score each output with src.dense_mix_sweep
write summary CSV
copy best checkpoint to chosen output path
```

Default output:

```text
runs/checkpoints/scorer_runtime_j_best.pt
```

Example:

```bash
scripts/select_best_runtime_scorer.sh \
  runs/checkpoints/scorer_dense_mix_0p25_finetune.runtime_best.pt \
  runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e4.runtime_best.pt \
  runs/checkpoints/scorer_dense_mix_0p25_finetune_lr3e5.runtime_best.pt \
  runs/checkpoints/scorer_dense_mix_0p25_finetune_lr1e5.runtime_best.pt
```

## Implementation Step 5 — Benchmark Learned K=8 Runtime

Once best runtime-J checkpoint is selected:

```text
runs/checkpoints/scorer_runtime_j_best.pt
```

Run learned quality:

```bash
SCORER=runs/checkpoints/scorer_runtime_j_best.pt \
scripts/run_learned_topology_quality.sh
```

Then benchmark speed against hand K=16.

Benchmark target:

```text
learned K=8 prepared sparse block faster than hand K=16
learned K=8 route_acc >= hand K=16 route_acc
```

If existing benchmark path cannot load learned topology, add:

```text
scripts/benchmark_learned_topology.sh
```

Minimum report:

```text
hand K=16 block_ms
learned K=8 block_ms
hand K=16 route_acc
learned K=8 route_acc
hidden_cos
logit_kl
```

## v10 Acceptance Gates

### Gate 1 — LR wrapper

```text
LR env var works
no trainer CLI error
```

### Gate 2 — resume fine-tune stability

At least one low-LR run improves runtime-J over:

```text
runs/checkpoints/scorer_dense_mix_0p25_finetune.runtime_best.pt
```

### Gate 3 — dense-equivalence improvement

Target minimum:

```text
hidden_cos > 0.978368
logit_kl < 0.013007
hidden_l1 < 0.162935
route_acc >= 0.9866
```

### Gate 4 — stretch dense-equivalence target

```text
hidden_cos >= 0.985
logit_kl <= 0.005
hidden_l1 <= 0.120
```

### Gate 5 — speed/quality proof

```text
learned K=8 route_acc >= hand K=16 route_acc
learned K=8 faster than hand K=16
```

## Risks

### Risk 1 — topology objective fights runtime-J

Training loss still optimizes local edge targets, while runtime-J is non-differentiable and only used for selection.

Mitigation:

```text
small LR
frequent runtime-J checkpointing
stop when runtime-J degrades
```

### Risk 2 — dense-QK teacher improves hidden geometry but hurts KL

Observed:

```text
hidden_cos improved
hidden_l1 improved
KL not always improved
```

Mitigation:

```text
rank by runtime-J
try smaller dense_mix if KL worsens
```

### Risk 3 — runtime-J subset overfits

Current runtime eval can use subset size:

```text
RUNTIME_QUALITY_MAX_EXAMPLES=1024
```

Mitigation:

```text
always final-evaluate on full val.jsonl
```

## Canonical Next Session Order

```text
1. Patch LR env in scripts/train_topology_scorer.sh.
2. Add scripts/select_best_runtime_scorer.sh.
3. Run lr=1e-4 resume fine-tune.
4. Run lr=3e-5 resume fine-tune.
5. Run lr=1e-5 resume fine-tune.
6. Select best by full-val runtime-J.
7. Evaluate scorer_runtime_j_best.pt.
8. Benchmark learned K=8 vs hand K=16.
```

## Current Best Baseline to Beat

```text
checkpoint = runs/checkpoints/scorer_dense_mix_0p25_finetune.runtime_best.pt
route_acc  = 0.9866
dense_agree = 0.9944
hidden_l1 = 0.162935
hidden_cos = 0.978368
logit_l1 = 0.100753
logit_kl = 0.013007
```

## Definition of Done

v10 is successful if:

```text
learned K=8 remains above hand K=16 route_acc
and dense-equivalence improves over the current runtime-best baseline
and the best checkpoint is selected by full validation runtime-J
```

Stretch success:

```text
hidden_cos >= 0.985
logit_kl <= 0.005
```
