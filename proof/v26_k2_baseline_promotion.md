---
schema_version: v26_baseline_promotion.v1
date: 2026-06-16
status: promoted
---

# v26 Baseline Promotion — qwen_topk_k2

## Promotion decision

```
Promote qwen_topk_k2 as the current fixed sparse topology candidate
for downstream heldout text evaluation and v26 rewiring experiments.
```

**`qwen_topk_k2` is now the v26 baseline.**

---

## Evidence

### KL surrogate proof

```
artifact:          runs/k2_kl_training/v2/k2_kl_comparison_report.json
kl_before_qwen:    0.5660
kl_before_random:  0.6134
kl_after_qwen:     0.5492
kl_after_random:   0.5978
delta_after:      -0.0486
qwen_wins_after:   true
train_steps:       20
```

Qwen k=2 topology beats matched random k=2 before and after identical KL training.

### Heldout text eval

```
artifact:           runs/v25_01_heldout_eval/qwen_style_tiny/k2_fixed/heldout_eval_report.json
n_train:            96
n_heldout:          32
train_loss_initial: 5.278
train_loss_final:   4.769
heldout_loss_mean:  4.405
generalization_gap: -0.364
heldout_generalizes: true
```

Student does not overfit. Held-out loss below train loss across all four families.

---

## v26 Constraints

These are hard rules for the v26 rewiring cycle. No exceptions.

```
A_qwen_k2 is the baseline.
Rewiring proposals must beat A_qwen_k2.
Random baselines remain required.
No edge mutation may be promoted without quality + KL + heldout reports.
```

| Constraint | Value |
|---|---|
| Baseline topology | `qwen_topk_k2` |
| Rewiring must beat baseline | **true** |
| Random baselines required | **true** |
| Required promotion gates | quality_report, kl_report, heldout_report |
| No edge mutation without full reports | **true** |
| Rewiring branch | v26 (separate from v25.01) |
| Mix into v25.01 | **false** |

---

## Safety flags (remain set through v26 rewiring cycle)

```
teacher_checkpoint_loaded:    false
raw_weight_payload_in_graph:  false
bounded_active_adjacency:     true
allow_rewiring:               false  ← until v26 gates pass
promotion_eligible:           false  ← until v26 gates pass
```

`allow_rewiring` and `promotion_eligible` stay `false` until the v26 rewiring cycle completes its own quality + KL + heldout gate suite and beats this baseline.
