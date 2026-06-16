---
schema_version: v25_01_surrogate_proof.v1
date: 2026-06-16
status: complete
---

# v25.01 Fixed-Topology KL Surrogate Proof — k=2

## Verified artifact

```
runs/k2_kl_training/v2/k2_kl_comparison_report.json
```

```
kl_before_qwen:    0.5660
kl_before_random:  0.6134
kl_after_qwen:     0.5492
kl_after_random:   0.5978
delta_after:      -0.0486
qwen_wins_after:   true
train_steps:       20
k:                 2
status:            k2_kl_comparison_ok
```

## Correct claim

$$\boxed{\text{v25.01 fixed-topology KL surrogate proof is complete for k=2.}}$$

More precise:

```
Qwen-derived k=2 topology beats matched random k=2
before and after identical KL training.
```

That is real evidence.

## What it proves

```
A_qwen_k2 has useful structural signal
under the frozen-logit KL / feature-propagation test.
```

Training did not erase the advantage:

```
before delta: -0.0474
after delta:  -0.0486
```

The margin slightly improved.

## What it does not prove yet

```
real language generation is better
student is faster than Qwen
student is smarter than Qwen
graph transformer architecture is complete
rewiring improves topology
```

## Status

```
v25.01 text behavior smoke:        green
v25.01 k=2 topology sweep:         green enough
v25.01 k=2 KL training comparison: green
v25.01 fixed-topology surrogate proof: complete
```

## Next boundary

Do not keep expanding v25.01 further.

Next phase:

```
k=2 fixed topology -> real heldout text/task eval
then
v26 bounded rewiring proposal
```

## Promotion phrase

```
Promote qwen_topk_k2 as the current fixed sparse topology candidate
for downstream heldout text evaluation and v26 rewiring experiments.
```
