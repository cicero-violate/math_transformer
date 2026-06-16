---
schema_version: v26_cycle_result.v1
date: 2026-06-16
branch: v26-rewiring
status: v26_negative
---

# v26 Cycle Results — qwen_topk_k2 Baseline

## Decision

```
candidate_not_promoted
```

Both cycles ran and were rejected. The qwen_topk_k2 baseline topology is stable under the KL surrogate metric.

---

## Cycle 1 — same_source_top_weight, max_swaps=4

```
artifact:              runs/v26_rewire/qwen_style_tiny/cycle_001/v26_cycle_report.json
proposal_policy:       same_source_top_weight
swap_count:            4
kl_baseline_after:     0.5492
kl_candidate_after:    0.6039  ← worse than baseline
kl_random_after:       0.5978
candidate_beats_baseline: false
candidate_beats_random:   false
quality_ok:            false
kl_ok:                 false
heldout_ok:            true
promote:               false
```

## Cycle 2 — deterministic_random, max_swaps=2, seed=7

```
artifact:              runs/v26_rewire/qwen_style_tiny/cycle_002/v26_cycle_report.json
proposal_policy:       deterministic_random
swap_count:            2
kl_baseline_after:     0.5492
kl_candidate_after:    0.5862  ← worse than baseline
kl_random_after:       0.5978
candidate_beats_baseline: false
candidate_beats_random:   true  (narrowly: 0.586 < 0.598)
quality_ok:            false
kl_ok:                 false
heldout_ok:            true
promote:               false
```

---

## What the result proves

```
qwen_topk_k2 is stable:
  no weight-graph edge swap improves KL under the surrogate test.
```

The Frobenius-norm top-k selection is near-optimal for this metric and model size.

## What it does not prove

```
qwen_topk_k2 is globally optimal
v26 rewiring can never help at larger k or different proposal policies
text-path accuracy cannot be improved by rewiring
```

## Baseline remains

```
qwen_topk_k2 retains its status as the v26 baseline.
No candidate was promoted.
allow_rewiring remains false.
```

## Next step options

```
1. Run more cycles with different seeds / policies (still exploratory)
2. Expand to k=3 adjacency as next candidate (bigger budget)
3. Accept v26_negative as the correct result at this model size
4. Move to downstream task eval to gather real gradient signal for rewiring
```
