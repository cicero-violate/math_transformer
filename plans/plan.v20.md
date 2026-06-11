# Plan v20 — Learned K=4 Is a Fragile Quality Leader; Mechanism and Runtime Gates Next

## Thesis

```text
The corrected fixed-K sweep changes the headline but not the theory.

Learned topology K=4 is the current quality leader by +1 sample over hand K=4.
However, the dominant effect is still sparsity/pruning denoising, not learned edge invention.
```

The next cycle must not promote from aggregate accuracy alone. It must prove:

```text
1. Mechanism: the 5 learned K=4 wins and 4 losses have interpretable edge-level causes.
2. Stability: the +1 sample learned advantage survives reruns / stress validation / replay attempts.
3. Runtime: any learned policy must pass repeated locked speed distribution gates, not single-run strict speed.
```

## Corrected Empirical Baseline

Validation set:

```text
data/synthetic_hard/val.jsonl
n = 1786
```

Artifact:

```text
runs/diagnostics/topology_k_sweep_summary.json
runs/diagnostics/topology_k_sweep_summary.csv
```

Important implementation correction:

```text
Hand sweep rows must run with fixed_k = k.
Do not batch hand K values under one shared fixed_k=max(K).
```

## Quality Table

| Topology                    |    K | route_acc | generic_expert     | affine_expert      | Correct   | Δ vs dense | Δ vs hand K=4 |
| ---                         | ---: |      ---: | ---:               | ---:               | ---:      |       ---: |          ---: |
| Dense full                  | full |  0.980963 | 225/259 = 0.868726 | 296/296 = 1.000000 | 1752/1786 |          0 |           -21 |
| Hand middle_preserving_topk |    2 |  0.896976 | 259/259 = 1.000000 | 112/296 = 0.378378 | 1602/1786 |       -150 |          -171 |
| Hand middle_preserving_topk |    3 |  0.978163 | 259/259 = 1.000000 | 257/296 = 0.868243 | 1747/1786 |         -5 |           -26 |
| Hand middle_preserving_topk |    4 |  0.992721 | 252/259 = 0.972973 | 290/296 = 0.979730 | 1773/1786 |        +21 |             0 |
| Hand middle_preserving_topk |    5 |  0.992721 | 247/259 = 0.953668 | 295/296 = 0.996622 | 1773/1786 |        +21 |             0 |
| Hand middle_preserving_topk |    6 |  0.989362 | 241/259 = 0.930502 | 295/296 = 0.996622 | 1767/1786 |        +15 |            -6 |
| Hand middle_preserving_topk |    8 |  0.988802 | 239/259 = 0.922780 | 296/296 = 1.000000 | 1766/1786 |        +14 |            -7 |
| Hand middle_preserving_topk |   12 |  0.987122 | 236/259 = 0.911197 | 296/296 = 1.000000 | 1763/1786 |        +11 |           -10 |
| Hand middle_preserving_topk |   16 |  0.983763 | 230/259 = 0.888031 | 296/296 = 1.000000 | 1757/1786 |         +5 |           -16 |
| Learned topology            |    2 |  0.898096 | 259/259 = 1.000000 | 114/296 = 0.385135 | 1604/1786 |       -148 |          -169 |
| Learned topology            |    3 |  0.977044 | 259/259 = 1.000000 | 255/296 = 0.861486 | 1745/1786 |         -7 |           -28 |
| Learned topology            |    4 |  0.993281 | 253/259 = 0.976834 | 290/296 = 0.979730 | 1774/1786 |        +22 |            +1 |
| Learned topology            |    5 |  0.992161 | 247/259 = 0.953668 | 294/296 = 0.993243 | 1772/1786 |        +20 |            -1 |
| Learned topology            |    6 |  0.989362 | 242/259 = 0.934363 | 294/296 = 0.993243 | 1767/1786 |        +15 |            -6 |
| Learned topology            |    8 |  0.989362 | 240/259 = 0.926641 | 296/296 = 1.000000 | 1767/1786 |        +15 |            -6 |
| Learned topology            |   12 |  0.984323 | 231/259 = 0.891892 | 296/296 = 1.000000 | 1758/1786 |         +6 |           -15 |
| Learned topology            |   16 |  0.981523 | 226/259 = 0.872587 | 296/296 = 1.000000 | 1753/1786 |         +1 |           -20 |

## Corrected Interpretation

```text
Pruning effect:
hand K=4 - dense = 1773 - 1752 = +21 samples

Learned edge-selection effect at best K:
learned K=4 - hand K=4 = 1774 - 1773 = +1 sample

Therefore:
pruning effect ≫ learned selection effect
21 ≫ 1
```

Allowed claim:

```text
Learned K=4 is the current quality leader and beats hand K=4 by one net validation sample.
```

Disallowed claim:

```text
Learned topology strongly dominates hand topology.
```

## Edge-Level Evidence

Artifact:

```text
runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl
```

Command:

```bash
python -m src.export_topology_edge_deltas \
  --examples data/synthetic_hard/val.jsonl \
  --checkpoint runs/checkpoints/synthetic_hard_dense.pt \
  --learned-scorer-checkpoint runs/checkpoints/topology_scorer.champion.pt \
  --k 4 \
  --quality-device auto \
  --output runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl
```

Observed result:

```text
records = 9
learned_win = 5
learned_loss = 4
net = +1
```

First observed learned win:

```text
sample_id = 60
target_expert = affine_expert
hand_pred = matmul_expert
learned_pred = affine_expert
hand_correct = false
learned_correct = true
hand_edges = 60
learned_edges = 60
removed_edges = 6
extra_edges = 6
```

## Current Risks

| Risk                                        | Current evidence                                                                                    | Required response                                                                                      |
| ---:                                        | ---                                                                                                 | ---                                                                                                    |
| Learned advantage is fragile                | Learned K=4 beats hand K=4 by only 1 sample; paired flips are 5 wins and 4 losses.                  | Treat learned K=4 as candidate, not settled champion, until mechanism and speed gates pass.            |
| Pruning still dominates                     | Hand K=4 gives +21 vs dense; learned adds only +1 over hand.                                        | Keep pruning-first framing in all reports.                                                             |
| Affine/generic tradeoff remains real        | K=2/K=3 max generic but damage affine heavily; K=4 improves generic while losing 6 affine vs dense. | Report generic and affine Pareto frontier, not route_acc alone.                                        |
| Edge mechanism is not yet summarized        | Edge-delta JSONL exists but has not been aggregated by relation/node pattern.                       | Build an edge-delta analyzer before explaining wins/losses.                                            |
| Runtime proof is still incomplete           | Single-run strict speed has been unstable.                                                          | Add repeated locked speed aggregator and gate on distribution.                                         |
| Promotion criteria not updated for hand K=4 | Historical promotion path uses champion-style quality gates, not hand K=4 Pareto baseline.          | Promotion/reporting must include dense, hand K=4, learned K=4, generic/affine, and speed distribution. |

## P0 Work Items

| Priority | Item                             | Acceptance                                                                                                                                                                                                |
| ---:     | ---                              | ---                                                                                                                                                                                                       |
| P0.1     | Edge-delta aggregate analyzer    | Reads `learned_k4_vs_hand_k4_edge_deltas.jsonl`; emits JSON/CSV summary by outcome, target expert, prediction flip, removed/extra edge count, edge score distribution, and recurring node-label patterns. |
| P0.2     | Repeated locked speed aggregator | Runs N locked benchmarks and emits median/p25/p75/pass-rate for dense, hand K=4, learned K=4, and current champion K=8. Speed gate uses distribution, not single run.                                     |
| P0.3     | Promotion report update          | Every promotion artifact includes dense, hand K=4, learned K=4, current champion policy, route/generic/affine, paired wins/losses, and repeated speed summary.                                            |
| P0.4     | K-sweep regression guard         | Tests ensure hand rows use `fixed_k=k`, learned rows use the requested learned K, and wins/losses are paired sample flips.                                                                                |
| P0.5     | Edge-delta compact view command  | Add a command that prints the 9 learned K=4 paired flips as a concise table: sample_id, outcome, target, dense_pred, hand_pred, learned_pred, removed_count, extra_count.                                 |

## P1 Work Items

| Priority | Item                                | Acceptance                                                                                                                                        |
| ---:     | ---                                 | ---                                                                                                                                               |
| P1.1     | Replay on learned K=4 losses        | Use the 4 learned losses as replay candidates without erasing the 5 wins; compare against hand K=4 and learned K=4 baseline.                      |
| P1.2     | Dynamic-K prototype                 | Per-expression or per-node K preserves K=4 generic gains while recovering affine regressions. Must beat or Pareto-match learned K=4 and hand K=4. |
| P1.3     | Generic-stress validation expansion | Add held-out composite expressions that separate affine-looking local subgraphs from truly generic global structure.                              |
| P1.4     | Timing-regime split                 | Artifacts separately report cold prepare, cache-hit prepare, static preloaded block, cache-hit block, and growing-node prepare.                   |
| P1.5     | Large-N static benchmark            | Show stable static/preloaded speed advantage at N >= 2048, then N >= 4096, with cold/dynamic cost reported separately.                            |

## Commands

### Corrected K sweep

```bash
python -m src.topology_k_sweep \
  --examples data/synthetic_hard/val.jsonl \
  --checkpoint runs/checkpoints/synthetic_hard_dense.pt \
  --k-values 2,3,4,5,6,8,12,16 \
  --learned-scorer-checkpoint runs/checkpoints/topology_scorer.champion.pt \
  --quality-device auto \
  --json-out runs/diagnostics/topology_k_sweep_summary.json \
  --csv-out runs/diagnostics/topology_k_sweep_summary.csv
```

### Learned K=4 vs hand K=4 edge deltas

```bash
python -m src.export_topology_edge_deltas \
  --examples data/synthetic_hard/val.jsonl \
  --checkpoint runs/checkpoints/synthetic_hard_dense.pt \
  --learned-scorer-checkpoint runs/checkpoints/topology_scorer.champion.pt \
  --k 4 \
  --quality-device auto \
  --output runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl
```

### Quick artifact sanity checks

```bash
python - <<'PY'
import json
from collections import Counter
rows = json.load(open('runs/diagnostics/topology_k_sweep_summary.json'))
print('rows', len(rows))
for r in rows:
    if (r['mode'], r['k']) in {('dense', 'full'), ('hand', 4), ('learned', 4)}:
        print(r['mode'], r['k'], r['correct_count'], r['route_acc'], r.get('generic_expert_correct'), r.get('affine_expert_correct'))
edge = [json.loads(line) for line in open('runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl')]
print('edge_records', len(edge), dict(Counter(r['outcome'] for r in edge)))
PY
```

## Decision Rules

Do not promote learned K=4 solely because it is +1 over hand K=4.

A topology policy can be considered promotion-ready only if it satisfies at least one of:

```text
1. Quality dominance:
   route_acc > hand K=4 and no worse affine/generic Pareto tradeoff on expanded validation.

2. Pareto improvement:
   same quality as hand K=4 or learned K=4 but materially faster or lower memory.

3. Stability improvement:
   lower variance speed/runtime at same quality under repeated locked benchmark distribution.

4. Dynamic-regime improvement:
   much lower cold/growing-graph prepare cost at acceptable quality.
```

## Non-Negotiable Gates

```text
1. Champion can only change through the promotion gate.
2. Hand K=4 remains a required quality baseline.
3. Learned K=4 is a quality candidate, not a mechanism proof.
4. Every topology quality report must include generic and affine accuracy.
5. Wins/losses must mean paired per-sample flips, not net correct-count deltas.
6. Single-run strict speed is diagnostic only; speed acceptance requires repeated locked aggregation.
7. Claims about learned topology must separate K/pruning effect from learned edge-selection effect.
8. Edge-level traces are required before explaining why a win happened.
9. Large-N scaling claims require static/preloaded and cold/dynamic regimes to be reported separately.
```
