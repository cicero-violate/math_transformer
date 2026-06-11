# Plan v19 — Sparsity Denoising, K-Sweep, and Edge-Level Evidence

## Thesis

```text
The strongest current result is not “learned topology invents better edges.”
The strongest current result is “sparsity/pruning denoises composite math expressions and fixes dense affine bias.”
```

The next development cycle must separate three effects:

```text
1. K effect: fewer edges improve routing by pruning distracting structure.
2. Learned selection effect: learned K selects better edges than hand K at the same K.
3. Runtime effect: sparse/static execution must win wall-clock outside single-run noise.
```

## Current Empirical Baseline

Validation set:

```text
data/synthetic_hard/val.jsonl
n = 1786
```

Quality table:

| Topology | K | route_acc | generic_expert | affine_expert | Correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense full | full | 0.980963 | 225/259 = 0.868726 | 296/296 = 1.000000 | 1752/1786 |
| Hand middle_preserving_topk | 4 | 0.992721 | 252/259 = 0.972973 | 290/296 = 0.979730 | 1773/1786 |
| Hand middle_preserving_topk | 8 | 0.988802 | 239/259 = 0.922780 | 296/296 = 1.000000 | 1766/1786 |
| Hand middle_preserving_topk | 12 | 0.987122 | 236/259 = 0.911197 | 296/296 = 1.000000 | 1763/1786 |
| Hand middle_preserving_topk | 16 | 0.983763 | 230/259 = 0.888031 | 296/296 = 1.000000 | 1757/1786 |
| Learned topology champion | 8 | 0.989362 | 240/259 = 0.926641 | 296/296 = 1.000000 | 1767/1786 |

Dense-correction trace evidence:

```text
learned_beats_dense = 15
learned_loses_to_dense = 0
dense_learned_disagreements = 15
all observed wins are generic_expert corrections from dense affine_expert mistakes
```

Key falsification:

```text
hand K=4 > learned K=8 > hand K=8 > hand K=16 > dense
```

Therefore:

```text
Dominant effect = pruning / structural denoising
Secondary effect = learned edge selection at K=8 (+1 sample vs hand K=8)
Unproven effect = learned topology beats hand topology at the best K
```

## Current Risks

| Risk | Current evidence | Required response |
| ---: | --- | --- |
| Learned topology overclaimed | Learned K=8 beats hand K=8 by only 1 sample; hand K=4 beats learned K=8 by 6 samples. | Reframe as pruning-first until learned K sweep proves otherwise. |
| Affine regressions under aggressive pruning | Hand K=4 improves generic but loses 6 affine examples. | Track affine/generic Pareto frontier, not only route_acc. |
| Single-run speed instability | Locked runs have flipped strict_speed_ok true/false. | Add repeated-run aggregation and gate on distribution. |
| Cold prepare misunderstood | topology_prepare_ms is real cold miss, not static benchmark loop cost. | Split cold, cache-hit, static, and growing-graph regimes in artifacts. |
| Edge mechanism unknown | Current traces summarize overlap but not actual edge lists. | Export edge-level wins with node labels and removed/extra edges. |

## P0 Work Items

| Priority | Item | Acceptance |
| ---: | --- | --- |
| P0.1 | Complete hand and learned fixed-K sweep | One command evaluates dense, hand K={2,3,4,5,6,8,12,16}, learned K={2,3,4,5,6,8,12,16}; emits CSV/JSON summary. |
| P0.2 | Add Pareto summary | Report route_acc, generic_acc, affine_acc, wins/losses vs dense, wins/losses vs hand K=4, and correct count. |
| P0.3 | Add edge-level win export | For dense-correction wins and affine regressions, export node labels, target edges, learned edges, removed edges, extra edges, and scores. |
| P0.4 | Add repeated locked speed aggregator | Run N repeated locked benchmarks, summarize median/p25/p75/pass-rate; speed gate uses distribution. |
| P0.5 | Split timing regimes in artifact | Emit cold_prepare_ms, cache_hit_prepare_ms, static_preloaded_block_ms, cache_hit_block_ms, and growing_nodes_prepare_ms where available. |
| P0.6 | Preserve hand K=4 as formal baseline | Promotion and reports must include hand K=4 until a learned/dynamic policy beats its quality/runtime Pareto point. |

## P1 Work Items

| Priority | Item | Acceptance |
| ---: | --- | --- |
| P1.1 | Learned K=4 retraining / selection | Learned K=4 beats hand K=4 or matches generic while recovering affine regressions. |
| P1.2 | Dynamic-K prototype | Per-node or per-expression K preserves generic improvements while avoiding affine regressions. |
| P1.3 | Generic-stress validation expansion | Add held-out composite expressions that separate affine-looking local subgraphs from truly generic global structure. |
| P1.4 | Large-N static topology benchmark | Show stable static/preloaded speed advantage at N >= 2048, then N >= 4096. |
| P1.5 | Baked topology cache workflow | For fixed node hashes, load prepared topology from artifact and bypass scorer/topology rebuild. |

## Experimental Commands

### Hand K sweep

```bash
for k in 2 3 4 5 6 8 12 16; do
  echo "=== hand topology K=$k ==="
  python -m src.eval \
    --quality \
    --examples data/synthetic_hard/val.jsonl \
    --checkpoint runs/checkpoints/synthetic_hard_dense.pt \
    --topology-mode middle_preserving_topk \
    --fixed-k "$k" \
    --quality-k "$k" \
    --quality-device auto
done
```

### Learned K sweep

```bash
for k in 2 3 4 5 6 8 12 16; do
  echo "=== learned topology K=$k ==="
  python -m src.eval \
    --quality \
    --examples data/synthetic_hard/val.jsonl \
    --checkpoint runs/checkpoints/synthetic_hard_dense.pt \
    --topology-mode middle_preserving_topk \
    --fixed-k 16 \
    --quality-k 16 \
    --quality-device auto \
    --learned-scorer-checkpoint runs/checkpoints/topology_scorer.champion.pt \
    --learned-k "$k"
done
```

### Dense-correction wins export

```bash
python - <<'PY'
import json
from pathlib import Path
src = Path("runs/diagnostics/demo_generation_traces.jsonl")
out = Path("runs/diagnostics/learned_topology_dense_wins.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
n = 0
with src.open() as f, out.open("w") as g:
    for line in f:
        d = json.loads(line)
        pred = d.get("prediction", {})
        if pred.get("learned_correct") and not pred.get("dense_correct"):
            g.write(json.dumps(d, sort_keys=True) + "\n")
            n += 1
print("wrote:", out)
print("wins:", n)
PY
```

## Decision Rules

Do not promote a learned scorer merely because it beats hand K=16.

A new topology policy must satisfy at least one of:

```text
1. Quality dominance: route_acc > hand K=4 and no worse affine/generic Pareto tradeoff.
2. Pareto improvement: same route_acc as hand K=4 but materially faster or lower memory.
3. Stability improvement: lower variance speed/runtime at same quality.
4. Dynamic-regime improvement: much lower cold/growing-graph prepare cost at acceptable quality.
```

## Immediate Implementation Target

```text
Build a fixed-K sweep summarizer:
- input: eval logs or direct calls to run_quality_eval
- output: runs/diagnostics/topology_k_sweep_summary.{json,csv}
- rows: mode, k, route_acc, by_expert, correct_count, wins/losses vs dense, wins/losses vs hand K=4
```

This should become the new first command before any topology promotion discussion.

## Non-Negotiable Gates

```text
1. Champion can only change through the promotion gate.
2. Hand K=4 is now a required quality baseline.
3. Every topology quality report must include generic and affine accuracy.
4. Single-run strict speed is diagnostic only; speed acceptance requires repeated locked aggregation.
5. Claims about learned topology must separate K/pruning effect from learned edge-selection effect.
6. Edge-level traces are required before explaining why a win happened.
7. Large-N scaling claims require static/preloaded and cold/dynamic regimes to be reported separately.
```
