# Plan v18 — Champion-Gated Learned Topology Handoff

## Current State

```text
Project = neurosymbolic/math_transformer
Status = champion-gated learned topology prototype; post-v18 evidence shifts the main hypothesis toward sparsity/pruning as structural denoising
Champion scorer = runs/checkpoints/topology_scorer.champion.pt
Champion metadata = runs/checkpoints/topology_scorer.champion.json
Historical champion proof = learned topology K=8 vs hand topology K=16 passes route quality; strict speed is not stable enough as a single-run claim
Current quality leader = hand topology K=4 on synthetic_hard val
Main bottleneck = distinguish pruning effect vs learned edge-selection effect, then prove stable wall-clock/large-T behavior
```

Current champion metrics remain the metadata baseline for promotion:

```text
route_acc = 0.9871220604703248
generic_expert = 236/259 = 0.9111969111969112
champion = learned topology scorer at K=8
```

Post-v18 empirical correction:

```text
Dense full route_acc              = 0.980963, generic = 225/259 = 0.868726
Hand middle_preserving_topk K=4   = 0.992721, generic = 252/259 = 0.972973  <-- current best quality
Hand middle_preserving_topk K=8   = 0.988802, generic = 239/259 = 0.922780
Hand middle_preserving_topk K=12  = 0.987122, generic = 236/259 = 0.911197
Hand middle_preserving_topk K=16  = 0.983763, generic = 230/259 = 0.888031
Learned topology K=8              = 0.989362, generic = 240/259 = 0.926641
```

Interpretation:

```text
1. The dominant quality gain is pruning / structural denoising, not learned edge novelty.
2. Learned K=8 beats hand K=8 by only 1 sample on this validation run.
3. Hand K=4 beats learned K=8 by 6 samples and beats dense by 21 samples.
4. The generic_expert class is the main beneficiary; dense overpredicts affine_expert on composite generic expressions.
5. Current speed signal is in the noise zone: locked runs have produced both strict_speed_ok=True and strict_speed_ok=False.
6. topology_prepare_ms is a real cold-cache cost, not benchmark-loop per-forward cost; it matters when node hashes change frequently.
```

## What Changed Since v17

| Area                           | Status                                                                      |
| ---                            | ---                                                                         |
| Standard topology trace schema | Implemented for scorer eval, quality eval, and failure diagnostics.         |
| Failure diagnostics            | Standardized into trace schema with missing/extra edges and route misses.   |
| Trace replay                   | Implemented replay candidate selection from failure traces.                 |
| Replay training                | Implemented appended replay records, loss weights, and replay oversampling. |
| Runtime checkpoint selection   | Implemented route-first/generic-aware runtime selection.                    |
| Promotion gate                 | Implemented champion checkpoint manager and metadata audit.                 |
| Default scorer wiring          | Inference/eval scripts now default to champion checkpoint.                  |
| Locked protocol/artifacts      | Implemented committed locked protocol, structured benchmark artifacts, and champion regression script. |
| Quality trace demonstration    | Learned topology generated route predictions and produced 15 dense-correction wins with 0 dense-regression losses. |
| K-sweep finding                | Hand K=4 is current quality leader; learned K=8 is only a small same-K improvement over hand K=8. |

## Canonical Workflow

```text
train experiment scorer
→ runtime route-first select checkpoint
→ run locked quality/speed artifact
→ promotion/regression gate compares candidate vs champion
→ if pass, copy to topology_scorer.champion.pt and update metadata
→ inference/eval scripts default to champion
```

Important correction:

```text
Champion immutability is still required, but the current champion is not necessarily the current best topology policy.
New topology policies must be compared against dense, hand K=4, hand K=8, hand K=16, and champion learned K=8.
```

## Current Default Commands

```bash
# Quality eval with champion default
scripts/run_learned_topology_quality.sh

# Historical champion proof, learned K=8 vs hand K=16
LEARNED_K=8 HAND_K=16 scripts/benchmark_learned_topology.sh

# Locked evaluator/benchmark protocol with structured artifacts
scripts/run_locked_topology_protocol.sh

# Champion regression proof with structured artifact audit
scripts/check_topology_champion_regression.sh

# Hand K sweep that exposed pruning/denoising effect
for k in 4 8 12 16; do
  python -m src.eval \
    --quality \
    --examples data/synthetic_hard/val.jsonl \
    --checkpoint runs/checkpoints/synthetic_hard_dense.pt \
    --topology-mode middle_preserving_topk \
    --fixed-k "$k" \
    --quality-k "$k" \
    --quality-device auto
done

# Learned K sweep still needed
for k in 2 3 4 5 6 8; do
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

## Pending Work Priority

| Priority | Work item                                  | Status  | Acceptance |
| ---:     | ---                                        | ---     | --- |
| P0.1     | Lock evaluator / benchmark protocol        | Done    | Committed protocol config, runner, structured artifacts, protocol hash, and champion regression command exist. |
| P0.2     | Champion regression check script           | Done    | One command verifies `topology_scorer.champion.pt` against metadata and artifact gates. |
| P0.3     | Full benchmark artifact trace              | Done    | Benchmark emits JSON/JSONL with checkpoint hash, hardware, config, quality, speed buckets, and gates. |
| P0.4     | Preserve champion default wiring           | Done    | Runtime/eval scripts default to champion; training cannot overwrite it. |
| P0.5     | Complete learned/hand K sweep              | Pending | Compare dense, hand K={2,3,4,5,6,8,12,16}, learned K={2,3,4,5,6,8}; report route/generic/affine tradeoff. |
| P0.6     | Add sweep summarizer artifact              | Pending | One command writes CSV/JSON table with by-expert accuracy, wins/losses vs dense, and topology overlap. |
| P0.7     | Add edge-level win trace export            | Pending | For dense-correction wins, export node labels plus learned/target/extra/missing edge lists. |
| P0.8     | Repeated locked speed aggregation          | Pending | Gate speed by median/p25/p75/pass-rate, not single-run strict speed. |
| P1.1     | Promote pruning/denoising baseline         | Pending | Preserve hand K=4 as a formal baseline; learned candidates must beat hand K=4 or reduce affine regressions at equal/better generic accuracy. |
| P1.2     | Expand hard validation set                 | Partial | Add deeper/longer/held-out/generic-stress examples; report by-expert breakdown. |
| P1.3     | Large-`T` scaling proof                    | Pending | Show topology policy advantage at `T=2048,4096,8192,16384` with prepared sparse/block paths. |
| P1.4     | Stable execution profile                   | Partial | Produce stable buckets: cold prepare, cache hit, static preloaded block, QKV, attention kernel, outproj, FFN, total. |
| P1.5     | Prepared topology cache / baked IR         | Partial | Reused graphs bypass scorer/topology rebuild and load neighbor tables directly. |
| P2.1     | Block sparse / memory locality proof       | Partial | Benchmark block sparse vs token sparse at `T >= 4096`; promote only if wall-clock wins. |
| P2.2     | Depth sweep with synced topology           | Pending | Compare `L ∈ {1,2,4,8}` using same topology and select by `J=Q-λT-γM`. |
| P2.3     | Replay sweep under promotion gate          | Partial | Replay variants use route-first runtime selection and promote only if route/generic/speed gates pass. |
| P3.1     | Feature registry / feature gates           | Pending | Versioned feature schemas and ablation reports exist before changing scorer input dimension. |
| P3.2     | Dynamic-K controller                       | Pending | Per-node `K_i` improves objective over fixed K baselines. |
| P3.3     | Active node selection                      | Pending | `V_active=TopM(q_i)` improves quality/runtime vs fixed-node baseline. |

## Implementation Bias

```text
Do less theory.
Measure more.
Protect the champion.
Do not confuse champion status with best-known topology policy.
Treat sparse topology as structural denoising until learned edge selection proves a larger same-K advantage.
Make every proposed improvement pass dense, hand K=4, hand K=8, hand K=16, and champion learned K=8 comparisons.
Do not claim speed from a single locked run.
```

## Files To Inspect First

| Area               | Files |
| ---                | --- |
| Roadmap            | `TODO.md`, `plan.v18.md`, `plan.v19.md` |
| Protocol/artifacts | `configs/learned_topology_locked_protocol.json`, `src/topology_protocol.py`, `src/topology_benchmark_artifact.py` |
| Champion promotion | `src/promote_topology_scorer.py`, `scripts/promote_topology_scorer.sh`, `src/check_topology_champion_regression.py`, `scripts/check_topology_champion_regression.sh` |
| Eval/benchmark     | `src/eval.py`, `src/eval_topology_scorer.py`, `scripts/run_learned_topology_quality.sh`, `scripts/benchmark_learned_topology.sh`, `scripts/run_locked_topology_protocol.sh` |
| Traces/failures    | `src/topology_trace.py`, `src/export_learned_topology_failures.py`, `runs/diagnostics/demo_generation_traces.jsonl`, `runs/diagnostics/learned_topology_dense_wins.jsonl` |
| Training/replay    | `src/train_topology_scorer.py`, `scripts/train_topology_scorer.sh`, `src/topology_trace_replay.py`, `scripts/select_topology_replay.sh` |
| Topology/cache     | `src/topology.py`, `src/learned_topology.py`, `src/learned_topology_runtime.py`, `src/topology_cache.py` |
| Attention/runtime  | `src/attention.py`, `src/sparse_attention.py`, `src/block_sparse_attention.py`, `src/triton_attention.py`, `src/triton_block_sparse_attention.py` |
| Tests              | `tests/test_promote_topology_scorer.py`, `tests/test_topology_benchmark_artifact.py`, `tests/test_topology_protocol.py`, `tests/test_topology_trace.py` |

## Non-Negotiable Gates

```text
1. Champion can only change through the promotion gate.
2. Training output paths must not default to champion paths.
3. Runtime selection must be route-first and generic-aware.
4. Benchmark speed claims require repeated locked wall-clock proof.
5. Feature schema changes require explicit versioning/checkpoint handling.
6. Architecture expansion waits until evaluator/trace/promotion gates stay green.
7. Any learned topology claim must compare against the new hand K=4 pruning baseline.
8. Any quality claim must report generic and affine tradeoff, not just route_acc.
```
