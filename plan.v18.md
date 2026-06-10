# Plan v18 — Champion-Gated Learned Topology Handoff

## Current State

```text
Project = neurosymbolic/math_transformer
Status = champion-gated learned topology prototype
Champion scorer = runs/checkpoints/topology_scorer.champion.pt
Champion metadata = runs/checkpoints/topology_scorer.champion.json
Accepted proof = learned topology K=8 vs hand topology K=16 passes route quality and strict speed gate at current benchmark scale
Main bottleneck = stable larger-T/runtime proof + execution speed + broader validation
```

Current champion metrics:

```text
route_acc = 0.9871220604703248
generic_expert = 236/259 = 0.9111969111969112
strict_speed_ok = True
benchmark speedup ≈ 1.0146x on current benchmark
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

## Canonical Workflow

```text
train experiment scorer
→ runtime route-first select checkpoint
→ run benchmark and save log
→ promotion gate compares candidate vs champion
→ if pass, copy to topology_scorer.champion.pt and update metadata
→ inference/eval scripts default to champion
```

## Current Default Commands

```bash
# Quality eval with champion default
scripts/run_learned_topology_quality.sh

# Benchmark champion K=8 vs hand K=16
LEARNED_K=8 HAND_K=16 scripts/benchmark_learned_topology.sh

# Failure diagnostics from champion
python -m src.export_learned_topology_failures \
  --output runs/diagnostics/learned_topology_failures.champion.jsonl

# Replay candidate selection
scripts/select_topology_replay.sh runs/diagnostics/learned_topology_failures.champion.jsonl

# Candidate promotion gate
CANDIDATE=runs/checkpoints/<candidate>.pt \
BENCHMARK_LOG=runs/benchmarks/<candidate>.log \
scripts/promote_topology_scorer.sh
```

## Pending Work Priority

| Priority | Work item                            | Status  | Acceptance                                                                                                                    |
| ---:     | ---                                  | ---     | ---                                                                                                                           |
| P0.1     | Lock evaluator / benchmark protocol  | Partial | Stable command configs, seeds, hardware notes, metrics, and pass/fail gates are committed.                                    |
| P0.2     | Champion regression check script     | Pending | One command verifies `topology_scorer.champion.pt` still passes quality/speed after code changes.                             |
| P0.3     | Full benchmark artifact trace        | Pending | Benchmark emits structured JSON/JSONL with checkpoint hash, hardware, configs, quality, speed buckets, and acceptance fields. |
| P0.4     | Preserve champion default wiring     | Done    | Runtime/eval scripts default to `runs/checkpoints/topology_scorer.champion.pt`; training cannot overwrite it.                 |
| P1.1     | Expand hard validation set           | Partial | Add deeper/longer/held-out/generic-stress examples; report by-expert breakdown.                                               |
| P1.2     | Large-`T` scaling proof              | Pending | Show learned sparse topology advantage at `T=2048,4096,8192,16384`.                                                           |
| P1.3     | Stable execution profile             | Partial | Produce stable buckets: topology prepare, QKV, attention kernel, outproj, FFN, router, total.                                 |
| P1.4     | Prepared topology cache / baked IR   | Partial | Reused graphs bypass scorer/topology rebuild and load neighbor tables directly.                                               |
| P2.1     | Block sparse / memory locality proof | Partial | Benchmark block sparse vs token sparse at `T >= 4096`; promote only if wall-clock wins.                                       |
| P2.2     | Depth sweep with synced topology     | Pending | Compare `L ∈ {1,2,4,8}` using same topology and select by `J=Q-λT-γM`.                                                        |
| P2.3     | Replay sweep under promotion gate    | Partial | Replay variants use route-first runtime selection and promote only if route/generic/speed gates pass.                         |
| P3.1     | Feature registry / feature gates     | Pending | Versioned feature schemas and ablation reports exist before changing scorer input dimension.                                  |
| P3.2     | Dynamic-K controller                 | Pending | Per-node `K_i` improves objective over fixed K=8 champion.                                                                    |
| P3.3     | Active node selection                | Pending | `V_active=TopM(q_i)` improves quality/runtime vs fixed-node baseline.                                                         |
| P3.4     | Code/AST dataset path                | Pending | JSONL supports AST/CFG/SSA/call/type nodes, edges, labels, and evaluator feedback.                                            |
| P3.5     | Edit/action head                     | Pending | Model proposes graph/code edits verified by tests/typechecker/evaluator.                                                      |

## Implementation Bias

```text
Do less theory.
Measure more.
Protect the champion.
Make every proposed improvement pass the promotion gate.
Do not let proxy metrics override route/generic accuracy.
```

## Files To Inspect First

| Area               | Files                                                                                                                                                                                                   |
| ---                | ---                                                                                                                                                                                                     |
| Roadmap            | `TODO.md`, `plan.v18.md`                                                                                                                                                                                |
| Champion promotion | `src/promote_topology_scorer.py`, `scripts/promote_topology_scorer.sh`                                                                                                                                  |
| Training           | `src/train_topology_scorer.py`, `scripts/train_topology_scorer.sh`                                                                                                                                      |
| Replay             | `src/topology_trace_replay.py`, `scripts/select_topology_replay.sh`                                                                                                                                     |
| Traces/failures    | `src/topology_trace.py`, `src/export_learned_topology_failures.py`                                                                                                                                      |
| Eval/benchmark     | `src/eval.py`, `src/eval_topology_scorer.py`, `scripts/run_learned_topology_quality.sh`, `scripts/benchmark_learned_topology.sh`                                                                        |
| Topology           | `src/topology.py`, `src/learned_topology.py`, `src/learned_topology_runtime.py`, `src/topology_cache.py`                                                                                                |
| Attention/runtime  | `src/attention.py`, `src/sparse_attention.py`, `src/block_sparse_attention.py`, `src/triton_attention.py`, `src/triton_block_sparse_attention.py`                                                       |
| Tests              | `tests/test_promote_topology_scorer.py`, `tests/test_train_topology_scorer.py`, `tests/test_topology_trace.py`, `tests/test_topology_trace_replay.py`, `tests/test_export_learned_topology_failures.py` |

## Non-Negotiable Gates

```text
1. Champion can only change through the promotion gate.
2. Training output paths must not default to champion paths.
3. Runtime selection must be route-first and generic-aware.
4. Benchmark speed claims require stable wall-clock proof.
5. Feature schema changes require explicit versioning/checkpoint handling.
6. Architecture expansion waits until evaluator/trace/promotion gates stay green.
```
