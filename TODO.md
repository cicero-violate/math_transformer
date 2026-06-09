# Pending Task Table

Leverage formula:

```text
L(move) = (quality_gain × speed_gain × reuse × money_flow × autonomy_gain) / cost
move* = argmax L(move)
```

Canonical objective:

```text
K=8 sparse learned topology = dense quality > current K=16 hand topology speed
```

## Research / Runtime Tasks

| Rank | Task | Status | Evidence / Current State | Next Action |
| ---: | --- | --- | --- | --- |
| 1 | Train learned topology scorer | Done | Learned topology scorer exists; runtime-best checkpoint selected; K=8 quality/speed proof passed. | Keep `scorer_runtime_aligned.runtime_best.pt` as baseline. |
| 2 | Dense teacher → sparse student distillation | Partial | Dense teacher trace path and soft dense-equivalence training path implemented; runtime-J training did not improve over baseline yet. | Tune dense-equivalence loss weights/objective; avoid runtime-J drift. |
| 3 | Build harder validation data | Partial | Synthetic hard data exists, but generic expert remains the main failure mode and current validation is still easy for several experts. | Add deeper trees, longer dependencies, ambiguous operators, held-out templates, larger `T`. |
| 4 | Scale graph size | Pending | K=8/K=16 proof done at current benchmark scale; large `T` sweep not complete. | Benchmark `T=2048,4096,8192,16384` with prepared sparse blocks. |
| 5 | Train dynamic-K controller | Pending | Fixed-K learned topology path exists; no per-node adaptive K controller yet. | Learn `K_i = controller(node_i, uncertainty_i, importance_i)`. |
| 6 | Program dependency graph | Pending | Not implemented in this module. | Model files/functions/types/variables and call/import/read/write edges. |
| 7 | Agent memory graph | Pending | Not implemented in this module. | Track tasks, decisions, failures, patches, files, tests, and dependencies. |
| 8 | Market value graph | Pending | Outside current model research track. | Build listings/products/specs/prices graph for mispricing detection. |
| 9 | Pain / manual work graph | Pending | Outside current model research track. | Model workflows, friction, repetition, handoffs, and automation value. |
| 10 | Shape / type validity model | Partial | Shape/env handling exists for topology and examples; no standalone validity model. | Train `expression + env → valid/invalid + output shape`. |
| 11 | Symbolic math embedding model | Partial | `MathEmbedder` exists; no dedicated contrastive/equivalence training objective. | Train equivalent expressions close and different expressions far. |
| 12 | Algebraic canonicalizer | Partial | Parser/normalizer/canonical-ish flow exists; no learned canonicalizer. | Expand rewrite/canonicalization coverage and tests. |
| 13 | Knowledge provenance graph | Pending | Not implemented. | Model claims, sources, evidence, contradictions, supports/refutes edges. |
| 14 | Kernel specialization | Partial | Prepared sparse block profiling exists; Triton sparse path exists; target kernel goals not met consistently. | Specialize for `H=4, D=16, K=16, T=1024+`; target `attention_kernel_ms <= 0.22`, `total_block_ms <= 0.75`. |
| 15 | Supply chain graph | Pending | Outside current model research track. | Acquire data and model companies/products/regions/dependencies. |

## Immediate Pending Tasks

| Priority | Task | Status | Target / Acceptance |
| ---: | --- | --- | --- |
| P0 | Preserve K=8 learned-topology baseline | Done | `quality_ok=True`, `speed_ok=True`, speedup ≈ `1.03×` vs hand K=16. |
| P0 | Re-run full K=6 benchmark after timing noise investigation | Pending | `quality_ok=True` and `speed_ok=True` at `BENCH_N=1024`, `BENCH_STEPS=100`. Current full run: quality passed, speed failed. |
| P1 | Investigate K=4 representation drift | Pending | Improve `hidden_cos` and reduce `logit_kl` while preserving high route accuracy. |
| P1 | Tune dense-equivalence training objective | Pending | Runtime-J improves over `scorer_runtime_j_best.pt`, not merely early-stops. |
| P1 | Export learned-topology failure diagnostics | Pending | Produce JSONL of generic-expert misses, missing/extra edges, hidden/logit drift. |
| P2 | Large-`T` scaling benchmark | Pending | Show `T² → T*K` advantage at `T >= 4096`. |
| P2 | Kernel specialization pass | Pending | Stable prepared block speed win beyond timing noise. |

## Useful Commands

```bash
# Main accepted proof, K=8 vs hand K=16
SCORER=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
LEARNED_K=8 \
HAND_K=16 \
scripts/benchmark_learned_topology.sh

# K sweep, allowed to report failures without aborting
SCORER=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
BENCH_STEPS=100 \
BENCH_N=1024 \
scripts/sweep_learned_k.sh

# Runtime-aligned training attempt
./run_training.sh
```
