# plan.v12.md — Make Learned K=6 a Real Speed Win

## Context

v11 proved the learned-topology path at K=8:

```text
learned K=8 quality >= hand K=16 quality
learned K=8 prepared block speed > hand K=16 prepared block speed
```

The next candidate is K=6.

Full sweep result at `BENCH_N=1024`, `BENCH_STEPS=100`:

```text
hand K=16 route_acc=0.9821 block_ms=0.974695
learned K=6 route_acc=0.9871 block_ms=1.037430
speedup=0.939529
quality_ok=True speed_ok=False
```

So K=6 improves quality but fails benchmark speed acceptance.

Profiler result shows the likely fix is kernel block shape:

```text
hand K=16 median total_block_ms = 0.957
hand K=16 median attention_kernel_ms = 0.340

learned K=6, triton_block_k=8:
  total_block_ms = 1.326
  attention_kernel_ms = 0.403

learned K=6, triton_block_k=16:
  total_block_ms = 0.881
  attention_kernel_ms = 0.315
```

Therefore learned K=6 can be faster in isolated profiler mode when `triton_block_k=16`.

But the full benchmark command using:

```bash
TRITON_BLOCK_K=16 scripts/benchmark_learned_topology.sh
```

still failed speed acceptance. That implies the benchmark script likely does not forward `TRITON_BLOCK_K` into the `src.eval --benchmark` calls, or the benchmark path uses a different timing/kernel path than `profile_sparse_block.py`.

---

## v12 Objective

Make `learned_topology K=6` either:

1. pass the same acceptance gate as K=8, or
2. produce a precise explanation showing why the isolated profiler win does not transfer to full benchmark acceptance.

Acceptance target:

```text
quality_ok=True
speed_ok=True
learned K=6 route_acc >= hand K=16 route_acc
learned K=6 block_ms < hand K=16 block_ms
```

Stretch target:

```text
speedup >= 1.10x
```

---

## Task 1 — Wire Triton Kernel Shape Through Benchmark Script

### Problem

`profile_sparse_block.py` accepts:

```bash
--triton-block-k 16
```

and shows K=6 speed improvement.

`benchmark_learned_topology.sh` was run with:

```bash
TRITON_BLOCK_K=16
```

but the full benchmark still failed. Inspect and patch the script so this environment variable is actually forwarded into both benchmark calls.

### Implementation

Add environment variables:

```bash
TRITON_BLOCK_D="${TRITON_BLOCK_D:-}"
TRITON_BLOCK_K="${TRITON_BLOCK_K:-}"
```

Build optional argv arrays:

```bash
TRITON_ARGS=()
if [[ -n "$TRITON_BLOCK_D" ]]; then
  TRITON_ARGS+=(--triton-block-d "$TRITON_BLOCK_D")
fi
if [[ -n "$TRITON_BLOCK_K" ]]; then
  TRITON_ARGS+=(--triton-block-k "$TRITON_BLOCK_K")
fi
```

Pass:

```bash
"${TRITON_ARGS[@]}"
```

into both:

```bash
python -m src.eval --benchmark ... hand K=16
python -m src.eval --benchmark ... learned K=6
```

### Validation

Run:

```bash
SCORER=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
LEARNED_K=6 \
HAND_K=16 \
BENCH_STEPS=100 \
BENCH_N=1024 \
TRITON_BLOCK_K=16 \
scripts/benchmark_learned_topology.sh
```

Expected:

```text
triton_block_k=16 is visible in the benchmark path
```

---

## Task 2 — Add Benchmark Metadata to JSON and Console Output

The benchmark output should explicitly report kernel shape so results are auditable.

Add to `BenchmarkReport` or JSON payload if missing:

```text
triton_block_d
triton_block_k
```

Console output should include:

```text
triton_block_d=<value> triton_block_k=<value>
```

This prevents silent false assumptions like:

```text
TRITON_BLOCK_K=16 was set in environment but not used by benchmark runtime.
```

---

## Task 3 — Confirm `src.eval` Forwards Triton Shape Into Prepared Block

Inspect `src.eval.run_benchmark(...)` and confirm that:

```text
args.triton_block_d
args.triton_block_k
```

are passed into every `MathRoutedTransformerBlock(...)` used for:

```text
prepared_static_sparse_block_ms
prepared_static_sparse_attention_ms
prepared_static_sparse_non_attention_ms
```

If missing, add function parameters:

```python
triton_block_d: int | None = None
triton_block_k: int | None = None
```

and forward them into block construction.

Acceptance:

```text
profile_sparse_block.py and benchmark_learned_topology.sh use the same kernel configuration.
```

---

## Task 4 — Reconcile Profiler vs Benchmark Timing

If `TRITON_BLOCK_K=16` is correctly wired but full benchmark still fails, compare:

```text
scripts/profile_sparse_block.py K=6 block_k=16
src.eval --benchmark learned K=6 block_k=16
```

Measure:

```text
prepared_static_sparse_block_ms
prepared_static_sparse_attention_ms
prepared_static_sparse_non_attention_ms
```

| Observation                                               | Interpretation                                   | Next Fix                                         |
| ---                                                       | ---                                              | ---                                              |
| Profiler attention faster, benchmark attention not faster | Different kernel path or missing flag forwarding | Align eval benchmark path with profiler path     |
| Attention faster, non-attention worse                     | Block overhead dominates                         | Fuse norm/qkv/out-proj or reduce benchmark noise |
| Both attention and block faster                           | K=6 accepted                                     | Promote K=6 baseline                             |
| K=6 still slower despite same kernel                      | Memory locality/layout issue                     | Move to neighbor locality/block topology         |

---

## Task 5 — Locality Experiment

If K=6 remains slower after kernel-shape wiring, test neighbor ordering.

Options:

1. Sort selected neighbor ids ascending:

```text
N_i = sort(N_i)
```

2. Locality-biased priority:

```text
score'(i,j) = score(i,j) - lambda * abs(i-j)
```

3. Stable order by `(block_id, score)`:

```text
block_id = j // block_size
```

Acceptance:

```text
route_acc does not drop below hand K=16
attention_kernel_ms decreases
prepared block speed improves
```

---

## Task 6 — Block-Sparse Learned Topology Design

If arbitrary learned edges remain GPU-hostile, switch from token-edge topology to block topology.

Current:

```text
edge_score(i,j) -> topK token neighbors
```

Proposed:

```text
edge_score(i,j) -> aggregate block_score(p,q) -> topK blocks
```

where:

```text
p = floor(i / block_size)
q = floor(j / block_size)
```

Benefits:

```text
coalesced reads
fewer random gathers
GPU-friendly sparse structure
better Triton tiling
```

This is the likely long-term fix if K-specific kernel settings are not enough.

---

## Task 7 — Update Sweep Script With Kernel Args

`sweep_learned_k.sh` should forward:

```bash
TRITON_BLOCK_D
TRITON_BLOCK_K
```

to `benchmark_learned_topology.sh`.

Validation:

```bash
SCORER=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
LEARNED_K_VALUES="6 4" \
BENCH_STEPS=100 \
BENCH_N=1024 \
TRITON_BLOCK_K=16 \
scripts/sweep_learned_k.sh
```

---

## Task 8 — Preserve Current K=8 Baseline

Do not overwrite the accepted v11 baseline until K=6 passes full acceptance.

Current accepted baseline:

```text
SCORER=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt
LEARNED_K=8
```

K=6 is candidate only.

Promotion rule:

```text
promote K=6 only if full benchmark prints:
acceptance_passed quality_ok=True speed_ok=True
```

---

## Expected v12 Outcomes

### Success Case

```text
K=6 passes quality and speed with TRITON_BLOCK_K=16.
```

Then update canonical objective:

```text
learned K=6 > hand K=16 quality
learned K=6 > hand K=16 speed
```

### Partial Success Case

```text
K=6 profiler passes but full benchmark fails.
```

Then v12 deliverable is a precise profiler-vs-benchmark discrepancy report.

### Failure Case

```text
K=6 remains slower after kernel shape and locality fixes.
```

Then move to block-sparse learned topology.

---

## Commands

### Baseline profile

```bash
.venv-cuda/bin/python scripts/profile_sparse_block.py \
  --device cuda \
  --n 1024 \
  --topology-mode middle_preserving_topk \
  --fixed-k 16 \
  --max-neighbors 16 \
  --middle-bridge-width 1
```

### Candidate profile

```bash
.venv-cuda/bin/python scripts/profile_sparse_block.py \
  --device cuda \
  --n 1024 \
  --topology-mode learned_topology \
  --learned-scorer-checkpoint runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
  --learned-k 6 \
  --max-neighbors 6 \
  --middle-bridge-width 1 \
  --triton-block-k 16
```

### Candidate benchmark

```bash
SCORER=runs/checkpoints/scorer_runtime_aligned.runtime_best.pt \
LEARNED_K=6 \
HAND_K=16 \
BENCH_STEPS=100 \
BENCH_N=1024 \
TRITON_BLOCK_K=16 \
scripts/benchmark_learned_topology.sh
```

---

## Final Rule

K=6 is not accepted from profiler evidence alone.

Only this output promotes K=6:

```text
acceptance_passed quality_ok=True speed_ok=True
```
