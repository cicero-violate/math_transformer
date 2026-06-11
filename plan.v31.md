# Plan v31 — Execution Trace Ledger

## Official Name

```text
Verified Execution Trace Ledger
```

Runtime intent:

```text
graph-dispatched kernel call -> input/output/cost/verifier trace -> replayable evidence -> utility signal for future graph updates
```

## Why v31 Exists

v30 lets the sparse graph call registered compiled kernels.

v31 records every meaningful execution as evidence.

The project moves from:

```text
the graph can execute compiled kernels
```

to:

```text
the graph can learn which executions are useful, stable, reusable, and safe enough to keep
```

The key correction is:

```text
Execution without trace is not learnable.
Execution traces become graph-memory evidence, not informal logs.
```

## Core Decision

```text
Every kernel run that influences model output must emit a replayable trace record.
```

Correct pipeline:

```text
Dispatch(K_i, x)
  -> y_i
  -> verifier result
  -> cost measurement
  -> fallback comparison when configured
  -> trace ledger append
  -> utility aggregation input
```

Rejected pipeline:

```text
kernel returns answer -> no provenance, no verifier, no replay data
```

## Core Objects

```text
T_i = execution trace record.
K_i = registered kernel.
x_i = canonicalized input digest or fixture reference.
y_i = canonicalized output digest or fixture reference.
s_i = verifier score.
c_i = measured cost.
r_i = reuse count.
u_i = utility estimate.
Replay(T_i) = deterministic rerun check.
```

Trace record:

```text
T_i = (
  trace_id,
  kernel_id,
  graph_node_id,
  input_digest,
  output_digest,
  verifier_result,
  cost_actual,
  fallback_result,
  graph_context,
  timestamp,
  replay_status
)
```

## Utility Formula

```text
u(K_i) = ΔA(K_i) - λ_C ΔC(K_i) - λ_R Risk(K_i) + λ_Ru Reuse(K_i)
```

where:

| term | meaning |
|---|---|
| `ΔA(K_i)` | quality or accuracy contribution |
| `ΔC(K_i)` | runtime/memory cost |
| `Risk(K_i)` | verifier failure, instability, policy failure |
| `Reuse(K_i)` | repeated successful use across contexts |

Bake-candidate rule:

```text
CandidateBake(K_i) iff u(K_i) > τ_u ∧ ReplayPassRate(K_i) >= τ_replay
```

## Artifact Contract

Output directory:

```text
runs/execution_trace_ledger/<experiment_id>/
```

Required artifacts:

```text
trace_config.json
execution_traces.jsonl
trace_index.sqlite
replay_report.json
kernel_utility_report.json
verifier_failure_report.jsonl
fallback_comparison_report.jsonl
cost_report.json
candidate_bake_kernels.jsonl
quality_report.json
```

Hard rule:

```text
No kernel can be baked into the preferred graph path without trace evidence.
```

## Implementation Plan

### P0.1 — Trace Schema

Add:

```text
src/execution_trace_ledger.py
tests/test_execution_trace_ledger.py
```

Acceptance:

```text
Writes deterministic JSONL traces for registered kernel calls.
Canonicalizes inputs/outputs by digest or fixture reference.
Records verifier result, cost, fallback status, and graph context.
```

### P0.2 — Replay Harness

Acceptance:

```text
Replays traces against the registered kernel and fallback.
Flags nondeterminism, mismatch, timeout, and verifier failure.
```

### P0.3 — Utility Aggregator

Acceptance:

```text
Aggregates per-kernel utility from trace evidence.
Emits candidate_bake_kernels.jsonl only for replay-stable kernels.
```

## Success Criteria

```text
Every graph-dispatched kernel call can be traced and replayed.
Trace evidence produces utility scores.
Only replay-stable, useful kernels become bake candidates for v32.
```
