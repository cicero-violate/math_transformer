# Plan v32 — Verified Kernel Baking

## Official Name

```text
Usefulness-Gated Executable Graph Baking
```

Runtime intent:

```text
execution trace evidence -> utility score -> regression gate -> preferred graph path update -> baked executable sparse routine
```

## Why v32 Exists

v31 records execution traces and identifies useful, replay-stable kernels.

v32 lets the graph permanently prefer verified compiled paths when they improve quality or efficiency.

The project moves from:

```text
compiled kernels are callable options
```

to:

```text
compiled kernels become preferred graph structure when evidence proves they are useful
```

The key correction is:

```text
Correct once is not enough.
Baking requires replay stability, held-out quality, old-domain regression safety, and cost improvement.
```

## Core Decision

```text
Bake only verified executable paths that improve usefulness under regression gates.
Archive or demote kernels that fail stability, quality, or safety checks.
```

Correct pipeline:

```text
candidate_bake_kernels.jsonl
  -> replay gate
  -> held-out quality gate
  -> old-domain regression gate
  -> cost/memory gate
  -> graph path preference update
  -> rollback manifest
```

Rejected pipeline:

```text
kernel produced a correct answer once -> make it permanent
```

## Core Objects

```text
K_i = candidate compiled kernel.
P_i = graph path using K_i.
P_ref = previous reference path.
U(K_i) = usefulness score.
Q_new = quality with baked path.
Q_old = quality before baking.
C_new = runtime/memory cost with baked path.
C_old = runtime/memory cost before baking.
Rollback(K_i) = manifest to undo baked preference.
```

## Bake Formula

Usefulness:

```text
U(K_i) = ΔQ(K_i) - λ_C ΔC(K_i) - λ_M ΔM(K_i) - λ_R Risk(K_i) + λ_Re Reuse(K_i)
```

Bake rule:

```text
Bake(K_i) iff
  U(K_i) > τ
  ∧ ReplayPassRate(K_i) >= τ_replay
  ∧ Q_heldout(K_i) >= Q_ref - ε_Q
  ∧ Q_old_domain(K_i) >= Q_old_domain(ref) - ε_old
  ∧ C_new <= C_old
```

Graph update:

```text
G_{t+1} = PreferPath(G_t, P_i) if Bake(K_i) else G_t
```

Rollback:

```text
Rollback(K_i): G_{t+1} -> G_t by restoring previous path preference and registry status
```

## Artifact Contract

Output directory:

```text
runs/kernel_baking/<experiment_id>/
```

Required artifacts:

```text
bake_config.json
candidate_bake_kernels.jsonl
accepted_bakes.jsonl
rejected_bakes.jsonl
path_preference_before.json
path_preference_after.json
heldout_quality_report.json
old_domain_regression_report.json
cost_memory_report.json
rollback_manifest.json
runtime_report.json
quality_report.json
```

Hard rule:

```text
Baked executable paths must remain reversible.
```

## Implementation Plan

### P0.1 — Bake Evaluator

Add:

```text
src/kernel_baking.py
tests/test_kernel_baking.py
```

Acceptance:

```text
Consumes candidate_bake_kernels.jsonl from v31.
Runs replay, held-out quality, regression, and cost gates.
Emits accepted_bakes.jsonl and rejected_bakes.jsonl.
```

### P0.2 — Path Preference Update

Acceptance:

```text
Updates graph path preference to use accepted kernels.
Does not remove reference fallback paths.
Emits before/after preference artifacts.
```

### P0.3 — Rollback Manifest

Acceptance:

```text
Records enough information to undo every baked path.
Rollback test restores the previous path preference exactly.
```

## Success Criteria

```text
Useful compiled kernels become preferred executable graph paths.
No baked path is irreversible.
No kernel is baked without trace evidence and regression gates.
The sparse graph can now reason, dispatch, execute, verify, and retain useful execution paths.
```
