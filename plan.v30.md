# Plan v30 — Runtime Kernel Registry

## Official Name

```text
Verified Kernel Registry for Executable Sparse Graphs
```

Runtime intent:

```text
compiled kernel artifact -> registry admission -> callable graph node -> runtime dispatch -> bounded execution policy
```

## Why v30 Exists

v29 proves that pure graph subgraphs can compile into verified CPU kernels.

v30 connects accepted kernels back to the sparse graph runtime.

The project moves from:

```text
compiled code exists as external artifact
```

to:

```text
compiled code is a typed graph node that can be selected, dispatched, traced, and compared against fallbacks
```

The key correction is:

```text
Registration is separate from compilation.
A compiled artifact must pass registry checks before the graph can call it.
```

## Core Decision

```text
Every callable kernel must have a registry record, content hash, ABI signature, verifier, fallback, and resource limits.
```

Correct pipeline:

```text
accepted_kernel.json
  -> registry validation
  -> callable symbol binding
  -> fallback binding
  -> dispatch policy
  -> V_code node insertion
```

Rejected pipeline:

```text
compiler output path -> direct runtime dlopen/call with no registry gate
```

## Core Objects

```text
KernelRegistry = registry of callable compiled kernels.
K_i = compiled kernel artifact.
Node(K_i) = graph code node.
Fallback(K_i) = reference or safe interpreted path.
Policy(K_i) = runtime limits and allowed contexts.
Dispatch(K_i, x) = controlled kernel invocation.
```

Kernel registry record:

```text
KernelRecord_i = (
  kernel_id,
  op_ids,
  abi_signature,
  artifact_path,
  source_hash,
  binary_hash,
  verifier_id,
  fallback_id,
  resource_limits,
  benchmark_summary,
  status
)
```

## Dispatch Rule

```text
Run(K_i, x) allowed iff
  Registered(K_i)
  ∧ HashCheck(K_i)
  ∧ Typecheck(x, ABI(K_i))
  ∧ ResourcePolicy(K_i)
  ∧ VerifierAvailable(K_i)
```

Fallback rule:

```text
if Run(K_i, x) fails policy or verifier:
  y = Fallback(K_i, x)
```

Runtime insertion:

```text
V_code,t+1 = V_code,t ∪ {Node(K_i) : RegistryAccept(K_i)}
```

## Artifact Contract

Output directory:

```text
runs/kernel_registry/<experiment_id>/
```

Required artifacts:

```text
registry_config.json
kernel_registry.json
registry_admission_report.json
callable_symbols.json
fallback_registry.json
dispatch_policy.json
hash_check_report.json
runtime_smoke_report.json
quality_report.json
```

`kernel_registry.json` must record:

```json
{
  "kernel_id": "kernel-topk-edge-select-v1",
  "op_ids": ["topk_edge_select"],
  "abi_signature": "float32[N], int64[K] -> int64[K]",
  "artifact_path": "runs/cpu_kernel_compile/.../kernel.so",
  "source_hash": "sha256:...",
  "binary_hash": "sha256:...",
  "verifier_id": "topk_membership_and_order_check",
  "fallback_id": "python_reference_topk_edge_select",
  "resource_limits": {"timeout_ms": 100, "max_bytes": 1048576},
  "status": "registered"
}
```

Hard rule:

```text
The graph calls registry IDs, not raw paths.
```

## Implementation Plan

### P0.1 — Registry Schema

Add:

```text
src/kernel_registry.py
tests/test_kernel_registry.py
```

Acceptance:

```text
Registers accepted kernels by content hash and ABI signature.
Rejects missing fallback, missing verifier, hash mismatch, and unknown symbols.
```

### P0.2 — Runtime Dispatch Wrapper

Acceptance:

```text
Dispatches registered pure kernels through a stable API.
Checks input type and resource limits.
Falls back to reference implementation on policy failure.
```

### P0.3 — Graph Node Binding

Acceptance:

```text
Adds V_code nodes for registered kernels.
Preserves original op provenance and source subgraph lineage.
```

## Success Criteria

```text
A verified CPU kernel can be registered and called through the graph runtime.
The registry blocks direct path execution.
Fallback works when dispatch is rejected or verification fails.
```
