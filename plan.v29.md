# Plan v29 — CPU Kernel Compiler Backend

## Official Name

```text
Verified Sparse Graph CPU Kernel Compiler
```

Runtime intent:

```text
ABI-admitted pure op subgraph -> IR -> CPU backend -> compiled kernel artifact -> verifier-gated callable function
```

## Why v29 Exists

v28 defines which graph nodes are safe and well-typed enough to be executable.

v29 lowers eligible pure subgraphs into CPU code.

The project moves from:

```text
graph can declare executable operations
```

to:

```text
graph subgraphs can become compiled kernels with stable signatures, benchmarks, hashes, and verifiers
```

The key correction is:

```text
Compilation is a backend optimization, not a permission bypass.
Only ABI-admitted pure subgraphs can compile in v29.
```

## Core Decision

```text
Start with deterministic CPU kernels for repeated sparse graph operations.
Do not compile arbitrary model-authored action code.
```

First target kernels:

```text
top-k edge selection
sparse adjacency gather/scatter
edge utility aggregation
graph traversal within bounded horizon
sparse attention mask construction
candidate edge filtering
ablation metric reduction
```

## Core Objects

```text
S ⊆ G_t = selected pure executable subgraph.
IR(S) = lowered intermediate representation.
K_i = compiled CPU kernel.
Hash(K_i) = content hash of compiler input and output.
Bench(K_i) = runtime/memory benchmark record.
Ver(K_i) = kernel verifier.
ABI(K_i) = callable runtime signature.
```

Backend options:

| backend | allowed use |
|---|---|
| Python reference | correctness oracle only |
| C shared object | primary simple CPU target |
| LLVM/MLIR | later optimized target |
| Rust shared library | optional safer systems target |
| Triton/GPU | not part of v29 CPU proof |

## Compilation Formula

Subgraph selection:

```text
S* = argmax_S [Reuse(S) + CostSaved(S) + Verifiability(S) - Risk(S)]
```

Lowering:

```text
IR_S = Lower(S*)
```

Compilation:

```text
K_i = CompileCPU(IR_S)
```

Admission of compiled artifact:

```text
K_i accepted iff Typecheck(K_i) ∧ SandboxCheck(K_i) ∧ Ver(K_i) ∧ Bench(K_i)
```

Equivalence check:

```text
∀x ∈ D_check: K_i(x) = Ref_S(x)
```

or, for floating point:

```text
max_x ||K_i(x) - Ref_S(x)|| <= ε
```

## Artifact Contract

Output directory:

```text
runs/cpu_kernel_compile/<experiment_id>/<kernel_id>/
```

Required artifacts:

```text
compile_config.json
source_subgraph.json
lowered_ir.json
reference_impl.py
generated_source.c
build_manifest.json
kernel_hashes.json
equivalence_report.json
benchmark_report.json
sandbox_report.json
accepted_kernel.json
rejected_kernel.json
```

Hard rule:

```text
Compiled kernels are inert artifacts until registered by the runtime registry in v30.
```

## Implementation Plan

### P0.1 — IR Lowering

Add:

```text
src/cpu_kernel_compiler.py
tests/test_cpu_kernel_compiler.py
```

Acceptance:

```text
Lowers a pure ABI op sequence into deterministic IR.
Rejects effectful or untyped ops.
Emits source_subgraph.json and lowered_ir.json.
```

### P0.2 — Reference Oracle

Acceptance:

```text
Runs Python reference implementation for each target kernel.
Stores input/output fixtures.
Uses fixtures for generated CPU equivalence tests.
```

### P0.3 — C Backend Proof

Acceptance:

```text
Generates C for at least one kernel class.
Builds a shared object or executable test binary.
Runs equivalence and benchmark checks.
Rejects kernel on mismatch or timeout.
```

## Success Criteria

```text
At least one repeated sparse graph operation compiles into a verified CPU kernel.
The compiled kernel matches the Python reference on generated fixtures.
The compiled kernel emits benchmark and hash artifacts.
No effectful operation is compiled.
```
