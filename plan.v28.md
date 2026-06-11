# Plan v28 — Executable Graph Op ABI

## Official Name

```text
Typed Executable Sparse Graph ABI
```

Runtime intent:

```text
sparse reasoning graph -> typed executable node contract -> safe op signatures -> effect policy -> compile-ready subgraphs
```

## Why v28 Exists

v27 keeps many teacher-derived graph priors available while bounding the active sparse student graph.

v28 adds the bridge from learned graph structure to executable computation.

The project moves from:

```text
the graph selects and routes reasoning structure
```

to:

```text
the graph can also declare pure executable operations with typed inputs, outputs, costs, effects, and verifiers
```

The key correction is:

```text
Executable graph nodes are not arbitrary generated programs.
They are typed operations admitted through an ABI and policy gate.
```

## Core Decision

```text
Every executable node must have a stable operation contract before it can be compiled or called.
```

Correct pipeline:

```text
graph node proposal
  -> op signature
  -> type/effect declaration
  -> verifier declaration
  -> cost model declaration
  -> ABI registry admission
  -> compile eligibility
```

Rejected pipeline:

```text
model emits raw code -> runtime executes it directly
```

## Core Objects

```text
G_t = active sparse graph at step t.
V_reason = reasoning nodes.
V_memory = memory/provenance/trace nodes.
V_code = executable operation nodes.
V_action = controlled effect/tool nodes.
E = typed graph edges.
Op = executable operation contract.
Sig(Op) = input/output type signature.
Eff(Op) = declared side-effect class.
Ver(Op) = verifier for operation output.
Cost(Op) = estimated runtime/memory cost.
ABI = stable callable interface exposed to graph runtime.
```

Allowed effect classes:

| effect | meaning | v28 status |
|---|---|---|
| pure | deterministic `X -> Y` with no external effects | allowed |
| read_state | reads bounded local state | gated |
| write_state | writes bounded local state | later |
| external_action | files, network, tools, devices | rejected in v28 |

## ABI Formula

Executable graph:

```text
G = (V_reason ∪ V_memory ∪ V_code ∪ V_action, E)
```

Operation contract:

```text
Op_i = (id_i, inputs_i, outputs_i, types_i, effect_i, cost_i, verifier_i, provenance_i)
```

Admission rule:

```text
Op_i ∈ ABI iff Typecheck(Op_i) ∧ EffectPolicy(Op_i) ∧ VerifierExists(Op_i)
```

Compile eligibility:

```text
Eligible(Op_i) = Pure(Op_i) ∧ StableTypes(Op_i) ∧ BoundedCost(Op_i)
```

## Artifact Contract

Output directory:

```text
runs/executable_graph_abi/<experiment_id>/
```

Required artifacts:

```text
abi_config.json
op_registry.json
type_registry.json
effect_policy.json
verifier_registry.json
cost_model_report.json
admitted_ops.jsonl
rejected_ops.jsonl
compile_eligibility_report.json
quality_report.json
```

`op_registry.json` must record:

```json
{
  "op_id": "stable-op-id",
  "name": "topk_edge_select",
  "inputs": [{"name": "scores", "type": "float32[N]"}],
  "outputs": [{"name": "indices", "type": "int64[K]"}],
  "effect": "pure",
  "cost_model": "O(N log K)",
  "verifier": "topk_membership_and_order_check",
  "provenance": "manual-or-mined-or-distilled",
  "status": "admitted"
}
```

Hard rule:

```text
No op reaches the compiler until it passes ABI admission.
```

## Implementation Plan

### P0.1 — Op Schema

Add:

```text
src/executable_graph_abi.py
tests/test_executable_graph_abi.py
```

Acceptance:

```text
Defines OpContract, TypeSpec, EffectSpec, VerifierSpec, and CostSpec.
Serializes/deserializes op contracts deterministically.
Rejects missing types, missing verifier, and unknown effects.
```

### P0.2 — Effect Policy

Acceptance:

```text
Allows pure ops by default.
Rejects external_action ops.
Marks read_state/write_state as gated but not compile eligible.
Emits rejected_ops.jsonl with reasons.
```

### P0.3 — Compile Eligibility Pass

Acceptance:

```text
Takes admitted ops and emits compile_eligibility_report.json.
Only pure, bounded, stable typed ops are eligible.
```

## Success Criteria

```text
Executable nodes can be represented in the graph without executing arbitrary code.
Pure operations can be admitted and marked compile eligible.
Effectful operations are rejected or gated with explicit reasons.
The next plan can lower eligible op subgraphs into CPU code.
```
