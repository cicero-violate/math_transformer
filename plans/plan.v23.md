# Plan v23 — Qwen Checkpoint Weight Graph Compiler + Sparse Student Bridge

## Official Name

```text
Checkpoint-Compiled Sparse Quantale Student
```

Runtime intent:

```text
large teacher checkpoint weights -> derived graph artifact G_0 -> sparse topology pruning -> graph-native student -> bounded quantale frontier reasoning
```

## Why v23 Exists

v22 implemented the quantale-native closure/planner layer.

v23 moves the project from:

```text
teacher behavior / prompt traces may seed a sparse student
```

to the selected path:

```text
teacher checkpoint weights seed a sparse graph-native student
```

The key correction is:

```text
Do not prompt Qwen to extract graph facts.
Compile Qwen checkpoint tensors into a derived graph artifact.
```

## v22 Completion State

Implemented before v23:

```text
P0 depth sweep formatter repair:
  DepthSweepRow.k supports int | None.
  k=None renders as full.

P0 expert key repair:
  generic -> generic_expert.
  affine -> affine_expert.

P0 L-sweep protocol note:
  multi-L sweeps record single_checkpoint_l_sweep limitation.

P0 repeated protocol harness:
  per-run failures do not abort the loop.
  total_runs / failed_runs / succeeded are printed.
  aggregator writes JSON/CSV summary even when runs fail.

P1.1 QuantaleSpec:
  QuantaleSpec(name, zero, one, join, compose, reduce, better, valid).
  BOOLEAN_SPEC = OR / AND.
  COST_SPEC = min / +.
  UTILITY_SPEC = max / +.
  quantale_closure(adj, h, spec).
  quantale_bounded_closure(node_ids, edges, h, spec).

P1.2 QuantaleFrontierPlanner:
  score_candidates_quantale(...).
  QuantaleFrontierPlanner.expand(...).
  closure_path_count and closure_compute_cost reported.
```

Current test state:

```text
429 passed
```

Important distinction:

```text
Repeated protocol artifact generation is fixed.
Repeated speed proof is not automatically proven.
Distributional speed acceptance remains a gate.
```

## v23 Decision

```text
Use Qwen / dense / MoE checkpoint weights as source structure.
Do not use prompt extraction as the primary distillation path.
Do not run the teacher at student runtime.
Do not store raw transformer weights in graph records.
```

Correct pipeline:

```text
Qwen safetensors checkpoint
  -> tensor manifest
  -> typed tensor parser
  -> block/head/expert graph compiler
  -> derived candidate graph G_0
  -> sparse edge scoring / pruning
  -> sparse quantale graph student
  -> repeated quality/runtime/memory proof gates
```

Rejected pipeline:

```text
Qwen prompts -> text answers -> extracted fact graph
```

Rejected representation:

```text
one raw parameter = one graph edge
```

## Core Objects

```text
Teacher checkpoint = model config + safetensors shards + index metadata.
Tensor manifest = typed inventory of tensors, shapes, dtypes, shards, and parsed module roles.
G_0 = candidate graph derived from tensor structure and tensor statistics.
A = sparse adjacency after learned or heuristic pruning.
Student = graph-native sparse model trained/evaluated under local memory and speed gates.
```

Node types:

| node type | meaning |
|---|---|
| model | checkpoint-level root |
| layer | transformer block index |
| tensor | source tensor metadata |
| head | attention head |
| channel_block | block of hidden dimensions |
| mlp_block | block of intermediate dimensions |
| expert | MoE expert |
| router | MoE router |
| projection | q/k/v/o/gate/up/down/router projection role |

Edge types:

| edge type | source evidence |
|---|---|
| tensor_contains_block | tensor shape / block partition |
| qk_affinity_prior | q_proj / k_proj block interaction approximation |
| value_flow_prior | v_proj / o_proj block flow |
| mlp_gate_flow | gate_proj block score |
| mlp_up_flow | up_proj block score |
| mlp_down_flow | down_proj block score |
| router_to_expert_prior | router tensor statistics |
| expert_flow_prior | expert gate/up/down tensors |
| residual_next | layer order / residual structure |

## Weight-Derived Graph Formula

For a 2D tensor:

```text
W ∈ R^{d_out × d_in}
```

split into blocks:

```text
W_ab ∈ R^{b_out × b_in}
```

score each candidate block edge:

```text
s_ab = ||W_ab||_F / sqrt(|W_ab|)
```

keep sparse block edges:

```text
E_K = {(a,b): b ∈ TopK_b(s_ab)}
```

Candidate graph:

```text
G_0 = (V_block ∪ V_head ∪ V_expert ∪ V_tensor, E_weight_prior, ω_G)
```

Student objective remains:

```text
L = KL(p_teacher || p_student) + α L_task + λ |A| + γ M + β T
```

but the first v23 implementation may train/evaluate against existing task labels before full teacher-logit loss exists.

## Artifact Contract

Output directory:

```text
runs/qwen_weight_graph/<model_id_or_hash>/
```

Required artifacts:

```text
manifest.json
nodes.jsonl
edges.jsonl
stats.json
```

Optional artifacts:

```text
adjacency.npz
topk_edges.csv
tensor_parse_failures.jsonl
```

`manifest.json` must record:

```text
source_model
source_config_hash
source_index_hash
compiler_version
block_size
topk
tensor_count
emitted_node_count
emitted_edge_count
raw_weight_payload_in_graph = false
```

`nodes.jsonl` schema:

```json
{
  "node_id": "hash",
  "type": "channel_block",
  "label": "metadata-only label",
  "layer": 0,
  "module": "self_attn",
  "projection": "q_proj",
  "block_index": 0,
  "range": [0, 64],
  "source_tensor": "model.layers.0.self_attn.q_proj.weight"
}
```

`edges.jsonl` schema:

```json
{
  "edge_id": "hash",
  "src": "hash",
  "rel": "weight_block_flow",
  "dst": "hash",
  "weight": 0.123,
  "score_name": "normalized_frobenius",
  "source_tensor": "model.layers.0.mlp.up_proj.weight",
  "provenance": {
    "shard": "model-00001-of-000NN.safetensors",
    "block_in": 3,
    "block_out": 12
  }
}
```

Hard rule:

```text
No graph record may contain raw tensor payloads.
Only derived statistics, ranges, hashes, and provenance are allowed.
```

## v23 Completion State

Implemented 2026-06-11:

```text
src/qwen_weight_graph.py — P0.1 + P0.2 + P1.1 + P1.2 + P1.3 + P2.1 stub
tests/test_qwen_weight_graph.py — 104 tests, all pass
Total test suite: 533 passed (429 prior + 104 new)
```

Implemented items:

```text
P0.1 Tensor manifest compiler:
  build_tensor_manifest_from_directory — reads safetensors headers only; no tensor data loaded.
  Supports single-file and sharded (model.safetensors.index.json) checkpoints.
  TensorSpec: name, shard, dtype, shape, offsets, layer, module, projection, expert_idx, role, parse_ok.

P0.2 Typed Qwen/MoE tensor parser:
  parse_qwen_tensor_name — recognizes self_attn q/k/v/o projections, mlp gate/up/down,
  MoE router (gate), MoE expert projections, shared_expert, embed_tokens, lm_head,
  final_norm, layernorm variants. Unknown tensors: parse_ok=False, role='unknown'.

P1.1 Block-energy edge compiler:
  compute_block_scores — partitions W into block_size×block_size blocks,
  scores by s_ab = ||W_ab||_F / sqrt(|W_ab|) = RMS norm,
  returns TopK (block_row, block_col, score) per row with deterministic stable tie-break.
  _build_block_graph_for_tensor — emits channel_block/mlp_block nodes and block-flow edges.
  Edge rels: qk_affinity_prior, value_flow_prior, mlp_gate_flow, mlp_up_flow, mlp_down_flow,
             expert_flow_prior, weight_block_flow; tensor_contains_block from projection node.

P1.2 Head/expert structural graph:
  _build_structural_graph — emits model/layer/projection/router/expert nodes;
  contains, residual_next, qk_affinity_prior, value_flow_prior, mlp_gate_flow, mlp_up_flow,
  router_to_expert_prior, expert_flow_prior edges from tensor metadata only (no weight data).

Artifact I/O:
  write_weight_graph_artifacts — manifest.json, nodes.jsonl, edges.jsonl, stats.json,
  optional tensor_parse_failures.jsonl.
  read_weight_graph_artifacts — round-trips all records.
  QwenWeightGraphCompiler — main class; compile() and compile_from_directory() methods.
  SafetensorsTensorLoader — streams one tensor at a time; no full shard in memory.

P1.3 WorldGraph schema adapter:
  load_weight_graph_as_world_graph — loads G_0 artifact into WorldGraph;
  edges with missing nodes silently skipped; opt-in; champion scorer unchanged (Gate 5).

P2.1 Sparse student stub:
  SparseStudentExperimentConfig — config dataclass.
  run_sparse_student_stub — reads G_0 manifest only (no teacher checkpoint at runtime; Gate 4);
  detects Gate 2 violations; returns stub_ok, artifact_missing, or gate_violation status.
```

Gate enforcement in implementation:

```text
Gate 2: raw_weight_payload_in_graph=false enforced in WeightGraphManifest.as_dict();
         run_sparse_student_stub rejects manifests with raw_weight_payload_in_graph=true.
Gate 3: No parameter-level edges; block-level and structural edges only.
Gate 4: Teacher checkpoint not loaded at student runtime; G_0 artifact is self-contained.
Gate 5: Champion scorer and topology eval paths are unaffected; this module is opt-in.
```

Remaining later:

```text
P2.1 full KL distillation training path.
P2.2 Student deletion gate (paired quality/speed/memory metrics).
P3 Head-level node refinement from model config (n_heads derivation).
P3 adjacency.npz optional sparse adjacency artifact.
```

## Implementation Plan (archived)

### P0.1 — Tensor Manifest Compiler

Add:

```text
src/qwen_weight_graph.py
```

Acceptance:

```text
Reads config/index/safetensors metadata.
Emits manifest.json.
Does not load all tensors into memory at once.
Records tensor name, shard, dtype, shape, and parse status.
```

### P0.2 — Typed Qwen Tensor Parser

Parse tensor names into:

```text
layer
module
projection
expert
head_group when derivable
```

Recognize at minimum:

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
router / gate
expert gate/up/down tensors for MoE variants
```

Acceptance:

```text
Unknown tensors are logged, not fatal.
Known dense and MoE naming patterns are covered by tests or fixtures.
```

### P1.1 — Block-Energy Edge Compiler

For each eligible 2D tensor:

```text
partition W into blocks
score by normalized Frobenius norm
emit TopK incoming or outgoing edges per block
```

Acceptance:

```text
Deterministic node/edge counts for fixture checkpoints.
TopK is stable under equal scores by deterministic tie-break.
No raw tensor values are written.
```

### P1.2 — Head/Expert Structural Graph

Emit coarse structural nodes and edges:

```text
layer -> attention heads
layer -> router -> experts
expert -> expert MLP blocks
layer -> next layer residual edge
```

Acceptance:

```text
MoE checkpoint emits expert and router graph.
Dense checkpoint emits attention/MLP graph without expert nodes.
```

### P1.3 — Candidate Graph Schema Adapter

Bridge:

```text
runs/qwen_weight_graph/.../nodes.jsonl, edges.jsonl
  -> current topology/scorer/student input structures
```

Acceptance:

```text
Existing champion scorer behavior is unchanged.
Qwen-weight graph path is opt-in.
Fixture graph can be loaded as G_0 and pruned to TopK adjacency.
```

### P2.1 — Sparse Student Training Stub

First target is a stub-compatible path:

```text
G_0 + existing task labels -> sparse topology/student training experiment
```

Later target:

```text
teacher logits / hidden summaries -> KL / representation distillation
```

Acceptance:

```text
Q_student / baseline >= 0.95 where baseline is explicitly defined.
M_student <= configured memory budget.
T_student reports cold/cache/static regimes separately.
```

## First Valid v23 Experiment

Use a small fixture first, then a real Qwen-style checkpoint.

```text
model_fixture ∈ {tiny_safetensors_fixture, small_qwen_style_checkpoint}
block_size ∈ {32,64,128}
topk ∈ {4,8,16}
graph_scope ∈ {mlp_only, attention_mlp, attention_mlp_moe}
```

Acceptance:

```text
compiler emits valid manifest/nodes/edges/stats
raw_weight_payload_in_graph=false
graph loader round-trips all nodes/edges
TopK adjacency is deterministic
existing test suite remains green
```

First student acceptance remains conservative:

```text
No quality claim until compared against dense, hand K=4, learned K=4, and relevant task baseline.
No speed claim until repeated distributional gate passes.
```

## Non-Negotiable Gates

1. Do not prompt/query Qwen as the primary v23 extraction path.
2. Do not store raw transformer weights in graph records.
3. Do not create parameter-level graph edges.
4. Do not require the large teacher checkpoint at student runtime.
5. Do not change champion scorer behavior for existing topology experiments.
6. Do not claim memory savings from derived graph artifacts unless end-to-end student memory is measured.
7. Do not claim speed proof unless repeated locked distributional artifacts pass.
8. Do not delete/archive student edges without paired quality, speed, and memory gates.
9. Keep quantale planner work inside `math_transformer`.
10. Keep full-`G_world` closure forbidden at runtime; closure is bounded/local only.

## Decision

```text
v23 is the checkpoint-weight graph compiler plan.
The next concrete artifact is not a Qwen prompt dataset.
The next concrete artifact is a derived graph G_0 compiled from Qwen/dense/MoE checkpoint tensors.
```

Summary formula:

```text
Qwen weights -> tensor manifest -> block/head/expert graph -> G_0 -> TopK sparse adjacency -> graph-native sparse student
```