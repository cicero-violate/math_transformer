# math_transformer handoff — v26 candidate promotion next session

Working directory:

```text
/workspace/ai_sandbox/canon-mini-agent/prototype/neurosymbolic/math_transformer
```

Project:

```text
Recurrent Frontier Graph Transformer / math_transformer
```

Branch:

```text
main
```

Remote:

```text
origin https://github.com/cicero-violate/math_transformer.git
```

Latest pushed commits:

```text
ed3c4cf Add v26 accepted candidate apply artifact
47931f0 Add v26 richer rewiring policies
bda0c61 Add v26 rewiring proposal search
e800467 Add v26 rewiring acceptance gate
8bbc773 Add v26 bounded rewiring proposals
50a2a30 Add v26 edge trace collection
dbd2206 Add sparse backend comparison CLI
c864ad0 Add CUDA-aware measured gate metadata
57bab86 Add device-aware sparse student backend
940daa3 Add measured distillation pipeline CLI
```

Expected current git state:

```text
git status --short
# should be clean except for this handoff file if not committed
```

## Current verified state

v25 measured distillation pipeline is green.

Known successful measured pipeline command:

```bash
python -m src.qwen_distillation_cli \
  --source-weight-graph-dir runs/qwen_weight_graph/qwen_style_tiny \
  --output-root runs/qwen_distillation_harness/qwen_style_tiny/v25_measured_torch_cpu \
  --k 1 \
  --k-values 1,2 \
  --random-seeds 0,1,2,3,4 \
  --vocab-size 16 \
  --target-seeds 0,1,2 \
  --feature-dim 8 \
  --forward-steps 1 \
  --train-steps 5 \
  --lr 0.1 \
  --runtime-repeats 2 \
  --max-runtime-seconds 10.0 \
  --max-peak-memory-bytes 134217728 \
  --max-cuda-peak-memory-bytes 134217728 \
  --device torch_cpu
```

Observed output:

```json
{
  "status": "measured_distillation_pipeline_ok",
  "device": "torch_cpu",
  "resolved_device": "cpu",
  "runtime_backend": "torch",
  "cuda_measurement_available": false,
  "quality_ok": true,
  "runtime_ok": true,
  "memory_ok": true,
  "safety_ok": true,
  "promote": true,
  "decision": "promoted",
  "missing_or_failed_gates": []
}
```

CUDA note:

```text
--device cuda fails clearly on this host because torch.cuda.is_available() is false.
```

## Core invariant

```text
Mine dense checkpoints offline.
Run sparse students online.
G_pool may grow.
Active adjacency A must stay bounded.
No teacher checkpoint at student runtime.
No raw weight payload in graph artifacts.
No champion scorer mutation.
No topology mutation unless promotion gates pass.
Promotion only passes when real measured gates pass.
```

## v26 progress

### P0 — edge trace collection

Implemented:

```text
src/qwen_edge_trace.py
src/qwen_edge_trace_cli.py
tests/test_qwen_edge_trace.py
```

Command:

```bash
python -m src.qwen_edge_trace_cli \
  --eval-output-dir runs/qwen_distillation_harness/qwen_style_tiny/v25_measured_torch_cpu/graph_prior_eval \
  --output-dir runs/qwen_edge_trace/qwen_style_tiny/v26_p0 \
  --k 1 \
  --feature-dim 8 \
  --steps 1 \
  --seeds 0,1,2 \
  --device torch_cpu
```

Verified output:

```json
{
  "adjacency_name": "qwen_topk_k1",
  "bounded_active_adjacency": true,
  "device": "torch_cpu",
  "edge_count": 32,
  "row_count": 96,
  "topology_mutated": false
}
```

Artifacts:

```text
runs/qwen_edge_trace/qwen_style_tiny/v26_p0/edge_trace_report.json
runs/qwen_edge_trace/qwen_style_tiny/v26_p0/edge_trace.jsonl
runs/qwen_edge_trace/qwen_style_tiny/v26_p0/edge_utility_summary.json
```

### P1 — bounded rewiring proposals

Implemented:

```text
src/qwen_rewire_proposal.py
src/qwen_rewire_proposal_cli.py
tests/test_qwen_rewire_proposal.py
```

Command:

```bash
python -m src.qwen_rewire_proposal_cli \
  --eval-output-dir runs/qwen_distillation_harness/qwen_style_tiny/v25_measured_torch_cpu/graph_prior_eval \
  --edge-trace-dir runs/qwen_edge_trace/qwen_style_tiny/v26_p0 \
  --output-dir runs/qwen_rewire_proposal/qwen_style_tiny/v26_p1 \
  --k 1 \
  --max-swaps 3
```

Verified output:

```json
{
  "accepted": false,
  "base_adjacency_name": "qwen_topk_k1",
  "base_edge_count": 32,
  "k": 1,
  "proposal_bounded": true,
  "proposed_edge_count": 32,
  "proposed_max_out_degree": 1,
  "swap_count": 3,
  "topology_mutated": false
}
```

### P2 — accept/reject gate

Implemented:

```text
src/qwen_rewire_acceptance.py
src/qwen_rewire_acceptance_cli.py
tests/test_qwen_rewire_acceptance.py
```

Command:

```bash
python -m src.qwen_rewire_acceptance_cli \
  --eval-output-dir runs/qwen_distillation_harness/qwen_style_tiny/v25_measured_torch_cpu/graph_prior_eval \
  --rewire-proposal-dir runs/qwen_rewire_proposal/qwen_style_tiny/v26_p1 \
  --output-dir runs/qwen_rewire_acceptance/qwen_style_tiny/v26_p2 \
  --k 1 \
  --vocab-size 16 \
  --target-seeds 0,1,2 \
  --feature-dim 8 \
  --forward-steps 1 \
  --train-steps 5 \
  --lr 0.1 \
  --device torch_cpu \
  --max-kl-regression 0.0
```

Verified result for P1 proposal:

```json
{
  "accepted": false,
  "base_kl_final": 0.4901987112836898,
  "candidate_kl_final": 0.4947830313900639,
  "candidate_minus_base_kl_final": 0.004584320106374107,
  "decision": "rejected",
  "proposal_applied": false,
  "quality_ok": false,
  "safety_ok": true
}
```

### P3 — proposal search loop

Implemented:

```text
src/qwen_rewire_search.py
src/qwen_rewire_search_cli.py
tests/test_qwen_rewire_search.py
```

Initial P3 searched only swap counts and found no accepted candidate:

```json
{
  "accepted_candidate_count": 0,
  "candidate_count": 3,
  "decision": "no_candidate_accepted",
  "proposal_applied": false,
  "topology_mutated": false
}
```

### P4 — richer proposal policies

Implemented by extending:

```text
src/qwen_rewire_proposal.py
src/qwen_rewire_search.py
tests/test_qwen_rewire_proposal.py
tests/test_qwen_rewire_search.py
```

Supported deterministic policies:

```text
same_source_top_weight
same_source_low_weight
same_relation_top_weight
utility_ratio
deterministic_random
```

Verified command:

```bash
python -m src.qwen_rewire_search_cli \
  --eval-output-dir runs/qwen_distillation_harness/qwen_style_tiny/v25_measured_torch_cpu/graph_prior_eval \
  --edge-trace-dir runs/qwen_edge_trace/qwen_style_tiny/v26_p0 \
  --output-dir runs/qwen_rewire_search/qwen_style_tiny/v26_p4 \
  --k 1 \
  --max-swaps-values 1,2 \
  --proposal-policies same_source_top_weight,same_source_low_weight,deterministic_random \
  --policy-seed 7 \
  --vocab-size 16 \
  --target-seeds 0,1,2 \
  --feature-dim 8 \
  --forward-steps 1 \
  --train-steps 5 \
  --lr 0.1 \
  --device torch_cpu \
  --max-kl-regression 0.0
```

Verified output:

```json
{
  "accepted_candidate_count": 1,
  "best_candidate_index": 4,
  "best_candidate_kl_delta": -0.03418879930089158,
  "candidate_count": 6,
  "decision": "accepted_candidate_found",
  "proposal_applied": false,
  "selected_candidate_accepted": true,
  "selected_candidate_index": 4,
  "selected_candidate_kl_delta": -0.03418879930089158,
  "topology_mutated": false
}
```

Known accepted candidate:

```json
{
  "selected_candidate_index": 4,
  "selected_candidate_policy": "deterministic_random",
  "selected_candidate_kl_delta": -0.03418879930089158,
  "candidate_adjacency_name": "qwen_topk_k1_v26_candidate"
}
```

### P5 — accepted-candidate apply artifact

Implemented:

```text
src/qwen_rewire_apply.py
src/qwen_rewire_apply_cli.py
tests/test_qwen_rewire_apply.py
```

Command:

```bash
python -m src.qwen_rewire_apply_cli \
  --rewire-search-dir runs/qwen_rewire_search/qwen_style_tiny/v26_p4 \
  --output-dir runs/qwen_rewire_apply/qwen_style_tiny/v26_p5 \
  --overwrite
```

Verified output:

```json
{
  "accepted_candidate_manifest": "runs/qwen_rewire_apply/qwen_style_tiny/v26_p5/accepted_candidate_manifest.json",
  "active_topology_mutated": false,
  "applied_candidate_eval_dir": "runs/qwen_rewire_apply/qwen_style_tiny/v26_p5/applied_candidate_eval",
  "base_topology_mutated": false,
  "candidate_adjacency_name": "qwen_topk_k1_v26_candidate",
  "candidate_materialized": true,
  "proposal_applied_to_base": false,
  "selected_candidate_index": 4,
  "selected_candidate_kl_delta": -0.03418879930089158,
  "selected_candidate_policy": "deterministic_random"
}
```

Important P5 artifact:

```text
runs/qwen_rewire_apply/qwen_style_tiny/v26_p5/accepted_candidate_manifest.json
runs/qwen_rewire_apply/qwen_style_tiny/v26_p5/applied_candidate_eval
```

## Latest full validation

After P5:

```text
python -m pytest -q
678 passed, 1 skipped
```

## Next task for new session

Implement:

```text
v26 P6: promote accepted candidate as next sparse prior
```

Suggested files:

```text
src/qwen_rewire_candidate_promotion.py
src/qwen_rewire_candidate_promotion_cli.py
tests/test_qwen_rewire_candidate_promotion.py
```

Input accepted candidate dir:

```text
runs/qwen_rewire_apply/qwen_style_tiny/v26_p5
```

Candidate eval handoff:

```text
runs/qwen_rewire_apply/qwen_style_tiny/v26_p5/applied_candidate_eval
```

Goal:
Run the measured fixed-topology distillation and promotion path against the materialized accepted candidate handoff, then write:

```text
candidate_distillation_harness_report.json
candidate_measured_gate_report.json
candidate_promotion_decision.json
candidate_next_prior_manifest.json
```

Suggested command shape:

```bash
python -m src.qwen_rewire_candidate_promotion_cli \
  --accepted-candidate-dir runs/qwen_rewire_apply/qwen_style_tiny/v26_p5 \
  --output-dir runs/qwen_rewire_candidate_promotion/qwen_style_tiny/v26_p6 \
  --k 1 \
  --vocab-size 16 \
  --target-seeds 0,1,2 \
  --feature-dim 8 \
  --forward-steps 1 \
  --train-steps 5 \
  --lr 0.1 \
  --runtime-repeats 2 \
  --max-runtime-seconds 10.0 \
  --max-peak-memory-bytes 134217728 \
  --max-cuda-peak-memory-bytes 134217728 \
  --device torch_cpu
```

Candidate promotion rule:

```text
candidate_promoted =
  quality_ok
  and runtime_ok
  and memory_ok
  and safety_ok
  and promote == true
  and candidate_materialized == true
  and base_topology_mutated == false
  and active_topology_mutated == false
  and proposal_applied_to_base == false
  and bounded_active_adjacency == true
```

If true:

```text
decision = "candidate_promoted_as_next_prior"
```

Else:

```text
decision = "candidate_not_promoted"
```

Do not mutate the original base topology.
Do not overwrite the original v25/v26 handoff.
Only write a candidate next-prior manifest.

Required public API for P6:

```text
build_candidate_next_prior_manifest
write_candidate_next_prior_manifest
load_candidate_next_prior_manifest
validate_candidate_next_prior_manifest
run_and_write_candidate_promotion_report
```

Reuse existing modules where possible:

```text
src.qwen_distillation_harness.run_fixed_topology_distillation_harness
src.qwen_distillation_measured_gates.run_and_write_measured_distillation_gate_report
src.qwen_distillation_promotion.run_and_write_distillation_promotion_decision
src.qwen_rewire_apply.load_accepted_candidate_manifest
src.qwen_rewire_apply.validate_accepted_candidate_manifest
```

Validation target:

```bash
python -m pytest -q tests/test_qwen_rewire_candidate_promotion.py
python -m pytest -q
```

Then commit:

```bash
git status --short
git add src/qwen_rewire_candidate_promotion.py src/qwen_rewire_candidate_promotion_cli.py tests/test_qwen_rewire_candidate_promotion.py
git commit -m 'Add v26 accepted candidate promotion'
git push origin main
```
