from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.qwen_distillation_harness import run_fixed_topology_distillation_harness
from src.qwen_logit_distillation_targets import (
    validate_frozen_logit_distillation_targets,
    write_frozen_logit_distillation_targets,
)
from src.qwen_random_adjacency_baseline import (
    generate_matched_random_adjacency,
    _build_handoff_dir as _build_random_handoff_dir,
)
from src.qwen_rewire_acceptance import (
    build_rewire_acceptance_report,
    write_rewire_acceptance_report,
)
from src.qwen_rewire_proposal import (
    build_rewire_proposal_report,
    write_rewire_proposal_report,
)
from src.qwen_sparse_student_handoff import (
    load_selected_adjacency,
    validate_selected_adjacency,
)
from src.v25_01_heldout_eval import run_v25_01_heldout_eval


SCHEMA_VERSION = "v26_rewire_k2_cycle.v1"
CYCLE_REPORT_FILENAME = "v26_cycle_report.json"

# Schema versions needed for synthetic edge trace
_EDGE_TRACE_SCHEMA = "qwen_edge_trace.v1"
_UTILITY_SUMMARY_SCHEMA = "qwen_edge_utility_summary.v1"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def build_synthetic_edge_trace_dir(
    adjacency: dict[str, Any],
    output_dir: str | Path,
    *,
    feature_dim: int = 8,
    steps: int = 1,
) -> Path:
    """Write a zero-utility edge trace for `adjacency` so the proposal generator can run.

    The utility scores are 0.0 for all edges — the proposal policy then selects
    swaps purely by edge weight from the weight graph candidate pool.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    edges = adjacency["edges"]
    adjacency_name = str(adjacency["adjacency_name"])
    k = int(adjacency["k"])
    max_out_degree = max(
        (sum(1 for e in edges if str(e["src_id"]) == src)
         for src in {str(e["src_id"]) for e in edges}),
        default=0,
    )
    edge_count = len(edges)
    ranked_edges = [
        {
            "edge_id": str(e["edge_id"]),
            "dst_id": str(e["dst_id"]),
            "utility_score": 0.0,
            "dst_delta_l1_mean": 0.0,
            "message_l1_mean": 0.0,
            "message_l1_max": 0.0,
            "message_l2_mean": 0.0,
            "message_l2_max": 0.0,
            "finite": True,
        }
        for e in edges
    ]
    utility_summary = {
        "schema_version": _UTILITY_SUMMARY_SCHEMA,
        "row_count": edge_count,
        "finite": True,
        "ranked_edges": ranked_edges,
    }
    report = {
        "schema_version": _EDGE_TRACE_SCHEMA,
        "status": "edge_trace_ok",
        "adjacency_name": adjacency_name,
        "k": k,
        "max_out_degree": max_out_degree,
        "feature_dim": feature_dim,
        "steps": steps,
        "edge_count": edge_count,
        "row_count": edge_count,
        "expected_row_count": edge_count,
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "topology_mutated": False,
        "finite": True,
        "edge_utility_summary": utility_summary,
        "note": "synthetic zero-utility trace for v26 proposal seeding; utilities supplied by weight-graph score",
    }
    _write_json(out / "edge_trace_report.json", report)
    # Write one row per edge so the row_count matches (proposal generator only reads the report)
    _write_jsonl(out / "edge_trace.jsonl", [{"edge_id": e["edge_id"], "utility_score": 0.0} for e in edges])
    return out


def _build_random_baseline_dir(
    baseline_eval_dir: Path,
    baseline_adjacency: dict[str, Any],
    output_dir: Path,
    seed: int = 0,
) -> dict[str, Any]:
    random_adjacency = generate_matched_random_adjacency(baseline_adjacency, seed=seed)
    random_summary = validate_selected_adjacency(random_adjacency)
    baseline_summary = validate_selected_adjacency(baseline_adjacency)
    _build_random_handoff_dir(
        output_dir,
        qwen_adjacency=baseline_adjacency,
        random_adjacency=random_adjacency,
        qwen_summary=baseline_summary,
        random_summary=random_summary,
        source_eval_dir=baseline_eval_dir,
    )
    return {
        "random_adjacency_name": random_summary["adjacency_name"],
        "random_eval_dir": str(output_dir),
        "random_seed": seed,
    }


def run_v26_rewire_cycle(
    qwen_eval_dir: str | Path,
    output_dir: str | Path,
    teacher_artifacts: str | Path,
    *,
    baseline_k: int = 2,
    baseline_adjacency_name: str = "qwen_topk_k2",
    max_swaps: int = 4,
    proposal_policy: str = "same_source_top_weight",
    policy_seed: int = 0,
    random_seed: int = 0,
    vocab_size: int = 16,
    target_seeds: list[int] | None = None,
    feature_dim: int = 8,
    forward_steps: int = 1,
    train_steps: int = 20,
    lr: float = 0.1,
    projection_seed: int = 0,
    temperature: float = 1.0,
    device: str = "cpu",
    held_out_per_family: int = 8,
    heldout_train_steps: int = 128,
    heldout_lr: float = 0.5,
    heldout_split_seed: int = 0,
    max_kl_regression: float = 0.0,
) -> dict[str, Any]:
    """Full v26 rewiring cycle: proposal → KL gate → random gate → heldout gate → decision.

    Gates:
      quality_ok   — candidate KL after <= baseline KL after (no regression)
      kl_ok        — candidate beats baseline AND beats random on KL after training
      heldout_ok   — heldout text eval generalizes (confirms text-path student stability)
      promote      — all three gates pass
    """
    base_eval_dir = Path(qwen_eval_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = target_seeds if target_seeds is not None else list(range(8))

    # ------------------------------------------------------------------
    # Shared frozen logit targets (same targets used for all KL runs)
    # ------------------------------------------------------------------
    targets_dir = out / "shared_logit_targets"
    write_frozen_logit_distillation_targets(
        targets_dir, vocab_size=vocab_size, seeds=seeds, temperature=temperature
    )
    validate_frozen_logit_distillation_targets(targets_dir)

    # ------------------------------------------------------------------
    # Baseline adjacency
    # ------------------------------------------------------------------
    baseline_adjacency = load_selected_adjacency(
        base_eval_dir, adjacency_name=baseline_adjacency_name
    )
    baseline_summary = validate_selected_adjacency(baseline_adjacency)

    # ------------------------------------------------------------------
    # Step 1 — Synthetic edge trace for baseline adjacency
    # ------------------------------------------------------------------
    trace_dir = build_synthetic_edge_trace_dir(
        baseline_adjacency,
        out / "edge_trace",
        feature_dim=feature_dim,
        steps=forward_steps,
    )

    # ------------------------------------------------------------------
    # Step 2 — Rewiring proposal (bounded edge swaps)
    # ------------------------------------------------------------------
    proposal_dir = out / "proposal"
    proposal_report = build_rewire_proposal_report(
        base_eval_dir,
        trace_dir,
        adjacency_name=baseline_adjacency_name,
        max_swaps=max_swaps,
        proposal_policy=proposal_policy,
        policy_seed=policy_seed,
    )
    write_rewire_proposal_report(proposal_report, proposal_dir)

    # ------------------------------------------------------------------
    # Step 3 — KL acceptance gate: candidate vs baseline
    # ------------------------------------------------------------------
    acceptance_dir = out / "acceptance"
    acceptance_report = build_rewire_acceptance_report(
        eval_output_dir=base_eval_dir,
        rewire_proposal_dir=proposal_dir,
        output_dir=acceptance_dir,
        k=baseline_k,
        logit_targets_dir=targets_dir,
        vocab_size=vocab_size,
        target_seeds=seeds,
        feature_dim=feature_dim,
        forward_steps=forward_steps,
        train_steps=train_steps,
        lr=lr,
        projection_seed=projection_seed,
        temperature=temperature,
        device=device,
        max_kl_regression=max_kl_regression,
    )
    write_rewire_acceptance_report(acceptance_report, acceptance_dir)

    kl_baseline_after = float(acceptance_report["base_kl_final"])
    kl_candidate_after = float(acceptance_report["candidate_kl_final"])
    quality_ok = bool(acceptance_report["quality_ok"])
    swap_count = int(proposal_report["swap_count"])

    # ------------------------------------------------------------------
    # Step 4 — Random baseline: matched random k=2 vs baseline
    # ------------------------------------------------------------------
    random_dir = out / "random_baseline_dir"
    random_meta = _build_random_baseline_dir(
        base_eval_dir, baseline_adjacency, random_dir, seed=random_seed
    )
    random_adj_name = random_meta["random_adjacency_name"]

    random_harness_report = run_fixed_topology_distillation_harness(
        random_dir,
        out / "random_harness",
        adjacency_name=random_adj_name,
        logit_targets_dir=targets_dir,
        vocab_size=vocab_size,
        target_seeds=seeds,
        feature_dim=feature_dim,
        forward_steps=forward_steps,
        train_steps=train_steps,
        lr=lr,
        projection_seed=projection_seed,
        temperature=temperature,
        device=device,
    )
    kl_random_after = float(random_harness_report["kl_after"])

    # ------------------------------------------------------------------
    # Step 5 — Heldout text eval (generalization gate)
    # ------------------------------------------------------------------
    heldout_report = run_v25_01_heldout_eval(
        teacher_artifacts,
        out / "heldout",
        held_out_per_family=held_out_per_family,
        train_steps=heldout_train_steps,
        lr=heldout_lr,
        split_seed=heldout_split_seed,
    )

    # ------------------------------------------------------------------
    # Step 6 — Tripartite gate decision
    # ------------------------------------------------------------------
    kl_ok = (kl_candidate_after < kl_baseline_after) and (kl_candidate_after < kl_random_after)
    heldout_ok = bool(heldout_report["heldout_generalizes"])
    promote = quality_ok and kl_ok and heldout_ok

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "v26_rewire_k2_cycle_ok",
        "baseline_adjacency": baseline_adjacency_name,
        "k": baseline_k,
        "proposal_policy": proposal_policy,
        "max_swaps": max_swaps,
        "swap_count": swap_count,
        "train_steps": train_steps,
        "vocab_size": vocab_size,
        "feature_dim": feature_dim,
        "forward_steps": forward_steps,
        "random_seed": random_seed,
        # KL values
        "kl_baseline_after": kl_baseline_after,
        "kl_candidate_after": kl_candidate_after,
        "kl_random_after": kl_random_after,
        "kl_delta_vs_baseline": kl_candidate_after - kl_baseline_after,
        "kl_delta_vs_random": kl_candidate_after - kl_random_after,
        "candidate_beats_baseline": kl_candidate_after < kl_baseline_after,
        "candidate_beats_random": kl_candidate_after < kl_random_after,
        # Heldout
        "heldout_loss_mean": float(heldout_report["heldout_loss_mean"]),
        "generalization_gap": float(heldout_report["generalization_gap"]),
        # Gates
        "quality_ok": quality_ok,
        "kl_ok": kl_ok,
        "heldout_ok": heldout_ok,
        "promote": promote,
        "decision": "candidate_promoted" if promote else "candidate_not_promoted",
        # Safety
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "topology_mutated": False,
        "finite": (
            math.isfinite(kl_baseline_after)
            and math.isfinite(kl_candidate_after)
            and math.isfinite(kl_random_after)
            and math.isfinite(heldout_report["heldout_loss_mean"])
        ),
        "note": (
            "v26 cycle 1: proposal via " + proposal_policy + "; "
            "gates: quality_ok (no KL regression) AND kl_ok (beats baseline AND random) "
            "AND heldout_ok (text generalizes); promote=true only if all three pass"
        ),
    }
    _write_json(out / CYCLE_REPORT_FILENAME, report)
    return report
