from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.qwen_edge_trace import load_edge_trace_report
from src.qwen_rewire_recursive_bootstrap import validate_recursive_bootstrap_manifest
from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency


SCHEMA_VERSION = "adaptive_rewire_contract.v1"
CONTRACT_MANIFEST_FILENAME = "adaptive_rewire_contract_manifest.json"
REQUIRED_ARTIFACT_FILENAMES = {
    "rewire_config": "rewire_config.json",
    "initial_adjacency": "initial_adjacency.json",
    "edge_utility": "edge_utility.jsonl",
    "proposal_batch": "proposal_batch.jsonl",
    "accepted_rewrites": "accepted_rewrites.jsonl",
    "rejected_rewrites": "rejected_rewrites.jsonl",
    "closure_preservation_report": "closure_preservation_report.jsonl",
    "old_domain_regression_report": "old_domain_regression_report.json",
    "rewire_iteration_metrics": "rewire_iteration_metrics.jsonl",
    "final_adjacency": "final_adjacency.json",
    "runtime_report": "runtime_report.json",
    "memory_report": "memory_report.json",
    "quality_report": "quality_report.json",
}
OPTIONAL_ARTIFACT_FILENAMES = {
    "rollback_manifest": "rollback_manifest.json",
    "candidate_edge_pool": "candidate_edge_pool.jsonl",
    "edge_tombstones": "edge_tombstones.jsonl",
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def load_recursive_bootstrap_manifest_path(path: str | Path) -> dict[str, Any]:
    return _read_json(Path(path))


def load_adaptive_rewire_contract_manifest(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / CONTRACT_MANIFEST_FILENAME)


def validate_contract_source_bootstrap_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != "qwen_rewire_recursive_bootstrap.v1":
        raise ValueError(f"bad recursive bootstrap schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "recursive_bootstrap_manifest_ok":
        raise ValueError(f"bad recursive bootstrap status={manifest.get('status')!r}")
    for key in ("recursive_seed_ready", "next_cycle_input_ready", "edge_trace_ready", "bounded_active_adjacency"):
        if not bool(manifest.get(key)):
            raise ValueError(f"{key} must be true")
    for key in (
        "base_topology_mutated",
        "active_topology_mutated",
        "proposal_applied_to_base",
        "teacher_checkpoint_loaded",
        "teacher_inference_runtime_required",
        "raw_weight_payload_in_graph",
    ):
        if bool(manifest.get(key)):
            raise ValueError(f"{key} must be false")
    return validate_recursive_bootstrap_manifest(manifest)


def _required_artifacts_present(output_dir: Path) -> bool:
    return all((output_dir / filename).exists() for filename in REQUIRED_ARTIFACT_FILENAMES.values())


def _noop_row(stream_name: str, *, cycle_index: int, reason: str = "p9_contract_only_no_rewrite_applied") -> dict[str, Any]:
    return {
        "schema_version": "adaptive_rewire_contract_stream_row.v1",
        "stream": stream_name,
        "cycle_index": cycle_index,
        "status": "noop_contract_row",
        "noop": True,
        "topology_mutated": False,
        "reason": reason,
    }


def _edge_utility_rows(edge_trace_report: dict[str, Any], *, cycle_index: int) -> list[dict[str, Any]]:
    utility = edge_trace_report.get("edge_utility_summary")
    if not isinstance(utility, dict):
        raise ValueError("edge trace report missing edge_utility_summary")
    ranked_edges = utility.get("ranked_edges")
    if not isinstance(ranked_edges, list):
        raise ValueError("edge utility ranked_edges must be a list")
    rows: list[dict[str, Any]] = []
    for rank, edge in enumerate(ranked_edges):
        if not isinstance(edge, dict):
            raise ValueError("edge utility ranked edge must be an object")
        rows.append({
            "schema_version": "adaptive_rewire_edge_utility.v1",
            "cycle_index": cycle_index,
            "rank": rank,
            "edge_id": edge.get("edge_id"),
            "src_id": edge.get("src_id"),
            "dst_id": edge.get("dst_id"),
            "relation": edge.get("relation"),
            "score_name": edge.get("score_name"),
            "utility_score": float(edge.get("utility_score", 0.0)),
            "message_l1_mean": float(edge.get("message_l1_mean", 0.0)),
            "message_l2_mean": float(edge.get("message_l2_mean", 0.0)),
            "trace_count": int(edge.get("trace_count", 0)),
            "gradient_stats_available": False,
            "topology_mutated": False,
            "source": "qwen_edge_trace_utility_projection",
        })
    return rows


def _build_rewire_config(
    *,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    student_id: str,
    experiment_id: str,
    edge_budget: int,
    proposal_budget: int,
    closure_horizon: int,
    acceptance_policy: str,
    old_domain_regression_budget: float,
    rollback_enabled: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "adaptive_rewire_config.v1",
        "status": "adaptive_rewire_config_ok",
        "source_recursive_bootstrap_manifest": str(source_manifest_path),
        "source_next_sparse_prior_manifest": source_manifest.get("source_next_sparse_prior_manifest"),
        "source_distill_run": None,
        "source_weight_graph_dir": None,
        "student_id": student_id,
        "experiment_id": experiment_id,
        "cycle_index": int(source_manifest["cycle_index"]),
        "initial_adjacency": REQUIRED_ARTIFACT_FILENAMES["initial_adjacency"],
        "candidate_pool": OPTIONAL_ARTIFACT_FILENAMES["candidate_edge_pool"],
        "edge_budget": int(edge_budget),
        "proposal_budget": int(proposal_budget),
        "closure_horizon": int(closure_horizon),
        "acceptance_policy": acceptance_policy,
        "rollback_enabled": bool(rollback_enabled),
        "old_domain_regression_budget": float(old_domain_regression_budget),
        "contract_only": True,
        "topology_mutated": False,
    }


def build_adaptive_rewire_contract_manifest(
    *,
    recursive_bootstrap_manifest: dict[str, Any],
    recursive_bootstrap_manifest_path: str | Path,
    output_dir: str | Path,
    student_id: str,
    experiment_id: str,
) -> dict[str, Any]:
    validate_contract_source_bootstrap_manifest(recursive_bootstrap_manifest)
    out = Path(output_dir)
    initial = _read_json(out / REQUIRED_ARTIFACT_FILENAMES["initial_adjacency"])
    final = _read_json(out / REQUIRED_ARTIFACT_FILENAMES["final_adjacency"])
    final_equals_initial = final == initial
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "adaptive_rewire_contract_ok",
        "source_recursive_bootstrap_manifest": str(Path(recursive_bootstrap_manifest_path)),
        "output_dir": str(out),
        "student_id": student_id,
        "experiment_id": experiment_id,
        "cycle_index": int(recursive_bootstrap_manifest["cycle_index"]),
        "contract_only": True,
        "topology_mutated": False,
        "final_equals_initial": final_equals_initial,
        "edge_trace_ready": True,
        "canonical_artifacts_ready": True,
        "required_artifacts_present": _required_artifacts_present(out),
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "next_stage": "edge_utility_aggregator",
        "artifacts": {
            **REQUIRED_ARTIFACT_FILENAMES,
            **OPTIONAL_ARTIFACT_FILENAMES,
            "adaptive_rewire_contract_manifest": CONTRACT_MANIFEST_FILENAME,
        },
        "note": "P9 normalizes a verified recursive bootstrap into the plan.v26 adaptive_rewire artifact contract without applying rewrites.",
    }
    validate_adaptive_rewire_contract_manifest(manifest)
    return manifest


def validate_adaptive_rewire_contract_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad adaptive rewire contract schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "adaptive_rewire_contract_ok":
        raise ValueError(f"bad adaptive rewire contract status={manifest.get('status')!r}")
    for key, expected in {
        "contract_only": True,
        "topology_mutated": False,
        "final_equals_initial": True,
        "edge_trace_ready": True,
        "canonical_artifacts_ready": True,
        "required_artifacts_present": True,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
    }.items():
        if bool(manifest.get(key)) is not expected:
            raise ValueError(f"adaptive rewire contract manifest {key} must be {expected}")
    out = Path(str(manifest.get("output_dir")))
    for filename in REQUIRED_ARTIFACT_FILENAMES.values():
        if not (out / filename).exists():
            raise ValueError(f"required adaptive rewire artifact missing: {filename}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("adaptive rewire contract artifacts must be an object")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "adaptive_rewire_contract_manifest_valid",
        "student_id": manifest["student_id"],
        "experiment_id": manifest["experiment_id"],
        "canonical_artifacts_ready": True,
        "required_artifacts_present": True,
    }


def write_adaptive_rewire_contract_manifest(manifest: dict[str, Any], output_dir: str | Path) -> None:
    validate_adaptive_rewire_contract_manifest(manifest)
    _write_json(Path(output_dir) / CONTRACT_MANIFEST_FILENAME, manifest)


def run_and_write_adaptive_rewire_contract(
    *,
    recursive_bootstrap_manifest: str | Path,
    output_dir: str | Path,
    student_id: str,
    experiment_id: str,
    edge_budget: int,
    proposal_budget: int,
    closure_horizon: int,
    acceptance_policy: str,
    old_domain_regression_budget: float,
    rollback_enabled: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if edge_budget < 1:
        raise ValueError(f"edge_budget must be >= 1, got {edge_budget!r}")
    if proposal_budget < 0:
        raise ValueError(f"proposal_budget must be >= 0, got {proposal_budget!r}")
    if closure_horizon < 0:
        raise ValueError(f"closure_horizon must be >= 0, got {closure_horizon!r}")
    source_manifest_path = Path(recursive_bootstrap_manifest)
    source_manifest = load_recursive_bootstrap_manifest_path(source_manifest_path)
    validate_contract_source_bootstrap_manifest(source_manifest)

    out = Path(output_dir)
    if out.exists():
        if not overwrite:
            raise ValueError(f"adaptive rewire output dir already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    adjacency_name = str(source_manifest["cycle_seed_adjacency_name"])
    seed_eval_dir = Path(str(source_manifest["cycle_seed_eval_dir"]))
    adjacency = load_selected_adjacency(seed_eval_dir, adjacency_name=adjacency_name)
    adjacency_summary = validate_selected_adjacency(adjacency)
    if int(adjacency_summary["edge_count"]) > int(edge_budget):
        raise ValueError("edge_budget must cover the current active adjacency edge_count")
    _write_json(out / REQUIRED_ARTIFACT_FILENAMES["initial_adjacency"], adjacency)
    _write_json(out / REQUIRED_ARTIFACT_FILENAMES["final_adjacency"], adjacency)

    edge_trace_report = load_edge_trace_report(source_manifest["cycle_edge_trace_dir"])
    edge_rows = _edge_utility_rows(edge_trace_report, cycle_index=int(source_manifest["cycle_index"]))
    _write_jsonl(out / REQUIRED_ARTIFACT_FILENAMES["edge_utility"], edge_rows)

    cycle_index = int(source_manifest["cycle_index"])
    for key in (
        "proposal_batch",
        "accepted_rewrites",
        "rejected_rewrites",
        "closure_preservation_report",
        "rewire_iteration_metrics",
    ):
        _write_jsonl(out / REQUIRED_ARTIFACT_FILENAMES[key], [_noop_row(key, cycle_index=cycle_index)])
    _write_jsonl(out / OPTIONAL_ARTIFACT_FILENAMES["candidate_edge_pool"], [_noop_row("candidate_edge_pool", cycle_index=cycle_index)])
    _write_jsonl(out / OPTIONAL_ARTIFACT_FILENAMES["edge_tombstones"], [_noop_row("edge_tombstones", cycle_index=cycle_index)])

    rewire_config = _build_rewire_config(
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        student_id=student_id,
        experiment_id=experiment_id,
        edge_budget=edge_budget,
        proposal_budget=proposal_budget,
        closure_horizon=closure_horizon,
        acceptance_policy=acceptance_policy,
        old_domain_regression_budget=old_domain_regression_budget,
        rollback_enabled=rollback_enabled,
    )
    _write_json(out / REQUIRED_ARTIFACT_FILENAMES["rewire_config"], rewire_config)
    _write_json(out / OPTIONAL_ARTIFACT_FILENAMES["rollback_manifest"], {
        "schema_version": "adaptive_rewire_rollback_manifest.v1",
        "status": "rollback_manifest_ok",
        "rollback_enabled": bool(rollback_enabled),
        "contract_only": True,
        "topology_mutated": False,
        "rollback_target": REQUIRED_ARTIFACT_FILENAMES["initial_adjacency"],
    })
    _write_json(out / REQUIRED_ARTIFACT_FILENAMES["old_domain_regression_report"], {
        "schema_version": "adaptive_rewire_old_domain_regression_report.v1",
        "status": "old_domain_regression_contract_ok",
        "old_domain_regression_budget": float(old_domain_regression_budget),
        "measured_regression": 0.0,
        "regression_ok": True,
        "contract_only": True,
        "topology_mutated": False,
    })
    _write_json(out / REQUIRED_ARTIFACT_FILENAMES["runtime_report"], {
        "schema_version": "adaptive_rewire_runtime_report.v1",
        "status": "runtime_contract_ok",
        "teacher_inference_runtime_required": False,
        "contract_only": True,
        "topology_mutated": False,
        "source_edge_trace_row_count": int(edge_trace_report["row_count"]),
    })
    _write_json(out / REQUIRED_ARTIFACT_FILENAMES["memory_report"], {
        "schema_version": "adaptive_rewire_memory_report.v1",
        "status": "memory_contract_ok",
        "raw_weight_payload_in_graph": False,
        "contract_only": True,
        "topology_mutated": False,
    })
    _write_json(out / REQUIRED_ARTIFACT_FILENAMES["quality_report"], {
        "schema_version": "adaptive_rewire_quality_report.v1",
        "status": "quality_contract_ok",
        "edge_trace_ready": True,
        "gradient_stats_available": False,
        "contract_only": True,
        "topology_mutated": False,
        "final_equals_initial": True,
    })

    manifest = build_adaptive_rewire_contract_manifest(
        recursive_bootstrap_manifest=source_manifest,
        recursive_bootstrap_manifest_path=source_manifest_path,
        output_dir=out,
        student_id=student_id,
        experiment_id=experiment_id,
    )
    write_adaptive_rewire_contract_manifest(manifest, out)
    return {
        "status": "adaptive_rewire_contract_run_ok",
        "adaptive_rewire_contract_manifest": str(out / CONTRACT_MANIFEST_FILENAME),
        "output_dir": str(out),
        "student_id": student_id,
        "experiment_id": experiment_id,
        "contract_only": True,
        "topology_mutated": False,
        "final_equals_initial": True,
        "canonical_artifacts_ready": True,
        "required_artifacts_present": True,
        "bounded_active_adjacency": True,
    }


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {raw!r}")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the canonical plan.v26 adaptive_rewire artifact contract.")
    parser.add_argument("--recursive-bootstrap-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--edge-budget", type=_positive_int, required=True)
    parser.add_argument("--proposal-budget", type=_nonnegative_int, required=True)
    parser.add_argument("--closure-horizon", type=_nonnegative_int, required=True)
    parser.add_argument("--acceptance-policy", required=True)
    parser.add_argument("--old-domain-regression-budget", type=float, required=True)
    parser.add_argument("--rollback-enabled", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_and_write_adaptive_rewire_contract(
            recursive_bootstrap_manifest=args.recursive_bootstrap_manifest,
            output_dir=args.output_dir,
            student_id=args.student_id,
            experiment_id=args.experiment_id,
            edge_budget=args.edge_budget,
            proposal_budget=args.proposal_budget,
            closure_horizon=args.closure_horizon,
            acceptance_policy=args.acceptance_policy,
            old_domain_regression_budget=args.old_domain_regression_budget,
            rollback_enabled=args.rollback_enabled,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
