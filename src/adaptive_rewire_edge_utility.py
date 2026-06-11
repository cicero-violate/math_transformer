from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.adaptive_rewire_contract import (
    REQUIRED_ARTIFACT_FILENAMES,
    validate_adaptive_rewire_contract_manifest,
)
from src.qwen_edge_trace import EDGE_TRACE_REPORT_FILENAME, load_edge_trace_report, validate_edge_trace_report
from src.qwen_rewire_recursive_bootstrap import load_recursive_bootstrap_manifest, validate_recursive_bootstrap_manifest
from src.qwen_sparse_student_handoff import validate_selected_adjacency


SCHEMA_VERSION = "adaptive_rewire_edge_utility.v1"
ROW_SCHEMA_VERSION = "adaptive_rewire_edge_utility.row.v1"
EDGE_UTILITY_FILENAME = "edge_utility.jsonl"
EDGE_UTILITY_REPORT_FILENAME = "edge_utility_report.json"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: expected JSON object rows")
                rows.append(row)
    return rows


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return {key: 0.0 for key in values}
    return {key: _clamp01((value - lo) / (hi - lo)) for key, value in values.items()}


def load_adaptive_rewire_edge_utility_report(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / EDGE_UTILITY_REPORT_FILENAME)


def _load_contract_manifest(path: str | Path) -> dict[str, Any]:
    return _read_json(Path(path))


def validate_edge_utility_contract_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != "adaptive_rewire_contract.v1":
        raise ValueError(f"bad adaptive rewire contract schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "adaptive_rewire_contract_ok":
        raise ValueError(f"bad adaptive rewire contract status={manifest.get('status')!r}")
    for key in ("canonical_artifacts_ready", "required_artifacts_present", "edge_trace_ready", "bounded_active_adjacency", "final_equals_initial"):
        if not bool(manifest.get(key)):
            raise ValueError(f"{key} must be true")
    for key in ("topology_mutated", "teacher_checkpoint_loaded", "teacher_inference_runtime_required", "raw_weight_payload_in_graph"):
        if bool(manifest.get(key)):
            raise ValueError(f"{key} must be false")
    return validate_adaptive_rewire_contract_manifest(manifest)


def _artifact_path(contract_manifest: dict[str, Any], artifact_key: str) -> Path:
    artifacts = contract_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("contract artifacts must be an object")
    rel = artifacts.get(artifact_key)
    if not rel:
        raise ValueError(f"contract artifact missing: {artifact_key}")
    return Path(str(contract_manifest["output_dir"])) / str(rel)


def _load_initial_adjacency(contract_manifest: dict[str, Any]) -> dict[str, Any]:
    path = _artifact_path(contract_manifest, "initial_adjacency")
    adjacency = _read_json(path)
    validate_selected_adjacency(adjacency)
    return adjacency


def _load_source_edge_trace_report(contract_manifest: dict[str, Any], *, allow_empty_trace: bool) -> tuple[dict[str, Any] | None, str | None]:
    source_bootstrap = contract_manifest.get("source_recursive_bootstrap_manifest")
    if source_bootstrap:
        bootstrap_path = Path(str(source_bootstrap))
        if bootstrap_path.exists():
            bootstrap = load_recursive_bootstrap_manifest(bootstrap_path.parent)
            validate_recursive_bootstrap_manifest(bootstrap)
            edge_trace_dir = Path(str(bootstrap["cycle_edge_trace_dir"]))
            report_path = edge_trace_dir / EDGE_TRACE_REPORT_FILENAME
            if report_path.exists():
                report = load_edge_trace_report(edge_trace_dir)
                validate_edge_trace_report(report)
                return report, str(report_path)
    fallback_path = _artifact_path(contract_manifest, "edge_utility")
    if fallback_path.exists():
        rows = _read_jsonl(fallback_path)
        projected = [row for row in rows if "edge_id" in row and ("utility_score" in row or "message_l1_mean" in row)]
        if projected:
            return {
                "schema_version": "adaptive_rewire_edge_utility_fallback.v1",
                "status": "projected_edge_utility_fallback_ok",
                "row_count": len(projected),
                "edge_utility_summary": {"ranked_edges": projected},
            }, str(fallback_path)
    if allow_empty_trace:
        return None, None
    raise FileNotFoundError("edge trace source missing")


def _trace_stats_by_edge(edge_trace_report: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], int]:
    if edge_trace_report is None:
        return {}, 0
    utility = edge_trace_report.get("edge_utility_summary")
    if not isinstance(utility, dict):
        raise ValueError("edge trace report missing edge_utility_summary")
    ranked_edges = utility.get("ranked_edges")
    if not isinstance(ranked_edges, list):
        raise ValueError("edge utility ranked_edges must be a list")
    by_edge: dict[str, dict[str, Any]] = {}
    total_rows = int(edge_trace_report.get("row_count", utility.get("row_count", 0)))
    for row in ranked_edges:
        if not isinstance(row, dict):
            raise ValueError("edge utility ranked edge must be an object")
        edge_id = str(row.get("edge_id"))
        by_edge[edge_id] = dict(row)
    return by_edge, total_rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def build_edge_utility_rows(
    *,
    contract_manifest: dict[str, Any],
    initial_adjacency: dict[str, Any],
    edge_trace_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    validate_edge_utility_contract_manifest(contract_manifest)
    adjacency_summary = validate_selected_adjacency(initial_adjacency)
    trace_by_edge, _trace_total = _trace_stats_by_edge(edge_trace_report)
    edges = sorted(
        initial_adjacency["edges"],
        key=lambda edge: (
            str(edge.get("src_id")),
            str(edge.get("dst_id")),
            str(edge.get("relation", "")),
            str(edge.get("edge_id")),
        ),
    )
    prior_scores = {str(edge["edge_id"]): abs(_safe_float(edge.get("weight"))) for edge in edges}
    prior_norm = _minmax(prior_scores)
    cost_values = {str(edge["edge_id"]): 1.0 for edge in edges}
    cost_norm = _minmax(cost_values)
    rows: list[dict[str, Any]] = []
    for edge in edges:
        edge_id = str(edge["edge_id"])
        trace = trace_by_edge.get(edge_id, {})
        activation_count = int(trace.get("used_count", trace.get("trace_count", 0)))
        trace_row_count = int(trace.get("trace_count", 0))
        activation_frequency = activation_count / trace_row_count if trace_row_count > 0 else 0.0
        activation_frequency_norm = _clamp01(activation_frequency)
        source_prior_score = prior_scores[edge_id]
        source_prior_score_norm = prior_norm.get(edge_id, 0.0)
        compute_cost = cost_values[edge_id]
        compute_cost_norm = cost_norm.get(edge_id, 0.0)
        closure_critical_flag = False
        utility_score = _clamp01(
            0.55 * activation_frequency_norm
            + 0.35 * source_prior_score_norm
            + 0.10 * (1.0 - compute_cost_norm)
        )
        archive_priority = _clamp01(
            0.60 * (1.0 - utility_score)
            + 0.30 * compute_cost_norm
            + 0.10 * (0.0 if closure_critical_flag else 1.0)
        )
        reason_codes = ["active_edge", "no_gradient_stats", "no_loss_contribution", "no_error_correlation"]
        if trace:
            reason_codes.append("edge_trace_observed")
        else:
            reason_codes.append("edge_trace_missing")
        rows.append({
            "schema_version": ROW_SCHEMA_VERSION,
            "student_id": contract_manifest["student_id"],
            "experiment_id": contract_manifest["experiment_id"],
            "cycle_index": int(contract_manifest["cycle_index"]),
            "edge_id": edge_id,
            "source_node": str(edge["src_id"]),
            "target_node": str(edge["dst_id"]),
            "edge_type": edge.get("relation"),
            "active": True,
            "in_initial_adjacency": True,
            "activation_count": activation_count,
            "trace_row_count": trace_row_count,
            "activation_frequency": activation_frequency,
            "activation_frequency_norm": activation_frequency_norm,
            "gradient_norm": None,
            "loss_contribution": None,
            "error_correlation": None,
            "mean_abs_activation": _safe_float(trace.get("message_l1_mean")) if trace else None,
            "max_abs_activation": _safe_float(trace.get("message_l1_max")) if trace else None,
            "compute_cost": compute_cost,
            "compute_cost_norm": compute_cost_norm,
            "source_prior_score": source_prior_score,
            "source_prior_score_norm": source_prior_score_norm,
            "closure_critical_flag": closure_critical_flag,
            "utility_score": utility_score,
            "archive_priority": archive_priority,
            "add_priority": 0.0,
            "reason_codes": reason_codes,
            "source_provenance": {
                "initial_adjacency_name": adjacency_summary["adjacency_name"],
                "trace_source": "cycle_edge_trace" if edge_trace_report is not None else "empty_trace_allowed",
                "gradient_stats_available": False,
                "loss_contribution_available": False,
                "error_correlation_available": False,
            },
        })
    return rows


def validate_edge_utility_rows(rows: list[dict[str, Any]], *, expected_edge_count: int) -> None:
    if len(rows) != expected_edge_count:
        raise ValueError("edge utility row count must match active edge count")
    previous_key: tuple[str, str, str] | None = None
    for row in rows:
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise ValueError("bad edge utility row schema_version")
        for key in ("utility_score", "archive_priority", "add_priority", "activation_frequency_norm", "source_prior_score_norm", "compute_cost_norm"):
            value = _safe_float(row.get(key), default=-1.0)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{key} must be in [0,1]")
        if not bool(row.get("active")) or not bool(row.get("in_initial_adjacency")):
            raise ValueError("edge utility rows must describe active initial adjacency edges")
        sort_key = (str(row.get("source_node")), str(row.get("target_node")), str(row.get("edge_type", "")))
        if previous_key is not None and sort_key < previous_key:
            raise ValueError("edge utility rows are not deterministically sorted")
        previous_key = sort_key


def build_edge_utility_report(
    *,
    contract_manifest_path: str | Path,
    contract_manifest: dict[str, Any],
    output_dir: str | Path,
    edge_utility_path: str | Path,
    rows: list[dict[str, Any]],
    edge_trace_report: dict[str, Any] | None,
    edge_trace_source: str | None,
) -> dict[str, Any]:
    validate_edge_utility_contract_manifest(contract_manifest)
    edge_count = len(rows)
    trace_row_count = 0 if edge_trace_report is None else int(edge_trace_report.get("row_count", 0))
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "edge_utility_report_ok",
        "source_contract_manifest": str(Path(contract_manifest_path)),
        "source_edge_trace": edge_trace_source,
        "output_dir": str(Path(output_dir)),
        "edge_utility_path": str(Path(edge_utility_path)),
        "edge_count": edge_count,
        "active_edge_count": edge_count,
        "trace_row_count": trace_row_count,
        "missing_gradient_stats": True,
        "missing_loss_contribution": True,
        "missing_error_correlation": True,
        "normalized": True,
        "topology_mutated": False,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "next_stage": "proposal_batch_generator",
    }
    validate_edge_utility_report(report)
    return report


def validate_edge_utility_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad edge utility report schema_version={report.get('schema_version')!r}")
    if report.get("status") != "edge_utility_report_ok":
        raise ValueError(f"bad edge utility report status={report.get('status')!r}")
    for key, expected in {
        "normalized": True,
        "topology_mutated": False,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
    }.items():
        if bool(report.get(key)) is not expected:
            raise ValueError(f"edge utility report {key} must be {expected}")
    if report.get("next_stage") != "proposal_batch_generator":
        raise ValueError("next_stage must be proposal_batch_generator")
    if int(report.get("edge_count", -1)) < 0:
        raise ValueError("edge_count must be non-negative")
    if int(report.get("active_edge_count", -1)) != int(report.get("edge_count", -2)):
        raise ValueError("active_edge_count must match edge_count for P10")
    for key in ("source_contract_manifest", "output_dir", "edge_utility_path"):
        path = Path(str(report.get(key)))
        if not path.exists():
            raise ValueError(f"edge utility report path missing: {key} -> {path}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "edge_utility_report_valid",
        "edge_count": int(report["edge_count"]),
        "normalized": True,
        "topology_mutated": False,
    }


def run_and_write_edge_utility_report(
    *,
    contract_manifest: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    allow_empty_trace: bool = False,
) -> dict[str, Any]:
    contract_path = Path(contract_manifest)
    contract = _load_contract_manifest(contract_path)
    validate_edge_utility_contract_manifest(contract)
    initial_adjacency = _load_initial_adjacency(contract)
    adjacency_summary = validate_selected_adjacency(initial_adjacency)

    out = Path(output_dir)
    if out.exists():
        if not overwrite:
            raise ValueError(f"edge utility output dir already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    edge_trace_report, edge_trace_source = _load_source_edge_trace_report(contract, allow_empty_trace=allow_empty_trace)
    rows = build_edge_utility_rows(
        contract_manifest=contract,
        initial_adjacency=initial_adjacency,
        edge_trace_report=edge_trace_report,
    )
    validate_edge_utility_rows(rows, expected_edge_count=int(adjacency_summary["edge_count"]))
    edge_utility_path = out / EDGE_UTILITY_FILENAME
    _write_jsonl(edge_utility_path, rows)
    report = build_edge_utility_report(
        contract_manifest_path=contract_path,
        contract_manifest=contract,
        output_dir=out,
        edge_utility_path=edge_utility_path,
        rows=rows,
        edge_trace_report=edge_trace_report,
        edge_trace_source=edge_trace_source,
    )
    _write_json(out / EDGE_UTILITY_REPORT_FILENAME, report)
    return {
        "status": "edge_utility_aggregation_ok",
        "edge_utility_report": str(out / EDGE_UTILITY_REPORT_FILENAME),
        "edge_utility_path": str(edge_utility_path),
        "output_dir": str(out),
        "edge_count": len(rows),
        "active_edge_count": len(rows),
        "trace_row_count": report["trace_row_count"],
        "normalized": True,
        "topology_mutated": False,
        "bounded_active_adjacency": True,
        "next_stage": "proposal_batch_generator",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate normalized adaptive_rewire edge utility rows from a P9 contract.")
    parser.add_argument("--contract-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-empty-trace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_and_write_edge_utility_report(
            contract_manifest=args.contract_manifest,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            allow_empty_trace=args.allow_empty_trace,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
