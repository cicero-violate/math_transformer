from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.adaptive_rewire_edge_utility import validate_edge_utility_report


SCHEMA_VERSION = "adaptive_rewire_proposal_batch.v1"
ROW_SCHEMA_VERSION = "adaptive_rewire_proposal.row.v1"
PROPOSAL_BATCH_FILENAME = "proposal_batch.jsonl"
PROPOSAL_BATCH_REPORT_FILENAME = "proposal_batch_report.json"
PROTECTED_EDGES_FILENAME = "protected_edges.jsonl"
CANDIDATE_POOL_SUMMARY_FILENAME = "candidate_pool_summary.json"


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


def _safe_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _bounded01(value: Any, *, name: str) -> float:
    number = _safe_float(value, name=name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return number


def load_proposal_batch_report(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / PROPOSAL_BATCH_REPORT_FILENAME)


def validate_source_edge_utility_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != "adaptive_rewire_edge_utility.v1":
        raise ValueError(f"bad edge utility schema_version={report.get('schema_version')!r}")
    if report.get("status") != "edge_utility_report_ok":
        raise ValueError(f"bad edge utility status={report.get('status')!r}")
    for key in ("normalized", "bounded_active_adjacency"):
        if not bool(report.get(key)):
            raise ValueError(f"{key} must be true")
    for key in ("topology_mutated", "teacher_checkpoint_loaded", "teacher_inference_runtime_required", "raw_weight_payload_in_graph"):
        if bool(report.get(key)):
            raise ValueError(f"{key} must be false")
    if report.get("next_stage") != "proposal_batch_generator":
        raise ValueError("next_stage must be proposal_batch_generator")
    return validate_edge_utility_report(report)


def validate_edge_utility_rows(rows: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(rows):
        if row.get("schema_version") != "adaptive_rewire_edge_utility.row.v1":
            raise ValueError(f"edge utility row {idx} bad schema_version")
        if not bool(row.get("active")):
            raise ValueError(f"edge utility row {idx} active must be true")
        if not bool(row.get("in_initial_adjacency")):
            raise ValueError(f"edge utility row {idx} in_initial_adjacency must be true")
        _bounded01(row.get("utility_score"), name="utility_score")
        _bounded01(row.get("archive_priority"), name="archive_priority")
        _bounded01(row.get("add_priority"), name="add_priority")


def _row_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("source_node")),
        str(row.get("target_node")),
        str(row.get("edge_type", "")),
        str(row.get("edge_id")),
    )


def _proposal_row(
    *,
    source_row: dict[str, Any],
    proposal_index: int,
    operation: str,
    reason_codes: list[str],
    proposed_delta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "student_id": source_row["student_id"],
        "experiment_id": source_row["experiment_id"],
        "cycle_index": int(source_row["cycle_index"]),
        "proposal_index": proposal_index,
        "operation": operation,
        "edge_id": source_row["edge_id"],
        "source_node": source_row["source_node"],
        "target_node": source_row["target_node"],
        "edge_type": source_row.get("edge_type"),
        "utility_score": float(source_row["utility_score"]),
        "archive_priority": float(source_row["archive_priority"]),
        "add_priority": float(source_row["add_priority"]),
        "closure_critical_flag": bool(source_row.get("closure_critical_flag", False)),
        "reason_codes": reason_codes,
        "source_provenance": {
            "edge_utility_row_schema": source_row.get("schema_version"),
            "edge_utility_reason_codes": list(source_row.get("reason_codes", [])),
            "edge_utility_source_provenance": source_row.get("source_provenance", {}),
        },
        "proposed_delta": proposed_delta,
        "rollback_data": {
            "operation": "restore_edge_state",
            "edge_id": source_row["edge_id"],
            "source_node": source_row["source_node"],
            "target_node": source_row["target_node"],
            "edge_type": source_row.get("edge_type"),
            "active_before": True,
        },
        "auto_accepted": False,
        "accepted": False,
        "rejected": False,
        "topology_mutated": False,
    }


def build_proposal_batch_rows(
    *,
    edge_utility_rows: list[dict[str, Any]],
    proposal_budget: int,
    archive_priority_threshold: float,
    downweight_priority_threshold: float,
    keep_utility_threshold: float,
) -> list[dict[str, Any]]:
    if proposal_budget < 0:
        raise ValueError(f"proposal_budget must be >= 0, got {proposal_budget!r}")
    archive_priority_threshold = _bounded01(archive_priority_threshold, name="archive_priority_threshold")
    downweight_priority_threshold = _bounded01(downweight_priority_threshold, name="downweight_priority_threshold")
    keep_utility_threshold = _bounded01(keep_utility_threshold, name="keep_utility_threshold")
    validate_edge_utility_rows(edge_utility_rows)
    rows = sorted(edge_utility_rows, key=_row_sort_key)

    candidates: list[tuple[tuple[Any, ...], str, dict[str, Any], list[str], dict[str, Any]]] = []
    for row in rows:
        utility = float(row["utility_score"])
        archive_priority = float(row["archive_priority"])
        activation = float(row.get("activation_frequency_norm", 0.0))
        compute_cost = float(row.get("compute_cost_norm", 0.0))
        closure = bool(row.get("closure_critical_flag", False))
        if closure or utility >= keep_utility_threshold:
            reasons = ["closure_critical_protected" if closure else "high_utility_keep", "bounded_proposal_budget"]
            candidates.append(((-utility, _row_sort_key(row)), "keep", row, reasons, {
                "type": "protect_active_edge",
                "weight_multiplier": 1.0,
                "topology_mutated": False,
            }))
        elif archive_priority >= archive_priority_threshold:
            reasons = ["low_utility", "high_archive_priority", "candidate_pool_unavailable", "bounded_proposal_budget"]
            if compute_cost >= 0.5:
                reasons.append("high_compute_cost")
            candidates.append(((-archive_priority, utility, _row_sort_key(row)), "archive", row, reasons, {
                "type": "archive_active_edge_candidate",
                "remove_edge": True,
                "topology_mutated": False,
            }))
        elif archive_priority >= downweight_priority_threshold or activation <= 0.25:
            reasons = ["low_activation_frequency", "candidate_pool_unavailable", "bounded_proposal_budget"]
            if archive_priority >= downweight_priority_threshold:
                reasons.append("moderate_archive_priority")
            candidates.append(((-archive_priority, activation, _row_sort_key(row)), "downweight", row, reasons, {
                "type": "downweight_active_edge_candidate",
                "weight_multiplier": 0.5,
                "topology_mutated": False,
            }))

    candidates.sort(key=lambda item: (item[1] != "archive", item[0]))
    proposals: list[dict[str, Any]] = []
    for proposal_index, (_rank, operation, source_row, reason_codes, proposed_delta) in enumerate(candidates[:proposal_budget]):
        proposals.append(_proposal_row(
            source_row=source_row,
            proposal_index=proposal_index,
            operation=operation,
            reason_codes=reason_codes,
            proposed_delta=proposed_delta,
        ))
    validate_proposal_rows(proposals, proposal_budget=proposal_budget)
    return proposals


def validate_proposal_rows(rows: list[dict[str, Any]], *, proposal_budget: int) -> None:
    if len(rows) > proposal_budget:
        raise ValueError("proposal_count must be <= proposal_budget")
    for idx, row in enumerate(rows):
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise ValueError(f"proposal row {idx} bad schema_version")
        if row.get("operation") not in {"keep", "downweight", "archive", "add", "tombstone"}:
            raise ValueError(f"proposal row {idx} unsupported operation")
        for key in ("utility_score", "archive_priority", "add_priority"):
            _bounded01(row.get(key), name=key)
        if bool(row.get("auto_accepted")):
            raise ValueError("proposal rows must not be auto accepted")
        if bool(row.get("accepted")) or bool(row.get("rejected")):
            raise ValueError("proposal rows must be undecided")
        if bool(row.get("topology_mutated")):
            raise ValueError("proposal rows must not mutate topology")
        if not isinstance(row.get("reason_codes"), list) or not row["reason_codes"]:
            raise ValueError("proposal rows must include reason_codes")
        if not isinstance(row.get("rollback_data"), dict) or not row["rollback_data"]:
            raise ValueError("proposal rows must include rollback_data")


def _count_operation(rows: list[dict[str, Any]], operation: str) -> int:
    return sum(1 for row in rows if row.get("operation") == operation)


def build_proposal_batch_report(
    *,
    edge_utility_report_path: str | Path,
    edge_utility_report: dict[str, Any],
    output_dir: str | Path,
    proposal_batch_path: str | Path,
    proposal_rows: list[dict[str, Any]],
    proposal_budget: int,
) -> dict[str, Any]:
    validate_source_edge_utility_report(edge_utility_report)
    validate_proposal_rows(proposal_rows, proposal_budget=proposal_budget)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "proposal_batch_report_ok",
        "source_edge_utility_report": str(Path(edge_utility_report_path)),
        "output_dir": str(Path(output_dir)),
        "proposal_batch_path": str(Path(proposal_batch_path)),
        "proposal_budget": int(proposal_budget),
        "proposal_count": len(proposal_rows),
        "archive_proposal_count": _count_operation(proposal_rows, "archive"),
        "downweight_proposal_count": _count_operation(proposal_rows, "downweight"),
        "keep_proposal_count": _count_operation(proposal_rows, "keep"),
        "add_proposal_count": _count_operation(proposal_rows, "add"),
        "tombstone_proposal_count": _count_operation(proposal_rows, "tombstone"),
        "candidate_pool_available": False,
        "bounded_proposal_count": len(proposal_rows) <= int(proposal_budget),
        "auto_accepted": False,
        "topology_mutated": False,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "next_stage": "bounded_active_graph_enforcement",
    }
    validate_proposal_batch_report(report)
    return report


def validate_proposal_batch_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad proposal batch schema_version={report.get('schema_version')!r}")
    if report.get("status") != "proposal_batch_report_ok":
        raise ValueError(f"bad proposal batch status={report.get('status')!r}")
    for key, expected in {
        "bounded_proposal_count": True,
        "auto_accepted": False,
        "topology_mutated": False,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
    }.items():
        if bool(report.get(key)) is not expected:
            raise ValueError(f"proposal batch report {key} must be {expected}")
    if report.get("next_stage") != "bounded_active_graph_enforcement":
        raise ValueError("next_stage must be bounded_active_graph_enforcement")
    if int(report.get("proposal_count", -1)) > int(report.get("proposal_budget", -2)):
        raise ValueError("proposal_count must be <= proposal_budget")
    for key in ("source_edge_utility_report", "output_dir", "proposal_batch_path"):
        path = Path(str(report.get(key)))
        if not path.exists():
            raise ValueError(f"proposal batch report path missing: {key} -> {path}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "proposal_batch_report_valid",
        "proposal_count": int(report["proposal_count"]),
        "proposal_budget": int(report["proposal_budget"]),
        "topology_mutated": False,
    }


def run_and_write_proposal_batch_report(
    *,
    edge_utility_report: str | Path,
    output_dir: str | Path,
    proposal_budget: int,
    archive_priority_threshold: float = 0.50,
    downweight_priority_threshold: float = 0.35,
    keep_utility_threshold: float = 0.75,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(edge_utility_report)
    source_report = _read_json(source_path)
    validate_source_edge_utility_report(source_report)
    edge_rows = _read_jsonl(Path(str(source_report["edge_utility_path"])))
    validate_edge_utility_rows(edge_rows)
    out = Path(output_dir)
    if out.exists():
        if not overwrite:
            raise ValueError(f"proposal batch output dir already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    proposal_rows = build_proposal_batch_rows(
        edge_utility_rows=edge_rows,
        proposal_budget=proposal_budget,
        archive_priority_threshold=archive_priority_threshold,
        downweight_priority_threshold=downweight_priority_threshold,
        keep_utility_threshold=keep_utility_threshold,
    )
    proposal_path = out / PROPOSAL_BATCH_FILENAME
    _write_jsonl(proposal_path, proposal_rows)
    protected = [row for row in proposal_rows if row["operation"] == "keep"]
    _write_jsonl(out / PROTECTED_EDGES_FILENAME, protected)
    _write_json(out / CANDIDATE_POOL_SUMMARY_FILENAME, {
        "schema_version": "adaptive_rewire_candidate_pool_summary.v1",
        "status": "candidate_pool_unavailable",
        "candidate_pool_available": False,
        "add_proposals_generated": False,
        "reason": "candidate pool not provided to P11",
    })
    report = build_proposal_batch_report(
        edge_utility_report_path=source_path,
        edge_utility_report=source_report,
        output_dir=out,
        proposal_batch_path=proposal_path,
        proposal_rows=proposal_rows,
        proposal_budget=proposal_budget,
    )
    _write_json(out / PROPOSAL_BATCH_REPORT_FILENAME, report)
    return {
        "status": "proposal_batch_generation_ok",
        "proposal_batch_report": str(out / PROPOSAL_BATCH_REPORT_FILENAME),
        "proposal_batch_path": str(proposal_path),
        "output_dir": str(out),
        "proposal_budget": int(proposal_budget),
        "proposal_count": len(proposal_rows),
        "bounded_proposal_count": True,
        "auto_accepted": False,
        "topology_mutated": False,
        "next_stage": "bounded_active_graph_enforcement",
    }


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {raw!r}")
    return value


def _bounded_float_arg(raw: str) -> float:
    value = float(raw)
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0,1], got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a bounded canonical adaptive_rewire proposal batch from edge utilities.")
    parser.add_argument("--edge-utility-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proposal-budget", type=_positive_int, required=True)
    parser.add_argument("--archive-priority-threshold", type=_bounded_float_arg, default=0.50)
    parser.add_argument("--downweight-priority-threshold", type=_bounded_float_arg, default=0.35)
    parser.add_argument("--keep-utility-threshold", type=_bounded_float_arg, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_and_write_proposal_batch_report(
            edge_utility_report=args.edge_utility_report,
            output_dir=args.output_dir,
            proposal_budget=args.proposal_budget,
            archive_priority_threshold=args.archive_priority_threshold,
            downweight_priority_threshold=args.downweight_priority_threshold,
            keep_utility_threshold=args.keep_utility_threshold,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
