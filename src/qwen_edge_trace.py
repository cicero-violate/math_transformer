from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency
from src.qwen_sparse_student_runtime import (
    _checksum_features,
    build_adjacency_index,
    initialize_node_features,
    resolve_runtime_device,
)


SCHEMA_VERSION = "qwen_edge_trace.v1"
ROW_SCHEMA_VERSION = "qwen_edge_trace_row.v1"
UTILITY_SUMMARY_SCHEMA_VERSION = "qwen_edge_utility_summary.v1"
EDGE_TRACE_REPORT_FILENAME = "edge_trace_report.json"
EDGE_TRACE_ROWS_FILENAME = "edge_trace.jsonl"
EDGE_UTILITY_SUMMARY_FILENAME = "edge_utility_summary.json"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _norm_l1(vector: list[float]) -> float:
    return sum(abs(_finite_float(value, name="feature")) for value in vector)


def _norm_l2(vector: list[float]) -> float:
    return math.sqrt(sum(_finite_float(value, name="feature") ** 2 for value in vector))


def _add_scaled(dst: list[float], src: list[float], scale: float) -> list[float]:
    if len(dst) != len(src):
        raise ValueError("feature dimensions must match")
    return [left + scale * right for left, right in zip(dst, src)]


def _propagate_with_rows(
    features: dict[str, list[float]],
    adjacency_index: dict[str, Any],
    *,
    seed: int,
    step: int,
    edge_metadata_by_id: dict[str, dict[str, Any]],
    residual: float = 1.0,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    next_features = {
        node_id: [residual * value for value in vector]
        for node_id, vector in features.items()
    }
    rows: list[dict[str, Any]] = []
    for src_id in sorted(adjacency_index["outgoing"]):
        src_features = features[src_id]
        src_l1 = _norm_l1(src_features)
        src_l2 = _norm_l2(src_features)
        for edge in adjacency_index["outgoing"][src_id]:
            edge_id = str(edge["edge_id"])
            dst_id = str(edge["dst_id"])
            normalized_weight = _finite_float(edge["normalized_weight"], name="normalized_weight")
            message = [normalized_weight * value for value in src_features]
            message_l1 = _norm_l1(message)
            message_l2 = _norm_l2(message)
            next_features[dst_id] = _add_scaled(next_features[dst_id], src_features, normalized_weight)
            meta = edge_metadata_by_id[edge_id]
            row = {
                "schema_version": ROW_SCHEMA_VERSION,
                "seed": seed,
                "step": step,
                "edge_id": edge_id,
                "src_id": src_id,
                "dst_id": dst_id,
                "relation": meta.get("relation"),
                "score_name": meta.get("score_name"),
                "weight": _finite_float(meta.get("weight"), name="edge.weight"),
                "normalized_weight": normalized_weight,
                "src_l1": src_l1,
                "src_l2": src_l2,
                "message_l1": message_l1,
                "message_l2": message_l2,
                "dst_delta_l1": message_l1,
                "dst_delta_l2": message_l2,
                "used": True,
                "finite": all(math.isfinite(value) for value in (src_l1, src_l2, message_l1, message_l2, normalized_weight)),
            }
            if not bool(row["finite"]):
                raise ValueError("edge trace row produced non-finite utility values")
            rows.append(row)
    return next_features, rows


def _aggregate_edge_utilities(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_edge[str(row["edge_id"])].append(row)
    edge_summaries: list[dict[str, Any]] = []
    for edge_id in sorted(by_edge):
        edge_rows = by_edge[edge_id]
        first = edge_rows[0]
        message_l1 = [float(row["message_l1"]) for row in edge_rows]
        message_l2 = [float(row["message_l2"]) for row in edge_rows]
        src_l1 = [float(row["src_l1"]) for row in edge_rows]
        dst_delta_l1 = [float(row["dst_delta_l1"]) for row in edge_rows]
        utility_score = statistics.mean(message_l1)
        edge_summaries.append(
            {
                "edge_id": edge_id,
                "src_id": first["src_id"],
                "dst_id": first["dst_id"],
                "relation": first.get("relation"),
                "score_name": first.get("score_name"),
                "weight": first["weight"],
                "normalized_weight": first["normalized_weight"],
                "trace_count": len(edge_rows),
                "used_count": sum(1 for row in edge_rows if bool(row.get("used"))),
                "utility_score": utility_score,
                "message_l1_mean": utility_score,
                "message_l1_max": max(message_l1),
                "message_l2_mean": statistics.mean(message_l2),
                "message_l2_max": max(message_l2),
                "src_l1_mean": statistics.mean(src_l1),
                "dst_delta_l1_mean": statistics.mean(dst_delta_l1),
                "finite": all(bool(row.get("finite")) for row in edge_rows),
            }
        )
    ranked = sorted(edge_summaries, key=lambda row: (-float(row["utility_score"]), str(row["edge_id"])))
    return {
        "schema_version": UTILITY_SUMMARY_SCHEMA_VERSION,
        "status": "edge_utility_summary_ok",
        "edge_count": len(edge_summaries),
        "row_count": len(rows),
        "ranked_edges": ranked,
        "finite": all(bool(row["finite"]) for row in edge_summaries),
    }


def build_edge_trace_report(
    eval_output_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    seeds: list[int] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if feature_dim < 1:
        raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    resolved_seeds = [0, 1, 2] if seeds is None else list(seeds)
    if not resolved_seeds:
        raise ValueError("seeds must contain at least one seed")
    device_info = resolve_runtime_device(device)
    selected_adjacency = load_selected_adjacency(eval_output_dir, adjacency_name=adjacency_name, k=k)
    adjacency_summary = validate_selected_adjacency(selected_adjacency)
    adjacency_index = build_adjacency_index(selected_adjacency)
    edge_metadata_by_id = {str(edge["edge_id"]): dict(edge) for edge in selected_adjacency["edges"]}

    rows: list[dict[str, Any]] = []
    final_output_checksums: dict[str, str] = {}
    initial_input_checksums: dict[str, str] = {}
    for seed in resolved_seeds:
        features = initialize_node_features(adjacency_index["node_ids"], feature_dim, seed=seed)
        initial_input_checksums[str(seed)] = _checksum_features(features)
        for step in range(steps):
            features, step_rows = _propagate_with_rows(
                features,
                adjacency_index,
                seed=seed,
                step=step,
                edge_metadata_by_id=edge_metadata_by_id,
            )
            rows.extend(step_rows)
        final_output_checksums[str(seed)] = _checksum_features(features)

    expected_row_count = int(adjacency_summary["edge_count"]) * len(resolved_seeds) * steps
    if len(rows) != expected_row_count:
        raise ValueError(f"edge trace row_count={len(rows)} expected={expected_row_count}")
    utility_summary = _aggregate_edge_utilities(rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "edge_trace_ok",
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "topology_mutated": False,
        "promotion_eligible": False,
        "eval_output_dir": str(Path(eval_output_dir)),
        "adjacency_name": adjacency_summary["adjacency_name"],
        "k": adjacency_summary["k"],
        "edge_count": adjacency_summary["edge_count"],
        "node_count": adjacency_summary["node_count"],
        "max_out_degree": adjacency_summary["max_out_degree"],
        "feature_dim": feature_dim,
        "steps": steps,
        "seeds": resolved_seeds,
        "row_count": len(rows),
        "expected_row_count": expected_row_count,
        "device": device,
        "device_info": device_info,
        "trace_backend": "python_utility_probe",
        "initial_input_checksum_by_seed": initial_input_checksums,
        "final_output_checksum_by_seed": final_output_checksums,
        "edge_trace_rows": rows,
        "edge_utility_summary": utility_summary,
        "artifacts": {
            "edge_trace_report": EDGE_TRACE_REPORT_FILENAME,
            "edge_trace_rows": EDGE_TRACE_ROWS_FILENAME,
            "edge_utility_summary": EDGE_UTILITY_SUMMARY_FILENAME,
        },
        "finite": bool(utility_summary["finite"]) and all(bool(row["finite"]) for row in rows),
        "note": "v26 P0 edge utility traces only; no topology mutation or teacher runtime",
    }
    validate_edge_trace_report(report)
    return report


def _serializable_report(report: dict[str, Any]) -> dict[str, Any]:
    clean = dict(report)
    clean.pop("edge_trace_rows", None)
    return clean


def write_edge_trace_report(report: dict[str, Any], output_dir: str | Path) -> None:
    validate_edge_trace_report(report)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = report.get("edge_trace_rows")
    if not isinstance(rows, list):
        raise ValueError("edge_trace_rows must be present when writing edge traces")
    with (out / EDGE_TRACE_ROWS_FILENAME).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _write_json(out / EDGE_UTILITY_SUMMARY_FILENAME, report["edge_utility_summary"])
    _write_json(out / EDGE_TRACE_REPORT_FILENAME, _serializable_report(report))


def load_edge_trace_report(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / EDGE_TRACE_REPORT_FILENAME)


def load_edge_trace_rows(output_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(output_dir) / EDGE_TRACE_ROWS_FILENAME
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError("edge trace rows must be JSON objects")
                rows.append(row)
    return rows


def validate_edge_trace_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad edge trace schema_version={report.get('schema_version')!r}")
    if report.get("status") != "edge_trace_ok":
        raise ValueError(f"bad edge trace status={report.get('status')!r}")
    for key, expected in {
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "topology_mutated": False,
        "finite": True,
    }.items():
        if bool(report.get(key)) is not expected:
            raise ValueError(f"edge trace {key} must be {expected}")
    if int(report.get("feature_dim", 0)) < 1:
        raise ValueError("edge trace feature_dim must be >= 1")
    if int(report.get("steps", 0)) < 1:
        raise ValueError("edge trace steps must be >= 1")
    if int(report.get("max_out_degree", 0)) > int(report.get("k", 0)):
        raise ValueError("edge trace max_out_degree must be <= k")
    row_count = int(report.get("row_count", -1))
    expected_row_count = int(report.get("expected_row_count", -2))
    if row_count != expected_row_count:
        raise ValueError("edge trace row_count must match expected_row_count")
    utility_summary = report.get("edge_utility_summary")
    if not isinstance(utility_summary, dict):
        raise ValueError("edge trace utility summary must be an object")
    if utility_summary.get("schema_version") != UTILITY_SUMMARY_SCHEMA_VERSION:
        raise ValueError("bad edge utility summary schema_version")
    if int(utility_summary.get("row_count", -1)) != row_count:
        raise ValueError("edge utility row_count must match edge trace row_count")
    ranked_edges = utility_summary.get("ranked_edges")
    if not isinstance(ranked_edges, list):
        raise ValueError("edge utility ranked_edges must be a list")
    if len(ranked_edges) != int(report.get("edge_count", -1)):
        raise ValueError("edge utility ranked_edges must match edge_count")
    for edge in ranked_edges:
        if not isinstance(edge, dict):
            raise ValueError("edge utility rows must be objects")
        if not bool(edge.get("finite")):
            raise ValueError("edge utility rows must be finite")
        _finite_float(edge.get("utility_score"), name="utility_score")
    rows = report.get("edge_trace_rows")
    if rows is not None:
        if not isinstance(rows, list) or len(rows) != row_count:
            raise ValueError("edge_trace_rows length must match row_count")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("edge trace row must be an object")
            if row.get("schema_version") != ROW_SCHEMA_VERSION:
                raise ValueError("bad edge trace row schema_version")
            for key in ("message_l1", "message_l2", "src_l1", "dst_delta_l1"):
                if _finite_float(row.get(key), name=key) < 0.0:
                    raise ValueError(f"edge trace row {key} must be non-negative")
            if not bool(row.get("used")) or not bool(row.get("finite")):
                raise ValueError("edge trace rows must be used and finite")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "edge_trace_report_valid",
        "edge_count": int(report["edge_count"]),
        "row_count": row_count,
        "adjacency_name": report["adjacency_name"],
        "topology_mutated": False,
    }


def run_and_write_edge_trace_report(
    eval_output_dir: str | Path,
    output_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    seeds: list[int] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    report = build_edge_trace_report(
        eval_output_dir,
        k=k,
        adjacency_name=adjacency_name,
        feature_dim=feature_dim,
        steps=steps,
        seeds=seeds,
        device=device,
    )
    write_edge_trace_report(report, output_dir)
    return _serializable_report(report)


def _parse_int_list(raw: str, *, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one integer")
    return values


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect v26 P0 active-edge utility traces for a fixed Qwen sparse adjacency.")
    parser.add_argument("--eval-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--k", type=_positive_int, default=None)
    group.add_argument("--adjacency-name", default=None)
    parser.add_argument("--feature-dim", type=_positive_int, default=8)
    parser.add_argument("--steps", type=_positive_int, default=1)
    parser.add_argument("--seeds", type=lambda raw: _parse_int_list(raw, name="seeds"), default=[0, 1, 2])
    parser.add_argument("--device", default="cpu", choices=["cpu", "torch_cpu", "cuda", "auto"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_and_write_edge_trace_report(
            args.eval_output_dir,
            args.output_dir,
            k=args.k,
            adjacency_name=args.adjacency_name,
            feature_dim=args.feature_dim,
            steps=args.steps,
            seeds=args.seeds,
            device=args.device,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "edge_trace_report": str(Path(args.output_dir) / EDGE_TRACE_REPORT_FILENAME),
        "edge_trace_rows": str(Path(args.output_dir) / EDGE_TRACE_ROWS_FILENAME),
        "edge_utility_summary": str(Path(args.output_dir) / EDGE_UTILITY_SUMMARY_FILENAME),
        "adjacency_name": report["adjacency_name"],
        "edge_count": report["edge_count"],
        "row_count": report["row_count"],
        "bounded_active_adjacency": report["bounded_active_adjacency"],
        "topology_mutated": report["topology_mutated"],
        "device": report["device"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
