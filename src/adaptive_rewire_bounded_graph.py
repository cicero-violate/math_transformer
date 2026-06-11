from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.adaptive_rewire_proposal_batch import validate_proposal_batch_report, validate_proposal_rows
from src.qwen_sparse_student_handoff import validate_selected_adjacency


SCHEMA_VERSION = "adaptive_rewire_bounded_graph.v1"
CANDIDATE_ADJACENCY_SCHEMA_VERSION = "adaptive_rewire_bounded_candidate_adjacency.v1"
PROTECTED_EDGE_ROW_SCHEMA_VERSION = "adaptive_rewire_protected_edge.row.v1"
BOUNDED_GRAPH_REPORT_FILENAME = "bounded_graph_report.json"
BOUNDED_CANDIDATE_ADJACENCY_FILENAME = "bounded_candidate_adjacency.json"
PROTECTED_EDGES_FILENAME = "protected_edges.jsonl"


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


def load_bounded_graph_report(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / BOUNDED_GRAPH_REPORT_FILENAME)


def validate_source_proposal_batch_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != "adaptive_rewire_proposal_batch.v1":
        raise ValueError(f"bad proposal batch schema_version={report.get('schema_version')!r}")
    if report.get("status") != "proposal_batch_report_ok":
        raise ValueError(f"bad proposal batch status={report.get('status')!r}")
    if not bool(report.get("bounded_proposal_count")):
        raise ValueError("bounded_proposal_count must be true")
    for key in ("auto_accepted", "topology_mutated", "teacher_checkpoint_loaded", "teacher_inference_runtime_required", "raw_weight_payload_in_graph"):
        if bool(report.get(key)):
            raise ValueError(f"{key} must be false")
    if not bool(report.get("bounded_active_adjacency")):
        raise ValueError("bounded_active_adjacency must be true")
    if report.get("next_stage") != "bounded_active_graph_enforcement":
        raise ValueError("next_stage must be bounded_active_graph_enforcement")
    return validate_proposal_batch_report(report)


def _load_source_adjacency(proposal_report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    edge_utility_report = _read_json(Path(str(proposal_report["source_edge_utility_report"])))
    contract_manifest_path = Path(str(edge_utility_report["source_contract_manifest"]))
    contract = _read_json(contract_manifest_path)
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("source contract artifacts must be an object")
    initial_rel = artifacts.get("initial_adjacency", "initial_adjacency.json")
    initial_path = Path(str(contract["output_dir"])) / str(initial_rel)
    adjacency = _read_json(initial_path)
    validate_selected_adjacency(adjacency)
    return adjacency, str(initial_path)


def _proposal_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (int(row.get("proposal_index", 0)), str(row.get("operation")), str(row.get("edge_id")))


def _protected_reasons(proposal: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if bool(proposal.get("closure_critical_flag")):
        reasons.append("closure_critical")
    if proposal.get("operation") == "keep":
        reasons.append("keep_proposal")
    for code in proposal.get("reason_codes", []):
        code_text = str(code)
        if code_text in {"closure_critical_protected", "high_utility_keep"}:
            reasons.append(code_text)
    return sorted(set(reasons))


def _protected_edge_rows(source_edges: dict[str, dict[str, Any]], proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proposal in sorted(proposals, key=_proposal_sort_key):
        reasons = _protected_reasons(proposal)
        if not reasons:
            continue
        edge_id = str(proposal["edge_id"])
        edge = source_edges.get(edge_id, {})
        rows.append({
            "schema_version": PROTECTED_EDGE_ROW_SCHEMA_VERSION,
            "edge_id": edge_id,
            "source_node": proposal.get("source_node", edge.get("src_id")),
            "target_node": proposal.get("target_node", edge.get("dst_id")),
            "edge_type": proposal.get("edge_type", edge.get("relation")),
            "protection_reasons": reasons,
            "source_provenance": proposal.get("source_provenance", {}),
        })
    return rows


def _max_out_degree(edges: list[dict[str, Any]]) -> int:
    counts: Counter[str] = Counter(str(edge["src_id"]) for edge in edges)
    return max(counts.values(), default=0)


def _candidate_pool_available(proposal: dict[str, Any]) -> bool:
    provenance = proposal.get("source_provenance")
    delta = proposal.get("proposed_delta")
    return (
        isinstance(provenance, dict)
        and bool(provenance.get("candidate_pool_available"))
        and isinstance(delta, dict)
        and isinstance(delta.get("edge"), dict)
    )


def simulate_bounded_candidate_adjacency(
    *,
    source_adjacency: dict[str, Any],
    proposal_rows: list[dict[str, Any]],
    edge_budget: int,
    max_out_degree_limit: int | None,
    source_adjacency_path: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    source_summary = validate_selected_adjacency(source_adjacency)
    if edge_budget < 0:
        raise ValueError(f"edge_budget must be >= 0, got {edge_budget!r}")
    if max_out_degree_limit is not None and max_out_degree_limit < 1:
        raise ValueError(f"max_out_degree_limit must be >= 1, got {max_out_degree_limit!r}")
    validate_proposal_rows(proposal_rows, proposal_budget=max(0, len(proposal_rows)))

    source_edges = {str(edge["edge_id"]): dict(edge) for edge in source_adjacency["edges"]}
    candidate_edges = {edge_id: dict(edge) for edge_id, edge in source_edges.items()}
    protected_rows = _protected_edge_rows(source_edges, proposal_rows)
    protected_edge_ids = {str(row["edge_id"]) for row in protected_rows}
    rejected_for_budget_count = 0
    rejected_for_protection_count = 0

    for proposal in sorted(proposal_rows, key=_proposal_sort_key):
        operation = str(proposal["operation"])
        edge_id = str(proposal["edge_id"])
        if operation == "keep":
            continue
        if operation == "downweight":
            if edge_id in candidate_edges:
                candidate_edges[edge_id]["adaptive_rewire_simulated_delta"] = proposal.get("proposed_delta", {})
            continue
        if operation == "archive":
            if edge_id in protected_edge_ids:
                rejected_for_protection_count += 1
                continue
            candidate_edges.pop(edge_id, None)
            continue
        if operation == "tombstone":
            if edge_id in candidate_edges:
                candidate_edges[edge_id]["adaptive_rewire_tombstone_rejected"] = True
            continue
        if operation == "add":
            if not _candidate_pool_available(proposal):
                rejected_for_budget_count += 1
                continue
            candidate_edge = dict(proposal["proposed_delta"]["edge"])
            candidate_edge.setdefault("edge_id", edge_id)
            candidate_edge.setdefault("src_id", proposal["source_node"])
            candidate_edge.setdefault("dst_id", proposal["target_node"])
            candidate_edge.setdefault("relation", proposal.get("edge_type"))
            candidate_edge.setdefault("weight", 0.0)
            candidate_edge.setdefault("score_name", "adaptive_rewire_candidate")
            trial_edges = list(candidate_edges.values()) + [candidate_edge]
            trial_max_out = _max_out_degree(trial_edges)
            if len(trial_edges) > edge_budget or (max_out_degree_limit is not None and trial_max_out > max_out_degree_limit):
                rejected_for_budget_count += 1
                continue
            candidate_edges[str(candidate_edge["edge_id"])] = candidate_edge

    edges = sorted(candidate_edges.values(), key=lambda edge: (str(edge.get("src_id")), str(edge.get("dst_id")), str(edge.get("relation", "")), str(edge.get("edge_id"))))
    max_out = _max_out_degree(edges)
    bounded_edge_count = len(edges) <= edge_budget
    bounded_out_degree = max_out_degree_limit is None or max_out <= max_out_degree_limit
    if not bounded_edge_count:
        raise ValueError("candidate edge count exceeds edge_budget")
    if not bounded_out_degree:
        raise ValueError("candidate max_out_degree exceeds limit")
    candidate = {
        "schema_version": CANDIDATE_ADJACENCY_SCHEMA_VERSION,
        "status": "bounded_candidate_adjacency_ok",
        "source_adjacency": source_adjacency_path,
        "edge_count": len(edges),
        "max_out_degree": max_out,
        "bounded_active_adjacency": True,
        "topology_mutated": False,
        "adjacency": {
            **source_adjacency,
            "edges": edges,
            "edge_count": len(edges),
            "max_out_degree": max_out,
        },
        "edges": edges,
    }
    counts = {
        "rejected_for_budget_count": rejected_for_budget_count,
        "rejected_for_protection_count": rejected_for_protection_count,
        "protected_violation_count": 0,
        "source_edge_count": int(source_summary["edge_count"]),
    }
    validate_bounded_candidate_adjacency(candidate)
    return candidate, protected_rows, counts


def validate_bounded_candidate_adjacency(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("schema_version") != CANDIDATE_ADJACENCY_SCHEMA_VERSION:
        raise ValueError(f"bad bounded candidate schema_version={candidate.get('schema_version')!r}")
    if candidate.get("status") != "bounded_candidate_adjacency_ok":
        raise ValueError(f"bad bounded candidate status={candidate.get('status')!r}")
    if not bool(candidate.get("bounded_active_adjacency")):
        raise ValueError("bounded candidate adjacency must be bounded")
    if bool(candidate.get("topology_mutated")):
        raise ValueError("bounded candidate adjacency must not mutate topology")
    if int(candidate.get("edge_count", -1)) != len(candidate.get("edges", [])):
        raise ValueError("bounded candidate edge_count must match edges")
    return {
        "schema_version": CANDIDATE_ADJACENCY_SCHEMA_VERSION,
        "status": "bounded_candidate_adjacency_valid",
        "edge_count": int(candidate["edge_count"]),
        "max_out_degree": int(candidate["max_out_degree"]),
    }


def build_bounded_graph_report(
    *,
    proposal_batch_report_path: str | Path,
    proposal_batch_report: dict[str, Any],
    output_dir: str | Path,
    proposal_rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    protected_rows: list[dict[str, Any]],
    edge_budget: int,
    max_out_degree_limit: int | None,
    counts: dict[str, int],
    candidate_path: str | Path,
    protected_path: str | Path,
) -> dict[str, Any]:
    validate_source_proposal_batch_report(proposal_batch_report)
    candidate_summary = validate_bounded_candidate_adjacency(candidate)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "bounded_graph_report_ok",
        "source_proposal_batch_report": str(Path(proposal_batch_report_path)),
        "output_dir": str(Path(output_dir)),
        "proposal_count": len(proposal_rows),
        "edge_budget": int(edge_budget),
        "source_edge_count": int(counts["source_edge_count"]),
        "candidate_edge_count": int(candidate_summary["edge_count"]),
        "bounded_edge_count": int(candidate_summary["edge_count"]) <= int(edge_budget),
        "max_out_degree": int(candidate_summary["max_out_degree"]),
        "max_out_degree_limit": None if max_out_degree_limit is None else int(max_out_degree_limit),
        "bounded_out_degree": max_out_degree_limit is None or int(candidate_summary["max_out_degree"]) <= int(max_out_degree_limit),
        "protected_edge_count": len(protected_rows),
        "protected_violation_count": int(counts["protected_violation_count"]),
        "rejected_for_budget_count": int(counts["rejected_for_budget_count"]),
        "rejected_for_protection_count": int(counts["rejected_for_protection_count"]),
        "candidate_adjacency_path": str(Path(candidate_path)),
        "protected_edges_path": str(Path(protected_path)),
        "topology_mutated": False,
        "proposals_auto_accepted": False,
        "rewrites_accepted": False,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "next_stage": "closure_preservation_gate",
    }
    validate_bounded_graph_report(report)
    return report


def validate_bounded_graph_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad bounded graph schema_version={report.get('schema_version')!r}")
    if report.get("status") != "bounded_graph_report_ok":
        raise ValueError(f"bad bounded graph status={report.get('status')!r}")
    for key, expected in {
        "bounded_edge_count": True,
        "bounded_out_degree": True,
        "topology_mutated": False,
        "proposals_auto_accepted": False,
        "rewrites_accepted": False,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
    }.items():
        if bool(report.get(key)) is not expected:
            raise ValueError(f"bounded graph report {key} must be {expected}")
    if report.get("next_stage") != "closure_preservation_gate":
        raise ValueError("next_stage must be closure_preservation_gate")
    if int(report.get("candidate_edge_count", -1)) > int(report.get("edge_budget", -2)):
        raise ValueError("candidate_edge_count must be <= edge_budget")
    for key in ("source_proposal_batch_report", "output_dir", "candidate_adjacency_path", "protected_edges_path"):
        path = Path(str(report.get(key)))
        if not path.exists():
            raise ValueError(f"bounded graph report path missing: {key} -> {path}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "bounded_graph_report_valid",
        "candidate_edge_count": int(report["candidate_edge_count"]),
        "bounded_active_adjacency": True,
        "topology_mutated": False,
    }


def run_and_write_bounded_graph_report(
    *,
    proposal_batch_report: str | Path,
    output_dir: str | Path,
    edge_budget: int,
    max_out_degree_limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(proposal_batch_report)
    source_report = _read_json(source_path)
    validate_source_proposal_batch_report(source_report)
    proposal_rows = _read_jsonl(Path(str(source_report["proposal_batch_path"])))
    validate_proposal_rows(proposal_rows, proposal_budget=int(source_report["proposal_budget"]))
    source_adjacency, source_adjacency_path = _load_source_adjacency(source_report)
    out = Path(output_dir)
    if out.exists():
        if not overwrite:
            raise ValueError(f"bounded graph output dir already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    candidate, protected_rows, counts = simulate_bounded_candidate_adjacency(
        source_adjacency=source_adjacency,
        proposal_rows=proposal_rows,
        edge_budget=edge_budget,
        max_out_degree_limit=max_out_degree_limit,
        source_adjacency_path=source_adjacency_path,
    )
    candidate_path = out / BOUNDED_CANDIDATE_ADJACENCY_FILENAME
    protected_path = out / PROTECTED_EDGES_FILENAME
    _write_json(candidate_path, candidate)
    _write_jsonl(protected_path, protected_rows)
    report = build_bounded_graph_report(
        proposal_batch_report_path=source_path,
        proposal_batch_report=source_report,
        output_dir=out,
        proposal_rows=proposal_rows,
        candidate=candidate,
        protected_rows=protected_rows,
        edge_budget=edge_budget,
        max_out_degree_limit=max_out_degree_limit,
        counts=counts,
        candidate_path=candidate_path,
        protected_path=protected_path,
    )
    _write_json(out / BOUNDED_GRAPH_REPORT_FILENAME, report)
    return {
        "status": "bounded_graph_enforcement_ok",
        "bounded_graph_report": str(out / BOUNDED_GRAPH_REPORT_FILENAME),
        "bounded_candidate_adjacency": str(candidate_path),
        "protected_edges": str(protected_path),
        "candidate_edge_count": report["candidate_edge_count"],
        "edge_budget": report["edge_budget"],
        "bounded_edge_count": True,
        "bounded_out_degree": True,
        "topology_mutated": False,
        "rewrites_accepted": False,
        "next_stage": "closure_preservation_gate",
    }


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate bounded active graph enforcement for an adaptive_rewire proposal batch.")
    parser.add_argument("--proposal-batch-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--edge-budget", type=_positive_int, required=True)
    parser.add_argument("--max-out-degree-limit", type=_positive_int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_and_write_bounded_graph_report(
            proposal_batch_report=args.proposal_batch_report,
            output_dir=args.output_dir,
            edge_budget=args.edge_budget,
            max_out_degree_limit=args.max_out_degree_limit,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
