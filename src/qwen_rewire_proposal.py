from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.qwen_edge_trace import load_edge_trace_report, validate_edge_trace_report
from src.qwen_graph_prior_eval import build_candidate_adjacency
from src.qwen_sparse_student_handoff import load_selected_adjacency, load_v25_handoff, validate_selected_adjacency
from src.qwen_weight_graph import read_weight_graph_artifacts


SCHEMA_VERSION = "qwen_rewire_proposal.v1"
PROPOSED_ADJACENCY_SCHEMA_VERSION = "qwen_rewire_proposed_adjacency.v1"
REWIRE_PROPOSAL_FILENAME = "rewire_proposal.json"
PROPOSED_ADJACENCY_FILENAME = "proposed_adjacency.json"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (str(edge["src_id"]), str(edge["dst_id"]), str(edge.get("relation", "")))


def _edge_id(edge: dict[str, Any]) -> str:
    return str(edge["edge_id"])


def _out_degrees(edges: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(edge["src_id"]) for edge in edges)
    return {src_id: int(count) for src_id, count in sorted(counts.items())}


def _max_out_degree(edges: list[dict[str, Any]]) -> int:
    return max(_out_degrees(edges).values(), default=0)


def _node_count(edges: list[dict[str, Any]]) -> int:
    return len({str(edge["src_id"]) for edge in edges} | {str(edge["dst_id"]) for edge in edges})


def _load_prior_config(eval_output_dir: Path) -> dict[str, Any]:
    path = eval_output_dir / "prior_config.json"
    if path.exists():
        return _read_json(path)
    return {}


def _active_utility_by_edge(edge_trace_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validate_edge_trace_report(edge_trace_report)
    summary = edge_trace_report["edge_utility_summary"]
    return {str(row["edge_id"]): dict(row) for row in summary["ranked_edges"]}


def _candidate_edges_from_g_pool(eval_output_dir: Path) -> list[dict[str, Any]]:
    handoff = load_v25_handoff(eval_output_dir)
    prior_config = _load_prior_config(eval_output_dir)
    source_weight_graph_dir = Path(str(handoff.get("source_weight_graph_dir") or prior_config.get("source_weight_graph_dir")))
    graph_scope = str(prior_config.get("graph_scope", "attention_mlp_moe"))
    edge_score_name = str(prior_config.get("edge_score_name", "normalized_frobenius"))
    result = read_weight_graph_artifacts(source_weight_graph_dir)
    candidates = build_candidate_adjacency(
        result,
        graph_scope=graph_scope,
        edge_score_name=edge_score_name,
    )
    return [edge.as_dict() for edge in candidates.edges]


def _select_swaps(
    *,
    active_edges: list[dict[str, Any]],
    candidate_edges: list[dict[str, Any]],
    utility_by_edge: dict[str, dict[str, Any]],
    max_swaps: int,
) -> list[dict[str, Any]]:
    active_ids = {_edge_id(edge) for edge in active_edges}
    active_keys = {_edge_key(edge) for edge in active_edges}
    inactive_by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_edges:
        if _edge_id(candidate) in active_ids or _edge_key(candidate) in active_keys:
            continue
        inactive_by_src[str(candidate["src_id"])].append(dict(candidate))
    for src_id in inactive_by_src:
        inactive_by_src[src_id].sort(key=lambda edge: (-float(edge["weight"]), str(edge.get("relation", "")), str(edge["dst_id"]), str(edge["edge_id"])))

    active_sorted = sorted(
        active_edges,
        key=lambda edge: (
            float(utility_by_edge.get(_edge_id(edge), {}).get("utility_score", 0.0)),
            str(edge.get("relation", "")),
            str(edge["src_id"]),
            str(edge["dst_id"]),
            str(edge["edge_id"]),
        ),
    )
    swaps: list[dict[str, Any]] = []
    used_add_ids: set[str] = set()
    for drop in active_sorted:
        if len(swaps) >= max_swaps:
            break
        src_id = str(drop["src_id"])
        replacement = None
        while inactive_by_src.get(src_id):
            candidate = inactive_by_src[src_id].pop(0)
            if _edge_id(candidate) not in used_add_ids:
                replacement = candidate
                break
        if replacement is None:
            continue
        used_add_ids.add(_edge_id(replacement))
        utility = utility_by_edge.get(_edge_id(drop), {})
        swaps.append(
            {
                "swap_index": len(swaps),
                "drop_edge_id": _edge_id(drop),
                "add_edge_id": _edge_id(replacement),
                "src_id": src_id,
                "drop_dst_id": str(drop["dst_id"]),
                "add_dst_id": str(replacement["dst_id"]),
                "drop_relation": drop.get("relation"),
                "add_relation": replacement.get("relation"),
                "drop_weight": float(drop["weight"]),
                "add_weight": float(replacement["weight"]),
                "drop_utility_score": float(utility.get("utility_score", 0.0)),
                "reason": "replace_low_utility_active_edge_with_same_source_inactive_graph_prior_candidate",
                "drop_edge": dict(drop),
                "add_edge": dict(replacement),
            }
        )
    return swaps


def _proposed_edges(active_edges: list[dict[str, Any]], swaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drop_ids = {str(swap["drop_edge_id"]) for swap in swaps}
    result = [dict(edge) for edge in active_edges if _edge_id(edge) not in drop_ids]
    for swap in swaps:
        edge = dict(swap["add_edge"])
        edge["proposal_source"] = "v26_low_utility_rewire_candidate"
        result.append(edge)
    return sorted(result, key=lambda edge: (str(edge["src_id"]), str(edge.get("relation", "")), str(edge["dst_id"]), str(edge["edge_id"])))


def _build_proposed_adjacency_payload(
    *,
    base_adjacency: dict[str, Any],
    proposed_edges: list[dict[str, Any]],
    swaps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": PROPOSED_ADJACENCY_SCHEMA_VERSION,
        "status": "bounded_rewire_proposed_adjacency_ok",
        "base_adjacency_name": base_adjacency["adjacency_name"],
        "proposal_name": f"{base_adjacency['adjacency_name']}_v26_p1_rewire_proposal",
        "source": "G_pool",
        "k": int(base_adjacency["k"]),
        "edge_count": len(proposed_edges),
        "base_edge_count": int(base_adjacency["edge_count"]),
        "node_count": _node_count(proposed_edges),
        "base_node_count": int(base_adjacency["node_count"]),
        "max_out_degree": _max_out_degree(proposed_edges),
        "bounded": True,
        "selection_policy": "drop_low_utility_same_source_add_graph_prior_candidate",
        "topology_mutated": False,
        "accepted": False,
        "swap_count": len(swaps),
        "edges": proposed_edges,
    }


def build_rewire_proposal_report(
    eval_output_dir: str | Path,
    edge_trace_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    k: int | None = None,
    adjacency_name: str | None = None,
    max_swaps: int = 1,
) -> dict[str, Any]:
    if max_swaps < 1:
        raise ValueError(f"max_swaps must be >= 1, got {max_swaps}")
    eval_dir = Path(eval_output_dir)
    trace_dir = Path(edge_trace_dir)
    edge_trace_report = load_edge_trace_report(trace_dir)
    validate_edge_trace_report(edge_trace_report)
    if adjacency_name is None and k is None:
        adjacency_name = str(edge_trace_report.get("adjacency_name"))
    active_adjacency = load_selected_adjacency(eval_dir, adjacency_name=adjacency_name, k=k)
    active_summary = validate_selected_adjacency(active_adjacency)
    if active_summary["adjacency_name"] != edge_trace_report.get("adjacency_name"):
        raise ValueError("edge trace adjacency does not match selected active adjacency")
    active_edges = [dict(edge) for edge in active_adjacency["edges"]]
    candidate_edges = _candidate_edges_from_g_pool(eval_dir)
    utility_by_edge = _active_utility_by_edge(edge_trace_report)
    swaps = _select_swaps(
        active_edges=active_edges,
        candidate_edges=candidate_edges,
        utility_by_edge=utility_by_edge,
        max_swaps=max_swaps,
    )
    proposed_edges = _proposed_edges(active_edges, swaps)
    proposed_payload = _build_proposed_adjacency_payload(
        base_adjacency=active_adjacency,
        proposed_edges=proposed_edges,
        swaps=swaps,
    )
    bounded = (
        proposed_payload["edge_count"] <= active_summary["edge_count"]
        and proposed_payload["max_out_degree"] <= active_summary["k"]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "bounded_rewire_proposal_ok",
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "topology_mutated": False,
        "accepted": False,
        "promotion_eligible": False,
        "eval_output_dir": str(eval_dir),
        "edge_trace_dir": str(trace_dir),
        "base_adjacency_name": active_summary["adjacency_name"],
        "k": active_summary["k"],
        "base_edge_count": active_summary["edge_count"],
        "candidate_edge_count": len(candidate_edges),
        "inactive_candidate_count": len([edge for edge in candidate_edges if _edge_id(edge) not in {_edge_id(active) for active in active_edges}]),
        "max_swaps": max_swaps,
        "swap_count": len(swaps),
        "proposal_bounded": bounded,
        "proposed_edge_count": proposed_payload["edge_count"],
        "proposed_max_out_degree": proposed_payload["max_out_degree"],
        "swaps": swaps,
        "proposed_adjacency": proposed_payload,
        "artifacts": {
            "rewire_proposal": REWIRE_PROPOSAL_FILENAME,
            "proposed_adjacency": PROPOSED_ADJACENCY_FILENAME,
        },
        "finite": all(float(swap["drop_utility_score"]) >= 0.0 for swap in swaps),
        "note": "v26 P1 bounded proposal only; active topology is not mutated and acceptance is not run",
    }
    validate_rewire_proposal_report(report)
    return report


def _serializable_report(report: dict[str, Any]) -> dict[str, Any]:
    clean = dict(report)
    clean.pop("proposed_adjacency", None)
    return clean


def validate_rewire_proposal_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad rewire proposal schema_version={report.get('schema_version')!r}")
    if report.get("status") != "bounded_rewire_proposal_ok":
        raise ValueError(f"bad rewire proposal status={report.get('status')!r}")
    for key, expected in {
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "topology_mutated": False,
        "accepted": False,
        "promotion_eligible": False,
        "proposal_bounded": True,
        "finite": True,
    }.items():
        if bool(report.get(key)) is not expected:
            raise ValueError(f"rewire proposal {key} must be {expected}")
    if int(report.get("proposed_edge_count", -1)) > int(report.get("base_edge_count", -2)):
        raise ValueError("proposed edge count must not exceed base edge count")
    if int(report.get("proposed_max_out_degree", -1)) > int(report.get("k", -2)):
        raise ValueError("proposed max_out_degree must not exceed k")
    swaps = report.get("swaps")
    if not isinstance(swaps, list):
        raise ValueError("rewire proposal swaps must be a list")
    if int(report.get("swap_count", -1)) != len(swaps):
        raise ValueError("swap_count must match len(swaps)")
    if len(swaps) > int(report.get("max_swaps", -1)):
        raise ValueError("swap_count must not exceed max_swaps")
    seen_drop: set[str] = set()
    seen_add: set[str] = set()
    for swap in swaps:
        if not isinstance(swap, dict):
            raise ValueError("swap rows must be objects")
        drop_id = str(swap.get("drop_edge_id"))
        add_id = str(swap.get("add_edge_id"))
        if drop_id in seen_drop or add_id in seen_add:
            raise ValueError("swap drop/add edge ids must be unique")
        seen_drop.add(drop_id)
        seen_add.add(add_id)
        if str(swap.get("src_id")) != str(swap.get("add_edge", {}).get("src_id")):
            raise ValueError("swap replacement must preserve source id")
        if float(swap.get("drop_utility_score", -1.0)) < 0.0:
            raise ValueError("drop utility score must be non-negative")
    proposed = report.get("proposed_adjacency")
    if proposed is not None:
        if not isinstance(proposed, dict):
            raise ValueError("proposed_adjacency must be an object")
        if proposed.get("schema_version") != PROPOSED_ADJACENCY_SCHEMA_VERSION:
            raise ValueError("bad proposed adjacency schema_version")
        if bool(proposed.get("topology_mutated")) or bool(proposed.get("accepted")):
            raise ValueError("proposed adjacency must not be accepted or applied")
        if int(proposed.get("edge_count", -1)) != int(report["proposed_edge_count"]):
            raise ValueError("proposed adjacency edge_count mismatch")
        if int(proposed.get("max_out_degree", -1)) > int(report["k"]):
            raise ValueError("proposed adjacency max_out_degree must be <= k")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "rewire_proposal_report_valid",
        "swap_count": len(swaps),
        "proposal_bounded": True,
        "accepted": False,
    }


def write_rewire_proposal_report(report: dict[str, Any], output_dir: str | Path) -> None:
    validate_rewire_proposal_report(report)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / PROPOSED_ADJACENCY_FILENAME, report["proposed_adjacency"])
    _write_json(out / REWIRE_PROPOSAL_FILENAME, _serializable_report(report))


def load_rewire_proposal_report(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / REWIRE_PROPOSAL_FILENAME)


def load_proposed_adjacency(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / PROPOSED_ADJACENCY_FILENAME)


def run_and_write_rewire_proposal_report(
    eval_output_dir: str | Path,
    edge_trace_dir: str | Path,
    output_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    max_swaps: int = 1,
) -> dict[str, Any]:
    report = build_rewire_proposal_report(
        eval_output_dir,
        edge_trace_dir,
        output_dir=output_dir,
        k=k,
        adjacency_name=adjacency_name,
        max_swaps=max_swaps,
    )
    write_rewire_proposal_report(report, output_dir)
    return _serializable_report(report)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v26 P1 bounded sparse-topology rewiring proposals from edge utility traces.")
    parser.add_argument("--eval-output-dir", required=True)
    parser.add_argument("--edge-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--k", type=_positive_int, default=None)
    group.add_argument("--adjacency-name", default=None)
    parser.add_argument("--max-swaps", type=_positive_int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_and_write_rewire_proposal_report(
            args.eval_output_dir,
            args.edge_trace_dir,
            args.output_dir,
            k=args.k,
            adjacency_name=args.adjacency_name,
            max_swaps=args.max_swaps,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "rewire_proposal": str(Path(args.output_dir) / REWIRE_PROPOSAL_FILENAME),
        "proposed_adjacency": str(Path(args.output_dir) / PROPOSED_ADJACENCY_FILENAME),
        "base_adjacency_name": report["base_adjacency_name"],
        "k": report["k"],
        "base_edge_count": report["base_edge_count"],
        "proposed_edge_count": report["proposed_edge_count"],
        "proposed_max_out_degree": report["proposed_max_out_degree"],
        "swap_count": report["swap_count"],
        "proposal_bounded": report["proposal_bounded"],
        "topology_mutated": report["topology_mutated"],
        "accepted": report["accepted"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
