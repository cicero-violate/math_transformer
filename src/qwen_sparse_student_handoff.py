from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


HANDOFF_SCHEMA_VERSION = "qwen_v25_handoff.v1"
SELECTED_INDEX_SCHEMA_VERSION = "qwen_selected_adjacency_index.v1"
SELECTED_ADJACENCY_SCHEMA_VERSION = "qwen_selected_adjacency.v1"

_FORBIDDEN_PAYLOAD_KEYS = {
    "payload",
    "raw_payload",
    "raw_tensor",
    "raw_tensor_payload",
    "raw_weight",
    "raw_weight_payload",
    "tensor_data",
    "tensor_payload",
    "tensor_values",
    "values",
    "weight_payload",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _reject_forbidden_payload_keys(value: Any, *, path: str = "edge") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"raw tensor payload field forbidden at {path}.{key_text}")
            _reject_forbidden_payload_keys(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_forbidden_payload_keys(child, path=f"{path}[{idx}]")


def load_v25_handoff(eval_output_dir: str | Path) -> dict[str, Any]:
    base = Path(eval_output_dir)
    handoff = _read_json(base / "v25_handoff_manifest.json")
    if handoff.get("schema_version") != HANDOFF_SCHEMA_VERSION:
        raise ValueError(f"bad handoff schema_version={handoff.get('schema_version')!r}")
    if bool(handoff.get("teacher_checkpoint_loaded", True)):
        raise ValueError("handoff must not load teacher checkpoint")
    if bool(handoff.get("raw_weight_payload_in_graph", True)):
        raise ValueError("handoff raw_weight_payload_in_graph must be false")
    if not bool(handoff.get("bounded_active_adjacency", False)):
        raise ValueError("handoff bounded_active_adjacency must be true")
    if bool(handoff.get("student_training_started", True)):
        raise ValueError("handoff student_training_started must be false")
    return handoff


def _load_selected_index(eval_output_dir: str | Path, handoff: dict[str, Any] | None = None) -> dict[str, Any]:
    base = Path(eval_output_dir)
    h = handoff or load_v25_handoff(base)
    rel = h.get("selected_adjacency_index", "selected_adjacencies/index.json")
    index = _read_json(base / str(rel))
    if index.get("schema_version") != SELECTED_INDEX_SCHEMA_VERSION:
        raise ValueError(f"bad selected adjacency index schema_version={index.get('schema_version')!r}")
    if not bool(index.get("bounded", False)):
        raise ValueError("selected adjacency index must be bounded")
    rows = index.get("adjacencies")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selected adjacency index must contain adjacencies")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("selected adjacency index rows must be objects")
        name = str(row.get("adjacency_name", ""))
        if not name.startswith("qwen_topk_k"):
            raise ValueError(f"v25 handoff may only reference qwen_topk adjacencies, got {name!r}")
    return index


def list_selected_adjacencies(eval_output_dir: str | Path) -> list[dict[str, Any]]:
    handoff = load_v25_handoff(eval_output_dir)
    index = _load_selected_index(eval_output_dir, handoff)
    return [dict(row) for row in index["adjacencies"]]


def validate_selected_adjacency(adjacency: dict[str, Any]) -> dict[str, Any]:
    if adjacency.get("schema_version") != SELECTED_ADJACENCY_SCHEMA_VERSION:
        raise ValueError(f"bad selected adjacency schema_version={adjacency.get('schema_version')!r}")
    if not bool(adjacency.get("bounded", False)):
        raise ValueError("selected adjacency must be bounded")
    if adjacency.get("source") != "G_0":
        raise ValueError(f"selected adjacency source must be G_0, got {adjacency.get('source')!r}")
    name = str(adjacency.get("adjacency_name", ""))
    if not name.startswith("qwen_topk_k"):
        raise ValueError(f"selected adjacency must be qwen_topk, got {name!r}")
    k = adjacency.get("k")
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"selected adjacency k must be a positive integer, got {k!r}")
    edges = adjacency.get("edges")
    if not isinstance(edges, list):
        raise ValueError("selected adjacency edges must be a list")
    if int(adjacency.get("edge_count", -1)) != len(edges):
        raise ValueError("selected adjacency edge_count does not match len(edges)")

    out_degree: Counter[str] = Counter()
    node_ids: set[str] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("selected adjacency edge rows must be objects")
        missing = [key for key in ("edge_id", "src_id", "dst_id", "relation", "weight", "score_name") if key not in edge]
        if missing:
            raise ValueError(f"selected adjacency edge missing fields {missing}")
        _reject_forbidden_payload_keys(edge)
        src_id = str(edge["src_id"])
        dst_id = str(edge["dst_id"])
        out_degree[src_id] += 1
        node_ids.add(src_id)
        node_ids.add(dst_id)
    max_out_degree = max(out_degree.values(), default=0)
    if max_out_degree > k:
        raise ValueError(f"selected adjacency max_out_degree={max_out_degree} exceeds k={k}")
    node_count = int(adjacency.get("node_count", 0))
    if node_count != len(node_ids):
        raise ValueError(f"selected adjacency node_count={node_count} does not match unique endpoint count={len(node_ids)}")

    return {
        "adjacency_name": name,
        "k": k,
        "edge_count": len(edges),
        "node_count": len(node_ids),
        "max_out_degree": max_out_degree,
        "selection_policy": adjacency.get("selection_policy"),
        "source": adjacency.get("source"),
        "bounded": True,
    }


def _resolve_selected_index_row(
    rows: list[dict[str, Any]],
    *,
    adjacency_name: str | None,
    k: int | None,
) -> dict[str, Any]:
    if adjacency_name is not None and k is not None:
        raise ValueError("request by adjacency_name or k, not both")
    if adjacency_name is not None:
        matches = [row for row in rows if row.get("adjacency_name") == adjacency_name]
    elif k is not None:
        matches = [row for row in rows if row.get("k") == k]
    else:
        matches = rows[:1]
    if not matches:
        target = adjacency_name if adjacency_name is not None else f"k={k}"
        raise ValueError(f"selected adjacency not found: {target}")
    return matches[0]


def load_selected_adjacency(
    eval_output_dir: str | Path,
    adjacency_name: str | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    base = Path(eval_output_dir)
    handoff = load_v25_handoff(base)
    index = _load_selected_index(base, handoff)
    row = _resolve_selected_index_row(index["adjacencies"], adjacency_name=adjacency_name, k=k)
    path = base / str(row["path"])
    adjacency = _read_json(path)
    summary = validate_selected_adjacency(adjacency)
    if row.get("adjacency_name") != summary["adjacency_name"]:
        raise ValueError("selected adjacency index/name mismatch")
    if row.get("k") != summary["k"]:
        raise ValueError("selected adjacency index/k mismatch")
    if row.get("edge_count") != summary["edge_count"]:
        raise ValueError("selected adjacency index/edge_count mismatch")
    if row.get("node_count") != summary["node_count"]:
        raise ValueError("selected adjacency index/node_count mismatch")
    return adjacency


def build_fixed_topology_student_stub(
    eval_output_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
) -> dict[str, Any]:
    try:
        handoff = load_v25_handoff(eval_output_dir)
    except FileNotFoundError as exc:
        return {
            "status": "handoff_missing",
            "error": str(exc),
            "student_training_started": False,
            "teacher_checkpoint_loaded": False,
        }
    adjacency = load_selected_adjacency(eval_output_dir, adjacency_name=adjacency_name, k=k)
    summary = validate_selected_adjacency(adjacency)
    return {
        "status": "fixed_topology_stub_ok",
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "adjacency_name": summary["adjacency_name"],
        "k": summary["k"],
        "edge_count": summary["edge_count"],
        "node_count": summary["node_count"],
        "max_out_degree": summary["max_out_degree"],
        "selection_policy": summary["selection_policy"],
        "source": summary["source"],
        "ready_for_v25_distillation": True,
        "promotion_required_before_deploy": bool(handoff.get("promotion_required_before_deploy", True)),
        "note": "fixed-topology runtime contract only; no distillation training yet",
    }
