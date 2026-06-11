from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency

_TARGET_MODES = {"identity", "zero", "scaled_identity"}


def _checksum_features(features: dict[str, list[float]]) -> str:
    payload = json.dumps(features, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def build_adjacency_index(selected_adjacency: dict[str, Any]) -> dict[str, Any]:
    summary = validate_selected_adjacency(selected_adjacency)
    edges = selected_adjacency["edges"]
    node_ids = sorted({str(edge["src_id"]) for edge in edges} | {str(edge["dst_id"]) for edge in edges})

    outgoing_weight_total: dict[str, float] = defaultdict(float)
    for edge in edges:
        src_id = str(edge["src_id"])
        outgoing_weight_total[src_id] += abs(_as_finite_float(edge["weight"], field="edge.weight"))

    outgoing: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        src_id = str(edge["src_id"])
        dst_id = str(edge["dst_id"])
        weight = _as_finite_float(edge["weight"], field="edge.weight")
        denom = outgoing_weight_total[src_id]
        normalized_weight = weight / denom if denom > 0.0 else 0.0
        outgoing[src_id].append(
            {
                "edge_id": edge["edge_id"],
                "src_id": src_id,
                "dst_id": dst_id,
                "weight": weight,
                "normalized_weight": normalized_weight,
            }
        )

    return {
        "adjacency_name": summary["adjacency_name"],
        "k": summary["k"],
        "node_ids": node_ids,
        "outgoing": outgoing,
        "edge_count": summary["edge_count"],
        "node_count": summary["node_count"],
        "max_out_degree": summary["max_out_degree"],
        "selection_policy": summary["selection_policy"],
        "source": summary["source"],
        "bounded": summary["bounded"],
    }


def initialize_node_features(node_ids: list[str], dim: int, seed: int = 0) -> dict[str, list[float]]:
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    rng = random.Random(seed)
    return {node_id: [rng.uniform(-1.0, 1.0) for _ in range(dim)] for node_id in sorted(node_ids)}


def _propagate_once(
    features: dict[str, list[float]],
    adjacency_index: dict[str, Any],
    *,
    residual: float = 1.0,
) -> dict[str, list[float]]:
    next_features = {
        node_id: [residual * value for value in vector]
        for node_id, vector in features.items()
    }
    for src_id, edges in adjacency_index["outgoing"].items():
        src_features = features[src_id]
        for edge in edges:
            dst_id = edge["dst_id"]
            scale = edge["normalized_weight"]
            dst_features = next_features[dst_id]
            for idx, value in enumerate(src_features):
                dst_features[idx] += scale * value
    return next_features


def _features_are_finite(features: dict[str, list[float]]) -> bool:
    return all(math.isfinite(value) for vector in features.values() for value in vector)


def _changed_node_count(before: dict[str, list[float]], after: dict[str, list[float]]) -> int:
    changed = 0
    for node_id, before_vector in before.items():
        after_vector = after[node_id]
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(before_vector, after_vector)):
            changed += 1
    return changed


def _forward_summary(
    adjacency_index: dict[str, Any],
    *,
    feature_dim: int,
    steps: int,
    input_features: dict[str, list[float]],
    output_features: dict[str, list[float]],
) -> dict[str, Any]:
    finite = _features_are_finite(output_features)
    changed_node_count = _changed_node_count(input_features, output_features)
    if adjacency_index["edge_count"] > 0 and changed_node_count <= 0:
        raise ValueError("fixed-topology forward did not change any node features")

    return {
        "status": "fixed_topology_forward_ok",
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "adjacency_name": adjacency_index["adjacency_name"],
        "k": adjacency_index["k"],
        "feature_dim": feature_dim,
        "steps": steps,
        "node_count": adjacency_index["node_count"],
        "edge_count": adjacency_index["edge_count"],
        "max_out_degree": adjacency_index["max_out_degree"],
        "input_checksum": _checksum_features(input_features),
        "output_checksum": _checksum_features(output_features),
        "changed_node_count": changed_node_count,
        "finite": finite,
        "ready_for_v25_distillation": False,
        "note": "dry-run fixed-topology forward only; no teacher distillation yet",
    }


def run_fixed_topology_forward_features(
    eval_output_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    if feature_dim < 1:
        raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    selected_adjacency = load_selected_adjacency(eval_output_dir, adjacency_name=adjacency_name, k=k)
    adjacency_index = build_adjacency_index(selected_adjacency)
    node_ids = adjacency_index["node_ids"]
    input_features = initialize_node_features(node_ids, feature_dim, seed=seed)

    output_features = input_features
    for _ in range(steps):
        output_features = _propagate_once(output_features, adjacency_index)

    return {
        "summary": _forward_summary(
            adjacency_index,
            feature_dim=feature_dim,
            steps=steps,
            input_features=input_features,
            output_features=output_features,
        ),
        "input_features": input_features,
        "output_features": output_features,
    }


def run_fixed_topology_forward(
    eval_output_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    return run_fixed_topology_forward_features(
        eval_output_dir,
        k=k,
        adjacency_name=adjacency_name,
        feature_dim=feature_dim,
        steps=steps,
        seed=seed,
    )["summary"]


def synthetic_target_features(
    input_features: dict[str, list[float]],
    *,
    mode: str = "identity",
    scale: float = 1.0,
) -> dict[str, list[float]]:
    if mode not in _TARGET_MODES:
        raise ValueError(f"target_mode must be one of {sorted(_TARGET_MODES)}, got {mode!r}")
    scale_value = _as_finite_float(scale, field="target_scale")
    if mode == "identity":
        return {node_id: list(vector) for node_id, vector in input_features.items()}
    if mode == "zero":
        return {node_id: [0.0 for _ in vector] for node_id, vector in input_features.items()}
    return {node_id: [scale_value * value for value in vector] for node_id, vector in input_features.items()}


def compute_feature_mse(
    output_features: dict[str, list[float]],
    target_features: dict[str, list[float]],
) -> dict[str, Any]:
    if set(output_features) != set(target_features):
        raise ValueError("output and target node ids must match")
    if not output_features:
        raise ValueError("features must contain at least one node")

    feature_dim: int | None = None
    squared_error = 0.0
    absolute_error = 0.0
    value_count = 0
    for node_id in sorted(output_features):
        output_vector = output_features[node_id]
        target_vector = target_features[node_id]
        if feature_dim is None:
            feature_dim = len(output_vector)
            if feature_dim < 1:
                raise ValueError("feature vectors must be non-empty")
        if len(output_vector) != feature_dim or len(target_vector) != feature_dim:
            raise ValueError("output and target feature dimensions must match")
        for output_value, target_value in zip(output_vector, target_vector):
            output_number = _as_finite_float(output_value, field="output_feature")
            target_number = _as_finite_float(target_value, field="target_feature")
            diff = output_number - target_number
            squared_error += diff * diff
            absolute_error += abs(diff)
            value_count += 1

    mse = squared_error / value_count
    l1 = absolute_error / value_count
    finite = math.isfinite(mse) and math.isfinite(l1)
    if not finite:
        raise ValueError("loss values must be finite")

    return {
        "mse": mse,
        "l1": l1,
        "node_count": len(output_features),
        "feature_dim": feature_dim,
        "finite": finite,
    }


def run_fixed_topology_loss_dry_run(
    eval_output_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    seed: int = 0,
    target_mode: str = "identity",
    target_scale: float = 1.0,
) -> dict[str, Any]:
    forward = run_fixed_topology_forward_features(
        eval_output_dir,
        k=k,
        adjacency_name=adjacency_name,
        feature_dim=feature_dim,
        steps=steps,
        seed=seed,
    )
    target_features = synthetic_target_features(
        forward["input_features"],
        mode=target_mode,
        scale=target_scale,
    )
    losses = compute_feature_mse(forward["output_features"], target_features)
    summary = forward["summary"]
    finite = bool(summary["finite"]) and bool(losses["finite"])
    if not finite:
        raise ValueError("loss dry run produced non-finite values")

    return {
        "status": "fixed_topology_loss_dry_run_ok",
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "adjacency_name": summary["adjacency_name"],
        "k": summary["k"],
        "feature_dim": feature_dim,
        "steps": steps,
        "target_mode": target_mode,
        "loss_mse": losses["mse"],
        "loss_l1": losses["l1"],
        "finite": finite,
        "input_checksum": summary["input_checksum"],
        "output_checksum": summary["output_checksum"],
        "target_checksum": _checksum_features(target_features),
        "ready_for_teacher_distillation": False,
        "promotion_eligible": False,
        "note": "synthetic teacher-free loss dry run only; no teacher distillation yet",
    }
