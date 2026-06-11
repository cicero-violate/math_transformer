from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.qwen_distillation_targets import (
    checksum_json,
    iter_frozen_target_rows,
    load_frozen_distillation_targets_manifest,
    validate_frozen_distillation_targets,
)
from src.qwen_sparse_student_runtime import compute_feature_mse, run_fixed_topology_forward_features


SCHEMA_VERSION = "qwen_frozen_target_eval.v1"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _require_false(data: dict[str, Any], key: str) -> None:
    if bool(data.get(key, True)):
        raise ValueError(f"frozen target manifest {key} must be false")


def _resolve_adjacency_request(
    manifest: dict[str, Any],
    *,
    k: int | None,
    adjacency_name: str | None,
) -> tuple[int, str]:
    manifest_k = int(manifest["selected_adjacency_k"])
    manifest_name = str(manifest["selected_adjacency_name"])
    if k is not None and k != manifest_k:
        raise ValueError(f"requested k={k} does not match frozen target selected_adjacency_k={manifest_k}")
    if adjacency_name is not None and adjacency_name != manifest_name:
        raise ValueError(
            f"requested adjacency_name={adjacency_name!r} does not match frozen target selected_adjacency_name={manifest_name!r}"
        )
    return manifest_k, manifest_name


def _validate_eval_inputs(manifest: dict[str, Any], *, steps: int) -> None:
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if manifest.get("target_type") != "node_features":
        raise ValueError(f"unsupported frozen target_type={manifest.get('target_type')!r}")
    _require_false(manifest, "teacher_checkpoint_loaded_at_runtime")
    _require_false(manifest, "teacher_inference_runtime_required")
    _require_false(manifest, "student_training_started")
    _require_false(manifest, "promotion_eligible")


def evaluate_fixed_topology_against_frozen_targets(
    eval_output_dir: str | Path,
    targets_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    steps: int = 1,
) -> dict[str, Any]:
    target_summary = validate_frozen_distillation_targets(targets_dir)
    manifest = load_frozen_distillation_targets_manifest(targets_dir)
    _validate_eval_inputs(manifest, steps=steps)
    resolved_k, resolved_name = _resolve_adjacency_request(manifest, k=k, adjacency_name=adjacency_name)
    feature_dim = int(manifest["feature_dim"])

    rows: list[dict[str, Any]] = []
    mse_values: list[float] = []
    l1_values: list[float] = []
    finite = True
    for row in iter_frozen_target_rows(targets_dir):
        seed = int(row["seed"])
        forward = run_fixed_topology_forward_features(
            eval_output_dir,
            k=resolved_k,
            adjacency_name=None,
            feature_dim=feature_dim,
            steps=steps,
            seed=seed,
        )
        losses = compute_feature_mse(forward["output_features"], row["target_features"])
        row_finite = bool(forward["summary"]["finite"]) and bool(losses["finite"])
        if not row_finite or not math.isfinite(losses["mse"]) or not math.isfinite(losses["l1"]):
            finite = False
        row_summary = {
            "row_id": row["row_id"],
            "seed": seed,
            "loss_mse": losses["mse"],
            "loss_l1": losses["l1"],
            "input_checksum": forward["summary"]["input_checksum"],
            "output_checksum": forward["summary"]["output_checksum"],
            "target_checksum": checksum_json(row["target_features"]),
            "finite": row_finite,
        }
        rows.append(row_summary)
        mse_values.append(losses["mse"])
        l1_values.append(losses["l1"])

    if not rows:
        raise ValueError("frozen target eval requires at least one row")
    if not finite:
        raise ValueError("frozen target eval produced non-finite row losses")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_target_eval_ok",
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": False,
        "eval_output_dir": str(Path(eval_output_dir)),
        "targets_dir": str(Path(targets_dir)),
        "adjacency_name": resolved_name,
        "k": resolved_k,
        "feature_dim": feature_dim,
        "steps": steps,
        "row_count": len(rows),
        "loss_mse_mean": _mean(mse_values),
        "loss_mse_min": min(mse_values),
        "loss_mse_max": max(mse_values),
        "loss_l1_mean": _mean(l1_values),
        "loss_l1_min": min(l1_values),
        "loss_l1_max": max(l1_values),
        "finite": finite,
        "target_rows_sha256": target_summary["target_rows_sha256"],
        "rows": rows,
        "note": "frozen target loss evaluation only; no training or online teacher inference",
    }


def write_frozen_target_eval_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_and_write_frozen_target_eval_report(
    eval_output_dir: str | Path,
    targets_dir: str | Path,
    output_path: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    steps: int = 1,
) -> dict[str, Any]:
    report = evaluate_fixed_topology_against_frozen_targets(
        eval_output_dir,
        targets_dir,
        k=k,
        adjacency_name=adjacency_name,
        steps=steps,
    )
    write_frozen_target_eval_report(report, output_path)
    return report
