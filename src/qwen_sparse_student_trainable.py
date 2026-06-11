from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.qwen_distillation_targets import (
    iter_frozen_target_rows,
    load_frozen_distillation_targets_manifest,
    validate_frozen_distillation_targets,
)
from src.qwen_sparse_student_runtime import run_fixed_topology_forward_features


SCHEMA_VERSION = "qwen_sparse_student_trainable.v1"
LOOP_SCHEMA_VERSION = "qwen_sparse_student_trainable_loop.v1"


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_train_inputs(
    manifest: dict[str, Any],
    *,
    k: int | None,
    adjacency_name: str | None,
    steps: int,
    lr: float,
    alpha: float,
    beta: float,
) -> tuple[int, str, float, float, float]:
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    lr_value = _finite_float(lr, name="lr")
    if lr_value <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr}")
    alpha_value = _finite_float(alpha, name="alpha")
    beta_value = _finite_float(beta, name="beta")
    if manifest.get("target_type") != "node_features":
        raise ValueError(f"unsupported frozen target_type={manifest.get('target_type')!r}")
    for key in (
        "teacher_checkpoint_loaded_at_runtime",
        "teacher_inference_runtime_required",
        "raw_weight_payload_in_graph",
        "student_training_started",
        "promotion_eligible",
    ):
        if bool(manifest.get(key, True)):
            raise ValueError(f"frozen target manifest {key} must be false")
    manifest_k = int(manifest["selected_adjacency_k"])
    manifest_name = str(manifest["selected_adjacency_name"])
    if k is not None and k != manifest_k:
        raise ValueError(f"requested k={k} does not match frozen target selected_adjacency_k={manifest_k}")
    if adjacency_name is not None and adjacency_name != manifest_name:
        raise ValueError(
            f"requested adjacency_name={adjacency_name!r} does not match frozen target selected_adjacency_name={manifest_name!r}"
        )
    return manifest_k, manifest_name, lr_value, alpha_value, beta_value


def apply_scalar_affine_head(
    features: dict[str, list[float]],
    *,
    alpha: float,
    beta: float,
) -> dict[str, list[float]]:
    alpha_value = _finite_float(alpha, name="alpha")
    beta_value = _finite_float(beta, name="beta")
    return {
        node_id: [alpha_value * _finite_float(value, name="feature") + beta_value for value in vector]
        for node_id, vector in features.items()
    }


def scalar_affine_mse_and_grads(
    output_features: dict[str, list[float]],
    target_features: dict[str, list[float]],
    *,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    alpha_value = _finite_float(alpha, name="alpha")
    beta_value = _finite_float(beta, name="beta")
    if set(output_features) != set(target_features):
        raise ValueError("output and target node ids must match")
    if not output_features:
        raise ValueError("features must contain at least one node")

    squared_error = 0.0
    absolute_error = 0.0
    grad_alpha_total = 0.0
    grad_beta_total = 0.0
    value_count = 0
    feature_dim: int | None = None
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
            h = _finite_float(output_value, name="output_feature")
            target = _finite_float(target_value, name="target_feature")
            diff = alpha_value * h + beta_value - target
            squared_error += diff * diff
            absolute_error += abs(diff)
            grad_alpha_total += 2.0 * diff * h
            grad_beta_total += 2.0 * diff
            value_count += 1

    loss_mse = squared_error / value_count
    loss_l1 = absolute_error / value_count
    grad_alpha = grad_alpha_total / value_count
    grad_beta = grad_beta_total / value_count
    finite = all(math.isfinite(value) for value in (loss_mse, loss_l1, grad_alpha, grad_beta))
    if not finite:
        raise ValueError("scalar affine loss or gradients must be finite")
    return {
        "loss_mse": loss_mse,
        "loss_l1": loss_l1,
        "grad_alpha": grad_alpha,
        "grad_beta": grad_beta,
        "value_count": value_count,
        "finite": finite,
    }


def _aggregate_metrics(
    row_metrics: list[dict[str, Any]],
    *,
    include_grads: bool,
) -> dict[str, float | int]:
    total_values = sum(int(row["value_count"]) for row in row_metrics)
    if total_values <= 0:
        raise ValueError("train step requires at least one target value")
    loss_mse = sum(float(row["loss_mse"]) * int(row["value_count"]) for row in row_metrics) / total_values
    loss_l1 = sum(float(row["loss_l1"]) * int(row["value_count"]) for row in row_metrics) / total_values
    result: dict[str, float | int] = {
        "loss_mse": loss_mse,
        "loss_l1": loss_l1,
        "value_count": total_values,
    }
    if include_grads:
        result["grad_alpha"] = sum(float(row["grad_alpha"]) * int(row["value_count"]) for row in row_metrics) / total_values
        result["grad_beta"] = sum(float(row["grad_beta"]) * int(row["value_count"]) for row in row_metrics) / total_values
    return result


def run_scalar_affine_train_step(
    eval_output_dir: str | Path,
    targets_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    steps: int = 1,
    lr: float = 0.01,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> dict[str, Any]:
    validate_frozen_distillation_targets(targets_dir)
    manifest = load_frozen_distillation_targets_manifest(targets_dir)
    resolved_k, resolved_name, lr_value, alpha_before, beta_before = _validate_train_inputs(
        manifest,
        k=k,
        adjacency_name=adjacency_name,
        steps=steps,
        lr=lr,
        alpha=alpha,
        beta=beta,
    )
    feature_dim = int(manifest["feature_dim"])

    before_metrics: list[dict[str, Any]] = []
    output_target_pairs: list[tuple[dict[str, list[float]], dict[str, list[float]]]] = []
    for row in iter_frozen_target_rows(targets_dir):
        forward = run_fixed_topology_forward_features(
            eval_output_dir,
            k=resolved_k,
            feature_dim=feature_dim,
            steps=steps,
            seed=int(row["seed"]),
        )
        target_features = row["target_features"]
        metrics = scalar_affine_mse_and_grads(
            forward["output_features"],
            target_features,
            alpha=alpha_before,
            beta=beta_before,
        )
        before_metrics.append(metrics)
        output_target_pairs.append((forward["output_features"], target_features))

    if not before_metrics:
        raise ValueError("train step requires at least one frozen target row")
    before = _aggregate_metrics(before_metrics, include_grads=True)
    grad_alpha = float(before["grad_alpha"])
    grad_beta = float(before["grad_beta"])
    alpha_after = alpha_before - lr_value * grad_alpha
    beta_after = beta_before - lr_value * grad_beta

    after_metrics = [
        scalar_affine_mse_and_grads(output_features, target_features, alpha=alpha_after, beta=beta_after)
        for output_features, target_features in output_target_pairs
    ]
    after = _aggregate_metrics(after_metrics, include_grads=False)
    finite = all(
        math.isfinite(value)
        for value in (
            alpha_before,
            beta_before,
            grad_alpha,
            grad_beta,
            alpha_after,
            beta_after,
            float(before["loss_mse"]),
            float(before["loss_l1"]),
            float(after["loss_mse"]),
            float(after["loss_l1"]),
        )
    )
    if not finite:
        raise ValueError("train step produced non-finite values")

    loss_decreased = float(after["loss_mse"]) <= float(before["loss_mse"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "scalar_affine_train_step_ok",
        "student_training_started": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": False,
        "optimizer": "analytic_sgd_scalar_affine",
        "adjacency_name": resolved_name,
        "k": resolved_k,
        "steps": steps,
        "lr": lr_value,
        "row_count": len(before_metrics),
        "feature_dim": feature_dim,
        "alpha_before": alpha_before,
        "beta_before": beta_before,
        "grad_alpha": grad_alpha,
        "grad_beta": grad_beta,
        "alpha_after": alpha_after,
        "beta_after": beta_after,
        "loss_mse_before": float(before["loss_mse"]),
        "loss_l1_before": float(before["loss_l1"]),
        "loss_mse_after": float(after["loss_mse"]),
        "loss_l1_after": float(after["loss_l1"]),
        "loss_decreased": loss_decreased,
        "finite": finite,
        "note": "one-step scalar affine optimizer dry run only; no teacher inference or KL distillation",
    }


def run_scalar_affine_training_loop(
    eval_output_dir: str | Path,
    targets_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    steps: int = 1,
    train_steps: int = 5,
    lr: float = 0.01,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> dict[str, Any]:
    if train_steps < 1:
        raise ValueError(f"train_steps must be >= 1, got {train_steps}")

    alpha_initial = _finite_float(alpha, name="alpha")
    beta_initial = _finite_float(beta, name="beta")
    alpha_state = alpha_initial
    beta_state = beta_initial
    history: list[dict[str, Any]] = []
    for train_step in range(train_steps):
        step_report = run_scalar_affine_train_step(
            eval_output_dir,
            targets_dir,
            k=k,
            adjacency_name=adjacency_name,
            steps=steps,
            lr=lr,
            alpha=alpha_state,
            beta=beta_state,
        )
        history.append(
            {
                "train_step": train_step,
                "alpha_before": step_report["alpha_before"],
                "beta_before": step_report["beta_before"],
                "alpha_after": step_report["alpha_after"],
                "beta_after": step_report["beta_after"],
                "loss_mse_before": step_report["loss_mse_before"],
                "loss_mse_after": step_report["loss_mse_after"],
                "loss_l1_before": step_report["loss_l1_before"],
                "loss_l1_after": step_report["loss_l1_after"],
                "grad_alpha": step_report["grad_alpha"],
                "grad_beta": step_report["grad_beta"],
                "loss_decreased": step_report["loss_decreased"],
            }
        )
        alpha_state = float(step_report["alpha_after"])
        beta_state = float(step_report["beta_after"])

    first = history[0]
    last = history[-1]
    monotonic_nonincreasing = all(
        history[idx]["loss_mse_after"] <= history[idx]["loss_mse_before"]
        and (idx == 0 or history[idx]["loss_mse_before"] <= history[idx - 1]["loss_mse_after"])
        for idx in range(len(history))
    )
    finite = all(
        math.isfinite(float(row[key]))
        for row in history
        for key in (
            "alpha_before",
            "beta_before",
            "alpha_after",
            "beta_after",
            "loss_mse_before",
            "loss_mse_after",
            "loss_l1_before",
            "loss_l1_after",
            "grad_alpha",
            "grad_beta",
        )
    )
    if not finite:
        raise ValueError("training loop produced non-finite values")

    return {
        "schema_version": LOOP_SCHEMA_VERSION,
        "status": "scalar_affine_training_loop_ok",
        "student_training_started": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": False,
        "optimizer": "analytic_sgd_scalar_affine",
        "adjacency_name": step_report["adjacency_name"],
        "k": step_report["k"],
        "steps": steps,
        "train_steps": train_steps,
        "lr": step_report["lr"],
        "row_count": step_report["row_count"],
        "feature_dim": step_report["feature_dim"],
        "alpha_initial": alpha_initial,
        "beta_initial": beta_initial,
        "alpha_final": alpha_state,
        "beta_final": beta_state,
        "loss_mse_initial": first["loss_mse_before"],
        "loss_mse_final": last["loss_mse_after"],
        "loss_l1_initial": first["loss_l1_before"],
        "loss_l1_final": last["loss_l1_after"],
        "loss_decreased": last["loss_mse_after"] <= first["loss_mse_before"],
        "monotonic_nonincreasing": monotonic_nonincreasing,
        "finite": finite,
        "history": history,
        "note": "bounded scalar affine training loop only; no teacher inference or KL distillation",
    }


def write_scalar_affine_training_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_and_write_scalar_affine_training_report(
    eval_output_dir: str | Path,
    targets_dir: str | Path,
    output_path: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    steps: int = 1,
    train_steps: int = 5,
    lr: float = 0.01,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> dict[str, Any]:
    report = run_scalar_affine_training_loop(
        eval_output_dir,
        targets_dir,
        k=k,
        adjacency_name=adjacency_name,
        steps=steps,
        train_steps=train_steps,
        lr=lr,
        alpha=alpha,
        beta=beta,
    )
    write_scalar_affine_training_report(report, output_path)
    return report
