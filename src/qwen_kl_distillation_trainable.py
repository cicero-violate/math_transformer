from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.qwen_distillation_targets import checksum_json
from src.qwen_kl_distillation_eval import kl_divergence, project_features_to_logits
from src.qwen_logit_distillation_targets import (
    iter_frozen_logit_target_rows,
    load_frozen_logit_distillation_targets_manifest,
    softmax,
    validate_frozen_logit_distillation_targets,
)
from src.qwen_sparse_student_runtime import run_fixed_topology_forward_features


STEP_SCHEMA_VERSION = "qwen_kl_distillation_trainable_step.v1"
LOOP_SCHEMA_VERSION = "qwen_kl_distillation_trainable_loop.v1"


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_runtime_args(
    manifest: dict[str, Any],
    *,
    feature_dim: int,
    steps: int,
    lr: float,
    temperature: float | None,
) -> tuple[int, float, float]:
    if feature_dim < 1:
        raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    lr_value = _finite_float(lr, name="lr")
    if lr_value <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr!r}")
    vocab_size = int(manifest.get("vocab_size", 0))
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    temp = float(manifest["temperature"]) if temperature is None else _finite_float(temperature, name="temperature")
    if temp <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    return vocab_size, temp, lr_value


def _validate_bias(bias: list[float] | None, *, vocab_size: int) -> list[float]:
    if bias is None:
        return [0.0 for _ in range(vocab_size)]
    if len(bias) != vocab_size:
        raise ValueError(f"bias length={len(bias)} does not match vocab_size={vocab_size}")
    return [_finite_float(value, name="bias") for value in bias]


def apply_logit_bias(logits: list[float], bias: list[float]) -> list[float]:
    if len(logits) != len(bias):
        raise ValueError("logits and bias lengths must match")
    return [
        _finite_float(logit, name="logit") + _finite_float(bias_value, name="bias")
        for logit, bias_value in zip(logits, bias)
    ]


def kl_loss_and_bias_grads(
    base_logits: list[float],
    teacher_probabilities: list[float],
    *,
    bias: list[float],
    temperature: float = 1.0,
) -> dict[str, Any]:
    temp = _finite_float(temperature, name="temperature")
    if temp <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    if len(base_logits) != len(teacher_probabilities) or len(base_logits) != len(bias):
        raise ValueError("base_logits, teacher_probabilities, and bias lengths must match")
    logits = apply_logit_bias(base_logits, bias)
    student_probabilities = softmax(logits, temp)
    metrics = kl_divergence(list(teacher_probabilities), student_probabilities)
    grad_bias = [
        (student_prob - _finite_float(teacher_prob, name="teacher_probability")) / temp
        for student_prob, teacher_prob in zip(student_probabilities, teacher_probabilities)
    ]
    finite = bool(metrics["finite"]) and all(math.isfinite(value) for value in grad_bias + student_probabilities)
    if not finite:
        raise ValueError("KL loss or bias gradients must be finite")
    return {
        "kl": metrics["kl"],
        "cross_entropy": metrics["cross_entropy"],
        "entropy_teacher": metrics["entropy_teacher"],
        "grad_bias": grad_bias,
        "student_probabilities": student_probabilities,
        "finite": finite,
    }


def _collect_base_rows(
    eval_output_dir: str | Path,
    logit_targets_dir: str | Path,
    *,
    k: int | None,
    adjacency_name: str | None,
    feature_dim: int,
    steps: int,
    projection_seed: int,
    vocab_size: int,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_row in iter_frozen_logit_target_rows(logit_targets_dir):
        forward = run_fixed_topology_forward_features(
            eval_output_dir,
            k=k,
            adjacency_name=adjacency_name,
            feature_dim=feature_dim,
            steps=steps,
            seed=int(target_row["seed"]),
            device=device,
        )
        base_logits = project_features_to_logits(forward["output_features"], vocab_size=vocab_size, seed=projection_seed)
        rows.append(
            {
                "row_id": target_row["row_id"],
                "seed": target_row["seed"],
                "base_logits": base_logits,
                "teacher_probabilities": list(target_row["probabilities"]),
            }
        )
    if not rows:
        raise ValueError("KL train step requires at least one frozen logit target row")
    return rows


def _score_rows(
    rows: list[dict[str, Any]],
    *,
    bias: list[float],
    temperature: float,
) -> dict[str, Any]:
    grad = [0.0 for _ in bias]
    kl_values: list[float] = []
    cross_entropy_values: list[float] = []
    for row in rows:
        metrics = kl_loss_and_bias_grads(
            row["base_logits"],
            row["teacher_probabilities"],
            bias=bias,
            temperature=temperature,
        )
        kl_values.append(metrics["kl"])
        cross_entropy_values.append(metrics["cross_entropy"])
        for idx, value in enumerate(metrics["grad_bias"]):
            grad[idx] += value
    row_count = len(rows)
    grad = [value / row_count for value in grad]
    kl_mean = sum(kl_values) / row_count
    cross_entropy_mean = sum(cross_entropy_values) / row_count
    finite = all(math.isfinite(value) for value in grad + kl_values + cross_entropy_values)
    if not finite:
        raise ValueError("KL train step produced non-finite losses or gradients")
    return {
        "kl": kl_mean,
        "cross_entropy": cross_entropy_mean,
        "grad_bias": grad,
        "finite": finite,
    }


def _run_logit_bias_train_step_internal(
    eval_output_dir: str | Path,
    logit_targets_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    projection_seed: int = 0,
    lr: float = 0.1,
    bias: list[float] | None = None,
    temperature: float | None = None,
    device: str = "cpu",
) -> tuple[dict[str, Any], list[float]]:
    validate_frozen_logit_distillation_targets(logit_targets_dir)
    manifest = load_frozen_logit_distillation_targets_manifest(logit_targets_dir)
    vocab_size, temp, lr_value = _validate_runtime_args(
        manifest,
        feature_dim=feature_dim,
        steps=steps,
        lr=lr,
        temperature=temperature,
    )
    bias_before = _validate_bias(bias, vocab_size=vocab_size)
    rows = _collect_base_rows(
        eval_output_dir,
        logit_targets_dir,
        k=k,
        adjacency_name=adjacency_name,
        feature_dim=feature_dim,
        steps=steps,
        projection_seed=projection_seed,
        vocab_size=vocab_size,
        device=device,
    )
    before = _score_rows(rows, bias=bias_before, temperature=temp)
    grad_bias = before["grad_bias"]
    bias_after = [bias_value - lr_value * grad_value for bias_value, grad_value in zip(bias_before, grad_bias)]
    after = _score_rows(rows, bias=bias_after, temperature=temp)
    finite = bool(before["finite"]) and bool(after["finite"]) and all(math.isfinite(value) for value in bias_after)
    if not finite:
        raise ValueError("KL train step produced non-finite values")
    report = {
        "schema_version": STEP_SCHEMA_VERSION,
        "status": "logit_bias_train_step_ok",
        "student_training_started": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "kl_training_started": True,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": False,
        "optimizer": "analytic_sgd_logit_bias_kl",
        "k": k,
        "adjacency_name": adjacency_name,
        "feature_dim": feature_dim,
        "steps": steps,
        "projection_seed": projection_seed,
        "temperature": temp,
        "device": device,
        "lr": lr_value,
        "vocab_size": vocab_size,
        "row_count": len(rows),
        "bias_before_checksum": checksum_json(bias_before),
        "bias_after_checksum": checksum_json(bias_after),
        "grad_bias_checksum": checksum_json(grad_bias),
        "kl_before": before["kl"],
        "kl_after": after["kl"],
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "kl_decreased": after["kl"] <= before["kl"],
        "finite": finite,
        "note": "one-step logit-bias KL optimizer dry run only; frozen targets only",
    }
    return report, bias_after


def run_logit_bias_train_step(
    eval_output_dir: str | Path,
    logit_targets_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    projection_seed: int = 0,
    lr: float = 0.1,
    bias: list[float] | None = None,
    temperature: float | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    report, _bias_after = _run_logit_bias_train_step_internal(
        eval_output_dir,
        logit_targets_dir,
        k=k,
        adjacency_name=adjacency_name,
        feature_dim=feature_dim,
        steps=steps,
        projection_seed=projection_seed,
        lr=lr,
        bias=bias,
        temperature=temperature,
        device=device,
    )
    return report


def run_logit_bias_training_loop(
    eval_output_dir: str | Path,
    logit_targets_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    train_steps: int = 5,
    projection_seed: int = 0,
    lr: float = 0.1,
    temperature: float | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if train_steps < 1:
        raise ValueError(f"train_steps must be >= 1, got {train_steps}")
    bias: list[float] | None = None
    history: list[dict[str, Any]] = []
    for train_step in range(train_steps):
        step, bias_after = _run_logit_bias_train_step_internal(
            eval_output_dir,
            logit_targets_dir,
            k=k,
            adjacency_name=adjacency_name,
            feature_dim=feature_dim,
            steps=steps,
            projection_seed=projection_seed,
            lr=lr,
            bias=bias,
            temperature=temperature,
            device=device,
        )
        history.append(
            {
                "train_step": train_step,
                "kl_before": step["kl_before"],
                "kl_after": step["kl_after"],
                "cross_entropy_before": step["cross_entropy_before"],
                "cross_entropy_after": step["cross_entropy_after"],
                "bias_before_checksum": step["bias_before_checksum"],
                "bias_after_checksum": step["bias_after_checksum"],
                "grad_bias_checksum": step["grad_bias_checksum"],
                "kl_decreased": step["kl_decreased"],
            }
        )
        bias = list(bias_after)

    first = history[0]
    last = history[-1]
    monotonic_nonincreasing = all(
        history[idx]["kl_after"] <= history[idx]["kl_before"]
        and (idx == 0 or history[idx]["kl_before"] <= history[idx - 1]["kl_after"])
        for idx in range(len(history))
    )
    finite = all(
        math.isfinite(float(row[key]))
        for row in history
        for key in ("kl_before", "kl_after", "cross_entropy_before", "cross_entropy_after")
    )
    if not finite:
        raise ValueError("KL training loop produced non-finite values")
    return {
        "schema_version": "qwen_kl_distillation_trainable_loop.v1",
        "status": "logit_bias_training_loop_ok",
        "student_training_started": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "kl_training_started": True,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": False,
        "optimizer": "analytic_sgd_logit_bias_kl",
        "k": k,
        "adjacency_name": adjacency_name,
        "feature_dim": feature_dim,
        "steps": steps,
        "train_steps": train_steps,
        "projection_seed": projection_seed,
        "temperature": step["temperature"],
        "device": device,
        "lr": step["lr"],
        "vocab_size": step["vocab_size"],
        "row_count": step["row_count"],
        "bias_initial_checksum": history[0]["bias_before_checksum"],
        "bias_final_checksum": history[-1]["bias_after_checksum"],
        "kl_initial": first["kl_before"],
        "kl_final": last["kl_after"],
        "cross_entropy_initial": first["cross_entropy_before"],
        "cross_entropy_final": last["cross_entropy_after"],
        "kl_decreased": last["kl_after"] <= first["kl_before"],
        "monotonic_nonincreasing": monotonic_nonincreasing,
        "finite": finite,
        "history": history,
        "note": "bounded logit-bias KL training loop only; frozen targets only, no online teacher",
    }


def write_logit_bias_training_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_and_write_logit_bias_training_report(
    eval_output_dir: str | Path,
    logit_targets_dir: str | Path,
    output_path: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    train_steps: int = 5,
    projection_seed: int = 0,
    lr: float = 0.1,
    temperature: float | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    report = run_logit_bias_training_loop(
        eval_output_dir,
        logit_targets_dir,
        k=k,
        adjacency_name=adjacency_name,
        feature_dim=feature_dim,
        steps=steps,
        train_steps=train_steps,
        projection_seed=projection_seed,
        lr=lr,
        temperature=temperature,
        device=device,
    )
    write_logit_bias_training_report(report, output_path)
    return report
