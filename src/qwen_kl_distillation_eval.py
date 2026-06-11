from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.qwen_distillation_targets import checksum_json
from src.qwen_logit_distillation_targets import (
    iter_frozen_logit_target_rows,
    load_frozen_logit_distillation_targets_manifest,
    softmax,
    validate_frozen_logit_distillation_targets,
)
from src.qwen_sparse_student_runtime import run_fixed_topology_forward_features


SCHEMA_VERSION = "qwen_kl_distillation_eval.v1"
PROBABILITY_SUM_TOLERANCE = 1e-8


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_eval_args(*, feature_dim: int, steps: int, vocab_size: int, temperature: float) -> tuple[int, int, int, float]:
    if feature_dim < 1:
        raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    temp = _finite_float(temperature, name="temperature")
    if temp <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    return feature_dim, steps, vocab_size, temp


def _require_false(data: dict[str, Any], key: str) -> None:
    if bool(data.get(key, True)):
        raise ValueError(f"frozen logit manifest {key} must be false")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _flatten_features(output_features: dict[str, list[float]]) -> list[float]:
    flat: list[float] = []
    for node_id in sorted(output_features):
        vector = output_features[node_id]
        for value in vector:
            flat.append(_finite_float(value, name="output_feature"))
    if not flat:
        raise ValueError("output_features must contain at least one feature value")
    return flat


def _deterministic_weight(feature_index: int, vocab_index: int, seed: int) -> float:
    phase = (feature_index + 1) * 12.9898 + (vocab_index + 1) * 78.233 + (seed + 1) * 37.719
    return math.sin(phase) + 0.5 * math.cos(phase * 0.61803398875)


def project_features_to_logits(
    output_features: dict[str, list[float]],
    *,
    vocab_size: int,
    seed: int = 0,
) -> list[float]:
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    flat = _flatten_features(output_features)
    norm = math.sqrt(len(flat))
    logits: list[float] = []
    for vocab_index in range(vocab_size):
        total = 0.0
        for feature_index, value in enumerate(flat):
            total += value * _deterministic_weight(feature_index, vocab_index, seed)
        logit = total / norm
        if not math.isfinite(logit):
            raise ValueError("projected student logits must be finite")
        logits.append(logit)
    return logits


def _validate_probability_vector(probabilities: list[float], *, name: str) -> list[float]:
    if not probabilities:
        raise ValueError(f"{name} must be non-empty")
    clean = [_finite_float(value, name=name) for value in probabilities]
    if any(value < 0.0 for value in clean):
        raise ValueError(f"{name} must be non-negative")
    total = sum(clean)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
        raise ValueError(f"{name} must sum to 1.0, got {total}")
    return clean


def kl_divergence(
    teacher_probabilities: list[float],
    student_probabilities: list[float],
    *,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    eps = _finite_float(epsilon, name="epsilon")
    if eps <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon!r}")
    teacher = _validate_probability_vector(teacher_probabilities, name="teacher_probabilities")
    student = _validate_probability_vector(student_probabilities, name="student_probabilities")
    if len(teacher) != len(student):
        raise ValueError("teacher and student probability lengths must match")

    kl = 0.0
    cross_entropy = 0.0
    entropy_teacher = 0.0
    for teacher_prob, student_prob in zip(teacher, student):
        kl += teacher_prob * math.log((teacher_prob + eps) / (student_prob + eps))
        cross_entropy -= teacher_prob * math.log(student_prob + eps)
        entropy_teacher -= teacher_prob * math.log(teacher_prob + eps)
    finite = all(math.isfinite(value) for value in (kl, cross_entropy, entropy_teacher))
    if not finite:
        raise ValueError("KL metrics must be finite")
    return {
        "kl": kl,
        "cross_entropy": cross_entropy,
        "entropy_teacher": entropy_teacher,
        "finite": finite,
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("target_type") != "logits":
        raise ValueError(f"unsupported target_type={manifest.get('target_type')!r}")
    for key in (
        "teacher_checkpoint_loaded_at_runtime",
        "teacher_inference_runtime_required",
        "raw_weight_payload_in_graph",
        "student_training_started",
        "kl_training_started",
        "promotion_eligible",
    ):
        _require_false(manifest, key)


def evaluate_kl_against_frozen_logits(
    eval_output_dir: str | Path,
    logit_targets_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    projection_seed: int = 0,
    temperature: float | None = None,
) -> dict[str, Any]:
    target_summary = validate_frozen_logit_distillation_targets(logit_targets_dir)
    manifest = load_frozen_logit_distillation_targets_manifest(logit_targets_dir)
    _validate_manifest(manifest)
    temp = float(manifest["temperature"]) if temperature is None else temperature
    feature_dim, steps, vocab_size, temp = _validate_eval_args(
        feature_dim=feature_dim,
        steps=steps,
        vocab_size=int(manifest["vocab_size"]),
        temperature=temp,
    )

    rows: list[dict[str, Any]] = []
    kl_values: list[float] = []
    cross_entropy_values: list[float] = []
    entropy_values: list[float] = []
    for row in iter_frozen_logit_target_rows(logit_targets_dir):
        teacher_probabilities = list(row["probabilities"])
        if len(teacher_probabilities) != vocab_size:
            raise ValueError("teacher probabilities length does not match vocab_size")
        forward = run_fixed_topology_forward_features(
            eval_output_dir,
            k=k,
            adjacency_name=adjacency_name,
            feature_dim=feature_dim,
            steps=steps,
            seed=int(row["seed"]),
        )
        student_logits = project_features_to_logits(
            forward["output_features"],
            vocab_size=vocab_size,
            seed=projection_seed,
        )
        student_probabilities = softmax(student_logits, temp)
        metrics = kl_divergence(teacher_probabilities, student_probabilities)
        row_finite = bool(forward["summary"]["finite"]) and bool(metrics["finite"])
        if not row_finite:
            raise ValueError("KL eval produced non-finite row")
        rows.append(
            {
                "row_id": row["row_id"],
                "seed": row["seed"],
                "kl": metrics["kl"],
                "cross_entropy": metrics["cross_entropy"],
                "entropy_teacher": metrics["entropy_teacher"],
                "student_logits_checksum": checksum_json(student_logits),
                "student_probabilities_checksum": checksum_json(student_probabilities),
                "teacher_probabilities_checksum": checksum_json(teacher_probabilities),
                "output_checksum": forward["summary"]["output_checksum"],
                "finite": row_finite,
            }
        )
        kl_values.append(metrics["kl"])
        cross_entropy_values.append(metrics["cross_entropy"])
        entropy_values.append(metrics["entropy_teacher"])

    if not rows:
        raise ValueError("KL eval requires at least one frozen logit target row")
    finite = all(math.isfinite(value) for value in kl_values + cross_entropy_values + entropy_values)
    if not finite:
        raise ValueError("KL eval aggregate metrics must be finite")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "kl_distillation_eval_ok",
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "kl_training_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": False,
        "eval_output_dir": str(Path(eval_output_dir)),
        "logit_targets_dir": str(Path(logit_targets_dir)),
        "adjacency_name": adjacency_name,
        "k": k,
        "feature_dim": feature_dim,
        "steps": steps,
        "vocab_size": vocab_size,
        "temperature": temp,
        "row_count": len(rows),
        "kl_mean": _mean(kl_values),
        "kl_min": min(kl_values),
        "kl_max": max(kl_values),
        "cross_entropy_mean": _mean(cross_entropy_values),
        "entropy_teacher_mean": _mean(entropy_values),
        "finite": finite,
        "target_rows_sha256": target_summary["target_rows_sha256"],
        "rows": rows,
        "note": "KL loss evaluation against frozen logits only; no teacher inference or KL training",
    }


def write_kl_eval_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_and_write_kl_eval_report(
    eval_output_dir: str | Path,
    logit_targets_dir: str | Path,
    output_path: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    projection_seed: int = 0,
    temperature: float | None = None,
) -> dict[str, Any]:
    report = evaluate_kl_against_frozen_logits(
        eval_output_dir,
        logit_targets_dir,
        k=k,
        adjacency_name=adjacency_name,
        feature_dim=feature_dim,
        steps=steps,
        projection_seed=projection_seed,
        temperature=temperature,
    )
    write_kl_eval_report(report, output_path)
    return report
