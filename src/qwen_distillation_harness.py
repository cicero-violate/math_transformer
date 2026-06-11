from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.qwen_kl_distillation_eval import run_and_write_kl_eval_report
from src.qwen_kl_distillation_trainable import run_and_write_logit_bias_training_report
from src.qwen_logit_distillation_targets import (
    load_frozen_logit_distillation_targets_manifest,
    validate_frozen_logit_distillation_targets,
    write_frozen_logit_distillation_targets,
)
from src.qwen_sparse_student_handoff import build_fixed_topology_student_stub


SCHEMA_VERSION = "qwen_fixed_topology_distillation_harness.v1"
HARNESS_REPORT_FILENAME = "distillation_harness_report.json"


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_args(
    *,
    vocab_size: int,
    feature_dim: int,
    forward_steps: int,
    train_steps: int,
    lr: float,
    temperature: float,
) -> tuple[int, int, int, int, float, float]:
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    if feature_dim < 1:
        raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
    if forward_steps < 1:
        raise ValueError(f"forward_steps must be >= 1, got {forward_steps}")
    if train_steps < 1:
        raise ValueError(f"train_steps must be >= 1, got {train_steps}")
    lr_value = _finite_float(lr, name="lr")
    if lr_value <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr!r}")
    temp = _finite_float(temperature, name="temperature")
    if temp <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    return vocab_size, feature_dim, forward_steps, train_steps, lr_value, temp


def _relative_or_str(path: Path, *, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def write_distillation_harness_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_distillation_harness_report(output_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(output_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("distillation harness report must be a JSON object")
    return data


def validate_distillation_harness_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad harness schema_version={report.get('schema_version')!r}")
    if report.get("status") != "fixed_topology_distillation_harness_ok":
        raise ValueError(f"bad harness status={report.get('status')!r}")
    expected_flags = {
        "student_training_started": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "kl_training_started": True,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": False,
        "finite": True,
    }
    for key, expected in expected_flags.items():
        if bool(report.get(key)) is not expected:
            raise ValueError(f"harness {key} must be {expected}")
    if report.get("promotion_decision") != "not_promoted":
        raise ValueError("harness promotion_decision must be not_promoted")
    if float(report.get("kl_before", float("nan"))) < 0.0:
        raise ValueError("harness kl_before must be non-negative")
    if float(report.get("kl_after", float("nan"))) < 0.0:
        raise ValueError("harness kl_after must be non-negative")
    if not bool(report.get("kl_decreased", False)):
        raise ValueError("harness kl_decreased must be true")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("harness artifacts must be an object")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fixed_topology_distillation_harness_valid",
        "kl_before": report["kl_before"],
        "kl_after": report["kl_after"],
        "kl_decreased": report["kl_decreased"],
        "promotion_decision": "not_promoted",
    }


def run_fixed_topology_distillation_harness(
    eval_output_dir: str | Path,
    output_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    logit_targets_dir: str | Path | None = None,
    vocab_size: int = 16,
    target_seeds: list[int] | None = None,
    feature_dim: int = 8,
    forward_steps: int = 1,
    train_steps: int = 5,
    lr: float = 0.1,
    projection_seed: int = 0,
    temperature: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    vocab_size, feature_dim, forward_steps, train_steps, lr, temperature = _validate_args(
        vocab_size=vocab_size,
        feature_dim=feature_dim,
        forward_steps=forward_steps,
        train_steps=train_steps,
        lr=lr,
        temperature=temperature,
    )

    stub = build_fixed_topology_student_stub(eval_output_dir, k=k, adjacency_name=adjacency_name)
    if stub.get("status") != "fixed_topology_stub_ok":
        raise ValueError(f"v25 handoff validation failed: {stub.get('status')}")

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    used_existing_targets = logit_targets_dir is not None
    if logit_targets_dir is None:
        targets_dir = out_base / "frozen_logit_targets"
        write_frozen_logit_distillation_targets(
            targets_dir,
            vocab_size=vocab_size,
            seeds=target_seeds,
            temperature=temperature,
        )
    else:
        targets_dir = Path(logit_targets_dir)
    target_summary = validate_frozen_logit_distillation_targets(targets_dir)
    target_manifest = load_frozen_logit_distillation_targets_manifest(targets_dir)

    kl_eval_before_path = out_base / "kl_eval_before.json"
    kl_before_report = run_and_write_kl_eval_report(
        eval_output_dir,
        targets_dir,
        kl_eval_before_path,
        k=stub["k"],
        feature_dim=feature_dim,
        steps=forward_steps,
        projection_seed=projection_seed,
        temperature=temperature,
        device=device,
    )

    training_report_path = out_base / "kl_training_report.json"
    training_report = run_and_write_logit_bias_training_report(
        eval_output_dir,
        targets_dir,
        training_report_path,
        k=stub["k"],
        feature_dim=feature_dim,
        steps=forward_steps,
        train_steps=train_steps,
        projection_seed=projection_seed,
        lr=lr,
        temperature=temperature,
    )

    manifest_path = targets_dir / "frozen_logit_targets_manifest.json"
    rows_path = targets_dir / "frozen_logit_targets.jsonl"
    artifacts = {
        "frozen_logit_targets_manifest": _relative_or_str(manifest_path, base=out_base),
        "frozen_logit_targets_rows": _relative_or_str(rows_path, base=out_base),
        "kl_eval_before": "kl_eval_before.json",
        "kl_training_report": "kl_training_report.json",
        "distillation_harness_report": HARNESS_REPORT_FILENAME,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "fixed_topology_distillation_harness_ok",
        "student_training_started": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "kl_training_started": True,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": False,
        "promotion_decision": "not_promoted",
        "eval_output_dir": str(Path(eval_output_dir)),
        "output_dir": str(out_base),
        "logit_targets_dir": str(targets_dir),
        "used_existing_logit_targets": used_existing_targets,
        "adjacency_name": stub["adjacency_name"],
        "k": stub["k"],
        "vocab_size": int(target_manifest["vocab_size"]),
        "feature_dim": feature_dim,
        "forward_steps": forward_steps,
        "train_steps": train_steps,
        "lr": lr,
        "projection_seed": projection_seed,
        "temperature": float(target_manifest["temperature"]),
        "device": device,
        "target_row_count": target_summary["row_count"],
        "kl_before": kl_before_report["kl_mean"],
        "kl_after": training_report["kl_final"],
        "kl_decreased": bool(training_report["kl_decreased"]),
        "monotonic_nonincreasing": bool(training_report["monotonic_nonincreasing"]),
        "finite": bool(kl_before_report["finite"]) and bool(training_report["finite"]),
        "artifacts": artifacts,
        "note": "fixed-topology frozen-logit distillation harness only; no online teacher inference or promotion",
    }
    validate_distillation_harness_report(report)
    write_distillation_harness_report(report, out_base / HARNESS_REPORT_FILENAME)
    return report
