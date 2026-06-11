from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.qwen_distillation_harness import load_distillation_harness_report, validate_distillation_harness_report


SCHEMA_VERSION = "qwen_distillation_gate.v1"
DEFAULT_GATE_REPORT_FILENAME = "distillation_gate_report.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _artifact_path(base: Path, raw_path: Any) -> Path:
    text = str(raw_path)
    path = Path(text)
    return path if path.is_absolute() else base / path


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_bool(report: dict[str, Any], key: str, expected: bool) -> None:
    if bool(report.get(key)) is not expected:
        raise ValueError(f"distillation gate {key} must be {expected}")


def _collect_checked_artifacts(harness_output_dir: Path, harness_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = harness_report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("harness artifacts must be an object")
    checked: dict[str, dict[str, Any]] = {}
    for name, raw_path in sorted(artifacts.items()):
        path = _artifact_path(harness_output_dir, raw_path)
        exists = path.exists()
        checked[name] = {
            "path": str(raw_path),
            "exists": exists,
            "byte_size": path.stat().st_size if exists and path.is_file() else None,
        }
        if not exists:
            raise FileNotFoundError(f"required harness artifact missing: {name} -> {path}")
    return checked


def build_distillation_gate_report(harness_output_dir: str | Path) -> dict[str, Any]:
    """Build a conservative v25 P11 gate report from a completed P10 harness output.

    The only available quality signal is the frozen-logit KL reduction proxy.
    Runtime and memory remain unavailable until real benchmark/memory protocols exist,
    so promotion is always false in this P11 report.
    """
    base = Path(harness_output_dir)
    harness_report_path = base / "distillation_harness_report.json"
    harness_report = load_distillation_harness_report(harness_report_path)
    validate_distillation_harness_report(harness_report)
    checked_artifacts = _collect_checked_artifacts(base, harness_report)

    kl_before = _finite_float(harness_report["kl_before"], name="kl_before")
    kl_after = _finite_float(harness_report["kl_after"], name="kl_after")
    kl_delta = kl_before - kl_after
    kl_relative_reduction = kl_delta / kl_before if kl_before > 0.0 else 0.0
    quality_ok = bool(harness_report.get("finite")) and bool(harness_report.get("kl_decreased")) and kl_after <= kl_before
    safety_ok = (
        bool(harness_report.get("student_training_started"))
        and not bool(harness_report.get("teacher_checkpoint_loaded", True))
        and not bool(harness_report.get("teacher_inference_runtime_required", True))
        and not bool(harness_report.get("teacher_distillation_started", True))
        and bool(harness_report.get("kl_training_started"))
        and not bool(harness_report.get("raw_weight_payload_in_graph", True))
        and bool(harness_report.get("bounded_active_adjacency"))
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "distillation_gate_report_ok",
        "student_training_started": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "kl_training_started": True,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": False,
        "promotion_decision": "not_promoted",
        "harness_output_dir": str(base),
        "harness_report": "distillation_harness_report.json",
        "quality_gate": {
            "gate_name": "kl_loss_reduction_proxy",
            "available": True,
            "proxy_only": True,
            "task_quality_available": False,
            "quality_ok": quality_ok,
            "metric": "kl_after <= kl_before",
            "kl_before": kl_before,
            "kl_after": kl_after,
            "kl_delta": kl_delta,
            "kl_relative_reduction": kl_relative_reduction,
            "reason": "frozen-logit KL reduction proxy only; real task quality unavailable",
        },
        "runtime_gate": {
            "gate_name": "runtime_benchmark",
            "available": False,
            "speed_ok": False,
            "reason": "real runtime benchmark unavailable for v25 P11",
        },
        "memory_gate": {
            "gate_name": "memory_benchmark",
            "available": False,
            "memory_ok": False,
            "reason": "real memory measurement unavailable for v25 P11",
        },
        "safety_gate": {
            "gate_name": "runtime_safety_invariants",
            "available": True,
            "safety_ok": safety_ok,
            "teacher_checkpoint_loaded": False,
            "teacher_inference_runtime_required": False,
            "raw_weight_payload_in_graph": False,
            "bounded_active_adjacency": True,
        },
        "promotion_report": {
            "promote": False,
            "quality_ok": quality_ok,
            "runtime_ok": False,
            "memory_ok": False,
            "safety_ok": safety_ok,
            "old_champion_scorer_behavior_unchanged": True,
            "reason": "runtime_memory_task_quality_pending",
        },
        "checked_artifacts": checked_artifacts,
        "finite": True,
        "note": "v25 P11 gate report only; KL quality is proxy-only and promotion remains blocked",
    }


def write_distillation_gate_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_distillation_gate_report(output_path: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_path))


def validate_distillation_gate_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad gate schema_version={report.get('schema_version')!r}")
    if report.get("status") != "distillation_gate_report_ok":
        raise ValueError(f"bad gate status={report.get('status')!r}")
    for key, expected in {
        "student_training_started": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "kl_training_started": True,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": False,
        "finite": True,
    }.items():
        _require_bool(report, key, expected)
    if report.get("promotion_decision") != "not_promoted":
        raise ValueError("distillation gate promotion_decision must be not_promoted")

    quality_gate = report.get("quality_gate")
    runtime_gate = report.get("runtime_gate")
    memory_gate = report.get("memory_gate")
    safety_gate = report.get("safety_gate")
    promotion_report = report.get("promotion_report")
    if not all(isinstance(gate, dict) for gate in (quality_gate, runtime_gate, memory_gate, safety_gate, promotion_report)):
        raise ValueError("distillation gate subreports must be objects")

    if not bool(quality_gate.get("available")):
        raise ValueError("quality gate must be available for KL proxy")
    if not bool(quality_gate.get("proxy_only")):
        raise ValueError("quality gate must be marked proxy_only")
    if bool(quality_gate.get("task_quality_available", True)):
        raise ValueError("task quality must remain unavailable in v25 P11")
    kl_before = _finite_float(quality_gate.get("kl_before"), name="quality_gate.kl_before")
    kl_after = _finite_float(quality_gate.get("kl_after"), name="quality_gate.kl_after")
    if kl_before < 0.0 or kl_after < 0.0:
        raise ValueError("KL metrics must be non-negative")
    if bool(quality_gate.get("quality_ok")) and kl_after > kl_before:
        raise ValueError("quality_ok cannot be true when KL increased")

    if bool(runtime_gate.get("available")) or bool(runtime_gate.get("speed_ok")):
        raise ValueError("runtime gate must remain unavailable/false in v25 P11")
    if bool(memory_gate.get("available")) or bool(memory_gate.get("memory_ok")):
        raise ValueError("memory gate must remain unavailable/false in v25 P11")
    if not bool(safety_gate.get("safety_ok")):
        raise ValueError("safety gate must pass")
    if bool(promotion_report.get("promote")):
        raise ValueError("v25 P11 must not promote")
    if bool(promotion_report.get("runtime_ok")) or bool(promotion_report.get("memory_ok")):
        raise ValueError("promotion runtime/memory gates must remain false")

    checked = report.get("checked_artifacts")
    if not isinstance(checked, dict) or not checked:
        raise ValueError("checked_artifacts must be a non-empty object")
    for name, row in checked.items():
        if not isinstance(row, dict):
            raise ValueError(f"checked artifact {name!r} must be an object")
        if not bool(row.get("exists")):
            raise ValueError(f"checked artifact {name!r} must exist")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "distillation_gate_report_valid",
        "quality_ok": bool(quality_gate.get("quality_ok")),
        "runtime_ok": False,
        "memory_ok": False,
        "promote": False,
        "promotion_decision": "not_promoted",
    }


def run_and_write_distillation_gate_report(
    harness_output_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_distillation_gate_report(harness_output_dir)
    validate_distillation_gate_report(report)
    path = Path(output_path) if output_path is not None else Path(harness_output_dir) / DEFAULT_GATE_REPORT_FILENAME
    write_distillation_gate_report(report, path)
    return report
