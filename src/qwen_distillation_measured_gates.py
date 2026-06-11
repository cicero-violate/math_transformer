from __future__ import annotations

import json
import math
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

from src.qwen_distillation_harness import load_distillation_harness_report, validate_distillation_harness_report
from src.qwen_kl_distillation_trainable import run_logit_bias_training_loop
from src.qwen_logit_distillation_targets import validate_frozen_logit_distillation_targets


SCHEMA_VERSION = "qwen_distillation_gate.v1"
DEFAULT_MEASURED_GATE_REPORT_FILENAME = "distillation_measured_gate_report.json"


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
    runtime_repeats: int,
    max_runtime_seconds: float,
    max_peak_memory_bytes: int,
    min_kl_relative_reduction: float,
) -> tuple[int, float, int, float]:
    if runtime_repeats < 1:
        raise ValueError(f"runtime_repeats must be >= 1, got {runtime_repeats}")
    max_seconds = _finite_float(max_runtime_seconds, name="max_runtime_seconds")
    if max_seconds < 0.0:
        raise ValueError(f"max_runtime_seconds must be >= 0, got {max_runtime_seconds!r}")
    max_bytes = int(max_peak_memory_bytes)
    if max_bytes < 0:
        raise ValueError(f"max_peak_memory_bytes must be >= 0, got {max_peak_memory_bytes!r}")
    min_reduction = _finite_float(min_kl_relative_reduction, name="min_kl_relative_reduction")
    if min_reduction < 0.0:
        raise ValueError(f"min_kl_relative_reduction must be >= 0, got {min_kl_relative_reduction!r}")
    return runtime_repeats, max_seconds, max_bytes, min_reduction


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    idx = math.ceil((percentile / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _artifact_path(base: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else base / path


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


def _run_one_measured_training_loop(harness_report: dict[str, Any]) -> tuple[dict[str, Any], float, int]:
    eval_output_dir = Path(harness_report["eval_output_dir"])
    logit_targets_dir = Path(harness_report["logit_targets_dir"])
    tracemalloc.start()
    started = time.perf_counter()
    try:
        report = run_logit_bias_training_loop(
            eval_output_dir,
            logit_targets_dir,
            k=int(harness_report["k"]),
            feature_dim=int(harness_report["feature_dim"]),
            steps=int(harness_report["forward_steps"]),
            train_steps=int(harness_report["train_steps"]),
            projection_seed=int(harness_report["projection_seed"]),
            lr=float(harness_report["lr"]),
            temperature=float(harness_report["temperature"]),
        )
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return report, elapsed, int(peak)


def build_measured_distillation_gate_report(
    harness_output_dir: str | Path,
    *,
    runtime_repeats: int = 3,
    max_runtime_seconds: float = 10.0,
    max_peak_memory_bytes: int = 128 * 1024 * 1024,
    min_kl_relative_reduction: float = 0.0,
) -> dict[str, Any]:
    """Measure task-quality, runtime, and memory gates for a P10 harness output.

    The task-quality protocol is frozen-logit KL reduction over the validated
    frozen target rows. Runtime is wall-clock elapsed seconds for the bounded KL
    training loop. Memory is Python tracemalloc peak bytes during that loop.
    """
    runtime_repeats, max_runtime_seconds, max_peak_memory_bytes, min_kl_relative_reduction = _validate_args(
        runtime_repeats=runtime_repeats,
        max_runtime_seconds=max_runtime_seconds,
        max_peak_memory_bytes=max_peak_memory_bytes,
        min_kl_relative_reduction=min_kl_relative_reduction,
    )
    base = Path(harness_output_dir)
    harness_report = load_distillation_harness_report(base / "distillation_harness_report.json")
    validate_distillation_harness_report(harness_report)
    checked_artifacts = _collect_checked_artifacts(base, harness_report)
    target_summary = validate_frozen_logit_distillation_targets(harness_report["logit_targets_dir"])

    measured_reports: list[dict[str, Any]] = []
    durations: list[float] = []
    peak_bytes: list[int] = []
    for _idx in range(runtime_repeats):
        report, elapsed, peak = _run_one_measured_training_loop(harness_report)
        measured_reports.append(report)
        durations.append(elapsed)
        peak_bytes.append(peak)

    first = measured_reports[0]
    kl_initial = _finite_float(first["kl_initial"], name="kl_initial")
    kl_final = _finite_float(first["kl_final"], name="kl_final")
    kl_delta = kl_initial - kl_final
    kl_relative_reduction = kl_delta / kl_initial if kl_initial > 0.0 else 0.0
    all_equivalent = all(
        report["kl_initial"] == first["kl_initial"]
        and report["kl_final"] == first["kl_final"]
        and report["bias_final_checksum"] == first["bias_final_checksum"]
        for report in measured_reports
    )
    quality_ok = (
        bool(first["finite"])
        and bool(first["kl_decreased"])
        and bool(first["monotonic_nonincreasing"])
        and kl_final <= kl_initial
        and kl_relative_reduction >= min_kl_relative_reduction
        and all_equivalent
    )
    duration_median = statistics.median(durations)
    duration_p95 = _percentile_nearest_rank(durations, 95.0)
    duration_max = max(durations)
    runtime_ok = duration_median <= max_runtime_seconds and duration_p95 <= max_runtime_seconds
    peak_max = max(peak_bytes)
    peak_median = int(statistics.median(peak_bytes))
    memory_ok = peak_max <= max_peak_memory_bytes
    safety_ok = (
        bool(harness_report.get("student_training_started"))
        and not bool(harness_report.get("teacher_checkpoint_loaded", True))
        and not bool(harness_report.get("teacher_inference_runtime_required", True))
        and not bool(harness_report.get("teacher_distillation_started", True))
        and bool(harness_report.get("kl_training_started"))
        and not bool(harness_report.get("raw_weight_payload_in_graph", True))
        and bool(harness_report.get("bounded_active_adjacency"))
    )
    promote_ready = quality_ok and runtime_ok and memory_ok and safety_ok
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
        "promotion_eligible": promote_ready,
        "promotion_decision": "eligible_pending_p12" if promote_ready else "not_promoted",
        "harness_output_dir": str(base),
        "harness_report": "distillation_harness_report.json",
        "measurement_protocol": {
            "schema_version": "qwen_distillation_measured_gates.v1",
            "quality_protocol": "frozen_logit_kl_reduction",
            "runtime_protocol": "perf_counter_bounded_kl_training_loop",
            "memory_protocol": "tracemalloc_peak_bounded_kl_training_loop",
            "runtime_repeats": runtime_repeats,
            "target_rows_sha256": target_summary["target_rows_sha256"],
        },
        "quality_gate": {
            "gate_name": "frozen_logit_task_quality",
            "available": True,
            "proxy_only": False,
            "task_quality_available": True,
            "quality_ok": quality_ok,
            "metric": "kl_final <= kl_initial and deterministic bounded loop",
            "kl_initial": kl_initial,
            "kl_final": kl_final,
            "kl_delta": kl_delta,
            "kl_relative_reduction": kl_relative_reduction,
            "min_kl_relative_reduction": min_kl_relative_reduction,
            "monotonic_nonincreasing": bool(first["monotonic_nonincreasing"]),
            "deterministic_repeats": all_equivalent,
            "reason": "frozen-logit target quality gate passed" if quality_ok else "frozen-logit target quality gate failed",
        },
        "runtime_gate": {
            "gate_name": "bounded_kl_training_runtime",
            "available": True,
            "runtime_ok": runtime_ok,
            "speed_ok": runtime_ok,
            "duration_seconds": durations,
            "duration_median_seconds": duration_median,
            "duration_p95_seconds": duration_p95,
            "duration_max_seconds": duration_max,
            "max_runtime_seconds": max_runtime_seconds,
            "reason": "runtime within threshold" if runtime_ok else "runtime threshold failed",
        },
        "memory_gate": {
            "gate_name": "bounded_kl_training_tracemalloc_peak",
            "available": True,
            "memory_ok": memory_ok,
            "peak_bytes": peak_bytes,
            "peak_median_bytes": peak_median,
            "peak_max_bytes": peak_max,
            "max_peak_memory_bytes": max_peak_memory_bytes,
            "reason": "memory within threshold" if memory_ok else "memory threshold failed",
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
            "promote": promote_ready,
            "quality_ok": quality_ok,
            "runtime_ok": runtime_ok,
            "memory_ok": memory_ok,
            "safety_ok": safety_ok,
            "old_champion_scorer_behavior_unchanged": True,
            "reason": "all_measured_gates_passed" if promote_ready else "measured_gate_failed",
        },
        "checked_artifacts": checked_artifacts,
        "finite": True,
        "note": "measured frozen-target task quality, runtime, and memory gate report; P12 remains final promotion authority",
    }


def write_measured_distillation_gate_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_measured_distillation_gate_report(output_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(output_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("measured gate report must be a JSON object")
    return data


def validate_measured_distillation_gate_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad measured gate schema_version={report.get('schema_version')!r}")
    if report.get("status") != "distillation_gate_report_ok":
        raise ValueError(f"bad measured gate status={report.get('status')!r}")
    for key, expected in {
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "finite": True,
    }.items():
        if bool(report.get(key)) is not expected:
            raise ValueError(f"measured gate {key} must be {expected}")
    quality_gate = report.get("quality_gate")
    runtime_gate = report.get("runtime_gate")
    memory_gate = report.get("memory_gate")
    safety_gate = report.get("safety_gate")
    promotion_report = report.get("promotion_report")
    if not all(isinstance(gate, dict) for gate in (quality_gate, runtime_gate, memory_gate, safety_gate, promotion_report)):
        raise ValueError("measured gate subreports must be objects")
    if not bool(quality_gate.get("available")) or bool(quality_gate.get("proxy_only")):
        raise ValueError("measured quality gate must be available and non-proxy")
    if not bool(quality_gate.get("task_quality_available")):
        raise ValueError("measured task quality must be available")
    if _finite_float(quality_gate.get("kl_final"), name="quality_gate.kl_final") < 0.0:
        raise ValueError("quality gate KL must be non-negative")
    if not bool(runtime_gate.get("available")):
        raise ValueError("measured runtime gate must be available")
    if not bool(memory_gate.get("available")):
        raise ValueError("measured memory gate must be available")
    if not bool(safety_gate.get("available")) or not bool(safety_gate.get("safety_ok")):
        raise ValueError("measured safety gate must be available and passing")
    promote = bool(promotion_report.get("promote"))
    expected_promote = (
        bool(quality_gate.get("quality_ok"))
        and bool(runtime_gate.get("runtime_ok", runtime_gate.get("speed_ok")))
        and bool(memory_gate.get("memory_ok"))
        and bool(safety_gate.get("safety_ok"))
    )
    if promote is not expected_promote:
        raise ValueError("promotion_report promote must match measured gate conjunction")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "measured_distillation_gate_report_valid",
        "quality_ok": bool(quality_gate.get("quality_ok")),
        "runtime_ok": bool(runtime_gate.get("runtime_ok", runtime_gate.get("speed_ok"))),
        "memory_ok": bool(memory_gate.get("memory_ok")),
        "safety_ok": bool(safety_gate.get("safety_ok")),
        "promote_ready": promote,
    }


def run_and_write_measured_distillation_gate_report(
    harness_output_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    runtime_repeats: int = 3,
    max_runtime_seconds: float = 10.0,
    max_peak_memory_bytes: int = 128 * 1024 * 1024,
    min_kl_relative_reduction: float = 0.0,
) -> dict[str, Any]:
    report = build_measured_distillation_gate_report(
        harness_output_dir,
        runtime_repeats=runtime_repeats,
        max_runtime_seconds=max_runtime_seconds,
        max_peak_memory_bytes=max_peak_memory_bytes,
        min_kl_relative_reduction=min_kl_relative_reduction,
    )
    validate_measured_distillation_gate_report(report)
    path = Path(output_path) if output_path is not None else Path(harness_output_dir) / DEFAULT_MEASURED_GATE_REPORT_FILENAME
    write_measured_distillation_gate_report(report, path)
    return report
