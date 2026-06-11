from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.qwen_distillation_gate import load_distillation_gate_report


SCHEMA_VERSION = "qwen_distillation_promotion.v1"
DEFAULT_PROMOTION_DECISION_FILENAME = "distillation_promotion_decision.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _bool(data: dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(data.get(key, default))


def _gate_report_path(gate_report_or_path: dict[str, Any] | str | Path) -> str | None:
    return None if isinstance(gate_report_or_path, dict) else str(Path(gate_report_or_path))


def _load_gate_report(gate_report_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(gate_report_or_path, dict):
        return dict(gate_report_or_path)
    return load_distillation_gate_report(gate_report_or_path)


def _validate_gate_report_for_promotion(gate_report: dict[str, Any]) -> None:
    """Validate the common P11/future-gate shape needed by P12.

    P11's own validator intentionally rejects runtime/memory availability and
    non-proxy quality. P12 must be able to evaluate both current conservative
    P11 reports and future reports where real gates are populated.
    """
    if gate_report.get("schema_version") != "qwen_distillation_gate.v1":
        raise ValueError(f"bad gate schema_version={gate_report.get('schema_version')!r}")
    if gate_report.get("status") != "distillation_gate_report_ok":
        raise ValueError(f"bad gate status={gate_report.get('status')!r}")
    for key, expected in {
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
    }.items():
        if bool(gate_report.get(key)) is not expected:
            raise ValueError(f"gate report {key} must be {expected}")
    for key in ("quality_gate", "runtime_gate", "memory_gate", "safety_gate"):
        if not isinstance(gate_report.get(key), dict):
            raise ValueError(f"gate report {key} must be an object")
    quality_gate = gate_report["quality_gate"]
    runtime_gate = gate_report["runtime_gate"]
    memory_gate = gate_report["memory_gate"]
    safety_gate = gate_report["safety_gate"]
    if not bool(quality_gate.get("available")):
        raise ValueError("quality gate must be available")
    if bool(quality_gate.get("quality_ok")) and not bool(quality_gate.get("task_quality_available")):
        # Current P11 has this exact proxy-only state; it is allowed as an input,
        # but P12 will not promote because task_quality_available is false.
        pass
    if bool(runtime_gate.get("speed_ok")) and not bool(runtime_gate.get("available")):
        raise ValueError("runtime_ok/speed_ok requires runtime gate availability")
    if bool(runtime_gate.get("runtime_ok")) and not bool(runtime_gate.get("available")):
        raise ValueError("runtime_ok requires runtime gate availability")
    if bool(memory_gate.get("memory_ok")) and not bool(memory_gate.get("available")):
        raise ValueError("memory_ok requires memory gate availability")
    if not bool(safety_gate.get("available")):
        raise ValueError("safety gate must be available")


def decide_distillation_promotion(gate_report_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Build the v25 P12 promotion decision from a P11 gate report.

    Promotion is allowed only when all real gates are explicitly true:
    real task quality, runtime, memory, and safety. KL-loss reduction remains a
    proxy-only quality signal and is therefore insufficient for promotion.
    """
    gate_report = _load_gate_report(gate_report_or_path)
    _validate_gate_report_for_promotion(gate_report)

    quality_gate = gate_report["quality_gate"]
    runtime_gate = gate_report["runtime_gate"]
    memory_gate = gate_report["memory_gate"]
    safety_gate = gate_report["safety_gate"]

    quality_available = _bool(quality_gate, "available")
    quality_ok = _bool(quality_gate, "quality_ok")
    task_quality_available = _bool(quality_gate, "task_quality_available")
    quality_proxy_only = _bool(quality_gate, "proxy_only")
    runtime_available = _bool(runtime_gate, "available")
    runtime_ok = _bool(runtime_gate, "speed_ok") or _bool(runtime_gate, "runtime_ok")
    memory_available = _bool(memory_gate, "available")
    memory_ok = _bool(memory_gate, "memory_ok")
    safety_available = _bool(safety_gate, "available")
    safety_ok = _bool(safety_gate, "safety_ok")

    required_gates = {
        "quality_gate_available": quality_available,
        "task_quality_available": task_quality_available,
        "quality_not_proxy_only": not quality_proxy_only,
        "quality_ok": quality_ok,
        "runtime_gate_available": runtime_available,
        "runtime_ok": runtime_ok,
        "memory_gate_available": memory_available,
        "memory_ok": memory_ok,
        "safety_gate_available": safety_available,
        "safety_ok": safety_ok,
    }
    missing_or_failed = [name for name, passed in required_gates.items() if not passed]
    promote = not missing_or_failed
    reason = "all_real_gates_passed" if promote else "required_real_gates_missing_or_failed"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "distillation_promotion_decision_ok",
        "promote": promote,
        "decision": "promoted" if promote else "not_promoted",
        "reason": reason,
        "source_gate_report": _gate_report_path(gate_report_or_path),
        "source_gate_schema_version": gate_report.get("schema_version"),
        "student_training_started": bool(gate_report.get("student_training_started", False)),
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "kl_training_started": bool(gate_report.get("kl_training_started", False)),
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_eligible": promote,
        "required_gates": required_gates,
        "missing_or_failed_gates": missing_or_failed,
        "gate_summary": {
            "quality_ok": quality_ok,
            "task_quality_available": task_quality_available,
            "quality_proxy_only": quality_proxy_only,
            "runtime_ok": runtime_ok,
            "runtime_available": runtime_available,
            "memory_ok": memory_ok,
            "memory_available": memory_available,
            "safety_ok": safety_ok,
            "safety_available": safety_available,
        },
        "quality_gate": {
            "gate_name": quality_gate.get("gate_name"),
            "quality_ok": quality_ok,
            "available": quality_available,
            "task_quality_available": task_quality_available,
            "proxy_only": quality_proxy_only,
            "reason": quality_gate.get("reason"),
        },
        "runtime_gate": {
            "gate_name": runtime_gate.get("gate_name"),
            "available": runtime_available,
            "runtime_ok": runtime_ok,
            "reason": runtime_gate.get("reason"),
        },
        "memory_gate": {
            "gate_name": memory_gate.get("gate_name"),
            "available": memory_available,
            "memory_ok": memory_ok,
            "reason": memory_gate.get("reason"),
        },
        "safety_gate": {
            "gate_name": safety_gate.get("gate_name"),
            "available": safety_available,
            "safety_ok": safety_ok,
        },
        "old_champion_scorer_behavior_unchanged": True,
        "note": "v25 P12 promotion decision; promotion requires real task quality, runtime, memory, and safety gates",
    }


def write_distillation_promotion_decision(decision: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_distillation_promotion_decision(output_path: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_path))


def validate_distillation_promotion_decision(decision: dict[str, Any]) -> dict[str, Any]:
    if decision.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad promotion schema_version={decision.get('schema_version')!r}")
    if decision.get("status") != "distillation_promotion_decision_ok":
        raise ValueError(f"bad promotion status={decision.get('status')!r}")
    for key, expected in {
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "old_champion_scorer_behavior_unchanged": True,
    }.items():
        if bool(decision.get(key)) is not expected:
            raise ValueError(f"promotion decision {key} must be {expected}")

    required_gates = decision.get("required_gates")
    if not isinstance(required_gates, dict) or not required_gates:
        raise ValueError("promotion decision required_gates must be a non-empty object")
    missing_or_failed = decision.get("missing_or_failed_gates")
    if not isinstance(missing_or_failed, list):
        raise ValueError("promotion decision missing_or_failed_gates must be a list")
    recomputed_missing = [name for name, passed in required_gates.items() if not bool(passed)]
    if missing_or_failed != recomputed_missing:
        raise ValueError("promotion decision missing_or_failed_gates does not match required_gates")

    promote = bool(decision.get("promote"))
    if promote:
        if missing_or_failed:
            raise ValueError("promotion cannot pass with missing or failed gates")
        if decision.get("decision") != "promoted":
            raise ValueError("promoted decision must have decision='promoted'")
        if decision.get("reason") != "all_real_gates_passed":
            raise ValueError("promoted decision must use all_real_gates_passed reason")
        if not bool(decision.get("promotion_eligible")):
            raise ValueError("promoted decision must be promotion_eligible")
    else:
        if decision.get("decision") != "not_promoted":
            raise ValueError("non-promoted decision must have decision='not_promoted'")
        if not missing_or_failed:
            raise ValueError("non-promoted decision must list missing or failed gates")
        if bool(decision.get("promotion_eligible")):
            raise ValueError("non-promoted decision must not be promotion_eligible")

    gate_summary = decision.get("gate_summary")
    if not isinstance(gate_summary, dict):
        raise ValueError("promotion decision gate_summary must be an object")
    if bool(gate_summary.get("quality_proxy_only")) and promote:
        raise ValueError("proxy-only quality cannot promote")
    if not bool(gate_summary.get("task_quality_available")) and promote:
        raise ValueError("task quality must be available to promote")
    if not bool(gate_summary.get("runtime_ok")) and promote:
        raise ValueError("runtime gate must pass to promote")
    if not bool(gate_summary.get("memory_ok")) and promote:
        raise ValueError("memory gate must pass to promote")
    if not bool(gate_summary.get("safety_ok")) and promote:
        raise ValueError("safety gate must pass to promote")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "distillation_promotion_decision_valid",
        "promote": promote,
        "decision": decision["decision"],
        "reason": decision["reason"],
        "missing_or_failed_gates": list(missing_or_failed),
    }


def run_and_write_distillation_promotion_decision(
    gate_report_or_path: dict[str, Any] | str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    decision = decide_distillation_promotion(gate_report_or_path)
    validate_distillation_promotion_decision(decision)
    if output_path is None:
        if isinstance(gate_report_or_path, dict):
            raise ValueError("output_path is required when gate_report_or_path is an in-memory dict")
        path = Path(gate_report_or_path).parent / DEFAULT_PROMOTION_DECISION_FILENAME
    else:
        path = Path(output_path)
    write_distillation_promotion_decision(decision, path)
    return decision
