from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.qwen_distillation_harness import run_fixed_topology_distillation_harness, validate_distillation_harness_report
from src.qwen_distillation_measured_gates import (
    run_and_write_measured_distillation_gate_report,
    validate_measured_distillation_gate_report,
)
from src.qwen_distillation_promotion import (
    run_and_write_distillation_promotion_decision,
    validate_distillation_promotion_decision,
)
from src.qwen_rewire_apply import load_accepted_candidate_manifest, validate_accepted_candidate_manifest
from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency


SCHEMA_VERSION = "qwen_rewire_candidate_promotion.v1"
CANDIDATE_DISTILLATION_HARNESS_REPORT_FILENAME = "candidate_distillation_harness_report.json"
CANDIDATE_MEASURED_GATE_REPORT_FILENAME = "candidate_measured_gate_report.json"
CANDIDATE_PROMOTION_DECISION_FILENAME = "candidate_promotion_decision.json"
CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME = "candidate_next_prior_manifest.json"
_HARNESS_WORK_DIRNAME = "candidate_distillation_harness"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_positive_int(value: int, *, name: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return number


def _validate_nonnegative_int(value: int, *, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return number


def _validate_nonnegative_float(value: float, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return number


def _validate_positive_float(value: float, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return number


def _copy_report_alias(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(str(source))
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def build_candidate_next_prior_manifest(
    *,
    accepted_candidate_manifest: dict[str, Any],
    accepted_candidate_manifest_path: str | Path,
    output_dir: str | Path,
    candidate_distillation_harness_report: str | Path,
    candidate_measured_gate_report: str | Path,
    candidate_promotion_decision: str | Path,
) -> dict[str, Any]:
    validate_accepted_candidate_manifest(accepted_candidate_manifest)
    out = Path(output_dir)
    applied_eval_dir = Path(str(accepted_candidate_manifest["applied_candidate_eval_dir"]))
    candidate_adjacency_name = str(accepted_candidate_manifest["candidate_adjacency_name"])
    adjacency = load_selected_adjacency(applied_eval_dir, adjacency_name=candidate_adjacency_name)
    adjacency_summary = validate_selected_adjacency(adjacency)
    if int(adjacency_summary["max_out_degree"]) > int(accepted_candidate_manifest["k"]):
        raise ValueError("candidate adjacency remains unbounded relative to k")

    harness_report = _read_json(Path(candidate_distillation_harness_report))
    validate_distillation_harness_report(harness_report)
    measured_gate_report = _read_json(Path(candidate_measured_gate_report))
    gate_summary = validate_measured_distillation_gate_report(measured_gate_report)
    promotion_decision_report = _read_json(Path(candidate_promotion_decision))
    validate_distillation_promotion_decision(promotion_decision_report)

    quality_ok = bool(gate_summary["quality_ok"])
    runtime_ok = bool(gate_summary["runtime_ok"])
    memory_ok = bool(gate_summary["memory_ok"])
    safety_ok = bool(gate_summary["safety_ok"])
    promoted_by_measured_decision = bool(promotion_decision_report["promote"])
    accepted_materialized = bool(accepted_candidate_manifest["candidate_materialized"])
    base_topology_mutated = bool(accepted_candidate_manifest["base_topology_mutated"])
    active_topology_mutated = bool(accepted_candidate_manifest["active_topology_mutated"])
    proposal_applied_to_base = bool(accepted_candidate_manifest["proposal_applied_to_base"])
    bounded_active_adjacency = bool(adjacency_summary["bounded"])
    candidate_promoted = (
        quality_ok
        and runtime_ok
        and memory_ok
        and safety_ok
        and promoted_by_measured_decision
        and accepted_materialized
        and not base_topology_mutated
        and not active_topology_mutated
        and not proposal_applied_to_base
        and bounded_active_adjacency
    )
    decision = "candidate_promoted_as_next_prior" if candidate_promoted else "candidate_not_promoted"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_next_prior_manifest_ok",
        "accepted_candidate_manifest": str(Path(accepted_candidate_manifest_path)),
        "applied_candidate_eval_dir": str(applied_eval_dir),
        "candidate_adjacency_name": candidate_adjacency_name,
        "selected_candidate_index": int(accepted_candidate_manifest["selected_candidate_index"]),
        "selected_candidate_policy": accepted_candidate_manifest["selected_candidate_policy"],
        "selected_candidate_kl_delta": float(accepted_candidate_manifest["selected_candidate_kl_delta"]),
        "candidate_distillation_harness_report": str(Path(candidate_distillation_harness_report)),
        "candidate_measured_gate_report": str(Path(candidate_measured_gate_report)),
        "candidate_promotion_decision": str(Path(candidate_promotion_decision)),
        "candidate_promoted": candidate_promoted,
        "decision": decision,
        "quality_ok": quality_ok,
        "runtime_ok": runtime_ok,
        "memory_ok": memory_ok,
        "safety_ok": safety_ok,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": bounded_active_adjacency,
        "base_topology_mutated": base_topology_mutated,
        "active_topology_mutated": active_topology_mutated,
        "proposal_applied_to_base": proposal_applied_to_base,
        "candidate_materialized": accepted_materialized,
        "promotion_eligible": candidate_promoted,
        "promote": promoted_by_measured_decision,
        "candidate_adjacency_summary": adjacency_summary,
        "missing_or_failed_gates": list(promotion_decision_report.get("missing_or_failed_gates", [])),
        "output_dir": str(out),
        "artifacts": {
            "candidate_next_prior_manifest": CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME,
            "candidate_distillation_harness_report": CANDIDATE_DISTILLATION_HARNESS_REPORT_FILENAME,
            "candidate_measured_gate_report": CANDIDATE_MEASURED_GATE_REPORT_FILENAME,
            "candidate_promotion_decision": CANDIDATE_PROMOTION_DECISION_FILENAME,
        },
        "note": "Promotes the accepted rewiring candidate as a next-prior artifact only; this does not mutate base A_t or active topology.",
    }
    validate_candidate_next_prior_manifest(manifest)
    return manifest


def validate_candidate_next_prior_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad candidate promotion schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "candidate_next_prior_manifest_ok":
        raise ValueError(f"bad candidate promotion status={manifest.get('status')!r}")
    for key, expected in {
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "base_topology_mutated": False,
        "active_topology_mutated": False,
        "proposal_applied_to_base": False,
        "candidate_materialized": True,
    }.items():
        if bool(manifest.get(key)) is not expected:
            raise ValueError(f"candidate next-prior manifest {key} must be {expected}")
    for key in (
        "accepted_candidate_manifest",
        "applied_candidate_eval_dir",
        "candidate_distillation_harness_report",
        "candidate_measured_gate_report",
        "candidate_promotion_decision",
    ):
        path = Path(str(manifest.get(key)))
        if not path.exists():
            raise ValueError(f"candidate next-prior manifest path missing: {key} -> {path}")
    adjacency_summary = manifest.get("candidate_adjacency_summary")
    if not isinstance(adjacency_summary, dict) or not bool(adjacency_summary.get("bounded")):
        raise ValueError("candidate adjacency summary must be bounded")
    candidate_promoted = bool(manifest.get("candidate_promoted"))
    expected_promoted = (
        bool(manifest.get("quality_ok"))
        and bool(manifest.get("runtime_ok"))
        and bool(manifest.get("memory_ok"))
        and bool(manifest.get("safety_ok"))
        and bool(manifest.get("promote"))
        and bool(manifest.get("candidate_materialized"))
        and not bool(manifest.get("base_topology_mutated"))
        and not bool(manifest.get("active_topology_mutated"))
        and not bool(manifest.get("proposal_applied_to_base"))
        and bool(manifest.get("bounded_active_adjacency"))
    )
    if candidate_promoted is not expected_promoted:
        raise ValueError("candidate_promoted must match candidate promotion conjunction")
    if candidate_promoted:
        if manifest.get("decision") != "candidate_promoted_as_next_prior":
            raise ValueError("promoted candidate must use candidate_promoted_as_next_prior decision")
        if not bool(manifest.get("promotion_eligible")):
            raise ValueError("promoted candidate must be promotion_eligible")
    else:
        if manifest.get("decision") != "candidate_not_promoted":
            raise ValueError("non-promoted candidate must use candidate_not_promoted decision")
        if bool(manifest.get("promotion_eligible")):
            raise ValueError("non-promoted candidate must not be promotion_eligible")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_next_prior_manifest_valid",
        "candidate_promoted": candidate_promoted,
        "decision": manifest["decision"],
        "candidate_adjacency_name": manifest["candidate_adjacency_name"],
    }


def write_candidate_next_prior_manifest(manifest: dict[str, Any], output_dir: str | Path) -> None:
    validate_candidate_next_prior_manifest(manifest)
    _write_json(Path(output_dir) / CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME, manifest)


def load_candidate_next_prior_manifest(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME)


def run_and_write_candidate_promotion_report(
    *,
    accepted_candidate_dir: str | Path,
    output_dir: str | Path,
    k: int = 1,
    vocab_size: int = 16,
    target_seeds: list[int] | None = None,
    feature_dim: int = 8,
    forward_steps: int = 1,
    train_steps: int = 5,
    lr: float = 0.1,
    projection_seed: int = 0,
    temperature: float = 1.0,
    runtime_repeats: int = 3,
    max_runtime_seconds: float = 10.0,
    max_peak_memory_bytes: int = 128 * 1024 * 1024,
    max_cuda_peak_memory_bytes: int | None = None,
    min_kl_relative_reduction: float = 0.0,
    device: str = "cpu",
) -> dict[str, Any]:
    k = _validate_positive_int(k, name="k")
    vocab_size = int(vocab_size)
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size!r}")
    feature_dim = _validate_positive_int(feature_dim, name="feature_dim")
    forward_steps = _validate_positive_int(forward_steps, name="forward_steps")
    train_steps = _validate_positive_int(train_steps, name="train_steps")
    lr = _validate_positive_float(lr, name="lr")
    temperature = _validate_positive_float(temperature, name="temperature")
    runtime_repeats = _validate_positive_int(runtime_repeats, name="runtime_repeats")
    max_runtime_seconds = _validate_nonnegative_float(max_runtime_seconds, name="max_runtime_seconds")
    max_peak_memory_bytes = _validate_nonnegative_int(max_peak_memory_bytes, name="max_peak_memory_bytes")
    if max_cuda_peak_memory_bytes is not None:
        max_cuda_peak_memory_bytes = _validate_nonnegative_int(max_cuda_peak_memory_bytes, name="max_cuda_peak_memory_bytes")
    min_kl_relative_reduction = _validate_nonnegative_float(min_kl_relative_reduction, name="min_kl_relative_reduction")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted_dir = Path(accepted_candidate_dir)
    accepted_manifest_path = accepted_dir / "accepted_candidate_manifest.json"
    accepted_manifest = load_accepted_candidate_manifest(accepted_dir)
    validate_accepted_candidate_manifest(accepted_manifest)
    if int(accepted_manifest["k"]) != k:
        raise ValueError(f"k must match accepted candidate k={accepted_manifest['k']}, got {k}")

    candidate_adjacency_name = str(accepted_manifest["candidate_adjacency_name"])
    applied_eval_dir = Path(str(accepted_manifest["applied_candidate_eval_dir"]))
    harness_dir = out / _HARNESS_WORK_DIRNAME
    harness_report = run_fixed_topology_distillation_harness(
        applied_eval_dir,
        harness_dir,
        adjacency_name=candidate_adjacency_name,
        vocab_size=vocab_size,
        target_seeds=target_seeds,
        feature_dim=feature_dim,
        forward_steps=forward_steps,
        train_steps=train_steps,
        lr=lr,
        projection_seed=projection_seed,
        temperature=temperature,
        device=device,
    )
    if int(harness_report["k"]) != k:
        raise ValueError(f"candidate harness resolved k={harness_report['k']}, expected {k}")
    candidate_harness_report_path = out / CANDIDATE_DISTILLATION_HARNESS_REPORT_FILENAME
    _copy_report_alias(harness_dir / "distillation_harness_report.json", candidate_harness_report_path)

    candidate_gate_report_path = out / CANDIDATE_MEASURED_GATE_REPORT_FILENAME
    measured_gate_report = run_and_write_measured_distillation_gate_report(
        harness_dir,
        candidate_gate_report_path,
        runtime_repeats=runtime_repeats,
        max_runtime_seconds=max_runtime_seconds,
        max_peak_memory_bytes=max_peak_memory_bytes,
        min_kl_relative_reduction=min_kl_relative_reduction,
        device=device,
        max_cuda_peak_memory_bytes=max_cuda_peak_memory_bytes,
    )
    candidate_promotion_decision_path = out / CANDIDATE_PROMOTION_DECISION_FILENAME
    promotion_decision = run_and_write_distillation_promotion_decision(
        candidate_gate_report_path,
        candidate_promotion_decision_path,
    )

    manifest = build_candidate_next_prior_manifest(
        accepted_candidate_manifest=accepted_manifest,
        accepted_candidate_manifest_path=accepted_manifest_path,
        output_dir=out,
        candidate_distillation_harness_report=candidate_harness_report_path,
        candidate_measured_gate_report=candidate_gate_report_path,
        candidate_promotion_decision=candidate_promotion_decision_path,
    )
    write_candidate_next_prior_manifest(manifest, out)
    summary = {
        "status": "candidate_promotion_pipeline_ok",
        "decision": manifest["decision"],
        "candidate_promoted": manifest["candidate_promoted"],
        "candidate_next_prior_manifest": str(out / CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME),
        "candidate_distillation_harness_report": str(candidate_harness_report_path),
        "candidate_measured_gate_report": str(candidate_gate_report_path),
        "candidate_promotion_decision": str(candidate_promotion_decision_path),
        "quality_ok": manifest["quality_ok"],
        "runtime_ok": manifest["runtime_ok"],
        "memory_ok": manifest["memory_ok"],
        "safety_ok": manifest["safety_ok"],
        "base_topology_mutated": manifest["base_topology_mutated"],
        "active_topology_mutated": manifest["active_topology_mutated"],
        "proposal_applied_to_base": manifest["proposal_applied_to_base"],
        "candidate_materialized": manifest["candidate_materialized"],
        "promote": bool(promotion_decision["promote"]),
        "missing_or_failed_gates": list(promotion_decision["missing_or_failed_gates"]),
        "device": str(measured_gate_report["measurement_protocol"].get("device", device)),
        "resolved_device": str(measured_gate_report["measurement_protocol"].get("resolved_device", "cpu")),
    }
    return summary


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("target-seeds must contain at least one integer")
    return values


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {raw!r}")
    return value


def _vocab_size(raw: str) -> int:
    value = int(raw)
    if value < 2:
        raise argparse.ArgumentTypeError(f"must be >= 2, got {raw!r}")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {raw!r}")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {raw!r}")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote a v26 accepted rewiring candidate as a next sparse prior artifact.")
    parser.add_argument("--accepted-candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=_positive_int, default=1)
    parser.add_argument("--vocab-size", type=_vocab_size, default=16)
    parser.add_argument("--target-seeds", type=_parse_int_list, default=[0, 1, 2])
    parser.add_argument("--feature-dim", type=_positive_int, default=8)
    parser.add_argument("--forward-steps", type=_positive_int, default=1)
    parser.add_argument("--train-steps", type=_positive_int, default=5)
    parser.add_argument("--lr", type=_positive_float, default=0.1)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--temperature", type=_positive_float, default=1.0)
    parser.add_argument("--runtime-repeats", type=_positive_int, default=3)
    parser.add_argument("--max-runtime-seconds", type=_nonnegative_float, default=10.0)
    parser.add_argument("--max-peak-memory-bytes", type=_nonnegative_int, default=128 * 1024 * 1024)
    parser.add_argument("--max-cuda-peak-memory-bytes", type=_nonnegative_int, default=None)
    parser.add_argument("--min-kl-relative-reduction", type=_nonnegative_float, default=0.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "torch_cpu", "cuda", "auto"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_and_write_candidate_promotion_report(
            accepted_candidate_dir=args.accepted_candidate_dir,
            output_dir=args.output_dir,
            k=args.k,
            vocab_size=args.vocab_size,
            target_seeds=args.target_seeds,
            feature_dim=args.feature_dim,
            forward_steps=args.forward_steps,
            train_steps=args.train_steps,
            lr=args.lr,
            projection_seed=args.projection_seed,
            temperature=args.temperature,
            runtime_repeats=args.runtime_repeats,
            max_runtime_seconds=args.max_runtime_seconds,
            max_peak_memory_bytes=args.max_peak_memory_bytes,
            max_cuda_peak_memory_bytes=args.max_cuda_peak_memory_bytes,
            min_kl_relative_reduction=args.min_kl_relative_reduction,
            device=args.device,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
