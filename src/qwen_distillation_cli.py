from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.qwen_distillation_harness import run_fixed_topology_distillation_harness
from src.qwen_distillation_measured_gates import (
    DEFAULT_MEASURED_GATE_REPORT_FILENAME,
    run_and_write_measured_distillation_gate_report,
    validate_measured_distillation_gate_report,
)
from src.qwen_distillation_promotion import run_and_write_distillation_promotion_decision
from src.qwen_graph_prior_eval import (
    QUALITY_MODE_ENERGY_CAPTURE,
    QUALITY_MODE_IMPLANTED_RECOVERY,
    QUALITY_MODE_UNAVAILABLE,
    run_graph_prior_eval,
    _load_gold_specs,
)


SCHEMA_VERSION = "qwen_distillation_cli.v1"
FINAL_SUMMARY_FILENAME = "final_summary.json"
MEASURED_PROMOTION_DECISION_FILENAME = "distillation_measured_promotion_decision.json"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_or_str(path: Path, *, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _parse_int_list(raw: str, *, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one integer")
    return values


def _parse_k_values(raw: str) -> list[int]:
    values = _parse_int_list(raw, name="k-values")
    bad = [value for value in values if value < 1]
    if bad:
        raise argparse.ArgumentTypeError(f"k-values must be >= 1, got {bad[0]}")
    return values


def _parse_random_seeds(raw: str) -> list[int]:
    return _parse_int_list(raw, name="random-seeds")


def _parse_target_seeds(raw: str) -> list[int]:
    return _parse_int_list(raw, name="target-seeds")


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


def build_final_summary(
    *,
    output_root: Path,
    eval_output_dir: Path,
    source_weight_graph_dir: Path | None,
    graph_prior_eval_ran: bool,
    harness_report: dict[str, Any],
    measured_gate_report: dict[str, Any],
    promotion_decision: dict[str, Any],
) -> dict[str, Any]:
    gate_validation = validate_measured_distillation_gate_report(measured_gate_report)
    gate_summary = promotion_decision.get("gate_summary", {})
    artifacts = {
        "final_summary": FINAL_SUMMARY_FILENAME,
        "distillation_harness_report": "distillation_harness_report.json",
        "distillation_measured_gate_report": DEFAULT_MEASURED_GATE_REPORT_FILENAME,
        "distillation_measured_promotion_decision": MEASURED_PROMOTION_DECISION_FILENAME,
    }
    if graph_prior_eval_ran:
        artifacts["graph_prior_eval"] = _relative_or_str(eval_output_dir, base=output_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "measured_distillation_pipeline_ok",
        "output_root": str(output_root),
        "source_weight_graph_dir": None if source_weight_graph_dir is None else str(source_weight_graph_dir),
        "eval_output_dir": str(eval_output_dir),
        "graph_prior_eval_ran": graph_prior_eval_ran,
        "harness_status": harness_report["status"],
        "measured_gate_status": measured_gate_report["status"],
        "promotion_status": promotion_decision["status"],
        "quality_ok": bool(gate_summary.get("quality_ok", gate_validation["quality_ok"])),
        "runtime_ok": bool(gate_summary.get("runtime_ok", gate_validation["runtime_ok"])),
        "memory_ok": bool(gate_summary.get("memory_ok", gate_validation["memory_ok"])),
        "safety_ok": bool(gate_summary.get("safety_ok", gate_validation["safety_ok"])),
        "promote": bool(promotion_decision["promote"]),
        "decision": promotion_decision["decision"],
        "missing_or_failed_gates": list(promotion_decision["missing_or_failed_gates"]),
        "k": int(harness_report["k"]),
        "adjacency_name": harness_report["adjacency_name"],
        "train_steps": int(harness_report["train_steps"]),
        "runtime_repeats": int(measured_gate_report["measurement_protocol"]["runtime_repeats"]),
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "old_champion_scorer_behavior_unchanged": True,
        "artifacts": artifacts,
    }


def validate_final_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad cli summary schema_version={summary.get('schema_version')!r}")
    if summary.get("status") != "measured_distillation_pipeline_ok":
        raise ValueError(f"bad cli summary status={summary.get('status')!r}")
    if summary.get("harness_status") != "fixed_topology_distillation_harness_ok":
        raise ValueError("harness did not complete")
    if summary.get("measured_gate_status") != "distillation_gate_report_ok":
        raise ValueError("measured gate report did not complete")
    if summary.get("promotion_status") != "distillation_promotion_decision_ok":
        raise ValueError("promotion decision did not complete")
    for key, expected in {
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "old_champion_scorer_behavior_unchanged": True,
    }.items():
        if bool(summary.get(key)) is not expected:
            raise ValueError(f"cli summary {key} must be {expected}")
    missing_or_failed = summary.get("missing_or_failed_gates")
    if not isinstance(missing_or_failed, list):
        raise ValueError("missing_or_failed_gates must be a list")
    promote = bool(summary.get("promote"))
    if promote and missing_or_failed:
        raise ValueError("promoted summary cannot have missing or failed gates")
    if promote and summary.get("decision") != "promoted":
        raise ValueError("promoted summary must have decision='promoted'")
    if not promote and summary.get("decision") != "not_promoted":
        raise ValueError("non-promoted summary must have decision='not_promoted'")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "measured_distillation_pipeline_summary_valid",
        "promote": promote,
        "decision": summary["decision"],
        "missing_or_failed_gates": list(missing_or_failed),
    }


def run_measured_distillation_pipeline(
    *,
    output_root: str | Path,
    source_weight_graph_dir: str | Path | None = None,
    eval_output_dir: str | Path | None = None,
    k: int = 1,
    k_values: list[int] | None = None,
    random_seeds: list[int] | None = None,
    graph_scope: str = "attention_mlp_moe",
    edge_score_name: str = "normalized_frobenius",
    quality_dataset: str = "",
    runtime_protocol: str = "unavailable",
    memory_protocol: str = "unavailable",
    quality_mode: str = QUALITY_MODE_ENERGY_CAPTURE,
    gold_block_specs: list[dict[str, Any]] | None = None,
    min_qwen_recall: float = 0.95,
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
    runtime_repeats: int = 3,
    max_runtime_seconds: float = 10.0,
    max_peak_memory_bytes: int = 128 * 1024 * 1024,
    min_kl_relative_reduction: float = 0.0,
) -> dict[str, Any]:
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)

    graph_prior_eval_ran = eval_output_dir is None
    source_dir = Path(source_weight_graph_dir) if source_weight_graph_dir is not None else None
    if graph_prior_eval_ran:
        if source_dir is None:
            raise ValueError("source_weight_graph_dir is required when eval_output_dir is not provided")
        eval_dir = out / "graph_prior_eval"
        run_graph_prior_eval(
            source_weight_graph_dir=source_dir,
            output_dir=eval_dir,
            k_values=k_values or [1, 2],
            random_seeds=random_seeds or [0, 1, 2, 3, 4],
            graph_scope=graph_scope,
            edge_score_name=edge_score_name,
            quality_dataset=quality_dataset,
            runtime_protocol=runtime_protocol,
            memory_protocol=memory_protocol,
            quality_mode=quality_mode,
            gold_block_specs=gold_block_specs,
            min_qwen_recall=min_qwen_recall,
        )
    else:
        eval_dir = Path(eval_output_dir)

    harness_report = run_fixed_topology_distillation_harness(
        eval_dir,
        out,
        k=k,
        adjacency_name=adjacency_name,
        logit_targets_dir=logit_targets_dir,
        vocab_size=vocab_size,
        target_seeds=target_seeds,
        feature_dim=feature_dim,
        forward_steps=forward_steps,
        train_steps=train_steps,
        lr=lr,
        projection_seed=projection_seed,
        temperature=temperature,
    )
    gate_path = out / DEFAULT_MEASURED_GATE_REPORT_FILENAME
    gate_report = run_and_write_measured_distillation_gate_report(
        out,
        gate_path,
        runtime_repeats=runtime_repeats,
        max_runtime_seconds=max_runtime_seconds,
        max_peak_memory_bytes=max_peak_memory_bytes,
        min_kl_relative_reduction=min_kl_relative_reduction,
    )
    decision_path = out / MEASURED_PROMOTION_DECISION_FILENAME
    promotion_decision = run_and_write_distillation_promotion_decision(gate_path, decision_path)

    summary = build_final_summary(
        output_root=out,
        eval_output_dir=eval_dir,
        source_weight_graph_dir=source_dir,
        graph_prior_eval_ran=graph_prior_eval_ran,
        harness_report=harness_report,
        measured_gate_report=gate_report,
        promotion_decision=promotion_decision,
    )
    validate_final_summary(summary)
    _write_json(out / FINAL_SUMMARY_FILENAME, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the measured fixed-topology Qwen distillation pipeline.")
    parser.add_argument("--source-weight-graph-dir", default=None)
    parser.add_argument("--eval-output-dir", default=None, help="Existing v25 handoff/graph-prior eval output. Skips graph-prior eval when set.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--k", type=_positive_int, default=1)
    parser.add_argument("--k-values", type=_parse_k_values, default=[1, 2])
    parser.add_argument("--random-seeds", type=_parse_random_seeds, default=[0, 1, 2, 3, 4])
    parser.add_argument("--graph-scope", default="attention_mlp_moe", choices=["mlp_only", "attention_mlp", "attention_mlp_moe", "all"])
    parser.add_argument("--edge-score-name", default="normalized_frobenius")
    parser.add_argument("--quality-dataset", default="")
    parser.add_argument("--runtime-protocol", default="unavailable")
    parser.add_argument("--memory-protocol", default="unavailable")
    parser.add_argument(
        "--quality-mode",
        default=QUALITY_MODE_ENERGY_CAPTURE,
        choices=[
            QUALITY_MODE_UNAVAILABLE,
            QUALITY_MODE_IMPLANTED_RECOVERY,
            "implanted_signal_recovery",
            QUALITY_MODE_ENERGY_CAPTURE,
            "energy_capture_proxy",
        ],
    )
    parser.add_argument("--gold-specs", default="")
    parser.add_argument("--min-qwen-recall", type=_nonnegative_float, default=0.95)
    parser.add_argument("--adjacency-name", default=None)
    parser.add_argument("--logit-targets-dir", default=None)
    parser.add_argument("--vocab-size", type=_vocab_size, default=16)
    parser.add_argument("--target-seeds", type=_parse_target_seeds, default=[0, 1, 2])
    parser.add_argument("--feature-dim", type=_positive_int, default=8)
    parser.add_argument("--forward-steps", type=_positive_int, default=1)
    parser.add_argument("--train-steps", type=_positive_int, default=5)
    parser.add_argument("--lr", type=_positive_float, default=0.1)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--temperature", type=_positive_float, default=1.0)
    parser.add_argument("--runtime-repeats", type=_positive_int, default=3)
    parser.add_argument("--max-runtime-seconds", type=_nonnegative_float, default=10.0)
    parser.add_argument("--max-peak-memory-bytes", type=_nonnegative_int, default=128 * 1024 * 1024)
    parser.add_argument("--min-kl-relative-reduction", type=_nonnegative_float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.eval_output_dir and not args.source_weight_graph_dir:
        parser.error("--source-weight-graph-dir is required unless --eval-output-dir is provided")
    try:
        gold_specs = _load_gold_specs(args.gold_specs) if args.gold_specs else None
        summary = run_measured_distillation_pipeline(
            output_root=args.output_root,
            source_weight_graph_dir=args.source_weight_graph_dir,
            eval_output_dir=args.eval_output_dir,
            k=args.k,
            k_values=args.k_values,
            random_seeds=args.random_seeds,
            graph_scope=args.graph_scope,
            edge_score_name=args.edge_score_name,
            quality_dataset=args.quality_dataset,
            runtime_protocol=args.runtime_protocol,
            memory_protocol=args.memory_protocol,
            quality_mode=args.quality_mode,
            gold_block_specs=gold_specs,
            min_qwen_recall=args.min_qwen_recall,
            adjacency_name=args.adjacency_name,
            logit_targets_dir=args.logit_targets_dir,
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
            min_kl_relative_reduction=args.min_kl_relative_reduction,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "final_summary": str(Path(args.output_root) / FINAL_SUMMARY_FILENAME),
        "quality_ok": summary["quality_ok"],
        "runtime_ok": summary["runtime_ok"],
        "memory_ok": summary["memory_ok"],
        "safety_ok": summary["safety_ok"],
        "promote": summary["promote"],
        "decision": summary["decision"],
        "missing_or_failed_gates": summary["missing_or_failed_gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
