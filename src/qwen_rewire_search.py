from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from src.qwen_rewire_acceptance import run_and_write_rewire_acceptance_report, validate_rewire_acceptance_report
from src.qwen_rewire_proposal import run_and_write_rewire_proposal_report, validate_rewire_proposal_report


SCHEMA_VERSION = "qwen_rewire_search.v1"
REWIRE_SEARCH_REPORT_FILENAME = "rewire_search_report.json"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
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


def _dedupe_positive_ints(values: Sequence[int], *, name: str) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for raw in values:
        value = int(raw)
        if value < 1:
            raise ValueError(f"{name} must contain only integers >= 1, got {value}")
        if value not in seen:
            result.append(value)
            seen.add(value)
    if not result:
        raise ValueError(f"{name} must contain at least one integer")
    return result


def _candidate_row(
    *,
    candidate_index: int,
    max_swaps: int,
    proposal_dir: Path,
    acceptance_dir: Path,
    proposal_report: dict[str, Any],
    acceptance_report: dict[str, Any],
) -> dict[str, Any]:
    validate_rewire_proposal_report({**proposal_report, "proposed_adjacency": _read_json(proposal_dir / "proposed_adjacency.json")})
    validate_rewire_acceptance_report(acceptance_report)
    return {
        "candidate_index": candidate_index,
        "max_swaps": max_swaps,
        "proposal_dir": str(proposal_dir),
        "acceptance_dir": str(acceptance_dir),
        "proposal_bounded": bool(proposal_report["proposal_bounded"]),
        "proposal_swap_count": int(proposal_report["swap_count"]),
        "proposal_topology_mutated": bool(proposal_report["topology_mutated"]),
        "accepted": bool(acceptance_report["accepted"]),
        "decision": acceptance_report["decision"],
        "quality_ok": bool(acceptance_report["quality_ok"]),
        "base_training_ok": bool(acceptance_report["base_training_ok"]),
        "candidate_training_ok": bool(acceptance_report["candidate_training_ok"]),
        "safety_ok": bool(acceptance_report["safety_ok"]),
        "base_kl_final": float(acceptance_report["base_kl_final"]),
        "candidate_kl_final": float(acceptance_report["candidate_kl_final"]),
        "candidate_minus_base_kl_final": float(acceptance_report["candidate_minus_base_kl_final"]),
        "proposal_applied": bool(acceptance_report["proposal_applied"]),
    }


def _best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate rows must be non-empty")
    return min(
        rows,
        key=lambda row: (
            float(row["candidate_minus_base_kl_final"]),
            int(row["proposal_swap_count"]),
            int(row["candidate_index"]),
        ),
    )


def _first_accepted_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if bool(row["accepted"]):
            return row
    return None


def build_rewire_search_report(
    *,
    eval_output_dir: str | Path,
    edge_trace_dir: str | Path,
    output_dir: str | Path,
    k: int | None = None,
    adjacency_name: str | None = None,
    max_swaps_values: Sequence[int] | None = None,
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
    max_kl_regression: float = 0.0,
    selection_policy: str = "first_accepted_else_best_kl_delta",
) -> dict[str, Any]:
    swaps_values = _dedupe_positive_ints(
        [1, 2, 3] if max_swaps_values is None else max_swaps_values,
        name="max_swaps_values",
    )
    if selection_policy != "first_accepted_else_best_kl_delta":
        raise ValueError(f"unsupported selection_policy={selection_policy!r}")
    max_kl_regression = _finite_float(max_kl_regression, name="max_kl_regression")
    if max_kl_regression < 0.0:
        raise ValueError("max_kl_regression must be >= 0")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    for idx, max_swaps in enumerate(swaps_values):
        candidate_root = out / "candidates" / f"candidate_{idx:03d}_swaps_{max_swaps}"
        proposal_dir = candidate_root / "proposal"
        acceptance_dir = candidate_root / "acceptance"
        proposal_report = run_and_write_rewire_proposal_report(
            eval_output_dir,
            edge_trace_dir,
            proposal_dir,
            k=k,
            adjacency_name=adjacency_name,
            max_swaps=max_swaps,
        )
        acceptance_report = run_and_write_rewire_acceptance_report(
            eval_output_dir=eval_output_dir,
            rewire_proposal_dir=proposal_dir,
            output_dir=acceptance_dir,
            k=k,
            logit_targets_dir=logit_targets_dir,
            vocab_size=vocab_size,
            target_seeds=target_seeds,
            feature_dim=feature_dim,
            forward_steps=forward_steps,
            train_steps=train_steps,
            lr=lr,
            projection_seed=projection_seed,
            temperature=temperature,
            device=device,
            max_kl_regression=max_kl_regression,
        )
        candidate_rows.append(
            _candidate_row(
                candidate_index=idx,
                max_swaps=max_swaps,
                proposal_dir=proposal_dir,
                acceptance_dir=acceptance_dir,
                proposal_report=proposal_report,
                acceptance_report=acceptance_report,
            )
        )

    first_accepted = _first_accepted_candidate(candidate_rows)
    best = _best_candidate(candidate_rows)
    selected = first_accepted if first_accepted is not None else best
    accepted_candidate_count = sum(1 for row in candidate_rows if bool(row["accepted"]))
    any_accepted = accepted_candidate_count > 0
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "rewire_search_ok",
        "student_training_started": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "kl_training_started": True,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "topology_mutated": False,
        "proposal_applied": False,
        "promotion_eligible": False,
        "eval_output_dir": str(Path(eval_output_dir)),
        "edge_trace_dir": str(Path(edge_trace_dir)),
        "output_dir": str(out),
        "k": k,
        "adjacency_name": adjacency_name,
        "max_swaps_values": swaps_values,
        "candidate_count": len(candidate_rows),
        "accepted_candidate_count": accepted_candidate_count,
        "any_accepted": any_accepted,
        "decision": "accepted_candidate_found" if any_accepted else "no_candidate_accepted",
        "selection_policy": selection_policy,
        "selected_candidate_index": int(selected["candidate_index"]),
        "selected_candidate_accepted": bool(selected["accepted"]),
        "selected_candidate_max_swaps": int(selected["max_swaps"]),
        "selected_candidate_kl_delta": float(selected["candidate_minus_base_kl_final"]),
        "best_candidate_index": int(best["candidate_index"]),
        "best_candidate_kl_delta": float(best["candidate_minus_base_kl_final"]),
        "device": device,
        "vocab_size": vocab_size,
        "feature_dim": feature_dim,
        "forward_steps": forward_steps,
        "train_steps": train_steps,
        "lr": lr,
        "projection_seed": projection_seed,
        "temperature": temperature,
        "max_kl_regression": max_kl_regression,
        "candidates": candidate_rows,
        "artifacts": {
            "rewire_search_report": REWIRE_SEARCH_REPORT_FILENAME,
            "candidate_root": "candidates",
        },
        "finite": all(math.isfinite(float(row["candidate_minus_base_kl_final"])) for row in candidate_rows),
        "note": "v26 P3 search only; selected proposals are not applied or promoted",
    }
    validate_rewire_search_report(report)
    return report


def validate_rewire_search_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad rewire search schema_version={report.get('schema_version')!r}")
    if report.get("status") != "rewire_search_ok":
        raise ValueError(f"bad rewire search status={report.get('status')!r}")
    for key, expected in {
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "topology_mutated": False,
        "proposal_applied": False,
        "promotion_eligible": False,
        "finite": True,
    }.items():
        if bool(report.get(key)) is not expected:
            raise ValueError(f"rewire search {key} must be {expected}")
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("rewire search candidates must be a non-empty list")
    if int(report.get("candidate_count", -1)) != len(candidates):
        raise ValueError("candidate_count must match len(candidates)")
    accepted_count = sum(1 for row in candidates if bool(row.get("accepted")))
    if int(report.get("accepted_candidate_count", -1)) != accepted_count:
        raise ValueError("accepted_candidate_count mismatch")
    if bool(report.get("any_accepted")) is not (accepted_count > 0):
        raise ValueError("any_accepted mismatch")
    expected_decision = "accepted_candidate_found" if accepted_count > 0 else "no_candidate_accepted"
    if report.get("decision") != expected_decision:
        raise ValueError("rewire search decision mismatch")
    selected_index = int(report.get("selected_candidate_index", -1))
    if selected_index < 0 or selected_index >= len(candidates):
        raise ValueError("selected_candidate_index out of range")
    selected = candidates[selected_index]
    if bool(report.get("selected_candidate_accepted")) is not bool(selected.get("accepted")):
        raise ValueError("selected_candidate_accepted mismatch")
    for row in candidates:
        if bool(row.get("proposal_topology_mutated")) or bool(row.get("proposal_applied")):
            raise ValueError("search candidates must not mutate or apply topology")
        if not bool(row.get("proposal_bounded")) or not bool(row.get("safety_ok")):
            raise ValueError("search candidates must remain bounded and safe")
        _finite_float(row.get("candidate_minus_base_kl_final"), name="candidate_minus_base_kl_final")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "rewire_search_report_valid",
        "candidate_count": len(candidates),
        "accepted_candidate_count": accepted_count,
        "decision": report["decision"],
        "selected_candidate_index": selected_index,
    }


def write_rewire_search_report(report: dict[str, Any], output_dir: str | Path) -> None:
    validate_rewire_search_report(report)
    _write_json(Path(output_dir) / REWIRE_SEARCH_REPORT_FILENAME, report)


def load_rewire_search_report(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / REWIRE_SEARCH_REPORT_FILENAME)


def run_and_write_rewire_search_report(
    *,
    eval_output_dir: str | Path,
    edge_trace_dir: str | Path,
    output_dir: str | Path,
    k: int | None = None,
    adjacency_name: str | None = None,
    max_swaps_values: Sequence[int] | None = None,
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
    max_kl_regression: float = 0.0,
) -> dict[str, Any]:
    report = build_rewire_search_report(
        eval_output_dir=eval_output_dir,
        edge_trace_dir=edge_trace_dir,
        output_dir=output_dir,
        k=k,
        adjacency_name=adjacency_name,
        max_swaps_values=max_swaps_values,
        logit_targets_dir=logit_targets_dir,
        vocab_size=vocab_size,
        target_seeds=target_seeds,
        feature_dim=feature_dim,
        forward_steps=forward_steps,
        train_steps=train_steps,
        lr=lr,
        projection_seed=projection_seed,
        temperature=temperature,
        device=device,
        max_kl_regression=max_kl_regression,
    )
    write_rewire_search_report(report, output_dir)
    return report


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {raw!r}")
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


def _parse_int_list(raw: str, *, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one integer")
    bad = [value for value in values if value < 1]
    if bad:
        raise argparse.ArgumentTypeError(f"{name} must contain only integers >= 1, got {bad[0]}")
    return values


def _parse_seed_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("target-seeds must contain at least one integer")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v26 P3 bounded rewiring proposal search.")
    parser.add_argument("--eval-output-dir", required=True)
    parser.add_argument("--edge-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--k", type=_positive_int, default=None)
    group.add_argument("--adjacency-name", default=None)
    parser.add_argument("--max-swaps-values", type=lambda raw: _parse_int_list(raw, name="max-swaps-values"), default=[1, 2, 3])
    parser.add_argument("--logit-targets-dir", default=None)
    parser.add_argument("--vocab-size", type=_positive_int, default=16)
    parser.add_argument("--target-seeds", type=_parse_seed_list, default=[0, 1, 2])
    parser.add_argument("--feature-dim", type=_positive_int, default=8)
    parser.add_argument("--forward-steps", type=_positive_int, default=1)
    parser.add_argument("--train-steps", type=_positive_int, default=5)
    parser.add_argument("--lr", type=_positive_float, default=0.1)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--temperature", type=_positive_float, default=1.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "torch_cpu", "cuda", "auto"])
    parser.add_argument("--max-kl-regression", type=_nonnegative_float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_and_write_rewire_search_report(
            eval_output_dir=args.eval_output_dir,
            edge_trace_dir=args.edge_trace_dir,
            output_dir=args.output_dir,
            k=args.k,
            adjacency_name=args.adjacency_name,
            max_swaps_values=args.max_swaps_values,
            logit_targets_dir=args.logit_targets_dir,
            vocab_size=args.vocab_size,
            target_seeds=args.target_seeds,
            feature_dim=args.feature_dim,
            forward_steps=args.forward_steps,
            train_steps=args.train_steps,
            lr=args.lr,
            projection_seed=args.projection_seed,
            temperature=args.temperature,
            device=args.device,
            max_kl_regression=args.max_kl_regression,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "rewire_search_report": str(Path(args.output_dir) / REWIRE_SEARCH_REPORT_FILENAME),
        "decision": report["decision"],
        "candidate_count": report["candidate_count"],
        "accepted_candidate_count": report["accepted_candidate_count"],
        "selected_candidate_index": report["selected_candidate_index"],
        "selected_candidate_accepted": report["selected_candidate_accepted"],
        "selected_candidate_kl_delta": report["selected_candidate_kl_delta"],
        "best_candidate_index": report["best_candidate_index"],
        "best_candidate_kl_delta": report["best_candidate_kl_delta"],
        "proposal_applied": report["proposal_applied"],
        "topology_mutated": report["topology_mutated"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
