from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from src.qwen_kl_distillation_trainable import run_logit_bias_training_loop
from src.qwen_logit_distillation_targets import (
    load_frozen_logit_distillation_targets_manifest,
    validate_frozen_logit_distillation_targets,
    write_frozen_logit_distillation_targets,
)
from src.qwen_rewire_proposal import (
    PROPOSED_ADJACENCY_FILENAME,
    load_proposed_adjacency,
    load_rewire_proposal_report,
    validate_rewire_proposal_report,
)
from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency


SCHEMA_VERSION = "qwen_rewire_acceptance.v1"
ACCEPTANCE_REPORT_FILENAME = "rewire_acceptance_report.json"
CANDIDATE_EVAL_DIRNAME = "candidate_eval"


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


def _node_count(edges: list[dict[str, Any]]) -> int:
    return len({str(edge["src_id"]) for edge in edges} | {str(edge["dst_id"]) for edge in edges})


def _max_out_degree(edges: list[dict[str, Any]]) -> int:
    counts: dict[str, int] = {}
    for edge in edges:
        src = str(edge["src_id"])
        counts[src] = counts.get(src, 0) + 1
    return max(counts.values(), default=0)


def _load_prior_config(eval_output_dir: Path) -> dict[str, Any]:
    path = eval_output_dir / "prior_config.json"
    return _read_json(path) if path.exists() else {}


def materialize_candidate_eval_dir(
    *,
    base_eval_output_dir: str | Path,
    proposed_adjacency: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    base = Path(base_eval_output_dir)
    out = Path(output_dir)
    selected_dir = out / "selected_adjacencies"
    selected_dir.mkdir(parents=True, exist_ok=True)
    prior_config = _load_prior_config(base)
    base_index_path = base / "selected_adjacencies" / "index.json"
    base_index = _read_json(base_index_path) if base_index_path.exists() else {}
    candidate_name = f"qwen_topk_k{int(proposed_adjacency['k'])}_v26_candidate"
    edges = [dict(edge) for edge in proposed_adjacency["edges"]]
    selected_payload = {
        "schema_version": "qwen_selected_adjacency.v1",
        "adjacency_name": candidate_name,
        "source": "G_0",
        "k": int(proposed_adjacency["k"]),
        "edge_count": len(edges),
        "node_count": _node_count(edges),
        "bounded": True,
        "selection_policy": "v26_proposal_candidate_not_applied",
        "edge_score_name": base_index.get("edge_score_name", prior_config.get("edge_score_name", "normalized_frobenius")),
        "graph_scope": base_index.get("graph_scope", prior_config.get("graph_scope", "attention_mlp_moe")),
        "edges": edges,
    }
    validate_selected_adjacency(selected_payload)
    rel_path = Path("selected_adjacencies") / f"{candidate_name}.json"
    _write_json(out / rel_path, selected_payload)

    index = {
        "schema_version": "qwen_selected_adjacency_index.v1",
        "source_weight_graph_dir": base_index.get("source_weight_graph_dir", prior_config.get("source_weight_graph_dir")),
        "graph_scope": selected_payload["graph_scope"],
        "edge_score_name": selected_payload["edge_score_name"],
        "selection_policy": "v26_proposal_candidate_not_applied",
        "bounded": True,
        "adjacencies": [
            {
                "adjacency_name": candidate_name,
                "k": int(selected_payload["k"]),
                "path": str(rel_path),
                "edge_count": int(selected_payload["edge_count"]),
                "node_count": int(selected_payload["node_count"]),
            }
        ],
    }
    _write_json(selected_dir / "index.json", index)
    handoff = {
        "schema_version": "qwen_v25_handoff.v1",
        "status": "ready_for_fixed_topology_sparse_student",
        "source_weight_graph_dir": index["source_weight_graph_dir"],
        "selected_adjacency_index": "selected_adjacencies/index.json",
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_required_before_deploy": True,
        "student_training_started": False,
        "quality_mode": prior_config.get("quality_mode", "energy_capture"),
        "graph_prior_quality_report": None,
        "promotion_decision": None,
    }
    _write_json(out / "v25_handoff_manifest.json", handoff)
    if prior_config:
        candidate_prior_config = dict(prior_config)
        candidate_prior_config["selected_adjacency_index"] = "selected_adjacencies/index.json"
        candidate_prior_config["v25_handoff_manifest"] = "v25_handoff_manifest.json"
        candidate_prior_config["v26_candidate_eval_dir"] = True
        candidate_prior_config["base_eval_output_dir"] = str(base)
        _write_json(out / "prior_config.json", candidate_prior_config)
    return {
        "candidate_eval_output_dir": str(out),
        "candidate_adjacency_name": candidate_name,
        "candidate_selected_adjacency": str(rel_path),
        "edge_count": int(selected_payload["edge_count"]),
        "node_count": int(selected_payload["node_count"]),
        "max_out_degree": _max_out_degree(edges),
        "bounded": True,
    }


def _validate_args(
    *,
    vocab_size: int,
    feature_dim: int,
    forward_steps: int,
    train_steps: int,
    lr: float,
    temperature: float,
    max_kl_regression: float,
) -> tuple[int, int, int, int, float, float, float]:
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
    kl_regression = _finite_float(max_kl_regression, name="max_kl_regression")
    if kl_regression < 0.0:
        raise ValueError(f"max_kl_regression must be >= 0, got {max_kl_regression!r}")
    return vocab_size, feature_dim, forward_steps, train_steps, lr_value, temp, kl_regression


def build_rewire_acceptance_report(
    *,
    eval_output_dir: str | Path,
    rewire_proposal_dir: str | Path,
    output_dir: str | Path,
    k: int | None = None,
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
    vocab_size, feature_dim, forward_steps, train_steps, lr, temperature, max_kl_regression = _validate_args(
        vocab_size=vocab_size,
        feature_dim=feature_dim,
        forward_steps=forward_steps,
        train_steps=train_steps,
        lr=lr,
        temperature=temperature,
        max_kl_regression=max_kl_regression,
    )
    base_eval_dir = Path(eval_output_dir)
    proposal_dir = Path(rewire_proposal_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    proposal_report = load_rewire_proposal_report(proposal_dir)
    validate_rewire_proposal_report(proposal_report)
    proposed_adjacency = load_proposed_adjacency(proposal_dir)
    base_k = int(proposal_report["k"] if k is None else k)
    base_adjacency = load_selected_adjacency(base_eval_dir, k=base_k)
    base_summary = validate_selected_adjacency(base_adjacency)

    candidate_eval = materialize_candidate_eval_dir(
        base_eval_output_dir=base_eval_dir,
        proposed_adjacency=proposed_adjacency,
        output_dir=out / CANDIDATE_EVAL_DIRNAME,
    )

    used_existing_targets = logit_targets_dir is not None
    if logit_targets_dir is None:
        targets_dir = out / "frozen_logit_targets"
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

    base_training = run_logit_bias_training_loop(
        base_eval_dir,
        targets_dir,
        k=base_k,
        feature_dim=feature_dim,
        steps=forward_steps,
        train_steps=train_steps,
        projection_seed=projection_seed,
        lr=lr,
        temperature=temperature,
        device=device,
    )
    candidate_training = run_logit_bias_training_loop(
        candidate_eval["candidate_eval_output_dir"],
        targets_dir,
        adjacency_name=candidate_eval["candidate_adjacency_name"],
        feature_dim=feature_dim,
        steps=forward_steps,
        train_steps=train_steps,
        projection_seed=projection_seed,
        lr=lr,
        temperature=temperature,
        device=device,
    )
    base_final = _finite_float(base_training["kl_final"], name="base.kl_final")
    candidate_final = _finite_float(candidate_training["kl_final"], name="candidate.kl_final")
    kl_delta = candidate_final - base_final
    quality_ok = candidate_final <= base_final + max_kl_regression
    candidate_training_ok = (
        bool(candidate_training["finite"])
        and bool(candidate_training["kl_decreased"])
        and bool(candidate_training["monotonic_nonincreasing"])
    )
    base_training_ok = (
        bool(base_training["finite"])
        and bool(base_training["kl_decreased"])
        and bool(base_training["monotonic_nonincreasing"])
    )
    safety_ok = (
        bool(proposal_report["proposal_bounded"])
        and not bool(proposal_report["topology_mutated"])
        and not bool(proposal_report["accepted"])
        and candidate_eval["edge_count"] <= base_summary["edge_count"]
        and candidate_eval["max_out_degree"] <= base_summary["k"]
    )
    accept = quality_ok and candidate_training_ok and base_training_ok and safety_ok
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "rewire_acceptance_decision_ok",
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
        "accepted": accept,
        "decision": "accepted_pending_apply" if accept else "rejected",
        "eval_output_dir": str(base_eval_dir),
        "rewire_proposal_dir": str(proposal_dir),
        "output_dir": str(out),
        "candidate_eval": candidate_eval,
        "base_adjacency_name": base_summary["adjacency_name"],
        "candidate_adjacency_name": candidate_eval["candidate_adjacency_name"],
        "k": base_summary["k"],
        "base_edge_count": base_summary["edge_count"],
        "candidate_edge_count": candidate_eval["edge_count"],
        "candidate_max_out_degree": candidate_eval["max_out_degree"],
        "swap_count": int(proposal_report["swap_count"]),
        "device": device,
        "vocab_size": int(target_manifest["vocab_size"]),
        "feature_dim": feature_dim,
        "forward_steps": forward_steps,
        "train_steps": train_steps,
        "lr": lr,
        "projection_seed": projection_seed,
        "temperature": float(target_manifest["temperature"]),
        "target_row_count": target_summary["row_count"],
        "used_existing_logit_targets": used_existing_targets,
        "base_kl_initial": base_training["kl_initial"],
        "base_kl_final": base_final,
        "candidate_kl_initial": candidate_training["kl_initial"],
        "candidate_kl_final": candidate_final,
        "candidate_minus_base_kl_final": kl_delta,
        "max_kl_regression": max_kl_regression,
        "quality_ok": quality_ok,
        "base_training_ok": base_training_ok,
        "candidate_training_ok": candidate_training_ok,
        "safety_ok": safety_ok,
        "acceptance_gate": {
            "quality_ok": quality_ok,
            "base_training_ok": base_training_ok,
            "candidate_training_ok": candidate_training_ok,
            "safety_ok": safety_ok,
            "accepted": accept,
            "rule": "candidate_kl_final <= base_kl_final + max_kl_regression and both bounded KL loops pass and proposal remains unapplied",
        },
        "base_training_report": base_training,
        "candidate_training_report": candidate_training,
        "artifacts": {
            "rewire_acceptance_report": ACCEPTANCE_REPORT_FILENAME,
            "candidate_eval_dir": CANDIDATE_EVAL_DIRNAME,
            "frozen_logit_targets": "frozen_logit_targets",
        },
        "finite": all(math.isfinite(float(v)) for v in (base_final, candidate_final, kl_delta)),
        "note": "v26 P2 accept/reject decision only; accepted proposals are not applied in this artifact",
    }
    validate_rewire_acceptance_report(report)
    return report


def validate_rewire_acceptance_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad rewire acceptance schema_version={report.get('schema_version')!r}")
    if report.get("status") != "rewire_acceptance_decision_ok":
        raise ValueError(f"bad rewire acceptance status={report.get('status')!r}")
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
            raise ValueError(f"rewire acceptance {key} must be {expected}")
    if int(report.get("candidate_edge_count", -1)) > int(report.get("base_edge_count", -2)):
        raise ValueError("candidate edge count must not exceed base edge count")
    if int(report.get("candidate_max_out_degree", -1)) > int(report.get("k", -2)):
        raise ValueError("candidate max_out_degree must not exceed k")
    expected_accept = (
        bool(report.get("quality_ok"))
        and bool(report.get("base_training_ok"))
        and bool(report.get("candidate_training_ok"))
        and bool(report.get("safety_ok"))
    )
    if bool(report.get("accepted")) is not expected_accept:
        raise ValueError("accepted must match acceptance gate conjunction")
    if bool(report.get("accepted")) and report.get("decision") != "accepted_pending_apply":
        raise ValueError("accepted report must use decision accepted_pending_apply")
    if not bool(report.get("accepted")) and report.get("decision") != "rejected":
        raise ValueError("rejected report must use decision rejected")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "rewire_acceptance_report_valid",
        "accepted": bool(report["accepted"]),
        "decision": report["decision"],
        "candidate_minus_base_kl_final": report["candidate_minus_base_kl_final"],
    }


def write_rewire_acceptance_report(report: dict[str, Any], output_dir: str | Path) -> None:
    validate_rewire_acceptance_report(report)
    _write_json(Path(output_dir) / ACCEPTANCE_REPORT_FILENAME, report)


def load_rewire_acceptance_report(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / ACCEPTANCE_REPORT_FILENAME)


def run_and_write_rewire_acceptance_report(
    *,
    eval_output_dir: str | Path,
    rewire_proposal_dir: str | Path,
    output_dir: str | Path,
    k: int | None = None,
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
    report = build_rewire_acceptance_report(
        eval_output_dir=eval_output_dir,
        rewire_proposal_dir=rewire_proposal_dir,
        output_dir=output_dir,
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
    write_rewire_acceptance_report(report, output_dir)
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


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("target-seeds must contain at least one integer")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v26 P2 bounded rewiring proposal accept/reject gate.")
    parser.add_argument("--eval-output-dir", required=True)
    parser.add_argument("--rewire-proposal-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=_positive_int, default=None)
    parser.add_argument("--logit-targets-dir", default=None)
    parser.add_argument("--vocab-size", type=_positive_int, default=16)
    parser.add_argument("--target-seeds", type=_parse_int_list, default=[0, 1, 2])
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
        report = run_and_write_rewire_acceptance_report(
            eval_output_dir=args.eval_output_dir,
            rewire_proposal_dir=args.rewire_proposal_dir,
            output_dir=args.output_dir,
            k=args.k,
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
        "rewire_acceptance_report": str(Path(args.output_dir) / ACCEPTANCE_REPORT_FILENAME),
        "decision": report["decision"],
        "accepted": report["accepted"],
        "quality_ok": report["quality_ok"],
        "base_training_ok": report["base_training_ok"],
        "candidate_training_ok": report["candidate_training_ok"],
        "safety_ok": report["safety_ok"],
        "base_kl_final": report["base_kl_final"],
        "candidate_kl_final": report["candidate_kl_final"],
        "candidate_minus_base_kl_final": report["candidate_minus_base_kl_final"],
        "proposal_applied": report["proposal_applied"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
