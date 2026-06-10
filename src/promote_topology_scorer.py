from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .eval import QualityReport, run_quality_eval
from .topology_benchmark_artifact import load_artifact


@dataclass(frozen=True)
class PromotionMetrics:
    checkpoint: str
    route_acc: float
    generic_acc: float
    generic_correct: int
    generic_total: int
    examples: int
    dense_agree: float | None = None
    hidden_cos: float | None = None
    logit_kl: float | None = None


@dataclass(frozen=True)
class BenchmarkGate:
    quality_ok: bool
    speed_ok: bool
    strict_speed_ok: bool
    speedup: float = 0.0
    learned_route_acc: float | None = None
    hand_route_acc: float | None = None

    @property
    def passed(self) -> bool:
        return self.quality_ok and self.speed_ok and self.strict_speed_ok


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    candidate: PromotionMetrics
    champion: PromotionMetrics | None
    benchmark: BenchmarkGate | None


def _generic_stats(report: QualityReport) -> tuple[float, int, int]:
    generic = (report.by_expert or {}).get("generic_expert", {})
    correct = int(generic.get("correct", 0) or 0)
    total = int(generic.get("total", 0) or 0)
    acc = float(generic.get("accuracy", 0.0) or 0.0)
    return acc, correct, total


def metrics_from_quality_report(report: QualityReport, checkpoint: str) -> PromotionMetrics:
    generic_acc, generic_correct, generic_total = _generic_stats(report)
    return PromotionMetrics(
        checkpoint=checkpoint,
        route_acc=float(report.route_accuracy),
        generic_acc=generic_acc,
        generic_correct=generic_correct,
        generic_total=generic_total,
        examples=int(report.n_examples),
        dense_agree=None if report.dense_agreement is None else float(report.dense_agreement),
        hidden_cos=None if report.hidden_cos is None else float(report.hidden_cos),
        logit_kl=None if report.logit_kl_dense_to_sparse is None else float(report.logit_kl_dense_to_sparse),
    )


def evaluate_scorer_metrics(
    *,
    checkpoint: str,
    examples_path: str,
    dense_checkpoint: str,
    learned_k: int = 8,
    hand_k: int = 16,
    device: str | None = "auto",
) -> PromotionMetrics:
    reports = run_quality_eval(
        examples_path=examples_path,
        k_values=[hand_k],
        checkpoint=dense_checkpoint,
        device=device,
        topology_mode="middle_preserving_topk",
        fixed_k=hand_k,
        middle_bridge_width=1,
        learned_scorer_checkpoint=checkpoint,
        learned_k=learned_k,
    )
    learned = [r for r in reports if r.mode == "learned_topology"]
    if not learned:
        raise RuntimeError("quality eval produced no learned_topology report")
    return metrics_from_quality_report(learned[-1], checkpoint)


def parse_benchmark_gate(text: str) -> BenchmarkGate:
    stripped = text.strip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
        return benchmark_gate_from_artifact(data)
    accept = re.search(
        r"acceptance_passed\s+quality_ok=(True|False)\s+speed_ok=(True|False)\s+strict_speed_ok=(True|False)",
        text,
    )
    if not accept:
        raise ValueError("benchmark log missing acceptance_passed quality/speed line")
    speedup_match = re.search(r"^speedup=([0-9.]+)", text, re.MULTILINE)
    learned_match = re.search(r"^learned_k\d+_route_acc=([0-9.]+)", text, re.MULTILINE)
    hand_match = re.search(r"^hand_k\d+_route_acc=([0-9.]+)", text, re.MULTILINE)
    return BenchmarkGate(
        quality_ok=accept.group(1) == "True",
        speed_ok=accept.group(2) == "True",
        strict_speed_ok=accept.group(3) == "True",
        speedup=float(speedup_match.group(1)) if speedup_match else 0.0,
        learned_route_acc=float(learned_match.group(1)) if learned_match else None,
        hand_route_acc=float(hand_match.group(1)) if hand_match else None,
    )


def benchmark_gate_from_artifact(data: dict[str, Any]) -> BenchmarkGate:
    acceptance = data.get("acceptance", {}) or {}
    speed = data.get("speed", {}) or {}
    quality = data.get("quality", {}) or {}
    learned = quality.get("learned", {}) or {}
    hand = quality.get("hand", {}) or {}
    return BenchmarkGate(
        quality_ok=bool(acceptance.get("quality_ok", quality.get("quality_ok", False))),
        speed_ok=bool(acceptance.get("speed_ok", speed.get("speed_ok", False))),
        strict_speed_ok=bool(acceptance.get("strict_speed_ok", speed.get("strict_speed_ok", False))),
        speedup=float(speed.get("speedup", 0.0) or 0.0),
        learned_route_acc=None if learned.get("route_acc") is None else float(learned.get("route_acc")),
        hand_route_acc=None if hand.get("route_acc") is None else float(hand.get("route_acc")),
    )


def parse_benchmark_gate_file(path: str | Path) -> BenchmarkGate:
    p = Path(path)
    if p.suffix in {".json", ".jsonl"}:
        return benchmark_gate_from_artifact(load_artifact(p))
    return parse_benchmark_gate(p.read_text(encoding="utf-8", errors="replace"))


def champion_regression_from_artifact(
    *,
    artifact: dict[str, Any],
    champion: PromotionMetrics,
    route_min_delta: float = 0.0,
    generic_min_delta: float = 0.0,
) -> PromotionDecision:
    quality = artifact.get("quality", {}) or {}
    learned = quality.get("learned", {}) or {}
    by_expert = learned.get("by_expert", {}) or {}
    generic = by_expert.get("generic_expert", {}) or {}
    total = int(generic.get("total", 0) or 0)
    correct = int(generic.get("correct", 0) or 0)
    generic_acc = float(generic.get("accuracy", 0.0) or 0.0)
    if total == 0 and champion.generic_total > 0:
        return PromotionDecision(
            False,
            "missing_generic_expert_metrics",
            PromotionMetrics(
                checkpoint=str((artifact.get("paths", {}) or {}).get("scorer", "")),
                route_acc=float(learned.get("route_acc", 0.0) or 0.0),
                generic_acc=0.0,
                generic_correct=0,
                generic_total=0,
                examples=int(learned.get("examples", 0) or 0),
            ),
            champion,
            benchmark_gate_from_artifact(artifact),
        )
    candidate = PromotionMetrics(
        checkpoint=str((artifact.get("paths", {}) or {}).get("scorer", "")),
        route_acc=float(learned.get("route_acc", 0.0) or 0.0),
        generic_acc=generic_acc,
        generic_correct=correct,
        generic_total=total,
        examples=int(learned.get("examples", 0) or 0),
        dense_agree=learned.get("dense_agree"),
        hidden_cos=learned.get("hidden_cos"),
        logit_kl=learned.get("logit_kl"),
    )
    return decide_promotion(
        candidate=candidate,
        champion=champion,
        benchmark=benchmark_gate_from_artifact(artifact),
        require_benchmark=True,
        route_min_delta=route_min_delta,
        generic_min_delta=generic_min_delta,
    )




def decide_promotion(
    *,
    candidate: PromotionMetrics,
    champion: PromotionMetrics | None,
    benchmark: BenchmarkGate | None,
    require_benchmark: bool = True,
    route_min_delta: float = 0.0,
    generic_min_delta: float = 0.0,
) -> PromotionDecision:
    if require_benchmark and benchmark is None:
        return PromotionDecision(False, "missing_benchmark_gate", candidate, champion, benchmark)
    if benchmark is not None and not benchmark.passed:
        return PromotionDecision(False, "benchmark_gate_failed", candidate, champion, benchmark)
    if champion is None:
        return PromotionDecision(True, "no_existing_champion", candidate, champion, benchmark)
    if candidate.route_acc + 1e-12 < champion.route_acc + route_min_delta:
        return PromotionDecision(False, "route_acc_regression", candidate, champion, benchmark)
    if candidate.generic_acc + 1e-12 < champion.generic_acc + generic_min_delta:
        return PromotionDecision(False, "generic_acc_regression", candidate, champion, benchmark)
    if candidate.route_acc > champion.route_acc + 1e-12:
        return PromotionDecision(True, "route_acc_improved", candidate, champion, benchmark)
    if candidate.generic_acc > champion.generic_acc + 1e-12:
        return PromotionDecision(True, "generic_acc_improved", candidate, champion, benchmark)
    return PromotionDecision(True, "non_regression", candidate, champion, benchmark)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_from_champion_metadata(path: str | Path) -> PromotionMetrics | None:
    meta_path = Path(path)
    if not meta_path.exists():
        return None
    data = _load_json(meta_path)
    metrics = data.get("metrics", data)
    return PromotionMetrics(
        checkpoint=str(data.get("checkpoint", metrics.get("checkpoint", ""))),
        route_acc=float(metrics.get("route_acc", 0.0)),
        generic_acc=float(metrics.get("generic_acc", 0.0)),
        generic_correct=int(metrics.get("generic_correct", 0) or 0),
        generic_total=int(metrics.get("generic_total", 0) or 0),
        examples=int(metrics.get("examples", 0) or 0),
        dense_agree=metrics.get("dense_agree"),
        hidden_cos=metrics.get("hidden_cos"),
        logit_kl=metrics.get("logit_kl"),
    )


def promote_checkpoint(
    *,
    decision: PromotionDecision,
    champion_checkpoint: str | Path,
    champion_metadata: str | Path,
    candidate_checkpoint: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    if not decision.promote and not force:
        return {
            "promoted": False,
            "reason": decision.reason,
            "candidate": asdict(decision.candidate),
            "champion": None if decision.champion is None else asdict(decision.champion),
            "benchmark": None if decision.benchmark is None else asdict(decision.benchmark),
        }
    src = Path(candidate_checkpoint)
    dst = Path(champion_checkpoint)
    meta = Path(champion_metadata)
    if not src.exists():
        raise FileNotFoundError(f"candidate checkpoint not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    payload = {
        "promoted": True,
        "reason": decision.reason if decision.promote else "forced",
        "checkpoint": str(dst),
        "source_checkpoint": str(src),
        "metrics": asdict(decision.candidate),
        "previous_champion": None if decision.champion is None else asdict(decision.champion),
        "benchmark": None if decision.benchmark is None else asdict(decision.benchmark),
    }
    meta.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a learned topology scorer checkpoint only if it passes champion gates.")
    parser.add_argument("--candidate", required=True, help="Candidate scorer checkpoint path.")
    parser.add_argument("--champion-checkpoint", default="runs/checkpoints/topology_scorer.champion.pt")
    parser.add_argument("--champion-metadata", default="runs/checkpoints/topology_scorer.champion.json")
    parser.add_argument("--examples", default="data/synthetic_hard/val.jsonl")
    parser.add_argument("--dense-checkpoint", default="runs/checkpoints/synthetic_hard_dense.pt")
    parser.add_argument("--learned-k", type=int, default=8)
    parser.add_argument("--hand-k", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--benchmark-log", default=None, help="Optional benchmark log containing acceptance_passed quality/speed line.")
    parser.add_argument("--no-require-benchmark", action="store_true")
    parser.add_argument("--route-min-delta", type=float, default=0.0)
    parser.add_argument("--generic-min-delta", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    candidate = evaluate_scorer_metrics(
        checkpoint=args.candidate,
        examples_path=args.examples,
        dense_checkpoint=args.dense_checkpoint,
        learned_k=args.learned_k,
        hand_k=args.hand_k,
        device=args.device,
    )
    champion = metrics_from_champion_metadata(args.champion_metadata)
    benchmark = parse_benchmark_gate_file(args.benchmark_log) if args.benchmark_log else None
    decision = decide_promotion(
        candidate=candidate,
        champion=champion,
        benchmark=benchmark,
        require_benchmark=not args.no_require_benchmark,
        route_min_delta=args.route_min_delta,
        generic_min_delta=args.generic_min_delta,
    )
    result = promote_checkpoint(
        decision=decision,
        champion_checkpoint=args.champion_checkpoint,
        champion_metadata=args.champion_metadata,
        candidate_checkpoint=args.candidate,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("promoted"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
