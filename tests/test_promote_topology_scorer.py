from __future__ import annotations

import json
from pathlib import Path

from src.promote_topology_scorer import (
    BenchmarkGate,
    PromotionMetrics,
    decide_promotion,
    parse_benchmark_gate,
    promote_checkpoint,
)


def _metrics(route: float, generic: float, ckpt: str = "candidate.pt") -> PromotionMetrics:
    return PromotionMetrics(
        checkpoint=ckpt,
        route_acc=route,
        generic_acc=generic,
        generic_correct=int(round(generic * 100)),
        generic_total=100,
        examples=1000,
        dense_agree=0.99,
        hidden_cos=0.97,
        logit_kl=0.01,
    )


def test_parse_benchmark_gate_acceptance_line():
    text = """
speedup=1.006301
hand_k16_route_acc=0.982100
learned_k8_route_acc=0.987100
acceptance_passed quality_ok=True speed_ok=True strict_speed_ok=True
"""
    gate = parse_benchmark_gate(text)
    assert gate.passed is True
    assert gate.speedup == 1.006301
    assert gate.learned_route_acc == 0.9871
    assert gate.hand_route_acc == 0.9821


def test_decide_promotion_rejects_benchmark_failure():
    candidate = _metrics(0.99, 0.91)
    champion = _metrics(0.98, 0.90, ckpt="champion.pt")
    gate = BenchmarkGate(quality_ok=True, speed_ok=False, strict_speed_ok=True)
    decision = decide_promotion(candidate=candidate, champion=champion, benchmark=gate)
    assert decision.promote is False
    assert decision.reason == "benchmark_gate_failed"


def test_decide_promotion_rejects_route_or_generic_regression():
    gate = BenchmarkGate(quality_ok=True, speed_ok=True, strict_speed_ok=True)
    champion = _metrics(0.9871, 0.9112, ckpt="champion.pt")

    route_bad = decide_promotion(candidate=_metrics(0.9866, 0.92), champion=champion, benchmark=gate)
    assert route_bad.promote is False
    assert route_bad.reason == "route_acc_regression"

    generic_bad = decide_promotion(candidate=_metrics(0.9871, 0.9073), champion=champion, benchmark=gate)
    assert generic_bad.promote is False
    assert generic_bad.reason == "generic_acc_regression"


def test_decide_promotion_accepts_non_regression_or_improvement():
    gate = BenchmarkGate(quality_ok=True, speed_ok=True, strict_speed_ok=True)
    champion = _metrics(0.9871, 0.9112, ckpt="champion.pt")
    same = decide_promotion(candidate=_metrics(0.9871, 0.9112), champion=champion, benchmark=gate)
    assert same.promote is True
    assert same.reason == "non_regression"
    improved = decide_promotion(candidate=_metrics(0.9872, 0.9112), champion=champion, benchmark=gate)
    assert improved.promote is True
    assert improved.reason == "route_acc_improved"


def test_promote_checkpoint_copies_and_writes_metadata(tmp_path: Path):
    candidate_path = tmp_path / "candidate.pt"
    champion_path = tmp_path / "champion.pt"
    metadata_path = tmp_path / "champion.json"
    candidate_path.write_bytes(b"checkpoint-bytes")
    gate = BenchmarkGate(quality_ok=True, speed_ok=True, strict_speed_ok=True, speedup=1.01)
    decision = decide_promotion(
        candidate=_metrics(0.9871, 0.9112, ckpt=str(candidate_path)),
        champion=None,
        benchmark=gate,
    )

    payload = promote_checkpoint(
        decision=decision,
        champion_checkpoint=champion_path,
        champion_metadata=metadata_path,
        candidate_checkpoint=candidate_path,
    )

    assert payload["promoted"] is True
    assert champion_path.read_bytes() == b"checkpoint-bytes"
    data = json.loads(metadata_path.read_text())
    assert data["checkpoint"] == str(champion_path)
    assert data["source_checkpoint"] == str(candidate_path)
    assert data["metrics"]["route_acc"] == 0.9871
    assert data["benchmark"]["speedup"] == 1.01
