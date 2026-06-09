from __future__ import annotations

from pathlib import Path

from src.dense_mix_sweep import parse_quality_reports, quality_score, score_log


QUALITY_TEXT = """
examples=data/synthetic_hard/val.jsonl  checkpoint=runs/checkpoints/synthetic_hard_dense.pt
mode=full  k=full  examples=1786  route_acc=0.9810
mode=topology_only  k=16  examples=1786  route_acc=0.9821  dense_agree=0.9989  hidden_l1=0.079942  hidden_cos=0.992537  logit_l1=0.046326  logit_kl=0.002046
mode=learned_topology  k=8  examples=1786  route_acc=0.9871  dense_agree=0.9938  hidden_l1=0.172168  hidden_cos=0.975618  logit_l1=0.114392  logit_kl=0.015902
"""


def test_parse_quality_reports_extracts_learned_metrics():
    rows = parse_quality_reports(QUALITY_TEXT)
    learned = [r for r in rows if r["mode"] == "learned_topology"][0]

    assert len(rows) == 3
    assert learned["k"] == "8"
    assert learned["examples"] == 1786
    assert learned["route_acc"] == 0.9871
    assert learned["hidden_cos"] == 0.975618
    assert learned["logit_kl"] == 0.015902


def test_score_log_returns_dense_equivalence_score():
    row = score_log(QUALITY_TEXT, mix="0.25")

    assert row["mix"] == "0.25"
    assert row["mode"] == "learned_topology"
    assert row["score"] == quality_score(row)
    assert float(row["score"]) > 1.0


def test_score_log_raises_without_learned_report():
    try:
        score_log("mode=full  k=full  examples=1  route_acc=1.0", mix="x")
    except ValueError as exc:
        assert "learned_topology" in str(exc)
    else:
        raise AssertionError("expected ValueError")
