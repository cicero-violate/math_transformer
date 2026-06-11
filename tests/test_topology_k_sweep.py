from pathlib import Path
import csv
import json

from src.eval import QualityReport
from src.topology_k_sweep import summarize_reports, run_topology_k_sweep

PROJ = Path(__file__).resolve().parents[1]


def _report(mode, k, correct, n=10, expert="generic_expert", flags=None):
    if flags is None:
        flags = [True] * correct + [False] * (n - correct)
    return QualityReport(
        mode=mode,
        k=k,
        n_examples=n,
        route_accuracy=correct / n,
        correct_count=correct,
        correct_by_example=flags,
        by_expert={expert: {"correct": correct, "total": n, "accuracy": correct / n}},
    )


def test_summarize_reports_adds_correct_and_baseline_deltas():
    dense_flags = [True, True, True, True, True, True, True, False, False, False]
    hand4_flags = [True, True, True, True, True, True, False, True, True, True]
    hand8_flags = [True, True, True, True, False, False, True, True, True, True]
    learned4_flags = [True] * 10
    rows = summarize_reports(
        dense_report=_report("full", None, 7, flags=dense_flags),
        hand_reports=[
            _report("topology_only", 4, 9, flags=hand4_flags),
            _report("topology_only", 8, 8, flags=hand8_flags),
        ],
        learned_reports=[_report("learned_topology", 4, 10, flags=learned4_flags)],
    )

    dense, hand4, hand8, learned4 = rows
    assert dense["mode"] == "dense"
    assert dense["k"] == "full"
    assert dense["correct_count"] == 7
    assert hand4["wins_vs_dense"] == 3
    assert hand4["losses_vs_dense"] == 1
    assert hand4["correct_delta_vs_hand_k4"] == 0
    assert hand8["wins_vs_hand_k4"] == 1
    assert hand8["losses_vs_hand_k4"] == 2
    assert learned4["mode"] == "learned"
    assert learned4["wins_vs_hand_k4"] == 1
    assert learned4["losses_vs_hand_k4"] == 0
    assert "_correct_by_example" not in learned4
    assert learned4["generic_expert_acc"] == 1.0


def test_run_topology_k_sweep_writes_json_and_csv_without_learned(tmp_path):
    json_out = tmp_path / "summary.json"
    csv_out = tmp_path / "summary.csv"
    rows = run_topology_k_sweep(
        examples_path=str(PROJ / "data" / "examples.jsonl"),
        checkpoint=None,
        k_values=[1, 2, 4],
        learned_scorer_checkpoint=None,
        device="cpu",
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        topk=1,
        local_window=1,
        json_out=json_out,
        csv_out=csv_out,
    )

    assert [r["mode"] for r in rows] == ["dense", "hand", "hand", "hand"]
    assert [r["k"] for r in rows] == ["full", 1, 2, 4]
    assert all("correct_count" in r for r in rows)
    assert all("wins_vs_dense" in r for r in rows)
    assert all("correct_delta_vs_hand_k4" in r for r in rows)
    assert json.loads(json_out.read_text()) == rows
    with csv_out.open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)
    assert "by_expert" not in csv_rows[0]


def test_run_topology_k_sweep_uses_per_k_fixed_k_for_hand(monkeypatch, tmp_path):
    import src.topology_k_sweep as sweep

    calls = []

    def fake_run_quality_eval(**kwargs):
        calls.append(kwargs)
        k_values = kwargs["k_values"]
        if k_values:
            k = k_values[0]
            return [
                _report("full", None, 7),
                _report("topology_only", k, 7 + k),
            ]
        learned_k = kwargs["learned_k"]
        return [
            _report("full", None, 7),
            _report("learned_topology", learned_k, 8 + learned_k),
        ]

    monkeypatch.setattr(sweep, "run_quality_eval", fake_run_quality_eval)
    rows = sweep.run_topology_k_sweep(
        examples_path="examples.jsonl",
        checkpoint="dense.pt",
        k_values=[2, 4],
        learned_scorer_checkpoint="scorer.pt",
        json_out=tmp_path / "summary.json",
        csv_out=tmp_path / "summary.csv",
    )

    hand_calls = [c for c in calls if c["k_values"]]
    learned_calls = [c for c in calls if not c["k_values"]]
    assert [(c["k_values"], c["fixed_k"]) for c in hand_calls] == [([2], 2), ([4], 4)]
    assert [c["learned_k"] for c in learned_calls] == [2, 4]
    assert [(c["learned_k"], c["fixed_k"]) for c in learned_calls] == [(2, 4), (4, 4)]
    assert [(r["mode"], r["k"]) for r in rows] == [
        ("dense", "full"),
        ("hand", 2),
        ("hand", 4),
        ("learned", 2),
        ("learned", 4),
    ]
