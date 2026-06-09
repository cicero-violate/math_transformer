from pathlib import Path

from src.eval import _load_route_eval_records, run_quality_eval


PROJ = Path(__file__).resolve().parents[1]


def test_load_route_eval_records_extracts_env():
    records = _load_route_eval_records(str(PROJ / "data" / "examples.jsonl"))

    assert records
    assert records[0]["expr"] == "add(matmul(A,x),b)"
    assert records[0]["expert_id"] >= 0
    assert records[0]["expert"] == "affine_expert"
    assert records[0]["env"]["A"] == (32, 64)
    assert "out" not in records[0]["env"]


def test_run_quality_eval_reports_full_and_sparse_k_values():
    reports = run_quality_eval(
        examples_path=str(PROJ / "data" / "examples.jsonl"),
        k_values=[1, 2],
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        topk=1,
        local_window=1,
        device="cpu",
    )

    assert [r.mode for r in reports] == ["full", "topology_only", "topology_only"]
    assert [r.k for r in reports] == [None, 1, 2]
    assert all(0.0 <= r.route_accuracy <= 1.0 for r in reports)
    assert reports[0].dense_agreement is None
    assert all(0.0 <= r.dense_agreement <= 1.0 for r in reports[1:])
    assert reports[0].hidden_l1 is None
    assert reports[0].hidden_cos is None
    assert reports[0].logit_l1 is None
    assert reports[0].logit_kl_dense_to_sparse is None
    for report in reports[1:]:
        assert report.hidden_l1 is not None
        assert report.hidden_l1 >= 0.0
        assert report.hidden_cos is not None
        assert -1.0 <= report.hidden_cos <= 1.0
        assert report.logit_l1 is not None
        assert report.logit_l1 >= 0.0
        assert report.logit_kl_dense_to_sparse is not None
        assert report.logit_kl_dense_to_sparse >= -1e-7
    assert reports[0].by_expert
    assert all("accuracy" in stats for stats in reports[0].by_expert.values())
    assert "by_expert" in str(reports[0])
    assert "hidden_l1" in str(reports[1])
    assert "logit_kl" in str(reports[1])
