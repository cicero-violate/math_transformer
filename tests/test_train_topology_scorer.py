from __future__ import annotations

import json
from pathlib import Path

import torch

from src.synthetic_data import generate_hard_records, write_jsonl
import src.train_topology_scorer as trainer_mod
from src.train_topology_scorer import (
    _compress_target,
    _load_replay_records,
    _load_replay_weights,
    _runtime_quality_selection_score,
    train_topology_scorer,
)


def test_train_topology_scorer_smoke(tmp_path: Path):
    data = tmp_path / "hard.jsonl"
    ckpt = tmp_path / "scorer.pt"
    records = generate_hard_records(8, seed=31, split="train", route_fraction=1.0, max_depth=3)
    write_jsonl(data, records)

    summary = train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(ckpt),
        max_steps=3,
        max_examples=4,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=1,
    )

    assert ckpt.exists()
    assert summary["examples"] == 4
    assert summary["steps"] == 3
    assert 0.0 <= float(summary["edge_recall"]) <= 1.0
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert "model_state_dict" in state
    assert state["target_k"] == 4

def test_compress_target_caps_rows_and_preserves_self():
    target = torch.ones(6, 6, dtype=torch.bool)
    compressed = _compress_target(target, keep_k=3)

    assert compressed.shape == target.shape
    assert torch.all(torch.diag(compressed))
    assert int(compressed.sum(dim=1).max().item()) == 3


def test_load_replay_weights_keeps_largest_weight_and_caps(tmp_path: Path):
    replay = tmp_path / "replay.jsonl"
    rows = [
        {"expr": "add(x, y)", "replay_score": 5.0},
        {"expr": "add(x, y)", "replay_score": 20.0},
        {"expr": "matmul(A, x)", "replay_score": 1.0},
        {"replay_score": 99.0},
    ]
    replay.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    weights = _load_replay_weights(str(replay), replay_weight_scale=0.5, replay_max_weight=4.0)

    assert weights["add(x, y)"] == 4.0
    assert weights["matmul(A, x)"] == 1.5
    assert len(weights) == 2


def test_train_topology_scorer_with_replay_candidates_records_metadata(tmp_path: Path):
    data = tmp_path / "hard.jsonl"
    ckpt = tmp_path / "scorer.pt"
    records = generate_hard_records(8, seed=43, split="train", route_fraction=1.0, max_depth=3)
    write_jsonl(data, records)
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        json.dumps({"expr": records[0]["expr"], "replay_score": 6.0}) + "\n",
        encoding="utf-8",
    )

    summary = train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(ckpt),
        max_steps=3,
        max_examples=4,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=1,
        replay_candidates_path=str(replay),
        replay_weight_scale=0.25,
        replay_max_weight=3.0,
    )

    assert ckpt.exists()
    assert summary["replay_candidates_path"] == str(replay)
    assert summary["replay_weighted_examples"] == 1
    assert summary["replay_weight_scale"] == 0.25
    assert summary["replay_max_weight"] == 3.0
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert state["replay_candidates_path"] == str(replay)
    assert state["replay_weighted_examples"] == 1
    assert state["replay_weight_scale"] == 0.25
    assert state["replay_max_weight"] == 3.0


def test_load_replay_records_loads_extra_training_records(tmp_path: Path):
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        json.dumps({
            "expr": "add(replay_x, replay_y)",
            "replay_score": 12.0,
            "target_expert": "generic_expert",
            "target_expert_id": 2,
            "env": {"replay_x": [4], "replay_y": [4]},
        }) + "\n",
        encoding="utf-8",
    )

    records = _load_replay_records(str(replay))

    assert len(records) == 1
    assert records[0]["expr"] == "add(replay_x, replay_y)"
    assert records[0]["expert"] == "generic_expert"
    assert records[0]["expert_id"] == 2
    assert records[0]["env"] == {"replay_x": (4,), "replay_y": (4,)}
    assert records[0]["source"] == "replay_candidate"


def test_train_topology_scorer_appends_replay_candidates_not_in_base_train(tmp_path: Path):
    data = tmp_path / "hard.jsonl"
    ckpt = tmp_path / "scorer.pt"
    records = generate_hard_records(8, seed=53, split="train", route_fraction=1.0, max_depth=3)
    write_jsonl(data, records)

    replay_expr = "add(replay_x, replay_y)"
    assert replay_expr not in {r["expr"] for r in records}
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        json.dumps({
            "expr": replay_expr,
            "replay_score": 10.0,
            "target_expert": "generic_expert",
            "target_expert_id": 2,
        }) + "\n",
        encoding="utf-8",
    )

    summary = train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(ckpt),
        max_steps=3,
        max_examples=4,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=1,
        replay_candidates_path=str(replay),
        replay_weight_scale=0.2,
        replay_max_weight=4.0,
    )

    assert summary["examples"] == 5
    assert summary["replay_appended_examples"] == 1
    assert summary["replay_weighted_examples"] == 1
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert state["replay_appended_examples"] == 1
    assert state["replay_weighted_examples"] == 1


def test_train_topology_scorer_replay_sample_ratio_forces_replay_steps(tmp_path: Path):
    data = tmp_path / "hard.jsonl"
    ckpt = tmp_path / "scorer.pt"
    records = generate_hard_records(8, seed=61, split="train", route_fraction=1.0, max_depth=3)
    write_jsonl(data, records)

    replay_expr = "add(replay_a, replay_b)"
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        json.dumps({
            "expr": replay_expr,
            "replay_score": 10.0,
            "target_expert": "generic_expert",
            "target_expert_id": 2,
        }) + "\n",
        encoding="utf-8",
    )

    summary = train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(ckpt),
        max_steps=3,
        max_examples=4,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=1,
        replay_candidates_path=str(replay),
        replay_weight_scale=0.2,
        replay_max_weight=4.0,
        replay_sample_ratio=1.0,
    )

    assert summary["examples"] == 5
    assert summary["replay_appended_examples"] == 1
    assert summary["replay_weighted_examples"] == 1
    assert summary["replay_sample_ratio"] == 1.0
    assert summary["replay_sampled_steps"] == 3
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert state["replay_sample_ratio"] == 1.0
    assert state["replay_sampled_steps"] == 3


def test_train_topology_scorer_runtime_quality_best_selection(monkeypatch, tmp_path: Path):
    data = tmp_path / "hard.jsonl"
    ckpt = tmp_path / "scorer.pt"
    runtime_best = tmp_path / "runtime_selected.pt"
    edge_best = tmp_path / "scorer.edge_best.pt"
    records = generate_hard_records(8, seed=71, split="train", route_fraction=1.0, max_depth=3)
    write_jsonl(data, records)
    calls = {"count": 0}

    def fake_runtime_score(**kwargs):
        calls["count"] += 1
        score = 0.1 * calls["count"]
        return score, {
            "route_acc": score,
            "dense_agree": 1.0,
            "hidden_cos": 0.99,
            "logit_kl": 0.01,
        }

    monkeypatch.setattr(trainer_mod, "_runtime_quality_score_for_checkpoint", fake_runtime_score)
    summary = train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(ckpt),
        max_steps=3,
        max_examples=4,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=1,
        eval_interval=1,
        val_examples_path=str(data),
        best_checkpoint=str(runtime_best),
        best_selection="runtime_quality",
        runtime_quality_examples_path=str(data),
        runtime_quality_checkpoint="dense_teacher_placeholder.pt",
        runtime_quality_interval=1,
        runtime_quality_best_checkpoint=str(runtime_best),
    )
    assert calls["count"] >= 2
    assert summary["best_selection"] == "runtime_quality"
    assert summary["selected_checkpoint"] == str(runtime_best)
    assert summary["best_checkpoint"] == str(runtime_best)
    assert summary["runtime_best_checkpoint"] == str(runtime_best)
    assert summary["edge_best_checkpoint"] == str(edge_best)
    assert runtime_best.exists()
    assert edge_best.exists()
    runtime_state = torch.load(runtime_best, map_location="cpu", weights_only=True)
    edge_state = torch.load(edge_best, map_location="cpu", weights_only=True)
    assert runtime_state["best_selection"] == "runtime_quality"
    assert edge_state["best_selection"] == "runtime_quality"


def test_runtime_quality_selection_score_route_dominates_kl_and_cos():
    better_route = {
        "route_acc": 0.9871,
        "generic_acc": 0.9112,
        "dense_agree": 0.9940,
        "hidden_cos": 0.9710,
        "logit_kl": 0.0147,
    }
    lower_route_better_proxy = {
        "route_acc": 0.9866,
        "generic_acc": 0.9073,
        "dense_agree": 0.9944,
        "hidden_cos": 0.9999,
        "logit_kl": 0.0001,
    }

    assert _runtime_quality_selection_score(better_route) > _runtime_quality_selection_score(lower_route_better_proxy)


def test_runtime_quality_selection_score_generic_breaks_route_tie():
    better_generic = {
        "route_acc": 0.9866,
        "generic_acc": 0.9112,
        "dense_agree": 0.9900,
        "hidden_cos": 0.9700,
        "logit_kl": 0.02,
    }
    worse_generic_better_proxy = {
        "route_acc": 0.9866,
        "generic_acc": 0.9073,
        "dense_agree": 0.9999,
        "hidden_cos": 0.9999,
        "logit_kl": 0.0001,
    }

    assert _runtime_quality_selection_score(better_generic) > _runtime_quality_selection_score(worse_generic_better_proxy)
