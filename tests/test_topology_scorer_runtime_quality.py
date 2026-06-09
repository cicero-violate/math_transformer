from __future__ import annotations

from pathlib import Path

import torch

from src.model import MathRoutedTransformer
from src.synthetic_data import generate_hard_records, write_jsonl
from src.train_topology_scorer import train_topology_scorer


def test_train_topology_scorer_saves_runtime_best_checkpoint(tmp_path: Path):
    train_data = tmp_path / "train.jsonl"
    val_data = tmp_path / "val.jsonl"
    dense_ckpt = tmp_path / "dense.pt"
    ckpt = tmp_path / "scorer.pt"
    best = tmp_path / "scorer.best.pt"
    runtime_best = tmp_path / "scorer.runtime.pt"

    write_jsonl(train_data, generate_hard_records(8, seed=71, split="train", route_fraction=1.0, max_depth=3))
    write_jsonl(val_data, generate_hard_records(6, seed=73, split="val", route_fraction=1.0, max_depth=3))
    model = MathRoutedTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0, attention_mode="full")
    torch.save({"model_state_dict": model.state_dict()}, dense_ckpt)

    summary = train_topology_scorer(
        examples_path=str(train_data),
        save_checkpoint=str(ckpt),
        max_steps=2,
        max_examples=4,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=1,
        val_examples_path=str(val_data),
        eval_interval=1,
        eval_max_examples=3,
        best_checkpoint=str(best),
        dense_checkpoint=str(dense_ckpt),
        dense_mix=0.1,
        runtime_quality_examples_path=str(val_data),
        runtime_quality_checkpoint=str(dense_ckpt),
        runtime_quality_interval=1,
        runtime_quality_max_examples=3,
        runtime_quality_best_checkpoint=str(runtime_best),
    )

    assert runtime_best.exists()
    assert summary["runtime_best_checkpoint"] == str(runtime_best)
    assert float(summary["runtime_best_score"]) > 0.0
    state = torch.load(runtime_best, map_location="cpu", weights_only=True)
    assert state["teacher_signal"] == "dense_qk_blend"
