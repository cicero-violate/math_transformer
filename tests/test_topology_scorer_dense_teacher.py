from __future__ import annotations

from pathlib import Path

import torch

from src.model import MathRoutedTransformer
from src.synthetic_data import generate_hard_records, write_jsonl
from src.train_topology_scorer import train_topology_scorer


def test_train_topology_scorer_dense_teacher_smoke(tmp_path: Path):
    data = tmp_path / "hard.jsonl"
    dense_ckpt = tmp_path / "dense.pt"
    scorer_ckpt = tmp_path / "scorer.pt"
    records = generate_hard_records(8, seed=67, split="train", route_fraction=1.0, max_depth=3)
    write_jsonl(data, records)
    model = MathRoutedTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0, attention_mode="full")
    torch.save({"model_state_dict": model.state_dict()}, dense_ckpt)

    summary = train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(scorer_ckpt),
        max_steps=2,
        max_examples=4,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=1,
        dense_checkpoint=str(dense_ckpt),
        dense_mix=0.25,
    )

    assert scorer_ckpt.exists()
    assert summary["examples"] == 4
    state = torch.load(scorer_ckpt, map_location="cpu", weights_only=True)
    assert state["teacher_signal"] == "dense_qk_blend"
    assert state["dense_mix"] == 0.25
