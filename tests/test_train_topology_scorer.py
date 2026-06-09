from __future__ import annotations

from pathlib import Path

import torch

from src.synthetic_data import generate_hard_records, write_jsonl
from src.train_topology_scorer import _compress_target, train_topology_scorer


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
