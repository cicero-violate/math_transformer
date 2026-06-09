from __future__ import annotations

from pathlib import Path

import torch

from src.synthetic_data import generate_hard_records, write_jsonl
from src.train_topology_scorer import train_topology_scorer


def test_train_topology_scorer_resumes_from_checkpoint(tmp_path: Path):
    data = tmp_path / "hard.jsonl"
    base_ckpt = tmp_path / "base.pt"
    finetuned_ckpt = tmp_path / "finetuned.pt"
    records = generate_hard_records(8, seed=79, split="train", route_fraction=1.0, max_depth=3)
    write_jsonl(data, records)

    train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(base_ckpt),
        max_steps=2,
        max_examples=4,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=1,
    )

    summary = train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(finetuned_ckpt),
        max_steps=1,
        max_examples=4,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=1,
        resume_scorer_checkpoint=str(base_ckpt),
    )

    assert finetuned_ckpt.exists()
    assert summary["checkpoint"] == str(finetuned_ckpt)
    state = torch.load(finetuned_ckpt, map_location="cpu", weights_only=True)
    assert state["resume_scorer_checkpoint"] == str(base_ckpt)
    assert state["model_state_dict"].keys() == torch.load(base_ckpt, map_location="cpu", weights_only=True)["model_state_dict"].keys()
