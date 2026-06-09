from __future__ import annotations

from pathlib import Path

import torch

from src.learned_topology_runtime import LearnedTopologyBuilder
from src.normalize import normalize
from src.parser import parse
from src.synthetic_data import generate_hard_records, write_jsonl
from src.train_topology_scorer import train_topology_scorer


def test_learned_topology_builder_outputs_capped_mask_and_priority(tmp_path: Path):
    data = tmp_path / "hard.jsonl"
    ckpt = tmp_path / "scorer.pt"
    records = generate_hard_records(8, seed=53, split="train", route_fraction=1.0, max_depth=3)
    write_jsonl(data, records)
    train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(ckpt),
        max_steps=3,
        max_examples=4,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=2,
    )

    nodes = normalize(parse(records[0]["normalized"])).collect_nodes()
    builder = LearnedTopologyBuilder(str(ckpt), fixed_k=3, device="cpu")
    mask, diag = builder.build_scored_topk(nodes, env=records[0].get("env") or None)
    priority = builder.priority_from_mask(mask)

    assert mask.shape == (len(nodes), len(nodes))
    assert mask.dtype == bool
    assert int(mask.sum(axis=1).max()) <= 3
    assert mask.diagonal().all()
    assert priority.shape == mask.shape
    assert priority.diagonal().min() == 1
    assert diag.by_relation["learned_topology"] == int(mask.sum())


def test_learned_topology_builder_torch_outputs_capped_mask(tmp_path: Path):
    data = tmp_path / "hard.jsonl"
    ckpt = tmp_path / "scorer.pt"
    records = generate_hard_records(6, seed=59, split="train", route_fraction=1.0, max_depth=3)
    write_jsonl(data, records)
    train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(ckpt),
        max_steps=2,
        max_examples=3,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=1,
    )

    nodes = normalize(parse(records[0]["normalized"])).collect_nodes()
    builder = LearnedTopologyBuilder(str(ckpt), fixed_k=3, device="cpu")
    mask, _diag = builder.build_scored_topk_torch(nodes, None, records[0].get("env") or None, torch.device("cpu"))
    priority = builder.priority_from_mask_torch(mask)

    assert mask.dtype == torch.bool
    assert int(mask.sum(dim=1).max().item()) <= 3
    assert torch.all(torch.diag(mask))
    assert priority.dtype == torch.int8
