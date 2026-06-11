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


def test_guarded_selector_protects_noncommutative_argument_edges():
    nodes = normalize(parse("div(x, y)")).collect_nodes()
    scores = torch.zeros(len(nodes), len(nodes))
    builder = LearnedTopologyBuilder(
        "unused.pt",
        fixed_k=1,
        device="cpu",
        protect_noncommutative=True,
    )

    mask = builder._mask_from_scores(nodes, scores, torch.device("cpu"))

    root_idx = next(i for i, node in enumerate(nodes) if node.op == "div")
    child_indices = [i for i, node in enumerate(nodes) if node.op == "var"]
    for child_idx in child_indices:
        assert mask[root_idx, child_idx]
        assert mask[child_idx, root_idx]
    assert torch.all(torch.diag(mask))


def test_polarity_bias_changes_extra_edge_selection(tmp_path: Path):
    nodes = normalize(parse("add(mul(x, y), z)")).collect_nodes()
    n = len(nodes)
    scores = torch.zeros(n, n)
    summary = {
        "edge_kind_polarity": [
            {"pattern": "extra_edges:mul->leaf", "record_polarity": -1.0},
            {"pattern": "extra_edges:add->leaf", "record_polarity": 1.0},
        ]
    }
    polarity_path = tmp_path / "polarity.json"
    polarity_path.write_text(__import__("json").dumps(summary))
    builder = LearnedTopologyBuilder(
        "unused.pt",
        fixed_k=2,
        device="cpu",
        polarity_summary=str(polarity_path),
        polarity_alpha=2.0,
    )

    mask = builder._mask_from_scores(nodes, scores, torch.device("cpu"))

    add_idx = next(i for i, node in enumerate(nodes) if node.op == "add")
    mul_idx = next(i for i, node in enumerate(nodes) if node.op == "mul")
    leaf_indices = [i for i, node in enumerate(nodes) if node.op == "var"]
    assert any(bool(mask[add_idx, j]) for j in leaf_indices)
    assert not any(bool(mask[mul_idx, j]) for j in leaf_indices)
