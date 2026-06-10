from __future__ import annotations

import json
from pathlib import Path

import torch

from src.normalize import normalize
from src.parser import parse
from src.topology_trace import (
    TopologyTraceWriter,
    hash_nodes,
    summarize_mask,
    summarize_overlap,
    summarize_scores,
)


def _nodes(expr: str):
    return normalize(parse(expr)).collect_nodes()


def test_topology_trace_summaries_and_writer(tmp_path: Path):
    scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pred = torch.tensor([[True, False], [True, True]])
    target = torch.tensor([[True, True], [False, True]])

    score_summary = summarize_scores(scores)
    assert score_summary["numel"] == 4
    assert score_summary["min"] == 1.0
    assert score_summary["max"] == 4.0

    mask_summary = summarize_mask(pred)
    assert mask_summary["n"] == 2
    assert mask_summary["active_edges"] == 3
    assert mask_summary["self_edges"] == 2

    overlap = summarize_overlap(pred, target)
    assert overlap["edge_hits"] == 2
    assert overlap["missing_edges"] == 1
    assert overlap["extra_edges"] == 1

    path = tmp_path / "trace.jsonl"
    with TopologyTraceWriter(path) as writer:
        writer.write({
            "sample_id": 0,
            "nodes_hash": hash_nodes(_nodes("add(x, y)")),
            "scores": score_summary,
            "pred_topology": mask_summary,
            "overlap": overlap,
        })

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["sample_id"] == 0
    assert len(rows[0]["nodes_hash"]) == 16
