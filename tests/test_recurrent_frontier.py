"""Tests for recurrent_frontier.py — P1.8 recurrent frontier benchmark."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.world_graph import WorldGraph, ActiveGraph, NodeRecord, EdgeRecord, make_node_id, make_edge_id
from src.recurrent_frontier import (
    SCHEMA_VERSION,
    RecurrentFrontierConfig,
    RecurrentFrontierTrace,
    RecurrentFrontierComparison,
    j_score,
    run_recurrent_frontier,
    run_recurrent_frontier_comparison,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _ring_world(n: int = 8) -> tuple[WorldGraph, list[str]]:
    """Ring graph: 0->1->...->n-1->0."""
    g = WorldGraph()
    ids = []
    for i in range(n):
        nid = make_node_id(f"r{i}", "ring_node", "test")
        g.add_node(NodeRecord(node_id=nid, label=f"r{i}", node_kind="ring_node",
                              features={"arity": 2, "depth": 0}))
        ids.append(nid)
    for i in range(n):
        eid = make_edge_id(ids[i], ids[(i + 1) % n], "next")
        g.add_edge(EdgeRecord(edge_id=eid, src_id=ids[i], dst_id=ids[(i + 1) % n], relation="next"))
    return g, ids


def _make_config(**overrides) -> RecurrentFrontierConfig:
    cfg = RecurrentFrontierConfig(
        k=2,
        l=2,
        t_outer_values=[1, 2, 3],
        budget=6,
        h_horizon=2,
        planner="one_hop_topk",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── RecurrentFrontierConfig ───────────────────────────────────────────────────

def test_config_as_dict():
    cfg = _make_config()
    d = cfg.as_dict()
    assert d["k"] == 2
    assert d["t_outer_values"] == [1, 2, 3]
    assert "planner" in d


# ── run_recurrent_frontier ────────────────────────────────────────────────────

def test_run_single_step():
    g, ids = _ring_world(8)
    cfg = _make_config(t_outer_values=[1])
    trace = run_recurrent_frontier(g, ids[:2], cfg, t_outer=1)
    assert isinstance(trace, RecurrentFrontierTrace)
    assert trace.t_outer == 1
    assert len(trace.steps) == 1
    assert trace.wall_ms >= 0


def test_run_multiple_steps_grows_active():
    g, ids = _ring_world(8)
    cfg = _make_config(budget=8)
    trace1 = run_recurrent_frontier(g, ids[:2], cfg, t_outer=1)
    trace3 = run_recurrent_frontier(g, ids[:2], cfg, t_outer=3)
    # More steps generally reach more nodes (ring has many neighbors)
    assert trace3.total_frontier_expansions >= trace1.total_frontier_expansions


def test_run_with_model_fn():
    g, ids = _ring_world(8)
    call_count = {"n": 0}

    def mock_model(active, world):
        call_count["n"] += 1
        return np.zeros((len(active.node_ids), 8), dtype=np.float32)

    cfg = _make_config()
    trace = run_recurrent_frontier(g, ids[:2], cfg, t_outer=2, model_fn=mock_model)
    assert call_count["n"] == 2  # called once per step


def test_run_model_fn_exception_continues():
    """If model_fn raises, h_t falls back to None — expansion still proceeds."""
    g, ids = _ring_world(8)

    def bad_model(active, world):
        raise RuntimeError("oops")

    cfg = _make_config()
    trace = run_recurrent_frontier(g, ids[:2], cfg, t_outer=1, model_fn=bad_model)
    assert len(trace.steps) == 1  # didn't crash


def test_run_respects_budget():
    g, ids = _ring_world(8)
    budget = 4
    cfg = _make_config(budget=budget, k=10)
    trace = run_recurrent_frontier(g, ids[:2], cfg, t_outer=3)
    assert trace.final_active_node_count <= budget


def test_run_closure_planner():
    g, ids = _ring_world(8)
    cfg = _make_config(planner="closure_boolean", h_horizon=2)
    trace = run_recurrent_frontier(g, ids[:2], cfg, t_outer=2)
    assert trace.t_outer == 2
    # closure planner records path_count
    assert trace.total_closure_path_count >= 0


# ── j_score ───────────────────────────────────────────────────────────────────

def test_j_score_baseline():
    g, ids = _ring_world(4)
    cfg = _make_config()
    trace = run_recurrent_frontier(g, ids[:1], cfg, t_outer=1)
    j = j_score(trace)
    assert isinstance(j, float)


def test_j_score_penalizes_runtime():
    g, ids = _ring_world(4)
    cfg = _make_config()
    trace = run_recurrent_frontier(g, ids[:1], cfg, t_outer=1)

    j_cheap = j_score(trace, lambda_time=0.0)
    j_expensive = j_score(trace, lambda_time=100.0)
    assert j_cheap > j_expensive  # higher λ penalizes runtime


# ── run_recurrent_frontier_comparison ────────────────────────────────────────

def test_comparison_returns_both_planners():
    g, ids = _ring_world(8)
    cfg = _make_config(t_outer_values=[1, 2])
    comp = run_recurrent_frontier_comparison(g, ids[:2], cfg)
    assert isinstance(comp, RecurrentFrontierComparison)
    assert len(comp.onehop_traces) == len(cfg.t_outer_values)
    assert len(comp.closure_traces) == len(cfg.t_outer_values)


def test_comparison_summary_has_rows():
    g, ids = _ring_world(8)
    cfg = _make_config(t_outer_values=[1, 2, 3])
    comp = run_recurrent_frontier_comparison(g, ids[:2], cfg)
    assert "rows" in comp.summary
    assert len(comp.summary["rows"]) == 3


def test_comparison_acceptance_is_bool_or_none():
    g, ids = _ring_world(8)
    cfg = _make_config(t_outer_values=[1, 3])
    comp = run_recurrent_frontier_comparison(g, ids[:2], cfg)
    am = comp.acceptance_met()
    assert am in (True, False, None)


def test_comparison_format_table():
    g, ids = _ring_world(8)
    cfg = _make_config(t_outer_values=[1, 2])
    comp = run_recurrent_frontier_comparison(g, ids[:2], cfg)
    table = comp.format_table()
    assert "T_out" in table
    assert "acceptance" in table.lower()


def test_comparison_save_json(tmp_path):
    g, ids = _ring_world(8)
    cfg = _make_config(t_outer_values=[1, 2])
    comp = run_recurrent_frontier_comparison(g, ids[:2], cfg)
    out = tmp_path / "cmp.json"
    comp.save_json(out)
    data = json.loads(out.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert "summary" in data
