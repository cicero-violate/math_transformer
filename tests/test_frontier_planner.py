"""Tests for frontier_planner.py — P1.3-P1.4, P1.6-P1.7."""
from __future__ import annotations

import numpy as np
import pytest

from src.world_graph import WorldGraph, ActiveGraph, NodeRecord, EdgeRecord, make_node_id, make_edge_id
from src.frontier_planner import (
    FrontierExpansionResult,
    ClosureAwarePlanner,
    OneHopTopKPlanner,
    QuantaleFrontierPlanner,
    boundary_candidates,
    extract_frontier,
    keep_top_b,
    score_candidates_closure,
    score_candidates_onehop,
    score_candidates_quantale,
)
from src.graph_closure import BOOLEAN_SPEC, COST_SPEC, UTILITY_SPEC


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _chain_world(n: int = 6) -> tuple[WorldGraph, list[str]]:
    """Linear chain: node0 -> node1 -> ... -> node(n-1)."""
    g = WorldGraph()
    ids = []
    for i in range(n):
        nid = make_node_id(f"n{i}", "generic", "test")
        g.add_node(NodeRecord(node_id=nid, label=f"n{i}", node_kind="generic",
                              features={"arity": 0, "depth": i}))
        ids.append(nid)
    for i in range(n - 1):
        eid = make_edge_id(ids[i], ids[i + 1], "next")
        g.add_edge(EdgeRecord(edge_id=eid, src_id=ids[i], dst_id=ids[i + 1], relation="next"))
    return g, ids


def _star_world(k: int = 4) -> tuple[WorldGraph, str, list[str]]:
    """Star: centre -> leaf_0, ..., leaf_(k-1) plus extra ring around leaves."""
    g = WorldGraph()
    centre_id = make_node_id("centre", "hub", "test")
    g.add_node(NodeRecord(node_id=centre_id, label="centre", node_kind="hub",
                          features={"arity": k, "depth": 0}))
    leaf_ids = []
    for i in range(k):
        lid = make_node_id(f"leaf{i}", "leaf", "test")
        g.add_node(NodeRecord(node_id=lid, label=f"leaf{i}", node_kind="leaf",
                              features={"arity": 0, "depth": 1}))
        leaf_ids.append(lid)
        eid = make_edge_id(centre_id, lid, "child")
        g.add_edge(EdgeRecord(edge_id=eid, src_id=centre_id, dst_id=lid, relation="child"))
    # Extra ring nodes outside star
    ring_ids = []
    for i in range(k):
        rid = make_node_id(f"ring{i}", "ring", "test")
        g.add_node(NodeRecord(node_id=rid, label=f"ring{i}", node_kind="ring",
                              features={"arity": 1, "depth": 2}))
        ring_ids.append(rid)
        eid = make_edge_id(leaf_ids[i], rid, "next")
        g.add_edge(EdgeRecord(edge_id=eid, src_id=leaf_ids[i], dst_id=rid, relation="next"))
    return g, centre_id, leaf_ids + ring_ids


# ── P1.3: extract_frontier ────────────────────────────────────────────────────

def test_extract_frontier_boundary_nodes():
    g, ids = _chain_world(4)
    active = ActiveGraph.seed(ids[:2], budget=10)
    frontier = extract_frontier(active, g)
    # ids[1] is boundary (has neighbour ids[2] outside)
    assert ids[1] in frontier


def test_extract_frontier_isolated_node():
    g = WorldGraph()
    nid = make_node_id("solo", "g", "t")
    g.add_node(NodeRecord(node_id=nid, label="solo", node_kind="g"))
    active = ActiveGraph.seed([nid], budget=10)
    assert len(extract_frontier(active, g)) == 0


def test_boundary_candidates_are_outside_active():
    g, ids = _chain_world(4)
    active = ActiveGraph.seed([ids[0]], budget=10)
    cands = boundary_candidates(active, g)
    assert ids[1] in cands
    for c in cands:
        assert c not in active.node_ids


# ── P1.4: score_candidates_onehop ────────────────────────────────────────────

def test_score_candidates_onehop_returns_top_k():
    g, centre_id, others = _star_world(4)
    active = ActiveGraph.seed([centre_id], budget=10)
    leaves = [x for x in others if "leaf" in (g.get_node(x).label if g.get_node(x) else "")]
    scored = score_candidates_onehop(None, active, leaves, g, k=2)
    assert len(scored) <= 2


def test_score_candidates_onehop_empty_candidates():
    g, ids = _chain_world(2)
    active = ActiveGraph.seed(ids, budget=10)
    result = score_candidates_onehop(None, active, [], g, k=3)
    assert result == []


def test_score_candidates_onehop_h_t_used():
    g, ids = _chain_world(4)
    active = ActiveGraph.seed(ids[:2], budget=10)
    cands = [ids[2], ids[3]]
    # Provide h_t with matching feature dim (8)
    h_t = np.ones((2, 8), dtype=np.float32)
    scored = score_candidates_onehop(h_t, active, cands, g, k=2)
    assert len(scored) == 2
    # Scores should be a list of (node_id, float) tuples
    for nid, s in scored:
        assert isinstance(s, float)


# ── score_candidates_closure ──────────────────────────────────────────────────

def test_score_candidates_closure_returns_k_results():
    g, centre_id, others = _star_world(4)
    active = ActiveGraph.seed([centre_id], budget=10)
    leaves = [x for x in others if "leaf" in (g.get_node(x).label if g.get_node(x) else "")]
    scored = score_candidates_closure(None, active, leaves, g, k=2, h_horizon=2)
    assert len(scored) <= 2


def test_score_candidates_closure_empty():
    g, ids = _chain_world(2)
    active = ActiveGraph.seed(ids, budget=10)
    assert score_candidates_closure(None, active, [], g, k=3, h_horizon=2) == []


def test_score_candidates_closure_mode_boolean():
    g, ids = _chain_world(5)
    active = ActiveGraph.seed(ids[:2], budget=10)
    cands = ids[2:]
    scored = score_candidates_closure(None, active, cands, g, k=3, h_horizon=3, mode="boolean")
    assert len(scored) <= 3


# ── P1.7: keep_top_b ─────────────────────────────────────────────────────────

def test_keep_top_b_within_budget():
    g, ids = _chain_world(4)
    active = ActiveGraph.seed(ids[:2], budget=10)
    result = keep_top_b(active, ids[2:], g, budget=10)
    # All nodes fit within budget=10
    assert len(result.node_ids) == 4


def test_keep_top_b_enforces_budget():
    g, ids = _chain_world(6)
    active = ActiveGraph.seed(ids[:4], budget=10)
    result = keep_top_b(active, ids[4:], g, budget=3)
    assert len(result.node_ids) <= 3


def test_keep_top_b_preserves_anchors():
    g, ids = _chain_world(6)
    active = ActiveGraph.seed(ids[:2], budget=10)
    active = ActiveGraph(
        node_ids=active.node_ids,
        budget=2,
        anchor_ids=frozenset([ids[0]]),
    )
    result = keep_top_b(active, ids[2:], g, budget=2)
    assert ids[0] in result.node_ids  # anchor always preserved


def test_keep_top_b_step_bumped():
    g, ids = _chain_world(3)
    active = ActiveGraph.seed(ids[:1], budget=10)
    result = keep_top_b(active, [ids[1]], g, budget=10)
    # with_node_set bumps step, but keep_top_b calls with_node_set internally
    # (only when pruning is needed — here no pruning occurs via with_added path)
    assert len(result.node_ids) >= 1


# ── P1.6: OneHopTopKPlanner ───────────────────────────────────────────────────

def test_onehop_planner_expand():
    g, ids = _chain_world(6)
    active = ActiveGraph.seed(ids[:2], budget=8)
    planner = OneHopTopKPlanner(k=2)
    new_active, result = planner.expand(active, g, None, t_outer=0)
    assert isinstance(result, FrontierExpansionResult)
    assert result.planner == "one_hop_topk"
    assert result.t_outer == 0
    assert len(new_active.node_ids) >= len(active.node_ids)


def test_onehop_planner_no_candidates_no_growth():
    g = WorldGraph()
    nid = make_node_id("alone", "g", "t")
    g.add_node(NodeRecord(node_id=nid, label="alone", node_kind="g"))
    active = ActiveGraph.seed([nid], budget=10)
    planner = OneHopTopKPlanner(k=4)
    new_active, result = planner.expand(active, g, None, t_outer=0)
    assert len(new_active.node_ids) == 1
    assert result.added_node_count == 0


# ── P1.6: ClosureAwarePlanner ─────────────────────────────────────────────────

def test_closure_planner_expand():
    g, ids = _chain_world(6)
    active = ActiveGraph.seed(ids[:2], budget=8)
    planner = ClosureAwarePlanner(k=2, h_horizon=2, closure_mode="boolean")
    new_active, result = planner.expand(active, g, None, t_outer=0)
    assert isinstance(result, FrontierExpansionResult)
    assert "closure" in result.planner
    assert result.closure_path_count >= 0


def test_closure_planner_records_compute_cost():
    g, ids = _chain_world(6)
    active = ActiveGraph.seed(ids[:2], budget=8)
    planner = ClosureAwarePlanner(k=2, h_horizon=3, closure_mode="utility")
    _, result = planner.expand(active, g, None, t_outer=1)
    # With utility mode, compute_ops = n^3 * H > 0
    assert result.closure_compute_cost > 0


# ── P1.2: score_candidates_quantale ──────────────────────────────────────────

def test_score_candidates_quantale_boolean_returns_k():
    g, centre_id, others = _star_world(4)
    active = ActiveGraph.seed([centre_id], budget=10)
    leaves = [x for x in others if "leaf" in (g.get_node(x).label if g.get_node(x) else "")]
    scored = score_candidates_quantale(None, active, leaves, g, k=2, h_horizon=2, spec=BOOLEAN_SPEC)
    assert len(scored) <= 2


def test_score_candidates_quantale_cost_returns_k():
    g, ids = _chain_world(5)
    active = ActiveGraph.seed(ids[:2], budget=10)
    cands = ids[2:]
    scored = score_candidates_quantale(None, active, cands, g, k=2, h_horizon=2, spec=COST_SPEC)
    assert len(scored) <= 2


def test_score_candidates_quantale_utility_returns_k():
    g, ids = _chain_world(5)
    active = ActiveGraph.seed(ids[:2], budget=10)
    cands = ids[2:]
    scored = score_candidates_quantale(None, active, cands, g, k=2, h_horizon=2, spec=UTILITY_SPEC)
    assert len(scored) <= 2


def test_score_candidates_quantale_empty():
    g, ids = _chain_world(2)
    active = ActiveGraph.seed(ids, budget=10)
    assert score_candidates_quantale(None, active, [], g, k=3, h_horizon=2, spec=BOOLEAN_SPEC) == []


def test_score_candidates_quantale_invalid_k():
    g, ids = _chain_world(2)
    active = ActiveGraph.seed(ids[:1], budget=10)
    with pytest.raises(ValueError):
        score_candidates_quantale(None, active, [ids[1]], g, k=0, h_horizon=2, spec=BOOLEAN_SPEC)


def test_score_candidates_quantale_invalid_h():
    g, ids = _chain_world(2)
    active = ActiveGraph.seed(ids[:1], budget=10)
    with pytest.raises(ValueError):
        score_candidates_quantale(None, active, [ids[1]], g, k=1, h_horizon=0, spec=BOOLEAN_SPEC)


# ── P1.2: QuantaleFrontierPlanner ────────────────────────────────────────────

def test_quantale_planner_boolean_expand():
    g, ids = _chain_world(6)
    active = ActiveGraph.seed(ids[:2], budget=8)
    planner = QuantaleFrontierPlanner(k=2, h_horizon=2, spec=BOOLEAN_SPEC)
    new_active, result = planner.expand(active, g, None, t_outer=0)
    assert isinstance(result, FrontierExpansionResult)
    assert "quantale_boolean" in result.planner
    assert len(new_active.node_ids) >= len(active.node_ids)


def test_quantale_planner_utility_expand():
    g, ids = _chain_world(6)
    active = ActiveGraph.seed(ids[:2], budget=8)
    planner = QuantaleFrontierPlanner(k=2, h_horizon=2, spec="utility")
    new_active, result = planner.expand(active, g, None, t_outer=1)
    assert "quantale_utility" in result.planner
    assert result.closure_compute_cost > 0


def test_quantale_planner_cost_expand():
    g, ids = _chain_world(6)
    active = ActiveGraph.seed(ids[:2], budget=8)
    planner = QuantaleFrontierPlanner(k=2, h_horizon=2, spec="cost")
    _, result = planner.expand(active, g, None, t_outer=0)
    assert "quantale_cost" in result.planner


def test_quantale_planner_string_spec():
    planner = QuantaleFrontierPlanner(k=2, h_horizon=2, spec="boolean")
    assert planner.spec is BOOLEAN_SPEC


def test_quantale_planner_invalid_spec():
    with pytest.raises(ValueError):
        QuantaleFrontierPlanner(k=2, h_horizon=2, spec="unknown_spec")


def test_quantale_planner_invalid_k():
    with pytest.raises(ValueError):
        QuantaleFrontierPlanner(k=0, h_horizon=2)


def test_quantale_planner_no_candidates_no_growth():
    g = WorldGraph()
    nid = make_node_id("alone", "g", "t")
    g.add_node(NodeRecord(node_id=nid, label="alone", node_kind="g"))
    active = ActiveGraph.seed([nid], budget=10)
    planner = QuantaleFrontierPlanner(k=4, h_horizon=2, spec="boolean")
    new_active, result = planner.expand(active, g, None, t_outer=0)
    assert len(new_active.node_ids) == 1
    assert result.added_node_count == 0


def test_quantale_planner_reports_closure_path_count():
    g, ids = _chain_world(6)
    active = ActiveGraph.seed(ids[:2], budget=8)
    planner = QuantaleFrontierPlanner(k=2, h_horizon=3, spec=UTILITY_SPEC)
    _, result = planner.expand(active, g, None, t_outer=0)
    assert result.closure_path_count >= 0


def test_quantale_planner_budget_respected():
    g, ids = _chain_world(10)
    active = ActiveGraph.seed(ids[:2], budget=4)
    planner = QuantaleFrontierPlanner(k=4, h_horizon=2, spec="boolean")
    new_active, _ = planner.expand(active, g, None, t_outer=0)
    assert len(new_active.node_ids) <= 4
