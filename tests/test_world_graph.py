"""Tests for world_graph.py — P1.1 graph identity schema, P1.2 world/active graph split."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.world_graph import (
    ActiveGraph,
    EdgeRecord,
    NodeRecord,
    WorldGraph,
    make_edge_id,
    make_node_id,
)
from src.ir import var, add, matmul


# ── Hash ID helpers ───────────────────────────────────────────────────────────

def test_make_node_id_is_deterministic():
    a = make_node_id("foo", "add", "prov")
    b = make_node_id("foo", "add", "prov")
    assert a == b


def test_make_node_id_differs_by_label():
    a = make_node_id("foo", "add")
    b = make_node_id("bar", "add")
    assert a != b


def test_make_edge_id_is_deterministic():
    a = make_edge_id("n1", "n2", "dep")
    b = make_edge_id("n1", "n2", "dep")
    assert a == b


def test_make_edge_id_differs_by_relation():
    a = make_edge_id("n1", "n2", "dep")
    b = make_edge_id("n1", "n2", "comp")
    assert a != b


# ── WorldGraph ────────────────────────────────────────────────────────────────

def _simple_world():
    g = WorldGraph()
    nid_a = make_node_id("A", "add", "test")
    nid_b = make_node_id("B", "var", "test")
    g.add_node(NodeRecord(node_id=nid_a, label="A", node_kind="add", provenance="test"))
    g.add_node(NodeRecord(node_id=nid_b, label="B", node_kind="var", provenance="test"))
    eid = make_edge_id(nid_a, nid_b, "symbolic_dependency")
    g.add_edge(EdgeRecord(edge_id=eid, src_id=nid_a, dst_id=nid_b, relation="symbolic_dependency"))
    return g, nid_a, nid_b, eid


def test_world_graph_add_and_query():
    g, nid_a, nid_b, eid = _simple_world()
    assert g.node_count() == 2
    assert g.edge_count() == 1
    assert g.has_node(nid_a)
    assert g.has_edge(eid)
    assert g.get_node(nid_a).label == "A"
    assert g.get_edge(eid).relation == "symbolic_dependency"


def test_world_graph_neighbors():
    g, nid_a, nid_b, eid = _simple_world()
    nb_a = g.neighbors(nid_a)
    nb_b = g.neighbors(nid_b)
    assert nid_b in nb_a
    assert nid_a in nb_b


def test_world_graph_add_node_idempotent():
    g = WorldGraph()
    nid = make_node_id("X", "mul", "test")
    rec = NodeRecord(node_id=nid, label="X", node_kind="mul")
    g.add_node(rec)
    g.add_node(rec)  # second add is a no-op
    assert g.node_count() == 1


def test_world_graph_add_edge_missing_node_raises():
    g = WorldGraph()
    with pytest.raises(ValueError, match="missing nodes"):
        g.add_edge(EdgeRecord(
            edge_id="e1", src_id="nonexistent1", dst_id="nonexistent2",
            relation="dep",
        ))


def test_world_graph_update_edge_weight():
    g, nid_a, nid_b, eid = _simple_world()
    g.update_edge_weight(eid, 0.75)
    assert g.get_edge(eid).weight == pytest.approx(0.75)
    assert g.get_edge(eid).version == 1


def test_world_graph_edges_between():
    g, nid_a, nid_b, eid = _simple_world()
    edges = g.edges_between(nid_a, nid_b)
    assert len(edges) == 1
    assert edges[0].edge_id == eid


def test_world_graph_serialization_roundtrip(tmp_path):
    g, nid_a, nid_b, eid = _simple_world()
    p = tmp_path / "world.jsonl"
    g.save_jsonl(p)
    g2 = WorldGraph.load_jsonl(p)
    assert g2.node_count() == g.node_count()
    assert g2.edge_count() == g.edge_count()
    assert g2.get_node(nid_a).label == "A"
    assert g2.get_edge(eid).relation == "symbolic_dependency"


# ── from_math_nodes factory ───────────────────────────────────────────────────

def test_from_math_nodes_creates_nodes():
    x = var("x")
    y = var("y")
    expr = add(x, y)
    nodes = expr.collect_nodes()
    world, ids = WorldGraph.from_math_nodes(nodes)
    assert world.node_count() == len(nodes)
    assert len(ids) == len(nodes)


def test_from_math_nodes_creates_dependency_edges():
    x = var("x")
    y = var("y")
    expr = add(x, y)
    nodes = expr.collect_nodes()
    world, ids = WorldGraph.from_math_nodes(nodes)
    # add node should have edges to x and y
    add_id = ids[0]
    nb = world.neighbors(add_id)
    assert len(nb) >= 2


def test_from_math_nodes_provenance_in_features():
    nodes = var("z").collect_nodes()
    world, ids = WorldGraph.from_math_nodes(nodes, provenance="unit_test")
    rec = world.get_node(ids[0])
    assert rec.provenance == "unit_test"
    assert rec.node_kind == "var"


# ── ActiveGraph ───────────────────────────────────────────────────────────────

def _world_with_chain(n: int = 5):
    """Build a linear chain: n0 -> n1 -> ... -> n(n-1)."""
    g = WorldGraph()
    ids = []
    for i in range(n):
        nid = make_node_id(f"node{i}", "generic", "test")
        g.add_node(NodeRecord(node_id=nid, label=f"node{i}", node_kind="generic"))
        ids.append(nid)
    for i in range(n - 1):
        eid = make_edge_id(ids[i], ids[i + 1], "next")
        g.add_edge(EdgeRecord(edge_id=eid, src_id=ids[i], dst_id=ids[i + 1], relation="next"))
    return g, ids


def test_active_graph_seed():
    g, ids = _world_with_chain()
    active = ActiveGraph.seed(ids[:2], budget=10)
    assert ids[0] in active.node_ids
    assert ids[1] in active.node_ids
    assert ids[2] not in active.node_ids


def test_active_graph_frontier():
    g, ids = _world_with_chain()
    # Seed with first node only — it has a neighbor outside active
    active = ActiveGraph.seed([ids[0]], budget=10)
    frontier = active.frontier(g)
    assert ids[0] in frontier  # ids[0] -> ids[1] which is outside


def test_active_graph_boundary_candidates():
    g, ids = _world_with_chain()
    active = ActiveGraph.seed([ids[0]], budget=10)
    candidates = active.boundary_candidates(g)
    assert ids[1] in candidates


def test_active_graph_no_frontier_when_isolated():
    g = WorldGraph()
    nid = make_node_id("lone", "generic", "test")
    g.add_node(NodeRecord(node_id=nid, label="lone", node_kind="generic"))
    active = ActiveGraph.seed([nid], budget=10)
    assert len(active.frontier(g)) == 0
    assert len(active.boundary_candidates(g)) == 0


def test_active_graph_active_edges():
    g, ids = _world_with_chain(3)
    active = ActiveGraph.seed(ids[:2], budget=10)
    edges = active.active_edges(g)
    assert len(edges) == 1
    assert edges[0].src_id == ids[0]
    assert edges[0].dst_id == ids[1]


def test_active_graph_with_added():
    g, ids = _world_with_chain()
    active = ActiveGraph.seed([ids[0]], budget=10)
    active2 = active.with_added([ids[1]])
    assert ids[1] in active2.node_ids
    assert active2.step == 0  # with_added doesn't bump step


def test_active_graph_with_node_set_bumps_step():
    g, ids = _world_with_chain()
    active = ActiveGraph.seed([ids[0]], budget=10)
    active2 = active.with_node_set(frozenset([ids[0], ids[1]]))
    assert active2.step == 1


def test_active_graph_budget_check():
    g, ids = _world_with_chain(3)
    active = ActiveGraph.seed(ids, budget=2)
    assert active.is_at_budget()
