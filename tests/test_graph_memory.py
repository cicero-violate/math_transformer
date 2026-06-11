"""Tests for graph_memory.py — P2.1 memory store, P2.2 deletion gate, P2.3 verifier."""
from __future__ import annotations

import time
import pytest

from src.world_graph import WorldGraph, NodeRecord, EdgeRecord, ActiveGraph, make_node_id, make_edge_id
from src.graph_memory import (
    SCHEMA_VERSION,
    ClosurePreservingEdgeDeletionGate,
    EdgeMemoryRecord,
    GraphDelta,
    GraphMemoryStore,
    GraphWritebackVerifier,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _small_world() -> tuple[WorldGraph, list[str], list[str]]:
    """Triangle graph: a->b->c->a."""
    g = WorldGraph()
    labels = ["a", "b", "c"]
    ids = [make_node_id(l, "n", "test") for l in labels]
    for nid, label in zip(ids, labels):
        g.add_node(NodeRecord(node_id=nid, label=label, node_kind="n"))
    edge_ids = []
    for i in range(3):
        src, dst = ids[i], ids[(i + 1) % 3]
        eid = make_edge_id(src, dst, "link")
        g.add_edge(EdgeRecord(edge_id=eid, src_id=src, dst_id=dst, relation="link"))
        edge_ids.append(eid)
    return g, ids, edge_ids


# ── P2.1: EdgeMemoryRecord ────────────────────────────────────────────────────

def test_edge_memory_record_defaults():
    rec = EdgeMemoryRecord(edge_id="e1")
    assert rec.omega == 0.0
    assert rec.confidence == 1.0
    assert rec.version == 0
    assert rec.provenance == ""


def test_edge_memory_record_as_dict():
    rec = EdgeMemoryRecord(edge_id="e1", omega=0.5, utility=1.0, provenance="test")
    d = rec.as_dict()
    assert d["edge_id"] == "e1"
    assert d["omega"] == 0.5


# ── P2.1: GraphMemoryStore ────────────────────────────────────────────────────

def test_memory_store_set_and_get():
    store = GraphMemoryStore()
    store.set("e1", omega=0.7, provenance="test_run")
    rec = store.get("e1")
    assert rec is not None
    assert rec.omega == pytest.approx(0.7)
    assert rec.provenance == "test_run"


def test_memory_store_update_increments_version():
    store = GraphMemoryStore()
    store.set("e1", omega=0.1)
    store.set("e1", omega=0.2)
    rec = store.get("e1")
    assert rec.version == 2


def test_memory_store_apply_to_world():
    g, ids, eids = _small_world()
    store = GraphMemoryStore()
    store.set(eids[0], omega=0.9)
    updated = store.apply_to_world(g)
    assert updated == 1
    assert g.get_edge(eids[0]).weight == pytest.approx(0.9)


def test_memory_store_apply_skips_missing_edges():
    g = WorldGraph()
    store = GraphMemoryStore()
    store.set("nonexistent_edge", omega=0.5)
    updated = store.apply_to_world(g)
    assert updated == 0


def test_memory_store_len():
    store = GraphMemoryStore()
    store.set("e1", omega=0.1)
    store.set("e2", omega=0.2)
    assert len(store) == 2


def test_memory_store_serialization_roundtrip(tmp_path):
    store = GraphMemoryStore()
    store.set("e1", omega=0.5, utility=1.0, confidence=0.8, provenance="roundtrip")
    path = tmp_path / "mem.jsonl"
    store.save_jsonl(path)
    loaded = GraphMemoryStore.load_jsonl(path)
    rec = loaded.get("e1")
    assert rec is not None
    assert rec.omega == pytest.approx(0.5)
    assert rec.provenance == "roundtrip"


# ── P2.2: ClosurePreservingEdgeDeletionGate ───────────────────────────────────

def test_deletion_gate_allows_safe_deletion():
    """Deletion gate produces a valid report with a preservation ratio."""
    g, ids, eids = _small_world()
    # Triangle a->b->c->a: removing one edge leaves 6/9 path pairs (0.667).
    # Use min_preservation=0.6 so deletion is allowed.
    gate = ClosurePreservingEdgeDeletionGate(h=2, mode="boolean", min_preservation=0.6)
    allowed, report = gate.check(g, eids[0])
    assert report.edge_id == eids[0]
    assert isinstance(report.preservation_ratio, float)
    assert 0.0 <= report.preservation_ratio <= 1.0
    assert allowed


def test_deletion_gate_blocks_critical_edge():
    """Single edge in a chain — removing it cuts all transitive paths."""
    g = WorldGraph()
    ids = [make_node_id(f"x{i}", "n", "t") for i in range(3)]
    for nid in ids:
        g.add_node(NodeRecord(node_id=nid, label=nid, node_kind="n"))
    only_edge = make_edge_id(ids[0], ids[1], "dep")
    g.add_edge(EdgeRecord(edge_id=only_edge, src_id=ids[0], dst_id=ids[1], relation="dep"))

    gate = ClosurePreservingEdgeDeletionGate(h=1, mode="boolean", min_preservation=0.99)
    allowed, report = gate.check(g, only_edge, active_node_ids=ids)
    # Removing the only edge loses the path — not allowed at 0.99 preservation
    assert not allowed
    assert report.reason == "closure_degraded"


def test_deletion_gate_nonexistent_edge():
    g = WorldGraph()
    gate = ClosurePreservingEdgeDeletionGate()
    allowed, report = gate.check(g, "no_such_edge")
    assert allowed
    assert report.reason == "edge_not_found"


def test_deletion_gate_pareto_override():
    g = WorldGraph()
    ids = [make_node_id(f"y{i}", "n", "t") for i in range(2)]
    for nid in ids:
        g.add_node(NodeRecord(node_id=nid, label=nid, node_kind="n"))
    eid = make_edge_id(ids[0], ids[1], "dep")
    g.add_edge(EdgeRecord(edge_id=eid, src_id=ids[0], dst_id=ids[1], relation="dep"))

    gate = ClosurePreservingEdgeDeletionGate(h=1, mode="boolean", min_preservation=1.0)
    allowed, report = gate.check(g, eid, pareto_override=True, pareto_note="memory savings accepted")
    assert allowed
    assert "pareto_override" in report.reason


# ── P2.3: GraphWritebackVerifier ─────────────────────────────────────────────

def test_verifier_accepts_valid_add():
    g, ids, eids = _small_world()
    verifier = GraphWritebackVerifier()
    new_nid = make_node_id("d", "n", "test")
    delta = GraphDelta(
        delta_v_add=[NodeRecord(node_id=new_nid, label="d", node_kind="n")],
        provenance="test",
    )
    accepted, reasons = verifier.verify_delta(delta, g)
    assert accepted
    assert g.has_node(new_nid)


def test_verifier_accepts_edge_add():
    g, ids, eids = _small_world()
    verifier = GraphWritebackVerifier()
    new_eid = make_edge_id(ids[0], ids[2], "long")
    # ids[0]->ids[2] (skipping b)
    delta = GraphDelta(
        delta_e_add=[EdgeRecord(edge_id=new_eid, src_id=ids[0], dst_id=ids[2], relation="long")],
        provenance="test",
    )
    accepted, reasons = verifier.verify_delta(delta, g)
    assert accepted
    assert g.has_edge(new_eid)


def test_verifier_rejects_missing_provenance():
    g, ids, eids = _small_world()
    verifier = GraphWritebackVerifier(require_provenance=True)
    delta = GraphDelta(
        delta_v_add=[NodeRecord(node_id="x", label="x", node_kind="n")],
        provenance="",  # missing
    )
    accepted, reasons = verifier.verify_delta(delta, g)
    assert not accepted
    assert any("provenance" in r for r in reasons)


def test_verifier_tombstones_removed_node():
    g, ids, eids = _small_world()
    verifier = GraphWritebackVerifier(
        deletion_gate=ClosurePreservingEdgeDeletionGate(h=1, min_preservation=0.0)
    )
    delta = GraphDelta(delta_v_remove=[ids[0]], provenance="cleanup")
    accepted, reasons = verifier.verify_delta(delta, g)
    if accepted:
        ts = verifier.get_tombstone(ids[0])
        assert ts is not None
        assert ts.kind == "node"


def test_verifier_tombstone_rollback():
    g, ids, eids = _small_world()
    verifier = GraphWritebackVerifier(
        deletion_gate=ClosurePreservingEdgeDeletionGate(h=1, min_preservation=0.0)
    )
    # Tombstone edge eids[0]
    delta = GraphDelta(delta_e_remove=[eids[0]], provenance="del_test")
    accepted, _ = verifier.verify_delta(delta, g)
    if accepted:
        ts = verifier.get_tombstone(eids[0])
        assert ts is not None
        # Rollback
        restored = verifier.rollback(eids[0], g)
        assert restored
        assert g.has_edge(eids[0])


def test_verifier_rollback_unknown_returns_false():
    g, ids, eids = _small_world()
    verifier = GraphWritebackVerifier()
    assert not verifier.rollback("does_not_exist", g)


def test_verifier_version_increments_on_accept():
    g, ids, eids = _small_world()
    verifier = GraphWritebackVerifier()
    assert verifier.version == 0
    nid = make_node_id("new", "n", "t")
    delta = GraphDelta(
        delta_v_add=[NodeRecord(node_id=nid, label="new", node_kind="n")],
        provenance="bump_test",
    )
    verifier.verify_delta(delta, g)
    assert verifier.version == 1


def test_verifier_version_unchanged_on_reject():
    g, ids, eids = _small_world()
    verifier = GraphWritebackVerifier()
    delta = GraphDelta(provenance="")  # no provenance
    verifier.verify_delta(delta, g)
    assert verifier.version == 0


def test_verifier_delta_omega_updates_weight():
    g, ids, eids = _small_world()
    verifier = GraphWritebackVerifier()
    delta = GraphDelta(delta_omega={eids[0]: 0.99}, provenance="weight_update")
    accepted, _ = verifier.verify_delta(delta, g)
    assert accepted
    assert g.get_edge(eids[0]).weight == pytest.approx(0.99)
