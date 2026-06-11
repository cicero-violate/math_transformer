"""Tests for graph_closure.py — P1.5 bounded Kleene closure API."""
from __future__ import annotations

import numpy as np
import pytest

from src.graph_closure import (
    BOOLEAN_SPEC,
    COST_SPEC,
    UTILITY_SPEC,
    ClosureResult,
    QuantaleSpec,
    _SPEC_MAP,
    boolean_closure,
    bounded_closure,
    closure_critical_edges,
    cost_closure,
    quantale_bounded_closure,
    quantale_closure,
    utility_closure,
)

_LARGE = 1e9


# ── boolean_closure ───────────────────────────────────────────────────────────

def test_boolean_closure_identity_h0():
    A = np.array([[False, True], [False, False]], dtype=bool)
    # h=1 should reach direct neighbors
    R = boolean_closure(A, h=1)
    # i=0 can reach j=1
    assert R[0, 1]
    # i=1 cannot reach j=0
    assert not R[1, 0]


def test_boolean_closure_transitive():
    # Chain: 0->1->2
    A = np.zeros((3, 3), dtype=bool)
    A[0, 1] = True
    A[1, 2] = True
    R = boolean_closure(A, h=2)
    assert R[0, 2]  # 0 can reach 2 via 1 in 2 hops


def test_boolean_closure_horizon_limits_reach():
    # 0->1->2->3
    A = np.zeros((4, 4), dtype=bool)
    A[0, 1] = A[1, 2] = A[2, 3] = True
    R1 = boolean_closure(A, h=1)
    R2 = boolean_closure(A, h=2)
    R3 = boolean_closure(A, h=3)
    assert not R1[0, 2]
    assert R2[0, 2]
    assert not R2[0, 3]
    assert R3[0, 3]


def test_boolean_closure_self_loops():
    A = np.array([[False, True], [False, False]], dtype=bool)
    R = boolean_closure(A, h=1)
    # Diagonal (self) always reachable via I
    assert R[0, 0]
    assert R[1, 1]


# ── cost_closure ──────────────────────────────────────────────────────────────

def test_cost_closure_direct_edge():
    A = np.full((3, 3), _LARGE)
    A[0, 1] = 2.0
    A[1, 2] = 3.0
    R = cost_closure(A, h=1)
    assert R[0, 1] == pytest.approx(2.0)
    assert R[0, 2] == pytest.approx(_LARGE)  # not reachable in 1 hop


def test_cost_closure_two_hop():
    A = np.full((3, 3), _LARGE)
    A[0, 1] = 2.0
    A[1, 2] = 3.0
    R = cost_closure(A, h=2)
    assert R[0, 2] == pytest.approx(5.0)


def test_cost_closure_shortest_path():
    # Two paths from 0 to 2: direct cost 10, or via 1 cost 2+3=5
    A = np.full((3, 3), _LARGE)
    A[0, 2] = 10.0
    A[0, 1] = 2.0
    A[1, 2] = 3.0
    R = cost_closure(A, h=2)
    assert R[0, 2] == pytest.approx(5.0)  # cheaper two-hop path wins


def test_cost_closure_self_is_zero():
    A = np.full((3, 3), _LARGE)
    A[0, 1] = 1.0
    R = cost_closure(A, h=1)
    assert R[0, 0] == pytest.approx(0.0)
    assert R[1, 1] == pytest.approx(0.0)


# ── utility_closure ───────────────────────────────────────────────────────────

def test_utility_closure_direct():
    NEG = -_LARGE
    A = np.full((3, 3), NEG)
    A[0, 1] = 5.0
    A[1, 2] = 3.0
    R = utility_closure(A, h=1)
    assert R[0, 1] == pytest.approx(5.0)
    assert R[0, 2] == pytest.approx(NEG)  # 2 hops needed


def test_utility_closure_two_hop():
    NEG = -_LARGE
    A = np.full((3, 3), NEG)
    A[0, 1] = 5.0
    A[1, 2] = 3.0
    R = utility_closure(A, h=2)
    assert R[0, 2] == pytest.approx(8.0)  # 5 + 3


def test_utility_closure_max_path():
    # Direct from 0->2 costs 4; via 1 costs 5+3=8
    NEG = -_LARGE
    A = np.full((3, 3), NEG)
    A[0, 2] = 4.0
    A[0, 1] = 5.0
    A[1, 2] = 3.0
    R = utility_closure(A, h=2)
    assert R[0, 2] == pytest.approx(8.0)  # highest-utility path via 1


# ── bounded_closure API ───────────────────────────────────────────────────────

def test_bounded_closure_boolean():
    node_ids = ["a", "b", "c"]
    edges = [("a", "b", 1.0), ("b", "c", 1.0)]
    result = bounded_closure(node_ids, edges, h=2, mode="boolean")
    assert isinstance(result, ClosureResult)
    assert result.mode == "boolean"
    assert result.n == 3
    idx = {nid: i for i, nid in enumerate(node_ids)}
    assert result.reachability[idx["a"], idx["c"]]  # reachable in 2 hops
    assert result.path_count > 0


def test_bounded_closure_cost():
    node_ids = ["a", "b", "c"]
    edges = [("a", "b", 1.0), ("b", "c", 2.0)]
    result = bounded_closure(node_ids, edges, h=2, mode="cost")
    assert result.mode == "cost"
    idx = {nid: i for i, nid in enumerate(node_ids)}
    assert result.reachability[idx["a"], idx["c"]] == pytest.approx(3.0)


def test_bounded_closure_utility():
    node_ids = ["a", "b", "c"]
    edges = [("a", "b", 5.0), ("b", "c", 3.0)]
    result = bounded_closure(node_ids, edges, h=2, mode="utility")
    assert result.mode == "utility"
    idx = {nid: i for i, nid in enumerate(node_ids)}
    assert result.reachability[idx["a"], idx["c"]] == pytest.approx(8.0)


def test_bounded_closure_empty_graph():
    result = bounded_closure([], [], h=2, mode="boolean")
    assert result.n == 0
    assert result.path_count == 0


def test_bounded_closure_ignores_out_of_slice_edges():
    node_ids = ["a", "b"]
    edges = [("a", "b", 1.0), ("a", "x", 1.0)]  # "x" not in node_ids
    result = bounded_closure(node_ids, edges, h=1, mode="boolean")
    assert result.n == 2  # "x" excluded


def test_bounded_closure_invalid_h():
    with pytest.raises(ValueError):
        bounded_closure(["a"], [], h=0, mode="boolean")


def test_bounded_closure_invalid_mode():
    with pytest.raises(ValueError):
        bounded_closure(["a"], [], h=1, mode="invalid")  # type: ignore


# ── closure_critical_edges ────────────────────────────────────────────────────

def test_closure_critical_edges_chain():
    # 0->1->2: removing 0->1 cuts 0's reach to 2
    nodes = ["a", "b", "c"]
    edges = [("a", "b", 1.0), ("b", "c", 1.0)]
    critical = closure_critical_edges(nodes, edges, h=2, mode="boolean")
    assert ("a", "b") in critical


def test_closure_critical_edges_redundant():
    # Two paths from a to c: a->b->c and a->c direct
    nodes = ["a", "b", "c"]
    edges = [("a", "b", 1.0), ("b", "c", 1.0), ("a", "c", 1.0)]
    critical = closure_critical_edges(nodes, edges, h=2, mode="boolean")
    # Removing a->c doesn't cut any connection (a->b->c still works)
    assert ("a", "c") not in critical


def test_closure_critical_edges_empty():
    assert closure_critical_edges(["a", "b"], [], h=2, mode="boolean") == set()


# ── QuantaleSpec — structure ──────────────────────────────────────────────────

def test_quantale_spec_repr():
    assert "boolean" in repr(BOOLEAN_SPEC)
    assert "cost" in repr(COST_SPEC)
    assert "utility" in repr(UTILITY_SPEC)


def test_quantale_spec_map_keys():
    assert set(_SPEC_MAP) == {"boolean", "cost", "utility"}
    assert _SPEC_MAP["boolean"] is BOOLEAN_SPEC
    assert _SPEC_MAP["cost"] is COST_SPEC
    assert _SPEC_MAP["utility"] is UTILITY_SPEC


def test_boolean_spec_gate5():
    assert BOOLEAN_SPEC.valid(True)
    assert not BOOLEAN_SPEC.valid(False)
    assert BOOLEAN_SPEC.better(True, False)
    assert not BOOLEAN_SPEC.better(False, True)
    assert not BOOLEAN_SPEC.better(True, True)


def test_cost_spec_gate5():
    assert COST_SPEC.valid(0.0)
    assert COST_SPEC.valid(5.0)
    assert not COST_SPEC.valid(1e9)
    assert COST_SPEC.better(1.0, 2.0)
    assert not COST_SPEC.better(2.0, 1.0)


def test_utility_spec_gate5():
    assert UTILITY_SPEC.valid(0.0)
    assert UTILITY_SPEC.valid(5.0)
    assert not UTILITY_SPEC.valid(-1e9)
    assert UTILITY_SPEC.better(5.0, 3.0)
    assert not UTILITY_SPEC.better(3.0, 5.0)


# ── quantale_closure — generic matrix close ───────────────────────────────────

def test_quantale_closure_boolean_matches_boolean_closure():
    A = np.zeros((4, 4), dtype=bool)
    A[0, 1] = A[1, 2] = A[2, 3] = True
    R_legacy = boolean_closure(A, h=3)
    R_generic = quantale_closure(A, h=3, spec=BOOLEAN_SPEC)
    np.testing.assert_array_equal(R_legacy, R_generic)


def test_quantale_closure_cost_matches_cost_closure():
    A = np.full((3, 3), _LARGE)
    A[0, 1] = 2.0
    A[1, 2] = 3.0
    R_legacy = cost_closure(A, h=2)
    R_generic = quantale_closure(A, h=2, spec=COST_SPEC)
    np.testing.assert_allclose(R_legacy, R_generic, atol=1e-6)


def test_quantale_closure_utility_matches_utility_closure():
    NEG = -_LARGE
    A = np.full((3, 3), NEG)
    A[0, 1] = 5.0
    A[1, 2] = 3.0
    R_legacy = utility_closure(A, h=2)
    R_generic = quantale_closure(A, h=2, spec=UTILITY_SPEC)
    np.testing.assert_allclose(R_legacy, R_generic, atol=1e-6)


def test_quantale_closure_boolean_horizon():
    A = np.zeros((4, 4), dtype=bool)
    A[0, 1] = A[1, 2] = A[2, 3] = True
    R1 = quantale_closure(A, h=1, spec=BOOLEAN_SPEC)
    R3 = quantale_closure(A, h=3, spec=BOOLEAN_SPEC)
    assert not R1[0, 2]  # needs 2 hops
    assert R3[0, 3]      # reachable in 3 hops


def test_quantale_closure_cost_shortest_path():
    A = np.full((3, 3), _LARGE)
    A[0, 2] = 10.0
    A[0, 1] = 2.0
    A[1, 2] = 3.0
    R = quantale_closure(A, h=2, spec=COST_SPEC)
    assert R[0, 2] == pytest.approx(5.0)  # 2+3 < 10


def test_quantale_closure_utility_max_path():
    NEG = -_LARGE
    A = np.full((3, 3), NEG)
    A[0, 2] = 4.0
    A[0, 1] = 5.0
    A[1, 2] = 3.0
    R = quantale_closure(A, h=2, spec=UTILITY_SPEC)
    assert R[0, 2] == pytest.approx(8.0)  # 5+3 > 4


# ── quantale_bounded_closure API ──────────────────────────────────────────────

def test_quantale_bounded_closure_boolean_agrees_with_bounded():
    node_ids = ["a", "b", "c"]
    edges = [("a", "b", 1.0), ("b", "c", 1.0)]
    r_mode = bounded_closure(node_ids, edges, h=2, mode="boolean")
    r_spec = quantale_bounded_closure(node_ids, edges, h=2, spec=BOOLEAN_SPEC)
    np.testing.assert_array_equal(r_mode.reachability, r_spec.reachability)
    assert r_spec.path_count == r_mode.path_count


def test_quantale_bounded_closure_cost_agrees_with_bounded():
    node_ids = ["a", "b", "c"]
    edges = [("a", "b", 1.0), ("b", "c", 2.0)]
    r_mode = bounded_closure(node_ids, edges, h=2, mode="cost")
    r_spec = quantale_bounded_closure(node_ids, edges, h=2, spec=COST_SPEC)
    idx = {n: i for i, n in enumerate(node_ids)}
    assert r_spec.reachability[idx["a"], idx["c"]] == pytest.approx(3.0)
    np.testing.assert_allclose(r_mode.reachability, r_spec.reachability, atol=1e-6)


def test_quantale_bounded_closure_utility_agrees_with_bounded():
    node_ids = ["a", "b", "c"]
    edges = [("a", "b", 5.0), ("b", "c", 3.0)]
    r_mode = bounded_closure(node_ids, edges, h=2, mode="utility")
    r_spec = quantale_bounded_closure(node_ids, edges, h=2, spec=UTILITY_SPEC)
    idx = {n: i for i, n in enumerate(node_ids)}
    assert r_spec.reachability[idx["a"], idx["c"]] == pytest.approx(8.0)
    np.testing.assert_allclose(r_mode.reachability, r_spec.reachability, atol=1e-6)


def test_quantale_bounded_closure_empty():
    r = quantale_bounded_closure([], [], h=2, spec=BOOLEAN_SPEC)
    assert r.n == 0
    assert r.path_count == 0


def test_quantale_bounded_closure_invalid_h():
    with pytest.raises(ValueError):
        quantale_bounded_closure(["a"], [], h=0, spec=BOOLEAN_SPEC)


def test_quantale_bounded_closure_returns_closure_result():
    r = quantale_bounded_closure(["a", "b"], [("a", "b", 1.0)], h=1, spec=BOOLEAN_SPEC)
    assert isinstance(r, ClosureResult)
    assert r.mode == "boolean"
    assert r.n == 2
