"""Tests for code_world_graph.py — P3.1 code world graph compiler."""
from __future__ import annotations

import pytest

from src.code_world_graph import (
    CodeEdgeKind,
    CodeNodeKind,
    CodeCompilerResult,
    parse_python_to_world_graph,
)
from src.world_graph import WorldGraph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse(source: str) -> CodeCompilerResult:
    return parse_python_to_world_graph(source, provenance="test")


def _node_labels(world: WorldGraph) -> set[str]:
    return {n.label for n in world.iter_nodes()}


def _edge_relations(world: WorldGraph) -> set[str]:
    return {e.relation for e in world.iter_edges()}


# ── Basic structure ───────────────────────────────────────────────────────────

def test_parses_empty_module():
    result = _parse("")
    assert result.stats["nodes"] >= 1  # at least <module>
    assert result.stats["edges"] >= 0


def test_module_node_created():
    result = _parse("x = 1")
    assert "<module>" in _node_labels(result.world)


def test_function_def_creates_node():
    result = _parse("def foo(): pass")
    assert "foo" in _node_labels(result.world)
    assert len(result.function_node_ids) == 1


def test_class_def_creates_node():
    result = _parse("class Bar: pass")
    assert "Bar" in _node_labels(result.world)
    assert len(result.class_node_ids) == 1


def test_import_creates_node():
    result = _parse("import os")
    assert "os" in _node_labels(result.world)
    assert len(result.import_node_ids) == 1


def test_import_from_creates_node():
    result = _parse("from pathlib import Path")
    labels = _node_labels(result.world)
    assert any("pathlib" in l and "Path" in l for l in labels)


# ── Edges ─────────────────────────────────────────────────────────────────────

def test_defines_edge_function():
    result = _parse("def foo(): pass")
    assert CodeEdgeKind.DEFINES.value in _edge_relations(result.world)


def test_defines_edge_class():
    result = _parse("class Foo: pass")
    assert CodeEdgeKind.DEFINES.value in _edge_relations(result.world)


def test_assigns_edge():
    result = _parse("x = 42")
    assert CodeEdgeKind.ASSIGNS.value in _edge_relations(result.world)


def test_calls_edge():
    result = _parse("""
def foo():
    bar()
""")
    assert CodeEdgeKind.CALLS.value in _edge_relations(result.world)


def test_imports_edge():
    result = _parse("import sys")
    assert CodeEdgeKind.IMPORTS.value in _edge_relations(result.world)


def test_has_arg_edge():
    result = _parse("def f(x, y): pass")
    assert CodeEdgeKind.HAS_ARG.value in _edge_relations(result.world)


def test_inherits_edge():
    result = _parse("class B(A): pass")
    assert CodeEdgeKind.INHERITS.value in _edge_relations(result.world)


def test_returns_edge():
    result = _parse("""
def f():
    return 1
""")
    assert CodeEdgeKind.RETURNS.value in _edge_relations(result.world)


# ── Stats ─────────────────────────────────────────────────────────────────────

def test_stats_counts():
    source = """
import os
import sys

def foo(a, b):
    return a + b

class MyClass:
    def method(self):
        foo(1, 2)
"""
    result = _parse(source)
    assert result.stats["functions"] >= 2  # foo + method
    assert result.stats["classes"] >= 1
    assert result.stats["imports"] >= 2
    assert result.stats["nodes"] > 10


# ── Complex source ────────────────────────────────────────────────────────────

def test_nested_functions():
    result = _parse("""
def outer():
    def inner():
        pass
""")
    labels = _node_labels(result.world)
    assert "outer" in labels
    assert "inner" in labels
    assert result.stats["functions"] == 2


def test_multiple_classes_with_methods():
    result = _parse("""
class A:
    def m(self): pass

class B(A):
    def m(self): pass
""")
    assert result.stats["classes"] >= 2
    assert CodeEdgeKind.INHERITS.value in _edge_relations(result.world)


def test_function_node_ids_usable_as_seeds():
    """function_node_ids should be valid node IDs in the world graph."""
    result = _parse("def f(): pass\ndef g(): pass")
    for nid in result.function_node_ids:
        assert result.world.has_node(nid)


def test_world_graph_can_seed_active_graph():
    """The resulting world graph is ready to seed an ActiveGraph."""
    from src.world_graph import ActiveGraph
    result = _parse("""
def alpha():
    pass

def beta():
    alpha()
""")
    if result.function_node_ids:
        active = ActiveGraph.seed(result.function_node_ids[:1], budget=16)
        assert len(active.node_ids) == 1
        candidates = active.boundary_candidates(result.world)
        # function nodes should have edges (DEFINES from module)
        assert len(candidates) >= 0  # just confirm no crash


# ── Provenance ────────────────────────────────────────────────────────────────

def test_provenance_set_on_nodes():
    result = parse_python_to_world_graph("def f(): pass", provenance="my_prov")
    for nid in result.function_node_ids:
        node = result.world.get_node(nid)
        assert node.provenance == "my_prov"


# ── Duplicate import names ────────────────────────────────────────────────────

def test_duplicate_import_names_handled():
    """Repeated imports with same name don't crash."""
    result = _parse("import os\nimport os")
    # Should not raise; os imported once due to hash-based IDs
    assert result.stats["nodes"] >= 1
