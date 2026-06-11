"""
Code world graph compiler (P3.1).

Compiles AST/CFG/SSA/CALL/TYPE/TEST structure from Python source into a
WorldGraph that can be used as G_world for frontier expansion experiments.

Node kinds: module, function, class, variable, call, import, argument, return, statement
Edge kinds: DEFINES, CALLS, IMPORTS, INHERITS, ASSIGNS, USES, RETURNS, HAS_ARG, CONTAINS

Usage:
    result = parse_python_to_world_graph(source)
    world = result.world        # G_world for recurrent frontier
    # Seed active graph from function entry points:
    active = ActiveGraph.seed(result.function_node_ids[:3], budget=32)
"""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .world_graph import WorldGraph, NodeRecord, EdgeRecord, make_node_id, make_edge_id


class CodeNodeKind(str, Enum):
    MODULE = "module"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    CALL = "call"
    IMPORT = "import"
    ARGUMENT = "argument"
    RETURN = "return"
    STATEMENT = "statement"
    ATTRIBUTE = "attribute"


class CodeEdgeKind(str, Enum):
    DEFINES = "DEFINES"
    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    INHERITS = "INHERITS"
    ASSIGNS = "ASSIGNS"
    USES = "USES"
    RETURNS = "RETURNS"
    HAS_ARG = "HAS_ARG"
    CONTAINS = "CONTAINS"


@dataclass
class CodeCompilerResult:
    world: WorldGraph
    node_ids: list[str]                # all node IDs in insertion order
    function_node_ids: list[str]       # function nodes only (good seed IDs)
    class_node_ids: list[str]
    import_node_ids: list[str]
    stats: dict[str, int]


def _attr_chain(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_attr_chain(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return "<expr>"


class _CodeVisitor(ast.NodeVisitor):

    def __init__(self, world: WorldGraph, provenance: str) -> None:
        self.world = world
        self.provenance = provenance
        self._scope: list[str] = []
        self.all_ids: list[str] = []
        self.fn_ids: list[str] = []
        self.cls_ids: list[str] = []
        self.imp_ids: list[str] = []

    def _add_node(self, label: str, kind: CodeNodeKind, features: dict[str, Any] | None = None) -> str:
        f = features or {}
        lineno = f.get("lineno", 0)
        prov = f"{self.provenance}:{lineno}" if lineno else self.provenance
        nid = make_node_id(label, kind.value, prov)
        rec = NodeRecord(
            node_id=nid,
            label=label,
            node_kind=kind.value,
            features=features or {},
            provenance=self.provenance,
        )
        self.world.add_node(rec)
        self.all_ids.append(nid)
        return nid

    def _add_edge(self, src: str, dst: str, kind: CodeEdgeKind) -> None:
        eid = make_edge_id(src, dst, kind.value)
        try:
            self.world.add_edge(EdgeRecord(
                edge_id=eid, src_id=src, dst_id=dst, relation=kind.value,
            ))
        except ValueError:
            pass

    def _scope_top(self) -> str | None:
        return self._scope[-1] if self._scope else None

    # ── Visitors ──────────────────────────────────────────────────────────────

    def visit_Module(self, node: ast.Module) -> None:
        mid = self._add_node("<module>", CodeNodeKind.MODULE, {"lineno": 0})
        self._scope.append(mid)
        self.generic_visit(node)
        self._scope.pop()

    def _visit_funcdef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        fid = self._add_node(
            node.name,
            CodeNodeKind.FUNCTION,
            {"lineno": node.lineno, "n_args": len(node.args.args), "is_async": isinstance(node, ast.AsyncFunctionDef)},
        )
        self.fn_ids.append(fid)
        scope = self._scope_top()
        if scope:
            self._add_edge(scope, fid, CodeEdgeKind.DEFINES)

        for arg in node.args.args:
            arg_id = self._add_node(
                f"{node.name}.{arg.arg}", CodeNodeKind.ARGUMENT,
                {"arg_name": arg.arg, "function": node.name},
            )
            self._add_edge(fid, arg_id, CodeEdgeKind.HAS_ARG)

        self._scope.append(fid)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _visit_funcdef
    visit_AsyncFunctionDef = _visit_funcdef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        cid = self._add_node(
            node.name, CodeNodeKind.CLASS,
            {"lineno": node.lineno, "n_bases": len(node.bases)},
        )
        self.cls_ids.append(cid)
        scope = self._scope_top()
        if scope:
            self._add_edge(scope, cid, CodeEdgeKind.DEFINES)

        for base in node.bases:
            if isinstance(base, (ast.Name, ast.Attribute)):
                base_name = _attr_chain(base)
                base_id = self._add_node(base_name, CodeNodeKind.CLASS, {"is_base": True})
                self._add_edge(cid, base_id, CodeEdgeKind.INHERITS)

        self._scope.append(cid)
        self.generic_visit(node)
        self._scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            imp_id = self._add_node(
                alias.name, CodeNodeKind.IMPORT,
                {"alias": alias.asname or alias.name, "lineno": node.lineno},
            )
            self.imp_ids.append(imp_id)
            scope = self._scope_top()
            if scope:
                self._add_edge(scope, imp_id, CodeEdgeKind.IMPORTS)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or "<unknown>"
        for alias in node.names:
            label = f"{module}.{alias.name}"
            imp_id = self._add_node(
                label, CodeNodeKind.IMPORT,
                {"module": module, "name": alias.name,
                 "alias": alias.asname or alias.name, "lineno": node.lineno},
            )
            self.imp_ids.append(imp_id)
            scope = self._scope_top()
            if scope:
                self._add_edge(scope, imp_id, CodeEdgeKind.IMPORTS)

    def visit_Assign(self, node: ast.Assign) -> None:
        scope = self._scope_top()
        for target in node.targets:
            if isinstance(target, ast.Name):
                vid = self._add_node(
                    target.id, CodeNodeKind.VARIABLE,
                    {"lineno": node.lineno, "scope": scope},
                )
                if scope:
                    self._add_edge(scope, vid, CodeEdgeKind.ASSIGNS)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        scope = self._scope_top()
        if isinstance(node.target, ast.Name):
            vid = self._add_node(
                node.target.id, CodeNodeKind.VARIABLE,
                {"lineno": node.lineno, "annotated": True, "scope": scope},
            )
            if scope:
                self._add_edge(scope, vid, CodeEdgeKind.ASSIGNS)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _attr_chain(node.func) if isinstance(node.func, (ast.Name, ast.Attribute)) else "<expr>"
        cid = self._add_node(
            f"call:{callee}", CodeNodeKind.CALL,
            {"callee": callee, "lineno": getattr(node, "lineno", 0)},
        )
        scope = self._scope_top()
        if scope:
            self._add_edge(scope, cid, CodeEdgeKind.CALLS)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        scope = self._scope_top()
        if scope:
            rid = self._add_node(
                f"return@{scope}", CodeNodeKind.RETURN,
                {"lineno": getattr(node, "lineno", 0), "function": scope},
            )
            self._add_edge(scope, rid, CodeEdgeKind.RETURNS)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Record USE edges for Name nodes in Load context
        if isinstance(node.ctx, ast.Load):
            scope = self._scope_top()
            if scope:
                vid = self._add_node(
                    node.id, CodeNodeKind.VARIABLE,
                    {"is_use": True, "lineno": getattr(node, "lineno", 0)},
                )
                self._add_edge(scope, vid, CodeEdgeKind.USES)
        # Do not call generic_visit for Name (no sub-nodes)


def parse_python_to_world_graph(
    source: str,
    provenance: str = "python_source",
) -> CodeCompilerResult:
    """
    Compile Python source into a WorldGraph (G_world).

    Nodes:  module, function, class, variable, call, import, argument, return
    Edges:  DEFINES, CALLS, IMPORTS, INHERITS, ASSIGNS, USES, RETURNS, HAS_ARG

    The resulting WorldGraph can seed an ActiveGraph for frontier expansion.
    Use result.function_node_ids as seed_ids for recurrent frontier runs on code tasks.
    """
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    world = WorldGraph()
    visitor = _CodeVisitor(world, provenance)
    visitor.visit(tree)

    stats = {
        "nodes": world.node_count(),
        "edges": world.edge_count(),
        "functions": len(visitor.fn_ids),
        "classes": len(visitor.cls_ids),
        "calls": sum(1 for n in world.iter_nodes() if n.node_kind == CodeNodeKind.CALL.value),
        "imports": len(visitor.imp_ids),
        "variables": sum(1 for n in world.iter_nodes() if n.node_kind == CodeNodeKind.VARIABLE.value),
    }

    return CodeCompilerResult(
        world=world,
        node_ids=visitor.all_ids,
        function_node_ids=visitor.fn_ids,
        class_node_ids=visitor.cls_ids,
        import_node_ids=visitor.imp_ids,
        stats=stats,
    )
