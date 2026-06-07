from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass(frozen=True)
class MathNode:
    op: str
    args: tuple[MathNode, ...]
    value: Union[str, float, int, None] = None
    shape: Optional[tuple[int, ...]] = None
    dtype: Optional[str] = None
    attrs: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))

    def __repr__(self) -> str:
        if self.op in ("var", "const"):
            return str(self.value)
        return f"{self.op}({', '.join(repr(a) for a in self.args)})"

    @property
    def arity(self) -> int:
        return len(self.args)

    @property
    def depth(self) -> int:
        if not self.args:
            return 0
        return 1 + max(a.depth for a in self.args)

    @property
    def subtree_size(self) -> int:
        return 1 + sum(a.subtree_size for a in self.args)

    def is_leaf(self) -> bool:
        return len(self.args) == 0

    def collect_nodes(self) -> list[MathNode]:
        """BFS order over the subtree rooted at this node."""
        result: list[MathNode] = []
        queue = [self]
        while queue:
            node = queue.pop(0)
            result.append(node)
            queue.extend(node.args)
        return result

    def collect_edges(self) -> list[tuple[MathNode, MathNode]]:
        """All parent->child edges in the subtree."""
        edges: list[tuple[MathNode, MathNode]] = []
        for node in self.collect_nodes():
            for child in node.args:
                edges.append((node, child))
        return edges


# ── Convenience constructors ──────────────────────────────────────────────────

def var(name: str, shape: Optional[tuple[int, ...]] = None,
        dtype: Optional[str] = None) -> MathNode:
    return MathNode(op="var", args=(), value=name, shape=shape, dtype=dtype)


def const(value: Union[int, float], shape: Optional[tuple[int, ...]] = None,
          dtype: Optional[str] = None) -> MathNode:
    return MathNode(op="const", args=(), value=value, shape=shape, dtype=dtype)


def add(a: MathNode, b: MathNode, shape: Optional[tuple[int, ...]] = None) -> MathNode:
    return MathNode(op="add", args=(a, b), shape=shape)


def matmul(a: MathNode, b: MathNode, shape: Optional[tuple[int, ...]] = None) -> MathNode:
    return MathNode(op="matmul", args=(a, b), shape=shape)


def affine(A: MathNode, x: MathNode, b: MathNode,
           shape: Optional[tuple[int, ...]] = None) -> MathNode:
    return MathNode(op="affine", args=(A, x, b), shape=shape)


def grad(f: MathNode, x: MathNode) -> MathNode:
    return MathNode(op="grad", args=(f, x))


def constraint(rel: MathNode) -> MathNode:
    return MathNode(op="constraint", args=(rel,))


def leq(a: MathNode, b: MathNode) -> MathNode:
    return MathNode(op="leq", args=(a, b))


# ── Op metadata ───────────────────────────────────────────────────────────────

COMMUTATIVE_OPS: frozenset[str] = frozenset({"add", "mul"})

_OPERATOR_CLASSES: dict[str, str] = {
    "var": "leaf",
    "const": "leaf",
    "add": "elementwise",
    "sub": "elementwise",
    "mul": "elementwise",
    "div": "elementwise",
    "neg": "elementwise",
    "matmul": "matmul",
    "affine": "affine",
    "sum": "reduction",
    "mean": "reduction",
    "norm": "reduction",
    "grad": "grad",
    "constraint": "constraint",
    "leq": "constraint",
    "geq": "constraint",
    "eq": "constraint",
    "transpose": "matmul",
    "inv": "matmul",
}


def op_class(node: MathNode) -> str:
    return _OPERATOR_CLASSES.get(node.op, "generic")
