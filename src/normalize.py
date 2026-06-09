from __future__ import annotations
from .ir import MathNode


def normalize(node: MathNode) -> MathNode:
    """Normalize a MathNode tree bottom-up, applying canonical rewrites."""
    new_args = tuple(normalize(a) for a in node.args)
    node = MathNode(
        op=node.op, args=new_args, value=node.value,
        shape=node.shape, dtype=node.dtype, attrs=dict(node.attrs),
    )
    node = _expand_affine(node)
    node = _identity_elimination(node)
    return node


def _expand_affine(node: MathNode) -> MathNode:
    """affine(A, x, b) -> add(matmul(A, x), b)"""
    if node.op == "affine" and len(node.args) == 3:
        A, x, b = node.args
        return MathNode(op="add", args=(MathNode(op="matmul", args=(A, x)), b))
    return node


def _identity_elimination(node: MathNode) -> MathNode:
    """
    Remove additive zeros and scalar multiplicative ones.
    NOTE: matmul identity is NOT simplified here — matrix identity != scalar 1.
    Only scalar mul(x, 1) is simplified.
    """
    if node.op == "add" and len(node.args) == 2:
        a, b = node.args
        if _is_zero(b):
            return a
        if _is_zero(a):
            return b

    # Only for scalar multiplication, not matmul
    if node.op == "mul" and len(node.args) == 2:
        a, b = node.args
        if _is_one(b):
            return a
        if _is_one(a):
            return b

    return node


def _is_zero(node: MathNode) -> bool:
    return node.op == "const" and node.value == 0


def _is_one(node: MathNode) -> bool:
    return node.op == "const" and node.value == 1
