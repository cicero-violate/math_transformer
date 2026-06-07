from __future__ import annotations
from typing import Optional
from .ir import MathNode


class ShapeError(Exception):
    pass


Shape = tuple[int, ...]


def infer_shape(
    node: MathNode,
    env: dict[str, Shape] | None = None,
) -> Optional[Shape]:
    """
    Infer the output shape of a MathNode given variable shapes in env.
    Returns None when shape cannot be determined.
    Raises ShapeError when shapes are provably incompatible.
    """
    if env is None:
        env = {}

    # ── Leaves ────────────────────────────────────────────────────────────
    if node.op == "var":
        # Check node.shape first, then env
        if node.shape is not None:
            return node.shape
        return env.get(str(node.value))

    if node.op == "const":
        return node.shape if node.shape is not None else ()  # scalar

    # ── Element-wise ──────────────────────────────────────────────────────
    if node.op in ("add", "sub", "mul", "div") and len(node.args) == 2:
        sa = infer_shape(node.args[0], env)
        sb = infer_shape(node.args[1], env)
        if sa is None and sb is None:
            return None
        if sa is None:
            return sb
        if sb is None:
            return sa
        if sa == ():
            return sb  # scalar broadcast
        if sb == ():
            return sa
        if sa == sb:
            return sa
        raise ShapeError(
            f"{node.op}: shape mismatch {sa} vs {sb}"
        )

    if node.op == "neg" and len(node.args) == 1:
        return infer_shape(node.args[0], env)

    # ── Matrix multiplication ─────────────────────────────────────────────
    if node.op == "matmul" and len(node.args) == 2:
        sa = infer_shape(node.args[0], env)
        sb = infer_shape(node.args[1], env)
        if sa is None or sb is None:
            return None
        if len(sa) == 0 or len(sb) == 0:
            raise ShapeError("matmul requires at least 1-D inputs")
        if sa[-1] != sb[0]:
            raise ShapeError(
                f"matmul inner-dimension mismatch: {sa} @ {sb}"
            )
        if len(sa) == 2 and len(sb) == 1:
            return (sa[0],)
        if len(sa) == 2 and len(sb) == 2:
            return (sa[0], sb[1])
        if len(sa) == 1 and len(sb) == 1:
            return ()  # dot product → scalar
        if len(sa) == 1 and len(sb) == 2:
            return (sb[1],)
        return None  # higher-rank batched matmul not handled in v0

    # ── Affine ────────────────────────────────────────────────────────────
    if node.op == "affine" and len(node.args) == 3:
        A, x, b = node.args
        mm_node = MathNode(op="matmul", args=(A, x))
        mm_shape = infer_shape(mm_node, env)
        b_shape = infer_shape(b, env)
        if mm_shape is not None and b_shape is not None and mm_shape != b_shape:
            raise ShapeError(
                f"affine: matmul output {mm_shape} != bias shape {b_shape}"
            )
        return mm_shape

    # ── Transpose ─────────────────────────────────────────────────────────
    if node.op == "transpose" and len(node.args) == 1:
        sa = infer_shape(node.args[0], env)
        if sa is None:
            return None
        if len(sa) == 2:
            return (sa[1], sa[0])
        if len(sa) == 1:
            return sa
        return None

    # ── Reductions ────────────────────────────────────────────────────────
    if node.op in ("sum", "mean") and len(node.args) >= 1:
        # v0: reduction collapses to scalar
        return (1,)

    if node.op == "norm" and len(node.args) >= 1:
        return (1,)

    # ── Gradient ──────────────────────────────────────────────────────────
    if node.op == "grad" and len(node.args) == 2:
        # grad(f, x) has same shape as x
        return infer_shape(node.args[1], env)

    return None


def check_shape(
    node: MathNode,
    env: dict[str, Shape] | None = None,
) -> Shape | None:
    """
    Like infer_shape but re-raises ShapeError with the node context attached.
    """
    try:
        return infer_shape(node, env)
    except ShapeError as e:
        raise ShapeError(f"In {node.op}: {e}") from e


def infer_tree(
    node: MathNode,
    env: dict[str, Shape] | None = None,
) -> dict[MathNode, Optional[Shape]]:
    """
    Recursively infer shapes for every node in the subtree.
    Returns a mapping node -> shape (or None if unknown).
    Raises ShapeError on first inconsistency found.
    """
    result: dict[MathNode, Optional[Shape]] = {}
    _infer_tree_recursive(node, env or {}, result)
    return result


def _infer_tree_recursive(
    node: MathNode,
    env: dict[str, Shape],
    result: dict[MathNode, Optional[Shape]],
) -> None:
    for child in node.args:
        _infer_tree_recursive(child, env, result)
    result[node] = infer_shape(node, env)
