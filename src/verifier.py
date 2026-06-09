from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from .ir import MathNode, op_class


@dataclass
class ExecutionPlan:
    node: MathNode
    expert: str
    input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    output_shape: Optional[tuple[int, ...]] = None
    input_dtypes: dict[str, str] = field(default_factory=dict)
    output_dtype: Optional[str] = None


@dataclass
class VerificationResult:
    passed: bool
    level: int   # highest level checked (0–5)
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class Verifier:
    """
    Symbolic verification gate.
    v0: L0 (shape), L1 (dtype), L2 (expert compat) at single-node level.
    check_tree(): recursive shape + compatibility checks over full subtree.
    """

    # ── Single-node check ──────────────────────────────────────────────────

    def check(self, node: MathNode, plan: ExecutionPlan) -> VerificationResult:
        for fn in (self._check_shapes, self._check_dtypes, self._check_op_compatibility):
            result = fn(node, plan)
            if not result.passed:
                return result
        return VerificationResult(True, 2, "All checks passed (L0-L2)")

    def _check_shapes(self, node: MathNode, plan: ExecutionPlan) -> VerificationResult:
        if node.op == "matmul" and len(node.args) == 2:
            A_shape = plan.input_shapes.get(_node_key(node.args[0]))
            x_shape = plan.input_shapes.get(_node_key(node.args[1]))
            if A_shape and x_shape:
                if A_shape[-1] != x_shape[0]:
                    return VerificationResult(
                        False, 0,
                        f"matmul inner-dim mismatch: {A_shape} @ {x_shape}",
                        {"A_shape": A_shape, "x_shape": x_shape},
                    )
        if node.op == "add" and len(node.args) == 2:
            a_shape = plan.input_shapes.get(_node_key(node.args[0]))
            b_shape = plan.input_shapes.get(_node_key(node.args[1]))
            if a_shape and b_shape and a_shape != b_shape and a_shape != () and b_shape != ():
                return VerificationResult(
                    False, 0,
                    f"add shape mismatch: {a_shape} vs {b_shape}",
                    {"a_shape": a_shape, "b_shape": b_shape},
                )
        return VerificationResult(True, 0, "Shape check passed")

    def _check_dtypes(self, node: MathNode, plan: ExecutionPlan) -> VerificationResult:
        if not plan.input_dtypes:
            return VerificationResult(True, 1, "No dtype info available")
        dtypes = set(plan.input_dtypes.values())
        if len(dtypes) > 1:
            return VerificationResult(
                True, 1,
                f"Mixed dtypes (implicit widening assumed): {dtypes}",
                {"dtypes": list(dtypes)},
            )
        return VerificationResult(True, 1, "Dtype check passed")

    def _check_op_compatibility(self, node: MathNode, plan: ExecutionPlan) -> VerificationResult:
        from .router import _OP_CLASS_TO_EXPERT
        if plan.expert == "generic_expert":
            return VerificationResult(True, 2, "Generic expert always valid")
        expected = _OP_CLASS_TO_EXPERT.get(op_class(node), "generic_expert")
        if plan.expert == expected:
            return VerificationResult(True, 2, "Expert matches op class")
        if plan.expert == "affine_expert" and node.op in ("add", "affine"):
            return VerificationResult(True, 2, "Affine expert compatible with add/affine")
        return VerificationResult(
            False, 2,
            f"Expert mismatch: assigned={plan.expert!r}, expected={expected!r}",
            {"op": node.op, "assigned": plan.expert, "expected": expected},
        )

    # ── Recursive tree check ───────────────────────────────────────────────

    def check_tree(
        self,
        root: MathNode,
        env: dict[str, tuple[int, ...]] | None = None,
    ) -> VerificationResult:
        """
        Recursively verify the entire expression tree using shape inference.
        Checks every matmul and add in the subtree for shape consistency.
        """
        from .shape import infer_shape, ShapeError
        try:
            _recursive_shape_check(root, env or {})
        except ShapeError as e:
            return VerificationResult(False, 0, str(e), {"error": str(e)})
        return VerificationResult(True, 0, "Tree shape check passed")


def _recursive_shape_check(
    node: MathNode,
    env: dict[str, tuple[int, ...]],
) -> None:
    """Walk the tree; raise ShapeError on first inconsistency."""
    from .shape import infer_shape, ShapeError
    # Check children first
    for child in node.args:
        _recursive_shape_check(child, env)
    # Check this node
    try:
        infer_shape(node, env)
    except ShapeError:
        raise


def _node_key(node: MathNode) -> str:
    if node.op == "var":
        return str(node.value)
    return repr(node)
