from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .ir import MathNode, op_class


EXPERT_NAMES: list[str] = [
    "affine_expert",
    "matmul_expert",
    "reduction_expert",
    "graph_expert",
    "grad_expert",
    "constraint_expert",
    "generic_expert",
]

_OP_CLASS_TO_EXPERT: dict[str, str] = {
    "affine":      "affine_expert",
    "matmul":      "matmul_expert",
    "reduction":   "reduction_expert",
    "grad":        "grad_expert",
    "constraint":  "constraint_expert",
    "elementwise": "generic_expert",
    "leaf":        "generic_expert",
    "generic":     "generic_expert",
}


def _is_affine_form(node: MathNode) -> bool:
    """True for add(matmul(...), ...) — the canonical affine pattern."""
    return (
        node.op == "add"
        and len(node.args) == 2
        and any(a.op == "matmul" for a in node.args)
    )


@dataclass
class RouteResult:
    expert: str
    confidence: float
    reason: str


class OperatorRouter:
    """
    v0: rule-based routing by op class.
    The generic_expert is always available as a fallback — no token is dropped.
    """

    def route(
        self,
        node: MathNode,
        z: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
    ) -> RouteResult:
        if _is_affine_form(node):
            return RouteResult("affine_expert", 1.0, "affine_pattern")
        oc = op_class(node)
        expert = _OP_CLASS_TO_EXPERT.get(oc, "generic_expert")
        return RouteResult(expert, 1.0, f"op_class={oc}")

    def route_batch(
        self,
        nodes: list[MathNode],
        z: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
    ) -> list[RouteResult]:
        return [
            self.route(
                n,
                z[i] if z is not None else None,
                h[i] if h is not None else None,
            )
            for i, n in enumerate(nodes)
        ]

    def route_diagnostics(self, nodes: list[MathNode]) -> dict:
        results = self.route_batch(nodes)
        counts: dict[str, int] = {e: 0 for e in EXPERT_NAMES}
        for r in results:
            counts[r.expert] = counts.get(r.expert, 0) + 1
        return {
            "routes": [(repr(n), r.expert, r.reason) for n, r in zip(nodes, results)],
            "counts": counts,
            "total": len(nodes),
        }
