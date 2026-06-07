from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .ir import MathNode
from .embedder import MathEmbedder, pairwise_cosine


# ── Individual relation matrices ──────────────────────────────────────────────

def symbolic_dependency_matrix(nodes: list[MathNode]) -> np.ndarray:
    """Bidirectional parent-child IR edges."""
    n = len(nodes)
    mat = np.zeros((n, n), dtype=bool)
    for i, node in enumerate(nodes):
        for child in node.args:
            for j, other in enumerate(nodes):
                if other is child or other == child:
                    mat[i, j] = True
                    mat[j, i] = True
    return mat


def same_operator_matrix(nodes: list[MathNode]) -> np.ndarray:
    """Edges between non-leaf nodes that share the same operator."""
    n = len(nodes)
    mat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if nodes[i].op == nodes[j].op:
                mat[i, j] = True
                mat[j, i] = True
    return mat


def shape_compatibility_matrix(
    nodes: list[MathNode],
    env: dict[str, tuple[int, ...]] | None = None,
) -> np.ndarray:
    """
    Edges between nodes with identical known output shapes.
    Requires env for meaningful coverage.
    """
    n = len(nodes)
    mat = np.zeros((n, n), dtype=bool)
    if not env:
        return mat
    try:
        from .shape import infer_shape
    except ImportError:
        return mat
    shapes = [infer_shape(nd, env) for nd in nodes]
    for i in range(n):
        if shapes[i] is None:
            continue
        for j in range(i + 1, n):
            if shapes[j] is not None and shapes[i] == shapes[j]:
                mat[i, j] = True
                mat[j, i] = True
    return mat


def composition_matrix(
    nodes: list[MathNode],
    env: dict[str, tuple[int, ...]] | None = None,
) -> np.ndarray:
    """
    Edges where node_i's output shape matches the expected input shape
    of one of node_j's argument slots.
    Proxy for "node_i could feed node_j's computation."
    """
    n = len(nodes)
    mat = np.zeros((n, n), dtype=bool)
    if not env:
        return mat
    try:
        from .shape import infer_shape
    except ImportError:
        return mat

    out_shapes = [infer_shape(nd, env) for nd in nodes]

    for j, node_j in enumerate(nodes):
        for arg in node_j.args:
            arg_shape = infer_shape(arg, env)
            if arg_shape is None:
                continue
            for i in range(n):
                if i == j:
                    continue
                if out_shapes[i] is not None and out_shapes[i] == arg_shape:
                    mat[i, j] = True
                    mat[j, i] = True
    return mat


def embedding_topk_matrix(Z: np.ndarray, k: int) -> np.ndarray:
    """Top-k cosine-similarity neighbours in embedding space."""
    n = Z.shape[0]
    mat = np.zeros((n, n), dtype=bool)
    if k <= 0 or n <= 1:
        return mat
    sims = pairwise_cosine(Z)
    for i in range(n):
        row = sims[i].copy()
        row[i] = -2.0
        k_actual = min(k, n - 1)
        topk_idx = np.argpartition(row, -k_actual)[-k_actual:]
        mat[i, topk_idx] = True
        mat[topk_idx, i] = True
    return mat


def local_window_matrix(n: int, w: int) -> np.ndarray:
    """Contiguous window edges."""
    mat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n, i + w + 1)
        mat[i, lo:hi] = True
    return mat


def identity_matrix(n: int) -> np.ndarray:
    return np.eye(n, dtype=bool)


# ── Priority matrix ───────────────────────────────────────────────────────────
# Lower integer = higher priority (more important to keep under truncation).

RELATION_PRIORITY: dict[str, int] = {
    "identity":            1,
    "symbolic_dependency": 2,
    "composition":         3,
    "shape_compat":        4,
    "embedding_topk":      5,
    "local_window":        6,
    "same_operator":       7,
}


def build_priority_matrix(
    nodes: list[MathNode],
    z: np.ndarray | None = None,
    env: dict[str, tuple[int, ...]] | None = None,
    topk: int = 4,
    local_window: int = 2,
) -> np.ndarray:
    """
    Returns an (n, n) int matrix where each cell holds the highest-priority
    relation type connecting that pair (0 = not connected).
    Lower value = higher priority.
    """
    embedder = MathEmbedder()
    n = len(nodes)
    if z is None and topk > 0:
        z = embedder.encode_batch(nodes)

    # Build each layer
    layers = [
        ("identity",            identity_matrix(n)),
        ("symbolic_dependency", symbolic_dependency_matrix(nodes)),
        ("composition",         composition_matrix(nodes, env)),
        ("shape_compat",        shape_compatibility_matrix(nodes, env)),
        ("embedding_topk",      embedding_topk_matrix(z, topk) if z is not None and topk > 0
                                else np.zeros((n, n), dtype=bool)),
        ("local_window",        local_window_matrix(n, local_window)),
        ("same_operator",       same_operator_matrix(nodes)),
    ]

    priority = np.zeros((n, n), dtype=np.int8)
    for name, mat in layers:
        p = RELATION_PRIORITY[name]
        # Set cells that haven't been assigned yet (priority=0 = unset)
        unset = (priority == 0) & mat
        priority[unset] = p
        # Also update cells where this relation has higher priority
        existing = (priority > 0) & (priority > p) & mat
        priority[existing] = p

    return priority


# ── Diagnostics ───────────────────────────────────────────────────────────────

@dataclass
class MaskDiagnostics:
    n: int
    full_edges: int
    allowed_edges: int
    sparsity_ratio: float
    relation_reduction: float
    avg_k: float
    max_k: int
    padding_ratio: float   # 1 - allowed / (n * max_k) — wasted sparse compute
    by_relation: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"n={self.n}",
            f"full_edges={self.full_edges}",
            f"allowed_edges={self.allowed_edges}",
            f"sparsity_ratio={self.sparsity_ratio:.4f}",
            f"relation_reduction={self.relation_reduction:.4f}",
            f"avg_k={self.avg_k:.2f}",
            f"max_k={self.max_k}",
            f"padding_ratio={self.padding_ratio:.4f}",
            "by_relation:",
        ]
        for rel, cnt in self.by_relation.items():
            lines.append(f"  {rel}: {cnt}")
        return "\n".join(lines)


# ── TopologyBuilder ───────────────────────────────────────────────────────────

class TopologyBuilder:
    """
    Builds boolean attention masks from multiple relation types.
    """

    def __init__(
        self,
        topk: int = 4,
        local_window: int = 2,
        use_shape_compat: bool = True,
        use_composition: bool = True,
    ) -> None:
        self.topk = topk
        self.local_window = local_window
        self.use_shape_compat = use_shape_compat
        self.use_composition = use_composition
        self.embedder = MathEmbedder()

    def build(
        self,
        nodes: list[MathNode],
        z: np.ndarray | None = None,
        env: dict[str, tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        mask, _ = self.build_detailed(nodes, z, env)
        return mask

    def build_detailed(
        self,
        nodes: list[MathNode],
        z: np.ndarray | None = None,
        env: dict[str, tuple[int, ...]] | None = None,
    ) -> tuple[np.ndarray, MaskDiagnostics]:
        n = len(nodes)

        if z is None and self.topk > 0:
            z = self.embedder.encode_batch(nodes)

        sym   = symbolic_dependency_matrix(nodes)
        sameop = same_operator_matrix(nodes)
        topk_m = (
            embedding_topk_matrix(z, self.topk)
            if z is not None and self.topk > 0
            else np.zeros((n, n), dtype=bool)
        )
        local  = local_window_matrix(n, self.local_window)
        ident  = identity_matrix(n)
        sc     = shape_compatibility_matrix(nodes, env) if self.use_shape_compat and env else np.zeros((n, n), dtype=bool)
        comp   = composition_matrix(nodes, env) if self.use_composition and env else np.zeros((n, n), dtype=bool)

        mask = sym | sameop | topk_m | local | ident | sc | comp

        per_row_k = mask.sum(axis=1)
        max_k = int(per_row_k.max()) if n > 0 else 0
        avg_k = float(per_row_k.mean()) if n > 0 else 0.0
        allowed = int(mask.sum())
        padding_ratio = 1.0 - (allowed / (n * max_k)) if max_k > 0 else 0.0

        diag = MaskDiagnostics(
            n=n,
            full_edges=n * n,
            allowed_edges=allowed,
            sparsity_ratio=allowed / (n * n) if n > 0 else 0.0,
            relation_reduction=1.0 - allowed / (n * n) if n > 0 else 0.0,
            avg_k=avg_k,
            max_k=max_k,
            padding_ratio=padding_ratio,
            by_relation={
                "symbolic_dependency": int(sym.sum()),
                "same_operator":       int(sameop.sum()),
                "embedding_topk":      int(topk_m.sum()),
                "local_window":        int(local.sum()),
                "shape_compat":        int(sc.sum()),
                "composition":         int(comp.sum()),
                "identity":            int(ident.sum()),
            },
        )
        return mask, diag

    def sparsity_ratio(self, mask: np.ndarray) -> float:
        n = mask.shape[0]
        return float(mask.sum()) / (n * n) if n > 0 else 0.0

    def edge_count(self, mask: np.ndarray) -> int:
        return int(mask.sum())
