from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
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


# ── Relation weights for scored top-K routing ────────────────────────────────
# Lower same_operator weight prevents it from dominating; continuous cosine sim
# replaces the binary embedding_topk for finer discrimination.

RELATION_WEIGHTS: dict[str, float] = {
    "identity":            10.0,
    "symbolic_dependency":  1.0,
    "composition":          0.9,
    "shape_compat":         0.8,
    "embedding":            0.6,   # applied to continuous cosine similarity
    "local_window":         0.4,
    "same_operator":        0.1,
}


def build_scored_topk_mask(
    nodes: list[MathNode],
    z: np.ndarray | None = None,
    env: dict[str, tuple[int, ...]] | None = None,
    fixed_k: int = 32,
    local_window: int = 2,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, "MaskDiagnostics"]:
    """
    Score-based top-K mask: each row keeps the fixed_k highest-scoring neighbors.

    Scoring combines binary relation membership (weighted) with continuous cosine
    similarity so that same_operator (w=0.1) cannot overwhelm symbolic/composition
    edges (w=0.9–1.0). Returns (bool mask, MaskDiagnostics).
    avg_k ≈ fixed_k regardless of n — breaks the k∝n quadratic scaling.
    """
    w = {**RELATION_WEIGHTS, **(weights or {})}
    embedder = MathEmbedder()
    n = len(nodes)

    if z is None:
        z = embedder.encode_batch(nodes)

    # Build individual relation matrices (used for both scoring and by_relation)
    ident   = identity_matrix(n)
    sym     = symbolic_dependency_matrix(nodes)
    comp    = composition_matrix(nodes, env)
    sc      = shape_compatibility_matrix(nodes, env)
    local   = local_window_matrix(n, local_window)
    sameop  = same_operator_matrix(nodes)

    # Continuous cosine similarity (replaces binary embedding_topk in scoring)
    cos_sim = pairwise_cosine(z).clip(0, 1).astype(np.float32)
    np.fill_diagonal(cos_sim, 0.0)

    # Weighted score matrix
    score = np.zeros((n, n), dtype=np.float32)
    score += w["identity"]            * ident.astype(np.float32)
    score += w["symbolic_dependency"] * sym.astype(np.float32)
    score += w["composition"]         * comp.astype(np.float32)
    score += w["shape_compat"]        * sc.astype(np.float32)
    score += w["local_window"]        * local.astype(np.float32)
    score += w["same_operator"]       * sameop.astype(np.float32)
    score += w["embedding"]           * cos_sim

    # Top-K selection per row
    k = min(fixed_k, n)
    mask = np.zeros((n, n), dtype=bool)
    if n > 0 and k > 0:
        top_idx = np.argpartition(score, -k, axis=1)[:, -k:] if k < n else np.tile(np.arange(n), (n, 1))
        mask[np.arange(n)[:, None], top_idx] = True
    np.fill_diagonal(mask, True)   # identity always included

    per_row_k = mask.sum(axis=1)
    max_k_val = int(per_row_k.max()) if n > 0 else 0
    avg_k     = float(per_row_k.mean()) if n > 0 else 0.0
    allowed   = int(mask.sum())

    diag = MaskDiagnostics(
        n=n,
        full_edges=n * n,
        allowed_edges=allowed,
        sparsity_ratio=allowed / (n * n) if n > 0 else 0.0,
        relation_reduction=1.0 - allowed / (n * n) if n > 0 else 0.0,
        avg_k=avg_k,
        max_k=max_k_val,
        padding_ratio=1.0 - (allowed / (n * max_k_val)) if max_k_val > 0 else 0.0,
        by_relation={
            "symbolic_dependency": int((mask & sym).sum()),
            "composition":         int((mask & comp).sum()),
            "shape_compat":        int((mask & sc).sum()),
            "local_window":        int((mask & local).sum()),
            "same_operator":       int((mask & sameop).sum()),
            "identity":            int((mask & ident).sum()),
            "embedding_topk":      0,
        },
    )
    return mask, diag


def build_scored_topk_mask_torch(
    nodes: list[MathNode],
    Z_t: torch.Tensor | None,
    env: dict[str, tuple[int, ...]] | None,
    fixed_k: int,
    local_window: int,
    weights: dict[str, float] | None,
    device: torch.device,
) -> tuple[torch.Tensor, "MaskDiagnostics"]:
    """GPU-resident scored top-K mask. Mirrors build_scored_topk_mask."""
    w = {**RELATION_WEIGHTS, **(weights or {})}
    n = len(nodes)

    ident   = identity_matrix_torch(n, device).float()
    sym     = symbolic_dependency_matrix_torch(nodes, device).float()
    comp    = composition_matrix_torch(nodes, env, device).float()
    sc      = shape_compat_matrix_torch(nodes, env, device).float()
    local   = local_window_matrix_torch(n, local_window, device).float()
    sameop  = same_operator_matrix_torch(nodes, device).float()

    score = (
        w["identity"]            * ident
        + w["symbolic_dependency"] * sym
        + w["composition"]         * comp
        + w["shape_compat"]        * sc
        + w["local_window"]        * local
        + w["same_operator"]       * sameop
    )

    if Z_t is not None:
        Z = Z_t.to(device)
        norms = Z.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_sim = (Z / norms) @ (Z / norms).T
        cos_sim = cos_sim.clamp(0, 1)
        cos_sim.fill_diagonal_(0.0)
        score = score + w["embedding"] * cos_sim

    k = min(fixed_k, n)
    topk_idx = score.topk(k, dim=1).indices  # (n, k)
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    rows = torch.arange(n, device=device).unsqueeze(1).expand_as(topk_idx)
    mask[rows.reshape(-1), topk_idx.reshape(-1)] = True
    mask.fill_diagonal_(True)

    per_row_k = mask.sum(dim=1)
    max_k_val = int(per_row_k.max().item()) if n > 0 else 0
    avg_k     = float(per_row_k.float().mean().item()) if n > 0 else 0.0
    allowed   = int(mask.sum().item())

    diag = MaskDiagnostics(
        n=n,
        full_edges=n * n,
        allowed_edges=allowed,
        sparsity_ratio=allowed / (n * n) if n > 0 else 0.0,
        relation_reduction=1.0 - allowed / (n * n) if n > 0 else 0.0,
        avg_k=avg_k,
        max_k=max_k_val,
        padding_ratio=1.0 - (allowed / (n * max_k_val)) if max_k_val > 0 else 0.0,
        by_relation={
            "symbolic_dependency": int((mask & sym.bool()).sum().item()),
            "composition":         int((mask & comp.bool()).sum().item()),
            "shape_compat":        int((mask & sc.bool()).sum().item()),
            "local_window":        int((mask & local.bool()).sum().item()),
            "same_operator":       int((mask & sameop.bool()).sum().item()),
            "identity":            int((mask & ident.bool()).sum().item()),
            "embedding_topk":      0,
        },
    )
    return mask, diag


# ── TopologyBuilder ───────────────────────────────────────────────────────────

class TopologyBuilder:
    """
    Builds boolean attention masks from multiple relation types.

    topology_mode="union"       — OR of all relation matrices (v1–v5 behaviour)
    topology_mode="scored_topk" — weighted score → top-K per row (v6, O(nK))
    """

    def __init__(
        self,
        topk: int = 4,
        local_window: int = 2,
        use_shape_compat: bool = True,
        use_composition: bool = True,
        topology_mode: str = "union",
        fixed_k: int = 32,
        relation_weights: dict[str, float] | None = None,
    ) -> None:
        self.topk = topk
        self.local_window = local_window
        self.use_shape_compat = use_shape_compat
        self.use_composition = use_composition
        self.topology_mode = topology_mode
        self.fixed_k = fixed_k
        self.relation_weights = relation_weights
        self.embedder = MathEmbedder()

    def build(
        self,
        nodes: list[MathNode],
        z: np.ndarray | None = None,
        env: dict[str, tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        if self.topology_mode == "scored_topk":
            mask, _ = self.build_scored_topk(nodes, z, env)
            return mask
        mask, _ = self.build_detailed(nodes, z, env)
        return mask

    def build_scored_topk(
        self,
        nodes: list[MathNode],
        z: np.ndarray | None = None,
        env: dict[str, tuple[int, ...]] | None = None,
    ) -> tuple[np.ndarray, MaskDiagnostics]:
        if z is None:
            z = self.embedder.encode_batch(nodes)
        return build_scored_topk_mask(
            nodes, z, env,
            fixed_k=self.fixed_k,
            local_window=self.local_window,
            weights=self.relation_weights,
        )

    def build_scored_topk_torch(
        self,
        nodes: list[MathNode],
        Z_t: torch.Tensor | None = None,
        env: dict[str, tuple[int, ...]] | None = None,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, MaskDiagnostics]:
        device = torch.device(device)
        if Z_t is None:
            z = self.embedder.encode_batch(nodes)
            Z_t = torch.tensor(z, dtype=torch.float32, device=device)
        return build_scored_topk_mask_torch(
            nodes, Z_t, env,
            fixed_k=self.fixed_k,
            local_window=self.local_window,
            weights=self.relation_weights,
            device=device,
        )

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

    def build_detailed_torch(
        self,
        nodes: list[MathNode],
        Z_t: torch.Tensor | None = None,
        env: dict[str, tuple[int, ...]] | None = None,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, MaskDiagnostics]:
        """
        GPU-resident topology build.
        Returns (bool mask tensor on device, MaskDiagnostics).
        """
        n = len(nodes)
        device = torch.device(device)

        if Z_t is None and self.topk > 0:
            z_np = self.embedder.encode_batch(nodes)
            Z_t = torch.tensor(z_np, dtype=torch.float32, device=device)

        sym    = symbolic_dependency_matrix_torch(nodes, device)
        sameop = same_operator_matrix_torch(nodes, device)
        topk_m = (
            embedding_topk_matrix_torch(Z_t, self.topk, device)
            if Z_t is not None and self.topk > 0
            else torch.zeros(n, n, dtype=torch.bool, device=device)
        )
        local = local_window_matrix_torch(n, self.local_window, device)
        ident = identity_matrix_torch(n, device)
        sc    = (
            shape_compat_matrix_torch(nodes, env, device)
            if self.use_shape_compat and env
            else torch.zeros(n, n, dtype=torch.bool, device=device)
        )
        comp  = (
            composition_matrix_torch(nodes, env, device)
            if self.use_composition and env
            else torch.zeros(n, n, dtype=torch.bool, device=device)
        )

        mask = sym | sameop | topk_m | local | ident | sc | comp

        per_row_k = mask.sum(dim=1)
        max_k = int(per_row_k.max().item()) if n > 0 else 0
        avg_k = float(per_row_k.float().mean().item()) if n > 0 else 0.0
        allowed = int(mask.sum().item())
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
                "symbolic_dependency": int(sym.sum().item()),
                "same_operator":       int(sameop.sum().item()),
                "embedding_topk":      int(topk_m.sum().item()),
                "local_window":        int(local.sum().item()),
                "shape_compat":        int(sc.sum().item()),
                "composition":         int(comp.sum().item()),
                "identity":            int(ident.sum().item()),
            },
        )
        return mask, diag


# ── GPU-native relation matrix builders ──────────────────────────────────────

def identity_matrix_torch(n: int, device: torch.device) -> torch.Tensor:
    return torch.eye(n, dtype=torch.bool, device=device)


def local_window_matrix_torch(n: int, w: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= w


def same_operator_matrix_torch(nodes: list[MathNode], device: torch.device) -> torch.Tensor:
    unique_ops = list(dict.fromkeys(nd.op for nd in nodes))
    op_to_id = {op: i for i, op in enumerate(unique_ops)}
    op_ids = torch.tensor([op_to_id[nd.op] for nd in nodes], dtype=torch.long, device=device)
    mat = op_ids.unsqueeze(0) == op_ids.unsqueeze(1)
    mat.fill_diagonal_(False)
    return mat


def symbolic_dependency_matrix_torch(
    nodes: list[MathNode], device: torch.device
) -> torch.Tensor:
    n = len(nodes)
    mat = torch.zeros(n, n, dtype=torch.bool, device=device)
    rows: list[int] = []
    cols: list[int] = []
    for i, node in enumerate(nodes):
        for child in node.args:
            for j, other in enumerate(nodes):
                if other is child or other == child:
                    rows.extend([i, j])
                    cols.extend([j, i])
    if rows:
        r = torch.tensor(rows, dtype=torch.long, device=device)
        c = torch.tensor(cols, dtype=torch.long, device=device)
        mat[r, c] = True
    return mat


def embedding_topk_matrix_torch(
    Z: torch.Tensor, k: int, device: torch.device
) -> torch.Tensor:
    n = Z.shape[0]
    mat = torch.zeros(n, n, dtype=torch.bool, device=device)
    if k <= 0 or n <= 1:
        return mat
    Z = Z.to(device)
    norms = Z.norm(dim=1, keepdim=True).clamp(min=1e-8)
    Zn = Z / norms
    sims = Zn @ Zn.T  # (n, n) cosine similarity
    sims.fill_diagonal_(-2.0)
    k_actual = min(k, n - 1)
    topk_idx = sims.topk(k_actual, dim=1).indices  # (n, k_actual)
    rows = torch.arange(n, device=device).unsqueeze(1).expand_as(topk_idx)
    mat[rows.reshape(-1), topk_idx.reshape(-1)] = True
    mat = mat | mat.T
    return mat


def shape_compat_matrix_torch(
    nodes: list[MathNode],
    env: dict[str, tuple[int, ...]] | None,
    device: torch.device,
) -> torch.Tensor:
    n = len(nodes)
    mat = torch.zeros(n, n, dtype=torch.bool, device=device)
    if not env:
        return mat
    try:
        from .shape import infer_shape
    except ImportError:
        return mat
    shapes = [infer_shape(nd, env) for nd in nodes]
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        if shapes[i] is None:
            continue
        for j in range(i + 1, n):
            if shapes[j] is not None and shapes[i] == shapes[j]:
                rows.extend([i, j])
                cols.extend([j, i])
    if rows:
        mat[
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(cols, dtype=torch.long, device=device),
        ] = True
    return mat


def composition_matrix_torch(
    nodes: list[MathNode],
    env: dict[str, tuple[int, ...]] | None,
    device: torch.device,
) -> torch.Tensor:
    n = len(nodes)
    mat = torch.zeros(n, n, dtype=torch.bool, device=device)
    if not env:
        return mat
    try:
        from .shape import infer_shape
    except ImportError:
        return mat
    out_shapes = [infer_shape(nd, env) for nd in nodes]
    rows: list[int] = []
    cols: list[int] = []
    for j, node_j in enumerate(nodes):
        for arg in node_j.args:
            arg_shape = infer_shape(arg, env)
            if arg_shape is None:
                continue
            for i in range(n):
                if i == j:
                    continue
                if out_shapes[i] is not None and out_shapes[i] == arg_shape:
                    rows.extend([i, j])
                    cols.extend([j, i])
    if rows:
        mat[
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(cols, dtype=torch.long, device=device),
        ] = True
    return mat


def build_priority_matrix_torch(
    nodes: list[MathNode],
    Z_t: torch.Tensor | None = None,
    env: dict[str, tuple[int, ...]] | None = None,
    topk: int = 4,
    local_window: int = 2,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """GPU-resident priority matrix. Returns (n, n) int8 tensor on `device`."""
    n = len(nodes)
    device = torch.device(device)

    if Z_t is None and topk > 0:
        embedder = MathEmbedder()
        z_np = embedder.encode_batch(nodes)
        Z_t = torch.tensor(z_np, dtype=torch.float32, device=device)

    layers = [
        ("identity",            identity_matrix_torch(n, device)),
        ("symbolic_dependency", symbolic_dependency_matrix_torch(nodes, device)),
        ("composition",         composition_matrix_torch(nodes, env, device)),
        ("shape_compat",        shape_compat_matrix_torch(nodes, env, device)),
        ("embedding_topk",
            embedding_topk_matrix_torch(Z_t, topk, device)
            if Z_t is not None and topk > 0
            else torch.zeros(n, n, dtype=torch.bool, device=device)),
        ("local_window",        local_window_matrix_torch(n, local_window, device)),
        ("same_operator",       same_operator_matrix_torch(nodes, device)),
    ]

    priority = torch.zeros(n, n, dtype=torch.int8, device=device)
    for name, mat in layers:
        p = RELATION_PRIORITY[name]
        unset = (priority == 0) & mat
        existing = (priority > 0) & (priority > p) & mat
        priority[unset] = p
        priority[existing] = p

    return priority
