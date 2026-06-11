"""
Bounded Kleene closure over active/candidate graph slices.

A^{<=H} = I ∨ A ∨ A^2 ∨ ... ∨ A^H

Three modes unified under one path algebra (plan v22 P1.1):
  boolean  — OR/AND reachability / permission
  cost     — min-plus cheapest bounded path
  utility  — max-plus highest-utility bounded path

The three modes are instances of QuantaleSpec. Use quantale_bounded_closure()
for the unified path; the legacy mode-string API (bounded_closure) is preserved
for backwards compatibility.

Do NOT compute full closure over G_world at runtime.
Only bounded/local A^{<=H} over active/candidate slices is allowed (gate 13).
Gate 5: each spec must define join, compose, better, and valid explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

ClosureMode = Literal["boolean", "cost", "utility"]

_LARGE = 1e9
_NEG = -_LARGE


# ── QuantaleSpec — unified path algebra (P1.1) ────────────────────────────────

class QuantaleSpec:
    """
    Algebraic specification for one bounded closure path (plan v22 P1.1).

    Gate 5: join, compose, better, and valid must be defined explicitly.

    join:    element-wise merge of alternative paths  (OR / min / max)
    compose: sequential composition for matrix step   (AND / + / + with guard)
    reduce:  join-reduction over intermediate nodes   (any / min / max on axis 1)
    better:  scalar ranking comparator                (a preferred over b)
    valid:   scalar reachability predicate            (x is a usable path value)
    zero:    sentinel for unreachable/impossible paths
    one:     identity weight for zero-length paths
    """

    def __init__(
        self,
        name: str,
        zero: Any,
        one: Any,
        join: Callable,
        compose: Callable,
        reduce: Callable,
        better: Callable,
        valid: Callable,
    ) -> None:
        self.name = name
        self.zero = zero
        self.one = one
        self._join = join
        self._compose = compose
        self._reduce = reduce
        self._better = better
        self._valid = valid

    def join(self, a: Any, b: Any) -> Any:
        return self._join(a, b)

    def compose(self, P_col: np.ndarray, A_row: np.ndarray) -> np.ndarray:
        """
        Compose power slice P_col (n,n,1) with adj slice A_row (1,n,n).
        Returns (n,n,n) composed values for one matrix-product step.
        """
        return self._compose(P_col, A_row)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        """Join-reduce (n,n,n) array over axis=1 → (n,n)."""
        return self._reduce(arr)

    def better(self, a: Any, b: Any) -> bool:
        return bool(self._better(a, b))

    def valid(self, x: Any) -> bool:
        return bool(self._valid(x))

    def __repr__(self) -> str:
        return f"QuantaleSpec(name={self.name!r})"


# Built-in specs (gate 5: join/compose/better/valid defined explicitly)

BOOLEAN_SPEC = QuantaleSpec(
    name="boolean",
    zero=False,
    one=True,
    join=np.logical_or,
    compose=lambda P, A: np.logical_and(P, A),
    reduce=lambda arr: np.any(arr, axis=1),
    better=lambda a, b: bool(a) and not bool(b),
    valid=bool,
)

COST_SPEC = QuantaleSpec(
    name="cost",
    zero=_LARGE,
    one=0.0,
    join=np.minimum,
    compose=lambda P, A: P + A,
    reduce=lambda arr: np.min(arr, axis=1),
    better=lambda a, b: float(a) < float(b),
    valid=lambda x: float(x) < _LARGE,
)

UTILITY_SPEC = QuantaleSpec(
    name="utility",
    zero=_NEG,
    one=0.0,
    join=np.maximum,
    compose=lambda P, A: np.where(
        (P < _NEG / 2) | (A < _NEG / 2), _NEG, P + A
    ),
    reduce=lambda arr: np.max(arr, axis=1),
    better=lambda a, b: float(a) > float(b),
    valid=lambda x: float(x) > _NEG / 2,
)

_SPEC_MAP: dict[str, QuantaleSpec] = {
    "boolean": BOOLEAN_SPEC,
    "cost": COST_SPEC,
    "utility": UTILITY_SPEC,
}


# ── Generic bounded closure via QuantaleSpec ──────────────────────────────────

def quantale_closure(adj: np.ndarray, h: int, spec: QuantaleSpec) -> np.ndarray:
    """
    A^{<=H} bounded Kleene closure under spec.

    Computes reach = I ⊕ A ⊕ A² ⊕ ... ⊕ A^H
    where ⊕ = spec.join, and ⊗ (matrix product) uses spec.compose / spec.reduce.

    This is the single generic closure path that unifies boolean, cost, and
    utility modes (plan v22 P1.1).
    """
    n = adj.shape[0]
    if spec.name == "boolean":
        dtype: Any = bool
    else:
        dtype = np.float64

    reach = np.full((n, n), spec.zero, dtype=dtype)
    np.fill_diagonal(reach, spec.one)
    power = reach.copy()

    for _ in range(h):
        # power_new[i,j] = reduce_k( compose(power[i,k], adj[k,j]) )
        composed = spec.compose(power[:, :, None], adj[None, :, :])  # (n,n,n)
        power = spec.reduce(composed)                                  # (n,n)
        reach = spec.join(reach, power)                                # (n,n)

    return reach


def quantale_bounded_closure(
    node_ids: list[str],
    edges: list[tuple[str, str, float]],
    h: int,
    spec: QuantaleSpec,
) -> "ClosureResult":
    """
    High-level bounded closure via explicit QuantaleSpec (P1.1).

    Equivalent to bounded_closure(..., mode=spec.name) but accepts a QuantaleSpec
    directly, enabling custom or extended algebras beyond the three built-ins.
    """
    if h < 1:
        raise ValueError(f"h must be ≥ 1, got {h}")

    n = len(node_ids)
    if n == 0:
        empty: np.ndarray = np.zeros((0, 0), dtype=bool if spec.name == "boolean" else np.float64)
        return ClosureResult(mode=spec.name, h=h, n=0, reachability=empty,
                             node_ids=[], path_count=0, compute_ops=0)

    idx = {nid: i for i, nid in enumerate(node_ids)}

    if spec.name == "boolean":
        A = np.full((n, n), spec.zero, dtype=bool)
        for src, dst, _ in edges:
            if src in idx and dst in idx:
                A[idx[src], idx[dst]] = True
    else:
        A = np.full((n, n), spec.zero, dtype=np.float64)
        # cost: keep minimum weight; utility: keep maximum weight
        is_cost = spec.name == "cost"
        for src, dst, w in edges:
            if src in idx and dst in idx:
                si, di = idx[src], idx[dst]
                A[si, di] = min(A[si, di], w) if is_cost else max(A[si, di], w)

    reach = quantale_closure(A, h, spec)

    # Count valid pairs, excluding self-loops
    if spec.name == "boolean":
        path_count = int(reach.sum()) - n
    elif spec.name == "cost":
        path_count = max(int((reach < _LARGE).sum()) - n, 0)
    else:
        path_count = max(int((reach > _NEG / 2).sum()) - n, 0)

    ops = (n * n * h) if spec.name == "boolean" else (n * n * n * h)
    return ClosureResult(mode=spec.name, h=h, n=n, reachability=reach,
                         node_ids=list(node_ids), path_count=path_count, compute_ops=ops)


@dataclass
class ClosureResult:
    mode: str
    h: int
    n: int
    reachability: np.ndarray   # (n, n): bool for boolean; float for cost/utility
    node_ids: list[str]        # index → node_id mapping
    path_count: int            # reachable pairs excluding self-loops
    compute_ops: int           # approximate op count for cost tracking


# ── Core matrix routines ──────────────────────────────────────────────────────

def boolean_closure(adj: np.ndarray, h: int) -> np.ndarray:
    """
    A^{<=H} boolean reachability.
    adj: (n, n) bool or 0/1.
    Returns (n, n) bool: reach[i,j]=True iff j reachable from i in ≤ H hops.
    """
    n = adj.shape[0]
    A = adj.astype(bool)
    reach = np.eye(n, dtype=bool)
    power = np.eye(n, dtype=bool)
    for _ in range(h):
        # Boolean matrix product: next[i,j] = any_k(power[i,k] and A[k,j])
        power = power @ A
        power = power.astype(bool)
        reach = reach | power
    return reach


def cost_closure(adj_cost: np.ndarray, h: int) -> np.ndarray:
    """
    Bounded shortest-path: minimum cost to reach j from i in ≤ H hops.
    adj_cost: (n, n) float, _LARGE if no direct edge.
    Returns (n, n) float; unreachable pairs have value _LARGE.
    """
    n = adj_cost.shape[0]
    d = np.full((n, n), _LARGE, dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    d = np.minimum(d, adj_cost)
    prev = d.copy()
    for _ in range(h - 1):
        # d[i][j] = min_k(prev[i][k] + adj_cost[k][j])
        relaxed = prev[:, :, None] + adj_cost[None, :, :]   # (n, n, n)
        one_more = relaxed.min(axis=1)                        # (n, n)
        prev = one_more
        d = np.minimum(d, one_more)
    return d


def utility_closure(adj_util: np.ndarray, h: int) -> np.ndarray:
    """
    Bounded max-utility path: highest utility to reach j from i in ≤ H hops.
    adj_util: (n, n) float, -_LARGE if no direct edge.
    Returns (n, n) float; unreachable pairs have value -_LARGE.
    """
    n = adj_util.shape[0]
    NEG = -_LARGE
    u = np.full((n, n), NEG, dtype=np.float64)
    np.fill_diagonal(u, 0.0)
    u = np.maximum(u, adj_util)
    prev = u.copy()
    for _ in range(h - 1):
        # max-plus: u[i][j] = max_k(prev[i][k] + adj_util[k][j])
        relaxed = prev[:, :, None] + adj_util[None, :, :]    # (n, n, n)
        # Mask -inf paths to avoid -inf + finite = finite confusion
        mask = (prev[:, :, None] < NEG / 2) | (adj_util[None, :, :] < NEG / 2)
        relaxed = np.where(mask, NEG, relaxed)
        one_more = relaxed.max(axis=1)
        prev = one_more
        u = np.maximum(u, one_more)
    return u


# ── Public API ────────────────────────────────────────────────────────────────

def bounded_closure(
    node_ids: list[str],
    edges: list[tuple[str, str, float]],
    h: int,
    mode: ClosureMode = "boolean",
) -> ClosureResult:
    """
    Compute A^{<=H} bounded Kleene closure over a graph slice.

    node_ids: ordered list of node IDs in the slice (G_t ∪ C_t)
    edges:    list of (src_id, dst_id, weight) within the slice
    h:        closure horizon H
    mode:     "boolean" | "cost" | "utility"

    Returns ClosureResult with reachability matrix indexed by node_ids order.
    """
    if h < 1:
        raise ValueError(f"h must be ≥ 1, got {h}")

    n = len(node_ids)
    if n == 0:
        empty = np.zeros((0, 0), dtype=bool)
        return ClosureResult(mode=mode, h=h, n=0, reachability=empty, node_ids=[], path_count=0, compute_ops=0)

    idx = {nid: i for i, nid in enumerate(node_ids)}

    if mode == "boolean":
        A = np.zeros((n, n), dtype=bool)
        for src, dst, _ in edges:
            if src in idx and dst in idx:
                A[idx[src], idx[dst]] = True
        reach = boolean_closure(A, h)
        path_count = int(reach.sum()) - n
        ops = n * n * h
        return ClosureResult(mode=mode, h=h, n=n, reachability=reach,
                             node_ids=list(node_ids), path_count=path_count, compute_ops=ops)

    if mode == "cost":
        A = np.full((n, n), _LARGE, dtype=np.float64)
        for src, dst, w in edges:
            if src in idx and dst in idx:
                si, di = idx[src], idx[dst]
                A[si, di] = min(A[si, di], w)
        reach = cost_closure(A, h)
        path_count = int((reach < _LARGE).sum()) - n
        ops = n * n * n * h
        return ClosureResult(mode=mode, h=h, n=n, reachability=reach,
                             node_ids=list(node_ids), path_count=max(path_count, 0), compute_ops=ops)

    if mode == "utility":
        NEG = -_LARGE
        A = np.full((n, n), NEG, dtype=np.float64)
        for src, dst, w in edges:
            if src in idx and dst in idx:
                si, di = idx[src], idx[dst]
                A[si, di] = max(A[si, di], w)
        reach = utility_closure(A, h)
        path_count = int((reach > NEG / 2).sum()) - n
        ops = n * n * n * h
        return ClosureResult(mode=mode, h=h, n=n, reachability=reach,
                             node_ids=list(node_ids), path_count=max(path_count, 0), compute_ops=ops)

    raise ValueError(f"Unknown closure mode: {mode!r}")


def closure_critical_edges(
    node_ids: list[str],
    edges: list[tuple[str, str, float]],
    h: int,
    mode: ClosureMode = "boolean",
) -> set[tuple[str, str]]:
    """
    Identify edges whose removal reduces bounded closure reachability.
    Used by P2.2 closure-preserving edge deletion gate.
    Returns set of (src_id, dst_id) pairs for critical edges.
    """
    if not edges:
        return set()

    base = bounded_closure(node_ids, edges, h, mode)
    critical: set[tuple[str, str]] = set()

    for src, dst, _ in edges:
        without = [(s, d, w) for s, d, w in edges if not (s == src and d == dst)]
        result = bounded_closure(node_ids, without, h, mode)

        if mode == "boolean":
            if int(result.reachability.sum()) < int(base.reachability.sum()):
                critical.add((src, dst))
        elif mode == "cost":
            # Any pair becomes strictly harder to reach?
            if np.any(result.reachability > base.reachability + 1e-9):
                critical.add((src, dst))
        elif mode == "utility":
            # Any pair loses utility?
            if np.any(result.reachability < base.reachability - 1e-9):
                critical.add((src, dst))

    return critical
