"""
Frontier planner for Recurrent Frontier Graph Transformer.

Implements:
  P1.3 — frontier extraction (∂G_t)
  P1.4 — boundary candidate scorer
  P1.6 — closure-aware frontier planner vs one-hop TopK baseline
  P1.7 — KeepTopB pruning

Plan formula:
  H_t = F_θ^L(X_t, G_t)
  C_t = BoundaryCandidates(∂G_t, G_world)
  G_{t+1} = KeepTopB(G_t ∪ TopK(Score(H_t, C_t)))
  G_world^{t+1} = Accept(G_world^t ⊕ ΔG_t)

Closure-aware claims must compare against one-hop/heuristic TopK under
the same K, L, T_outer, B, H, evaluator (non-negotiable gate 15).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .world_graph import WorldGraph, ActiveGraph, NodeRecord, EdgeRecord
from .graph_closure import (
    bounded_closure,
    ClosureMode,
    QuantaleSpec,
    quantale_bounded_closure,
    BOOLEAN_SPEC,
    COST_SPEC,
    UTILITY_SPEC,
    _SPEC_MAP,
)

_FEAT_DIM = 8


@dataclass
class FrontierExpansionResult:
    """Metrics from one frontier expansion step (one T_outer iteration)."""
    t_outer: int
    planner: str
    active_node_count: int
    active_edge_count: int
    frontier_node_count: int
    candidate_count: int
    added_node_count: int
    pruned_node_count: int
    closure_path_count: int = 0
    closure_compute_cost: int = 0
    closure_critical_preserved: int = 0
    stale_noisy_edge_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ── P1.3: Frontier extraction ─────────────────────────────────────────────────

def extract_frontier(active: ActiveGraph, world: WorldGraph) -> frozenset[str]:
    """∂G_t: active nodes with at least one neighbor outside G_t."""
    return active.frontier(world)


def boundary_candidates(active: ActiveGraph, world: WorldGraph) -> list[str]:
    """C_t: world-graph neighbors of ∂G_t not currently in G_t."""
    return active.boundary_candidates(world)


# ── P1.4: Candidate scoring ───────────────────────────────────────────────────

def _node_feat(node: NodeRecord | None) -> np.ndarray:
    """Minimal feature vector for graph-structure scoring."""
    if node is None:
        return np.zeros(_FEAT_DIM, dtype=np.float32)
    f = node.features
    return np.array([
        float(f.get("arity", 0)),
        float(f.get("depth", 0)),
        float(f.get("subtree_size", 0)),
        float(f.get("weight", node.__class__.__dict__.get("weight", 0.0))),
        1.0 if node.node_kind == "add" else 0.0,
        1.0 if node.node_kind == "matmul" else 0.0,
        1.0 if node.node_kind == "affine" else 0.0,
        1.0 if node.node_kind in ("var", "const") else 0.0,
    ], dtype=np.float32)


def score_candidates_onehop(
    h_t: np.ndarray | None,
    active: ActiveGraph,
    candidates: list[str],
    world: WorldGraph,
    k: int,
) -> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    """
    One-hop TopK baseline scorer.

    Scores each candidate by:
      - Cosine similarity of its feature vector to the mean active representation
      - Bonus from existing edge weights ω_ij to active nodes

    h_t: (|G_t|, d) hidden states from F_θ^L, or None.
         If d == _FEAT_DIM, used as anchor; otherwise ignored.
    Returns sorted [(node_id, score)] descending, at most k entries.
    """
    if not candidates:
        return []

    cand_recs = [world.get_node(c) for c in candidates]
    cand_vecs = np.stack([_node_feat(r) for r in cand_recs])  # (|C|, _FEAT_DIM)

    if h_t is not None and h_t.ndim == 2 and h_t.shape[0] > 0 and h_t.shape[1] == _FEAT_DIM:
        anchor = h_t.mean(axis=0)
    else:
        # Fall back to mean feature of active nodes
        active_recs = [world.get_node(nid) for nid in active.node_ids]
        active_vecs = np.stack([_node_feat(r) for r in active_recs])
        anchor = active_vecs.mean(axis=0) if len(active_vecs) > 0 else np.zeros(_FEAT_DIM, dtype=np.float32)

    anchor_norm = np.linalg.norm(anchor) + 1e-8
    cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8
    scores = (cand_vecs / cand_norms) @ (anchor / anchor_norm)  # (|C|,)

    # Add edge weight bonus from world graph
    for i, cid in enumerate(candidates):
        for nid in active.node_ids:
            for e in world.edges_between(nid, cid) + world.edges_between(cid, nid):
                scores[i] += e.weight

    k_actual = min(k, len(candidates))
    top_idx = np.argpartition(scores, -k_actual)[-k_actual:]
    return sorted([(candidates[i], float(scores[i])) for i in top_idx], key=lambda x: -x[1])


def score_candidates_closure(
    h_t: np.ndarray | None,
    active: ActiveGraph,
    candidates: list[str],
    world: WorldGraph,
    k: int,
    h_horizon: int,
    mode: ClosureMode = "utility",
) -> list[tuple[str, float]]:
    """
    Closure-aware scorer.

    Builds A^{<=H} over G_t ∪ C_t and scores each candidate by its closure
    contribution to/from active nodes.

    For cost mode: lower cost → higher score (negated internally).
    Gate 15: results must later be compared against one-hop TopK under same K, H.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if h_horizon < 1:
        raise ValueError(f"h_horizon must be >= 1, got {h_horizon}")
    if not candidates:
        return []

    active_set = set(active.node_ids)
    slice_ids = list(active.node_ids) + [c for c in candidates if c not in active_set]
    slice_set = set(slice_ids)

    slice_edges: list[tuple[str, str, float]] = []
    for nid in slice_ids:
        for eid in world.out_edge_ids(nid):
            e = world.get_edge(eid)
            if e is not None and e.dst_id in slice_set:
                slice_edges.append((e.src_id, e.dst_id, e.weight))

    cr = bounded_closure(slice_ids, slice_edges, h_horizon, mode)
    reach = cr.reachability
    idx_map = {nid: i for i, nid in enumerate(slice_ids)}
    active_indices = [idx_map[a] for a in active_set if a in idx_map]

    scores: list[float] = []
    NEG = -1e9
    LARGE = 1e9
    for cid in candidates:
        if cid not in idx_map or not active_indices:
            scores.append(0.0)
            continue
        ci = idx_map[cid]
        if mode == "boolean":
            from_active = sum(bool(reach[ai, ci]) for ai in active_indices)
            to_active = sum(bool(reach[ci, ai]) for ai in active_indices)
            scores.append(float(from_active + to_active))
        elif mode == "cost":
            # Lower cost → better candidate: negate so argmax picks cheapest.
            from_active = sum(
                float(reach[ai, ci])
                for ai in active_indices
                if reach[ai, ci] < LARGE
            )
            to_active = sum(
                float(reach[ci, ai])
                for ai in active_indices
                if reach[ci, ai] < LARGE
            )
            scores.append(-(from_active + to_active))
        else:  # utility
            from_active = sum(
                float(reach[ai, ci])
                for ai in active_indices
                if reach[ai, ci] > NEG / 2
            )
            to_active = sum(
                float(reach[ci, ai])
                for ai in active_indices
                if reach[ci, ai] > NEG / 2
            )
            scores.append(from_active + to_active)

    k_actual = min(k, len(candidates))
    scores_arr = np.array(scores, dtype=np.float32)
    top_idx = np.argpartition(scores_arr, -k_actual)[-k_actual:]
    return sorted([(candidates[i], float(scores_arr[i])) for i in top_idx], key=lambda x: -x[1])


# ── P1.7: KeepTopB pruning ────────────────────────────────────────────────────

def keep_top_b(
    active: ActiveGraph,
    new_node_ids: list[str],
    world: WorldGraph,
    budget: int,
    anchor_ids: set[str] | None = None,
) -> ActiveGraph:
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")
    """
    KeepTopB: merge new nodes into active graph, then prune to budget B.

    Retention priority (highest = keep):
      1. anchor_ids — never pruned
      2. nodes with high edge weight ω_ij to anchors
      3. frontier nodes (have unloaded neighbors — preserves exploration)
      4. degree in world graph
    """
    merged = active.node_ids | frozenset(new_node_ids)
    if len(merged) <= budget:
        return active.with_node_set(merged)

    anchors = (anchor_ids or set()) | set(active.anchor_ids)
    merged_list = list(merged)

    priority: dict[str, float] = {}
    for nid in merged_list:
        if nid in anchors:
            priority[nid] = float("inf")
            continue
        degree = len(world.neighbors(nid))
        anchor_bonus = sum(
            e.weight + 1.0
            for anc in anchors
            for e in world.edges_between(nid, anc) + world.edges_between(anc, nid)
        )
        frontier_bonus = 2.0 if any(nb not in merged for nb in world.neighbors(nid)) else 0.0
        priority[nid] = degree + anchor_bonus * 3.0 + frontier_bonus

    kept = sorted(merged_list, key=lambda nid: -priority.get(nid, 0.0))[:budget]
    return active.with_node_set(frozenset(kept))


# ── P1.6: Planners ────────────────────────────────────────────────────────────

class OneHopTopKPlanner:
    """
    Baseline: one-hop TopK frontier expansion.

    Scores candidates from ∂G_t by cosine similarity + edge weight bonus.
    No closure computation — O(|C| * d_feat) per step.
    """

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.k = k

    def expand(
        self,
        active: ActiveGraph,
        world: WorldGraph,
        h_t: np.ndarray | None,
        t_outer: int,
    ) -> tuple[ActiveGraph, FrontierExpansionResult]:
        frontier = extract_frontier(active, world)
        candidates = boundary_candidates(active, world)
        scored = score_candidates_onehop(h_t, active, candidates, world, self.k)

        added_ids = [nid for nid, _ in scored]
        expanded = active.with_added(added_ids)
        new_active = keep_top_b(expanded, [], world, active.budget)

        pruned = max(0, len(active.node_ids) + len(added_ids) - len(new_active.node_ids))
        result = FrontierExpansionResult(
            t_outer=t_outer,
            planner="one_hop_topk",
            active_node_count=len(new_active.node_ids),
            active_edge_count=len(new_active.active_edges(world)),
            frontier_node_count=len(frontier),
            candidate_count=len(candidates),
            added_node_count=len(added_ids),
            pruned_node_count=pruned,
        )
        return new_active, result


class ClosureAwarePlanner:
    """
    Closure-aware frontier planner.

    Scores candidates using A^{<=H} over G_t ∪ C_t.
    O(|G_t ∪ C_t|^2 * H) per step for boolean; O(n^3 * H) for cost/utility.

    Gate 13: only bounded/local closure over active/candidate slice — never G_world.
    Gate 15: results must compare against OneHopTopKPlanner under same K, H, T_outer.
    """

    def __init__(
        self,
        k: int,
        h_horizon: int,
        closure_mode: ClosureMode = "utility",
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if h_horizon < 1:
            raise ValueError(f"h_horizon must be >= 1, got {h_horizon}")
        self.k = k
        self.h_horizon = h_horizon
        self.closure_mode = closure_mode

    def expand(
        self,
        active: ActiveGraph,
        world: WorldGraph,
        h_t: np.ndarray | None,
        t_outer: int,
    ) -> tuple[ActiveGraph, FrontierExpansionResult]:
        frontier = extract_frontier(active, world)
        candidates = boundary_candidates(active, world)
        scored = score_candidates_closure(
            h_t, active, candidates, world,
            self.k, self.h_horizon, self.closure_mode,
        )

        added_ids = [nid for nid, _ in scored]
        expanded = active.with_added(added_ids)
        new_active = keep_top_b(expanded, [], world, active.budget)

        # Compute closure stats on final active+candidate slice
        slice_ids = list(active.node_ids) + [c for c in candidates if c not in active.node_ids]
        slice_set = set(slice_ids)
        slice_edges: list[tuple[str, str, float]] = []
        for nid in slice_ids:
            for eid in world.out_edge_ids(nid):
                e = world.get_edge(eid)
                if e is not None and e.dst_id in slice_set:
                    slice_edges.append((e.src_id, e.dst_id, e.weight))
        cr = bounded_closure(slice_ids, slice_edges, self.h_horizon, self.closure_mode)

        pruned = max(0, len(active.node_ids) + len(added_ids) - len(new_active.node_ids))
        result = FrontierExpansionResult(
            t_outer=t_outer,
            planner=f"closure_{self.closure_mode}_H{self.h_horizon}",
            active_node_count=len(new_active.node_ids),
            active_edge_count=len(new_active.active_edges(world)),
            frontier_node_count=len(frontier),
            candidate_count=len(candidates),
            added_node_count=len(added_ids),
            pruned_node_count=pruned,
            closure_path_count=cr.path_count,
            closure_compute_cost=cr.compute_ops,
        )
        return new_active, result


# ── P1.2: Quantale-native scoring and planner ─────────────────────────────────

def score_candidates_quantale(
    h_t: np.ndarray | None,
    active: ActiveGraph,
    candidates: list[str],
    world: WorldGraph,
    k: int,
    h_horizon: int,
    spec: QuantaleSpec,
) -> list[tuple[str, float]]:
    """
    Quantale-native candidate scorer (P1.2).

    Uses spec.join / spec.compose / spec.valid / spec.better to rank candidates
    by their aggregate bounded path value to/from active nodes.

    For cost spec: lower total cost → higher score (negated internally).
    Gate 13: only bounded/local closure over active+candidate slice — never G_world.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if h_horizon < 1:
        raise ValueError(f"h_horizon must be >= 1, got {h_horizon}")
    if not candidates:
        return []

    active_set = set(active.node_ids)
    slice_ids = list(active.node_ids) + [c for c in candidates if c not in active_set]
    slice_set = set(slice_ids)

    slice_edges: list[tuple[str, str, float]] = []
    for nid in slice_ids:
        for eid in world.out_edge_ids(nid):
            e = world.get_edge(eid)
            if e is not None and e.dst_id in slice_set:
                slice_edges.append((e.src_id, e.dst_id, e.weight))

    cr = quantale_bounded_closure(slice_ids, slice_edges, h_horizon, spec)
    reach = cr.reachability
    idx_map = {nid: i for i, nid in enumerate(slice_ids)}
    active_indices = [idx_map[a] for a in active_set if a in idx_map]

    scores: list[float] = []
    for cid in candidates:
        if cid not in idx_map or not active_indices:
            scores.append(0.0)
            continue
        ci = idx_map[cid]
        from_active = sum(float(reach[ai, ci]) for ai in active_indices if spec.valid(reach[ai, ci]))
        to_active = sum(float(reach[ci, ai]) for ai in active_indices if spec.valid(reach[ci, ai]))
        raw = from_active + to_active
        # For cost: lower total cost → negate so argmax picks cheapest path.
        scores.append(-raw if spec.name == "cost" else raw)

    k_actual = min(k, len(candidates))
    scores_arr = np.array(scores, dtype=np.float32)
    top_idx = np.argpartition(scores_arr, -k_actual)[-k_actual:]
    return sorted([(candidates[i], float(scores_arr[i])) for i in top_idx], key=lambda x: -x[1])


class QuantaleFrontierPlanner:
    """
    Quantale-native frontier planner (P1.2).

    Uses a QuantaleSpec to drive candidate scoring via A^{<=H} bounded closure.
    The spec encodes join, compose, better, and valid for the path algebra.

    Acceptance gate: results must be compared against OneHopTopKPlanner and
    ClosureAwarePlanner under identical K, L, B, H, T_outer, evaluator (gate 15 / v22).

    Gate 13: only bounded/local closure over active+candidate slice — never G_world.
    """

    def __init__(
        self,
        k: int,
        h_horizon: int,
        spec: QuantaleSpec | str = "utility",
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if h_horizon < 1:
            raise ValueError(f"h_horizon must be >= 1, got {h_horizon}")
        if isinstance(spec, str):
            if spec not in _SPEC_MAP:
                raise ValueError(f"Unknown quantale spec name: {spec!r}; choose from {list(_SPEC_MAP)}")
            spec = _SPEC_MAP[spec]
        self.k = k
        self.h_horizon = h_horizon
        self.spec = spec

    def expand(
        self,
        active: ActiveGraph,
        world: WorldGraph,
        h_t: np.ndarray | None,
        t_outer: int,
    ) -> tuple[ActiveGraph, FrontierExpansionResult]:
        frontier = extract_frontier(active, world)
        candidates = boundary_candidates(active, world)
        scored = score_candidates_quantale(
            h_t, active, candidates, world,
            self.k, self.h_horizon, self.spec,
        )

        added_ids = [nid for nid, _ in scored]
        expanded = active.with_added(added_ids)
        new_active = keep_top_b(expanded, [], world, active.budget)

        # Closure stats on final active+candidate slice for cost reporting
        active_set = set(active.node_ids)
        slice_ids = list(active.node_ids) + [c for c in candidates if c not in active_set]
        slice_set = set(slice_ids)
        slice_edges: list[tuple[str, str, float]] = []
        for nid in slice_ids:
            for eid in world.out_edge_ids(nid):
                e = world.get_edge(eid)
                if e is not None and e.dst_id in slice_set:
                    slice_edges.append((e.src_id, e.dst_id, e.weight))
        cr = quantale_bounded_closure(slice_ids, slice_edges, self.h_horizon, self.spec)

        pruned = max(0, len(active.node_ids) + len(added_ids) - len(new_active.node_ids))
        result = FrontierExpansionResult(
            t_outer=t_outer,
            planner=f"quantale_{self.spec.name}_H{self.h_horizon}",
            active_node_count=len(new_active.node_ids),
            active_edge_count=len(new_active.active_edges(world)),
            frontier_node_count=len(frontier),
            candidate_count=len(candidates),
            added_node_count=len(added_ids),
            pruned_node_count=pruned,
            closure_path_count=cr.path_count,
            closure_compute_cost=cr.compute_ops,
        )
        return new_active, result
