from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .ir import MathNode


def _hash_id(*parts: str) -> str:
    canonical = "\x00".join(parts)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def make_node_id(label: str, node_kind: str, provenance: str = "") -> str:
    return _hash_id("node", label, node_kind, provenance)


def make_edge_id(src_id: str, dst_id: str, relation: str) -> str:
    return _hash_id("edge", src_id, dst_id, relation)


@dataclass
class NodeRecord:
    """A node in the world graph. node_id is identity; label is metadata."""
    node_id: str
    label: str
    node_kind: str
    features: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    version: int = 0

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeRecord):
            return NotImplemented
        return self.node_id == other.node_id


@dataclass
class EdgeRecord:
    """A directed edge in the world graph. edge_id is identity; weight is ω_ij."""
    edge_id: str
    src_id: str
    dst_id: str
    relation: str
    weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 0

    def __hash__(self) -> int:
        return hash(self.edge_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EdgeRecord):
            return NotImplemented
        return self.edge_id == other.edge_id


class WorldGraph:
    """
    Persistent world graph G_world.

    Raw facts (nodes/edges) are append-only once added.
    Derived state (edge weights ω_ij, version counters) is mutable.
    Hash IDs are identity; labels are metadata.

    Tombstoned records are held in _inactive_* sets and are invisible to all
    normal query methods.  Use include_inactive=True variants for archival access.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeRecord] = {}
        self._edges: dict[str, EdgeRecord] = {}
        self._out_edge_ids: dict[str, list[str]] = {}
        self._in_edge_ids: dict[str, list[str]] = {}
        self._inactive_nodes: set[str] = set()
        self._inactive_edges: set[str] = set()

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_node(self, node: NodeRecord) -> None:
        if node.node_id not in self._nodes:
            self._nodes[node.node_id] = node
            self._out_edge_ids.setdefault(node.node_id, [])
            self._in_edge_ids.setdefault(node.node_id, [])

    def add_edge(self, edge: EdgeRecord) -> None:
        if edge.edge_id in self._edges:
            return
        missing = [x for x in (edge.src_id, edge.dst_id) if x not in self._nodes]
        if missing:
            raise ValueError(f"Cannot add edge {edge.edge_id!r}: missing nodes {missing}")
        self._edges[edge.edge_id] = edge
        self._out_edge_ids.setdefault(edge.src_id, []).append(edge.edge_id)
        self._in_edge_ids.setdefault(edge.dst_id, []).append(edge.edge_id)

    def update_edge_weight(self, edge_id: str, weight: float) -> None:
        """Update ω_ij — mutable derived state."""
        e = self._edges.get(edge_id)
        if e is not None:
            e.weight = weight
            e.version += 1

    def deactivate_node(self, node_id: str) -> None:
        """Tombstone a node — it stays in raw storage but is invisible to queries."""
        if node_id in self._nodes:
            self._inactive_nodes.add(node_id)

    def deactivate_edge(self, edge_id: str) -> None:
        """Tombstone an edge — stays in raw storage but invisible to queries."""
        if edge_id in self._edges:
            self._inactive_edges.add(edge_id)

    def reactivate_node(self, node_id: str) -> None:
        self._inactive_nodes.discard(node_id)

    def reactivate_edge(self, edge_id: str) -> None:
        self._inactive_edges.discard(edge_id)

    def is_node_active(self, node_id: str) -> bool:
        return node_id in self._nodes and node_id not in self._inactive_nodes

    def is_edge_active(self, edge_id: str) -> bool:
        return edge_id in self._edges and edge_id not in self._inactive_edges

    # ── Query (active records only by default) ────────────────────────────────

    def get_node(self, node_id: str, *, include_inactive: bool = False) -> NodeRecord | None:
        if not include_inactive and node_id in self._inactive_nodes:
            return None
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str, *, include_inactive: bool = False) -> EdgeRecord | None:
        if not include_inactive and edge_id in self._inactive_edges:
            return None
        return self._edges.get(edge_id)

    def out_edge_ids(self, node_id: str) -> list[str]:
        return [eid for eid in self._out_edge_ids.get(node_id, []) if eid not in self._inactive_edges]

    def in_edge_ids(self, node_id: str) -> list[str]:
        return [eid for eid in self._in_edge_ids.get(node_id, []) if eid not in self._inactive_edges]

    def neighbors(self, node_id: str) -> list[str]:
        """All active neighbors (undirected union of active out- and in-edges)."""
        seen: dict[str, None] = {}
        for eid in self._out_edge_ids.get(node_id, []):
            if eid in self._inactive_edges:
                continue
            nb = self._edges[eid].dst_id
            if nb not in self._inactive_nodes:
                seen[nb] = None
        for eid in self._in_edge_ids.get(node_id, []):
            if eid in self._inactive_edges:
                continue
            nb = self._edges[eid].src_id
            if nb not in self._inactive_nodes:
                seen[nb] = None
        seen.pop(node_id, None)
        return list(seen)

    def out_neighbors(self, node_id: str) -> list[str]:
        return [
            self._edges[eid].dst_id
            for eid in self._out_edge_ids.get(node_id, [])
            if eid not in self._inactive_edges
        ]

    def edges_between(self, src_id: str, dst_id: str) -> list[EdgeRecord]:
        return [
            self._edges[eid]
            for eid in self._out_edge_ids.get(src_id, [])
            if eid not in self._inactive_edges and self._edges[eid].dst_id == dst_id
        ]

    def node_count(self) -> int:
        return len(self._nodes) - len(self._inactive_nodes)

    def edge_count(self) -> int:
        return len(self._edges) - len(self._inactive_edges)

    def iter_nodes(self) -> Iterator[NodeRecord]:
        return (n for n in self._nodes.values() if n.node_id not in self._inactive_nodes)

    def iter_edges(self) -> Iterator[EdgeRecord]:
        return (e for e in self._edges.values() if e.edge_id not in self._inactive_edges)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes and node_id not in self._inactive_nodes

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self._edges and edge_id not in self._inactive_edges

    # ── Serialization ─────────────────────────────────────────────────────────

    def save_jsonl(self, path: Path) -> None:
        with Path(path).open("w") as f:
            for n in self._nodes.values():
                f.write(json.dumps({
                    "kind": "node",
                    "node_id": n.node_id,
                    "label": n.label,
                    "node_kind": n.node_kind,
                    "features": n.features,
                    "provenance": n.provenance,
                    "version": n.version,
                }) + "\n")
            for e in self._edges.values():
                f.write(json.dumps({
                    "kind": "edge",
                    "edge_id": e.edge_id,
                    "src_id": e.src_id,
                    "dst_id": e.dst_id,
                    "relation": e.relation,
                    "weight": e.weight,
                    "metadata": e.metadata,
                    "version": e.version,
                }) + "\n")
            for nid in self._inactive_nodes:
                f.write(json.dumps({"kind": "inactive_node", "node_id": nid}) + "\n")
            for eid in self._inactive_edges:
                f.write(json.dumps({"kind": "inactive_edge", "edge_id": eid}) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> WorldGraph:
        g = cls()
        with Path(path).open() as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                kind = rec.pop("kind")
                if kind == "node":
                    g.add_node(NodeRecord(**rec))
                elif kind == "edge":
                    g.add_edge(EdgeRecord(**rec))
                elif kind == "inactive_node":
                    g._inactive_nodes.add(rec["node_id"])
                elif kind == "inactive_edge":
                    g._inactive_edges.add(rec["edge_id"])
        return g

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_math_nodes(
        cls,
        nodes: list[MathNode],
        provenance: str = "math_expr",
    ) -> tuple[WorldGraph, list[str]]:
        """
        Build a WorldGraph from a flat list of MathNodes.
        Returns (world, node_ids) where node_ids[i] corresponds to nodes[i].
        Symbolic dependency edges are added for all parent-child arg relationships.
        """
        g = cls()
        id_map: dict[int, str] = {}

        for i, mn in enumerate(nodes):
            label = repr(mn)
            # Include position index so repeated identical nodes (e.g. x+x) get distinct IDs.
            nid = make_node_id(label, mn.op, f"{provenance}:{i}")
            rec = NodeRecord(
                node_id=nid,
                label=label,
                node_kind=mn.op,
                features={
                    "op": mn.op,
                    "value": str(mn.value) if mn.value is not None else None,
                    "arity": mn.arity,
                    "depth": mn.depth,
                    "subtree_size": mn.subtree_size,
                },
                provenance=provenance,
            )
            g.add_node(rec)
            id_map[id(mn)] = nid

        for mn in nodes:
            src_id = id_map[id(mn)]
            for child in mn.args:
                if id(child) in id_map:
                    dst_id = id_map[id(child)]
                    eid = make_edge_id(src_id, dst_id, "symbolic_dependency")
                    try:
                        g.add_edge(EdgeRecord(
                            edge_id=eid,
                            src_id=src_id,
                            dst_id=dst_id,
                            relation="symbolic_dependency",
                        ))
                    except ValueError:
                        pass

        return g, [id_map[id(mn)] for mn in nodes]


@dataclass
class ActiveGraph:
    """
    Bounded active graph G_t: a slice of G_world with node budget B.

    node_ids: frozenset — the current active node set
    budget: int — maximum |G_t|; enforced by keep_top_b in frontier_planner
    anchor_ids: frozenset — nodes that must not be pruned
    step: int — which T_outer step produced this graph
    """

    node_ids: frozenset[str]
    budget: int
    anchor_ids: frozenset[str] = field(default_factory=frozenset)
    step: int = 0

    @classmethod
    def seed(
        cls,
        seed_ids: list[str],
        budget: int,
        anchor_ids: list[str] | None = None,
    ) -> ActiveGraph:
        if budget <= 0:
            raise ValueError(f"budget must be > 0, got {budget}")
        ids = frozenset(seed_ids)
        return cls(
            node_ids=ids,
            budget=budget,
            anchor_ids=frozenset(anchor_ids if anchor_ids is not None else seed_ids),
            step=0,
        )

    def frontier(self, world: WorldGraph) -> frozenset[str]:
        """∂G_t: active nodes with at least one neighbor outside G_t."""
        boundary: set[str] = set()
        for nid in self.node_ids:
            for nb in world.neighbors(nid):
                if nb not in self.node_ids:
                    boundary.add(nid)
                    break
        return frozenset(boundary)

    def boundary_candidates(self, world: WorldGraph) -> list[str]:
        """C_t: world-graph neighbors of ∂G_t that are not in G_t."""
        candidates: set[str] = set()
        for nid in self.frontier(world):
            for nb in world.neighbors(nid):
                if nb not in self.node_ids and world.has_node(nb):
                    candidates.add(nb)
        return list(candidates)

    def active_edges(self, world: WorldGraph) -> list[EdgeRecord]:
        """Edges where both endpoints are in G_t."""
        result: list[EdgeRecord] = []
        for nid in self.node_ids:
            for eid in world.out_edge_ids(nid):
                e = world.get_edge(eid)
                if e is not None and e.dst_id in self.node_ids:
                    result.append(e)
        return result

    def to_node_records(self, world: WorldGraph) -> list[NodeRecord]:
        return [r for nid in self.node_ids for r in [world.get_node(nid)] if r is not None]

    def contains(self, node_id: str) -> bool:
        return node_id in self.node_ids

    def with_added(self, new_ids: list[str]) -> ActiveGraph:
        return ActiveGraph(
            node_ids=self.node_ids | frozenset(new_ids),
            budget=self.budget,
            anchor_ids=self.anchor_ids,
            step=self.step,
        )

    def with_node_set(self, new_ids: frozenset[str]) -> ActiveGraph:
        return ActiveGraph(
            node_ids=new_ids,
            budget=self.budget,
            anchor_ids=self.anchor_ids,
            step=self.step + 1,
        )

    def is_at_budget(self) -> bool:
        return len(self.node_ids) >= self.budget
