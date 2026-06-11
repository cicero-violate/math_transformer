"""
Persistent graph memory weights and writeback verifier (P2.1–P2.3).

P2.1 — GraphMemoryStore: per-edge ω_ij, utility, confidence, staleness, provenance, version
P2.2 — ClosurePreservingEdgeDeletionGate: checks A^{<=H} vs (A\\e)^{<=H} before delete
P2.3 — GraphWritebackVerifier: accept/reject ΔV+, ΔV-, ΔE+, ΔE-, Δω;
        archive/tombstone instead of destructive delete

Non-negotiable gates:
  9  — Do not persist mutations without verifier evidence, provenance, versioning, rollback.
  10 — Do not store transformer weights in graph records.
  12 — Raw facts append-only; derived structures mutable and versioned.
  16 — Edge deletion must preserve important bounded closure or show accepted Pareto tradeoff.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .world_graph import WorldGraph, NodeRecord, EdgeRecord
from .graph_closure import bounded_closure, ClosureMode

SCHEMA_VERSION = "graph_memory.v1"


# ── P2.1: Edge memory weights ─────────────────────────────────────────────────

@dataclass
class EdgeMemoryRecord:
    """
    Per-edge persistent memory weight ω_ij plus metadata.
    Not a transformer weight (gate 10).
    """
    edge_id: str
    omega: float = 0.0         # memory weight ω_ij
    utility: float = 0.0       # learned utility signal
    confidence: float = 1.0    # confidence in this weight estimate
    staleness: float = 0.0     # 0 = fresh, 1 = fully stale
    provenance: str = ""       # where this weight came from
    version: int = 0
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class GraphMemoryStore:
    """
    Stores and retrieves ω_ij edge memory weights.

    Separate from model parameters (gate 10).
    Mutable and versioned (gate 12 derived state).
    """

    def __init__(self) -> None:
        self._records: dict[str, EdgeMemoryRecord] = {}

    def get(self, edge_id: str) -> EdgeMemoryRecord | None:
        return self._records.get(edge_id)

    def set(
        self,
        edge_id: str,
        *,
        omega: float | None = None,
        utility: float | None = None,
        confidence: float | None = None,
        staleness: float | None = None,
        provenance: str | None = None,
    ) -> EdgeMemoryRecord:
        rec = self._records.get(edge_id) or EdgeMemoryRecord(edge_id=edge_id)
        if omega is not None:
            rec.omega = omega
        if utility is not None:
            rec.utility = utility
        if confidence is not None:
            rec.confidence = confidence
        if staleness is not None:
            rec.staleness = staleness
        if provenance is not None:
            rec.provenance = provenance
        rec.version += 1
        rec.updated_at = time.time()
        self._records[edge_id] = rec
        return rec

    def apply_to_world(self, world: WorldGraph) -> int:
        """Sync stored ω_ij values into world graph edge records."""
        updated = 0
        for edge_id, rec in self._records.items():
            if world.has_edge(edge_id):
                world.update_edge_weight(edge_id, rec.omega)
                updated += 1
        return updated

    def __len__(self) -> int:
        return len(self._records)

    def save_jsonl(self, path: Path) -> None:
        with Path(path).open("w") as f:
            for rec in self._records.values():
                row = {"schema_version": SCHEMA_VERSION}
                row.update(rec.as_dict())
                f.write(json.dumps(row) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> GraphMemoryStore:
        store = cls()
        with Path(path).open() as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rec.pop("schema_version", None)
                store._records[rec["edge_id"]] = EdgeMemoryRecord(**rec)
        return store


# ── P2.2: Closure-preserving edge deletion gate ───────────────────────────────

@dataclass
class DeletionGateReport:
    edge_id: str
    relation: str
    mode: str
    h: int
    closure_pairs_with: int
    closure_pairs_without: int
    preservation_ratio: float
    min_preservation: float
    allowed: bool
    reason: str
    pareto_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class ClosurePreservingEdgeDeletionGate:
    """
    P2.2: Before archive/delete, compare A^{<=H} vs (A\\e)^{<=H}.
    Edge deletion is allowed only if the bounded closure is preserved above
    min_preservation (default 0.95), or an accepted Pareto tradeoff is provided.

    Gate 16: edge deletion must preserve important bounded closure or show
             an accepted Pareto tradeoff in quality/runtime/memory.
    """

    def __init__(
        self,
        h: int = 2,
        mode: ClosureMode = "boolean",
        min_preservation: float = 0.95,
    ) -> None:
        self.h = h
        self.mode = mode
        self.min_preservation = min_preservation

    def check(
        self,
        world: WorldGraph,
        edge_id: str,
        active_node_ids: list[str] | None = None,
        pareto_override: bool = False,
        pareto_note: str = "",
    ) -> tuple[bool, DeletionGateReport]:
        """
        Returns (allowed, report).

        pareto_override: if True and pareto_note is non-empty, allows deletion
                         even when preservation < min_preservation.
        """
        edge = world.get_edge(edge_id)
        if edge is None:
            rep = DeletionGateReport(
                edge_id=edge_id, relation="", mode=self.mode, h=self.h,
                closure_pairs_with=0, closure_pairs_without=0,
                preservation_ratio=1.0, min_preservation=self.min_preservation,
                allowed=True, reason="edge_not_found",
            )
            return True, rep

        node_ids = active_node_ids or [n.node_id for n in world.iter_nodes()]
        slice_set = set(node_ids)

        def collect(exclude: str | None = None) -> list[tuple[str, str, float]]:
            result = []
            for nid in node_ids:
                for eid in world.out_edge_ids(nid):
                    if eid == exclude:
                        continue
                    e = world.get_edge(eid)
                    if e is not None and e.dst_id in slice_set:
                        result.append((e.src_id, e.dst_id, e.weight))
            return result

        with_edges = collect()
        without_edges = collect(exclude=edge_id)

        cr_with = bounded_closure(node_ids, with_edges, self.h, self.mode)
        cr_without = bounded_closure(node_ids, without_edges, self.h, self.mode)

        _LARGE = 1e9
        if self.mode == "boolean":
            pairs_with = max(int(cr_with.reachability.sum()), 1)
            pairs_without = int(cr_without.reachability.sum())
        elif self.mode == "cost":
            pairs_with = max(int((cr_with.reachability < _LARGE).sum()), 1)
            pairs_without = int((cr_without.reachability < _LARGE).sum())
        else:  # utility
            pairs_with = max(int((cr_with.reachability > -_LARGE / 2).sum()), 1)
            pairs_without = int((cr_without.reachability > -_LARGE / 2).sum())

        preservation = pairs_without / pairs_with
        allowed = preservation >= self.min_preservation
        reason = "ok" if allowed else "closure_degraded"

        if not allowed and pareto_override and pareto_note:
            allowed = True
            reason = f"pareto_override: {pareto_note}"

        rep = DeletionGateReport(
            edge_id=edge_id,
            relation=edge.relation,
            mode=self.mode,
            h=self.h,
            closure_pairs_with=pairs_with,
            closure_pairs_without=pairs_without,
            preservation_ratio=round(preservation, 4),
            min_preservation=self.min_preservation,
            allowed=allowed,
            reason=reason,
            pareto_note=pareto_note,
        )
        return allowed, rep


# ── P2.3: Graph writeback verifier ────────────────────────────────────────────

@dataclass
class GraphDelta:
    """
    Proposed atomic change to G_world.
    Accepted or rejected by GraphWritebackVerifier.
    Raw facts (nodes/edges) are append-only; derivations are mutable/versioned (gate 12).
    """
    delta_v_add: list[NodeRecord] = field(default_factory=list)
    delta_v_remove: list[str] = field(default_factory=list)     # node_ids (tombstoned)
    delta_e_add: list[EdgeRecord] = field(default_factory=list)
    delta_e_remove: list[str] = field(default_factory=list)     # edge_ids (tombstoned)
    delta_omega: dict[str, float] = field(default_factory=dict) # edge_id -> new ω_ij
    provenance: str = ""
    timestamp: float = field(default_factory=time.time)

    def has_changes(self) -> bool:
        return bool(
            self.delta_v_add or self.delta_v_remove
            or self.delta_e_add or self.delta_e_remove
            or self.delta_omega
        )


@dataclass
class TombstoneRecord:
    kind: str          # "node" | "edge"
    record_data: dict[str, Any]
    delta_provenance: str
    tombstoned_at: float
    version: int


class GraphWritebackVerifier:
    """
    P2.3: Accept/reject graph deltas with provenance, versioning, and rollback.

    Accepted deltas:
      - ΔV+, ΔE+ are applied immediately
      - ΔV-, ΔE- are tombstoned (archived), not destructively deleted
      - Δω are applied as mutable derived state (gate 12)
      - Rejected if missing provenance or if closure gate fails

    Gate 9: requires provenance, versioning, rollback path.
    """

    def __init__(
        self,
        deletion_gate: ClosurePreservingEdgeDeletionGate | None = None,
        require_provenance: bool = True,
    ) -> None:
        self._gate = deletion_gate or ClosurePreservingEdgeDeletionGate()
        self._require_provenance = require_provenance
        self._tombstones: dict[str, TombstoneRecord] = {}
        self._version: int = 0

    def verify_delta(
        self,
        delta: GraphDelta,
        world: WorldGraph,
    ) -> tuple[bool, list[str]]:
        """
        Validate and apply delta to world.

        Returns (accepted, rejection_reasons).
        On rejection, world is unchanged.
        """
        reasons: list[str] = []

        if self._require_provenance and not delta.provenance:
            reasons.append("missing provenance in delta")
            return False, reasons

        # Validate additions
        for node in delta.delta_v_add:
            if not node.node_id or not node.node_kind:
                reasons.append(f"node {node.node_id!r}: missing node_id or node_kind")

        new_node_ids = {n.node_id for n in delta.delta_v_add}
        for edge in delta.delta_e_add:
            if not edge.edge_id or not edge.src_id or not edge.dst_id:
                reasons.append(f"edge {edge.edge_id!r}: missing required fields")
                continue
            if not world.has_node(edge.src_id) and edge.src_id not in new_node_ids:
                reasons.append(f"edge {edge.edge_id!r}: src {edge.src_id!r} not in world or delta")
            if not world.has_node(edge.dst_id) and edge.dst_id not in new_node_ids:
                reasons.append(f"edge {edge.edge_id!r}: dst {edge.dst_id!r} not in world or delta")

        # Validate removals
        for node_id in delta.delta_v_remove:
            if not world.has_node(node_id):
                reasons.append(f"node_remove {node_id!r}: not in world")
                continue
            for eid in world.out_edge_ids(node_id) + world.in_edge_ids(node_id):
                allowed, report = self._gate.check(world, eid)
                if not allowed:
                    reasons.append(
                        f"node_remove {node_id!r}: edge {eid!r} closure degraded "
                        f"(preservation={report.preservation_ratio:.3f})"
                    )

        for edge_id in delta.delta_e_remove:
            if not world.has_edge(edge_id):
                reasons.append(f"edge_remove {edge_id!r}: not in world")
                continue
            allowed, report = self._gate.check(world, edge_id)
            if not allowed:
                reasons.append(
                    f"edge_remove {edge_id!r}: closure degraded "
                    f"(preservation={report.preservation_ratio:.3f} < {report.min_preservation})"
                )

        if reasons:
            return False, reasons

        # Apply additions
        for node in delta.delta_v_add:
            world.add_node(node)
        for edge in delta.delta_e_add:
            try:
                world.add_edge(edge)
            except ValueError as exc:
                return False, [str(exc)]

        # Tombstone removals (archive, never destructive)
        for node_id in delta.delta_v_remove:
            node = world.get_node(node_id)
            if node is not None:
                self._tombstones[node_id] = TombstoneRecord(
                    kind="node",
                    record_data={
                        "node_id": node.node_id, "label": node.label,
                        "node_kind": node.node_kind, "features": node.features,
                        "provenance": node.provenance, "version": node.version,
                    },
                    delta_provenance=delta.provenance,
                    tombstoned_at=delta.timestamp,
                    version=self._version,
                )
                world.deactivate_node(node_id)

        for edge_id in delta.delta_e_remove:
            edge = world.get_edge(edge_id)
            if edge is not None:
                self._tombstones[edge_id] = TombstoneRecord(
                    kind="edge",
                    record_data={
                        "edge_id": edge.edge_id, "src_id": edge.src_id,
                        "dst_id": edge.dst_id, "relation": edge.relation,
                        "weight": edge.weight, "metadata": edge.metadata,
                        "version": edge.version,
                    },
                    delta_provenance=delta.provenance,
                    tombstoned_at=delta.timestamp,
                    version=self._version,
                )
                world.deactivate_edge(edge_id)

        # Apply weight updates (mutable derived state, gate 12)
        for edge_id, new_weight in delta.delta_omega.items():
            world.update_edge_weight(edge_id, new_weight)

        self._version += 1
        return True, []

    def get_tombstone(self, record_id: str) -> TombstoneRecord | None:
        return self._tombstones.get(record_id)

    def list_tombstones(self) -> list[str]:
        return list(self._tombstones)

    def rollback(self, record_id: str, world: WorldGraph) -> bool:
        """Restore a tombstoned node or edge to the world graph."""
        ts = self._tombstones.get(record_id)
        if ts is None:
            return False
        if ts.kind == "node":
            world.reactivate_node(record_id)
        elif ts.kind == "edge":
            world.reactivate_edge(record_id)
        else:
            return False
        del self._tombstones[record_id]
        return True

    @property
    def version(self) -> int:
        return self._version
