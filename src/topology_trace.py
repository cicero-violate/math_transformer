from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from .ir import MathNode


JsonDict = dict[str, Any]


def _jsonable(value: Any) -> Any:
    """Convert common tensor/numpy scalar values into JSON-serializable values."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def hash_nodes(nodes: Iterable[MathNode]) -> str:
    """Stable content hash for a node sequence used in topology tracing."""
    payload = [repr(node) for node in nodes]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def summarize_scores(scores: torch.Tensor | None) -> JsonDict:
    """Compact score summary. Avoids writing dense N^2 score matrices by default."""
    if scores is None:
        return {}
    s = scores.detach().float().cpu()
    if s.numel() == 0:
        return {"numel": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "numel": int(s.numel()),
        "min": float(s.min().item()),
        "max": float(s.max().item()),
        "mean": float(s.mean().item()),
        "std": float(s.std(unbiased=False).item()) if s.numel() > 1 else 0.0,
    }


def summarize_mask(mask: torch.Tensor | None) -> JsonDict:
    """Compact topology mask summary for JSONL traces."""
    if mask is None:
        return {}
    m = mask.detach().bool().cpu()
    if m.ndim != 2:
        raise ValueError(f"expected a 2D topology mask, got shape {tuple(m.shape)}")
    n = int(m.shape[0])
    dense_edges = int(m.numel())
    active_edges = int(m.sum().item())
    per_row = m.sum(dim=1) if n else torch.zeros(0, dtype=torch.long)
    return {
        "n": n,
        "dense_edges": dense_edges,
        "active_edges": active_edges,
        "sparsity_ratio": float(active_edges / dense_edges) if dense_edges else 0.0,
        "min_k": int(per_row.min().item()) if n else 0,
        "max_k": int(per_row.max().item()) if n else 0,
        "mean_k": float(per_row.float().mean().item()) if n else 0.0,
        "self_edges": int(torch.diag(m).sum().item()) if n else 0,
    }


def summarize_overlap(pred: torch.Tensor | None, target: torch.Tensor | None) -> JsonDict:
    """Summarize overlap between two topology masks without storing full matrices."""
    if pred is None or target is None:
        return {}
    p = pred.detach().bool().cpu()
    t = target.detach().bool().cpu()
    if p.shape != t.shape:
        raise ValueError(f"pred/target mask shape mismatch: {tuple(p.shape)} vs {tuple(t.shape)}")
    hit = int((p & t).sum().item())
    pred_edges = int(p.sum().item())
    target_edges = int(t.sum().item())
    return {
        "edge_hits": hit,
        "pred_edges": pred_edges,
        "target_edges": target_edges,
        "micro_precision": float(hit / pred_edges) if pred_edges else 0.0,
        "micro_recall": float(hit / target_edges) if target_edges else 0.0,
        "missing_edges": int((t & ~p).sum().item()),
        "extra_edges": int((p & ~t).sum().item()),
    }


@dataclass
class TopologyTraceWriter:
    """Append-only JSONL writer for topology/evaluator traces.

    The writer intentionally stores compact summaries by default. Full dense feature
    tensors and score matrices are too large for routine tracing and should be
    exported only by specialized diagnostics.
    """

    path: str | Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, record: JsonDict) -> None:
        self._fh.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        self.count += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TopologyTraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()
