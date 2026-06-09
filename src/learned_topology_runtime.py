from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from .embedder import MathEmbedder
from .ir import MathNode
from .learned_topology import FEATURE_NAMES, LearnedTopologyScorer, build_edge_feature_tensor, topk_mask_from_scores
from .topology import MaskDiagnostics


class LearnedTopologyBuilder:
    """TopologyBuilder-compatible learned scorer runtime."""
    is_learned_topology = True
    topology_mode = "learned_topology"

    def __init__(self, checkpoint: str, *, fixed_k: int = 8, topk: int = 1,
                 local_window: int = 1, middle_bridge_width: int = 1,
                 device: torch.device | str | None = None) -> None:
        self.checkpoint = str(checkpoint)
        self.fixed_k = int(fixed_k)
        self.topk = int(topk)
        self.local_window = int(local_window)
        self.middle_bridge_width = int(middle_bridge_width)
        self.embedder = MathEmbedder()
        self._requested_device = torch.device(device) if device is not None else None
        self._scorer: LearnedTopologyScorer | None = None
        self._hidden_dim = 64
        self._feature_names = FEATURE_NAMES

    @property
    def include_middle_bridge(self) -> bool:
        return self.middle_bridge_width > 0

    def _load_state(self, device: torch.device) -> LearnedTopologyScorer:
        if self._scorer is not None and next(self._scorer.parameters()).device == device:
            return self._scorer
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"learned topology scorer checkpoint not found: {self.checkpoint}")
        state = torch.load(self.checkpoint, map_location=device, weights_only=True)
        self._hidden_dim = int(state.get("hidden_dim", 64))
        self._feature_names = tuple(state.get("feature_names", FEATURE_NAMES))
        scorer = LearnedTopologyScorer(feature_dim=len(self._feature_names), hidden_dim=self._hidden_dim).to(device)
        scorer.load_state_dict(state["model_state_dict"])
        scorer.eval()
        self._scorer = scorer
        return scorer

    def _scores(self, nodes: list[MathNode], z: np.ndarray | None, env: dict | None,
                device: torch.device) -> torch.Tensor:
        if z is None:
            z = self.embedder.encode_batch(nodes)
        scorer = self._load_state(device)
        features = build_edge_feature_tensor(
            nodes, z, env,
            local_window=self.local_window,
            middle_bridge_width=self.middle_bridge_width,
            device=device,
        )
        with torch.no_grad():
            return scorer(features)

    @staticmethod
    def _diag(mask: torch.Tensor | np.ndarray) -> MaskDiagnostics:
        if isinstance(mask, torch.Tensor):
            n = int(mask.shape[0])
            per_row = mask.sum(dim=1) if n else torch.tensor([], device=mask.device)
            allowed = int(mask.sum().item()) if n else 0
            max_k = int(per_row.max().item()) if n else 0
            avg_k = float(per_row.float().mean().item()) if n else 0.0
            ident = int(torch.diag(mask).sum().item()) if n else 0
        else:
            n = int(mask.shape[0])
            per_row = mask.sum(axis=1) if n else np.array([])
            allowed = int(mask.sum()) if n else 0
            max_k = int(per_row.max()) if n else 0
            avg_k = float(per_row.mean()) if n else 0.0
            ident = int(np.diag(mask).sum()) if n else 0
        full = n * n
        return MaskDiagnostics(
            n=n, full_edges=full, allowed_edges=allowed,
            sparsity_ratio=allowed / full if full else 0.0,
            relation_reduction=1.0 - allowed / full if full else 0.0,
            avg_k=avg_k, max_k=max_k,
            padding_ratio=1.0 - allowed / (n * max_k) if n and max_k else 0.0,
            by_relation={"learned_topology": allowed, "identity": ident},
        )

    @staticmethod
    def priority_from_mask_torch(mask: torch.Tensor) -> torch.Tensor:
        priority = torch.zeros(mask.shape, dtype=torch.int8, device=mask.device)
        priority[mask] = 2
        if mask.numel():
            idx = torch.arange(mask.shape[0], device=mask.device)
            priority[idx, idx] = 1
        return priority

    @staticmethod
    def priority_from_mask(mask: np.ndarray) -> np.ndarray:
        priority = np.zeros(mask.shape, dtype=np.int8)
        priority[mask] = 2
        if mask.size:
            np.fill_diagonal(priority, 1)
        return priority

    def build_scored_topk_torch(self, nodes: list[MathNode], Z_t: torch.Tensor | None,
                                env: dict | None, device: torch.device) -> tuple[torch.Tensor, MaskDiagnostics]:
        z = Z_t.detach().cpu().numpy() if Z_t is not None else None
        scores = self._scores(nodes, z, env, device)
        mask = topk_mask_from_scores(scores, self.fixed_k)
        return mask, self._diag(mask)

    def build_scored_topk(self, nodes: list[MathNode], z: np.ndarray | None = None,
                          env: dict | None = None) -> tuple[np.ndarray, MaskDiagnostics]:
        device = self._requested_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scores = self._scores(nodes, z, env, device)
        mask_t = topk_mask_from_scores(scores, self.fixed_k)
        mask = mask_t.detach().cpu().numpy().astype(bool)
        return mask, self._diag(mask)
