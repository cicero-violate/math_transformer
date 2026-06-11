from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch

from .embedder import MathEmbedder
from .ir import MathNode
from .learned_topology import FEATURE_NAMES, LearnedTopologyScorer, build_edge_feature_tensor, topk_mask_from_scores
from .topology import MaskDiagnostics, build_hand_score_matrix


class LearnedTopologyBuilder:
    """TopologyBuilder-compatible learned scorer runtime."""
    is_learned_topology = True
    topology_mode = "learned_topology"

    def __init__(self, checkpoint: str, *, fixed_k: int = 8, topk: int = 1,
                 local_window: int = 1, middle_bridge_width: int = 1,
                 device: torch.device | str | None = None,
                 protect_noncommutative: bool = False,
                 polarity_summary: str | None = None,
                 polarity_alpha: float = 0.0) -> None:
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
        self.protect_noncommutative = bool(protect_noncommutative)
        self.polarity_summary = polarity_summary
        self.polarity_alpha = float(polarity_alpha)
        self._polarity: dict[str, float] | None = None

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

    def _load_polarity(self) -> dict[str, float]:
        if self._polarity is not None:
            return self._polarity
        if not self.polarity_summary:
            self._polarity = {}
            return self._polarity
        data = json.loads(Path(self.polarity_summary).read_text(encoding="utf-8"))
        self._polarity = {str(r["pattern"]): float(r.get("record_polarity", 0.0) or 0.0) for r in data.get("edge_kind_polarity", []) if r.get("pattern") is not None}
        return self._polarity

    @staticmethod
    def _edge_kind(src: MathNode, dst: MathNode) -> str:
        a = "leaf" if src.is_leaf() else src.op
        b = "leaf" if dst.is_leaf() else dst.op
        return f"{a}->{b}"

    @staticmethod
    def _protected_mask(nodes: list[MathNode], device: torch.device) -> torch.Tensor:
        mat = torch.zeros(len(nodes), len(nodes), dtype=torch.bool, device=device)
        for i, node in enumerate(nodes):
            if node.op not in {"sub", "div", "matmul"}:
                continue
            for child in node.args:
                for j, other in enumerate(nodes):
                    if other is child or other == child:
                        mat[i, j] = True
                        mat[j, i] = True
        return mat

    def _polarity_bias(self, nodes: list[MathNode], base_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        polarity = self._load_polarity()
        bias = torch.zeros(len(nodes), len(nodes), dtype=torch.float32, device=device)
        if self.polarity_alpha == 0.0 or not polarity:
            return bias
        base = base_mask.to(device=device, dtype=torch.bool)
        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                kind = self._edge_kind(src, dst)
                if bool(base[i, j].item()):
                    value = -float(polarity.get(f"removed_edges:{kind}", 0.0))
                else:
                    value = float(polarity.get(f"extra_edges:{kind}", 0.0))
                if value:
                    bias[i, j] = self.polarity_alpha * value
        return bias

    def _mask_from_scores(self, nodes: list[MathNode], scores: torch.Tensor, device: torch.device) -> torch.Tensor:
        protected = self._protected_mask(nodes, device) if self.protect_noncommutative else None
        hand_scores_np, _ = build_hand_score_matrix(
            nodes,
            None,
            None,
            local_window=self.local_window,
            include_middle_bridge=self.include_middle_bridge,
            middle_bridge_width=self.middle_bridge_width,
        )
        hand_scores = torch.as_tensor(hand_scores_np, dtype=scores.dtype, device=device)
        base_mask = topk_mask_from_scores(hand_scores, self.fixed_k)
        bias = self._polarity_bias(nodes, base_mask, device)
        work = scores + bias.to(device=device, dtype=scores.dtype)
        n = int(scores.shape[0])
        k = min(self.fixed_k, n)
        if protected is None:
            return topk_mask_from_scores(work, self.fixed_k)
        idx = torch.arange(n, device=device)
        protected = protected.clone()
        protected[idx, idx] = True
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for row in range(n):
            row_mask = protected[row].clone()
            remaining = max(0, k - int(row_mask.sum().item()))
            if remaining > 0:
                eligible = ~row_mask
                count = int(eligible.sum().item())
                if count > 0:
                    take = min(remaining, count)
                    row_scores = work[row].masked_fill(~eligible, -torch.inf)
                    chosen = row_scores.topk(take).indices
                    row_mask[chosen] = True
            mask[row] = row_mask
        return mask

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
        mask = self._mask_from_scores(nodes, scores, device)
        return mask, self._diag(mask)

    def build_scored_topk(self, nodes: list[MathNode], z: np.ndarray | None = None,
                          env: dict | None = None) -> tuple[np.ndarray, MaskDiagnostics]:
        device = self._requested_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scores = self._scores(nodes, z, env, device)
        mask_t = self._mask_from_scores(nodes, scores, device)
        mask = mask_t.detach().cpu().numpy().astype(bool)
        return mask, self._diag(mask)
