from __future__ import annotations

import math
import time
from dataclasses import asdict

import numpy as np
import torch

from .block_topology import BlockTopologyConfig, PreparedBlockTopology
from .ir import MathNode, op_class
from .topology import MaskDiagnostics


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _block_ranges(n: int, block_size: int) -> list[tuple[int, int]]:
    return [(i, min(i + block_size, n)) for i in range(0, n, block_size)]


class HeuristicBlockTopologyBuilder:
    """TopologyBuilder-compatible block-pair heuristic runtime.

    The direct prepared path builds O(B^2 + N*K) tensors and deliberately avoids
    materializing an O(N^2) token mask.
    """

    is_learned_topology = True
    is_block_topology = True
    topology_mode = "learned_block_topk"

    def __init__(
        self,
        *,
        block_size: int = 64,
        topk_blocks: int = 4,
        include_local_blocks: int = 1,
        block_token_cap: int = 16,
        include_symbolic_bridge: bool = True,
        fixed_k: int | None = None,
        topk: int = 1,
        local_window: int = 1,
        middle_bridge_width: int = 1,
        device: torch.device | str | None = None,
        lambda_distance: float = 0.08,
    ) -> None:
        self.config = BlockTopologyConfig(
            block_size=int(block_size),
            topk_blocks=int(topk_blocks),
            include_local_blocks=int(include_local_blocks),
            include_self_block=True,
            include_symbolic_bridge=bool(include_symbolic_bridge),
            block_token_cap=int(block_token_cap),
        )
        self.fixed_k = int(fixed_k) if fixed_k is not None else max(1, int(topk_blocks) * int(block_token_cap) + 2 * int(local_window) + 1)
        self.topk = int(topk)
        self.local_window = int(local_window)
        self.middle_bridge_width = int(middle_bridge_width)
        self.lambda_distance = float(lambda_distance)
        self._requested_device = torch.device(device) if device is not None else None
        self.last_timing: dict[str, float] = {}

    @property
    def include_middle_bridge(self) -> bool:
        return self.middle_bridge_width > 0

    @property
    def cache_config_key(self) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(asdict(self.config).items())) + f"|fixed_k={self.fixed_k}"

    def _block_op_histograms(self, nodes: list[MathNode], ranges: list[tuple[int, int]]) -> list[dict[str, int]]:
        hists: list[dict[str, int]] = []
        for start, end in ranges:
            hist: dict[str, int] = {}
            for nd in nodes[start:end]:
                key = op_class(nd)
                hist[key] = hist.get(key, 0) + 1
                op_key = f"op:{nd.op}"
                hist[op_key] = hist.get(op_key, 0) + 1
            hists.append(hist)
        return hists

    @staticmethod
    def _hist_overlap(a: dict[str, int], b: dict[str, int]) -> float:
        if not a or not b:
            return 0.0
        inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in set(a) & set(b))
        denom = max(1, min(sum(a.values()), sum(b.values())))
        return inter / denom

    def _symbolic_block_pairs(self, nodes: list[MathNode]) -> set[tuple[int, int]]:
        bs = self.config.block_size
        index: dict[MathNode, int] = {node: i for i, node in enumerate(nodes)}
        pairs: set[tuple[int, int]] = set()
        for i, node in enumerate(nodes):
            bi = i // bs
            for child in node.args:
                j = index.get(child)
                if j is not None:
                    bj = j // bs
                    pairs.add((bi, bj)); pairs.add((bj, bi))
        return pairs

    def _build_block_neighbors(self, nodes: list[MathNode], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
        n = len(nodes)
        B = max(1, math.ceil(n / self.config.block_size))
        ranges = _block_ranges(n, self.config.block_size)
        hists = self._block_op_histograms(nodes, ranges)
        symbolic_pairs = self._symbolic_block_pairs(nodes) if self.config.include_symbolic_bridge else set()
        middle = (B - 1) / 2.0
        topk_eff = min(B, max(1, self.config.topk_blocks + 2 * self.config.include_local_blocks + 1))

        scores = np.zeros((B, B), dtype=np.float32)
        for i in range(B):
            for j in range(B):
                dist = abs(i - j)
                score = -self.lambda_distance * dist
                if i == j:
                    score += 10.0
                if dist <= self.config.include_local_blocks:
                    score += 3.0
                if (i, j) in symbolic_pairs:
                    score += 4.0
                score += 0.5 * self._hist_overlap(hists[i], hists[j])
                score += 0.25 / (1.0 + abs(j - middle))
                scores[i, j] = score

        if topk_eff < B:
            idx = np.argpartition(scores, -topk_eff, axis=1)[:, -topk_eff:]
            row_scores = np.take_along_axis(scores, idx, axis=1)
            order = np.argsort(-row_scores, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
        else:
            idx = np.tile(np.arange(B, dtype=np.int64), (B, 1))
        valid = np.ones_like(idx, dtype=np.int8)
        return torch.tensor(idx, dtype=torch.long, device=device), torch.tensor(valid, dtype=torch.int8, device=device), float(B * B)

    def prepare_topology(
        self,
        nodes: list[MathNode],
        z: np.ndarray | None = None,
        env: dict | None = None,
        *,
        max_neighbors: int | None = None,
        device: torch.device | None = None,
    ) -> PreparedBlockTopology:
        dev = device or self._requested_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = len(nodes)
        _sync_if_cuda(dev); t0 = time.perf_counter()
        block_neighbors, block_valid_i8, score_entries = self._build_block_neighbors(nodes, dev)
        _sync_if_cuda(dev); t1 = time.perf_counter()

        K = max(1, int(max_neighbors if max_neighbors is not None else self.fixed_k))
        bs = self.config.block_size
        cap = max(1, self.config.block_token_cap)
        local = max(0, self.local_window)
        mid = n // 2 if n else 0

        _sync_if_cuda(dev); t2 = time.perf_counter()
        # Build the token table on CPU using Python lists, then transfer once.
        # This avoids thousands of tiny CUDA tensor writes and removes the
        # previous O(N^2) child lookup inside symbolic bridge expansion.
        node_to_first_index: dict[MathNode, int] = {}
        for idx, nd in enumerate(nodes):
            node_to_first_index.setdefault(nd, idx)
        block_neighbors_cpu = block_neighbors.detach().cpu().tolist()
        block_valid_cpu = block_valid_i8.detach().cpu().numpy().astype(np.int8)
        kb_eff = int(block_neighbors.shape[1])
        block_token_indices_cpu = np.zeros((int(block_neighbors.shape[0]), kb_eff, cap), dtype=np.int64)
        block_token_valid_cpu = np.zeros((int(block_neighbors.shape[0]), kb_eff, cap), dtype=np.int8)
        for bi, row in enumerate(block_neighbors_cpu):
            for slot, bj_raw in enumerate(row):
                if slot >= kb_eff or not bool(block_valid_cpu[bi, slot]):
                    continue
                bj = int(bj_raw)
                start = bj * bs
                end = min(start + bs, n)
                if end <= start:
                    continue
                if cap >= (end - start):
                    cand = list(range(start, end))[:cap]
                else:
                    step = max(1, (end - start) // cap)
                    cand = list(range(start, end, step))[:cap]
                if cand:
                    block_token_indices_cpu[bi, slot, :len(cand)] = cand
                    block_token_valid_cpu[bi, slot, :len(cand)] = 1
        neighbors_cpu = np.zeros((n, K), dtype=np.int64)
        valid_cpu = np.zeros((n, K), dtype=np.int8)
        for t in range(n):
            vals: list[int] = []
            seen: set[int] = set()

            def add(u: int) -> None:
                if 0 <= u < n and u not in seen and len(vals) < K:
                    seen.add(u)
                    vals.append(u)

            add(t)
            for u in range(t - local, t + local + 1):
                add(u)
            if self.config.include_symbolic_bridge:
                for child in nodes[t].args:
                    j = node_to_first_index.get(child)
                    if j is not None:
                        add(j)
                if self.middle_bridge_width > 0:
                    for u in range(mid - self.middle_bridge_width, mid + self.middle_bridge_width + 1):
                        add(u)
            b = t // bs
            for bj in block_neighbors_cpu[b]:
                start = int(bj) * bs
                end = min(start + bs, n)
                if end <= start:
                    continue
                if cap >= (end - start):
                    cand = range(start, end)
                else:
                    step = max(1, (end - start) // cap)
                    cand = range(start, end, step)
                for u in cand:
                    add(int(u))
                    if len(vals) >= K:
                        break
                if len(vals) >= K:
                    break
            if vals:
                neighbors_cpu[t, :len(vals)] = vals
                valid_cpu[t, :len(vals)] = 1
        neighbors = torch.from_numpy(neighbors_cpu).to(dev, non_blocking=True)
        valid = torch.from_numpy(valid_cpu).to(dev, non_blocking=True)
        block_token_indices = torch.from_numpy(block_token_indices_cpu).to(dev, non_blocking=True)
        block_token_valid_i8 = torch.from_numpy(block_token_valid_cpu).to(dev, non_blocking=True)
        _sync_if_cuda(dev); t3 = time.perf_counter()

        per_row = valid_cpu.sum(axis=1) if n else np.array([], dtype=np.int64)
        allowed = int(per_row.sum()) if n else 0
        max_k = int(per_row.max()) if n else 0
        avg_k = float(per_row.mean()) if n else 0.0
        full = n * n
        diag = MaskDiagnostics(
            n=n,
            full_edges=full,
            allowed_edges=allowed,
            sparsity_ratio=allowed / full if full else 0.0,
            relation_reduction=1.0 - allowed / full if full else 0.0,
            avg_k=avg_k,
            max_k=max_k,
            padding_ratio=1.0 - allowed / (n * max_k) if n and max_k else 0.0,
            by_relation={
                "block_topology": allowed,
                "block_score_entries": int(score_entries),
                "block_count": int(block_neighbors.shape[0]),
                "block_token_cap": int(cap),
                "block_effective_token_k": int(block_neighbors.shape[1] * cap),
            },
        )
        self.last_timing = {
            "topology_prepare_ms": (t3 - t0) * 1000.0,
            "learned_scorer_ms": (t1 - t0) * 1000.0,
            "neighbor_table_build_ms": (t3 - t2) * 1000.0,
        }
        return PreparedBlockTopology(
            block_neighbors=block_neighbors,
            block_valid_i8=block_valid_i8,
            block_token_indices=block_token_indices,
            block_token_valid_i8=block_token_valid_i8,
            token_neighbors=neighbors,
            token_valid_i8=valid,
            diagnostics=diag,
            block_size=self.config.block_size,
            is_block_topology=True,
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

    # Compatibility fallback for dense-mask callers. Avoid using this for large N.
    def build_scored_topk_torch(self, nodes, Z_t, env, device):
        prepared = self.prepare_topology(nodes, None, env, max_neighbors=self.fixed_k, device=device)
        n = prepared.token_neighbors.shape[0]
        mask = torch.zeros((n, n), dtype=torch.bool, device=device)
        rows = torch.arange(n, device=device).unsqueeze(1).expand_as(prepared.token_neighbors)
        mask[rows.reshape(-1), prepared.token_neighbors.reshape(-1)] = prepared.token_valid_i8.reshape(-1).bool()
        return mask, prepared.diagnostics

    def build_scored_topk(self, nodes, z=None, env=None):
        dev = self._requested_device or torch.device("cpu")
        mask, diag = self.build_scored_topk_torch(nodes, None, env, dev)
        return mask.cpu().numpy().astype(bool), diag
