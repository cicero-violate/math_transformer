"""
Qwen checkpoint weight graph compiler — plan v23 P0.1, P0.2, P1.1, P1.2, P1.3, P2.1 stub.

Pipeline:
    safetensors checkpoint
        -> typed tensor manifest (P0.1, P0.2)
        -> block-energy edge compiler (P1.1)
        -> head/expert structural graph (P1.2)
        -> artifact files: manifest.json, nodes.jsonl, edges.jsonl, stats.json
        -> WorldGraph schema adapter (P1.3)

Non-negotiable gates (plan v23):
    Gate 2: raw_weight_payload_in_graph = false — no raw tensor data in any graph record.
    Gate 3: No parameter-level edges; block-level and structural edges only.
    Gate 4: Teacher checkpoint not required at student runtime.
    Gate 5: Champion scorer behavior is unchanged; this module is opt-in only.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .world_graph import WorldGraph, NodeRecord, EdgeRecord

# ── Constants ──────────────────────────────────────────────────────────────────

COMPILER_VERSION = "qwen_weight_graph.v1"
SCHEMA_VERSION_MANIFEST = "weight_graph_manifest.v1"
SCHEMA_VERSION_STATS = "weight_graph_stats.v1"

_ATTN_PROJS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_MLP_PROJS = frozenset({"gate_proj", "up_proj", "down_proj"})

_PROJ_ROLE_MAP: dict[str, str] = {
    "q_proj": "attention_query",
    "k_proj": "attention_key",
    "v_proj": "attention_value",
    "o_proj": "attention_output",
    "gate_proj": "mlp_gate",
    "up_proj": "mlp_up",
    "down_proj": "mlp_down",
}

_PROJ_TO_BLOCK_REL: dict[str, str] = {
    "q_proj": "qk_affinity_prior",
    "k_proj": "qk_affinity_prior",
    "v_proj": "value_flow_prior",
    "o_proj": "value_flow_prior",
    "gate_proj": "mlp_gate_flow",
    "up_proj": "mlp_up_flow",
    "down_proj": "mlp_down_flow",
}

_RE_LAYER = re.compile(r"^model\.layers\.(\d+)\.")
_RE_EXPERT = re.compile(r"\.experts\.(\d+)\.")
_RE_SHARED_EXPERT = re.compile(r"\.shared_expert\.")
_RE_PROJ = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.(weight|bias)$"
)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TensorRole:
    """Parsed structural role of one checkpoint tensor (P0.2 output)."""
    layer: int | None
    module: str | None
    projection: str | None
    expert_idx: int | None
    is_shared_expert: bool
    role: str
    parse_ok: bool


@dataclass
class TensorSpec:
    """Manifest entry for one checkpoint tensor (P0.1 output)."""
    name: str
    shard: str
    dtype: str
    shape: list[int]
    offset_start: int
    offset_end: int
    layer: int | None
    module: str | None
    projection: str | None
    expert_idx: int | None
    is_shared_expert: bool
    role: str
    parse_ok: bool


@dataclass
class WeightGraphNode:
    node_id: str
    node_type: str
    label: str
    features: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "node_id": self.node_id,
            "type": self.node_type,
            "label": self.label,
        }
        d.update(self.features)
        return d


@dataclass
class WeightGraphEdge:
    edge_id: str
    src_id: str
    dst_id: str
    rel: str
    weight: float
    score_name: str
    source_tensor: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "src": self.src_id,
            "rel": self.rel,
            "dst": self.dst_id,
            "weight": self.weight,
            "score_name": self.score_name,
            "source_tensor": self.source_tensor,
            "provenance": self.provenance,
        }


@dataclass
class WeightGraphManifest:
    source_model: str
    source_config_hash: str
    source_index_hash: str
    compiler_version: str
    block_size: int
    topk: int
    tensor_count: int
    emitted_node_count: int
    emitted_edge_count: int
    raw_weight_payload_in_graph: bool  # Gate 2: always False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION_MANIFEST,
            "source_model": self.source_model,
            "source_config_hash": self.source_config_hash,
            "source_index_hash": self.source_index_hash,
            "compiler_version": self.compiler_version,
            "block_size": self.block_size,
            "topk": self.topk,
            "tensor_count": self.tensor_count,
            "emitted_node_count": self.emitted_node_count,
            "emitted_edge_count": self.emitted_edge_count,
            "raw_weight_payload_in_graph": False,  # Gate 2: never store raw weights
        }


@dataclass
class WeightGraphResult:
    manifest: WeightGraphManifest
    nodes: list[WeightGraphNode]
    edges: list[WeightGraphEdge]
    stats: dict[str, Any]
    parse_failures: list[dict[str, Any]]


@dataclass
class SparseStudentExperimentConfig:
    """P2.1 stub config: derived G_0 + task labels -> sparse student experiment."""
    artifact_dir: Path
    task_examples_path: Path
    checkpoint: Path
    topk: int = 4
    block_size: int = 64
    device: str = "cpu"
    quality_baseline: float = 0.95
    memory_budget_mb: float = 1024.0


# ── Hash utilities ─────────────────────────────────────────────────────────────

def _hash_id(*parts: str) -> str:
    canonical = "\x00".join(parts)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── Safetensors utilities ──────────────────────────────────────────────────────

def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read safetensors JSON header without loading any tensor data."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            return {}
        header_len = struct.unpack("<Q", raw_len)[0]
        raw_header = f.read(header_len)
    header: dict[str, Any] = json.loads(raw_header.decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def _bytes_to_float32(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray | None:
    """Convert safetensors raw bytes to a float32 numpy array. Gate 2: no raw data stored."""
    try:
        if dtype == "BF16":
            # BF16 = top 16 bits of F32; reconstruct by shifting uint16 left 16 bits
            arr_u16 = np.frombuffer(raw, dtype=np.uint16)
            arr_u32 = arr_u16.astype(np.uint32) << 16
            arr: np.ndarray = arr_u32.view(np.float32).copy()
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dtype == "I32":
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        elif dtype == "I64":
            arr = np.frombuffer(raw, dtype=np.int64).astype(np.float32)
        elif dtype == "I16":
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        elif dtype in ("I8", "U8"):
            arr = np.frombuffer(raw, dtype=np.uint8 if dtype == "U8" else np.int8).astype(np.float32)
        else:
            return None
        return arr.reshape(shape)
    except (ValueError, struct.error, OverflowError, BufferError):
        return None


def _load_safetensors_tensor(path: Path, spec: TensorSpec) -> np.ndarray | None:
    """Stream one tensor from a safetensors shard without loading the full shard."""
    try:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            data_start = 8 + header_len
            f.seek(data_start + spec.offset_start)
            n_bytes = spec.offset_end - spec.offset_start
            raw = f.read(n_bytes)
    except (OSError, struct.error):
        return None
    return _bytes_to_float32(raw, spec.dtype, spec.shape)


# ── P0.2: Typed Qwen/MoE tensor name parser ───────────────────────────────────

def parse_qwen_tensor_name(name: str) -> TensorRole:
    """
    Parse a Qwen/dense/MoE checkpoint tensor name into a typed TensorRole.

    Recognises:
        model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
        model.layers.{L}.mlp.{gate,up,down}_proj.weight
        model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight
        model.layers.{L}.mlp.shared_expert.{gate,up,down}_proj.weight
        model.layers.{L}.mlp.gate.weight  (MoE router)
        model.layers.{L}.block_sparse_moe.gate.weight  (Mixtral-style)
        model.embed_tokens.weight
        model.norm.weight / model.final_norm.weight
        lm_head.weight
    Unknown tensors are returned with parse_ok=True and role='layer_weight' if layer
    is identifiable, or parse_ok=False and role='unknown' otherwise.
    """
    # ── Fast paths for well-known non-layer tensors ────────────────────────────
    if name.startswith("model.embed_tokens."):
        return TensorRole(None, "embed", None, None, False, "embed_tokens", True)
    if re.match(r"model\.(norm|final_norm)\.", name):
        return TensorRole(None, "norm", None, None, False, "final_norm", True)
    if name.startswith("lm_head."):
        return TensorRole(None, "lm_head", None, None, False, "lm_head", True)

    # ── Determine layer index ──────────────────────────────────────────────────
    m_layer = _RE_LAYER.match(name)
    if not m_layer:
        return TensorRole(None, None, None, None, False, "unknown", False)
    layer = int(m_layer.group(1))

    # ── MoE router: .mlp.gate.weight or .block_sparse_moe.gate.weight ─────────
    if re.search(r"\.(mlp|block_sparse_moe)\.gate\.(weight|bias)$", name):
        return TensorRole(layer, "mlp", "gate", None, False, "router", True)

    # ── Expert projections: .experts.{E}.{proj}.weight ────────────────────────
    m_expert = _RE_EXPERT.search(name)
    if m_expert:
        expert_idx = int(m_expert.group(1))
        m_proj = _RE_PROJ.search(name)
        if m_proj:
            projection = m_proj.group(1)
            role = f"expert_{_PROJ_ROLE_MAP.get(projection, projection)}"
            return TensorRole(layer, "mlp", projection, expert_idx, False, role, True)
        return TensorRole(layer, "mlp", None, expert_idx, False, "expert_weight", True)

    # ── Shared expert: .shared_expert.{proj}.weight ────────────────────────────
    if _RE_SHARED_EXPERT.search(name):
        m_proj = _RE_PROJ.search(name)
        if m_proj:
            projection = m_proj.group(1)
            role = f"shared_expert_{_PROJ_ROLE_MAP.get(projection, projection)}"
            return TensorRole(layer, "mlp", projection, None, True, role, True)
        return TensorRole(layer, "mlp", None, None, True, "shared_expert_weight", True)

    # ── Standard projection: self_attn.{q,k,v,o}_proj or mlp.{gate,up,down}_proj ──
    m_proj = _RE_PROJ.search(name)
    if m_proj:
        projection = m_proj.group(1)
        module = "self_attn" if projection in _ATTN_PROJS else "mlp"
        role = _PROJ_ROLE_MAP.get(projection, "weight")
        return TensorRole(layer, module, projection, None, False, role, True)

    # ── Layer norm and other per-layer tensors ─────────────────────────────────
    if re.search(r"(layernorm|layer_norm)\.", name, re.IGNORECASE):
        return TensorRole(layer, "layernorm", None, None, False, "layernorm", True)

    return TensorRole(layer, None, None, None, False, "layer_weight", True)


# ── P0.1: Tensor manifest compiler ────────────────────────────────────────────

def build_tensor_manifest_from_directory(
    checkpoint_dir: Path | str,
    source_model: str = "",
) -> tuple[list[TensorSpec], str, str]:
    """
    Build a tensor manifest by reading only safetensors headers (no tensor data loaded).

    Returns (specs, config_hash, index_hash). Supports single-file and sharded checkpoints.
    """
    checkpoint_dir = Path(checkpoint_dir)

    config_path = checkpoint_dir / "config.json"
    config_hash = _file_hash(config_path) if config_path.exists() else ""

    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"

    index_hash = ""
    # shard_filename -> sorted tensor names expected in that shard (None = read from header)
    shard_map: dict[str, list[str] | None] = {}

    if index_path.exists():
        index_hash = _file_hash(index_path)
        with index_path.open() as f:
            index = json.load(f)
        weight_map: dict[str, str] = index.get("weight_map", {})
        for tname, shard_file in weight_map.items():
            entry = shard_map.setdefault(shard_file, [])
            if entry is not None:
                entry.append(tname)
    elif single_path.exists():
        shard_map["model.safetensors"] = None

    specs: list[TensorSpec] = []
    for shard_file in shard_map:
        shard_path = checkpoint_dir / shard_file
        if not shard_path.exists():
            continue
        header = _read_safetensors_header(shard_path)
        for tensor_name, tensor_info in header.items():
            shape = list(tensor_info.get("shape", []))
            dtype = str(tensor_info.get("dtype", "F32"))
            offsets = tensor_info.get("data_offsets", [0, 0])
            tr = parse_qwen_tensor_name(tensor_name)
            specs.append(TensorSpec(
                name=tensor_name,
                shard=shard_file,
                dtype=dtype,
                shape=shape,
                offset_start=int(offsets[0]),
                offset_end=int(offsets[1]),
                layer=tr.layer,
                module=tr.module,
                projection=tr.projection,
                expert_idx=tr.expert_idx,
                is_shared_expert=tr.is_shared_expert,
                role=tr.role,
                parse_ok=tr.parse_ok,
            ))

    return specs, config_hash, index_hash


# ── P1.1: Block-energy edge compiler ──────────────────────────────────────────

def compute_block_scores(
    W: np.ndarray,
    block_size: int,
    topk: int,
) -> list[tuple[int, int, float]]:
    """
    Partition 2D tensor W into block_size×block_size blocks, score each block.

    Score: s_ab = ||W_ab||_F / sqrt(|W_ab|) = RMS norm of the block.
    Returns TopK highest-scoring (block_row, block_col, score) tuples per block row,
    sorted descending by score with stable tie-break on ascending block_col index.

    Gate 2: only derived float scores are returned; no raw tensor values.
    """
    if W.ndim != 2:
        raise ValueError(f"compute_block_scores requires 2D tensor, got shape {W.shape}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")

    d_out, d_in = W.shape
    n_rows = max(1, (d_out + block_size - 1) // block_size)
    n_cols = max(1, (d_in + block_size - 1) // block_size)

    W_f = W.astype(np.float64)
    pad_r = n_rows * block_size - d_out
    pad_c = n_cols * block_size - d_in
    if pad_r > 0 or pad_c > 0:
        W_f = np.pad(W_f, ((0, pad_r), (0, pad_c)))

    # Reshape to (n_rows, block_size, n_cols, block_size) then compute per-block norms
    W_blocks = W_f.reshape(n_rows, block_size, n_cols, block_size)
    sq_sums = np.einsum("ibjc,ibjc->ij", W_blocks, W_blocks)  # (n_rows, n_cols)
    scores = np.sqrt(sq_sums) / float(block_size)              # ||W_ab||_F / sqrt(|W_ab|)

    k = min(topk, n_cols)
    edges: list[tuple[int, int, float]] = []
    for br in range(n_rows):
        order = np.argsort(-scores[br], kind="stable")
        for bc in order[:k]:
            edges.append((br, int(bc), float(scores[br, bc])))

    return edges


# ── P1.2: Structural graph ─────────────────────────────────────────────────────

def _make_node(
    node_type: str,
    label: str,
    source_model: str,
    *id_parts: str,
    features: dict[str, Any] | None = None,
) -> WeightGraphNode:
    node_id = _hash_id(node_type, source_model, *id_parts)
    return WeightGraphNode(node_id=node_id, node_type=node_type, label=label,
                           features=features or {})


def _make_edge(
    src_id: str,
    dst_id: str,
    rel: str,
    *,
    weight: float = 0.0,
    score_name: str = "structural",
    source_tensor: str = "",
    provenance: dict[str, Any] | None = None,
) -> WeightGraphEdge:
    edge_id = _hash_id("edge", src_id, dst_id, rel)
    return WeightGraphEdge(edge_id=edge_id, src_id=src_id, dst_id=dst_id, rel=rel,
                           weight=weight, score_name=score_name, source_tensor=source_tensor,
                           provenance=provenance or {})


def _build_structural_graph(
    specs: list[TensorSpec],
    source_model: str,
) -> tuple[list[WeightGraphNode], list[WeightGraphEdge], dict[int, dict[str, str]]]:
    """
    Emit coarse structural nodes and edges from tensor metadata only (no weight data).

    Returns (nodes, edges, proj_node_ids) where proj_node_ids maps
    layer -> {proj_key -> node_id} for use by the block graph builder.

    Gate 3: no parameter-level edges; coarse structural edges only.
    """
    nodes: list[WeightGraphNode] = []
    edges: list[WeightGraphEdge] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    def _add_node(n: WeightGraphNode) -> None:
        if n.node_id not in seen_nodes:
            nodes.append(n)
            seen_nodes.add(n.node_id)

    def _add_edge(e: WeightGraphEdge) -> None:
        if e.edge_id not in seen_edges:
            edges.append(e)
            seen_edges.add(e.edge_id)

    # ── Model root ─────────────────────────────────────────────────────────────
    model_node = _make_node("model", f"model:{source_model}", source_model, "root",
                            features={"source_model": source_model})
    _add_node(model_node)

    # ── Collect layer indices, router/expert info ──────────────────────────────
    layers: set[int] = set()
    layer_has_router: set[int] = set()
    layer_expert_indices: dict[int, set[int]] = defaultdict(set)

    for spec in specs:
        if spec.layer is None:
            continue
        layers.add(spec.layer)
        if spec.role == "router":
            layer_has_router.add(spec.layer)
        elif spec.expert_idx is not None:
            layer_expert_indices[spec.layer].add(spec.expert_idx)

    # ── Layer nodes, contains edges, residual_next edges ──────────────────────
    sorted_layers = sorted(layers)
    layer_node_ids: dict[int, str] = {}
    for layer_idx in sorted_layers:
        ln = _make_node("layer", f"layer:{layer_idx}", source_model, str(layer_idx),
                        features={"layer": layer_idx})
        _add_node(ln)
        layer_node_ids[layer_idx] = ln.node_id
        _add_edge(_make_edge(model_node.node_id, ln.node_id, "contains",
                             provenance={"layer": layer_idx}))

    for i in range(len(sorted_layers) - 1):
        src = layer_node_ids[sorted_layers[i]]
        dst = layer_node_ids[sorted_layers[i + 1]]
        _add_edge(_make_edge(src, dst, "residual_next",
                             provenance={"from_layer": sorted_layers[i],
                                         "to_layer": sorted_layers[i + 1]}))

    # ── Projection nodes (non-expert) ─────────────────────────────────────────
    proj_node_ids: dict[int, dict[str, str]] = defaultdict(dict)

    for spec in specs:
        if spec.layer is None or spec.projection is None or spec.module is None:
            continue
        if spec.expert_idx is not None:
            continue  # handled in expert section below
        if spec.role == "router":
            continue  # handled in router section below

        proj_key = (f"shared_expert.{spec.projection}" if spec.is_shared_expert
                    else f"{spec.module}.{spec.projection}")
        if proj_key in proj_node_ids[spec.layer]:
            continue

        pn = _make_node(
            "projection",
            f"layer:{spec.layer}:{proj_key}",
            source_model,
            str(spec.layer), proj_key,
            features={
                "layer": spec.layer,
                "module": spec.module,
                "projection": spec.projection,
                "is_shared_expert": spec.is_shared_expert,
                "role": spec.role,
                "shape": spec.shape,
                "dtype": spec.dtype,
                "shard": spec.shard,
            },
        )
        _add_node(pn)
        proj_node_ids[spec.layer][proj_key] = pn.node_id
        if spec.layer in layer_node_ids:
            _add_edge(_make_edge(
                layer_node_ids[spec.layer], pn.node_id, "contains",
                source_tensor=spec.name,
                provenance={"layer": spec.layer, "proj_key": proj_key},
            ))

    # ── Structural attention / MLP cross-projection edges ─────────────────────
    for layer_idx in sorted_layers:
        pm = proj_node_ids[layer_idx]
        q = pm.get("self_attn.q_proj")
        k = pm.get("self_attn.k_proj")
        v = pm.get("self_attn.v_proj")
        o = pm.get("self_attn.o_proj")
        gate = pm.get("mlp.gate_proj")
        up = pm.get("mlp.up_proj")
        down = pm.get("mlp.down_proj")

        if q and k:
            _add_edge(_make_edge(q, k, "qk_affinity_prior",
                                 provenance={"layer": layer_idx}))
        if v and o:
            _add_edge(_make_edge(v, o, "value_flow_prior",
                                 provenance={"layer": layer_idx}))
        if gate and up:
            _add_edge(_make_edge(gate, up, "mlp_gate_flow",
                                 provenance={"layer": layer_idx}))
        if up and down:
            _add_edge(_make_edge(up, down, "mlp_up_flow",
                                 provenance={"layer": layer_idx}))

    # ── Router nodes ──────────────────────────────────────────────────────────
    layer_router_ids: dict[int, str] = {}
    for layer_idx in sorted(layer_has_router):
        router_spec: TensorSpec | None = next(
            (s for s in specs if s.layer == layer_idx and s.role == "router"), None
        )
        n_experts = (router_spec.shape[0]
                     if router_spec and len(router_spec.shape) >= 1 else 0)
        rn = _make_node("router", f"layer:{layer_idx}:router", source_model,
                        str(layer_idx), "router",
                        features={"layer": layer_idx, "n_experts": n_experts,
                                  "shape": router_spec.shape if router_spec else []})
        _add_node(rn)
        layer_router_ids[layer_idx] = rn.node_id
        if layer_idx in layer_node_ids:
            _add_edge(_make_edge(layer_node_ids[layer_idx], rn.node_id, "contains",
                                 provenance={"layer": layer_idx}))

    # ── Expert nodes and their projection nodes ────────────────────────────────
    for layer_idx in sorted(layer_expert_indices):
        router_id = layer_router_ids.get(layer_idx)
        for expert_idx in sorted(layer_expert_indices[layer_idx]):
            en = _make_node("expert", f"layer:{layer_idx}:expert:{expert_idx}",
                            source_model, str(layer_idx), str(expert_idx),
                            features={"layer": layer_idx, "expert_idx": expert_idx})
            _add_node(en)
            if router_id:
                _add_edge(_make_edge(router_id, en.node_id, "router_to_expert_prior",
                                     provenance={"layer": layer_idx,
                                                 "expert_idx": expert_idx}))
            for spec in specs:
                if spec.layer != layer_idx or spec.expert_idx != expert_idx:
                    continue
                if spec.projection is None:
                    continue
                proj_key = f"expert{expert_idx}.{spec.projection}"
                if proj_key in proj_node_ids[layer_idx]:
                    continue
                epn = _make_node(
                    "projection",
                    f"layer:{layer_idx}:expert:{expert_idx}:{spec.projection}",
                    source_model,
                    str(layer_idx), f"expert{expert_idx}", spec.projection,
                    features={
                        "layer": layer_idx,
                        "module": "mlp",
                        "projection": spec.projection,
                        "expert_idx": expert_idx,
                        "is_shared_expert": False,
                        "role": spec.role,
                        "shape": spec.shape,
                        "dtype": spec.dtype,
                        "shard": spec.shard,
                    },
                )
                _add_node(epn)
                proj_node_ids[layer_idx][proj_key] = epn.node_id
                _add_edge(_make_edge(en.node_id, epn.node_id, "expert_flow_prior",
                                     source_tensor=spec.name,
                                     provenance={"layer": layer_idx,
                                                 "expert_idx": expert_idx}))

    return nodes, edges, dict(proj_node_ids)


def _build_block_graph_for_tensor(
    spec: TensorSpec,
    W: np.ndarray,
    block_size: int,
    topk: int,
    source_model: str,
    proj_node_ids: dict[int, dict[str, str]],
) -> tuple[list[WeightGraphNode], list[WeightGraphEdge]]:
    """
    Emit channel_block / mlp_block nodes and block-flow edges for one 2D tensor.

    Output node per (tensor, block_row, dim_side='out') and (tensor, block_col, dim_side='in').
    Edges: tensor_contains_block (projection -> output block) and block-flow (in -> out).

    Gate 2: only Frobenius-norm scores are stored; W values are not.
    Gate 3: block-level only.
    """
    if W.ndim != 2:
        return [], []

    block_scores = compute_block_scores(W, block_size, topk)
    if not block_scores:
        return [], []

    d_out, d_in = W.shape
    nodes: list[WeightGraphNode] = []
    edges: list[WeightGraphEdge] = []

    proj = spec.projection
    is_expert = spec.expert_idx is not None
    rel = ("expert_flow_prior" if is_expert
           else _PROJ_TO_BLOCK_REL.get(proj or "", "weight_block_flow"))
    block_type = ("mlp_block" if spec.module == "mlp" and proj in _MLP_PROJS
                  else "channel_block")

    seen_br = sorted({br for br, _, _ in block_scores})
    seen_bc = sorted({bc for _, bc, _ in block_scores})

    out_ids: dict[int, str] = {}
    in_ids: dict[int, str] = {}

    for br in seen_br:
        nid = _hash_id(block_type, "out", source_model, spec.name, str(br))
        r_start, r_end = br * block_size, min((br + 1) * block_size, d_out)
        nodes.append(WeightGraphNode(
            node_id=nid, node_type=block_type,
            label=f"{spec.name}:out:{br}",
            features={
                "layer": spec.layer, "module": spec.module,
                "projection": spec.projection, "expert_idx": spec.expert_idx,
                "dim_side": "out", "block_index": br,
                "range": [r_start, r_end], "source_tensor": spec.name,
            },
        ))
        out_ids[br] = nid

    for bc in seen_bc:
        nid = _hash_id(block_type, "in", source_model, spec.name, str(bc))
        c_start, c_end = bc * block_size, min((bc + 1) * block_size, d_in)
        nodes.append(WeightGraphNode(
            node_id=nid, node_type=block_type,
            label=f"{spec.name}:in:{bc}",
            features={
                "layer": spec.layer, "module": spec.module,
                "projection": spec.projection, "expert_idx": spec.expert_idx,
                "dim_side": "in", "block_index": bc,
                "range": [c_start, c_end], "source_tensor": spec.name,
            },
        ))
        in_ids[bc] = nid

    # tensor_contains_block: projection node -> output channel_block nodes
    proj_key: str | None = None
    if spec.layer is not None and spec.projection and spec.module:
        if spec.expert_idx is not None:
            proj_key = f"expert{spec.expert_idx}.{spec.projection}"
        elif spec.is_shared_expert:
            proj_key = f"shared_expert.{spec.projection}"
        else:
            proj_key = f"{spec.module}.{spec.projection}"
    proj_nid = (proj_node_ids.get(spec.layer, {}).get(proj_key)  # type: ignore[arg-type]
                if proj_key is not None and spec.layer is not None else None)

    for br, out_nid in out_ids.items():
        if proj_nid:
            eid = _hash_id("edge", proj_nid, out_nid, "tensor_contains_block")
            edges.append(WeightGraphEdge(
                edge_id=eid, src_id=proj_nid, dst_id=out_nid,
                rel="tensor_contains_block", weight=0.0,
                score_name="structural", source_tensor=spec.name,
                provenance={"block_index": br, "dim_side": "out"},
            ))

    # Block-flow edges: in_node(bc) -> out_node(br) with Frobenius-norm score
    for br, bc, score in block_scores:
        src_nid = in_ids.get(bc)
        dst_nid = out_ids.get(br)
        if src_nid is None or dst_nid is None:
            continue
        eid = _hash_id("edge", src_nid, dst_nid, rel)
        edges.append(WeightGraphEdge(
            edge_id=eid, src_id=src_nid, dst_id=dst_nid,
            rel=rel, weight=float(score),
            score_name="normalized_frobenius",
            source_tensor=spec.name,
            provenance={"shard": spec.shard, "block_in": bc, "block_out": br},
        ))

    return nodes, edges


# ── QwenWeightGraphCompiler ────────────────────────────────────────────────────

class QwenWeightGraphCompiler:
    """
    Compile a Qwen/dense/MoE checkpoint into a derived graph artifact G_0.

    The resulting artifact contains only derived statistics and structural metadata;
    no raw transformer weights are stored (Gate 2). The artifact is independent of
    the teacher checkpoint at runtime (Gate 4). Champion scorer is unaffected (Gate 5).
    """

    def __init__(
        self,
        block_size: int = 64,
        topk: int = 4,
        source_model: str = "unknown",
    ) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        self.block_size = block_size
        self.topk = topk
        self.source_model = source_model

    def compile(
        self,
        specs: list[TensorSpec],
        tensor_loader: Callable[[TensorSpec], np.ndarray | None] | None = None,
        *,
        config_hash: str = "",
        index_hash: str = "",
    ) -> WeightGraphResult:
        """
        Compile from a TensorSpec list into a WeightGraphResult.

        tensor_loader: called per eligible 2D spec; returns float32 array or None.
        If None, only the structural graph is built (manifest-only mode).
        """
        parse_failures = [
            {"name": s.name, "shard": s.shard, "dtype": s.dtype,
             "shape": s.shape, "role": s.role}
            for s in specs if not s.parse_ok
        ]

        struct_nodes, struct_edges, proj_node_ids = _build_structural_graph(
            specs, self.source_model
        )

        block_nodes: list[WeightGraphNode] = []
        block_edges: list[WeightGraphEdge] = []
        eligible_count = 0

        if tensor_loader is not None:
            for spec in specs:
                if not spec.parse_ok or len(spec.shape) != 2:
                    continue
                W = tensor_loader(spec)
                if W is None:
                    continue
                eligible_count += 1
                bn, be = _build_block_graph_for_tensor(
                    spec, W, self.block_size, self.topk,
                    self.source_model, proj_node_ids,
                )
                block_nodes.extend(bn)
                block_edges.extend(be)

        # Deduplicate nodes (in practice struct and block nodes do not overlap)
        seen_ids: set[str] = set()
        all_nodes: list[WeightGraphNode] = []
        for n in struct_nodes + block_nodes:
            if n.node_id not in seen_ids:
                all_nodes.append(n)
                seen_ids.add(n.node_id)

        all_edges = struct_edges + block_edges

        manifest = WeightGraphManifest(
            source_model=self.source_model,
            source_config_hash=config_hash,
            source_index_hash=index_hash,
            compiler_version=COMPILER_VERSION,
            block_size=self.block_size,
            topk=self.topk,
            tensor_count=len(specs),
            emitted_node_count=len(all_nodes),
            emitted_edge_count=len(all_edges),
            raw_weight_payload_in_graph=False,
        )

        node_by_type: dict[str, int] = defaultdict(int)
        for n in all_nodes:
            node_by_type[n.node_type] += 1
        edge_by_rel: dict[str, int] = defaultdict(int)
        for e in all_edges:
            edge_by_rel[e.rel] += 1

        stats: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION_STATS,
            "total_tensors": len(specs),
            "parse_ok_count": sum(1 for s in specs if s.parse_ok),
            "parse_fail_count": len(parse_failures),
            "eligible_2d_count": eligible_count,
            "node_count_by_type": dict(node_by_type),
            "edge_count_by_rel": dict(edge_by_rel),
            "layer_count": len({s.layer for s in specs if s.layer is not None}),
            "expert_count": len({
                (s.layer, s.expert_idx)
                for s in specs if s.expert_idx is not None
            }),
            "block_size": self.block_size,
            "topk": self.topk,
        }

        return WeightGraphResult(
            manifest=manifest,
            nodes=all_nodes,
            edges=all_edges,
            stats=stats,
            parse_failures=parse_failures,
        )

    def compile_from_directory(
        self,
        checkpoint_dir: Path | str,
        output_dir: Path | str | None = None,
    ) -> WeightGraphResult:
        """Full pipeline: read checkpoint directory, compile G_0, optionally write artifacts."""
        checkpoint_dir = Path(checkpoint_dir)
        specs, config_hash, index_hash = build_tensor_manifest_from_directory(
            checkpoint_dir, self.source_model
        )
        loader = SafetensorsTensorLoader(checkpoint_dir)
        result = self.compile(specs, loader, config_hash=config_hash, index_hash=index_hash)
        if output_dir is not None:
            write_weight_graph_artifacts(result, Path(output_dir))
        return result


class SafetensorsTensorLoader:
    """Stream one tensor at a time from safetensors shards (no full shard in memory)."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)

    def __call__(self, spec: TensorSpec) -> np.ndarray | None:
        shard_path = self.base_dir / spec.shard
        if not shard_path.exists():
            return None
        return _load_safetensors_tensor(shard_path, spec)


# ── Artifact I/O ──────────────────────────────────────────────────────────────

def write_weight_graph_artifacts(result: WeightGraphResult, output_dir: Path) -> None:
    """Write manifest.json, nodes.jsonl, edges.jsonl, stats.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "manifest.json").write_text(
        json.dumps(result.manifest.as_dict(), indent=2)
    )
    with (output_dir / "nodes.jsonl").open("w") as f:
        for node in result.nodes:
            f.write(json.dumps(node.as_dict()) + "\n")
    with (output_dir / "edges.jsonl").open("w") as f:
        for edge in result.edges:
            f.write(json.dumps(edge.as_dict()) + "\n")
    (output_dir / "stats.json").write_text(json.dumps(result.stats, indent=2))

    if result.parse_failures:
        with (output_dir / "tensor_parse_failures.jsonl").open("w") as f:
            for rec in result.parse_failures:
                f.write(json.dumps(rec) + "\n")


def read_weight_graph_artifacts(artifact_dir: Path | str) -> WeightGraphResult:
    """Read a compiled G_0 artifact directory back into a WeightGraphResult."""
    artifact_dir = Path(artifact_dir)

    raw = json.loads((artifact_dir / "manifest.json").read_text())
    manifest = WeightGraphManifest(
        source_model=raw.get("source_model", ""),
        source_config_hash=raw.get("source_config_hash", ""),
        source_index_hash=raw.get("source_index_hash", ""),
        compiler_version=raw.get("compiler_version", COMPILER_VERSION),
        block_size=raw.get("block_size", 64),
        topk=raw.get("topk", 4),
        tensor_count=raw.get("tensor_count", 0),
        emitted_node_count=raw.get("emitted_node_count", 0),
        emitted_edge_count=raw.get("emitted_edge_count", 0),
        raw_weight_payload_in_graph=False,  # always False — reject any True from disk
    )

    nodes: list[WeightGraphNode] = []
    with (artifact_dir / "nodes.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            nid = d.pop("node_id")
            ntype = d.pop("type")
            label = d.pop("label")
            nodes.append(WeightGraphNode(node_id=nid, node_type=ntype, label=label,
                                         features=d))

    edges: list[WeightGraphEdge] = []
    with (artifact_dir / "edges.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            edges.append(WeightGraphEdge(
                edge_id=d.get("edge_id", ""),
                src_id=d.get("src", ""),
                dst_id=d.get("dst", ""),
                rel=d.get("rel", ""),
                weight=float(d.get("weight", 0.0)),
                score_name=d.get("score_name", ""),
                source_tensor=d.get("source_tensor", ""),
                provenance=d.get("provenance", {}),
            ))

    stats_path = artifact_dir / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    failures_path = artifact_dir / "tensor_parse_failures.jsonl"
    failures: list[dict[str, Any]] = []
    if failures_path.exists():
        with failures_path.open() as f:
            for line in f:
                failures.append(json.loads(line))

    return WeightGraphResult(manifest=manifest, nodes=nodes, edges=edges,
                             stats=stats, parse_failures=failures)


# ── P1.3: WorldGraph schema adapter ───────────────────────────────────────────

def load_weight_graph_as_world_graph(artifact_dir: Path | str) -> WorldGraph:
    """
    Load a compiled G_0 artifact as a WorldGraph for planner/scorer/student use (P1.3).

    Edges referencing missing nodes are silently skipped.
    Gate 5: does not affect champion scorer or existing topology experiments; opt-in only.
    """
    result = read_weight_graph_artifacts(Path(artifact_dir))
    world = WorldGraph()

    for node in result.nodes:
        world.add_node(NodeRecord(
            node_id=node.node_id,
            label=node.label,
            node_kind=node.node_type,
            features=node.features,
        ))

    for edge in result.edges:
        if not (world.is_node_active(edge.src_id) and world.is_node_active(edge.dst_id)):
            continue
        try:
            world.add_edge(EdgeRecord(
                edge_id=edge.edge_id,
                src_id=edge.src_id,
                dst_id=edge.dst_id,
                relation=edge.rel,
                weight=edge.weight,
                metadata=edge.provenance,
            ))
        except ValueError:
            pass

    return world


# ── P2.1: Sparse student stub ─────────────────────────────────────────────────

def run_sparse_student_stub(config: SparseStudentExperimentConfig) -> dict[str, Any]:
    """
    Minimal stub: G_0 artifact + task config -> sparse student experiment entry point (P2.1).

    Full KL distillation training is a later target (plan v23 P2.1, later stage).
    Gate 4: does not load the teacher checkpoint; only the derived G_0 artifact.
    Gate 5: does not change champion scorer behavior.
    """
    artifact_dir = Path(config.artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "status": "artifact_missing",
            "artifact_dir": str(artifact_dir),
            "error": f"manifest.json not found at {manifest_path}",
        }

    raw = json.loads(manifest_path.read_text())
    if raw.get("raw_weight_payload_in_graph", False):
        return {
            "status": "gate_violation",
            "error": "Gate 2 violation: raw_weight_payload_in_graph=true in manifest",
        }

    return {
        "status": "stub_ok",
        "artifact_dir": str(artifact_dir),
        "source_model": raw.get("source_model", ""),
        "compiler_version": raw.get("compiler_version", ""),
        "block_size": config.block_size,
        "topk": config.topk,
        "device": config.device,
        "quality_baseline": config.quality_baseline,
        "memory_budget_mb": config.memory_budget_mb,
        "g0_node_count": raw.get("emitted_node_count", 0),
        "g0_edge_count": raw.get("emitted_edge_count", 0),
        "raw_weight_payload_in_graph": False,
        "note": "stub; full KL distillation is plan v23 P2.1 later target",
    }
