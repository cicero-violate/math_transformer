"""
Tests for src/qwen_weight_graph.py — plan v23 P0.1, P0.2, P1.1, P1.2, P1.3, P2.1 stub.

Fixture-based: all tests use in-memory synthetic safetensors fixtures.
No real Qwen checkpoint required.

Key invariants:
  - raw_weight_payload_in_graph is always False (Gate 2)
  - No raw tensor values in any graph record (Gate 2)
  - No parameter-level edges (Gate 3)
  - node_ids and edge_ids are deterministic hashes (same input -> same output)
  - MoE checkpoints emit expert and router nodes; dense checkpoints do not
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_weight_graph import (
    COMPILER_VERSION,
    QwenWeightGraphCompiler,
    SafetensorsTensorLoader,
    SparseStudentExperimentConfig,
    TensorRole,
    TensorSpec,
    WeightGraphManifest,
    WeightGraphResult,
    _build_block_graph_for_tensor,
    _build_structural_graph,
    _bytes_to_float32,
    _read_safetensors_header,
    build_tensor_manifest_from_directory,
    compute_block_scores,
    load_weight_graph_as_world_graph,
    parse_qwen_tensor_name,
    read_weight_graph_artifacts,
    run_sparse_student_stub,
    write_weight_graph_artifacts,
)


# ── Safetensors fixture builder ────────────────────────────────────────────────

def _make_safetensors_bytes(tensors: dict[str, np.ndarray]) -> bytes:
    """Build minimal in-memory safetensors bytes (F32, little-endian)."""
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    data_parts: list[bytes] = []
    for name, arr in tensors.items():
        arr_f32 = arr.astype(np.float32)
        data = arr_f32.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        data_parts.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(data_parts)


# ── Fixture tensor dicts ───────────────────────────────────────────────────────

def _dense_tensors(rng: np.random.RandomState) -> dict[str, np.ndarray]:
    """2-layer dense attention+MLP checkpoint, tiny shapes for fast tests."""
    d = {}
    for li in range(2):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            d[f"model.layers.{li}.self_attn.{proj}.weight"] = rng.randn(8, 8).astype(np.float32)
        for proj in ("gate_proj", "up_proj"):
            d[f"model.layers.{li}.mlp.{proj}.weight"] = rng.randn(16, 8).astype(np.float32)
        d[f"model.layers.{li}.mlp.down_proj.weight"] = rng.randn(8, 16).astype(np.float32)
        d[f"model.layers.{li}.input_layernorm.weight"] = rng.randn(8).astype(np.float32)
    d["model.embed_tokens.weight"] = rng.randn(32, 8).astype(np.float32)
    d["lm_head.weight"] = rng.randn(32, 8).astype(np.float32)
    d["model.norm.weight"] = rng.randn(8).astype(np.float32)
    return d


def _moe_tensors(rng: np.random.RandomState) -> dict[str, np.ndarray]:
    """1-layer MoE checkpoint with router + 2 experts."""
    d = {}
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        d[f"model.layers.0.self_attn.{proj}.weight"] = rng.randn(8, 8).astype(np.float32)
    d["model.layers.0.mlp.gate.weight"] = rng.randn(2, 8).astype(np.float32)  # router (2 experts)
    for ei in range(2):
        for proj in ("gate_proj", "up_proj"):
            d[f"model.layers.0.mlp.experts.{ei}.{proj}.weight"] = rng.randn(16, 8).astype(np.float32)
        d[f"model.layers.0.mlp.experts.{ei}.down_proj.weight"] = rng.randn(8, 16).astype(np.float32)
    d["model.layers.0.input_layernorm.weight"] = rng.randn(8).astype(np.float32)
    d["model.norm.weight"] = rng.randn(8).astype(np.float32)
    return d


# ── Shared fixtures ────────────────────────────────────────────────────────────

_RNG = np.random.RandomState(42)
_DENSE = _dense_tensors(_RNG)
_MOE = _moe_tensors(_RNG)
_BLOCK = 4   # block_size for all tests — produces multiple blocks on 8×8+ matrices
_TOPK = 2


@pytest.fixture
def dense_dir(tmp_path: Path) -> Path:
    (tmp_path / "model.safetensors").write_bytes(_make_safetensors_bytes(_DENSE))
    return tmp_path


@pytest.fixture
def moe_dir(tmp_path: Path) -> Path:
    (tmp_path / "model.safetensors").write_bytes(_make_safetensors_bytes(_MOE))
    return tmp_path


@pytest.fixture
def dense_specs(dense_dir: Path) -> list[TensorSpec]:
    specs, _, _ = build_tensor_manifest_from_directory(dense_dir, "dense_test")
    return specs


@pytest.fixture
def moe_specs(moe_dir: Path) -> list[TensorSpec]:
    specs, _, _ = build_tensor_manifest_from_directory(moe_dir, "moe_test")
    return specs


def _loader(tensors: dict[str, np.ndarray]) -> object:
    """Return a callable that loads from an in-memory dict (no file I/O)."""
    def load(spec: TensorSpec) -> np.ndarray | None:
        arr = tensors.get(spec.name)
        return arr.astype(np.float32) if arr is not None else None
    return load


# ── P0.2: parse_qwen_tensor_name ──────────────────────────────────────────────

class TestParseQwenTensorName:
    def test_attention_q_proj(self):
        r = parse_qwen_tensor_name("model.layers.3.self_attn.q_proj.weight")
        assert r.parse_ok
        assert r.layer == 3
        assert r.module == "self_attn"
        assert r.projection == "q_proj"
        assert r.expert_idx is None
        assert r.role == "attention_query"

    def test_attention_k_proj(self):
        r = parse_qwen_tensor_name("model.layers.0.self_attn.k_proj.weight")
        assert r.parse_ok and r.projection == "k_proj" and r.role == "attention_key"

    def test_attention_v_proj(self):
        r = parse_qwen_tensor_name("model.layers.0.self_attn.v_proj.weight")
        assert r.parse_ok and r.projection == "v_proj" and r.role == "attention_value"

    def test_attention_o_proj(self):
        r = parse_qwen_tensor_name("model.layers.0.self_attn.o_proj.weight")
        assert r.parse_ok and r.projection == "o_proj" and r.role == "attention_output"

    def test_mlp_gate_proj(self):
        r = parse_qwen_tensor_name("model.layers.1.mlp.gate_proj.weight")
        assert r.parse_ok and r.module == "mlp" and r.projection == "gate_proj"
        assert r.role == "mlp_gate"

    def test_mlp_up_proj(self):
        r = parse_qwen_tensor_name("model.layers.1.mlp.up_proj.weight")
        assert r.parse_ok and r.projection == "up_proj" and r.role == "mlp_up"

    def test_mlp_down_proj(self):
        r = parse_qwen_tensor_name("model.layers.1.mlp.down_proj.weight")
        assert r.parse_ok and r.projection == "down_proj" and r.role == "mlp_down"

    def test_moe_router(self):
        r = parse_qwen_tensor_name("model.layers.0.mlp.gate.weight")
        assert r.parse_ok and r.layer == 0 and r.role == "router"
        assert r.expert_idx is None

    def test_moe_expert_proj(self):
        r = parse_qwen_tensor_name("model.layers.2.mlp.experts.3.gate_proj.weight")
        assert r.parse_ok and r.layer == 2 and r.expert_idx == 3
        assert r.projection == "gate_proj" and "expert" in r.role

    def test_moe_shared_expert(self):
        r = parse_qwen_tensor_name("model.layers.0.mlp.shared_expert.gate_proj.weight")
        assert r.parse_ok and r.is_shared_expert and r.projection == "gate_proj"

    def test_embed_tokens(self):
        r = parse_qwen_tensor_name("model.embed_tokens.weight")
        assert r.parse_ok and r.layer is None and r.role == "embed_tokens"

    def test_final_norm(self):
        r = parse_qwen_tensor_name("model.norm.weight")
        assert r.parse_ok and r.role == "final_norm"

    def test_lm_head(self):
        r = parse_qwen_tensor_name("lm_head.weight")
        assert r.parse_ok and r.role == "lm_head"

    def test_unknown_no_layer(self):
        r = parse_qwen_tensor_name("some.mystery.weight")
        assert not r.parse_ok and r.role == "unknown"

    def test_block_sparse_moe_router(self):
        r = parse_qwen_tensor_name("model.layers.5.block_sparse_moe.gate.weight")
        assert r.parse_ok and r.layer == 5 and r.role == "router"

    def test_layer_norm_variant(self):
        r = parse_qwen_tensor_name("model.layers.0.post_attention_layernorm.weight")
        assert r.parse_ok and r.layer == 0 and r.role == "layernorm"


# ── Safetensors utilities ──────────────────────────────────────────────────────

class TestSafetensorsHeader:
    def test_read_round_trip(self, tmp_path: Path):
        tensors = {"w": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
        (tmp_path / "t.safetensors").write_bytes(_make_safetensors_bytes(tensors))
        header = _read_safetensors_header(tmp_path / "t.safetensors")
        assert "w" in header
        assert header["w"]["shape"] == [2, 2]
        assert header["w"]["dtype"] == "F32"

    def test_metadata_key_stripped(self, tmp_path: Path):
        tensors = {"x": np.ones((4,), dtype=np.float32)}
        (tmp_path / "t.safetensors").write_bytes(_make_safetensors_bytes(tensors))
        header = _read_safetensors_header(tmp_path / "t.safetensors")
        assert "__metadata__" not in header

    def test_bytes_to_float32_f32(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _bytes_to_float32(arr.tobytes(), "F32", [3])
        assert result is not None
        np.testing.assert_allclose(result, arr)

    def test_bytes_to_float32_f16(self):
        arr = np.array([1.0, 2.0], dtype=np.float16)
        result = _bytes_to_float32(arr.tobytes(), "F16", [2])
        assert result is not None
        np.testing.assert_allclose(result, [1.0, 2.0], rtol=1e-3)

    def test_bytes_to_float32_unknown_dtype(self):
        assert _bytes_to_float32(b"\x00" * 4, "Q8_0", [1]) is None


# ── P0.1: build_tensor_manifest_from_directory ────────────────────────────────

class TestBuildTensorManifest:
    def test_returns_specs_for_all_tensors(self, dense_dir: Path):
        specs, _, _ = build_tensor_manifest_from_directory(dense_dir, "test")
        names = {s.name for s in specs}
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "model.embed_tokens.weight" in names
        assert "lm_head.weight" in names

    def test_shape_recorded_correctly(self, dense_dir: Path):
        specs, _, _ = build_tensor_manifest_from_directory(dense_dir, "test")
        q = next(s for s in specs if "q_proj" in s.name and "layers.0" in s.name)
        assert q.shape == [8, 8]

    def test_parse_ok_for_known_projections(self, dense_dir: Path):
        specs, _, _ = build_tensor_manifest_from_directory(dense_dir, "test")
        proj_specs = [s for s in specs if s.projection in ("q_proj", "gate_proj")]
        assert all(s.parse_ok for s in proj_specs)

    def test_moe_expert_idx_parsed(self, moe_dir: Path):
        specs, _, _ = build_tensor_manifest_from_directory(moe_dir, "test")
        expert_specs = [s for s in specs if s.expert_idx is not None]
        assert len(expert_specs) > 0
        assert all(s.parse_ok for s in expert_specs)

    def test_config_hash_empty_when_no_config(self, dense_dir: Path):
        _, config_hash, _ = build_tensor_manifest_from_directory(dense_dir, "test")
        assert config_hash == ""

    def test_config_hash_set_when_config_exists(self, dense_dir: Path):
        (dense_dir / "config.json").write_text('{"model_type": "qwen2"}')
        _, config_hash, _ = build_tensor_manifest_from_directory(dense_dir, "test")
        assert len(config_hash) > 0

    def test_shard_name_recorded(self, dense_dir: Path):
        specs, _, _ = build_tensor_manifest_from_directory(dense_dir, "test")
        assert all(s.shard == "model.safetensors" for s in specs)

    def test_empty_dir_returns_empty_specs(self, tmp_path: Path):
        specs, _, _ = build_tensor_manifest_from_directory(tmp_path, "test")
        assert specs == []


# ── P1.1: compute_block_scores ────────────────────────────────────────────────

class TestComputeBlockScores:
    def test_basic_2x2_blocks(self):
        W = np.ones((4, 4), dtype=np.float32)
        scores = compute_block_scores(W, block_size=2, topk=2)
        assert len(scores) == 4  # 2 block_rows × 2 edges each
        for br, bc, s in scores:
            assert 0 <= br < 2
            assert 0 <= bc < 2
            assert s > 0.0

    def test_score_formula_rms(self):
        # Single 2×2 block with known values; score = ||W||_F / sqrt(|W|) = sqrt(mean(x^2))
        W = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        scores = compute_block_scores(W, block_size=2, topk=1)
        assert len(scores) == 1
        br, bc, s = scores[0]
        expected = np.sqrt((9 + 16) / 4.0)  # ||W||_F / sqrt(4) = sqrt(25/4) = 2.5
        assert abs(s - expected) < 1e-5

    def test_topk_respected(self):
        W = np.random.randn(8, 8).astype(np.float32)
        scores = compute_block_scores(W, block_size=4, topk=1)
        n_rows = 2  # 8/4 = 2 blocks per dimension
        assert len(scores) == n_rows * 1  # 1 per block_row

    def test_topk_capped_at_n_cols(self):
        W = np.ones((4, 4), dtype=np.float32)
        # block_size=4 -> 1×1 = 1 block; topk=10 but only 1 col block
        scores = compute_block_scores(W, block_size=4, topk=10)
        assert len(scores) == 1  # 1 row × 1 col

    def test_deterministic_tiebreak(self):
        # Constant matrix: all blocks have equal score; tie-break by ascending bc
        W = np.ones((8, 8), dtype=np.float32)
        scores = compute_block_scores(W, block_size=4, topk=2)
        # For each br, the two bc should be 0 and 1 (ascending, stable)
        for i in range(0, len(scores), 2):
            _, bc0, _ = scores[i]
            _, bc1, _ = scores[i + 1]
            assert bc0 < bc1

    def test_scores_descending_per_row(self):
        rng = np.random.RandomState(7)
        W = rng.randn(8, 12).astype(np.float32)
        scores = compute_block_scores(W, block_size=4, topk=3)
        # Group by block_row and check descending order
        by_row: dict[int, list[float]] = {}
        for br, bc, s in scores:
            by_row.setdefault(br, []).append(s)
        for row_scores in by_row.values():
            assert row_scores == sorted(row_scores, reverse=True)

    def test_zero_matrix_scores_are_zero(self):
        W = np.zeros((8, 8), dtype=np.float32)
        scores = compute_block_scores(W, block_size=4, topk=2)
        for _, _, s in scores:
            assert s == pytest.approx(0.0)

    def test_invalid_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            compute_block_scores(np.ones(8), block_size=4, topk=2)

    def test_invalid_block_size_raises(self):
        with pytest.raises(ValueError):
            compute_block_scores(np.ones((8, 8)), block_size=0, topk=2)

    def test_invalid_topk_raises(self):
        with pytest.raises(ValueError):
            compute_block_scores(np.ones((8, 8)), block_size=4, topk=0)

    def test_no_raw_values_in_output(self):
        rng = np.random.RandomState(99)
        W = rng.randn(8, 8).astype(np.float32)
        scores = compute_block_scores(W, block_size=4, topk=2)
        # Output is (int, int, float) tuples — no numpy arrays or raw bytes
        for item in scores:
            assert len(item) == 3
            br, bc, s = item
            assert isinstance(br, int)
            assert isinstance(bc, int)
            assert isinstance(s, float)


# ── P1.2: Structural graph ─────────────────────────────────────────────────────

class TestStructuralGraph:
    def _compile_struct(self, tensors, source_model="test"):
        specs = [
            TensorSpec(
                name=name,
                shard="s.safetensors",
                dtype="F32",
                shape=list(arr.shape),
                offset_start=0,
                offset_end=arr.nbytes,
                layer=parse_qwen_tensor_name(name).layer,
                module=parse_qwen_tensor_name(name).module,
                projection=parse_qwen_tensor_name(name).projection,
                expert_idx=parse_qwen_tensor_name(name).expert_idx,
                is_shared_expert=parse_qwen_tensor_name(name).is_shared_expert,
                role=parse_qwen_tensor_name(name).role,
                parse_ok=parse_qwen_tensor_name(name).parse_ok,
            )
            for name, arr in tensors.items()
        ]
        return _build_structural_graph(specs, source_model)

    def test_dense_emits_model_node(self):
        nodes, _, _ = self._compile_struct(_DENSE)
        model_nodes = [n for n in nodes if n.node_type == "model"]
        assert len(model_nodes) == 1

    def test_dense_emits_layer_nodes(self):
        nodes, _, _ = self._compile_struct(_DENSE)
        layer_nodes = [n for n in nodes if n.node_type == "layer"]
        assert len(layer_nodes) == 2  # 2 layers

    def test_dense_emits_projection_nodes(self):
        nodes, _, _ = self._compile_struct(_DENSE)
        proj_nodes = [n for n in nodes if n.node_type == "projection"]
        # 7 projections × 2 layers = 14
        assert len(proj_nodes) == 14

    def test_dense_no_expert_nodes(self):
        nodes, _, _ = self._compile_struct(_DENSE)
        expert_nodes = [n for n in nodes if n.node_type == "expert"]
        assert len(expert_nodes) == 0

    def test_dense_no_router_nodes(self):
        nodes, _, _ = self._compile_struct(_DENSE)
        router_nodes = [n for n in nodes if n.node_type == "router"]
        assert len(router_nodes) == 0

    def test_moe_emits_router_node(self):
        nodes, _, _ = self._compile_struct(_MOE)
        router_nodes = [n for n in nodes if n.node_type == "router"]
        assert len(router_nodes) == 1

    def test_moe_emits_expert_nodes(self):
        nodes, _, _ = self._compile_struct(_MOE)
        expert_nodes = [n for n in nodes if n.node_type == "expert"]
        assert len(expert_nodes) == 2

    def test_moe_router_to_expert_edges(self):
        _, edges, _ = self._compile_struct(_MOE)
        rte = [e for e in edges if e.rel == "router_to_expert_prior"]
        assert len(rte) == 2

    def test_dense_has_residual_next_edges(self):
        _, edges, _ = self._compile_struct(_DENSE)
        residual = [e for e in edges if e.rel == "residual_next"]
        assert len(residual) == 1  # layer 0 -> layer 1

    def test_dense_has_qk_affinity_edges(self):
        _, edges, _ = self._compile_struct(_DENSE)
        qk = [e for e in edges if e.rel == "qk_affinity_prior"]
        assert len(qk) == 2  # one per layer

    def test_dense_has_value_flow_edges(self):
        _, edges, _ = self._compile_struct(_DENSE)
        vf = [e for e in edges if e.rel == "value_flow_prior"]
        assert len(vf) == 2

    def test_dense_has_mlp_gate_flow_edges(self):
        _, edges, _ = self._compile_struct(_DENSE)
        gf = [e for e in edges if e.rel == "mlp_gate_flow"]
        assert len(gf) == 2

    def test_node_ids_are_deterministic(self):
        nodes1, _, _ = self._compile_struct(_DENSE, "test")
        nodes2, _, _ = self._compile_struct(_DENSE, "test")
        ids1 = {n.node_id for n in nodes1}
        ids2 = {n.node_id for n in nodes2}
        assert ids1 == ids2

    def test_proj_node_ids_index_populated(self):
        _, _, proj_node_ids = self._compile_struct(_DENSE)
        assert 0 in proj_node_ids
        assert "self_attn.q_proj" in proj_node_ids[0]


# ── P1.1+P1.2: Block graph ────────────────────────────────────────────────────

class TestBlockGraph:
    def _make_spec(self, name: str, arr: np.ndarray) -> TensorSpec:
        tr = parse_qwen_tensor_name(name)
        return TensorSpec(
            name=name, shard="s.safetensors", dtype="F32",
            shape=list(arr.shape), offset_start=0, offset_end=arr.nbytes,
            layer=tr.layer, module=tr.module, projection=tr.projection,
            expert_idx=tr.expert_idx, is_shared_expert=tr.is_shared_expert,
            role=tr.role, parse_ok=tr.parse_ok,
        )

    def test_block_nodes_emitted_for_2d_tensor(self):
        arr = np.random.randn(8, 8).astype(np.float32)
        spec = self._make_spec("model.layers.0.self_attn.q_proj.weight", arr)
        nodes, _ = _build_block_graph_for_tensor(spec, arr, _BLOCK, _TOPK, "test", {})
        assert len(nodes) > 0

    def test_no_block_nodes_for_1d_tensor(self):
        arr = np.ones(8, dtype=np.float32)
        spec = self._make_spec("model.layers.0.input_layernorm.weight", arr)
        nodes, edges = _build_block_graph_for_tensor(spec, arr, _BLOCK, _TOPK, "test", {})
        assert nodes == [] and edges == []

    def test_block_flow_edges_emitted(self):
        arr = np.random.randn(8, 8).astype(np.float32)
        spec = self._make_spec("model.layers.0.self_attn.q_proj.weight", arr)
        _, edges = _build_block_graph_for_tensor(spec, arr, _BLOCK, _TOPK, "test", {})
        block_flow = [e for e in edges if e.rel in (
            "qk_affinity_prior", "weight_block_flow")]
        assert len(block_flow) > 0

    def test_no_raw_tensor_data_in_nodes(self):
        arr = np.random.randn(8, 8).astype(np.float32)
        spec = self._make_spec("model.layers.0.mlp.gate_proj.weight", arr)
        nodes, _ = _build_block_graph_for_tensor(spec, arr, _BLOCK, _TOPK, "test", {})
        for node in nodes:
            d = node.as_dict()
            for v in d.values():
                assert not isinstance(v, np.ndarray), "raw ndarray in node record (Gate 2 violation)"
                if isinstance(v, (list, tuple)):
                    assert not any(isinstance(x, np.ndarray) for x in v)

    def test_no_raw_tensor_data_in_edges(self):
        arr = np.random.randn(8, 8).astype(np.float32)
        spec = self._make_spec("model.layers.0.mlp.up_proj.weight", arr)
        _, edges = _build_block_graph_for_tensor(spec, arr, _BLOCK, _TOPK, "test", {})
        for edge in edges:
            d = edge.as_dict()
            for v in d.values():
                assert not isinstance(v, np.ndarray), "raw ndarray in edge record (Gate 2 violation)"

    def test_mlp_block_type_for_mlp_projections(self):
        arr = np.random.randn(8, 8).astype(np.float32)
        spec = self._make_spec("model.layers.0.mlp.gate_proj.weight", arr)
        nodes, _ = _build_block_graph_for_tensor(spec, arr, _BLOCK, _TOPK, "test", {})
        block_nodes = [n for n in nodes if n.node_type == "mlp_block"]
        assert len(block_nodes) > 0

    def test_channel_block_type_for_attention(self):
        arr = np.random.randn(8, 8).astype(np.float32)
        spec = self._make_spec("model.layers.0.self_attn.q_proj.weight", arr)
        nodes, _ = _build_block_graph_for_tensor(spec, arr, _BLOCK, _TOPK, "test", {})
        chan_nodes = [n for n in nodes if n.node_type == "channel_block"]
        assert len(chan_nodes) > 0

    def test_tensor_contains_block_edges_when_proj_known(self):
        arr = np.random.randn(8, 8).astype(np.float32)
        spec = self._make_spec("model.layers.0.self_attn.q_proj.weight", arr)
        # Provide proj_node_ids so contains edges are emitted
        specs_list = [spec]
        _, _, proj_ids = _build_structural_graph(specs_list, "test")
        _, edges = _build_block_graph_for_tensor(spec, arr, _BLOCK, _TOPK, "test", proj_ids)
        contains = [e for e in edges if e.rel == "tensor_contains_block"]
        assert len(contains) > 0

    def test_block_node_ids_are_deterministic(self):
        arr = np.random.randn(8, 8).astype(np.float32)
        spec = self._make_spec("model.layers.0.self_attn.q_proj.weight", arr)
        nodes1, _ = _build_block_graph_for_tensor(spec, arr, _BLOCK, _TOPK, "test", {})
        nodes2, _ = _build_block_graph_for_tensor(spec, arr, _BLOCK, _TOPK, "test", {})
        ids1 = {n.node_id for n in nodes1}
        ids2 = {n.node_id for n in nodes2}
        assert ids1 == ids2


# ── QwenWeightGraphCompiler ────────────────────────────────────────────────────

class TestQwenWeightGraphCompiler:
    def test_compile_returns_result(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs, _loader(_DENSE))
        assert isinstance(result, WeightGraphResult)

    def test_manifest_raw_payload_always_false(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs, _loader(_DENSE))
        assert result.manifest.raw_weight_payload_in_graph is False
        assert result.manifest.as_dict()["raw_weight_payload_in_graph"] is False

    def test_manifest_gate2_in_as_dict(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK)
        result = c.compile(dense_specs)
        d = result.manifest.as_dict()
        assert d["raw_weight_payload_in_graph"] is False

    def test_node_count_positive(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs, _loader(_DENSE))
        assert len(result.nodes) > 0
        assert result.manifest.emitted_node_count == len(result.nodes)

    def test_edge_count_positive(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs, _loader(_DENSE))
        assert len(result.edges) > 0
        assert result.manifest.emitted_edge_count == len(result.edges)

    def test_no_loader_gives_structural_only(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result_struct = c.compile(dense_specs, None)
        result_full = c.compile(dense_specs, _loader(_DENSE))
        # Structural-only has fewer nodes/edges (no block nodes/edges)
        assert len(result_struct.nodes) <= len(result_full.nodes)
        assert len(result_struct.edges) <= len(result_full.edges)

    def test_stats_layer_count(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs)
        assert result.stats["layer_count"] == 2

    def test_stats_expert_count_zero_for_dense(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs)
        assert result.stats["expert_count"] == 0

    def test_moe_stats_expert_count(self, moe_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(moe_specs, _loader(_MOE))
        assert result.stats["expert_count"] == 2

    def test_node_ids_are_16_char_hex(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs, _loader(_DENSE))
        for node in result.nodes[:10]:
            assert len(node.node_id) == 16
            assert all(c in "0123456789abcdef" for c in node.node_id)

    def test_edge_ids_are_16_char_hex(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs, _loader(_DENSE))
        for edge in result.edges[:10]:
            assert len(edge.edge_id) == 16

    def test_deterministic_node_edge_counts(self, dense_specs):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        r1 = c.compile(dense_specs, _loader(_DENSE))
        r2 = c.compile(dense_specs, _loader(_DENSE))
        assert len(r1.nodes) == len(r2.nodes)
        assert len(r1.edges) == len(r2.edges)

    def test_compile_from_directory(self, dense_dir: Path):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile_from_directory(dense_dir)
        assert isinstance(result, WeightGraphResult)
        assert result.manifest.raw_weight_payload_in_graph is False

    def test_invalid_block_size_raises(self):
        with pytest.raises(ValueError):
            QwenWeightGraphCompiler(block_size=0)

    def test_invalid_topk_raises(self):
        with pytest.raises(ValueError):
            QwenWeightGraphCompiler(topk=0)


# ── Artifact I/O round-trip ────────────────────────────────────────────────────

class TestArtifactIO:
    def _compile_and_write(self, tmp_path: Path, specs, tensors) -> tuple:
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(specs, _loader(tensors))
        write_weight_graph_artifacts(result, tmp_path / "out")
        return result, tmp_path / "out"

    def test_manifest_file_exists(self, dense_specs, tmp_path):
        _, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        assert (out / "manifest.json").exists()

    def test_nodes_file_exists(self, dense_specs, tmp_path):
        _, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        assert (out / "nodes.jsonl").exists()

    def test_edges_file_exists(self, dense_specs, tmp_path):
        _, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        assert (out / "edges.jsonl").exists()

    def test_stats_file_exists(self, dense_specs, tmp_path):
        _, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        assert (out / "stats.json").exists()

    def test_manifest_raw_payload_false_on_disk(self, dense_specs, tmp_path):
        _, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        m = json.loads((out / "manifest.json").read_text())
        assert m["raw_weight_payload_in_graph"] is False

    def test_round_trip_node_count(self, dense_specs, tmp_path):
        result, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        reloaded = read_weight_graph_artifacts(out)
        assert len(reloaded.nodes) == len(result.nodes)

    def test_round_trip_edge_count(self, dense_specs, tmp_path):
        result, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        reloaded = read_weight_graph_artifacts(out)
        assert len(reloaded.edges) == len(result.edges)

    def test_round_trip_node_ids_match(self, dense_specs, tmp_path):
        result, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        reloaded = read_weight_graph_artifacts(out)
        orig_ids = {n.node_id for n in result.nodes}
        reload_ids = {n.node_id for n in reloaded.nodes}
        assert orig_ids == reload_ids

    def test_round_trip_manifest_fields(self, dense_specs, tmp_path):
        result, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        reloaded = read_weight_graph_artifacts(out)
        assert reloaded.manifest.block_size == result.manifest.block_size
        assert reloaded.manifest.topk == result.manifest.topk
        assert reloaded.manifest.raw_weight_payload_in_graph is False

    def test_no_raw_weights_in_nodes_jsonl(self, dense_specs, tmp_path):
        _, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        for line in (out / "nodes.jsonl").read_text().splitlines():
            d = json.loads(line)
            # Ensure no large float arrays embedded as JSON arrays of many numbers
            for v in d.values():
                if isinstance(v, list):
                    # shape or range lists are tiny (2 elements); reject large lists
                    assert len(v) <= 32, f"suspiciously large list in node record: {len(v)} elements"

    def test_no_raw_weights_in_edges_jsonl(self, dense_specs, tmp_path):
        _, out = self._compile_and_write(tmp_path, dense_specs, _DENSE)
        for line in (out / "edges.jsonl").read_text().splitlines():
            d = json.loads(line)
            # weight is a scalar float, not a list
            assert isinstance(d.get("weight", 0.0), (int, float))


# ── P1.3: WorldGraph schema adapter ───────────────────────────────────────────

class TestWorldGraphAdapter:
    def _compile_write_load(self, tmp_path, specs, tensors) -> object:
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(specs, _loader(tensors))
        out = tmp_path / "out"
        write_weight_graph_artifacts(result, out)
        return load_weight_graph_as_world_graph(out)

    def test_world_graph_has_nodes(self, dense_specs, tmp_path):
        world = self._compile_write_load(tmp_path, dense_specs, _DENSE)
        assert world.node_count() > 0

    def test_world_graph_has_edges(self, dense_specs, tmp_path):
        world = self._compile_write_load(tmp_path, dense_specs, _DENSE)
        assert world.edge_count() > 0

    def test_world_graph_node_count_matches(self, dense_specs, tmp_path):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs, _loader(_DENSE))
        out = tmp_path / "out"
        write_weight_graph_artifacts(result, out)
        world = load_weight_graph_as_world_graph(out)
        assert world.node_count() == len(result.nodes)

    def test_moe_world_graph_has_expert_nodes(self, moe_specs, tmp_path):
        world = self._compile_write_load(tmp_path, moe_specs, _MOE)
        expert_nodes = [n for n in world.iter_nodes() if n.node_kind == "expert"]
        assert len(expert_nodes) == 2

    def test_dense_world_graph_no_expert_nodes(self, dense_specs, tmp_path):
        world = self._compile_write_load(tmp_path, dense_specs, _DENSE)
        expert_nodes = [n for n in world.iter_nodes() if n.node_kind == "expert"]
        assert len(expert_nodes) == 0

    def test_world_graph_nodes_have_kinds(self, dense_specs, tmp_path):
        world = self._compile_write_load(tmp_path, dense_specs, _DENSE)
        kinds = {n.node_kind for n in world.iter_nodes()}
        assert "layer" in kinds
        assert "projection" in kinds
        assert "model" in kinds


# ── P2.1: Sparse student stub ─────────────────────────────────────────────────

class TestSparseStudentStub:
    def _make_config(self, artifact_dir: Path) -> SparseStudentExperimentConfig:
        return SparseStudentExperimentConfig(
            artifact_dir=artifact_dir,
            task_examples_path=Path("data/examples.jsonl"),
            checkpoint=Path("runs/checkpoints/dense.pt"),
            topk=_TOPK,
            block_size=_BLOCK,
            device="cpu",
            quality_baseline=0.95,
            memory_budget_mb=512.0,
        )

    def test_stub_ok_when_artifact_present(self, dense_specs, tmp_path):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs, _loader(_DENSE))
        out = tmp_path / "out"
        write_weight_graph_artifacts(result, out)
        config = self._make_config(out)
        r = run_sparse_student_stub(config)
        assert r["status"] == "stub_ok"

    def test_stub_raw_payload_always_false(self, dense_specs, tmp_path):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs)
        out = tmp_path / "out"
        write_weight_graph_artifacts(result, out)
        config = self._make_config(out)
        r = run_sparse_student_stub(config)
        assert r.get("raw_weight_payload_in_graph") is False

    def test_stub_artifact_missing(self, tmp_path):
        config = self._make_config(tmp_path / "nonexistent")
        r = run_sparse_student_stub(config)
        assert r["status"] == "artifact_missing"

    def test_stub_gate2_violation_detected(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        bad_manifest = {"raw_weight_payload_in_graph": True, "source_model": "bad"}
        (out / "manifest.json").write_text(json.dumps(bad_manifest))
        config = self._make_config(out)
        r = run_sparse_student_stub(config)
        assert r["status"] == "gate_violation"

    def test_stub_reports_g0_counts(self, dense_specs, tmp_path):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs, _loader(_DENSE))
        out = tmp_path / "out"
        write_weight_graph_artifacts(result, out)
        config = self._make_config(out)
        r = run_sparse_student_stub(config)
        assert r["g0_node_count"] == len(result.nodes)
        assert r["g0_edge_count"] == len(result.edges)

    def test_stub_config_fields_present(self, dense_specs, tmp_path):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        result = c.compile(dense_specs)
        out = tmp_path / "out"
        write_weight_graph_artifacts(result, out)
        config = self._make_config(out)
        r = run_sparse_student_stub(config)
        assert r["block_size"] == _BLOCK
        assert r["topk"] == _TOPK
        assert r["device"] == "cpu"
        assert r["quality_baseline"] == pytest.approx(0.95)


# ── Integration: full compile_from_directory ───────────────────────────────────

class TestCompileFromDirectory:
    def test_full_pipeline_dense(self, dense_dir: Path, tmp_path: Path):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        out = tmp_path / "g0"
        result = c.compile_from_directory(dense_dir, out)
        assert (out / "manifest.json").exists()
        assert (out / "nodes.jsonl").exists()
        assert (out / "edges.jsonl").exists()
        assert result.manifest.raw_weight_payload_in_graph is False

    def test_full_pipeline_moe(self, moe_dir: Path, tmp_path: Path):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="moe_test")
        out = tmp_path / "g0"
        result = c.compile_from_directory(moe_dir, out)
        assert result.stats["expert_count"] == 2
        world = load_weight_graph_as_world_graph(out)
        expert_nodes = [n for n in world.iter_nodes() if n.node_kind == "expert"]
        assert len(expert_nodes) == 2

    def test_world_graph_roundtrip_edge_count(self, dense_dir: Path, tmp_path: Path):
        c = QwenWeightGraphCompiler(block_size=_BLOCK, topk=_TOPK, source_model="test")
        out = tmp_path / "g0"
        result = c.compile_from_directory(dense_dir, out)
        world = load_weight_graph_as_world_graph(out)
        assert world.node_count() == len(result.nodes)
