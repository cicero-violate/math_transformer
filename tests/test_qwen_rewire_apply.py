from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_edge_trace import run_and_write_edge_trace_report
from src.qwen_graph_prior_eval import run_graph_prior_eval
from src.qwen_rewire_apply import (
    ACCEPTED_CANDIDATE_MANIFEST_FILENAME,
    APPLIED_CANDIDATE_EVAL_DIRNAME,
    build_accepted_candidate_manifest,
    load_accepted_candidate_manifest,
    main,
    run_and_write_accepted_candidate_manifest,
    validate_accepted_candidate_manifest,
)
from src.qwen_rewire_search import run_and_write_rewire_search_report
from src.qwen_sparse_student_handoff import load_selected_adjacency
from src.qwen_weight_graph import QwenWeightGraphCompiler, build_tensor_manifest_from_directory, write_weight_graph_artifacts


def _make_safetensors_bytes(tensors: dict[str, np.ndarray]) -> bytes:
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


def _energy_proxy_tensors() -> dict[str, np.ndarray]:
    rng = np.random.RandomState(77)
    tensors = {}
    for li in range(2):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"model.layers.{li}.self_attn.{proj}.weight"] = rng.randn(8, 8).astype(np.float32)
        for proj in ("gate_proj", "up_proj"):
            tensors[f"model.layers.{li}.mlp.{proj}.weight"] = rng.randn(16, 8).astype(np.float32)
        tensors[f"model.layers.{li}.mlp.down_proj.weight"] = rng.randn(8, 16).astype(np.float32)
    tensors["model.norm.weight"] = rng.randn(8).astype(np.float32)
    return tensors


def _loader(tensors: dict[str, np.ndarray]):
    def load(spec):
        arr = tensors.get(spec.name)
        return arr.astype(np.float32) if arr is not None else None

    return load


def _compile_g0(tmp_path: Path) -> Path:
    tensors = _energy_proxy_tensors()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(_make_safetensors_bytes(tensors))
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "rewire_apply_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="rewire_apply_test")
    result = compiler.compile(specs, _loader(tensors), config_hash=config_hash, index_hash=index_hash)
    out = tmp_path / "g0"
    write_weight_graph_artifacts(result, out)
    return out


def _build_accepted_search(tmp_path: Path) -> tuple[Path, Path, Path]:
    g0 = _compile_g0(tmp_path)
    eval_out = tmp_path / "prior_eval"
    run_graph_prior_eval(
        source_weight_graph_dir=g0,
        output_dir=eval_out,
        k_values=[1, 2],
        random_seeds=[0, 1],
        quality_mode="energy_capture",
    )
    trace_out = tmp_path / "edge_trace"
    run_and_write_edge_trace_report(
        eval_out,
        trace_out,
        k=1,
        feature_dim=8,
        steps=1,
        seeds=[0, 1, 2],
        device="torch_cpu",
    )
    search_out = tmp_path / "search"
    search = run_and_write_rewire_search_report(
        eval_output_dir=eval_out,
        edge_trace_dir=trace_out,
        output_dir=search_out,
        k=1,
        max_swaps_values=[1, 2],
        proposal_policies=["same_source_top_weight", "same_source_low_weight", "deterministic_random"],
        policy_seed=7,
        target_seeds=[0, 1, 2],
        train_steps=5,
        lr=0.1,
        device="torch_cpu",
    )
    if search["accepted_candidate_count"] < 1:
        _force_selected_candidate_accepted(search_out)
    return eval_out, trace_out, search_out


def _force_selected_candidate_accepted(search_out: Path) -> None:
    """Create an accepted-search fixture from a real generated candidate artifact.

    P5 tests materialization only. P4 has separate coverage for actually finding an
    accepted candidate on the measured handoff, while tiny synthetic graphs may not
    always produce a strict KL-improving candidate.
    """
    search_path = search_out / "rewire_search_report.json"
    search = json.loads(search_path.read_text(encoding="utf-8"))
    selected_idx = int(search["best_candidate_index"])
    candidate = search["candidates"][selected_idx]
    acceptance_path = Path(candidate["acceptance_dir"]) / "rewire_acceptance_report.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    base_final = float(acceptance["base_kl_final"])
    candidate_final = max(0.0, base_final - 0.001)
    acceptance["candidate_kl_final"] = candidate_final
    acceptance["candidate_training_report"]["kl_final"] = candidate_final
    acceptance["candidate_minus_base_kl_final"] = candidate_final - base_final
    acceptance["quality_ok"] = True
    acceptance["base_training_ok"] = True
    acceptance["candidate_training_ok"] = True
    acceptance["safety_ok"] = True
    acceptance["accepted"] = True
    acceptance["decision"] = "accepted_pending_apply"
    acceptance["acceptance_gate"]["quality_ok"] = True
    acceptance["acceptance_gate"]["base_training_ok"] = True
    acceptance["acceptance_gate"]["candidate_training_ok"] = True
    acceptance["acceptance_gate"]["safety_ok"] = True
    acceptance["acceptance_gate"]["accepted"] = True
    acceptance_path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    candidate["accepted"] = True
    candidate["decision"] = "accepted_pending_apply"
    candidate["quality_ok"] = True
    candidate["base_training_ok"] = True
    candidate["candidate_training_ok"] = True
    candidate["safety_ok"] = True
    candidate["base_kl_final"] = base_final
    candidate["candidate_kl_final"] = candidate_final
    candidate["candidate_minus_base_kl_final"] = candidate_final - base_final
    search["candidates"][selected_idx] = candidate
    search["accepted_candidate_count"] = 1
    search["any_accepted"] = True
    search["decision"] = "accepted_candidate_found"
    search["selected_candidate_index"] = selected_idx
    search["selected_candidate_accepted"] = True
    search["selected_candidate_max_swaps"] = int(candidate["max_swaps"])
    search["selected_candidate_kl_delta"] = candidate_final - base_final
    search["best_candidate_index"] = selected_idx
    search["best_candidate_kl_delta"] = candidate_final - base_final
    search_path.write_text(json.dumps(search, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_apply_materializes_accepted_candidate_without_mutating_base(tmp_path):
    eval_out, _trace_out, search_out = _build_accepted_search(tmp_path)
    base_before = load_selected_adjacency(eval_out, k=1)
    out = tmp_path / "apply"
    manifest = build_accepted_candidate_manifest(
        rewire_search_dir=search_out,
        output_dir=out,
    )
    assert manifest["status"] == "accepted_candidate_apply_artifact_ok"
    assert manifest["candidate_materialized"] is True
    assert manifest["proposal_applied_to_base"] is False
    assert manifest["base_topology_mutated"] is False
    assert manifest["active_topology_mutated"] is False
    assert manifest["quality_ok"] is True
    assert manifest["candidate_minus_base_kl_final"] <= 0.0
    assert validate_accepted_candidate_manifest(manifest)["candidate_materialized"] is True
    applied_dir = Path(manifest["applied_candidate_eval_dir"])
    assert applied_dir.name == APPLIED_CANDIDATE_EVAL_DIRNAME
    assert (applied_dir / "v25_handoff_manifest.json").exists()
    candidate = load_selected_adjacency(applied_dir, adjacency_name=manifest["candidate_adjacency_name"])
    assert candidate["edge_count"] == manifest["candidate_edge_count"]
    base_after = load_selected_adjacency(eval_out, k=1)
    assert base_after == base_before


def test_apply_writes_manifest(tmp_path):
    _eval_out, _trace_out, search_out = _build_accepted_search(tmp_path)
    out = tmp_path / "apply_write"
    manifest = run_and_write_accepted_candidate_manifest(
        rewire_search_dir=search_out,
        output_dir=out,
    )
    path = out / ACCEPTED_CANDIDATE_MANIFEST_FILENAME
    assert path.exists()
    loaded = load_accepted_candidate_manifest(out)
    assert loaded == manifest
    assert (out / APPLIED_CANDIDATE_EVAL_DIRNAME).exists()


def test_apply_cli_writes_manifest(tmp_path):
    _eval_out, _trace_out, search_out = _build_accepted_search(tmp_path)
    out = tmp_path / "apply_cli"
    rc = main([
        "--rewire-search-dir",
        str(search_out),
        "--output-dir",
        str(out),
    ])
    assert rc == 0
    loaded = load_accepted_candidate_manifest(out)
    assert loaded["candidate_materialized"] is True
    assert loaded["proposal_applied_to_base"] is False


def test_apply_rejects_no_accepted_candidate(tmp_path):
    g0 = _compile_g0(tmp_path)
    eval_out = tmp_path / "prior_eval_no_accept"
    run_graph_prior_eval(
        source_weight_graph_dir=g0,
        output_dir=eval_out,
        k_values=[1, 2],
        random_seeds=[0, 1],
        quality_mode="energy_capture",
    )
    trace_out = tmp_path / "edge_trace_no_accept"
    run_and_write_edge_trace_report(eval_out, trace_out, k=1, seeds=[0, 1, 2])
    search_out = tmp_path / "search_no_accept"
    search = run_and_write_rewire_search_report(
        eval_output_dir=eval_out,
        edge_trace_dir=trace_out,
        output_dir=search_out,
        k=1,
        max_swaps_values=[1],
        proposal_policies=["same_source_top_weight"],
        target_seeds=[0, 1, 2],
        train_steps=5,
        lr=0.1,
    )
    if search["accepted_candidate_count"] > 0:
        pytest.skip("fixture unexpectedly produced an accepted candidate")
    with pytest.raises(ValueError, match="selected search candidate is not accepted"):
        build_accepted_candidate_manifest(rewire_search_dir=search_out, output_dir=tmp_path / "apply_bad")


def test_apply_rejects_existing_output_without_overwrite(tmp_path):
    _eval_out, _trace_out, search_out = _build_accepted_search(tmp_path)
    out = tmp_path / "apply_existing"
    run_and_write_accepted_candidate_manifest(rewire_search_dir=search_out, output_dir=out)
    with pytest.raises(ValueError, match="already exists"):
        build_accepted_candidate_manifest(rewire_search_dir=search_out, output_dir=out)
    manifest = build_accepted_candidate_manifest(rewire_search_dir=search_out, output_dir=out, overwrite=True)
    assert manifest["candidate_materialized"] is True
