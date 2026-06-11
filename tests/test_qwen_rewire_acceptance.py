from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_edge_trace import run_and_write_edge_trace_report
from src.qwen_graph_prior_eval import run_graph_prior_eval
from src.qwen_rewire_acceptance import (
    ACCEPTANCE_REPORT_FILENAME,
    build_rewire_acceptance_report,
    load_rewire_acceptance_report,
    main,
    run_and_write_rewire_acceptance_report,
    validate_rewire_acceptance_report,
)
from src.qwen_rewire_proposal import run_and_write_rewire_proposal_report
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "rewire_acceptance_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="rewire_acceptance_test")
    result = compiler.compile(specs, _loader(tensors), config_hash=config_hash, index_hash=index_hash)
    out = tmp_path / "g0"
    write_weight_graph_artifacts(result, out)
    return out


def _build_eval_trace_proposal(tmp_path: Path) -> tuple[Path, Path, Path]:
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
        device="cpu",
    )
    proposal_out = tmp_path / "proposal"
    run_and_write_rewire_proposal_report(
        eval_out,
        trace_out,
        proposal_out,
        k=1,
        max_swaps=2,
    )
    return eval_out, trace_out, proposal_out


def test_rewire_acceptance_builds_decision_without_applying_topology(tmp_path):
    eval_out, _trace_out, proposal_out = _build_eval_trace_proposal(tmp_path)
    out = tmp_path / "acceptance"
    report = build_rewire_acceptance_report(
        eval_output_dir=eval_out,
        rewire_proposal_dir=proposal_out,
        output_dir=out,
        k=1,
        vocab_size=16,
        target_seeds=[0, 1, 2],
        feature_dim=8,
        forward_steps=1,
        train_steps=5,
        lr=0.1,
        device="torch_cpu",
    )
    assert report["status"] == "rewire_acceptance_decision_ok"
    assert report["teacher_checkpoint_loaded"] is False
    assert report["teacher_inference_runtime_required"] is False
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["topology_mutated"] is False
    assert report["proposal_applied"] is False
    assert report["promotion_eligible"] is False
    assert report["candidate_edge_count"] <= report["base_edge_count"]
    assert report["candidate_max_out_degree"] <= report["k"]
    assert report["base_training_ok"] is True
    assert report["candidate_training_ok"] is True
    assert report["safety_ok"] is True
    assert report["decision"] in {"accepted_pending_apply", "rejected"}
    assert validate_rewire_acceptance_report(report)["decision"] == report["decision"]
    candidate_eval = Path(report["candidate_eval"]["candidate_eval_output_dir"])
    assert (candidate_eval / "v25_handoff_manifest.json").exists()
    assert load_selected_adjacency(candidate_eval, adjacency_name=report["candidate_adjacency_name"])["edge_count"] == report["candidate_edge_count"]
    base_after = load_selected_adjacency(eval_out, k=1)
    assert base_after["adjacency_name"] == "qwen_topk_k1"


def test_rewire_acceptance_writes_report(tmp_path):
    eval_out, _trace_out, proposal_out = _build_eval_trace_proposal(tmp_path)
    out = tmp_path / "acceptance_write"
    report = run_and_write_rewire_acceptance_report(
        eval_output_dir=eval_out,
        rewire_proposal_dir=proposal_out,
        output_dir=out,
        k=1,
        vocab_size=16,
        target_seeds=[0, 1, 2],
        train_steps=5,
        lr=0.1,
        device="cpu",
    )
    path = out / ACCEPTANCE_REPORT_FILENAME
    assert path.exists()
    loaded = load_rewire_acceptance_report(out)
    assert loaded == report
    assert loaded["proposal_applied"] is False


def test_rewire_acceptance_cli_writes_report(tmp_path):
    eval_out, _trace_out, proposal_out = _build_eval_trace_proposal(tmp_path)
    out = tmp_path / "acceptance_cli"
    rc = main([
        "--eval-output-dir",
        str(eval_out),
        "--rewire-proposal-dir",
        str(proposal_out),
        "--output-dir",
        str(out),
        "--k",
        "1",
        "--vocab-size",
        "16",
        "--target-seeds",
        "0,1,2",
        "--feature-dim",
        "8",
        "--forward-steps",
        "1",
        "--train-steps",
        "5",
        "--lr",
        "0.1",
        "--device",
        "torch_cpu",
    ])
    assert rc == 0
    loaded = load_rewire_acceptance_report(out)
    assert loaded["proposal_applied"] is False
    assert loaded["topology_mutated"] is False
    assert loaded["decision"] in {"accepted_pending_apply", "rejected"}


def test_rewire_acceptance_can_force_reject_with_zero_regression_gate(tmp_path):
    eval_out, _trace_out, proposal_out = _build_eval_trace_proposal(tmp_path)
    report = build_rewire_acceptance_report(
        eval_output_dir=eval_out,
        rewire_proposal_dir=proposal_out,
        output_dir=tmp_path / "acceptance_force",
        k=1,
        target_seeds=[0, 1, 2],
        train_steps=5,
        lr=0.1,
        max_kl_regression=0.0,
    )
    expected = report["candidate_kl_final"] <= report["base_kl_final"]
    assert report["quality_ok"] is expected
    assert report["accepted"] is (expected and report["base_training_ok"] and report["candidate_training_ok"] and report["safety_ok"])


def test_rewire_acceptance_rejects_bad_args(tmp_path):
    eval_out, _trace_out, proposal_out = _build_eval_trace_proposal(tmp_path)
    with pytest.raises(ValueError, match="train_steps"):
        build_rewire_acceptance_report(
            eval_output_dir=eval_out,
            rewire_proposal_dir=proposal_out,
            output_dir=tmp_path / "bad",
            train_steps=0,
        )
    with pytest.raises(SystemExit) as bad_cli:
        main([
            "--eval-output-dir",
            str(eval_out),
            "--rewire-proposal-dir",
            str(proposal_out),
            "--output-dir",
            str(tmp_path / "bad_cli"),
            "--max-kl-regression",
            "-1",
        ])
    assert bad_cli.value.code == 2
