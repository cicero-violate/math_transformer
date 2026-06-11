from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from src.qwen_static_teacher_data import (
    STATIC_TEACHER_MANIFEST_FILENAME,
    STATIC_TEACHER_VALIDATION_FILENAME,
    STATIC_TENSOR_INVENTORY_FILENAME,
    STATIC_TENSOR_SUMMARY_FILENAME,
    build_static_teacher_validation,
    parse_gguf,
    resolve_ollama_gguf_blob_path,
    run_and_write_static_teacher_data,
    validate_static_teacher_manifest,
    validate_static_teacher_validation,
)


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_metadata_value(value) -> tuple[int, bytes]:
    if isinstance(value, str):
        return 8, _gguf_string(value)
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, int):
        return 4, struct.pack("<I", value)
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    raise TypeError(value)


def _make_tiny_gguf() -> bytes:
    metadata = {
        "general.architecture": "qwen2",
        "general.name": "qwen2.5-coder:1.5b fixture",
        "general.alignment": 32,
        "tokenizer.ggml.model": "gpt2",
    }
    tensor_specs = [
        ("model.layers.0.self_attn.q_proj.weight", [2, 2], 0, bytes(range(16))),
        ("model.layers.0.mlp.down_proj.weight", [4], 24, bytes([10, 20, 30, 40])),
    ]
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensor_specs))
    header += struct.pack("<Q", len(metadata))
    for key, value in metadata.items():
        value_type, encoded = _gguf_metadata_value(value)
        header += _gguf_string(key)
        header += struct.pack("<I", value_type)
        header += encoded
    for name, shape, offset, ggml_type_id, in [
        (tensor_specs[0][0], tensor_specs[0][1], tensor_specs[0][2], 0),
        (tensor_specs[1][0], tensor_specs[1][1], tensor_specs[1][2], 24),
    ]:
        header += _gguf_string(name)
        header += struct.pack("<I", len(shape))
        for dim in shape:
            header += struct.pack("<Q", dim)
        header += struct.pack("<I", ggml_type_id)
        header += struct.pack("<Q", offset)
    pad = (-len(header)) % 32
    body = bytearray(b"\x00" * pad)
    body += tensor_specs[0][3]
    body += b"\x00" * 8
    body += tensor_specs[1][3]
    return bytes(header + body)


def _build_ollama_fixture(tmp_path: Path, *, model_name: str = "qwen2.5-coder:1.5b") -> tuple[Path, Path]:
    root = tmp_path / "ollama-models"
    name, tag = model_name.split(":", 1)
    manifest_path = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    blob_path = root / "blobs" / ("sha256-" + "a" * 64)
    manifest_path.parent.mkdir(parents=True)
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(_make_tiny_gguf())
    manifest = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": "sha256:" + "a" * 64,
                "size": blob_path.stat().st_size,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root, blob_path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_parse_gguf_reads_metadata_and_tensor_inventory(tmp_path):
    gguf_path = tmp_path / "fixture.gguf"
    gguf_path.write_bytes(_make_tiny_gguf())

    parsed = parse_gguf(gguf_path)

    assert parsed.version == 3
    assert parsed.metadata["general.architecture"] == "qwen2"
    assert parsed.alignment == 32
    assert len(parsed.tensors) == 2
    assert parsed.tensors[0].name == "model.layers.0.self_attn.q_proj.weight"
    assert parsed.tensors[0].ggml_type == "F32"
    assert parsed.tensors[0].size_bytes_estimate == 16
    assert parsed.tensors[1].offset == 24
    assert parsed.tensors[1].size_bytes_estimate == 4


def test_resolve_ollama_gguf_blob_path(tmp_path):
    root, blob = _build_ollama_fixture(tmp_path)

    manifest_path, resolved_blob = resolve_ollama_gguf_blob_path("qwen2.5-coder:1.5b", root)

    assert manifest_path.exists()
    assert resolved_blob == blob


def test_static_teacher_data_happy_path_writes_static_artifacts(monkeypatch, tmp_path):
    root, _blob = _build_ollama_fixture(tmp_path)
    out = tmp_path / "static_out"
    monkeypatch.setattr("src.qwen_static_teacher_data.capture_runtime_state", lambda: {
        "ollama_processes": [],
        "gpu_compute_apps": [],
    })

    summary = run_and_write_static_teacher_data(
        model_name="qwen2.5-coder:1.5b",
        ollama_models_dir=root,
        output_dir=out,
        max_tensor_sample_bytes=8,
        max_output_json_bytes=1024 * 1024,
    )

    assert summary["status"] == "static_teacher_data_valid"
    assert (out / STATIC_TEACHER_MANIFEST_FILENAME).exists()
    assert (out / STATIC_TENSOR_INVENTORY_FILENAME).exists()
    assert (out / STATIC_TENSOR_SUMMARY_FILENAME).exists()
    assert (out / STATIC_TEACHER_VALIDATION_FILENAME).exists()
    manifest = _read_json(out / STATIC_TEACHER_MANIFEST_FILENAME)
    assert validate_static_teacher_manifest(manifest)["promotion_eligible_as_static_prior"] is True
    assert manifest["teacher_inference_ran"] is False
    assert manifest["ollama_api_called"] is False
    assert manifest["promotion_eligible_as_logits"] is False
    rows = _read_jsonl(out / STATIC_TENSOR_INVENTORY_FILENAME)
    assert len(rows) == 2
    assert rows[0]["sample_bytes_read"] == 8
    assert rows[0]["raw_payload_exported"] is False
    assert "sample_checksum" in rows[0]
    tensor_summary = _read_json(out / STATIC_TENSOR_SUMMARY_FILENAME)
    assert tensor_summary["status"] == "static_tensor_summary_ok"
    assert tensor_summary["tensor_count"] == 2
    validation = _read_json(out / STATIC_TEACHER_VALIDATION_FILENAME)
    assert validate_static_teacher_validation(validation)["status"] == "static_teacher_validation_valid"
    assert validation["teacher_cuda_memory_bytes"] == 0


def test_static_teacher_validation_rejects_runner_and_logits_claims():
    with pytest.raises(ValueError, match="ollama runner started"):
        build_static_teacher_validation(
            before_state={"ollama_processes": [], "gpu_compute_apps": []},
            after_state={"ollama_processes": ["123 ollama runner --model x"], "gpu_compute_apps": []},
            artifacts=[],
        )

    with pytest.raises(ValueError, match="forbidden target_mode"):
        build_static_teacher_validation(
            before_state={"ollama_processes": [], "gpu_compute_apps": []},
            after_state={"ollama_processes": [], "gpu_compute_apps": []},
            artifacts=[{"target_mode": "synthetic_logits"}],
        )

    with pytest.raises(ValueError, match="must not claim logits"):
        build_static_teacher_validation(
            before_state={"ollama_processes": [], "gpu_compute_apps": []},
            after_state={"ollama_processes": [], "gpu_compute_apps": []},
            artifacts=[{"contains_logits": True}],
        )


def test_static_teacher_data_rejects_bad_args_and_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr("src.qwen_static_teacher_data.capture_runtime_state", lambda: {
        "ollama_processes": [],
        "gpu_compute_apps": [],
    })
    with pytest.raises(ValueError, match="max_tensor_sample_bytes must be >= 0"):
        run_and_write_static_teacher_data(
            model_name="qwen2.5-coder:1.5b",
            ollama_models_dir=tmp_path,
            output_dir=tmp_path / "out",
            max_tensor_sample_bytes=-1,
            max_output_json_bytes=1024,
        )

    with pytest.raises(FileNotFoundError, match="could not resolve Ollama manifest"):
        run_and_write_static_teacher_data(
            model_name="qwen2.5-coder:1.5b",
            ollama_models_dir=tmp_path,
            output_dir=tmp_path / "out",
            max_tensor_sample_bytes=8,
            max_output_json_bytes=1024,
        )
