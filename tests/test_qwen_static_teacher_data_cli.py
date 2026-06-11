from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from src.qwen_static_teacher_data import (
    STATIC_TEACHER_MANIFEST_FILENAME,
    STATIC_TEACHER_VALIDATION_FILENAME,
    STATIC_TENSOR_INVENTORY_FILENAME,
)
from src.qwen_static_teacher_data_cli import main


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _tiny_gguf() -> bytes:
    metadata = {
        "general.architecture": "qwen2",
        "general.alignment": 32,
    }
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", 1)
    header += struct.pack("<Q", len(metadata))
    for key, value in metadata.items():
        header += _gguf_string(key)
        if isinstance(value, str):
            header += struct.pack("<I", 8)
            header += _gguf_string(value)
        else:
            header += struct.pack("<I", 4)
            header += struct.pack("<I", value)
    header += _gguf_string("model.embed_tokens.weight")
    header += struct.pack("<I", 2)
    header += struct.pack("<Q", 2)
    header += struct.pack("<Q", 2)
    header += struct.pack("<I", 0)
    header += struct.pack("<Q", 0)
    return bytes(header + (b"\x00" * ((-len(header)) % 32)) + bytes(range(16)))


def _build_ollama_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "ollama-models"
    manifest_path = root / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5-coder" / "1.5b"
    blob_path = root / "blobs" / ("sha256-" + "b" * 64)
    manifest_path.parent.mkdir(parents=True)
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(_tiny_gguf())
    manifest_path.write_text(json.dumps({
        "schemaVersion": 2,
        "layers": [{
            "mediaType": "application/vnd.ollama.image.model",
            "digest": "sha256:" + "b" * 64,
            "size": blob_path.stat().st_size,
        }],
    }), encoding="utf-8")
    return root


def test_static_teacher_data_cli_writes_artifacts_and_prints_summary(monkeypatch, tmp_path, capsys):
    root = _build_ollama_fixture(tmp_path)
    out = tmp_path / "static_cli"
    monkeypatch.setattr("src.qwen_static_teacher_data.capture_runtime_state", lambda: {
        "ollama_processes": [],
        "gpu_compute_apps": [],
    })

    rc = main([
        "--model-name",
        "qwen2.5-coder:1.5b",
        "--ollama-models-dir",
        str(root),
        "--output-dir",
        str(out),
        "--max-tensor-sample-bytes",
        "8",
        "--max-output-json-bytes",
        str(1024 * 1024),
    ])

    assert rc == 0
    assert (out / STATIC_TEACHER_MANIFEST_FILENAME).exists()
    assert (out / STATIC_TENSOR_INVENTORY_FILENAME).exists()
    assert (out / STATIC_TEACHER_VALIDATION_FILENAME).exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "static_teacher_data_valid"
    assert printed["static_teacher_manifest"] == str(out / STATIC_TEACHER_MANIFEST_FILENAME)
    assert printed["teacher_inference_ran"] is False
    assert printed["bounded_static_extraction_only"] is True


def test_static_teacher_data_cli_rejects_bad_args(tmp_path):
    with pytest.raises(SystemExit) as bad_sample:
        main([
            "--model-name",
            "qwen2.5-coder:1.5b",
            "--ollama-models-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-tensor-sample-bytes",
            "-1",
            "--max-output-json-bytes",
            "1024",
        ])
    assert bad_sample.value.code == 2

    with pytest.raises(SystemExit) as missing_manifest:
        main([
            "--model-name",
            "qwen2.5-coder:1.5b",
            "--ollama-models-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-tensor-sample-bytes",
            "8",
            "--max-output-json-bytes",
            "1024",
        ])
    assert missing_manifest.value.code == 2
