from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Sequence


SCHEMA_VERSION = "qwen_static_teacher_data.v1"
TENSOR_ROW_SCHEMA_VERSION = "qwen_static_tensor_inventory_row.v1"
SUMMARY_SCHEMA_VERSION = "qwen_static_tensor_summary.v1"
VALIDATION_SCHEMA_VERSION = "qwen_static_teacher_validation.v1"

STATIC_TEACHER_MANIFEST_FILENAME = "static_teacher_manifest.json"
STATIC_TENSOR_INVENTORY_FILENAME = "static_tensor_inventory.jsonl"
STATIC_TENSOR_SUMMARY_FILENAME = "static_tensor_summary.json"
STATIC_TEACHER_VALIDATION_FILENAME = "static_teacher_validation.json"

_GGUF_MAGIC = b"GGUF"

_GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

_GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
}


@dataclass(frozen=True)
class GGUFTensorInfo:
    tensor_index: int
    name: str
    shape: list[int]
    ggml_type: str
    ggml_type_id: int
    n_dims: int
    offset: int
    size_bytes_estimate: int


@dataclass(frozen=True)
class GGUFParseResult:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: list[GGUFTensorInfo]
    tensor_data_start: int
    alignment: int


def _write_json(path: Path, data: dict[str, Any], *, max_output_json_bytes: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if max_output_json_bytes is not None and len(text.encode("utf-8")) > max_output_json_bytes:
        raise ValueError(f"{path.name} exceeds max_output_json_bytes={max_output_json_bytes}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, max_output_json_bytes: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    total = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            total += len(line.encode("utf-8"))
            if max_output_json_bytes is not None and total > max_output_json_bytes:
                tmp.unlink(missing_ok=True)
                raise ValueError(f"{path.name} exceeds max_output_json_bytes={max_output_json_bytes}")
            handle.write(line)
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: expected JSON object rows")
                rows.append(row)
    return rows


def _read_exact(handle: BinaryIO, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise ValueError("truncated GGUF file")
    return data


def _read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_i32(handle: BinaryIO) -> int:
    return struct.unpack("<i", _read_exact(handle, 4))[0]


def _read_i64(handle: BinaryIO) -> int:
    return struct.unpack("<q", _read_exact(handle, 8))[0]


def _read_f32(handle: BinaryIO) -> float:
    return float(struct.unpack("<f", _read_exact(handle, 4))[0])


def _read_f64(handle: BinaryIO) -> float:
    return float(struct.unpack("<d", _read_exact(handle, 8))[0])


def _read_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _read_metadata_value(handle: BinaryIO, value_type: int) -> Any:
    if value_type == 0:
        return struct.unpack("<B", _read_exact(handle, 1))[0]
    if value_type == 1:
        return struct.unpack("<b", _read_exact(handle, 1))[0]
    if value_type == 2:
        return struct.unpack("<H", _read_exact(handle, 2))[0]
    if value_type == 3:
        return struct.unpack("<h", _read_exact(handle, 2))[0]
    if value_type == 4:
        return _read_u32(handle)
    if value_type == 5:
        return _read_i32(handle)
    if value_type == 6:
        return _read_f32(handle)
    if value_type == 7:
        return bool(struct.unpack("<?", _read_exact(handle, 1))[0])
    if value_type == 8:
        return _read_string(handle)
    if value_type == 9:
        item_type = _read_u32(handle)
        length = _read_u64(handle)
        return {
            "type": _GGUF_VALUE_TYPES.get(item_type, f"unknown_{item_type}"),
            "values": [_read_metadata_value(handle, item_type) for _ in range(length)],
        }
    if value_type == 10:
        return _read_u64(handle)
    if value_type == 11:
        return _read_i64(handle)
    if value_type == 12:
        return _read_f64(handle)
    raise ValueError(f"unsupported GGUF metadata value type {value_type}")


def _align_offset(offset: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("GGUF alignment must be positive")
    return ((offset + alignment - 1) // alignment) * alignment


def _estimate_tensor_size(shape: list[int], ggml_type_id: int) -> int:
    type_name, block_size, type_size = _GGML_TYPES.get(ggml_type_id, (f"UNKNOWN_{ggml_type_id}", 1, 1))
    del type_name
    n_elements = 1
    for dim in shape:
        n_elements *= max(0, int(dim))
    if n_elements == 0:
        return 0
    return int(math.ceil(n_elements / block_size) * type_size)


def parse_gguf(path: str | Path) -> GGUFParseResult:
    gguf_path = Path(path)
    with gguf_path.open("rb") as handle:
        if _read_exact(handle, 4) != _GGUF_MAGIC:
            raise ValueError(f"{gguf_path} is not a GGUF file")
        version = _read_u32(handle)
        tensor_count = _read_u64(handle)
        metadata_kv_count = _read_u64(handle)
        metadata: dict[str, Any] = {}
        for _ in range(metadata_kv_count):
            key = _read_string(handle)
            value_type = _read_u32(handle)
            metadata[key] = _read_metadata_value(handle, value_type)

        tensors: list[GGUFTensorInfo] = []
        for tensor_index in range(tensor_count):
            name = _read_string(handle)
            n_dims = _read_u32(handle)
            shape = [_read_u64(handle) for _ in range(n_dims)]
            ggml_type_id = _read_u32(handle)
            offset = _read_u64(handle)
            ggml_type = _GGML_TYPES.get(ggml_type_id, (f"UNKNOWN_{ggml_type_id}", 1, 1))[0]
            tensors.append(GGUFTensorInfo(
                tensor_index=tensor_index,
                name=name,
                shape=[int(dim) for dim in shape],
                ggml_type=ggml_type,
                ggml_type_id=ggml_type_id,
                n_dims=int(n_dims),
                offset=int(offset),
                size_bytes_estimate=_estimate_tensor_size([int(dim) for dim in shape], ggml_type_id),
            ))

        alignment = int(metadata.get("general.alignment", 32))
        tensor_data_start = _align_offset(handle.tell(), alignment)

    return GGUFParseResult(
        path=gguf_path,
        version=int(version),
        metadata=metadata,
        tensors=tensors,
        tensor_data_start=tensor_data_start,
        alignment=alignment,
    )


def _ollama_manifest_candidates(model_name: str, ollama_models_dir: Path) -> list[Path]:
    if ":" in model_name:
        name, tag = model_name.split(":", 1)
    else:
        name, tag = model_name, "latest"
    candidates = [
        ollama_models_dir / "manifests" / "registry.ollama.ai" / "library" / name / tag,
        ollama_models_dir / "manifests" / "registry.ollama.ai" / name / tag,
    ]
    if "/" in name:
        namespace, bare_name = name.split("/", 1)
        candidates.append(ollama_models_dir / "manifests" / "registry.ollama.ai" / namespace / bare_name / tag)
    return candidates


def resolve_ollama_manifest_path(model_name: str, ollama_models_dir: str | Path) -> Path:
    root = Path(ollama_models_dir)
    for candidate in _ollama_manifest_candidates(model_name, root):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve Ollama manifest for {model_name!r} under {root}")


def _blob_path_from_digest(ollama_models_dir: Path, digest: str) -> Path | None:
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        return None
    return ollama_models_dir / "blobs" / f"sha256-{digest.split(':', 1)[1]}"


def resolve_ollama_gguf_blob_path(model_name: str, ollama_models_dir: str | Path) -> tuple[Path, Path]:
    root = Path(ollama_models_dir)
    manifest_path = resolve_ollama_manifest_path(model_name, root)
    manifest = _read_json(manifest_path)
    digests: list[str] = []
    config = manifest.get("config")
    if isinstance(config, dict) and isinstance(config.get("digest"), str):
        digests.append(config["digest"])
    for layer in manifest.get("layers", []):
        if isinstance(layer, dict) and isinstance(layer.get("digest"), str):
            media_type = str(layer.get("mediaType", ""))
            if "model" in media_type:
                digests.insert(0, layer["digest"])
            else:
                digests.append(layer["digest"])
    seen: set[str] = set()
    for digest in digests:
        if digest in seen:
            continue
        seen.add(digest)
        blob_path = _blob_path_from_digest(root, digest)
        if blob_path is None or not blob_path.exists():
            continue
        with blob_path.open("rb") as handle:
            if handle.read(4) == _GGUF_MAGIC:
                return manifest_path, blob_path
    raise FileNotFoundError(f"could not locate GGUF blob referenced by {manifest_path}")


def _capture_command(args: list[str]) -> list[str]:
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    lines = (completed.stdout or "").splitlines()
    return [line.strip() for line in lines if line.strip()]


def capture_runtime_state() -> dict[str, Any]:
    return {
        "ollama_processes": _capture_command(["pgrep", "-a", "ollama"]),
        "gpu_compute_apps": _capture_command([
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]),
    }


def _contains_ollama_runner(lines: list[str]) -> bool:
    return any("ollama" in line.lower() and "runner" in line.lower() for line in lines)


def _unsafe_gpu_teacher_lines(lines: list[str]) -> list[str]:
    unsafe: list[str] = []
    for line in lines:
        lowered = line.lower()
        if "ollama" in lowered or "qwen" in lowered:
            unsafe.append(line)
    return unsafe


def _sample_tensor_bytes(gguf: GGUFParseResult, tensor: GGUFTensorInfo, max_tensor_sample_bytes: int) -> bytes:
    if max_tensor_sample_bytes < 0:
        raise ValueError("max_tensor_sample_bytes must be >= 0")
    sample_len = min(int(max_tensor_sample_bytes), int(tensor.size_bytes_estimate))
    if sample_len == 0:
        return b""
    with gguf.path.open("rb") as handle:
        handle.seek(gguf.tensor_data_start + tensor.offset)
        return handle.read(sample_len)


def _sample_stats(sample: bytes) -> dict[str, Any]:
    checksum = hashlib.sha256(sample).hexdigest()
    if not sample:
        return {
            "sample_bytes_read": 0,
            "sample_checksum": checksum,
            "sample_byte_min": None,
            "sample_byte_max": None,
            "sample_byte_mean": None,
        }
    return {
        "sample_bytes_read": len(sample),
        "sample_checksum": checksum,
        "sample_byte_min": min(sample),
        "sample_byte_max": max(sample),
        "sample_byte_mean": sum(sample) / float(len(sample)),
    }


def build_static_tensor_inventory_rows(
    *,
    gguf: GGUFParseResult,
    model_name: str,
    max_tensor_sample_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tensor in gguf.tensors:
        sample = _sample_tensor_bytes(gguf, tensor, max_tensor_sample_bytes)
        stats = _sample_stats(sample)
        rows.append({
            "schema_version": TENSOR_ROW_SCHEMA_VERSION,
            "model_name": model_name,
            "tensor_index": tensor.tensor_index,
            "name": tensor.name,
            "shape": tensor.shape,
            "ggml_type": tensor.ggml_type,
            "n_dims": tensor.n_dims,
            "offset": tensor.offset,
            "size_bytes_estimate": tensor.size_bytes_estimate,
            "sample_checksum": stats["sample_checksum"],
            "sample_bytes_read": stats["sample_bytes_read"],
            "sample_byte_min": stats["sample_byte_min"],
            "sample_byte_max": stats["sample_byte_max"],
            "sample_byte_mean": stats["sample_byte_mean"],
            "raw_payload_exported": False,
        })
    return rows


def _name_prefix(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        return ".".join(parts[:4])
    return ".".join(parts[:2]) if len(parts) >= 2 else name


def build_static_tensor_summary(gguf: GGUFParseResult, rows: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(str(row["ggml_type"]) for row in rows)
    prefix_counts = Counter(_name_prefix(str(row["name"])) for row in rows)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "static_tensor_summary_ok",
        "tensor_count": len(rows),
        "metadata_key_count": len(gguf.metadata),
        "total_size_bytes_estimate": sum(int(row["size_bytes_estimate"]) for row in rows),
        "tensor_type_counts": dict(sorted(type_counts.items())),
        "name_prefix_counts": dict(sorted(prefix_counts.items())),
        "bounded_sample_stats_available": all("sample_checksum" in row for row in rows),
        "source_format": "gguf",
        "gguf_version": gguf.version,
        "gguf_alignment": gguf.alignment,
    }


def build_static_teacher_manifest(
    *,
    model_name: str,
    manifest_path: Path,
    gguf_blob_path: Path,
    output_dir: Path,
    tensor_count: int,
    metadata_key_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "static_teacher_data_ready",
        "model_name": model_name,
        "source_format": "gguf",
        "ollama_manifest_path": str(manifest_path),
        "gguf_blob_path": str(gguf_blob_path),
        "output_dir": str(output_dir),
        "tensor_count": int(tensor_count),
        "metadata_key_count": int(metadata_key_count),
        "static_tensor_inventory": str(output_dir / STATIC_TENSOR_INVENTORY_FILENAME),
        "static_tensor_summary": str(output_dir / STATIC_TENSOR_SUMMARY_FILENAME),
        "static_teacher_validation": str(output_dir / STATIC_TEACHER_VALIDATION_FILENAME),
        "teacher_inference_ran": False,
        "teacher_checkpoint_loaded_into_runtime": False,
        "teacher_cuda_visible": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "ollama_api_called": False,
        "ollama_runner_started": False,
        "raw_full_weight_payload_exported": False,
        "bounded_static_extraction_only": True,
        "promotion_eligible_as_logits": False,
        "promotion_eligible_as_static_prior": True,
        "target_mode": "static_qwen_weight_graph_prior",
        "contains_logits": False,
    }


def build_static_teacher_validation(
    *,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    before_ollama = set(before_state.get("ollama_processes", []))
    after_ollama = set(after_state.get("ollama_processes", []))
    new_ollama = sorted(after_ollama - before_ollama)
    runner_started = _contains_ollama_runner(new_ollama)
    unsafe_gpu = _unsafe_gpu_teacher_lines(list(after_state.get("gpu_compute_apps", [])))
    forbidden_targets = {"synthetic_logits", "ollama_sampled_hard_label_pseudo_logits"}
    for artifact in artifacts:
        if artifact.get("target_mode") in forbidden_targets:
            raise ValueError(f"forbidden target_mode={artifact.get('target_mode')!r}")
        if bool(artifact.get("teacher_inference_ran")):
            raise ValueError("teacher_inference_ran must be false")
        if bool(artifact.get("ollama_api_called")):
            raise ValueError("ollama_api_called must be false")
        if bool(artifact.get("ollama_runner_started")):
            raise ValueError("ollama_runner_started must be false")
        if bool(artifact.get("raw_full_weight_payload_exported")) or bool(artifact.get("raw_payload_exported")):
            raise ValueError("raw full tensor payload export is forbidden")
        if bool(artifact.get("contains_logits")) or bool(artifact.get("promotion_eligible_as_logits")):
            raise ValueError("static teacher artifacts must not claim logits")
    if runner_started:
        raise ValueError("ollama runner started during static extraction")
    if unsafe_gpu:
        raise ValueError("Ollama/Qwen process appears in GPU compute apps")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "static_teacher_data_valid",
        "teacher_inference_ran": False,
        "teacher_cuda_memory_bytes": 0,
        "ollama_process_delta": len(new_ollama),
        "ollama_runner_started": False,
        "raw_full_weight_payload_exported": False,
        "bounded_static_extraction_only": True,
        "ollama_api_called": False,
        "teacher_checkpoint_loaded_into_runtime": False,
        "gpu_teacher_process_detected": False,
    }


def validate_static_teacher_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad static teacher schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "static_teacher_data_ready":
        raise ValueError(f"bad static teacher status={manifest.get('status')!r}")
    expected = {
        "teacher_inference_ran": False,
        "teacher_checkpoint_loaded_into_runtime": False,
        "ollama_api_called": False,
        "ollama_runner_started": False,
        "raw_full_weight_payload_exported": False,
        "bounded_static_extraction_only": True,
        "promotion_eligible_as_logits": False,
        "promotion_eligible_as_static_prior": True,
        "contains_logits": False,
    }
    for key, value in expected.items():
        if bool(manifest.get(key)) is not value:
            raise ValueError(f"static teacher manifest {key} must be {value}")
    if manifest.get("source_format") != "gguf":
        raise ValueError("source_format must be gguf")
    forbidden_name_tokens = ("real_qwen_logits", "teacher_logits", "behavioral_distillation", "full_qwen_distillation", "KL_teacher_distribution")
    text = json.dumps(manifest, sort_keys=True)
    for token in forbidden_name_tokens:
        if token in text:
            raise ValueError(f"forbidden naming token in static teacher manifest: {token}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "static_teacher_manifest_valid",
        "promotion_eligible_as_static_prior": True,
    }


def validate_static_tensor_inventory_rows(rows: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(rows):
        if row.get("schema_version") != TENSOR_ROW_SCHEMA_VERSION:
            raise ValueError(f"tensor inventory row {idx} bad schema_version")
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError(f"tensor inventory row {idx} missing name")
        if not isinstance(row.get("shape"), list):
            raise ValueError(f"tensor inventory row {idx} shape must be a list")
        if int(row.get("tensor_index", -1)) != idx:
            raise ValueError(f"tensor inventory row {idx} tensor_index mismatch")
        if int(row.get("size_bytes_estimate", -1)) < 0:
            raise ValueError(f"tensor inventory row {idx} size_bytes_estimate must be >= 0")
        if bool(row.get("raw_payload_exported")):
            raise ValueError(f"tensor inventory row {idx} raw_payload_exported must be false")
        if "sample_checksum" not in row:
            raise ValueError(f"tensor inventory row {idx} missing sample_checksum")


def validate_static_teacher_validation(validation: dict[str, Any]) -> dict[str, Any]:
    if validation.get("schema_version") != VALIDATION_SCHEMA_VERSION:
        raise ValueError(f"bad validation schema_version={validation.get('schema_version')!r}")
    if validation.get("status") != "static_teacher_data_valid":
        raise ValueError(f"bad validation status={validation.get('status')!r}")
    for key, expected in {
        "teacher_inference_ran": False,
        "teacher_cuda_memory_bytes": 0,
        "ollama_runner_started": False,
        "raw_full_weight_payload_exported": False,
        "bounded_static_extraction_only": True,
    }.items():
        value = validation.get(key)
        if key == "teacher_cuda_memory_bytes":
            if int(value) != expected:
                raise ValueError("teacher_cuda_memory_bytes must be 0")
        elif bool(value) is not expected:
            raise ValueError(f"validation {key} must be {expected}")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "static_teacher_validation_valid",
    }


def run_and_write_static_teacher_data(
    *,
    model_name: str,
    ollama_models_dir: str | Path,
    output_dir: str | Path,
    max_tensor_sample_bytes: int,
    max_output_json_bytes: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    if max_tensor_sample_bytes < 0:
        raise ValueError("max_tensor_sample_bytes must be >= 0")
    if max_output_json_bytes < 1:
        raise ValueError("max_output_json_bytes must be >= 1")
    out = Path(output_dir)
    if out.exists():
        if not overwrite:
            raise ValueError(f"static teacher output dir already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    before_state = capture_runtime_state()
    manifest_path, gguf_blob_path = resolve_ollama_gguf_blob_path(model_name, ollama_models_dir)
    gguf = parse_gguf(gguf_blob_path)
    rows = build_static_tensor_inventory_rows(
        gguf=gguf,
        model_name=model_name,
        max_tensor_sample_bytes=max_tensor_sample_bytes,
    )
    validate_static_tensor_inventory_rows(rows)
    summary = build_static_tensor_summary(gguf, rows)
    manifest = build_static_teacher_manifest(
        model_name=model_name,
        manifest_path=manifest_path,
        gguf_blob_path=gguf_blob_path,
        output_dir=out,
        tensor_count=len(rows),
        metadata_key_count=len(gguf.metadata),
    )
    validate_static_teacher_manifest(manifest)
    after_state = capture_runtime_state()
    validation = build_static_teacher_validation(
        before_state=before_state,
        after_state=after_state,
        artifacts=[manifest, summary, *rows],
    )
    validate_static_teacher_validation(validation)

    _write_jsonl(out / STATIC_TENSOR_INVENTORY_FILENAME, rows, max_output_json_bytes=max_output_json_bytes)
    _write_json(out / STATIC_TENSOR_SUMMARY_FILENAME, summary, max_output_json_bytes=max_output_json_bytes)
    _write_json(out / STATIC_TEACHER_MANIFEST_FILENAME, manifest, max_output_json_bytes=max_output_json_bytes)
    _write_json(out / STATIC_TEACHER_VALIDATION_FILENAME, validation, max_output_json_bytes=max_output_json_bytes)

    return {
        "status": "static_teacher_data_valid",
        "static_teacher_manifest": str(out / STATIC_TEACHER_MANIFEST_FILENAME),
        "static_tensor_inventory": str(out / STATIC_TENSOR_INVENTORY_FILENAME),
        "static_tensor_summary": str(out / STATIC_TENSOR_SUMMARY_FILENAME),
        "static_teacher_validation": str(out / STATIC_TEACHER_VALIDATION_FILENAME),
        "model_name": model_name,
        "source_format": "gguf",
        "tensor_count": len(rows),
        "teacher_inference_ran": False,
        "teacher_cuda_memory_bytes": 0,
        "ollama_runner_started": False,
        "raw_full_weight_payload_exported": False,
        "bounded_static_extraction_only": True,
    }


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {raw!r}")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract bounded static GGUF/Ollama Qwen checkpoint data without teacher inference.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--ollama-models-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tensor-sample-bytes", type=_nonnegative_int, required=True)
    parser.add_argument("--max-output-json-bytes", type=_positive_int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_and_write_static_teacher_data(
            model_name=args.model_name,
            ollama_models_dir=args.ollama_models_dir,
            output_dir=args.output_dir,
            max_tensor_sample_bytes=args.max_tensor_sample_bytes,
            max_output_json_bytes=args.max_output_json_bytes,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
