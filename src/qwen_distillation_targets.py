from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.qwen_sparse_student_runtime import run_fixed_topology_forward_features, synthetic_target_features


MANIFEST_SCHEMA_VERSION = "qwen_frozen_distillation_targets.v1"
ROW_SCHEMA_VERSION = "qwen_frozen_distillation_target_row.v1"
TARGET_ROWS_FILENAME = "frozen_targets.jsonl"
MANIFEST_FILENAME = "frozen_targets_manifest.json"

_FORBIDDEN_PAYLOAD_KEYS = {
    "payload",
    "raw_payload",
    "raw_tensor",
    "raw_tensor_payload",
    "raw_weight",
    "raw_weight_payload",
    "tensor_data",
    "tensor_payload",
    "tensor_values",
    "values",
    "weight_payload",
}


def checksum_json(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _reject_forbidden_payload_keys(value: Any, *, path: str = "row") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"raw tensor payload field forbidden at {path}.{key_text}")
            _reject_forbidden_payload_keys(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_forbidden_payload_keys(child, path=f"{path}[{idx}]")


def _assert_false(data: dict[str, Any], key: str, *, label: str) -> None:
    if bool(data.get(key, True)):
        raise ValueError(f"{label} {key} must be false")


def _validate_feature_dict(features: Any, *, feature_dim: int, node_count: int) -> dict[str, list[float]]:
    if not isinstance(features, dict):
        raise ValueError("target_features must be an object")
    if len(features) != node_count:
        raise ValueError(f"target_features node_count={len(features)} does not match {node_count}")
    clean: dict[str, list[float]] = {}
    for node_id, vector in features.items():
        if not isinstance(vector, list):
            raise ValueError(f"target_features[{node_id!r}] must be a list")
        if len(vector) != feature_dim:
            raise ValueError(f"target_features[{node_id!r}] feature_dim={len(vector)} does not match {feature_dim}")
        clean_vector: list[float] = []
        for value in vector:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("target feature values must be finite")
            clean_vector.append(number)
        clean[str(node_id)] = clean_vector
    return clean


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return text.encode("utf-8")


def write_frozen_distillation_targets(
    eval_output_dir: str | Path,
    output_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    feature_dim: int = 8,
    seeds: list[int] | None = None,
    target_mode: str = "scaled_identity",
    target_scale: float = 0.5,
) -> dict[str, Any]:
    seed_values = [0, 1, 2] if seeds is None else list(seeds)
    if feature_dim < 1:
        raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    selected_summary: dict[str, Any] | None = None
    manifest_node_ids_sha256: str | None = None
    for idx, seed in enumerate(seed_values):
        forward = run_fixed_topology_forward_features(
            eval_output_dir,
            k=k,
            adjacency_name=adjacency_name,
            feature_dim=feature_dim,
            steps=1,
            seed=seed,
        )
        summary = forward["summary"]
        if selected_summary is None:
            selected_summary = summary
        else:
            expected = (selected_summary["adjacency_name"], selected_summary["k"], selected_summary["node_count"])
            observed = (summary["adjacency_name"], summary["k"], summary["node_count"])
            if observed != expected:
                raise ValueError("selected adjacency changed while writing frozen targets")

        target_features = synthetic_target_features(
            forward["input_features"],
            mode=target_mode,
            scale=target_scale,
        )
        node_ids = sorted(target_features)
        node_ids_sha256 = checksum_json(node_ids)
        if manifest_node_ids_sha256 is None:
            manifest_node_ids_sha256 = node_ids_sha256
        elif node_ids_sha256 != manifest_node_ids_sha256:
            raise ValueError("node ids changed while writing frozen targets")
        row = {
            "schema_version": ROW_SCHEMA_VERSION,
            "row_id": f"target_{idx:06d}",
            "seed": seed,
            "target_type": "node_features",
            "feature_dim": feature_dim,
            "node_count": len(node_ids),
            "node_ids_sha256": node_ids_sha256,
            "target_checksum": checksum_json(target_features),
            "target_features": target_features,
            "metadata": {
                "target_mode": f"synthetic_{target_mode}",
                "teacher_checkpoint_loaded_at_runtime": False,
            },
        }
        rows.append(row)

    if selected_summary is None:
        raise ValueError("at least one frozen target seed is required")
    rows_path = output_base / TARGET_ROWS_FILENAME
    rows_path.write_bytes(_jsonl_bytes(rows))

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_targets_ready",
        "target_type": "node_features",
        "producer": "offline_teacher_or_synthetic_fixture",
        "teacher_checkpoint_loaded_at_runtime": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "student_training_started": False,
        "feature_dim": feature_dim,
        "row_count": len(rows),
        "target_rows_path": TARGET_ROWS_FILENAME,
        "target_rows_sha256": _sha256_file(rows_path),
        "selected_adjacency_name": selected_summary["adjacency_name"],
        "selected_adjacency_k": selected_summary["k"],
        "selected_adjacency_node_count": selected_summary["node_count"],
        "selected_adjacency_edge_count": selected_summary["edge_count"],
        "node_ids_sha256": manifest_node_ids_sha256,
        "promotion_eligible": False,
    }
    (output_base / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_frozen_distillation_targets(output_dir: str | Path) -> dict[str, Any]:
    base = Path(output_dir)
    manifest = _read_json(base / MANIFEST_FILENAME)
    rows = list(iter_frozen_target_rows(base))
    return {"manifest": manifest, "rows": rows}


def load_frozen_distillation_targets_manifest(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / MANIFEST_FILENAME)


def iter_frozen_target_rows(output_dir: str | Path) -> Iterable[dict[str, Any]]:
    base = Path(output_dir)
    rows_path = base / TARGET_ROWS_FILENAME
    if not rows_path.exists():
        raise FileNotFoundError(str(rows_path))
    with rows_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{rows_path}:{lineno}: expected JSON object row")
            yield row


def validate_frozen_distillation_targets(output_dir: str | Path) -> dict[str, Any]:
    base = Path(output_dir)
    manifest = _read_json(base / MANIFEST_FILENAME)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"bad frozen targets schema_version={manifest.get('schema_version')!r}")
    _assert_false(manifest, "teacher_checkpoint_loaded_at_runtime", label="manifest")
    _assert_false(manifest, "teacher_inference_runtime_required", label="manifest")
    _assert_false(manifest, "raw_weight_payload_in_graph", label="manifest")
    _assert_false(manifest, "student_training_started", label="manifest")
    _assert_false(manifest, "promotion_eligible", label="manifest")
    if manifest.get("target_type") != "node_features":
        raise ValueError(f"unsupported target_type={manifest.get('target_type')!r}")

    rows_rel = str(manifest.get("target_rows_path", TARGET_ROWS_FILENAME))
    rows_path = base / rows_rel
    if not rows_path.exists():
        raise FileNotFoundError(str(rows_path))
    rows_sha = _sha256_file(rows_path)
    if rows_sha != manifest.get("target_rows_sha256"):
        raise ValueError("target_rows_sha256 mismatch")

    feature_dim = int(manifest.get("feature_dim", 0))
    node_count = int(manifest.get("selected_adjacency_node_count", -1))
    expected_node_ids_sha = str(manifest.get("node_ids_sha256"))
    rows = list(iter_frozen_target_rows(base))
    if len(rows) != int(manifest.get("row_count", -1)):
        raise ValueError("row_count does not match frozen target rows")

    for row in rows:
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise ValueError(f"bad frozen target row schema_version={row.get('schema_version')!r}")
        _reject_forbidden_payload_keys({key: value for key, value in row.items() if key != "target_features"})
        if row.get("target_type") != manifest.get("target_type"):
            raise ValueError("row target_type does not match manifest")
        if int(row.get("feature_dim", 0)) != feature_dim:
            raise ValueError("row feature_dim does not match manifest")
        if int(row.get("node_count", -1)) != node_count:
            raise ValueError("row node_count does not match selected adjacency node_count")
        if row.get("node_ids_sha256") != expected_node_ids_sha:
            raise ValueError("row node_ids_sha256 does not match manifest")
        target_features = _validate_feature_dict(row.get("target_features"), feature_dim=feature_dim, node_count=node_count)
        if row.get("target_checksum") != checksum_json(target_features):
            raise ValueError("row target_checksum mismatch")

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_targets_valid",
        "target_type": manifest["target_type"],
        "feature_dim": feature_dim,
        "row_count": len(rows),
        "target_rows_sha256": rows_sha,
        "selected_adjacency_name": manifest.get("selected_adjacency_name"),
        "selected_adjacency_k": manifest.get("selected_adjacency_k"),
        "teacher_checkpoint_loaded_at_runtime": False,
        "teacher_inference_runtime_required": False,
        "student_training_started": False,
        "promotion_eligible": False,
    }
