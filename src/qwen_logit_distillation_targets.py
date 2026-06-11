from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.qwen_distillation_targets import checksum_json


MANIFEST_SCHEMA_VERSION = "qwen_frozen_logit_distillation_targets.v1"
ROW_SCHEMA_VERSION = "qwen_frozen_logit_distillation_target_row.v1"
MANIFEST_FILENAME = "frozen_logit_targets_manifest.json"
TARGET_ROWS_FILENAME = "frozen_logit_targets.jsonl"
PROBABILITY_SUM_TOLERANCE = 1e-8

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


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_vocab_temperature(vocab_size: int, temperature: float) -> tuple[int, float]:
    if not isinstance(vocab_size, int) or vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size!r}")
    temperature_value = _finite_float(temperature, name="temperature")
    if temperature_value <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    return vocab_size, temperature_value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _assert_false(data: dict[str, Any], key: str, *, label: str) -> None:
    if bool(data.get(key, True)):
        raise ValueError(f"{label} {key} must be false")


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


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if not logits:
        raise ValueError("logits must be non-empty")
    temperature_value = _finite_float(temperature, name="temperature")
    if temperature_value <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    clean_logits = [_finite_float(value, name="logit") for value in logits]
    max_logit = max(clean_logits)
    exps = [math.exp((value - max_logit) / temperature_value) for value in clean_logits]
    total = sum(exps)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("softmax normalization must be finite and positive")
    return [value / total for value in exps]


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return text.encode("utf-8")


def write_frozen_logit_distillation_targets(
    output_dir: str | Path,
    *,
    vocab_size: int = 16,
    seeds: list[int] | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    vocab, temp = _validate_vocab_temperature(vocab_size, temperature)
    seed_values = [0, 1, 2] if seeds is None else list(seeds)
    if not seed_values:
        raise ValueError("at least one frozen logit target seed is required")

    rows: list[dict[str, Any]] = []
    for idx, seed in enumerate(seed_values):
        rng = random.Random(seed)
        logits = [rng.uniform(-2.0, 2.0) for _ in range(vocab)]
        probabilities = softmax(logits, temp)
        rows.append(
            {
                "schema_version": ROW_SCHEMA_VERSION,
                "row_id": f"logit_target_{idx:06d}",
                "seed": seed,
                "target_type": "logits",
                "vocab_size": vocab,
                "temperature": temp,
                "logits_checksum": checksum_json(logits),
                "probabilities_checksum": checksum_json(probabilities),
                "logits": logits,
                "probabilities": probabilities,
                "metadata": {
                    "target_mode": "synthetic_logits",
                    "teacher_checkpoint_loaded_at_runtime": False,
                },
            }
        )

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    rows_path = output_base / TARGET_ROWS_FILENAME
    rows_path.write_bytes(_jsonl_bytes(rows))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_logit_targets_ready",
        "target_type": "logits",
        "producer": "offline_teacher_or_synthetic_fixture",
        "teacher_checkpoint_loaded_at_runtime": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "student_training_started": False,
        "kl_training_started": False,
        "vocab_size": vocab,
        "row_count": len(rows),
        "target_rows_path": TARGET_ROWS_FILENAME,
        "target_rows_sha256": _sha256_file(rows_path),
        "temperature": temp,
        "promotion_eligible": False,
    }
    (output_base / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def iter_frozen_logit_target_rows(output_dir: str | Path) -> Iterable[dict[str, Any]]:
    rows_path = Path(output_dir) / TARGET_ROWS_FILENAME
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


def load_frozen_logit_distillation_targets_manifest(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / MANIFEST_FILENAME)


def _validate_float_list(values: Any, *, length: int, name: str) -> list[float]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    if len(values) != length:
        raise ValueError(f"{name} length={len(values)} does not match vocab_size={length}")
    return [_finite_float(value, name=name) for value in values]


def validate_frozen_logit_distillation_targets(output_dir: str | Path) -> dict[str, Any]:
    base = Path(output_dir)
    manifest = load_frozen_logit_distillation_targets_manifest(base)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"bad frozen logit targets schema_version={manifest.get('schema_version')!r}")
    if manifest.get("target_type") != "logits":
        raise ValueError(f"unsupported target_type={manifest.get('target_type')!r}")
    for key in (
        "teacher_checkpoint_loaded_at_runtime",
        "teacher_inference_runtime_required",
        "raw_weight_payload_in_graph",
        "student_training_started",
        "kl_training_started",
        "promotion_eligible",
    ):
        _assert_false(manifest, key, label="manifest")
    vocab_size, temperature = _validate_vocab_temperature(int(manifest.get("vocab_size", 0)), manifest.get("temperature"))

    rows_path = base / str(manifest.get("target_rows_path", TARGET_ROWS_FILENAME))
    if not rows_path.exists():
        raise FileNotFoundError(str(rows_path))
    rows_sha = _sha256_file(rows_path)
    if rows_sha != manifest.get("target_rows_sha256"):
        raise ValueError("target_rows_sha256 mismatch")

    rows = list(iter_frozen_logit_target_rows(base))
    if len(rows) != int(manifest.get("row_count", -1)):
        raise ValueError("row_count does not match frozen logit target rows")

    for row in rows:
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise ValueError(f"bad frozen logit target row schema_version={row.get('schema_version')!r}")
        _reject_forbidden_payload_keys({key: value for key, value in row.items() if key not in {"logits", "probabilities"}})
        if row.get("target_type") != "logits":
            raise ValueError("row target_type must be logits")
        if int(row.get("vocab_size", 0)) != vocab_size:
            raise ValueError("row vocab_size does not match manifest")
        if float(row.get("temperature")) != temperature:
            raise ValueError("row temperature does not match manifest")
        logits = _validate_float_list(row.get("logits"), length=vocab_size, name="logits")
        probabilities = _validate_float_list(row.get("probabilities"), length=vocab_size, name="probabilities")
        if any(probability < 0.0 for probability in probabilities):
            raise ValueError("probabilities must be non-negative")
        probability_sum = sum(probabilities)
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
            raise ValueError(f"probabilities must sum to 1.0, got {probability_sum}")
        if row.get("logits_checksum") != checksum_json(logits):
            raise ValueError("logits_checksum mismatch")
        if row.get("probabilities_checksum") != checksum_json(probabilities):
            raise ValueError("probabilities_checksum mismatch")

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_logit_targets_valid",
        "target_type": "logits",
        "vocab_size": vocab_size,
        "row_count": len(rows),
        "target_rows_sha256": rows_sha,
        "temperature": temperature,
        "teacher_checkpoint_loaded_at_runtime": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "student_training_started": False,
        "kl_training_started": False,
        "promotion_eligible": False,
    }
