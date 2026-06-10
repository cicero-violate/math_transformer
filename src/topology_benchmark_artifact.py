from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import re
import socket
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "learned_topology_benchmark.v1"

_QUALITY_RE = re.compile(
    r"mode=(?P<mode>\w+)\s+"
    r"k=(?P<k>\w+)\s+"
    r"examples=(?P<examples>\d+)\s+"
    r"route_acc=(?P<route_acc>[0-9.]+)"
    r"(?:\s+dense_agree=(?P<dense_agree>[0-9.]+))?"
    r"(?:\s+hidden_l1=(?P<hidden_l1>[0-9.]+))?"
    r"(?:\s+hidden_cos=(?P<hidden_cos>[0-9.]+))?"
    r"(?:\s+logit_l1=(?P<logit_l1>[0-9.]+))?"
    r"(?:\s+logit_kl=(?P<logit_kl>[0-9.]+))?"
)
_EXPERT_RE = re.compile(r"(?P<name>[A-Za-z0-9_]+)=(?P<correct>\d+)/(?P<total>\d+)\((?P<accuracy>[0-9.]+)\)")


def utc_timestamp() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_fingerprint() -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }
    try:
        import torch

        data.update(
            {
                "torch": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", None),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive metadata only
        data["torch_error"] = repr(exc)
    return data


def parse_quality_log(text: str) -> list[dict[str, Any]]:
    """Parse the stable text emitted by ``QualityReport.__str__``.

    The benchmark shell script remains text-compatible for humans and existing
    tooling, while this parser builds a structured quality section for artifacts.
    """
    rows: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        match = _QUALITY_RE.search(line)
        if match:
            gd = match.groupdict()
            k_raw = gd["k"]
            row: dict[str, Any] = {
                "mode": gd["mode"],
                "k": None if k_raw == "full" else int(k_raw),
                "examples": int(gd["examples"]),
                "route_acc": float(gd["route_acc"]),
                "by_expert": {},
            }
            for key in ("dense_agree", "hidden_l1", "hidden_cos", "logit_l1", "logit_kl"):
                row[key] = None if gd.get(key) is None else float(gd[key])
            rows.append(row)
            continue
        if "by_expert" in line and rows:
            experts: dict[str, dict[str, float | int]] = {}
            for exp in _EXPERT_RE.finditer(line):
                experts[exp.group("name")] = {
                    "correct": int(exp.group("correct")),
                    "total": int(exp.group("total")),
                    "accuracy": float(exp.group("accuracy")),
                }
            rows[-1]["by_expert"] = experts
    return rows


def latest_report_json(path: str | Path) -> dict[str, Any]:
    files = sorted(Path(path).glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no benchmark json in {path}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def paired_buckets(report: dict[str, Any]) -> dict[str, Any]:
    selector = report.get("selector_results", {})
    return selector.get("paired_prepared_shared_block", {}) or {}


def report_attention_ms(report: dict[str, Any]) -> float:
    val = float(report.get("prepared_static_sparse_attention_ms") or 0.0)
    if val > 0.0:
        return val
    paired = paired_buckets(report)
    outproj = float(paired.get("attention_outproj_ms", 0.0) or 0.0)
    kernel = float(paired.get("attention_kernel_ms", 0.0) or 0.0)
    return outproj if outproj > 0.0 else kernel


def learned_mode_from_report(report: dict[str, Any]) -> str:
    return "learned_block_topk" if report.get("by_relation", {}).get("block_score_entries", 0) else "learned_topology"


def pick_quality_row(rows: list[dict[str, Any]], mode: str, k: int) -> dict[str, Any]:
    for row in rows:
        if row.get("mode") == mode and row.get("k") == k:
            return row
    raise ValueError(f"missing quality row mode={mode} k={k}")


def bucket_group_stats(hand_buckets: dict[str, Any], learned_buckets: dict[str, Any]) -> dict[str, str]:
    groups = {
        "attention kernel": ["attention_kernel_ms", "attention_outproj_ms"],
        "qkv projection": ["qkv_ms", "norm_qkv_ms"],
        "out projection": ["out_proj_ms", "attention_outproj_ms"],
        "layernorm": ["norm1_ms", "norm2_ms", "norm_qkv_ms"],
        "ffn": ["ffn_ms"],
        "topology prepare/table": ["topology_prepare_ms", "learned_scorer_ms", "neighbor_table_build_ms"],
        "residual/measurement overhead": ["residual1_ms", "residual2_ms"],
    }
    rows: list[tuple[float, float, str]] = []
    for name, keys in groups.items():
        hv = sum(float(hand_buckets.get(k, 0.0) or 0.0) for k in keys)
        lv = sum(float(learned_buckets.get(k, 0.0) or 0.0) for k in keys)
        rows.append((lv - hv, lv, name))
    by_regression = sorted(rows, reverse=True)
    by_absolute = sorted(rows, key=lambda x: x[1], reverse=True)
    return {
        "dominant_regression_bucket": by_regression[0][2] if by_regression else "unknown",
        "dominant_absolute_bucket": by_absolute[0][2] if by_absolute else "unknown",
    }


def build_benchmark_artifact(
    *,
    quality_log_text: str,
    hand_report: dict[str, Any],
    learned_report: dict[str, Any],
    hand_k: int,
    learned_k: int,
    acceptance_tolerance_ms: float,
    config: dict[str, Any] | None = None,
    paths: dict[str, Any] | None = None,
    include_reports: bool = True,
) -> dict[str, Any]:
    rows = parse_quality_log(quality_log_text)
    learned_mode = learned_mode_from_report(learned_report)
    hand_quality = pick_quality_row(rows, "topology_only", hand_k)
    learned_quality = pick_quality_row(rows, learned_mode, learned_k)

    hand_ms = float(hand_report.get("prepared_static_sparse_block_ms") or 0.0)
    learned_ms = float(learned_report.get("prepared_static_sparse_block_ms") or 0.0)
    hand_attn = report_attention_ms(hand_report)
    learned_attn = report_attention_ms(learned_report)
    hand_non = max(0.0, hand_ms - hand_attn)
    learned_non = max(0.0, learned_ms - learned_attn)
    speedup = hand_ms / learned_ms if learned_ms > 0 else 0.0
    speed_gap_ms = learned_ms - hand_ms
    quality_ok = float(learned_quality["route_acc"]) >= float(hand_quality["route_acc"])
    strict_speed_ok = learned_ms > 0 and hand_ms > 0 and learned_ms < hand_ms
    speed_ok = learned_ms > 0 and hand_ms > 0 and speed_gap_ms <= acceptance_tolerance_ms

    hand_p = paired_buckets(hand_report)
    learned_p = paired_buckets(learned_report)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_timestamp(),
        "hardware": runtime_fingerprint(),
        "config": config or {},
        "paths": paths or {},
        "hashes": {
            key: sha256_file(value)
            for key, value in (paths or {}).items()
            if isinstance(value, str) and key in {"scorer", "checkpoint", "examples"}
        },
        "quality": {
            "hand": hand_quality,
            "learned": learned_quality,
            "learned_mode": learned_mode,
            "quality_ok": quality_ok,
        },
        "speed": {
            "hand_block_ms": hand_ms,
            "learned_block_ms": learned_ms,
            "hand_attention_ms": hand_attn,
            "learned_attention_ms": learned_attn,
            "hand_non_attention_ms": hand_non,
            "learned_non_attention_ms": learned_non,
            "speedup": speedup,
            "speed_gap_ms": speed_gap_ms,
            "acceptance_tolerance_ms": acceptance_tolerance_ms,
            "speed_ok": speed_ok,
            "strict_speed_ok": strict_speed_ok,
        },
        "buckets": {
            "hand": hand_p,
            "learned": learned_p,
        },
        "acceptance": {
            "passed": quality_ok and speed_ok and strict_speed_ok,
            "quality_ok": quality_ok,
            "speed_ok": speed_ok,
            "strict_speed_ok": strict_speed_ok,
        },
        "diagnostics": bucket_group_stats(hand_p, learned_p),
    }
    if include_reports:
        artifact["reports"] = {"hand": hand_report, "learned": learned_report}
    return artifact


def write_artifacts(artifact: dict[str, Any], *, json_path: str | Path | None = None, jsonl_path: str | Path | None = None) -> None:
    if json_path:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if jsonl_path:
        p = Path(jsonl_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(artifact, sort_keys=True) + "\n")


def load_artifact(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"empty benchmark artifact: {p}")
    if p.suffix == ".jsonl":
        return json.loads(text.splitlines()[-1])
    return json.loads(text)