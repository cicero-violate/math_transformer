from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any, Sequence

from src.qwen_sparse_student_runtime import resolve_runtime_device, run_fixed_topology_forward_features


SCHEMA_VERSION = "qwen_backend_compare.v1"
BACKEND_COMPARE_REPORT_FILENAME = "backend_compare_report.json"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_int_list(raw: str, *, name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one integer")
    return values


def _parse_devices(raw: str) -> list[str]:
    devices = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"cpu", "torch_cpu", "cuda", "auto"}
    if not devices:
        raise argparse.ArgumentTypeError("devices must contain at least one device")
    invalid = [device for device in devices if device not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown device={invalid[0]!r}; expected one of {sorted(allowed)}")
    return devices


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {raw!r}")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {raw!r}")
    return value


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    idx = math.ceil((percentile / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _flatten_features(features: dict[str, list[float]]) -> list[float]:
    flat: list[float] = []
    for node_id in sorted(features):
        flat.extend(float(value) for value in features[node_id])
    if not flat:
        raise ValueError("features must contain at least one value")
    return flat


def _max_abs_diff(a: dict[str, list[float]], b: dict[str, list[float]]) -> float:
    if set(a) != set(b):
        return float("inf")
    max_diff = 0.0
    for node_id in sorted(a):
        av = a[node_id]
        bv = b[node_id]
        if len(av) != len(bv):
            return float("inf")
        for left, right in zip(av, bv):
            max_diff = max(max_diff, abs(float(left) - float(right)))
    return max_diff


def _measure_forward_once(
    eval_output_dir: Path,
    *,
    k: int,
    feature_dim: int,
    steps: int,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], float, int]:
    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = run_fixed_topology_forward_features(
            eval_output_dir,
            k=k,
            feature_dim=feature_dim,
            steps=steps,
            seed=seed,
            device=device,
        )
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, int(peak)


def _build_device_report(
    eval_output_dir: Path,
    *,
    device: str,
    k: int,
    feature_dim: int,
    steps: int,
    seeds: list[int],
    runtime_repeats: int,
    baseline_outputs_by_seed: dict[int, dict[str, list[float]]] | None,
    parity_tolerance: float,
) -> tuple[dict[str, Any], dict[int, dict[str, list[float]]] | None]:
    try:
        device_info = resolve_runtime_device(device)
    except ValueError as exc:
        return (
            {
                "device": device,
                "available": False,
                "status": "backend_unavailable",
                "reason": str(exc),
                "parity_ok": False,
                "runtime_ok": False,
                "memory_ok": False,
            },
            baseline_outputs_by_seed,
        )

    durations: list[float] = []
    peak_bytes: list[int] = []
    seed_rows: list[dict[str, Any]] = []
    first_outputs_by_seed: dict[int, dict[str, list[float]]] = {}
    output_checksums_by_seed: dict[int, str] = {}
    input_checksums_by_seed: dict[int, str] = {}
    summaries: list[dict[str, Any]] = []

    for seed in seeds:
        seed_durations: list[float] = []
        seed_peak_bytes: list[int] = []
        first_result: dict[str, Any] | None = None
        for repeat in range(runtime_repeats):
            result, elapsed, peak = _measure_forward_once(
                eval_output_dir,
                k=k,
                feature_dim=feature_dim,
                steps=steps,
                seed=seed,
                device=device,
            )
            durations.append(elapsed)
            peak_bytes.append(peak)
            seed_durations.append(elapsed)
            seed_peak_bytes.append(peak)
            if repeat == 0:
                first_result = result
                first_outputs_by_seed[seed] = result["output_features"]
                output_checksums_by_seed[seed] = result["summary"]["output_checksum"]
                input_checksums_by_seed[seed] = result["summary"]["input_checksum"]
                summaries.append(result["summary"])
        if first_result is None:
            raise ValueError("runtime_repeats must be >= 1")
        seed_rows.append(
            {
                "seed": seed,
                "input_checksum": input_checksums_by_seed[seed],
                "output_checksum": output_checksums_by_seed[seed],
                "duration_seconds": seed_durations,
                "duration_median_seconds": statistics.median(seed_durations),
                "peak_bytes": seed_peak_bytes,
                "peak_max_bytes": max(seed_peak_bytes),
            }
        )

    is_baseline = baseline_outputs_by_seed is None
    if is_baseline:
        baseline_outputs_by_seed = first_outputs_by_seed
    parity_diffs = {
        str(seed): _max_abs_diff(first_outputs_by_seed[seed], baseline_outputs_by_seed[seed])
        for seed in seeds
    }
    parity_ok = all(diff <= parity_tolerance for diff in parity_diffs.values())
    output_checksum_matches = {
        str(seed): output_checksums_by_seed[seed] == summaries[0]["output_checksum"] if len(seeds) == 1 else None
        for seed in seeds
    }
    report = {
        "device": device,
        "available": True,
        "status": "backend_compare_device_ok",
        "baseline": is_baseline,
        "parity_ok": parity_ok,
        "parity_tolerance": parity_tolerance,
        "max_abs_diff_by_seed": parity_diffs,
        "max_abs_diff": max(parity_diffs.values()) if parity_diffs else 0.0,
        "output_checksum_by_seed": {str(seed): checksum for seed, checksum in output_checksums_by_seed.items()},
        "input_checksum_by_seed": {str(seed): checksum for seed, checksum in input_checksums_by_seed.items()},
        "output_checksum_matches_single_seed": output_checksum_matches,
        "duration_seconds": durations,
        "duration_median_seconds": statistics.median(durations),
        "duration_p95_seconds": _percentile_nearest_rank(durations, 95.0),
        "duration_max_seconds": max(durations),
        "peak_bytes": peak_bytes,
        "peak_median_bytes": int(statistics.median(peak_bytes)),
        "peak_max_bytes": max(peak_bytes),
        "runtime_ok": True,
        "memory_ok": True,
        "finite": all(bool(summary["finite"]) for summary in summaries),
        "adjacency_name": summaries[0]["adjacency_name"],
        "k": summaries[0]["k"],
        "node_count": summaries[0]["node_count"],
        "edge_count": summaries[0]["edge_count"],
        "max_out_degree": summaries[0]["max_out_degree"],
        "device_info": device_info,
        "rows": seed_rows,
    }
    return report, baseline_outputs_by_seed


def build_backend_compare_report(
    eval_output_dir: str | Path,
    *,
    k: int = 1,
    feature_dim: int = 8,
    steps: int = 1,
    seeds: list[int] | None = None,
    devices: list[str] | None = None,
    runtime_repeats: int = 3,
    parity_tolerance: float = 0.0,
) -> dict[str, Any]:
    if runtime_repeats < 1:
        raise ValueError(f"runtime_repeats must be >= 1, got {runtime_repeats}")
    if parity_tolerance < 0.0:
        raise ValueError(f"parity_tolerance must be >= 0, got {parity_tolerance}")
    resolved_seeds = [0, 1, 2] if seeds is None else list(seeds)
    if not resolved_seeds:
        raise ValueError("seeds must contain at least one seed")
    resolved_devices = ["cpu", "torch_cpu", "auto"] if devices is None else list(devices)
    if not resolved_devices:
        raise ValueError("devices must contain at least one device")

    eval_dir = Path(eval_output_dir)
    device_reports: list[dict[str, Any]] = []
    baseline_outputs_by_seed: dict[int, dict[str, list[float]]] | None = None
    baseline_device: str | None = None
    for device in resolved_devices:
        device_report, baseline_outputs_by_seed = _build_device_report(
            eval_dir,
            device=device,
            k=k,
            feature_dim=feature_dim,
            steps=steps,
            seeds=resolved_seeds,
            runtime_repeats=runtime_repeats,
            baseline_outputs_by_seed=baseline_outputs_by_seed,
            parity_tolerance=parity_tolerance,
        )
        if device_report.get("available") and baseline_device is None:
            baseline_device = device
        device_reports.append(device_report)

    available_reports = [report for report in device_reports if bool(report.get("available"))]
    if not available_reports:
        raise ValueError("no requested backend device was available")
    parity_ok = all(bool(report.get("parity_ok")) for report in available_reports)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "backend_compare_ok",
        "eval_output_dir": str(eval_dir),
        "k": k,
        "feature_dim": feature_dim,
        "steps": steps,
        "seeds": resolved_seeds,
        "devices": resolved_devices,
        "runtime_repeats": runtime_repeats,
        "parity_tolerance": parity_tolerance,
        "baseline_device": baseline_device,
        "available_device_count": len(available_reports),
        "unavailable_device_count": len(device_reports) - len(available_reports),
        "parity_ok": parity_ok,
        "runtime_ok": all(bool(report.get("runtime_ok")) for report in available_reports),
        "memory_ok": all(bool(report.get("memory_ok")) for report in available_reports),
        "all_available_backends_finite": all(bool(report.get("finite")) for report in available_reports),
        "device_reports": device_reports,
        "note": "fixed-topology backend comparison only; no teacher checkpoint or online teacher inference",
    }


def validate_backend_compare_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad backend compare schema_version={report.get('schema_version')!r}")
    if report.get("status") != "backend_compare_ok":
        raise ValueError(f"bad backend compare status={report.get('status')!r}")
    device_reports = report.get("device_reports")
    if not isinstance(device_reports, list) or not device_reports:
        raise ValueError("backend compare device_reports must be a non-empty list")
    available = [entry for entry in device_reports if bool(entry.get("available"))]
    if not available:
        raise ValueError("backend compare requires at least one available backend")
    if bool(report.get("parity_ok")) is not all(bool(entry.get("parity_ok")) for entry in available):
        raise ValueError("backend compare parity_ok must match available device reports")
    for entry in available:
        if entry.get("status") != "backend_compare_device_ok":
            raise ValueError("available backend report has bad status")
        for key in ("duration_median_seconds", "duration_p95_seconds", "peak_max_bytes", "max_abs_diff"):
            value = entry.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"available backend report {key} must be finite")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "backend_compare_report_valid",
        "available_device_count": len(available),
        "parity_ok": bool(report["parity_ok"]),
        "baseline_device": report.get("baseline_device"),
    }


def write_backend_compare_report(report: dict[str, Any], output_path: str | Path) -> None:
    _write_json(Path(output_path), report)


def run_and_write_backend_compare_report(
    eval_output_dir: str | Path,
    output_dir: str | Path,
    *,
    k: int = 1,
    feature_dim: int = 8,
    steps: int = 1,
    seeds: list[int] | None = None,
    devices: list[str] | None = None,
    runtime_repeats: int = 3,
    parity_tolerance: float = 0.0,
) -> dict[str, Any]:
    report = build_backend_compare_report(
        eval_output_dir,
        k=k,
        feature_dim=feature_dim,
        steps=steps,
        seeds=seeds,
        devices=devices,
        runtime_repeats=runtime_repeats,
        parity_tolerance=parity_tolerance,
    )
    validate_backend_compare_report(report)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_backend_compare_report(report, out / BACKEND_COMPARE_REPORT_FILENAME)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare fixed-topology sparse-student runtime backends.")
    parser.add_argument("--eval-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=_positive_int, default=1)
    parser.add_argument("--feature-dim", type=_positive_int, default=8)
    parser.add_argument("--steps", type=_positive_int, default=1)
    parser.add_argument("--seeds", type=lambda raw: _parse_int_list(raw, name="seeds"), default=[0, 1, 2])
    parser.add_argument("--devices", type=_parse_devices, default=["cpu", "torch_cpu", "auto"])
    parser.add_argument("--runtime-repeats", type=_positive_int, default=3)
    parser.add_argument("--parity-tolerance", type=_nonnegative_float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_and_write_backend_compare_report(
            args.eval_output_dir,
            args.output_dir,
            k=args.k,
            feature_dim=args.feature_dim,
            steps=args.steps,
            seeds=args.seeds,
            devices=args.devices,
            runtime_repeats=args.runtime_repeats,
            parity_tolerance=args.parity_tolerance,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "backend_compare_report": str(Path(args.output_dir) / BACKEND_COMPARE_REPORT_FILENAME),
        "baseline_device": report["baseline_device"],
        "available_device_count": report["available_device_count"],
        "unavailable_device_count": report["unavailable_device_count"],
        "parity_ok": report["parity_ok"],
        "runtime_ok": report["runtime_ok"],
        "memory_ok": report["memory_ok"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
