from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


SCHEMA_VERSION = "locked_speed_distribution.v1"
DEFAULT_REQUIRED_POLICIES = ["dense_full", "hand_k4", "learned_k4", "current_champion_k8"]


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _numeric(values: Iterable[float]) -> dict[str, Any]:
    xs = [float(v) for v in values]
    if not xs:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    return {
        "count": len(xs),
        "min": min(xs),
        "p25": _quantile(xs, 0.25),
        "median": median(xs),
        "p75": _quantile(xs, 0.75),
        "max": max(xs),
        "mean": mean(xs),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(data)
    return rows


def load_artifacts(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.glob("*.json")):
                artifacts.append(json.loads(child.read_text(encoding="utf-8")))
            for child in sorted(p.glob("*.jsonl")):
                artifacts.extend(_read_jsonl(child))
        elif p.suffix == ".jsonl":
            artifacts.extend(_read_jsonl(p))
        else:
            artifacts.append(json.loads(p.read_text(encoding="utf-8")))
    return artifacts


def _first_positive(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if f > 0.0:
            return f
    return None


def _run_key(artifact: dict[str, Any], ordinal: int) -> str:
    cfg = artifact.get("config", {}) or {}
    seed = cfg.get("bench_seed")
    bench_n = cfg.get("bench_n")
    node_mode = cfg.get("bench_node_mode")
    if seed is None:
        return f"artifact-{ordinal}"
    return f"seed={seed}|n={bench_n}|node_mode={node_mode}"


def speed_observations(artifact: dict[str, Any], *, ordinal: int = 0) -> list[dict[str, Any]]:
    cfg = artifact.get("config", {}) or {}
    quality = artifact.get("quality", {}) or {}
    speed = artifact.get("speed", {}) or {}
    reports = artifact.get("reports", {}) or {}
    hand_report = reports.get("hand", {}) or {}
    learned_report = reports.get("learned", {}) or {}

    hand_k = cfg.get("hand_k") or (quality.get("hand", {}) or {}).get("k")
    learned_k = cfg.get("learned_k") or (quality.get("learned", {}) or {}).get("k")
    run_key = _run_key(artifact, ordinal)
    common = {
        "run_key": run_key,
        "bench_seed": cfg.get("bench_seed"),
        "bench_n": cfg.get("bench_n"),
        "bench_node_mode": cfg.get("bench_node_mode"),
        "artifact_index": ordinal,
    }
    observations: list[dict[str, Any]] = []

    dense_ms = _first_positive(hand_report.get("full_block_ms"), learned_report.get("full_block_ms"))
    if dense_ms is not None:
        observations.append({**common, "policy": "dense_full", "block_ms": dense_ms, "source": "reports.full_block_ms"})

    hand_ms = _first_positive(speed.get("hand_block_ms"))
    if hand_ms is not None and hand_k is not None:
        observations.append({**common, "policy": f"hand_k{int(hand_k)}", "block_ms": hand_ms, "source": "speed.hand_block_ms"})

    learned_ms = _first_positive(speed.get("learned_block_ms"))
    if learned_ms is not None and learned_k is not None:
        labels = [f"learned_k{int(learned_k)}"]
        if int(learned_k) == 8:
            labels.append("current_champion_k8")
        for label in dict.fromkeys(labels):
            observations.append({**common, "policy": label, "block_ms": learned_ms, "source": "speed.learned_block_ms"})
    return observations


def aggregate_locked_speed(
    artifacts: Iterable[dict[str, Any]],
    *,
    required_policies: Iterable[str] = DEFAULT_REQUIRED_POLICIES,
    baseline_policy: str = "hand_k4",
    tolerance_ms: float = 0.05,
    min_pass_rate: float = 0.75,
) -> dict[str, Any]:
    artifact_rows = list(artifacts)
    observations: list[dict[str, Any]] = []
    for idx, artifact in enumerate(artifact_rows):
        observations.extend(speed_observations(artifact, ordinal=idx))

    values_by_policy: dict[str, list[float]] = defaultdict(list)
    by_run_policy: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for obs in observations:
        policy = str(obs["policy"])
        value = float(obs["block_ms"])
        values_by_policy[policy].append(value)
        by_run_policy[str(obs["run_key"])][policy].append(value)

    policy_run_values: dict[str, list[float]] = defaultdict(list)
    for run_values in by_run_policy.values():
        for policy, values in run_values.items():
            policy_run_values[policy].append(float(median(values)))

    policies: dict[str, dict[str, Any]] = {}
    for policy, values in sorted(policy_run_values.items()):
        stats = _numeric(values)
        comparable = 0
        passed = 0
        for run_values in by_run_policy.values():
            if policy not in run_values or baseline_policy not in run_values:
                continue
            comparable += 1
            policy_ms = median(run_values[policy])
            baseline_ms = median(run_values[baseline_policy])
            if policy_ms <= baseline_ms + tolerance_ms:
                passed += 1
        stats["pass_count"] = passed if comparable else None
        stats["comparable_count"] = comparable
        stats["pass_rate"] = (passed / comparable) if comparable else None
        policies[policy] = stats

    required = list(required_policies)
    missing = [policy for policy in required if policy not in policies]
    gated = [p for p in required if p not in {baseline_policy, "dense_full"}]
    failing = []
    baseline_median = (policies.get(baseline_policy) or {}).get("median")
    for policy in gated:
        stats = policies.get(policy)
        if not stats:
            continue
        median_ok = True if baseline_median is None else float(stats["median"]) <= float(baseline_median) + tolerance_ms
        pass_rate = stats.get("pass_rate")
        pass_rate_ok = pass_rate is not None and float(pass_rate) >= min_pass_rate
        if not (median_ok and pass_rate_ok):
            failing.append({"policy": policy, "median_ok": median_ok, "pass_rate_ok": pass_rate_ok, "median": stats.get("median"), "baseline_median": baseline_median, "pass_rate": pass_rate})

    acceptance_flags: dict[str, list[float]] = defaultdict(list)
    for artifact in artifact_rows:
        acceptance = artifact.get("acceptance", {}) or {}
        for key in ("quality_ok", "speed_ok", "strict_speed_ok", "passed"):
            if key in acceptance:
                acceptance_flags[key].append(1.0 if bool(acceptance[key]) else 0.0)

    return {
        "schema_version": SCHEMA_VERSION,
        "n_artifacts": len(artifact_rows),
        "n_observations": len(observations),
        "baseline_policy": baseline_policy,
        "required_policies": required,
        "tolerance_ms": tolerance_ms,
        "min_pass_rate": min_pass_rate,
        "policies": policies,
        "pair_acceptance": {key: _numeric(vals) for key, vals in sorted(acceptance_flags.items())},
        "gate": {"speed_distribution_ok": not missing and not failing, "missing_policies": missing, "failing_policies": failing},
    }


def format_speed_table(summary: dict[str, Any]) -> str:
    columns = ["policy", "count", "median", "p25", "p75", "pass_rate"]
    rows = []
    for policy, stats in (summary.get("policies", {}) or {}).items():
        rows.append({"policy": policy, "count": stats.get("count"), "median": stats.get("median"), "p25": stats.get("p25"), "p75": stats.get("p75"), "pass_rate": stats.get("pass_rate")})
    rows.sort(key=lambda r: str(r["policy"]))

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(fmt(row.get(col))))
    lines = ["  ".join(col.ljust(widths[col]) for col in columns)]
    lines.append("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        lines.append("  ".join(fmt(row.get(col)).ljust(widths[col]) for col in columns))
    return "\n".join(lines)


def write_locked_speed_summary(summary: dict[str, Any], *, json_out: str | Path | None = None, csv_out: str | Path | None = None) -> None:
    if json_out:
        p = Path(json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_out:
        p = Path(csv_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["policy", "count", "min", "p25", "median", "p75", "max", "mean", "pass_count", "comparable_count", "pass_rate"]
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for policy, stats in sorted((summary.get("policies", {}) or {}).items()):
                row = {"policy": policy}
                row.update(stats)
                writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated locked topology benchmark artifacts.")
    parser.add_argument("--artifacts", nargs="+", default=["runs/benchmarks/repeated_locked_protocol_artifacts.jsonl"])
    parser.add_argument("--json-out", default="runs/benchmarks/repeated_locked_speed_summary.json")
    parser.add_argument("--csv-out", default="runs/benchmarks/repeated_locked_speed_summary.csv")
    parser.add_argument("--required-policies", default=",".join(DEFAULT_REQUIRED_POLICIES))
    parser.add_argument("--baseline-policy", default="hand_k4")
    parser.add_argument("--tolerance-ms", type=float, default=0.05)
    parser.add_argument("--min-pass-rate", type=float, default=0.75)
    args = parser.parse_args()

    required = [x.strip() for x in args.required_policies.split(",") if x.strip()]
    summary = aggregate_locked_speed(load_artifacts(args.artifacts), required_policies=required, baseline_policy=args.baseline_policy, tolerance_ms=args.tolerance_ms, min_pass_rate=args.min_pass_rate)
    write_locked_speed_summary(summary, json_out=args.json_out, csv_out=args.csv_out)
    print(format_speed_table(summary))
    print(f"speed_distribution_ok={summary['gate']['speed_distribution_ok']}")
    if not summary["gate"]["speed_distribution_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
