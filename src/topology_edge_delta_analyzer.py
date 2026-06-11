from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


SCHEMA_VERSION = "topology_edge_delta_summary.v1"


def load_edge_delta_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"{p}:{line_no}: expected JSON object")
            records.append(rec)
    return records


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def numeric_summary(values: Iterable[float | int]) -> dict[str, float | int | None]:
    xs = [float(v) for v in values]
    if not xs:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None, "sum": 0.0}
    return {
        "count": len(xs),
        "min": min(xs),
        "p25": _quantile(xs, 0.25),
        "median": median(xs),
        "p75": _quantile(xs, 0.75),
        "max": max(xs),
        "mean": mean(xs),
        "sum": sum(xs),
    }


def _top_counter(counter: Counter[str], limit: int | None = None) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.most_common(limit)}


def _node_kind(label: Any) -> str:
    text = str(label)
    op = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(", text)
    if op:
        return op.group(1)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        return "leaf"
    if re.match(r"^-?[0-9]+(?:\.[0-9]+)?$", text):
        return "number"
    return "other"


def _edge_pattern(edge: dict[str, Any]) -> str:
    return f"{_node_kind(edge.get('src_label', ''))}->{_node_kind(edge.get('dst_label', ''))}"


def _raw_edge_label(edge: dict[str, Any]) -> str:
    return f"{edge.get('src_label', '')}->{edge.get('dst_label', '')}"


def _score_values(edges: Iterable[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for edge in edges:
        score = edge.get("score")
        if score is not None:
            out.append(float(score))
    return out


def _prediction_flip(record: dict[str, Any]) -> str:
    return f"{record.get('hand_pred')}->{record.get('learned_pred')}"


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    removed = record.get("removed_edges") or []
    extra = record.get("extra_edges") or []
    return {
        "sample_id": record.get("sample_id"),
        "outcome": record.get("outcome"),
        "target": record.get("target_expert"),
        "dense_pred": record.get("dense_pred"),
        "hand_pred": record.get("hand_pred"),
        "learned_pred": record.get("learned_pred"),
        "removed_count": len(removed),
        "extra_count": len(extra),
    }


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _pattern_polarity_rows(
    *,
    pattern_counts_by_outcome: dict[str, Counter[str]],
    pattern_record_counts_by_outcome: dict[str, Counter[str]],
    outcome_counts: Counter[str],
    win_label: str = "learned_win",
    loss_label: str = "learned_loss",
) -> list[dict[str, Any]]:
    win_counts = pattern_counts_by_outcome.get(win_label, Counter())
    loss_counts = pattern_counts_by_outcome.get(loss_label, Counter())
    win_record_counts = pattern_record_counts_by_outcome.get(win_label, Counter())
    loss_record_counts = pattern_record_counts_by_outcome.get(loss_label, Counter())
    win_total = sum(win_counts.values())
    loss_total = sum(loss_counts.values())
    win_record_total = int(outcome_counts.get(win_label, 0))
    loss_record_total = int(outcome_counts.get(loss_label, 0))
    patterns = sorted(set(win_counts) | set(loss_counts) | set(win_record_counts) | set(loss_record_counts))
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        win_count = int(win_counts.get(pattern, 0))
        loss_count = int(loss_counts.get(pattern, 0))
        win_rate = _rate(win_count, win_total)
        loss_rate = _rate(loss_count, loss_total)
        win_rec_count = int(win_record_counts.get(pattern, 0))
        loss_rec_count = int(loss_record_counts.get(pattern, 0))
        win_record_rate = _rate(win_rec_count, win_record_total)
        loss_record_rate = _rate(loss_rec_count, loss_record_total)
        rows.append(
            {
                "pattern": pattern,
                "win_count": win_count,
                "loss_count": loss_count,
                "support_count": win_count + loss_count,
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "polarity": win_rate - loss_rate,
                "win_record_count": win_rec_count,
                "loss_record_count": loss_rec_count,
                "record_support_count": win_rec_count + loss_rec_count,
                "win_record_rate": win_record_rate,
                "loss_record_rate": loss_record_rate,
                "record_polarity": win_record_rate - loss_record_rate,
            }
        )
    rows.sort(key=lambda r: (abs(float(r["polarity"])), int(r["support_count"]), abs(float(r["record_polarity"]))), reverse=True)
    return rows


def analyze_edge_delta_records(records: Iterable[dict[str, Any]], *, max_patterns: int = 20) -> dict[str, Any]:
    rows = list(records)
    outcome_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    prediction_flip_counts: Counter[str] = Counter()
    outcome_target_counts: dict[str, Counter[str]] = defaultdict(Counter)
    outcome_prediction_counts: dict[str, Counter[str]] = defaultdict(Counter)
    edge_count_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    edge_score_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    edge_kind_patterns: dict[str, Counter[str]] = defaultdict(Counter)
    edge_kind_record_patterns: dict[str, Counter[str]] = defaultdict(Counter)
    raw_node_patterns: dict[str, Counter[str]] = defaultdict(Counter)
    compact = []

    for record in rows:
        outcome = str(record.get("outcome", "unknown"))
        target = str(record.get("target_expert", "unknown"))
        flip = _prediction_flip(record)
        removed = list(record.get("removed_edges") or [])
        extra = list(record.get("extra_edges") or [])

        outcome_counts[outcome] += 1
        target_counts[target] += 1
        prediction_flip_counts[flip] += 1
        outcome_target_counts[outcome][target] += 1
        outcome_prediction_counts[outcome][flip] += 1
        compact.append(_compact_record(record))

        for group in ("all", outcome):
            edge_count_values[group]["removed_count"].append(float(len(removed)))
            edge_count_values[group]["extra_count"].append(float(len(extra)))
            edge_count_values[group]["delta_count"].append(float(len(extra) - len(removed)))

        for edge_set, edges in (("removed_edges", removed), ("extra_edges", extra)):
            scores = _score_values(edges)
            record_kind_patterns = set()
            for group in ("all", outcome):
                edge_score_values[group][edge_set].extend(scores)
            for edge in edges:
                kind_pattern = f"{edge_set}:{_edge_pattern(edge)}"
                raw_pattern = f"{edge_set}:{_raw_edge_label(edge)}"
                record_kind_patterns.add(kind_pattern)
                for group in ("all", outcome):
                    edge_kind_patterns[group][kind_pattern] += 1
                    raw_node_patterns[group][raw_pattern] += 1
            for pattern in record_kind_patterns:
                for group in ("all", outcome):
                    edge_kind_record_patterns[group][pattern] += 1

    edge_kind_polarity = _pattern_polarity_rows(
        pattern_counts_by_outcome=edge_kind_patterns,
        pattern_record_counts_by_outcome=edge_kind_record_patterns,
        outcome_counts=outcome_counts,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "n_records": len(rows),
        "outcome_counts": _top_counter(outcome_counts),
        "target_expert_counts": _top_counter(target_counts),
        "prediction_flip_counts": _top_counter(prediction_flip_counts),
        "outcome_target_counts": {outcome: _top_counter(counter) for outcome, counter in sorted(outcome_target_counts.items())},
        "outcome_prediction_flip_counts": {outcome: _top_counter(counter) for outcome, counter in sorted(outcome_prediction_counts.items())},
        "edge_count_summary": {group: {name: numeric_summary(vals) for name, vals in values.items()} for group, values in sorted(edge_count_values.items())},
        "edge_score_summary": {group: {name: numeric_summary(vals) for name, vals in values.items()} for group, values in sorted(edge_score_values.items())},
        "recurring_node_label_patterns": {group: _top_counter(counter, max_patterns) for group, counter in sorted(raw_node_patterns.items())},
        "recurring_edge_kind_patterns": {group: _top_counter(counter, max_patterns) for group, counter in sorted(edge_kind_patterns.items())},
        "edge_kind_polarity": edge_kind_polarity,
        "top_win_edge_kind_polarity": [row for row in edge_kind_polarity if float(row["polarity"]) > 0.0][:max_patterns],
        "top_loss_edge_kind_polarity": [row for row in edge_kind_polarity if float(row["polarity"]) < 0.0][:max_patterns],
        "compact_records": compact,
    }


def format_compact_table(records: Iterable[dict[str, Any]]) -> str:
    rows = [_compact_record(r) for r in records]
    columns = ["sample_id", "outcome", "target", "dense_pred", "hand_pred", "learned_pred", "removed_count", "extra_count"]
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)
    lines = [header, sep]
    for row in rows:
        lines.append("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
    return "\n".join(lines)


def format_polarity_table(summary: dict[str, Any], *, limit: int = 20) -> str:
    rows = list(summary.get("edge_kind_polarity", []) or [])[:limit]
    columns = [
        "pattern",
        "win_count",
        "loss_count",
        "win_rate",
        "loss_rate",
        "polarity",
        "win_record_rate",
        "loss_record_rate",
        "record_polarity",
    ]

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(fmt(row.get(col, ""))))
    lines = ["  ".join(col.ljust(widths[col]) for col in columns)]
    lines.append("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        lines.append("  ".join(fmt(row.get(col, "")).ljust(widths[col]) for col in columns))
    return "\n".join(lines)


def _summary_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_counter(section: str, group: str, counter: dict[str, int]) -> None:
        for key, value in counter.items():
            rows.append({"section": section, "group": group, "key": key, "count": value})

    add_counter("outcome_counts", "all", summary.get("outcome_counts", {}))
    add_counter("target_expert_counts", "all", summary.get("target_expert_counts", {}))
    add_counter("prediction_flip_counts", "all", summary.get("prediction_flip_counts", {}))
    for group, counter in (summary.get("outcome_target_counts", {}) or {}).items():
        add_counter("outcome_target_counts", group, counter)
    for group, counter in (summary.get("outcome_prediction_flip_counts", {}) or {}).items():
        add_counter("outcome_prediction_flip_counts", group, counter)
    for group, counter in (summary.get("recurring_edge_kind_patterns", {}) or {}).items():
        add_counter("recurring_edge_kind_patterns", group, counter)
    for group, counter in (summary.get("recurring_node_label_patterns", {}) or {}).items():
        add_counter("recurring_node_label_patterns", group, counter)

    for row in summary.get("edge_kind_polarity", []) or []:
        out = {"section": "edge_kind_polarity", "group": "learned_win_vs_loss", "key": row.get("pattern")}
        out.update(row)
        rows.append(out)

    for section in ("edge_count_summary", "edge_score_summary"):
        for group, metrics_by_key in (summary.get(section, {}) or {}).items():
            for key, metrics in metrics_by_key.items():
                row = {"section": section, "group": group, "key": key}
                row.update(metrics)
                rows.append(row)
    return rows


def write_edge_delta_summary(summary: dict[str, Any], *, json_out: str | Path | None = None, csv_out: str | Path | None = None) -> None:
    if json_out:
        p = Path(json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_out:
        p = Path(csv_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = _summary_csv_rows(summary)
        fieldnames = ["section", "group", "key", "count", "min", "p25", "median", "p75", "max", "mean", "sum", "pattern", "win_count", "loss_count", "support_count", "win_rate", "loss_rate", "polarity", "win_record_count", "loss_record_count", "record_support_count", "win_record_rate", "loss_record_rate", "record_polarity"]
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate learned-vs-hand topology edge-delta JSONL artifacts.")
    parser.add_argument("--input", default="runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl")
    parser.add_argument("--json-out", default="runs/diagnostics/learned_k4_vs_hand_k4_edge_delta_summary.json")
    parser.add_argument("--csv-out", default="runs/diagnostics/learned_k4_vs_hand_k4_edge_delta_summary.csv")
    parser.add_argument("--compact", action="store_true", help="Print paired flips as a compact table.")
    parser.add_argument("--polarity", action="store_true", help="Print edge-kind win/loss polarity table.")
    parser.add_argument("--max-patterns", type=int, default=20)
    args = parser.parse_args()

    records = load_edge_delta_records(args.input)
    summary = analyze_edge_delta_records(records, max_patterns=args.max_patterns)
    write_edge_delta_summary(summary, json_out=args.json_out, csv_out=args.csv_out)
    if args.polarity:
        print(format_polarity_table(summary, limit=args.max_patterns))
    elif args.compact:
        print(format_compact_table(records))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
