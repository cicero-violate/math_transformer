from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .eval import QualityReport, run_quality_eval

DEFAULT_K_VALUES = [2, 3, 4, 5, 6, 8, 12, 16]
DEFAULT_JSON_OUT = Path("runs/diagnostics/topology_k_sweep_summary.json")
DEFAULT_CSV_OUT = Path("runs/diagnostics/topology_k_sweep_summary.csv")


def _expert_accuracy(report: QualityReport, expert: str) -> float | None:
    stats = report.by_expert.get(expert)
    if not stats:
        return None
    return float(stats.get("accuracy", 0.0))


def _expert_correct(report: QualityReport, expert: str) -> int | None:
    stats = report.by_expert.get(expert)
    if not stats:
        return None
    return int(stats.get("correct", 0))


def _correct_count(report: QualityReport) -> int:
    if report.correct_count is not None:
        return int(report.correct_count)
    return int(round(report.route_accuracy * report.n_examples))


def _base_row(report: QualityReport, sweep_mode: str, k: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mode": sweep_mode,
        "report_mode": report.mode,
        "k": "full" if k is None else int(k),
        "route_acc": float(report.route_accuracy),
        "correct_count": _correct_count(report),
        "n_examples": int(report.n_examples),
        "dense_agreement": report.dense_agreement,
        "hidden_l1": report.hidden_l1,
        "hidden_cos": report.hidden_cos,
        "logit_l1": report.logit_l1,
        "logit_kl_dense_to_sparse": report.logit_kl_dense_to_sparse,
        "by_expert": report.by_expert,
        "_correct_by_example": list(report.correct_by_example),
    }
    for expert in sorted(report.by_expert):
        prefix = expert.replace(" ", "_")
        row[f"{prefix}_acc"] = _expert_accuracy(report, expert)
        row[f"{prefix}_correct"] = _expert_correct(report, expert)
        row[f"{prefix}_total"] = int(report.by_expert[expert].get("total", 0))
    return row


def _attach_comparisons(rows: list[dict[str, Any]]) -> None:
    dense = next((r for r in rows if r["mode"] == "dense"), None)
    hand_k4 = next((r for r in rows if r["mode"] == "hand" and r["k"] == 4), None)
    for row in rows:
        for label, base in (("dense", dense), ("hand_k4", hand_k4)):
            if base is None:
                row[f"wins_vs_{label}"] = None
                row[f"losses_vs_{label}"] = None
                row[f"correct_delta_vs_{label}"] = None
                row[f"route_acc_delta_vs_{label}"] = None
                continue
            delta = int(row["correct_count"]) - int(base["correct_count"])
            row_flags = row.get("_correct_by_example") or []
            base_flags = base.get("_correct_by_example") or []
            if len(row_flags) == len(base_flags) and row_flags:
                wins = sum(1 for current, baseline in zip(row_flags, base_flags) if current and not baseline)
                losses = sum(1 for current, baseline in zip(row_flags, base_flags) if baseline and not current)
            else:
                # Fallback for manually constructed legacy reports without per-example flags.
                wins = max(delta, 0)
                losses = max(-delta, 0)
            row[f"wins_vs_{label}"] = wins
            row[f"losses_vs_{label}"] = losses
            row[f"correct_delta_vs_{label}"] = delta
            row[f"route_acc_delta_vs_{label}"] = float(row["route_acc"]) - float(base["route_acc"])


def summarize_reports(
    dense_report: QualityReport,
    hand_reports: list[QualityReport],
    learned_reports: list[QualityReport],
) -> list[dict[str, Any]]:
    rows = [_base_row(dense_report, "dense", None)]
    rows.extend(_base_row(r, "hand", r.k) for r in hand_reports)
    rows.extend(_base_row(r, "learned", r.k) for r in learned_reports)
    _attach_comparisons(rows)
    for row in rows:
        row.pop("_correct_by_example", None)
    return rows


def run_topology_k_sweep(
    examples_path: str,
    checkpoint: str | None,
    k_values: list[int] | None = None,
    learned_scorer_checkpoint: str | None = None,
    device: str | None = "auto",
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    topk: int = 3,
    local_window: int = 1,
    topology_mode: str = "middle_preserving_topk",
    middle_bridge_width: int = 0,
    json_out: str | Path = DEFAULT_JSON_OUT,
    csv_out: str | Path = DEFAULT_CSV_OUT,
) -> list[dict[str, Any]]:
    ks = list(k_values or DEFAULT_K_VALUES)
    dense_report: QualityReport | None = None
    hand_reports: list[QualityReport] = []
    for k in ks:
        # Plan v19 hand sweep semantics are explicit: --fixed-k "$k" --quality-k "$k".
        # Do not batch hand K values under one fixed_k=max(K), because fixed_k changes
        # the topology mask before neighbor truncation and can change quality results.
        reports = run_quality_eval(
            examples_path=examples_path,
            k_values=[k],
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            topk=topk,
            local_window=local_window,
            checkpoint=checkpoint,
            device=device,
            topology_mode=topology_mode,
            fixed_k=k,
            middle_bridge_width=middle_bridge_width,
        )
        if dense_report is None:
            dense_report = reports[0]
        hand_reports.extend(reports[1:])
    if dense_report is None:
        dense_only = run_quality_eval(
            examples_path=examples_path,
            k_values=[],
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            topk=topk,
            local_window=local_window,
            checkpoint=checkpoint,
            device=device,
            topology_mode=topology_mode,
            fixed_k=0,
            middle_bridge_width=middle_bridge_width,
        )
        dense_report = dense_only[0]

    learned_reports: list[QualityReport] = []
    if learned_scorer_checkpoint:
        for k in ks:
            reports = run_quality_eval(
                examples_path=examples_path,
                k_values=[],
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                topk=topk,
                local_window=local_window,
                checkpoint=checkpoint,
                device=device,
                topology_mode=topology_mode,
                fixed_k=max(ks) if ks else k,
                middle_bridge_width=middle_bridge_width,
                learned_scorer_checkpoint=learned_scorer_checkpoint,
                learned_k=k,
            )
            learned_reports.extend(r for r in reports if r.mode == "learned_topology")

    rows = summarize_reports(dense_report, hand_reports, learned_reports)
    write_summary(rows, json_out=json_out, csv_out=csv_out)
    return rows


def write_summary(rows: list[dict[str, Any]], json_out: str | Path, csv_out: str | Path) -> None:
    json_path = Path(json_out)
    csv_path = Path(csv_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key in {"by_expert", "_correct_by_example"}:
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-K topology quality sweep and emit summary artifacts.")
    parser.add_argument("--examples", default="data/synthetic_hard/val.jsonl")
    parser.add_argument("--checkpoint", default="runs/checkpoints/synthetic_hard_dense.pt")
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES), dest="k_values")
    parser.add_argument("--learned-scorer-checkpoint", default="runs/checkpoints/topology_scorer.champion.pt", dest="learned_scorer_checkpoint")
    parser.add_argument("--no-learned", action="store_true", dest="no_learned")
    parser.add_argument("--quality-device", default="auto", dest="quality_device")
    parser.add_argument("--d-model", type=int, default=64, dest="d_model")
    parser.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    parser.add_argument("--n-layers", type=int, default=2, dest="n_layers")
    parser.add_argument("--d-ff", type=int, default=128, dest="d_ff")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--local-window", type=int, default=1, dest="local_window")
    parser.add_argument("--topology-mode", default="middle_preserving_topk", choices=["union", "scored_topk", "middle_preserving_topk"])
    parser.add_argument("--middle-bridge-width", type=int, default=0, dest="middle_bridge_width")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), dest="json_out")
    parser.add_argument("--csv-out", default=str(DEFAULT_CSV_OUT), dest="csv_out")
    args = parser.parse_args()

    ks = [int(x) for x in args.k_values.split(",") if x.strip()]
    learned_ckpt = None if args.no_learned else args.learned_scorer_checkpoint
    rows = run_topology_k_sweep(
        examples_path=args.examples,
        checkpoint=args.checkpoint,
        k_values=ks,
        learned_scorer_checkpoint=learned_ckpt,
        device=args.quality_device,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        topk=args.topk,
        local_window=args.local_window,
        topology_mode=args.topology_mode,
        middle_bridge_width=args.middle_bridge_width,
        json_out=args.json_out,
        csv_out=args.csv_out,
    )
    print(f"wrote {args.json_out}")
    print(f"wrote {args.csv_out}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
