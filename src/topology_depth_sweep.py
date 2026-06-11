"""
Topology depth sweep — P0.4.

Measures route_accuracy at L ∈ {1, 2, 4, 8} with fixed K before claiming
multi-hop value from T_outer > 1.

Non-negotiable gate 4: every topology quality report must include
generic and affine accuracy alongside route_accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "topology_depth_sweep.v1"
DEFAULT_L_VALUES = [1, 2, 4, 8]
DEFAULT_K = 4


@dataclass
class DepthSweepRow:
    l: int
    k: int | None     # None for dense/full mode (k=None in QualityReport)
    topology_mode: str
    route_accuracy: float
    generic_accuracy: float | None
    affine_accuracy: float | None
    n_examples: int
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "l": self.l,
            "k": self.k,
            "topology_mode": self.topology_mode,
            "route_accuracy": self.route_accuracy,
            "generic_accuracy": self.generic_accuracy,
            "affine_accuracy": self.affine_accuracy,
            "n_examples": self.n_examples,
            "notes": self.notes,
        }


def _extract_expert_acc(report, expert: str) -> tuple[float | None, int, int]:
    """Extract (accuracy, correct, total) for a named expert from a QualityReport."""
    if not report.by_expert:
        return None, 0, 0
    entry = report.by_expert.get(expert)
    if not entry:
        return None, 0, 0
    correct = int(entry.get("correct", 0))
    total = int(entry.get("total", 0))
    acc = correct / total if total > 0 else None
    return acc, correct, total


def run_topology_depth_sweep(
    examples_path: str | Path,
    checkpoint: str | Path,
    l_values: list[int] = DEFAULT_L_VALUES,
    fixed_k: int = DEFAULT_K,
    device: str = "cpu",
    d_model: int = 128,
    n_heads: int = 4,
    d_ff: int = 256,
    topk: int = 4,
    local_window: int = 2,
    topology_mode: str = "middle_preserving_topk",
    middle_bridge_width: int = 0,
    json_out: str | Path | None = None,
    csv_out: str | Path | None = None,
) -> list[DepthSweepRow]:
    """
    Evaluate route_accuracy, generic_accuracy, affine_accuracy
    at each L in l_values with the given fixed K.

    Every row includes generic and affine accuracy (gate 4).
    Returns list[DepthSweepRow] in l_values order.
    """
    from .eval import run_quality_eval

    rows: list[DepthSweepRow] = []

    for l in l_values:
        reports = run_quality_eval(
            examples_path=str(examples_path),
            k_values=[fixed_k],
            d_model=d_model,
            n_heads=n_heads,
            n_layers=l,
            d_ff=d_ff,
            topk=topk,
            local_window=local_window,
            checkpoint=str(checkpoint),
            device=device,
            topology_mode=topology_mode,
            fixed_k=fixed_k,
            middle_bridge_width=middle_bridge_width,
        )
        for report in reports:
            generic_acc, _, _ = _extract_expert_acc(report, "generic_expert")
            affine_acc, _, _ = _extract_expert_acc(report, "affine_expert")
            k_val = getattr(report, "k", fixed_k)
            # Single-checkpoint L-sweep: results reflect a fixed-checkpoint eval,
            # not per-L training. Valid only for relative comparison; a strict
            # L-sweep proof requires per-L checkpoints (plan gate P0.3).
            note = "single_checkpoint_l_sweep" if len(l_values) > 1 else ""
            rows.append(DepthSweepRow(
                l=l,
                k=k_val,
                topology_mode=topology_mode,
                route_accuracy=report.route_accuracy,
                generic_accuracy=generic_acc,
                affine_accuracy=affine_acc,
                n_examples=report.n_examples,
                notes=note,
            ))

    if json_out is not None or csv_out is not None:
        write_depth_sweep_summary(rows, json_out=json_out, csv_out=csv_out)

    return rows


def write_depth_sweep_summary(
    rows: list[DepthSweepRow],
    *,
    json_out: str | Path | None = None,
    csv_out: str | Path | None = None,
) -> None:
    data = [r.as_dict() for r in rows]
    if json_out is not None:
        Path(json_out).write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "rows": data,
        }, indent=2))
    if csv_out is not None and data:
        with Path(csv_out).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)


def format_depth_sweep_table(rows: list[DepthSweepRow]) -> str:
    header = f"{'L':<4} {'K':<6} {'topology_mode':<30} {'route_acc':<11} {'generic_acc':<13} {'affine_acc':<11} {'n'}"
    sep = "-" * 87
    lines = [header, sep]
    for r in rows:
        k_str = "full" if r.k is None else str(r.k)
        gen = f"{r.generic_accuracy:.4f}" if r.generic_accuracy is not None else "N/A    "
        aff = f"{r.affine_accuracy:.4f}" if r.affine_accuracy is not None else "N/A    "
        lines.append(
            f"{r.l:<4} {k_str:<6} {r.topology_mode:<30} {r.route_accuracy:.4f}      "
            f"{gen:<13} {aff:<11} {r.n_examples}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Topology depth sweep: quality at L ∈ {1,2,4,8}")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--l-values", nargs="+", type=int, default=DEFAULT_L_VALUES)
    parser.add_argument("--fixed-k", type=int, default=DEFAULT_K)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--topology-mode", default="middle_preserving_topk")
    parser.add_argument("--middle-bridge-width", type=int, default=0)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    rows = run_topology_depth_sweep(
        examples_path=args.examples,
        checkpoint=args.checkpoint,
        l_values=args.l_values,
        fixed_k=args.fixed_k,
        device=args.device,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        topology_mode=args.topology_mode,
        middle_bridge_width=args.middle_bridge_width,
        json_out=args.json_out,
        csv_out=args.csv_out,
    )
    print(format_depth_sweep_table(rows))


if __name__ == "__main__":
    main()
