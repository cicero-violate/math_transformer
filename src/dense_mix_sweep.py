from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

_RE = re.compile(
    r"mode=(?P<mode>\S+)\s+k=(?P<k>\S+)\s+examples=(?P<examples>\d+)\s+route_acc=(?P<route_acc>[0-9.]+)"
    r"(?:\s+dense_agree=(?P<dense_agree>[0-9.]+))?"
    r"(?:\s+hidden_l1=(?P<hidden_l1>[0-9.]+))?"
    r"(?:\s+hidden_cos=(?P<hidden_cos>[0-9.]+))?"
    r"(?:\s+logit_l1=(?P<logit_l1>[0-9.]+))?"
    r"(?:\s+logit_kl=(?P<logit_kl>[0-9.]+))?"
)


def _f(groups: dict[str, str | None], key: str) -> float | None:
    return float(groups[key]) if groups.get(key) is not None else None


def parse_quality_reports(text: str) -> list[dict[str, float | int | str | None]]:
    rows = []
    for m in _RE.finditer(text):
        g = m.groupdict()
        rows.append({
            "mode": g["mode"],
            "k": g["k"],
            "examples": int(g["examples"]),
            "route_acc": float(g["route_acc"]),
            "dense_agree": _f(g, "dense_agree"),
            "hidden_l1": _f(g, "hidden_l1"),
            "hidden_cos": _f(g, "hidden_cos"),
            "logit_l1": _f(g, "logit_l1"),
            "logit_kl": _f(g, "logit_kl"),
        })
    return rows


def quality_score(row: dict[str, float | int | str | None]) -> float:
    return (
        float(row["route_acc"])
        + 0.5 * float(row.get("hidden_cos") or 0.0)
        + 0.25 * float(row.get("dense_agree") or 0.0)
        - 2.0 * float(row.get("logit_kl") or 0.0)
        - 0.1 * float(row.get("hidden_l1") or 0.0)
    )


def score_log(text: str, mix: str) -> dict[str, float | int | str | None]:
    learned = [r for r in parse_quality_reports(text) if r["mode"] == "learned_topology"]
    if not learned:
        raise ValueError("quality log has no learned_topology report")
    row = dict(learned[-1])
    row["mix"] = mix
    row["score"] = quality_score(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--mix", action="append", default=[])
    ap.add_argument("--summary-csv", default=None)
    args = ap.parse_args()
    rows = []
    for i, log in enumerate(args.logs):
        mix = args.mix[i] if i < len(args.mix) else Path(log).stem
        rows.append(score_log(Path(log).read_text(), mix))
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    if args.summary_csv:
        out = Path(args.summary_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["mix", "score", "mode", "k", "examples", "route_acc", "dense_agree", "hidden_l1", "hidden_cos", "logit_l1", "logit_kl"]
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    for r in rows:
        print(
            f"mix={r['mix']} score={float(r['score']):.6f} route_acc={float(r['route_acc']):.4f} "
            f"dense_agree={float(r['dense_agree'] or 0):.4f} hidden_cos={float(r['hidden_cos'] or 0):.6f} "
            f"hidden_l1={float(r['hidden_l1'] or 0):.6f} logit_kl={float(r['logit_kl'] or 0):.6f}"
        )


if __name__ == "__main__":
    main()
