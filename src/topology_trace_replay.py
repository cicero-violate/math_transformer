from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def iter_trace_records(paths: Iterable[str | Path]) -> Iterable[tuple[Path, int, JsonDict]]:
    """Yield `(path, line_no, record)` triples from one or more JSONL trace files."""
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"trace row must be an object: {path}:{line_no}")
                yield path, line_no, row


def is_failure_trace(row: JsonDict) -> bool:
    """Return true when a standardized trace row represents a failed learned route."""
    diag = row.get("diagnostics") or {}
    pred = row.get("prediction") or {}
    if bool(diag.get("failure")):
        return True
    if "learned_correct" in pred:
        return not bool(pred.get("learned_correct"))
    return False


def replay_score(row: JsonDict) -> float:
    """Score trace rows for replay value.

    Higher means more useful for failure-driven scorer improvement. The score is
    intentionally simple and transparent. It is for data selection, not model
    evaluation.
    """
    diag = row.get("diagnostics") or {}
    pred = row.get("prediction") or {}
    agreement = row.get("agreement") or {}
    overlap = row.get("overlap") or {}

    score = 0.0
    if is_failure_trace(row):
        score += 10.0
    if bool(pred.get("dense_correct")) and not bool(pred.get("learned_correct", True)):
        score += 5.0
    if bool(pred.get("hand_correct")) and not bool(pred.get("learned_correct", True)):
        score += 3.0
    if bool(diag.get("is_generic_expert")):
        score += 2.0

    score += 0.25 * float(diag.get("missing_edge_count", overlap.get("missing_edges", 0)) or 0)
    score += 0.10 * float(diag.get("extra_edge_count", overlap.get("extra_edges", 0)) or 0)
    score += 2.0 * float(agreement.get("logit_kl", agreement.get("logit_kl_dense_to_sparse", 0.0)) or 0.0)
    score += 1.0 * float(agreement.get("hidden_l1", 0.0) or 0.0)
    if "hidden_cos" in agreement and agreement.get("hidden_cos") is not None:
        score += max(0.0, 1.0 - float(agreement.get("hidden_cos") or 0.0))
    return float(score)


def make_replay_candidate(path: Path, line_no: int, row: JsonDict) -> JsonDict:
    """Convert one trace row to a compact replay-candidate row."""
    pred = row.get("prediction") or {}
    diag = row.get("diagnostics") or {}
    overlap = row.get("overlap") or {}
    agreement = row.get("agreement") or {}
    candidate = {
        "source_trace": str(path),
        "source_line": line_no,
        "sample_id": row.get("sample_id"),
        "domain": row.get("domain"),
        "expr": row.get("expr"),
        "nodes_hash": row.get("nodes_hash"),
        "replay_score": replay_score(row),
        "failure": is_failure_trace(row),
        "target_expert": pred.get("target_expert") or (row.get("target") or {}).get("expert"),
        "target_expert_id": pred.get("target_expert_id") or (row.get("target") or {}).get("expert_id"),
        "learned_pred": pred.get("learned_pred"),
        "learned_pred_id": pred.get("learned_pred_id"),
        "dense_pred": pred.get("dense_pred"),
        "dense_pred_id": pred.get("dense_pred_id"),
        "hand_pred": pred.get("hand_pred"),
        "hand_pred_id": pred.get("hand_pred_id"),
        "scorer_checkpoint": row.get("scorer_checkpoint"),
        "feature_schema": row.get("feature_schema"),
        "topology_config": row.get("topology_config") or {},
        "missing_edge_count": diag.get("missing_edge_count", overlap.get("missing_edges", 0)),
        "extra_edge_count": diag.get("extra_edge_count", overlap.get("extra_edges", 0)),
        "is_generic_expert": bool(diag.get("is_generic_expert")),
        "agreement": {
            "hidden_l1": agreement.get("hidden_l1"),
            "hidden_cos": agreement.get("hidden_cos"),
            "logit_l1": agreement.get("logit_l1"),
            "logit_kl": agreement.get("logit_kl", agreement.get("logit_kl_dense_to_sparse")),
        },
    }
    if "env" in row:
        candidate["env"] = row["env"]
    return candidate


def select_replay_candidates(
    paths: Iterable[str | Path],
    *,
    max_records: int = 100,
    min_score: float = 0.0,
    failures_only: bool = True,
) -> list[JsonDict]:
    candidates: list[JsonDict] = []
    for path, line_no, row in iter_trace_records(paths):
        if failures_only and not is_failure_trace(row):
            continue
        candidate = make_replay_candidate(path, line_no, row)
        if float(candidate["replay_score"]) < min_score:
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda r: (-float(r["replay_score"]), str(r.get("source_trace")), int(r.get("source_line") or 0)))
    if max_records > 0:
        candidates = candidates[:max_records]
    return candidates


def summarize_replay_candidates(candidates: list[JsonDict]) -> JsonDict:
    by_expert: dict[str, int] = {}
    generic = 0
    for row in candidates:
        expert = str(row.get("target_expert") or "unknown")
        by_expert[expert] = by_expert.get(expert, 0) + 1
        generic += int(bool(row.get("is_generic_expert")))
    return {
        "records": len(candidates),
        "generic_expert_records": generic,
        "by_expert": dict(sorted(by_expert.items())),
        "max_replay_score": max((float(r["replay_score"]) for r in candidates), default=0.0),
        "mean_replay_score": (
            sum(float(r["replay_score"]) for r in candidates) / len(candidates)
            if candidates else 0.0
        ),
    }


def write_jsonl(path: str | Path, rows: Iterable[JsonDict]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Select replay candidates from standardized topology trace JSONL files.")
    parser.add_argument("traces", nargs="+", help="Input topology trace JSONL files.")
    parser.add_argument("--output", default="runs/replay/topology_replay_candidates.jsonl")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--include-successes", action="store_true", help="Include non-failure rows if their replay score passes the threshold.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    candidates = select_replay_candidates(
        args.traces,
        max_records=args.max_records,
        min_score=args.min_score,
        failures_only=not args.include_successes,
    )
    written = write_jsonl(args.output, candidates)
    summary = summarize_replay_candidates(candidates)
    summary["output"] = args.output
    summary["written"] = written
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary:
        out = Path(args.summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
