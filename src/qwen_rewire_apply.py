from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.qwen_rewire_acceptance import load_rewire_acceptance_report, validate_rewire_acceptance_report
from src.qwen_rewire_search import load_rewire_search_report, validate_rewire_search_report
from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency


SCHEMA_VERSION = "qwen_rewire_apply.v1"
ACCEPTED_CANDIDATE_MANIFEST_FILENAME = "accepted_candidate_manifest.json"
APPLIED_CANDIDATE_EVAL_DIRNAME = "applied_candidate_eval"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _copy_candidate_eval(candidate_eval_dir: Path, applied_eval_dir: Path, *, overwrite: bool) -> None:
    if not candidate_eval_dir.exists():
        raise FileNotFoundError(candidate_eval_dir)
    if applied_eval_dir.exists():
        if not overwrite:
            raise ValueError(f"applied candidate eval dir already exists: {applied_eval_dir}")
        shutil.rmtree(applied_eval_dir)
    shutil.copytree(candidate_eval_dir, applied_eval_dir)


def _selected_candidate(search_report: dict[str, Any]) -> dict[str, Any]:
    validate_rewire_search_report(search_report)
    idx = int(search_report["selected_candidate_index"])
    candidates = search_report["candidates"]
    if idx < 0 or idx >= len(candidates):
        raise ValueError("selected_candidate_index out of range")
    candidate = dict(candidates[idx])
    if not bool(candidate.get("accepted")):
        raise ValueError("selected search candidate is not accepted")
    if not bool(search_report.get("selected_candidate_accepted")):
        raise ValueError("search report selected_candidate_accepted must be true")
    return candidate


def build_accepted_candidate_manifest(
    *,
    rewire_search_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    search_dir = Path(rewire_search_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    search_report = load_rewire_search_report(search_dir)
    selected = _selected_candidate(search_report)
    acceptance_dir = Path(str(selected["acceptance_dir"]))
    acceptance_report = load_rewire_acceptance_report(acceptance_dir)
    validate_rewire_acceptance_report(acceptance_report)
    if not bool(acceptance_report["accepted"]):
        raise ValueError("selected candidate acceptance report is not accepted")
    if bool(acceptance_report["proposal_applied"]):
        raise ValueError("accepted candidate was already marked applied")
    candidate_eval_dir = Path(str(acceptance_report["candidate_eval"]["candidate_eval_output_dir"]))
    applied_eval_dir = out / APPLIED_CANDIDATE_EVAL_DIRNAME
    _copy_candidate_eval(candidate_eval_dir, applied_eval_dir, overwrite=overwrite)
    candidate_adjacency_name = str(acceptance_report["candidate_adjacency_name"])
    applied_adjacency = load_selected_adjacency(applied_eval_dir, adjacency_name=candidate_adjacency_name)
    applied_summary = validate_selected_adjacency(applied_adjacency)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted_candidate_apply_artifact_ok",
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "base_topology_mutated": False,
        "active_topology_mutated": False,
        "proposal_applied_to_base": False,
        "candidate_materialized": True,
        "promotion_eligible": False,
        "rewire_search_dir": str(search_dir),
        "output_dir": str(out),
        "applied_candidate_eval_dir": str(applied_eval_dir),
        "selected_candidate_index": int(selected["candidate_index"]),
        "selected_candidate_accepted": True,
        "selected_candidate_policy": selected.get("proposal_policy"),
        "selected_candidate_max_swaps": int(selected["max_swaps"]),
        "selected_candidate_kl_delta": float(selected["candidate_minus_base_kl_final"]),
        "base_eval_output_dir": acceptance_report["eval_output_dir"],
        "base_adjacency_name": acceptance_report["base_adjacency_name"],
        "candidate_adjacency_name": candidate_adjacency_name,
        "candidate_edge_count": int(acceptance_report["candidate_edge_count"]),
        "candidate_max_out_degree": int(acceptance_report["candidate_max_out_degree"]),
        "k": int(acceptance_report["k"]),
        "quality_ok": bool(acceptance_report["quality_ok"]),
        "base_training_ok": bool(acceptance_report["base_training_ok"]),
        "candidate_training_ok": bool(acceptance_report["candidate_training_ok"]),
        "safety_ok": bool(acceptance_report["safety_ok"]),
        "acceptance_decision": acceptance_report["decision"],
        "base_kl_final": float(acceptance_report["base_kl_final"]),
        "candidate_kl_final": float(acceptance_report["candidate_kl_final"]),
        "candidate_minus_base_kl_final": float(acceptance_report["candidate_minus_base_kl_final"]),
        "applied_adjacency_summary": applied_summary,
        "artifacts": {
            "accepted_candidate_manifest": ACCEPTED_CANDIDATE_MANIFEST_FILENAME,
            "applied_candidate_eval": APPLIED_CANDIDATE_EVAL_DIRNAME,
        },
        "note": "v26 P5 materializes an accepted candidate eval handoff without mutating the base active topology",
    }
    validate_accepted_candidate_manifest(manifest)
    return manifest


def validate_accepted_candidate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad apply schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "accepted_candidate_apply_artifact_ok":
        raise ValueError(f"bad apply status={manifest.get('status')!r}")
    for key, expected in {
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "base_topology_mutated": False,
        "active_topology_mutated": False,
        "proposal_applied_to_base": False,
        "candidate_materialized": True,
        "promotion_eligible": False,
        "selected_candidate_accepted": True,
        "quality_ok": True,
        "base_training_ok": True,
        "candidate_training_ok": True,
        "safety_ok": True,
    }.items():
        if bool(manifest.get(key)) is not expected:
            raise ValueError(f"accepted candidate manifest {key} must be {expected}")
    if int(manifest.get("candidate_edge_count", -1)) > int(manifest.get("k", -2)) * max(1, int(manifest.get("applied_adjacency_summary", {}).get("node_count", 1))):
        raise ValueError("candidate edge count failed coarse boundedness check")
    if int(manifest.get("candidate_max_out_degree", -1)) > int(manifest.get("k", -2)):
        raise ValueError("candidate max_out_degree must not exceed k")
    if manifest.get("acceptance_decision") != "accepted_pending_apply":
        raise ValueError("acceptance_decision must be accepted_pending_apply")
    if float(manifest.get("candidate_minus_base_kl_final", 1.0)) > 0.0:
        raise ValueError("accepted candidate must not regress KL under this conservative gate")
    applied_dir = Path(str(manifest.get("applied_candidate_eval_dir")))
    if not applied_dir.exists():
        raise ValueError("applied candidate eval dir must exist")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted_candidate_manifest_valid",
        "selected_candidate_index": int(manifest["selected_candidate_index"]),
        "candidate_adjacency_name": manifest["candidate_adjacency_name"],
        "candidate_materialized": True,
    }


def write_accepted_candidate_manifest(manifest: dict[str, Any], output_dir: str | Path) -> None:
    validate_accepted_candidate_manifest(manifest)
    _write_json(Path(output_dir) / ACCEPTED_CANDIDATE_MANIFEST_FILENAME, manifest)


def load_accepted_candidate_manifest(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / ACCEPTED_CANDIDATE_MANIFEST_FILENAME)


def run_and_write_accepted_candidate_manifest(
    *,
    rewire_search_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest = build_accepted_candidate_manifest(
        rewire_search_dir=rewire_search_dir,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    write_accepted_candidate_manifest(manifest, output_dir)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize a v26 accepted rewiring candidate as an isolated eval artifact.")
    parser.add_argument("--rewire-search-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_and_write_accepted_candidate_manifest(
            rewire_search_dir=args.rewire_search_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "accepted_candidate_manifest": str(Path(args.output_dir) / ACCEPTED_CANDIDATE_MANIFEST_FILENAME),
        "applied_candidate_eval_dir": manifest["applied_candidate_eval_dir"],
        "selected_candidate_index": manifest["selected_candidate_index"],
        "selected_candidate_policy": manifest["selected_candidate_policy"],
        "selected_candidate_kl_delta": manifest["selected_candidate_kl_delta"],
        "candidate_adjacency_name": manifest["candidate_adjacency_name"],
        "candidate_materialized": manifest["candidate_materialized"],
        "proposal_applied_to_base": manifest["proposal_applied_to_base"],
        "base_topology_mutated": manifest["base_topology_mutated"],
        "active_topology_mutated": manifest["active_topology_mutated"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
