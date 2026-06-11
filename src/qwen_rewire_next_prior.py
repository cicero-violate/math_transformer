from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.qwen_rewire_candidate_promotion import validate_candidate_next_prior_manifest
from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency


SCHEMA_VERSION = "qwen_rewire_next_prior.v1"
NEXT_SPARSE_PRIOR_MANIFEST_FILENAME = "next_sparse_prior_manifest.json"
NEXT_SPARSE_PRIOR_EVAL_DIRNAME = "next_sparse_prior_eval"
NEXT_SPARSE_PRIOR_ADJACENCY_SUMMARY_FILENAME = "next_sparse_prior_adjacency_summary.json"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def load_candidate_next_prior_manifest_path(path: str | Path) -> dict[str, Any]:
    return _read_json(Path(path))


def validate_promoted_candidate_next_prior_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != "qwen_rewire_candidate_promotion.v1":
        raise ValueError(f"bad candidate promotion schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "candidate_next_prior_manifest_ok":
        raise ValueError(f"bad candidate promotion status={manifest.get('status')!r}")
    if not bool(manifest.get("candidate_promoted")):
        raise ValueError("candidate_promoted must be true")
    if manifest.get("decision") != "candidate_promoted_as_next_prior":
        raise ValueError("decision must be candidate_promoted_as_next_prior")
    for key in ("quality_ok", "runtime_ok", "memory_ok", "safety_ok"):
        if not bool(manifest.get(key)):
            raise ValueError(f"{key} must be true")
    for key in ("base_topology_mutated", "active_topology_mutated", "proposal_applied_to_base"):
        if bool(manifest.get(key)):
            raise ValueError(f"{key} must be false")
    for key in ("bounded_active_adjacency", "candidate_materialized"):
        if not bool(manifest.get(key)):
            raise ValueError(f"{key} must be true")
    validation = validate_candidate_next_prior_manifest(manifest)
    return validation


def _copy_next_prior_eval(source_eval_dir: Path, destination_eval_dir: Path, *, overwrite: bool) -> None:
    if not source_eval_dir.exists():
        raise FileNotFoundError(str(source_eval_dir))
    if destination_eval_dir.exists():
        if not overwrite:
            raise ValueError(f"next sparse prior eval dir already exists: {destination_eval_dir}")
        shutil.rmtree(destination_eval_dir)
    shutil.copytree(source_eval_dir, destination_eval_dir)


def build_next_sparse_prior_manifest(
    *,
    candidate_next_prior_manifest: dict[str, Any],
    candidate_next_prior_manifest_path: str | Path,
    output_dir: str | Path,
    next_sparse_prior_eval_dir: str | Path,
    adjacency_summary: dict[str, Any],
) -> dict[str, Any]:
    validate_promoted_candidate_next_prior_manifest(candidate_next_prior_manifest)
    if not bool(adjacency_summary.get("bounded")):
        raise ValueError("copied next sparse prior adjacency must be bounded")
    out = Path(output_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "next_sparse_prior_manifest_ok",
        "source_candidate_next_prior_manifest": str(Path(candidate_next_prior_manifest_path)),
        "source_applied_candidate_eval_dir": str(Path(str(candidate_next_prior_manifest["applied_candidate_eval_dir"]))),
        "next_sparse_prior_eval_dir": str(Path(next_sparse_prior_eval_dir)),
        "next_sparse_prior_adjacency_name": str(candidate_next_prior_manifest["candidate_adjacency_name"]),
        "selected_candidate_index": int(candidate_next_prior_manifest["selected_candidate_index"]),
        "selected_candidate_policy": candidate_next_prior_manifest["selected_candidate_policy"],
        "selected_candidate_kl_delta": float(candidate_next_prior_manifest["selected_candidate_kl_delta"]),
        "candidate_promoted": True,
        "promotion_decision": "candidate_promoted_as_next_prior",
        "bounded_active_adjacency": True,
        "base_topology_mutated": False,
        "active_topology_mutated": False,
        "proposal_applied_to_base": False,
        "original_base_topology_mutated": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "recursive_seed_ready": True,
        "promotion_eligible": True,
        "next_cycle_input_ready": True,
        "next_sparse_prior_adjacency_summary": adjacency_summary,
        "artifacts": {
            "next_sparse_prior_manifest": NEXT_SPARSE_PRIOR_MANIFEST_FILENAME,
            "next_sparse_prior_eval": NEXT_SPARSE_PRIOR_EVAL_DIRNAME,
            "next_sparse_prior_adjacency_summary": NEXT_SPARSE_PRIOR_ADJACENCY_SUMMARY_FILENAME,
        },
        "output_dir": str(out),
        "note": "Isolated next sparse prior handoff for the next rewiring cycle; original base topology and P6 artifacts are not mutated.",
    }
    validate_next_sparse_prior_manifest(manifest)
    return manifest


def validate_next_sparse_prior_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad next prior schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "next_sparse_prior_manifest_ok":
        raise ValueError(f"bad next prior status={manifest.get('status')!r}")
    for key, expected in {
        "candidate_promoted": True,
        "bounded_active_adjacency": True,
        "base_topology_mutated": False,
        "active_topology_mutated": False,
        "proposal_applied_to_base": False,
        "original_base_topology_mutated": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "recursive_seed_ready": True,
        "promotion_eligible": True,
        "next_cycle_input_ready": True,
    }.items():
        if bool(manifest.get(key)) is not expected:
            raise ValueError(f"next sparse prior manifest {key} must be {expected}")
    if manifest.get("promotion_decision") != "candidate_promoted_as_next_prior":
        raise ValueError("promotion_decision must be candidate_promoted_as_next_prior")
    for key in (
        "source_candidate_next_prior_manifest",
        "source_applied_candidate_eval_dir",
        "next_sparse_prior_eval_dir",
    ):
        path = Path(str(manifest.get(key)))
        if not path.exists():
            raise ValueError(f"next sparse prior manifest path missing: {key} -> {path}")
    summary = manifest.get("next_sparse_prior_adjacency_summary")
    if not isinstance(summary, dict) or not bool(summary.get("bounded")):
        raise ValueError("next sparse prior adjacency summary must be bounded")
    if summary.get("adjacency_name") != manifest.get("next_sparse_prior_adjacency_name"):
        raise ValueError("next sparse prior adjacency summary/name mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("next sparse prior artifacts must be an object")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "next_sparse_prior_manifest_valid",
        "next_sparse_prior_adjacency_name": manifest["next_sparse_prior_adjacency_name"],
        "recursive_seed_ready": True,
        "next_cycle_input_ready": True,
    }


def write_next_sparse_prior_manifest(manifest: dict[str, Any], output_dir: str | Path) -> None:
    validate_next_sparse_prior_manifest(manifest)
    _write_json(Path(output_dir) / NEXT_SPARSE_PRIOR_MANIFEST_FILENAME, manifest)


def load_next_sparse_prior_manifest(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / NEXT_SPARSE_PRIOR_MANIFEST_FILENAME)


def run_and_write_next_sparse_prior_handoff(
    *,
    candidate_next_prior_manifest: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_manifest_path = Path(candidate_next_prior_manifest)
    source_manifest = load_candidate_next_prior_manifest_path(source_manifest_path)
    validate_promoted_candidate_next_prior_manifest(source_manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_eval_dir = Path(str(source_manifest["applied_candidate_eval_dir"]))
    next_eval_dir = out / NEXT_SPARSE_PRIOR_EVAL_DIRNAME
    _copy_next_prior_eval(source_eval_dir, next_eval_dir, overwrite=overwrite)

    adjacency_name = str(source_manifest["candidate_adjacency_name"])
    adjacency = load_selected_adjacency(next_eval_dir, adjacency_name=adjacency_name)
    adjacency_summary = validate_selected_adjacency(adjacency)
    _write_json(out / NEXT_SPARSE_PRIOR_ADJACENCY_SUMMARY_FILENAME, adjacency_summary)

    manifest = build_next_sparse_prior_manifest(
        candidate_next_prior_manifest=source_manifest,
        candidate_next_prior_manifest_path=source_manifest_path,
        output_dir=out,
        next_sparse_prior_eval_dir=next_eval_dir,
        adjacency_summary=adjacency_summary,
    )
    write_next_sparse_prior_manifest(manifest, out)
    return {
        "status": "next_sparse_prior_handoff_ok",
        "next_sparse_prior_manifest": str(out / NEXT_SPARSE_PRIOR_MANIFEST_FILENAME),
        "next_sparse_prior_eval_dir": str(next_eval_dir),
        "next_sparse_prior_adjacency_summary": str(out / NEXT_SPARSE_PRIOR_ADJACENCY_SUMMARY_FILENAME),
        "next_sparse_prior_adjacency_name": manifest["next_sparse_prior_adjacency_name"],
        "recursive_seed_ready": manifest["recursive_seed_ready"],
        "next_cycle_input_ready": manifest["next_cycle_input_ready"],
        "bounded_active_adjacency": manifest["bounded_active_adjacency"],
        "base_topology_mutated": manifest["base_topology_mutated"],
        "active_topology_mutated": manifest["active_topology_mutated"],
        "proposal_applied_to_base": manifest["proposal_applied_to_base"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an isolated next sparse prior handoff from a promoted v26 candidate.")
    parser.add_argument("--candidate-next-prior-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_and_write_next_sparse_prior_handoff(
            candidate_next_prior_manifest=args.candidate_next_prior_manifest,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
