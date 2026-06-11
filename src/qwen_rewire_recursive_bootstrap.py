from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.qwen_edge_trace import run_and_write_edge_trace_report, validate_edge_trace_report
from src.qwen_rewire_next_prior import validate_next_sparse_prior_manifest
from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency


SCHEMA_VERSION = "qwen_rewire_recursive_bootstrap.v1"
RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME = "recursive_bootstrap_manifest.json"
CYCLE_SEED_ADJACENCY_SUMMARY_FILENAME_TEMPLATE = "cycle_{cycle:03d}_seed_adjacency_summary.json"
CYCLE_SEED_EVAL_DIRNAME_TEMPLATE = "cycle_{cycle:03d}_seed_eval"
CYCLE_EDGE_TRACE_DIRNAME_TEMPLATE = "cycle_{cycle:03d}_edge_trace"


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


def load_next_sparse_prior_manifest_path(path: str | Path) -> dict[str, Any]:
    return _read_json(Path(path))


def _cycle_dirname(template: str, cycle_index: int) -> str:
    return template.format(cycle=cycle_index)


def _validate_positive_int(value: int, *, name: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return number


def validate_recursive_seed_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != "qwen_rewire_next_prior.v1":
        raise ValueError(f"bad next prior schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "next_sparse_prior_manifest_ok":
        raise ValueError(f"bad next prior status={manifest.get('status')!r}")
    if not bool(manifest.get("recursive_seed_ready")):
        raise ValueError("recursive_seed_ready must be true")
    if not bool(manifest.get("next_cycle_input_ready")):
        raise ValueError("next_cycle_input_ready must be true")
    for key in ("bounded_active_adjacency",):
        if not bool(manifest.get(key)):
            raise ValueError(f"{key} must be true")
    for key in ("base_topology_mutated", "active_topology_mutated", "proposal_applied_to_base"):
        if bool(manifest.get(key)):
            raise ValueError(f"{key} must be false")
    for key in ("teacher_checkpoint_loaded", "teacher_inference_runtime_required", "raw_weight_payload_in_graph"):
        if bool(manifest.get(key)):
            raise ValueError(f"{key} must be false")
    return validate_next_sparse_prior_manifest(manifest)


def _copy_cycle_seed_eval(source_eval_dir: Path, destination_eval_dir: Path, *, overwrite: bool) -> None:
    if not source_eval_dir.exists():
        raise FileNotFoundError(str(source_eval_dir))
    if destination_eval_dir.exists():
        if not overwrite:
            raise ValueError(f"cycle seed eval dir already exists: {destination_eval_dir}")
        shutil.rmtree(destination_eval_dir)
    shutil.copytree(source_eval_dir, destination_eval_dir)


def build_recursive_bootstrap_manifest(
    *,
    next_sparse_prior_manifest: dict[str, Any],
    next_sparse_prior_manifest_path: str | Path,
    output_dir: str | Path,
    cycle_index: int,
    cycle_seed_eval_dir: str | Path,
    cycle_edge_trace_dir: str | Path,
    seed_adjacency_summary: dict[str, Any],
    edge_trace_report: dict[str, Any],
) -> dict[str, Any]:
    cycle_index = _validate_positive_int(cycle_index, name="cycle_index")
    validate_recursive_seed_manifest(next_sparse_prior_manifest)
    edge_trace_validation = validate_edge_trace_report(edge_trace_report)
    if edge_trace_validation["adjacency_name"] != next_sparse_prior_manifest["next_sparse_prior_adjacency_name"]:
        raise ValueError("edge trace adjacency/name mismatch")
    if not bool(seed_adjacency_summary.get("bounded")):
        raise ValueError("cycle seed adjacency must be bounded")
    out = Path(output_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "recursive_bootstrap_manifest_ok",
        "cycle_index": cycle_index,
        "source_next_sparse_prior_manifest": str(Path(next_sparse_prior_manifest_path)),
        "source_next_sparse_prior_eval_dir": str(Path(str(next_sparse_prior_manifest["next_sparse_prior_eval_dir"]))),
        "cycle_seed_eval_dir": str(Path(cycle_seed_eval_dir)),
        "cycle_edge_trace_dir": str(Path(cycle_edge_trace_dir)),
        "cycle_seed_adjacency_name": str(next_sparse_prior_manifest["next_sparse_prior_adjacency_name"]),
        "selected_candidate_index": int(next_sparse_prior_manifest["selected_candidate_index"]),
        "selected_candidate_policy": next_sparse_prior_manifest["selected_candidate_policy"],
        "selected_candidate_kl_delta": float(next_sparse_prior_manifest["selected_candidate_kl_delta"]),
        "recursive_seed_ready": True,
        "next_cycle_input_ready": True,
        "edge_trace_ready": True,
        "bounded_active_adjacency": True,
        "base_topology_mutated": False,
        "active_topology_mutated": False,
        "proposal_applied_to_base": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "cycle_seed_adjacency_summary": seed_adjacency_summary,
        "edge_trace_summary": {
            "edge_count": int(edge_trace_report["edge_count"]),
            "row_count": int(edge_trace_report["row_count"]),
            "adjacency_name": edge_trace_report["adjacency_name"],
            "topology_mutated": False,
        },
        "artifacts": {
            "recursive_bootstrap_manifest": RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME,
            "cycle_001_seed_eval": _cycle_dirname(CYCLE_SEED_EVAL_DIRNAME_TEMPLATE, cycle_index),
            "cycle_001_edge_trace": _cycle_dirname(CYCLE_EDGE_TRACE_DIRNAME_TEMPLATE, cycle_index),
            "cycle_001_seed_adjacency_summary": _cycle_dirname(CYCLE_SEED_ADJACENCY_SUMMARY_FILENAME_TEMPLATE, cycle_index),
        },
        "output_dir": str(out),
        "note": "Bootstraps the next recursive rewiring cycle from an isolated promoted sparse prior copy; no prior artifacts are mutated.",
    }
    validate_recursive_bootstrap_manifest(manifest)
    return manifest


def validate_recursive_bootstrap_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bad recursive bootstrap schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "recursive_bootstrap_manifest_ok":
        raise ValueError(f"bad recursive bootstrap status={manifest.get('status')!r}")
    for key, expected in {
        "recursive_seed_ready": True,
        "next_cycle_input_ready": True,
        "edge_trace_ready": True,
        "bounded_active_adjacency": True,
        "base_topology_mutated": False,
        "active_topology_mutated": False,
        "proposal_applied_to_base": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
    }.items():
        if bool(manifest.get(key)) is not expected:
            raise ValueError(f"recursive bootstrap manifest {key} must be {expected}")
    if int(manifest.get("cycle_index", 0)) < 1:
        raise ValueError("cycle_index must be >= 1")
    for key in (
        "source_next_sparse_prior_manifest",
        "source_next_sparse_prior_eval_dir",
        "cycle_seed_eval_dir",
        "cycle_edge_trace_dir",
    ):
        path = Path(str(manifest.get(key)))
        if not path.exists():
            raise ValueError(f"recursive bootstrap manifest path missing: {key} -> {path}")
    summary = manifest.get("cycle_seed_adjacency_summary")
    if not isinstance(summary, dict) or not bool(summary.get("bounded")):
        raise ValueError("cycle seed adjacency summary must be bounded")
    if summary.get("adjacency_name") != manifest.get("cycle_seed_adjacency_name"):
        raise ValueError("cycle seed adjacency summary/name mismatch")
    edge_summary = manifest.get("edge_trace_summary")
    if not isinstance(edge_summary, dict) or bool(edge_summary.get("topology_mutated", True)):
        raise ValueError("edge trace summary must be present and non-mutating")
    if edge_summary.get("adjacency_name") != manifest.get("cycle_seed_adjacency_name"):
        raise ValueError("edge trace summary/name mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("recursive bootstrap artifacts must be an object")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "recursive_bootstrap_manifest_valid",
        "cycle_index": int(manifest["cycle_index"]),
        "cycle_seed_adjacency_name": manifest["cycle_seed_adjacency_name"],
        "recursive_seed_ready": True,
        "next_cycle_input_ready": True,
        "edge_trace_ready": True,
    }


def write_recursive_bootstrap_manifest(manifest: dict[str, Any], output_dir: str | Path) -> None:
    validate_recursive_bootstrap_manifest(manifest)
    _write_json(Path(output_dir) / RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME, manifest)


def load_recursive_bootstrap_manifest(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME)


def run_and_write_recursive_bootstrap_handoff(
    *,
    next_sparse_prior_manifest: str | Path,
    output_dir: str | Path,
    cycle_index: int = 1,
    k: int | None = None,
    feature_dim: int = 8,
    steps: int = 1,
    seeds: list[int] | None = None,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, Any]:
    cycle_index = _validate_positive_int(cycle_index, name="cycle_index")
    feature_dim = _validate_positive_int(feature_dim, name="feature_dim")
    steps = _validate_positive_int(steps, name="steps")
    if k is not None:
        k = _validate_positive_int(k, name="k")
    if seeds is not None and not list(seeds):
        raise ValueError("seeds must contain at least one seed")

    source_manifest_path = Path(next_sparse_prior_manifest)
    source_manifest = load_next_sparse_prior_manifest_path(source_manifest_path)
    validate_recursive_seed_manifest(source_manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_eval_dir = Path(str(source_manifest["next_sparse_prior_eval_dir"]))
    cycle_seed_eval_dir = out / _cycle_dirname(CYCLE_SEED_EVAL_DIRNAME_TEMPLATE, cycle_index)
    _copy_cycle_seed_eval(source_eval_dir, cycle_seed_eval_dir, overwrite=overwrite)

    adjacency_name = str(source_manifest["next_sparse_prior_adjacency_name"])
    adjacency = load_selected_adjacency(cycle_seed_eval_dir, adjacency_name=adjacency_name)
    adjacency_summary = validate_selected_adjacency(adjacency)
    if k is not None and int(adjacency_summary["k"]) != k:
        raise ValueError(f"k must match cycle seed adjacency k={adjacency_summary['k']}, got {k}")
    seed_summary_path = out / _cycle_dirname(CYCLE_SEED_ADJACENCY_SUMMARY_FILENAME_TEMPLATE, cycle_index)
    _write_json(seed_summary_path, adjacency_summary)

    cycle_edge_trace_dir = out / _cycle_dirname(CYCLE_EDGE_TRACE_DIRNAME_TEMPLATE, cycle_index)
    if cycle_edge_trace_dir.exists():
        if not overwrite:
            raise ValueError(f"cycle edge trace dir already exists: {cycle_edge_trace_dir}")
        shutil.rmtree(cycle_edge_trace_dir)
    edge_trace_report = run_and_write_edge_trace_report(
        cycle_seed_eval_dir,
        cycle_edge_trace_dir,
        adjacency_name=adjacency_name,
        feature_dim=feature_dim,
        steps=steps,
        seeds=seeds,
        device=device,
    )

    manifest = build_recursive_bootstrap_manifest(
        next_sparse_prior_manifest=source_manifest,
        next_sparse_prior_manifest_path=source_manifest_path,
        output_dir=out,
        cycle_index=cycle_index,
        cycle_seed_eval_dir=cycle_seed_eval_dir,
        cycle_edge_trace_dir=cycle_edge_trace_dir,
        seed_adjacency_summary=adjacency_summary,
        edge_trace_report=edge_trace_report,
    )
    write_recursive_bootstrap_manifest(manifest, out)
    return {
        "status": "recursive_bootstrap_handoff_ok",
        "recursive_bootstrap_manifest": str(out / RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME),
        "cycle_seed_eval_dir": str(cycle_seed_eval_dir),
        "cycle_edge_trace_dir": str(cycle_edge_trace_dir),
        "cycle_seed_adjacency_summary": str(seed_summary_path),
        "cycle_seed_adjacency_name": manifest["cycle_seed_adjacency_name"],
        "recursive_seed_ready": manifest["recursive_seed_ready"],
        "next_cycle_input_ready": manifest["next_cycle_input_ready"],
        "edge_trace_ready": manifest["edge_trace_ready"],
        "bounded_active_adjacency": manifest["bounded_active_adjacency"],
        "base_topology_mutated": manifest["base_topology_mutated"],
        "active_topology_mutated": manifest["active_topology_mutated"],
        "proposal_applied_to_base": manifest["proposal_applied_to_base"],
    }


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("seeds must contain at least one integer")
    return values


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap a recursive rewiring cycle from a v26 next sparse prior.")
    parser.add_argument("--next-sparse-prior-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cycle-index", type=_positive_int, default=1)
    parser.add_argument("--k", type=_positive_int, default=None)
    parser.add_argument("--feature-dim", type=_positive_int, default=8)
    parser.add_argument("--steps", type=_positive_int, default=1)
    parser.add_argument("--seeds", type=_parse_int_list, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu", choices=["cpu", "torch_cpu", "cuda", "auto"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_and_write_recursive_bootstrap_handoff(
            next_sparse_prior_manifest=args.next_sparse_prior_manifest,
            output_dir=args.output_dir,
            cycle_index=args.cycle_index,
            k=args.k,
            feature_dim=args.feature_dim,
            steps=args.steps,
            seeds=args.seeds,
            device=args.device,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
