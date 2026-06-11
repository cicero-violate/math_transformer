from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "learned_topology_locked_protocol.v1"
DEFAULT_PROTOCOL = "configs/learned_topology_locked_protocol.json"


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def protocol_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def validate_protocol(data: dict[str, Any]) -> None:
    schema = data.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported protocol schema_version={schema!r}; expected {SCHEMA_VERSION!r}")
    for section in ("paths", "quality", "benchmark", "gates"):
        if not isinstance(data.get(section), dict):
            raise ValueError(f"protocol missing object section: {section}")
    paths = data["paths"]
    for key in ("scorer", "champion_metadata", "checkpoint", "examples", "tmp_dir", "artifact_json", "artifact_jsonl"):
        if not paths.get(key):
            raise ValueError(f"protocol paths.{key} is required")
    benchmark = data["benchmark"]
    for key in ("hand_k", "learned_k", "bench_n", "bench_steps", "bench_seed", "acceptance_tolerance_ms"):
        if key not in benchmark:
            raise ValueError(f"protocol benchmark.{key} is required")
    quality = data["quality"]
    for key in ("learned_k", "quality_k", "device"):
        if key not in quality:
            raise ValueError(f"protocol quality.{key} is required")
    gates = data["gates"]
    for key in ("require_quality_ok", "require_speed_ok", "require_strict_speed_ok", "route_min_delta", "generic_min_delta"):
        if key not in gates:
            raise ValueError(f"protocol gates.{key} is required")


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    validate_protocol(data)
    data = dict(data)
    base = {k: v for k, v in data.items() if k not in {"protocol_path", "protocol_hash"}}
    data["protocol_path"] = str(p)
    data["protocol_hash"] = protocol_hash(base)
    return data


def shell_env(data: dict[str, Any]) -> dict[str, str]:
    paths = data["paths"]
    quality = data["quality"]
    benchmark = data["benchmark"]
    gates = data["gates"]
    env = {}
    env["LOCKED_PROTOCOL_NAME"] = str(data.get("name", ""))
    env["LOCKED_PROTOCOL_CONFIG"] = str(data.get("protocol_path", DEFAULT_PROTOCOL))
    env["LOCKED_PROTOCOL_HASH"] = str(data.get("protocol_hash", protocol_hash(data)))
    env["SCORER"] = str(paths["scorer"])
    env["CHAMPION_CHECKPOINT"] = str(paths["scorer"])
    env["CHAMPION_METADATA"] = str(paths["champion_metadata"])
    env["CHECKPOINT"] = str(paths["checkpoint"])
    env["DENSE_CHECKPOINT"] = str(paths["checkpoint"])
    env["EXAMPLES"] = str(paths["examples"])
    env["TMP_DIR"] = str(paths["tmp_dir"])
    env["TRACE_OUTPUT"] = str(paths.get("quality_trace", ""))
    env["ARTIFACT_JSON"] = str(paths["artifact_json"])
    env["ARTIFACT_JSONL"] = str(paths["artifact_jsonl"])
    env["LEARNED_K"] = str(benchmark["learned_k"])
    env["HAND_K"] = str(benchmark["hand_k"])
    env["QUALITY_K"] = str(quality["quality_k"])
    env["DEVICE"] = str(benchmark.get("device", quality.get("device", "auto")))
    env["BENCH_N"] = str(benchmark["bench_n"])
    env["BENCH_STEPS"] = str(benchmark["bench_steps"])
    env["BENCH_NODE_MODE"] = str(benchmark.get("bench_node_mode", "trees"))
    env["BENCH_SEED"] = str(benchmark["bench_seed"])
    env["ACCEPTANCE_TOL_MS"] = str(benchmark["acceptance_tolerance_ms"])
    env["ALLOW_FAIL"] = str(benchmark.get("allow_fail", 0))
    env["BLOCK_TOPOLOGY"] = str(benchmark.get("block_topology", 0))
    env["NATIVE_BLOCK_SPARSE"] = str(benchmark.get("native_block_sparse", 0))
    env["FUSED_NORM_QKV"] = str(benchmark.get("fused_norm_qkv", 0))
    env["FUSED_ATTN_OUTPROJ"] = str(benchmark.get("fused_attn_outproj", 0))
    env["ROUTE_MIN_DELTA"] = str(gates["route_min_delta"])
    env["GENERIC_MIN_DELTA"] = str(gates["generic_min_delta"])
    return env


def emit_shell_exports(data: dict[str, Any]) -> str:
    lines = []
    for key, value in shell_env(data).items():
        lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or export the locked learned-topology evaluation protocol.")
    parser.add_argument("--config", default=DEFAULT_PROTOCOL)
    parser.add_argument("--emit-shell-env", action="store_true")
    parser.add_argument("--hash", action="store_true")
    args = parser.parse_args()
    data = load_protocol(args.config)
    if args.emit_shell_env:
        print(emit_shell_exports(data), end="")
    elif args.hash:
        print(data["protocol_hash"])
    else:
        print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
