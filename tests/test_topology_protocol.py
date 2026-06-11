from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.topology_protocol import emit_shell_exports, load_protocol, protocol_hash, shell_env, validate_protocol


def test_locked_protocol_loads_and_exports_env():
    protocol = load_protocol("configs/learned_topology_locked_protocol.json")
    env = shell_env(protocol)

    assert protocol["schema_version"] == "learned_topology_locked_protocol.v1"
    assert len(protocol["protocol_hash"]) == 64
    assert env["SCORER"] == "runs/checkpoints/topology_scorer.champion.pt"
    assert env["HAND_K"] == "16"
    assert env["LEARNED_K"] == "8"
    assert env["BENCH_SEED"] == "0"
    assert env["LOCKED_PROTOCOL_HASH"] == protocol["protocol_hash"]

    exports = emit_shell_exports(protocol)
    assert "export SCORER=runs/checkpoints/topology_scorer.champion.pt" in exports
    assert f"export LOCKED_PROTOCOL_HASH={protocol['protocol_hash']}" in exports


def test_protocol_hash_is_canonical_order_independent():
    left = {"b": 2, "a": {"d": 4, "c": 3}}
    right = {"a": {"c": 3, "d": 4}, "b": 2}
    assert protocol_hash(left) == protocol_hash(right)


def test_validate_protocol_rejects_missing_required_sections(tmp_path: Path):
    bad = {"schema_version": "learned_topology_locked_protocol.v1", "paths": {}}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(ValueError, match="missing object section"):
        load_protocol(p)


def test_validate_protocol_rejects_wrong_schema():
    with pytest.raises(ValueError, match="unsupported protocol"):
        validate_protocol({"schema_version": "wrong"})