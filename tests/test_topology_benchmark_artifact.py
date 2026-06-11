from __future__ import annotations

from src.topology_benchmark_artifact import build_benchmark_artifact, parse_quality_log


def test_parse_quality_log_with_expert_breakdown():
    text = """
mode=topology_only  k=16  examples=259  route_acc=0.9800
         by_expert generic_expert=230/259(0.8880)
mode=learned_topology  k=8  examples=259  route_acc=0.9900  hidden_cos=0.970000  logit_kl=0.010000
         by_expert generic_expert=236/259(0.9112)
"""
    rows = parse_quality_log(text)
    assert rows[0]["by_expert"]["generic_expert"]["correct"] == 230
    assert rows[1]["hidden_cos"] == 0.97
    assert rows[1]["by_expert"]["generic_expert"]["accuracy"] == 0.9112


def test_build_benchmark_artifact_acceptance():
    quality = """
mode=full  k=full  examples=259  route_acc=0.9800
         by_expert generic_expert=225/259(0.8687) affine_expert=296/296(1.0000)
mode=topology_only  k=4  examples=259  route_acc=0.9927
         by_expert generic_expert=252/259(0.9730) affine_expert=290/296(0.9797)
mode=topology_only  k=16  examples=259  route_acc=0.9800
         by_expert generic_expert=230/259(0.8880)
mode=learned_topology  k=4  examples=259  route_acc=0.9933
         by_expert generic_expert=253/259(0.9768) affine_expert=290/296(0.9797)
mode=learned_topology  k=8  examples=259  route_acc=0.9900
         by_expert generic_expert=236/259(0.9112)
"""
    hand = {
        "n": 1024,
        "node_mode": "trees",
        "prepared_static_sparse_block_ms": 10.0,
        "prepared_static_sparse_attention_ms": 4.0,
        "selector_results": {"paired_prepared_shared_block": {"qkv_ms": 1.0}},
        "by_relation": {},
    }
    learned = dict(hand)
    learned["prepared_static_sparse_block_ms"] = 9.0
    learned["prepared_static_sparse_attention_ms"] = 3.5
    artifact = build_benchmark_artifact(
        quality_log_text=quality,
        hand_report=hand,
        learned_report=learned,
        hand_k=16,
        learned_k=8,
        acceptance_tolerance_ms=0.05,
    )
    assert artifact["acceptance"]["passed"] is True
    assert artifact["quality"]["learned"]["by_expert"]["generic_expert"]["correct"] == 236
    assert artifact["speed"]["speedup"] > 1.0
    report = artifact["promotion_report"]
    assert report["quality_policies"]["dense_full"]["route_acc"] == 0.98
    assert report["quality_policies"]["hand_k4"]["generic_expert"]["correct"] == 252
    assert report["quality_policies"]["learned_k4"]["affine_expert"]["total"] == 296
    assert report["quality_policies"]["current_learned_policy"]["k"] == 8
    assert report["required_quality_policies_present"]["hand_k4"] is True
