from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.qwen_distillation_harness import run_fixed_topology_distillation_harness
from src.qwen_logit_distillation_targets import (
    validate_frozen_logit_distillation_targets,
    write_frozen_logit_distillation_targets,
)


SCHEMA_VERSION = "qwen_k2_kl_comparison.v1"
COMPARISON_REPORT_FILENAME = "k2_kl_comparison_report.json"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_k2_kl_comparison(
    qwen_eval_dir: str | Path,
    random_eval_dir: str | Path,
    output_dir: str | Path,
    *,
    k: int = 2,
    random_adjacency_name: str | None = None,
    vocab_size: int = 16,
    target_seeds: list[int] | None = None,
    feature_dim: int = 8,
    forward_steps: int = 1,
    train_steps: int = 20,
    lr: float = 0.1,
    projection_seed: int = 0,
    temperature: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seeds = target_seeds if target_seeds is not None else list(range(8))

    # Shared frozen logit targets — both runs see identical teacher distributions.
    targets_dir = out / "shared_logit_targets"
    write_frozen_logit_distillation_targets(
        targets_dir, vocab_size=vocab_size, seeds=seeds, temperature=temperature
    )
    validate_frozen_logit_distillation_targets(targets_dir)

    # A_qwen k=2
    qwen_report = run_fixed_topology_distillation_harness(
        qwen_eval_dir,
        out / "qwen",
        k=k,
        logit_targets_dir=targets_dir,
        vocab_size=vocab_size,
        target_seeds=seeds,
        feature_dim=feature_dim,
        forward_steps=forward_steps,
        train_steps=train_steps,
        lr=lr,
        projection_seed=projection_seed,
        temperature=temperature,
        device=device,
    )

    # A_random k=2 — use the matched-random adjacency from the comparison dir.
    random_adj_name = random_adjacency_name or f"qwen_topk_k{k}_random_seed0"
    random_report = run_fixed_topology_distillation_harness(
        random_eval_dir,
        out / "random",
        adjacency_name=random_adj_name,
        logit_targets_dir=targets_dir,
        vocab_size=vocab_size,
        target_seeds=seeds,
        feature_dim=feature_dim,
        forward_steps=forward_steps,
        train_steps=train_steps,
        lr=lr,
        projection_seed=projection_seed,
        temperature=temperature,
        device=device,
    )

    kl_before_qwen = float(qwen_report["kl_before"])
    kl_after_qwen = float(qwen_report["kl_after"])
    kl_before_random = float(random_report["kl_before"])
    kl_after_random = float(random_report["kl_after"])

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "k2_kl_comparison_ok",
        "candidate_topology": f"qwen_topk_k{k}",
        "k": k,
        "train_steps": train_steps,
        "vocab_size": vocab_size,
        "feature_dim": feature_dim,
        "forward_steps": forward_steps,
        "temperature": temperature,
        "device": device,
        # Before training
        "kl_before_qwen": kl_before_qwen,
        "kl_before_random": kl_before_random,
        "delta_before": kl_before_qwen - kl_before_random,
        "qwen_wins_before": kl_before_qwen < kl_before_random,
        # After training
        "kl_after_qwen": kl_after_qwen,
        "kl_after_random": kl_after_random,
        "delta_after": kl_after_qwen - kl_after_random,
        "qwen_wins_after": kl_after_qwen < kl_after_random,
        # Training reduction
        "kl_reduction_qwen": kl_before_qwen - kl_after_qwen,
        "kl_reduction_random": kl_before_random - kl_after_random,
        # Safety flags
        "kl_decreased_qwen": bool(qwen_report["kl_decreased"]),
        "kl_decreased_random": bool(random_report["kl_decreased"]),
        "finite": (
            math.isfinite(kl_before_qwen)
            and math.isfinite(kl_after_qwen)
            and math.isfinite(kl_before_random)
            and math.isfinite(kl_after_random)
        ),
        "note": (
            "lower KL is better; qwen_wins_after=true means the Qwen k=2 topology "
            "achieves lower final KL than matched random after identical training"
        ),
    }
    _write_json(out / COMPARISON_REPORT_FILENAME, report)
    return report
