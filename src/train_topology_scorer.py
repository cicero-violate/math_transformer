from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .eval import _load_route_eval_records, run_quality_eval
from .embedder import MathEmbedder
from .model import MathRoutedTransformer
from .learned_topology import (
    FEATURE_NAMES,
    LearnedTopologyScorer,
    build_edge_feature_tensor,
    topk_mask_from_scores,
)
from .normalize import normalize
from .parser import parse
from .topology import TopologyBuilder, build_hand_score_matrix


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return dev


def _edge_metrics(scores: torch.Tensor, target: torch.Tensor, eval_k: int) -> dict[str, float]:
    with torch.no_grad():
        pred = topk_mask_from_scores(scores, eval_k)
        target_bool = target.bool()
        hits = (pred & target_bool).sum(dim=1).float()
        denom = target_bool.sum(dim=1).clamp(min=1).float()
        recall = (hits / denom).mean().item()
        precision = (hits / pred.sum(dim=1).clamp(min=1).float()).mean().item()
        diag = torch.diag(pred).float().mean().item() if pred.numel() else 1.0
        return {
            "edge_recall": float(recall),
            "edge_precision": float(precision),
            "self_loop_rate": float(diag),
        }






def _aggregate_metrics(
    scorer: LearnedTopologyScorer,
    cached: list[tuple],
    eval_k: int,
) -> dict[str, float]:
    if not cached:
        return {
            "mean_row_recall": 0.0,
            "mean_row_precision": 0.0,
            "micro_recall": 0.0,
            "micro_precision": 0.0,
            "self_loop_rate": 0.0,
            "row_cap_violations": 0.0,
        }
    hits_total = 0.0
    pred_total = 0.0
    target_total = 0.0
    row_recall_sum = 0.0
    row_precision_sum = 0.0
    row_count = 0
    self_hits = 0
    self_total = 0
    row_cap_violations = 0
    scorer.eval()
    with torch.no_grad():
        for item in cached:
            features, target_f, _n_nodes, _teacher_score = item[:4]
            target = target_f.bool()
            scores = scorer(features)
            pred = topk_mask_from_scores(scores, eval_k)
            hits = (pred & target).sum(dim=1).float()
            pred_rows = pred.sum(dim=1).clamp(min=1).float()
            target_rows = target.sum(dim=1).clamp(min=1).float()
            row_recall_sum += float((hits / target_rows).sum().item())
            row_precision_sum += float((hits / pred_rows).sum().item())
            row_count += int(pred.shape[0])
            hits_total += float(hits.sum().item())
            pred_total += float(pred.sum().item())
            target_total += float(target.sum().item())
            if pred.numel():
                self_hits += int(torch.diag(pred).sum().item())
                self_total += int(pred.shape[0])
                row_cap_violations += int((pred.sum(dim=1) > eval_k).sum().item())
    scorer.train()
    return {
        "mean_row_recall": row_recall_sum / max(row_count, 1),
        "mean_row_precision": row_precision_sum / max(row_count, 1),
        "micro_recall": hits_total / max(target_total, 1.0),
        "micro_precision": hits_total / max(pred_total, 1.0),
        "self_loop_rate": self_hits / max(self_total, 1),
        "row_cap_violations": float(row_cap_violations),
    }


def _save_scorer_checkpoint(
    path: str | Path,
    scorer: LearnedTopologyScorer,
    *,
    hidden_dim: int,
    target_k: int,
    eval_k: int,
    local_window: int,
    middle_bridge_width: int,
    topology_mode: str,
    examples_path: str,
    best_metric: float | None = None,
    teacher_signal: str = "hand_score_topk",
    dense_checkpoint: str | None = None,
    dense_mix: float = 0.0,
    resume_scorer_checkpoint: str | None = None,
    replay_candidates_path: str | None = None,
    replay_weight_scale: float = 0.0,
    replay_max_weight: float = 1.0,
    replay_weighted_examples: int = 0,
    replay_appended_examples: int = 0,
    replay_sample_ratio: float = 0.0,
    replay_sampled_steps: int = 0,
    best_selection: str = "edge_recall",
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": scorer.state_dict(),
        "feature_names": FEATURE_NAMES,
        "hidden_dim": hidden_dim,
        "target_k": target_k,
        "eval_k": eval_k,
        "trained_target_k": eval_k,
        "teacher_signal": teacher_signal,
        "dense_checkpoint": dense_checkpoint or "",
        "dense_mix": float(dense_mix),
        "resume_scorer_checkpoint": resume_scorer_checkpoint or "",
        "replay_candidates_path": replay_candidates_path or "",
        "replay_weight_scale": float(replay_weight_scale),
        "replay_max_weight": float(replay_max_weight),
        "replay_weighted_examples": int(replay_weighted_examples),
        "replay_appended_examples": int(replay_appended_examples),
        "replay_sample_ratio": float(replay_sample_ratio),
        "replay_sampled_steps": int(replay_sampled_steps),
        "best_selection": best_selection,
        "local_window": local_window,
        "middle_bridge_width": middle_bridge_width,
        "topology_mode": topology_mode,
        "examples_path": examples_path,
    }
    if best_metric is not None:
        payload["best_mean_row_recall"] = float(best_metric)
    torch.save(payload, out)

def _compress_target(target: torch.Tensor, keep_k: int) -> torch.Tensor:
    """Reduce a target mask to at most keep_k positives per row."""
    if keep_k <= 0:
        raise ValueError("keep_k must be positive")
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError(f"target must be square, got {tuple(target.shape)}")
    n = target.shape[0]
    if n == 0:
        return target.bool()
    k = min(keep_k, n)
    work = target.float().clone()
    idx = torch.arange(n, device=target.device)
    work[idx, idx] = 2.0
    top_idx = work.topk(k, dim=1).indices
    compressed = torch.zeros_like(target, dtype=torch.bool)
    rows = idx.unsqueeze(1).expand_as(top_idx)
    compressed[rows.reshape(-1), top_idx.reshape(-1)] = True
    compressed.fill_diagonal_(True)
    return compressed


def _topk_mask_from_teacher_scores(score: torch.Tensor, keep_k: int) -> torch.Tensor:
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError(f"score must be square, got {tuple(score.shape)}")
    n = score.shape[0]
    if n == 0:
        return torch.zeros_like(score, dtype=torch.bool)
    k = min(keep_k, n)
    work = score.clone()
    idx = torch.arange(n, device=score.device)
    work[idx, idx] = torch.finfo(work.dtype).max / 4
    top_idx = work.topk(k, dim=1).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    rows = idx.unsqueeze(1).expand_as(top_idx)
    mask[rows.reshape(-1), top_idx.reshape(-1)] = True
    mask.fill_diagonal_(True)
    return mask


def _normalize_teacher_scores(score: torch.Tensor) -> torch.Tensor:
    if score.numel() == 0:
        return score
    max_val = score.max().clamp(min=1e-6)
    return score / max_val


def _pairwise_rank_loss(pred: torch.Tensor, target_mask: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for row_pred, row_target in zip(pred, target_mask):
        pos = row_pred[row_target.bool()]
        neg = row_pred[~row_target.bool()]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        hardest_pos = pos.min()
        hardest_neg = neg.max()
        losses.append(torch.relu(torch.as_tensor(margin, device=pred.device, dtype=pred.dtype) - hardest_pos + hardest_neg))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def _load_dense_teacher_model(
    checkpoint: str,
    device: torch.device,
    *,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
) -> MathRoutedTransformer:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = MathRoutedTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.0,
        attention_mode="full",
        share_topology_cache=False,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _dense_qk_affinity_score(
    model: MathRoutedTransformer,
    nodes,
    env: dict | None,
) -> torch.Tensor:
    """Mean-head first-layer dense QK attention affinity, normalized to [0,1]."""
    with torch.no_grad():
        x = model.embed_nodes(nodes)
        layer = model.layers[0]
        x_norm = layer.norm1(x)
        q, k, _v = layer.attn._project(x_norm)
        scores = (q @ k.transpose(-2, -1)) / max(float(q.shape[-1]) ** 0.5, 1e-6)
        probs = torch.softmax(scores, dim=-1).mean(dim=(0, 1))
        if probs.numel():
            probs = probs / probs.max().clamp(min=1e-8)
            probs.fill_diagonal_(1.0)
        return probs


def _dense_targets(
    model: MathRoutedTransformer,
    nodes,
    env: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen dense teacher hidden/logit targets for one expression graph."""
    with torch.no_grad():
        x = model.embed_nodes(nodes)
        hidden = model(x, None, env=env)[0].detach()
        logits = model.route_logits(hidden).detach()
    return hidden, logits


def _soft_topology_forward(
    model: MathRoutedTransformer,
    nodes,
    env: dict | None,
    edge_scores: torch.Tensor,
) -> torch.Tensor:
    """Differentiable dense model forward with scorer-derived soft topology gates.

    The runtime path still uses hard TopK masks. This surrogate is used only for
    training: sigmoid(edge_scores) becomes a multiplicative attention prior, so
    hidden/logit dense-equivalence losses send gradients into the scorer.
    """
    x = model.embed_nodes(nodes)
    if edge_scores.ndim != 2 or edge_scores.shape[0] != edge_scores.shape[1]:
        raise ValueError(f"edge_scores must be square, got {tuple(edge_scores.shape)}")
    t = edge_scores.shape[0]
    gate = torch.sigmoid(edge_scores).clamp_min(1e-6)
    if t:
        gate = gate.clone()
        idx = torch.arange(t, device=gate.device)
        gate[idx, idx] = 1.0
    log_gate = torch.log(gate).view(1, 1, t, t)

    for layer in model.layers:
        x_norm = layer.norm1(x)
        q, k, v = layer.attn._project(x_norm)
        attn_logits = q @ k.transpose(-2, -1) / math.sqrt(max(q.shape[-1], 1))
        probs = F.softmax(attn_logits + log_gate, dim=-1)
        attn_raw = probs @ v
        b, _h, seq, _d = attn_raw.shape
        attn_out = layer.attn._collect(attn_raw, b, seq)
        x = x + attn_out
        x = x + layer._ff_block(layer.norm2(x))
    return model.head(x)


def _runtime_dense_equivalence_loss(
    model: MathRoutedTransformer,
    payload: dict,
    edge_scores: torch.Tensor,
    *,
    runtime_kl_loss: float,
    runtime_cos_loss: float,
    runtime_hidden_l1_loss: float,
) -> torch.Tensor:
    soft_hidden = _soft_topology_forward(model, payload["nodes"], payload["env"], edge_scores)
    soft_logits = model.route_logits(soft_hidden)
    dense_hidden = payload["dense_hidden"]
    dense_logits = payload["dense_logits"]
    loss = edge_scores.sum() * 0.0
    if runtime_kl_loss > 0.0:
        loss = loss + runtime_kl_loss * F.kl_div(
            F.log_softmax(soft_logits, dim=-1),
            F.softmax(dense_logits, dim=-1),
            reduction="batchmean",
        )
    if runtime_cos_loss > 0.0:
        loss = loss + runtime_cos_loss * (1.0 - F.cosine_similarity(
            soft_hidden.reshape(1, -1),
            dense_hidden.reshape(1, -1),
            dim=1,
        ).mean())
    if runtime_hidden_l1_loss > 0.0:
        loss = loss + runtime_hidden_l1_loss * F.l1_loss(soft_hidden, dense_hidden)
    return loss



def _write_runtime_subset(source_path: str, max_examples: int) -> str:
    if max_examples <= 0:
        return source_path
    out = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    with open(source_path, encoding="utf-8") as f, out:
        for idx, line in enumerate(f):
            if idx >= max_examples:
                break
            out.write(line)
    return out.name


def _load_replay_weights(
    replay_candidates_path: str | None,
    *,
    replay_weight_scale: float = 0.1,
    replay_max_weight: float = 8.0,
) -> dict[str, float]:
    """Load expression-level loss weights from replay candidate JSONL.

    Replay rows are selected from standardized topology traces. They should carry
    an `expr` and `replay_score`. Training remains unchanged when no replay file
    is supplied. Multiple rows for the same expression keep the largest weight.
    """
    if not replay_candidates_path:
        return {}
    path = Path(replay_candidates_path)
    if not path.exists():
        raise FileNotFoundError(f"replay candidates file not found: {replay_candidates_path}")
    scale = max(float(replay_weight_scale), 0.0)
    max_weight = max(float(replay_max_weight), 1.0)
    weights: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"replay row must be an object: {path}:{line_no}")
            expr = row.get("expr")
            if not expr:
                continue
            score = float(row.get("replay_score", 1.0) or 1.0)
            weight = min(max_weight, 1.0 + scale * max(score, 0.0))
            weights[str(expr)] = max(weights.get(str(expr), 1.0), weight)
    return weights


def _canonicalize_shape_env(env):
    """Convert JSON-loaded shape env lists into tuple shapes expected by infer_shape."""
    if env is None:
        return None
    if not isinstance(env, dict):
        return env
    out = {}
    for key, value in env.items():
        if isinstance(value, list):
            out[key] = tuple(value)
        else:
            out[key] = value
    return out


def _load_replay_records(replay_candidates_path: str | None) -> list[dict]:
    """Load replay candidate rows as route-record-like training examples.

    This is separate from replay weighting. If a replay expression is not present
    in the base training JSONL, it can still be appended to the training cache and
    trained with its replay weight. The topology scorer target is still generated
    from the hand/dense teacher; replay rows only choose *which expressions* get
    extra training pressure.
    """
    if not replay_candidates_path:
        return []
    path = Path(replay_candidates_path)
    if not path.exists():
        raise FileNotFoundError(f"replay candidates file not found: {replay_candidates_path}")
    records: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"replay row must be an object: {path}:{line_no}")
            expr = row.get("expr")
            if not expr:
                continue
            expr = str(expr)
            if expr in seen:
                continue
            seen.add(expr)
            records.append({
                "expr": expr,
                "env": _canonicalize_shape_env(row.get("env") or None),
                "expert": row.get("target_expert") or "",
                "expert_id": row.get("target_expert_id", -1),
                "source": "replay_candidate",
            })
    return records


def _quality_report_to_score_row(report) -> dict[str, float | int | str | None]:
    by_expert = getattr(report, "by_expert", None) or {}
    generic = by_expert.get("generic_expert", {}) if isinstance(by_expert, dict) else {}
    generic_correct = int(generic.get("correct", 0) or 0) if isinstance(generic, dict) else 0
    generic_total = int(generic.get("total", 0) or 0) if isinstance(generic, dict) else 0
    generic_acc = float(generic.get("accuracy", 0.0) or 0.0) if isinstance(generic, dict) else 0.0
    return {
        "mode": report.mode,
        "k": str(report.k if report.k is not None else "full"),
        "examples": report.n_examples,
        "route_acc": report.route_accuracy,
        "generic_acc": generic_acc,
        "generic_correct": generic_correct,
        "generic_total": generic_total,
        "dense_agree": report.dense_agreement,
        "hidden_l1": report.hidden_l1,
        "hidden_cos": report.hidden_cos,
        "logit_l1": report.logit_l1,
        "logit_kl": report.logit_kl_dense_to_sparse,
    }


def _runtime_quality_selection_score(row: dict[str, float | int | str | None]) -> float:
    """Route-first/generic-aware runtime checkpoint score.

    This intentionally makes route accuracy dominate representation metrics.
    Generic-expert accuracy is second because current learned-topology failures
    are concentrated there. Dense/hidden/logit agreement only break quality ties.
    """
    route = float(row.get("route_acc") or 0.0)
    generic = float(row.get("generic_acc") or 0.0)
    dense_agree = float(row.get("dense_agree") or 0.0)
    hidden_cos = float(row.get("hidden_cos") or 0.0)
    logit_kl = float(row.get("logit_kl") or 0.0)
    return (
        1_000_000.0 * route
        + 10_000.0 * generic
        + 100.0 * dense_agree
        + hidden_cos
        - logit_kl
    )


def _runtime_quality_score_for_checkpoint(
    *,
    scorer_checkpoint: str,
    examples_path: str,
    dense_checkpoint: str,
    device: str | None,
    learned_k: int,
    hand_k: int,
    middle_bridge_width: int,
    max_examples: int,
) -> tuple[float, dict[str, float | int | str | None]]:
    eval_examples = _write_runtime_subset(examples_path, max_examples)
    try:
        reports = run_quality_eval(
            examples_path=eval_examples,
            k_values=[hand_k],
            checkpoint=dense_checkpoint,
            device=device,
            topology_mode="middle_preserving_topk",
            fixed_k=hand_k,
            middle_bridge_width=middle_bridge_width,
            learned_scorer_checkpoint=scorer_checkpoint,
            learned_k=learned_k,
        )
        learned = [r for r in reports if r.mode == "learned_topology"]
        if not learned:
            raise RuntimeError("runtime quality eval produced no learned_topology report")
        row = _quality_report_to_score_row(learned[-1])
        # Keep checkpoint selection route-first. The older quality_score helper is
        # intentionally not used here because it can prefer lower KL despite lower
        # route accuracy.
        return _runtime_quality_selection_score(row), row
    finally:
        if max_examples > 0 and eval_examples != examples_path:
            try:
                Path(eval_examples).unlink(missing_ok=True)
            except OSError:
                pass



def _load_scorer_state(
    scorer: LearnedTopologyScorer,
    checkpoint: str,
    device: torch.device,
) -> dict:
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"resume scorer checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model_state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    scorer.load_state_dict(model_state)
    return state if isinstance(state, dict) else {"model_state_dict": model_state}


def train_topology_scorer(
    *,
    examples_path: str,
    save_checkpoint: str,
    max_steps: int = 1000,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    target_k: int = 16,
    eval_k: int = 8,
    local_window: int = 1,
    middle_bridge_width: int = 1,
    topology_mode: str = "middle_preserving_topk",
    device: str | None = "auto",
    log_interval: int = 50,
    max_examples: int = 0,
    seed: int = 0,
    dense_checkpoint: str | None = None,
    dense_mix: float = 0.0,
    dense_d_model: int = 64,
    dense_n_heads: int = 4,
    dense_n_layers: int = 2,
    dense_d_ff: int = 128,
    resume_scorer_checkpoint: str | None = None,
    val_examples_path: str | None = None,
    eval_interval: int = 250,
    eval_max_examples: int = 512,
    best_checkpoint: str | None = None,
    runtime_quality_examples_path: str | None = None,
    runtime_quality_checkpoint: str | None = None,
    runtime_quality_interval: int = 0,
    runtime_quality_max_examples: int = 0,
    runtime_quality_best_checkpoint: str | None = None,
    runtime_quality_patience: int = 0,
    runtime_quality_min_delta: float = 1e-5,
    runtime_quality_stop_on_degrade: bool = False,
    runtime_kl_loss: float = 0.0,
    runtime_cos_loss: float = 0.0,
    runtime_hidden_l1_loss: float = 0.0,
    replay_candidates_path: str | None = None,
    replay_weight_scale: float = 0.1,
    replay_max_weight: float = 8.0,
    replay_sample_ratio: float = 0.0,
    best_selection: str = "edge_recall",
) -> dict[str, float | int | str]:
    if not Path(examples_path).exists():
        raise FileNotFoundError(f"examples file not found: {examples_path}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _resolve_device(device)
    best_selection = str(best_selection)
    if best_selection not in {"edge_recall", "runtime_quality"}:
        raise ValueError("best_selection must be 'edge_recall' or 'runtime_quality'")
    runtime_quality_enabled = bool(runtime_quality_examples_path and runtime_quality_checkpoint and runtime_quality_interval > 0)
    if best_selection == "runtime_quality" and not runtime_quality_enabled:
        raise ValueError("best_selection='runtime_quality' requires runtime quality examples, checkpoint, and interval")

    records = _load_route_eval_records(examples_path)
    if max_examples and max_examples > 0:
        records = records[:max_examples]
    if not records:
        raise ValueError(f"no route examples loaded from {examples_path}")

    replay_weights = _load_replay_weights(
        replay_candidates_path,
        replay_weight_scale=replay_weight_scale,
        replay_max_weight=replay_max_weight,
    )
    replay_records = _load_replay_records(replay_candidates_path)
    base_exprs = {str(rec.get("expr", "")) for rec in records}
    replay_appended_examples = 0
    replay_sample_ratio = float(max(0.0, min(1.0, replay_sample_ratio)))
    for replay_rec in replay_records:
        replay_expr = str(replay_rec.get("expr", ""))
        if not replay_expr or replay_expr in base_exprs:
            continue
        records.append(replay_rec)
        base_exprs.add(replay_expr)
        replay_appended_examples += 1
    replay_weighted_examples = 0

    embedder = MathEmbedder()
    teacher = TopologyBuilder(
        topk=1,
        local_window=local_window,
        topology_mode=topology_mode,
        fixed_k=target_k,
        middle_bridge_width=middle_bridge_width,
    )
    scorer = LearnedTopologyScorer(feature_dim=len(FEATURE_NAMES), hidden_dim=hidden_dim).to(dev)
    if resume_scorer_checkpoint:
        _load_scorer_state(scorer, resume_scorer_checkpoint, dev)
        print(f"Resumed learned topology scorer from {resume_scorer_checkpoint}")
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr)
    dense_teacher = None
    dense_mix = float(max(0.0, min(1.0, dense_mix)))
    if dense_checkpoint:
        dense_teacher = _load_dense_teacher_model(
            dense_checkpoint, dev,
            d_model=dense_d_model,
            n_heads=dense_n_heads,
            n_layers=dense_n_layers,
            d_ff=dense_d_ff,
        )
    teacher_signal = "dense_qk_blend" if dense_teacher is not None and dense_mix > 0.0 else "hand_score_topk"

    runtime_equiv_enabled = dense_teacher is not None and (
        runtime_kl_loss > 0.0 or runtime_cos_loss > 0.0 or runtime_hidden_l1_loss > 0.0
    )

    if dense_teacher is not None:
        for param in dense_teacher.parameters():
            param.requires_grad_(False)

    def _build_cached(route_records, *, apply_replay_weights: bool = False):
        nonlocal replay_weighted_examples
        built: list[tuple] = []
        for rec in route_records:
            root = normalize(parse(rec["expr"]))
            nodes = root.collect_nodes()
            z = embedder.encode_batch(nodes)
            teacher_score_np, _ = build_hand_score_matrix(
                nodes,
                z,
                rec["env"] or None,
                local_window=local_window,
                include_middle_bridge=(topology_mode == "middle_preserving_topk"),
                middle_bridge_width=middle_bridge_width,
            )
            features = build_edge_feature_tensor(
                nodes,
                z,
                rec["env"] or None,
                local_window=local_window,
                middle_bridge_width=middle_bridge_width,
                device=dev,
            )
            teacher_score = torch.tensor(teacher_score_np, dtype=torch.float32, device=dev)
            if dense_teacher is not None and dense_mix > 0.0:
                hand_norm = _normalize_teacher_scores(teacher_score)
                dense_score = _dense_qk_affinity_score(dense_teacher, nodes, rec["env"] or None).to(dev)
                teacher_score = (1.0 - dense_mix) * hand_norm + dense_mix * dense_score
            target = _topk_mask_from_teacher_scores(teacher_score, eval_k).float()
            runtime_payload = None
            if runtime_equiv_enabled and dense_teacher is not None:
                dense_hidden, dense_logits = _dense_targets(dense_teacher, nodes, rec["env"] or None)
                runtime_payload = {
                    "nodes": nodes,
                    "env": rec["env"] or None,
                    "dense_hidden": dense_hidden,
                    "dense_logits": dense_logits,
                }
            sample_weight = float(replay_weights.get(str(rec.get("expr", "")), 1.0)) if apply_replay_weights else 1.0
            is_replay_weighted = bool(apply_replay_weights and sample_weight > 1.0)
            if is_replay_weighted:
                replay_weighted_examples += 1
            built.append((features, target, len(nodes), _normalize_teacher_scores(teacher_score), runtime_payload, sample_weight, is_replay_weighted))
        return built

    cached = _build_cached(records, apply_replay_weights=True)

    if not cached:
        raise ValueError("no trainable topology examples built")

    val_cached: list[tuple] = []
    if val_examples_path:
        val_records = _load_route_eval_records(val_examples_path)
        if eval_max_examples and eval_max_examples > 0:
            val_records = val_records[:eval_max_examples]
        val_cached = _build_cached(val_records, apply_replay_weights=False)

    rng = random.Random(seed)
    order = list(range(len(cached)))
    rng.shuffle(order)
    replay_indices = [idx for idx, item in enumerate(cached) if bool(item[6])]
    replay_sampled_steps = 0
    best_metric = -1.0
    default_best_path = best_checkpoint or str(Path(save_checkpoint).with_name(Path(save_checkpoint).stem + ".best.pt"))
    if best_selection == "runtime_quality":
        runtime_best_path = runtime_quality_best_checkpoint or default_best_path
        best_path = str(Path(save_checkpoint).with_name(Path(save_checkpoint).stem + ".edge_best.pt"))
    else:
        best_path = default_best_path
        runtime_best_path = runtime_quality_best_checkpoint or str(Path(save_checkpoint).with_name(Path(save_checkpoint).stem + ".runtime_best.pt"))
    runtime_best_score = -1.0
    runtime_stale_count = 0
    last_runtime_quality: dict[str, float | int | str | None] = {}
    best_runtime_quality: dict[str, float | int | str | None] = {}
    stopped_early = False
    stop_step = max_steps

    def _save_runtime_temp(path: str) -> None:
        _save_scorer_checkpoint(
            path, scorer, hidden_dim=hidden_dim, target_k=target_k,
            eval_k=eval_k, local_window=local_window,
            middle_bridge_width=middle_bridge_width, topology_mode=topology_mode,
            examples_path=examples_path, best_metric=(best_metric if best_metric >= 0 else None),
            teacher_signal=teacher_signal, dense_checkpoint=dense_checkpoint, dense_mix=dense_mix,
            resume_scorer_checkpoint=resume_scorer_checkpoint,
            replay_candidates_path=replay_candidates_path,
            replay_weight_scale=replay_weight_scale,
            replay_max_weight=replay_max_weight,
            replay_weighted_examples=replay_weighted_examples,
            replay_appended_examples=replay_appended_examples,
            replay_sample_ratio=replay_sample_ratio,
            replay_sampled_steps=replay_sampled_steps,
            best_selection=best_selection,
        )

    def _maybe_update_runtime_best(step_label: str) -> None:
        nonlocal runtime_best_score, runtime_stale_count, last_runtime_quality, best_runtime_quality, stopped_early, stop_step
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_ckpt:
            runtime_tmp = tmp_ckpt.name
        _save_runtime_temp(runtime_tmp)
        try:
            runtime_score, runtime_row = _runtime_quality_score_for_checkpoint(
                scorer_checkpoint=runtime_tmp,
                examples_path=runtime_quality_examples_path or "",
                dense_checkpoint=runtime_quality_checkpoint or "",
                device=device,
                learned_k=eval_k,
                hand_k=target_k,
                middle_bridge_width=middle_bridge_width,
                max_examples=runtime_quality_max_examples,
            )
            last_runtime_quality = dict(runtime_row)
            print(
                f"         runtime_J={runtime_score:.6f} "
                f"runtime_route={float(runtime_row['route_acc']):.4f} "
                f"runtime_cos={float(runtime_row.get('hidden_cos') or 0.0):.6f} "
                f"runtime_kl={float(runtime_row.get('logit_kl') or 0.0):.6f}"
            )
            if runtime_score > runtime_best_score + runtime_quality_min_delta:
                runtime_best_score = runtime_score
                best_runtime_quality = dict(runtime_row)
                runtime_stale_count = 0
                _save_runtime_temp(runtime_best_path)
                print(f"         saved runtime-best scorer to {runtime_best_path}")
            else:
                runtime_stale_count += 1
                print(
                    f"         runtime_quality_stale={runtime_stale_count} "
                    f"best_runtime_J={runtime_best_score:.6f}"
                )
                if runtime_quality_stop_on_degrade and runtime_stale_count >= max(runtime_quality_patience, 0):
                    stopped_early = True
                    try:
                        stop_step = int(step_label)
                    except ValueError:
                        stop_step = 0
                    print(f"         runtime quality early stop at step={step_label}")
        finally:
            try:
                Path(runtime_tmp).unlink(missing_ok=True)
            except OSError:
                pass

    if runtime_quality_enabled:
        print("         runtime quality baseline before training")
        _maybe_update_runtime_best("baseline")

    last_loss = 0.0
    last_metrics = {"edge_recall": 0.0, "edge_precision": 0.0, "self_loop_rate": 0.0}
    last_val_metrics: dict[str, float] = {}
    for step in range(max_steps):
        if step > 0 and step % len(order) == 0:
            rng.shuffle(order)
        use_replay_sample = bool(replay_indices and replay_sample_ratio > 0.0 and rng.random() < replay_sample_ratio)
        if use_replay_sample:
            item_idx = rng.choice(replay_indices)
            replay_sampled_steps += 1
        else:
            item_idx = order[step % len(order)]
        features, target, n_nodes, teacher_score, runtime_payload, sample_weight, is_replay_weighted = cached[item_idx]
        scores = scorer(features)
        # Positives are sparse; use a per-example positive weight so identity/relations
        # are not drowned by disconnected pairs.
        pos = target.sum().clamp(min=1.0)
        neg = target.numel() - pos
        pos_weight = (neg / pos).clamp(min=1.0, max=128.0)
        bce_loss = F.binary_cross_entropy_with_logits(scores, target, pos_weight=pos_weight)
        score_loss = F.mse_loss(torch.sigmoid(scores), teacher_score)
        rank_loss = _pairwise_rank_loss(scores, target.bool())
        loss = bce_loss + 0.5 * score_loss + 0.25 * rank_loss
        # Runtime-J-aligned dense-equivalence loss. When a dense teacher is
        # available, use a differentiable soft-topology surrogate through the
        # frozen dense model so hidden/logit losses backpropagate into scorer
        # edge scores. Without a dense teacher, fall back to the old score-space
        # proxy instead of silently dropping configured loss weights.
        if runtime_payload is not None and dense_teacher is not None:
            loss = loss + _runtime_dense_equivalence_loss(
                dense_teacher,
                runtime_payload,
                scores,
                runtime_kl_loss=runtime_kl_loss,
                runtime_cos_loss=runtime_cos_loss,
                runtime_hidden_l1_loss=runtime_hidden_l1_loss,
            )
        else:
            if runtime_kl_loss > 0.0:
                loss = loss + runtime_kl_loss * F.kl_div(
                    F.log_softmax(scores.reshape(1, -1), dim=-1),
                    F.softmax(teacher_score.reshape(1, -1), dim=-1),
                    reduction="batchmean",
                )
            if runtime_cos_loss > 0.0:
                loss = loss + runtime_cos_loss * (1.0 - F.cosine_similarity(
                    torch.sigmoid(scores).reshape(1, -1),
                    teacher_score.reshape(1, -1),
                    dim=1,
                ).mean())
            if runtime_hidden_l1_loss > 0.0:
                loss = loss + runtime_hidden_l1_loss * F.l1_loss(torch.sigmoid(scores), teacher_score)

        loss = loss * float(sample_weight)

        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

        if step % log_interval == 0 or step == max_steps - 1:
            last_metrics = _edge_metrics(scores.detach(), target.bool(), eval_k)
            print(
                f"step={step:5d} loss={last_loss:.6f} n={n_nodes:3d} w={float(sample_weight):.3f} replay={int(is_replay_weighted)} "
                f"recall@{eval_k}={last_metrics['edge_recall']:.4f} "
                f"precision@{eval_k}={last_metrics['edge_precision']:.4f} "
                f"self={last_metrics['self_loop_rate']:.4f}"
            )
        if val_cached and (step % eval_interval == 0 or step == max_steps - 1):
            last_val_metrics = _aggregate_metrics(scorer, val_cached, eval_k)
            val_metric = float(last_val_metrics["mean_row_recall"])
            print(
                f"         val_mean_recall@{eval_k}={val_metric:.4f} "
                f"val_micro_recall@{eval_k}={last_val_metrics['micro_recall']:.4f} "
                f"val_self={last_val_metrics['self_loop_rate']:.4f}"
            )
            if val_metric > best_metric:
                best_metric = val_metric
                _save_scorer_checkpoint(
                    best_path, scorer, hidden_dim=hidden_dim, target_k=target_k,
                    eval_k=eval_k, local_window=local_window,
                    middle_bridge_width=middle_bridge_width, topology_mode=topology_mode,
                    examples_path=examples_path, best_metric=best_metric,
                    teacher_signal=teacher_signal, dense_checkpoint=dense_checkpoint, dense_mix=dense_mix,
                    resume_scorer_checkpoint=resume_scorer_checkpoint,
                    replay_candidates_path=replay_candidates_path,
                    replay_weight_scale=replay_weight_scale,
                    replay_max_weight=replay_max_weight,
                    replay_weighted_examples=replay_weighted_examples,
                    replay_appended_examples=replay_appended_examples,
                    replay_sample_ratio=replay_sample_ratio,
                    replay_sampled_steps=replay_sampled_steps,
                    best_selection=best_selection,
                )
                print(f"         saved best scorer to {best_path}")

        if (
            runtime_quality_enabled
            and (step % runtime_quality_interval == 0 or step == max_steps - 1)
        ):
            _maybe_update_runtime_best(str(step))
            if stopped_early:
                break

    _save_scorer_checkpoint(
        save_checkpoint, scorer, hidden_dim=hidden_dim, target_k=target_k,
        eval_k=eval_k, local_window=local_window,
        middle_bridge_width=middle_bridge_width, topology_mode=topology_mode,
        examples_path=examples_path, best_metric=(best_metric if best_metric >= 0 else None),
        teacher_signal=teacher_signal, dense_checkpoint=dense_checkpoint, dense_mix=dense_mix,
        resume_scorer_checkpoint=resume_scorer_checkpoint,
        replay_candidates_path=replay_candidates_path,
        replay_weight_scale=replay_weight_scale,
        replay_max_weight=replay_max_weight,
        replay_weighted_examples=replay_weighted_examples,
        replay_appended_examples=replay_appended_examples,
        replay_sample_ratio=replay_sample_ratio,
        replay_sampled_steps=replay_sampled_steps,
    )
    print(f"Saved learned topology scorer to {save_checkpoint}")
    edge_best_checkpoint = best_path if val_cached and best_metric >= 0 else ""
    runtime_selected_checkpoint = runtime_best_path if runtime_best_score >= 0 else ""
    selected_checkpoint = runtime_selected_checkpoint if best_selection == "runtime_quality" else edge_best_checkpoint
    selected_score = runtime_best_score if best_selection == "runtime_quality" else (best_metric if best_metric >= 0 else 0.0)
    return {
        "examples": len(cached),
        "steps": stop_step + 1 if stopped_early else max_steps,
        "stopped_early": int(stopped_early),
        "loss": last_loss,
        "checkpoint": str(save_checkpoint),
        "best_selection": best_selection,
        "selected_checkpoint": selected_checkpoint,
        "selected_score": selected_score if selected_score >= 0 else 0.0,
        "best_checkpoint": selected_checkpoint,
        "edge_best_checkpoint": edge_best_checkpoint,
        "best_mean_row_recall": best_metric if best_metric >= 0 else 0.0,
        "runtime_best_checkpoint": runtime_selected_checkpoint,
        "runtime_best_score": runtime_best_score if runtime_best_score >= 0 else 0.0,
        "replay_candidates_path": replay_candidates_path or "",
        "replay_weighted_examples": replay_weighted_examples,
        "replay_appended_examples": replay_appended_examples,
        "replay_sample_ratio": float(replay_sample_ratio),
        "replay_sampled_steps": replay_sampled_steps,
        "replay_weight_scale": float(replay_weight_scale),
        "replay_max_weight": float(replay_max_weight),
        **{f"runtime_{k}": v for k, v in last_runtime_quality.items()},
        **{f"runtime_best_{k}": v for k, v in best_runtime_quality.items()},
        **last_metrics,
        **{f"val_{k}": v for k, v in last_val_metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned topology scorer from hand-scored TopK pseudo-targets.")
    parser.add_argument("--examples", default="data/synthetic_hard/train.jsonl")
    parser.add_argument("--save-checkpoint", default="runs/checkpoints/learned_topology_scorer.pt")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--target-k", type=int, default=16)
    parser.add_argument("--eval-k", type=int, default=8)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--middle-bridge-width", type=int, default=1)
    parser.add_argument("--topology-mode", default="middle_preserving_topk", choices=["scored_topk", "middle_preserving_topk"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dense-checkpoint", default=None, dest="dense_checkpoint")
    parser.add_argument("--dense-mix", type=float, default=0.0, dest="dense_mix")
    parser.add_argument("--dense-d-model", type=int, default=64, dest="dense_d_model")
    parser.add_argument("--dense-n-heads", type=int, default=4, dest="dense_n_heads")
    parser.add_argument("--dense-n-layers", type=int, default=2, dest="dense_n_layers")
    parser.add_argument("--dense-d-ff", type=int, default=128, dest="dense_d_ff")
    parser.add_argument("--resume-scorer-checkpoint", default=None, dest="resume_scorer_checkpoint")
    parser.add_argument("--val-examples", default=None, dest="val_examples_path")
    parser.add_argument("--eval-interval", type=int, default=250, dest="eval_interval")
    parser.add_argument("--eval-max-examples", type=int, default=512, dest="eval_max_examples")
    parser.add_argument("--best-checkpoint", default=None, dest="best_checkpoint")
    parser.add_argument("--best-selection", default="edge_recall", choices=["edge_recall", "runtime_quality"], dest="best_selection")
    parser.add_argument("--runtime-quality-examples", default=None, dest="runtime_quality_examples_path")
    parser.add_argument("--runtime-quality-checkpoint", default=None, dest="runtime_quality_checkpoint")
    parser.add_argument("--runtime-quality-interval", type=int, default=0, dest="runtime_quality_interval")
    parser.add_argument("--runtime-quality-max-examples", type=int, default=0, dest="runtime_quality_max_examples")
    parser.add_argument("--runtime-quality-best-checkpoint", default=None, dest="runtime_quality_best_checkpoint")
    parser.add_argument("--runtime-quality-patience", type=int, default=0, dest="runtime_quality_patience")
    parser.add_argument("--runtime-quality-min-delta", type=float, default=1e-5, dest="runtime_quality_min_delta")
    parser.add_argument("--runtime-quality-stop-on-degrade", action="store_true", dest="runtime_quality_stop_on_degrade")
    parser.add_argument("--runtime-kl-loss", type=float, default=0.0, dest="runtime_kl_loss")
    parser.add_argument("--runtime-cos-loss", type=float, default=0.0, dest="runtime_cos_loss")
    parser.add_argument("--runtime-hidden-l1-loss", type=float, default=0.0, dest="runtime_hidden_l1_loss")
    parser.add_argument("--replay-candidates", default=None, dest="replay_candidates_path")
    parser.add_argument("--replay-weight-scale", type=float, default=0.1, dest="replay_weight_scale")
    parser.add_argument("--replay-max-weight", type=float, default=8.0, dest="replay_max_weight")
    parser.add_argument("--replay-sample-ratio", type=float, default=0.0, dest="replay_sample_ratio")
    args = parser.parse_args()
    summary = train_topology_scorer(
        examples_path=args.examples,
        save_checkpoint=args.save_checkpoint,
        max_steps=args.max_steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        target_k=args.target_k,
        eval_k=args.eval_k,
        local_window=args.local_window,
        middle_bridge_width=args.middle_bridge_width,
        topology_mode=args.topology_mode,
        device=args.device,
        log_interval=args.log_interval,
        max_examples=args.max_examples,
        seed=args.seed,
        dense_checkpoint=args.dense_checkpoint,
        dense_mix=args.dense_mix,
        dense_d_model=args.dense_d_model,
        dense_n_heads=args.dense_n_heads,
        dense_n_layers=args.dense_n_layers,
        dense_d_ff=args.dense_d_ff,
        resume_scorer_checkpoint=args.resume_scorer_checkpoint,
        val_examples_path=args.val_examples_path,
        eval_interval=args.eval_interval,
        eval_max_examples=args.eval_max_examples,
        best_checkpoint=args.best_checkpoint,
        best_selection=args.best_selection,
        runtime_quality_examples_path=args.runtime_quality_examples_path,
        runtime_quality_checkpoint=args.runtime_quality_checkpoint,
        runtime_quality_interval=args.runtime_quality_interval,
        runtime_quality_max_examples=args.runtime_quality_max_examples,
        runtime_quality_best_checkpoint=args.runtime_quality_best_checkpoint,
        runtime_quality_patience=args.runtime_quality_patience,
        runtime_quality_min_delta=args.runtime_quality_min_delta,
        runtime_quality_stop_on_degrade=args.runtime_quality_stop_on_degrade,
        runtime_kl_loss=args.runtime_kl_loss,
        runtime_cos_loss=args.runtime_cos_loss,
        runtime_hidden_l1_loss=args.runtime_hidden_l1_loss,
        replay_candidates_path=args.replay_candidates_path,
        replay_weight_scale=args.replay_weight_scale,
        replay_max_weight=args.replay_max_weight,
        replay_sample_ratio=args.replay_sample_ratio,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()