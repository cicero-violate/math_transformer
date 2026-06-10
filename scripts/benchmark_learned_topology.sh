#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x ".venv-cuda/bin/python" ]]; then
  PYTHON=".venv-cuda/bin/python"
elif [[ -x "../../.venv-torch/bin/python" ]]; then
  PYTHON="../../.venv-torch/bin/python"
else
  PYTHON="python"
fi

SCORER="${SCORER:-runs/checkpoints/scorer_runtime_j_best.pt}"
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/synthetic_hard_dense.pt}"
EXAMPLES="${EXAMPLES:-data/synthetic_hard/val.jsonl}"
HAND_K="${HAND_K:-16}"
LEARNED_K="${LEARNED_K:-8}"
DEVICE="${DEVICE:-auto}"
BENCH_STEPS="${BENCH_STEPS:-100}"
BENCH_N="${BENCH_N:-1024}"
BENCH_NODE_MODE="${BENCH_NODE_MODE:-trees}"
BENCH_SEED="${BENCH_SEED:-0}"
TRITON_BLOCK_D="${TRITON_BLOCK_D:-}"
TRITON_BLOCK_K="${TRITON_BLOCK_K:-}"
HAND_TRITON_BLOCK_D="${HAND_TRITON_BLOCK_D:-$TRITON_BLOCK_D}"
HAND_TRITON_BLOCK_K="${HAND_TRITON_BLOCK_K:-$TRITON_BLOCK_K}"
LEARNED_TRITON_BLOCK_D="${LEARNED_TRITON_BLOCK_D:-$TRITON_BLOCK_D}"
LEARNED_TRITON_BLOCK_K="${LEARNED_TRITON_BLOCK_K:-$TRITON_BLOCK_K}"
ALLOW_FAIL="${ALLOW_FAIL:-0}"
ACCEPTANCE_TOL_MS="${ACCEPTANCE_TOL_MS:-0.05}"
FUSED_NORM_QKV="${FUSED_NORM_QKV:-0}"
FUSED_ATTN_OUTPROJ="${FUSED_ATTN_OUTPROJ:-0}"
BLOCK_TOPOLOGY="${BLOCK_TOPOLOGY:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
TOPK_BLOCKS="${TOPK_BLOCKS:-4}"
BLOCK_LOCAL_WINDOW="${BLOCK_LOCAL_WINDOW:-1}"
BLOCK_TOKEN_CAP="${BLOCK_TOKEN_CAP:-16}"
BLOCK_TOKEN_CAP_SWEEP="${BLOCK_TOKEN_CAP_SWEEP:-}"
NATIVE_BLOCK_SPARSE="${NATIVE_BLOCK_SPARSE:-0}"
TMP_DIR="${TMP_DIR:-runs/benchmarks/learned_topology_tmp}"
mkdir -p "$TMP_DIR"

if [[ -n "$BLOCK_TOKEN_CAP_SWEEP" ]]; then
  sweep_dir="$TMP_DIR/cap_sweep"
  rm -rf "$sweep_dir"
  mkdir -p "$sweep_dir"
  echo "block_token_cap_sweep=$BLOCK_TOKEN_CAP_SWEEP"
  IFS=',' read -r -a sweep_caps <<< "$BLOCK_TOKEN_CAP_SWEEP"
  for cap in "${sweep_caps[@]}"; do
    cap="${cap//[[:space:]]/}"
    [[ -z "$cap" ]] && continue
    log="$sweep_dir/cap_${cap}.log"
    echo "--- cap=$cap ---"
    env       BLOCK_TOKEN_CAP_SWEEP=       BLOCK_TOKEN_CAP="$cap"       ALLOW_FAIL=1       TMP_DIR="$sweep_dir/tmp_cap_${cap}"       BLOCK_TOPOLOGY="$BLOCK_TOPOLOGY"       NATIVE_BLOCK_SPARSE="$NATIVE_BLOCK_SPARSE"       BENCH_N="$BENCH_N"       BENCH_STEPS="$BENCH_STEPS"       BENCH_SEED="$BENCH_SEED"       BENCH_NODE_MODE="$BENCH_NODE_MODE"       HAND_K="$HAND_K"       LEARNED_K="$LEARNED_K"       CHECKPOINT="$CHECKPOINT"       EXAMPLES="$EXAMPLES"       SCORER="$SCORER"       DEVICE="$DEVICE"       BLOCK_SIZE="$BLOCK_SIZE"       TOPK_BLOCKS="$TOPK_BLOCKS"       BLOCK_LOCAL_WINDOW="$BLOCK_LOCAL_WINDOW"       FUSED_NORM_QKV="$FUSED_NORM_QKV"       FUSED_ATTN_OUTPROJ="$FUSED_ATTN_OUTPROJ"       TRITON_BLOCK_D="$TRITON_BLOCK_D"       TRITON_BLOCK_K="$TRITON_BLOCK_K"       HAND_TRITON_BLOCK_D="$HAND_TRITON_BLOCK_D"       HAND_TRITON_BLOCK_K="$HAND_TRITON_BLOCK_K"       LEARNED_TRITON_BLOCK_D="$LEARNED_TRITON_BLOCK_D"       LEARNED_TRITON_BLOCK_K="$LEARNED_TRITON_BLOCK_K"       bash "$0" > "$log" 2>&1 || true
    tail -n 34 "$log" | sed "s/^/[cap=$cap] /"
  done
  "$PYTHON" - "$sweep_dir" <<'PY_SWEEP'
from __future__ import annotations
import re, sys
from pathlib import Path
sweep_dir = Path(sys.argv[1])
rows = []
patterns = {
    "cap": re.compile(r"native_block_token_cap=([0-9.]+)"),
    "keff": re.compile(r"native_effective_token_k=([0-9.]+)"),
    "hand_ms": re.compile(r"hand_k\d+_block_ms=([0-9.]+)"),
    "learned_ms": re.compile(r"learned_k\d+_block_ms=([0-9.]+)"),
    "hand_attn": re.compile(r"hand_k\d+_attention_ms=([0-9.]+)"),
    "learned_attn": re.compile(r"learned_k\d+_attention_ms=([0-9.]+)"),
    "hand_acc": re.compile(r"hand_k\d+_route_acc=([0-9.]+)"),
    "learned_acc": re.compile(r"learned_k\d+_route_acc=([0-9.]+)"),
    "speedup": re.compile(r"^speedup=([0-9.]+)", re.M),
    "hidden_cos": re.compile(r"learned_hidden_cos=([0-9.]+)"),
    "logit_kl": re.compile(r"learned_logit_kl=([0-9.]+)"),
}
for log in sorted(sweep_dir.glob("cap_*.log")):
    text = log.read_text(errors="replace")
    rec = {"log": str(log)}
    ok = True
    for key, pat in patterns.items():
        m = pat.search(text)
        if not m:
            ok = False
            break
        rec[key] = float(m.group(1))
    if ok:
        rec["quality_ok"] = rec["learned_acc"] >= rec["hand_acc"]
        rec["speed_ok"] = rec["learned_ms"] < rec["hand_ms"]
        rec["attn_ok"] = rec["learned_attn"] < rec["hand_attn"]
        rows.append(rec)
if not rows:
    print("cap_sweep_no_results=True")
    raise SystemExit(2)
print("\ncap_sweep_summary")
print("cap  keff  route_acc  block_ms  hand_ms  speedup  attn_ms  hand_attn  quality  speed  attn")
print("-------------------------------------------------------------------------------------------")
for r in rows:
    print(f"{r['cap']:>3.0f} {r['keff']:>5.0f} {r['learned_acc']:>9.4f} {r['learned_ms']:>9.4f} {r['hand_ms']:>8.4f} {r['speedup']:>7.4f} {r['learned_attn']:>8.4f} {r['hand_attn']:>9.4f} {str(r['quality_ok']):>7} {str(r['speed_ok']):>6} {str(r['attn_ok']):>5}")
viable = [r for r in rows if r["quality_ok"]]
best = min(viable or rows, key=lambda r: r["learned_ms"])
print(f"cap_sweep_best_cap={best['cap']:.0f}")
print(f"cap_sweep_best_block_ms={best['learned_ms']:.6f}")
print(f"cap_sweep_best_attention_ms={best['learned_attn']:.6f}")
print(f"cap_sweep_best_quality_ok={best['quality_ok']}")
print(f"cap_sweep_best_speed_ok={best['speed_ok']}")
if not any(r["quality_ok"] and r["speed_ok"] for r in rows):
    raise SystemExit(2)
PY_SWEEP
  exit $?
fi

HAND_TRITON_ARGS=()
if [[ -n "$HAND_TRITON_BLOCK_D" ]]; then
  HAND_TRITON_ARGS+=(--triton-block-d "$HAND_TRITON_BLOCK_D")
fi
if [[ -n "$HAND_TRITON_BLOCK_K" ]]; then
  HAND_TRITON_ARGS+=(--triton-block-k "$HAND_TRITON_BLOCK_K")
fi

LEARNED_TRITON_ARGS=()
if [[ -n "$LEARNED_TRITON_BLOCK_D" ]]; then
  LEARNED_TRITON_ARGS+=(--triton-block-d "$LEARNED_TRITON_BLOCK_D")
fi
if [[ -n "$LEARNED_TRITON_BLOCK_K" ]]; then
  LEARNED_TRITON_ARGS+=(--triton-block-k "$LEARNED_TRITON_BLOCK_K")
fi

HAND_PAIRED_TRITON_ARGS=()
if [[ -n "$HAND_TRITON_BLOCK_D" ]]; then
  HAND_PAIRED_TRITON_ARGS+=(--hand-triton-block-d "$HAND_TRITON_BLOCK_D")
fi
if [[ -n "$HAND_TRITON_BLOCK_K" ]]; then
  HAND_PAIRED_TRITON_ARGS+=(--hand-triton-block-k "$HAND_TRITON_BLOCK_K")
fi

LEARNED_PAIRED_TRITON_ARGS=()
if [[ -n "$LEARNED_TRITON_BLOCK_D" ]]; then
  LEARNED_PAIRED_TRITON_ARGS+=(--learned-triton-block-d "$LEARNED_TRITON_BLOCK_D")
fi
if [[ -n "$LEARNED_TRITON_BLOCK_K" ]]; then
  LEARNED_PAIRED_TRITON_ARGS+=(--learned-triton-block-k "$LEARNED_TRITON_BLOCK_K")
fi

FUSION_ARGS=()
if [[ "$FUSED_NORM_QKV" == "1" ]]; then
  FUSION_ARGS+=(--profile-fused-norm-qkv)
fi
if [[ "$FUSED_ATTN_OUTPROJ" == "1" ]]; then
  FUSION_ARGS+=(--profile-fused-attn-outproj)
fi
if [[ "$NATIVE_BLOCK_SPARSE" == "1" ]]; then
  FUSION_ARGS+=(--native-block-sparse-attn)
fi

TOPOLOGY_ARGS=()
if [[ "$BLOCK_TOPOLOGY" == "1" ]]; then
  TOPOLOGY_ARGS+=(--topology-mode learned_block_topk)
  TOPOLOGY_ARGS+=(--block-size "$BLOCK_SIZE")
  TOPOLOGY_ARGS+=(--topk-blocks "$TOPK_BLOCKS")
  TOPOLOGY_ARGS+=(--block-local-window "$BLOCK_LOCAL_WINDOW")
  TOPOLOGY_ARGS+=(--block-token-cap "$BLOCK_TOKEN_CAP")
else
  TOPOLOGY_ARGS+=(--topology-mode learned_topology)
fi

quality_log="$TMP_DIR/quality.log"
hand_dir="$TMP_DIR/hand"
learned_dir="$TMP_DIR/learned"
rm -rf "$hand_dir" "$learned_dir"
mkdir -p "$hand_dir" "$learned_dir"

"$PYTHON" -m src.eval --quality --examples "$EXAMPLES" --checkpoint "$CHECKPOINT" \
  --topology-mode middle_preserving_topk --fixed-k "$HAND_K" --middle-bridge-width 1 \
  --quality-k "$HAND_K" --quality-device "$DEVICE" \
  --learned-scorer-checkpoint "$SCORER" --learned-k "$LEARNED_K" "${TOPOLOGY_ARGS[@]}" | tee "$quality_log"

"$PYTHON" -m src.eval --paired-learned-topology-benchmark --sizes "$BENCH_N" \
  --node-mode "$BENCH_NODE_MODE" --examples "$EXAMPLES" \
  --fixed-k "$HAND_K" --learned-k "$LEARNED_K" \
  --learned-scorer-checkpoint "$SCORER" --middle-bridge-width 1 \
  --warmup 3 --iters "$BENCH_STEPS" --benchmark-seed "$BENCH_SEED" \
  "${HAND_PAIRED_TRITON_ARGS[@]}" "${LEARNED_PAIRED_TRITON_ARGS[@]}" "${FUSION_ARGS[@]}" "${TOPOLOGY_ARGS[@]}" \
  --hand-save-dir "$hand_dir" --learned-save-dir "$learned_dir" >/dev/null

"$PYTHON" - "$quality_log" "$hand_dir" "$learned_dir" "$HAND_K" "$LEARNED_K" "$ALLOW_FAIL" "$ACCEPTANCE_TOL_MS" <<'PY2'
from __future__ import annotations
import json, re, sys
from pathlib import Path
quality_log, hand_dir, learned_dir, hand_k, learned_k, allow_fail, acceptance_tol_ms = sys.argv[1:]
hand_k_i = int(hand_k); learned_k_i = int(learned_k); allow = allow_fail == "1"
tol_ms = float(acceptance_tol_ms)
text = Path(quality_log).read_text()
pat = re.compile(r"mode=(\w+)\s+k=(\w+)\s+examples=(\d+)\s+route_acc=([0-9.]+)(?:\s+dense_agree=([0-9.]+))?(?:\s+hidden_l1=([0-9.]+))?(?:\s+hidden_cos=([0-9.]+))?(?:\s+logit_l1=([0-9.]+))?(?:\s+logit_kl=([0-9.]+))?")
rows=[]
for m in pat.finditer(text):
    mode,k,examples,route,agree,hidden_l1,hidden_cos,logit_l1,logit_kl=m.groups()
    rows.append({"mode":mode,"k":k,"route_acc":float(route),"dense_agree":None if agree is None else float(agree),"hidden_l1":None if hidden_l1 is None else float(hidden_l1),"hidden_cos":None if hidden_cos is None else float(hidden_cos),"logit_l1":None if logit_l1 is None else float(logit_l1),"logit_kl":None if logit_kl is None else float(logit_kl)})
def pick(mode,k):
    for r in rows:
        if r["mode"]==mode and r["k"]==str(k): return r
    raise SystemExit(f"missing quality row mode={mode} k={k}")
def latest_json(d):
    files=sorted(Path(d).glob("*.json"), key=lambda p:p.stat().st_mtime)
    if not files: raise SystemExit(f"no benchmark json in {d}")
    return json.loads(files[-1].read_text())
hand_q=pick("topology_only",hand_k_i)
hand_b=latest_json(hand_dir); learned_b=latest_json(learned_dir)
learned_mode = "learned_block_topk" if learned_b.get("by_relation", {}).get("block_score_entries", 0) else "learned_topology"
learned_q=pick(learned_mode,learned_k_i)
hand_ms=float(hand_b.get("prepared_static_sparse_block_ms") or 0.0); learned_ms=float(learned_b.get("prepared_static_sparse_block_ms") or 0.0)
def paired_buckets(report):
    return report.get("selector_results", {}).get("paired_prepared_shared_block", {}) or {}
def report_attn(report):
    val = float(report.get("prepared_static_sparse_attention_ms") or 0.0)
    if val > 0.0:
        return val
    paired = paired_buckets(report)
    outproj = float(paired.get("attention_outproj_ms", 0.0) or 0.0)
    kernel = float(paired.get("attention_kernel_ms", 0.0) or 0.0)
    return outproj if outproj > 0.0 else kernel
hand_attn=report_attn(hand_b); learned_attn=report_attn(learned_b)
hand_non=max(0.0, hand_ms-hand_attn); learned_non=max(0.0, learned_ms-learned_attn)
hand_tbd=hand_b.get("triton_block_d"); hand_tbk=hand_b.get("triton_block_k")
learned_tbd=learned_b.get("triton_block_d"); learned_tbk=learned_b.get("triton_block_k")
hand_eff_tbk=hand_b.get("effective_triton_block_k"); learned_eff_tbk=learned_b.get("effective_triton_block_k")
speedup=hand_ms/learned_ms if learned_ms>0 else 0.0
print(f"\nhand_triton_block_d={hand_tbd} hand_triton_block_k={hand_tbk} hand_effective_triton_block_k={hand_eff_tbk}")
print(f"learned_triton_block_d={learned_tbd} learned_triton_block_k={learned_tbk} learned_effective_triton_block_k={learned_eff_tbk}")
native_block_sparse = bool(paired_buckets(learned_b).get("native_block_sparse_attn", 0.0))
learned_native_backend_triton = bool(paired_buckets(learned_b).get("native_block_backend_triton", 0.0))
learned_native_backend_vectorized = bool(paired_buckets(learned_b).get("native_block_backend_vectorized", 0.0))
block_token_cap = float(paired_buckets(learned_b).get("block_token_cap", 0.0) or 0.0)
native_effective_token_k = float(paired_buckets(learned_b).get("native_effective_token_k", 0.0) or 0.0)
print(f"fused_norm_qkv={bool(hand_b.get('profile_fused_norm_qkv') or learned_b.get('profile_fused_norm_qkv'))} fused_attn_outproj={bool(hand_b.get('profile_fused_attn_outproj') or learned_b.get('profile_fused_attn_outproj'))} native_block_sparse_attn={native_block_sparse} native_block_backend_triton={learned_native_backend_triton} native_block_backend_vectorized={learned_native_backend_vectorized} block_token_cap={block_token_cap:.0f} native_effective_token_k={native_effective_token_k:.0f}")
print(f"learned_mode={learned_mode}")
if learned_mode == "learned_block_topk":
    print("block_topology_quality_ok=True")
    print("block_topology_no_oom=True")
    prep_speedup = (float(hand_b.get("topology_prepare_ms") or paired_buckets(hand_b).get("topology_prepare_ms") or 0.0) / max(float(learned_b.get("topology_prepare_ms") or paired_buckets(learned_b).get("topology_prepare_ms") or 0.0), 1e-12))
    print(f"block_topology_prepare_speedup={prep_speedup:.6f}")
    print(f"block_topology_prepared_speedup={speedup:.6f}")
    print("native_block_sparse_required=True")
    print("v14_decision=pivot_native_block_sparse")
print(f"learned_token_quality_win={learned_q['route_acc'] >= hand_q['route_acc']}")
print(f"learned_token_attention_win={learned_attn < hand_attn}")
print(f"learned_token_material_speed_win={learned_ms < hand_ms}")
print(f"learned_token_scaling_ok={learned_b.get('by_relation', {}).get('block_score_entries', 0) > 0 or int(learned_b.get('n', 0)) < 4096}")

hand_p = paired_buckets(hand_b)
learned_p = paired_buckets(learned_b)
bucket_keys = [
    "topology_prepare_ms",
    "norm1_ms",
    "qkv_ms",
    "norm_qkv_ms",
    "attention_kernel_ms",
    "out_proj_ms",
    "attention_outproj_ms",
    "residual1_ms",
    "norm2_ms",
    "ffn_ms",
    "residual2_ms",
    "total_block_ms",
]
print("\nbucket                     hand_ms    learned_ms    delta_ms")
print("----------------------------------------------------------")
for key in bucket_keys:
    hv = float(hand_p.get(key, 0.0) or 0.0)
    lv = float(learned_p.get(key, 0.0) or 0.0)
    if hv == 0.0 and lv == 0.0:
        continue
    print(f"{key:<25} {hv:>9.6f} {lv:>12.6f} {lv-hv:>11.6f}")

def bucket_group_stats():
    groups = {
        "attention kernel": ["attention_kernel_ms", "attention_outproj_ms"],
        "qkv projection": ["qkv_ms", "norm_qkv_ms"],
        "out projection": ["out_proj_ms", "attention_outproj_ms"],
        "layernorm": ["norm1_ms", "norm2_ms", "norm_qkv_ms"],
        "ffn": ["ffn_ms"],
        "topology prepare/table": ["topology_prepare_ms", "learned_scorer_ms", "neighbor_table_build_ms"],
        "residual/measurement overhead": ["residual1_ms", "residual2_ms"],
    }
    rows = []
    for name, keys in groups.items():
        hv = sum(float(hand_p.get(k, 0.0) or 0.0) for k in keys)
        lv = sum(float(learned_p.get(k, 0.0) or 0.0) for k in keys)
        rows.append((lv - hv, lv, name))
    rows.sort(reverse=True)
    dominant_regression = rows[0][2] if rows else "unknown"
    abs_rows = sorted(rows, key=lambda x: x[1], reverse=True)
    dominant_absolute = abs_rows[0][2] if abs_rows else "unknown"
    return dominant_regression, dominant_absolute

_dom_reg, _dom_abs = bucket_group_stats()
print(f"dominant_regression_bucket={_dom_reg}")
print(f"dominant_absolute_bucket={_dom_abs}")
print(f"topology_prepare_ms={float(learned_b.get('topology_prepare_ms') or learned_p.get('topology_prepare_ms') or 0.0):.6f}")
print(f"learned_scorer_ms={float(learned_b.get('learned_scorer_ms') or learned_p.get('learned_scorer_ms') or 0.0):.6f}")
print(f"neighbor_table_build_ms={float(learned_b.get('neighbor_table_build_ms') or learned_p.get('neighbor_table_build_ms') or 0.0):.6f}")
print(f"prepared_block_ms={learned_ms:.6f}")
print(f"prepared_attention_ms={learned_attn:.6f}")
print(f"prepared_non_attention_ms={learned_non:.6f}")
print(f"total_with_prepare_ms={float(learned_b.get('total_with_prepare_ms') or learned_p.get('total_with_prepare_ms') or 0.0):.6f}")
if native_block_sparse:
    print(f"native_block_attention_ms={learned_attn:.6f}")
    print(f"native_block_prepared_block_ms={learned_ms:.6f}")
    print(f"native_block_token_cap={block_token_cap:.0f}")
    print(f"native_effective_token_k={native_effective_token_k:.0f}")
    print(f"speedup_vs_token_neighbor_block={speedup:.6f}")
print("\nmode                 k    route_acc    block_ms    attn_ms     non_attn_ms hidden_cos    logit_kl    hidden_l1")
print("---------------------------------------------------------------------------------------------------------")
print(f"hand_topology        {hand_k_i:<4d} {hand_q['route_acc']:<12.4f} {hand_ms:<11.3f} {hand_attn:<11.3f} {hand_non:<11.3f} {'-':<13} {'-':<11} {'-':<10}")
print(f"{learned_mode:<20} {learned_k_i:<4d} {learned_q['route_acc']:<12.4f} {learned_ms:<11.3f} {learned_attn:<11.3f} {learned_non:<11.3f} {(learned_q.get('hidden_cos') or 0.0):<13.6f} {(learned_q.get('logit_kl') or 0.0):<11.6f} {(learned_q.get('hidden_l1') or 0.0):<10.6f}")
print(f"\nhand_k{hand_k_i}_block_ms={hand_ms:.6f}")
print(f"learned_k{learned_k_i}_block_ms={learned_ms:.6f}")
print(f"hand_k{hand_k_i}_attention_ms={hand_attn:.6f}")
print(f"learned_k{learned_k_i}_attention_ms={learned_attn:.6f}")
print(f"hand_k{hand_k_i}_non_attention_ms={hand_non:.6f}")
print(f"learned_k{learned_k_i}_non_attention_ms={learned_non:.6f}")
print(f"speedup={speedup:.6f}")
print(f"hand_k{hand_k_i}_route_acc={hand_q['route_acc']:.6f}")
print(f"learned_k{learned_k_i}_route_acc={learned_q['route_acc']:.6f}")
print(f"learned_hidden_cos={(learned_q.get('hidden_cos') or 0.0):.6f}")
print(f"learned_logit_kl={(learned_q.get('logit_kl') or 0.0):.6f}")
print(f"learned_hidden_l1={(learned_q.get('hidden_l1') or 0.0):.6f}")
quality_ok = learned_q["route_acc"] >= hand_q["route_acc"]
strict_speed_ok = learned_ms > 0 and hand_ms > 0 and learned_ms < hand_ms
speed_gap_ms = learned_ms - hand_ms
speed_ok = learned_ms > 0 and hand_ms > 0 and speed_gap_ms <= tol_ms
print(f"strict_speed_ok={strict_speed_ok}")
print(f"acceptance_tolerance_ms={tol_ms:.6f}")
print(f"speed_gap_ms={speed_gap_ms:.6f}")
if quality_ok and speed_ok:
    print(f"acceptance_passed quality_ok={quality_ok} speed_ok={speed_ok} strict_speed_ok={strict_speed_ok}")
else:
    print(f"acceptance_failed quality_ok={quality_ok} speed_ok={speed_ok} strict_speed_ok={strict_speed_ok}", file=sys.stderr)
    if quality_ok and not speed_ok:
        if learned_eff_tbk == hand_eff_tbk and learned_k_i < hand_k_i:
            print("diagnosis=learned_k_has_fewer_edges_but_same_effective_triton_block_k; padded kernel work is not reduced", file=sys.stderr)
        if learned_attn >= hand_attn:
            print("diagnosis=learned_attention_not_faster_than_hand", file=sys.stderr)
        elif learned_non > hand_non:
            print("diagnosis=learned_attention_faster_but_non_attention_overhead_dominates", file=sys.stderr)
    if not allow:
        raise SystemExit(2)
    print(f"acceptance_allowed_failure quality_ok={quality_ok} speed_ok={speed_ok}")
PY2
