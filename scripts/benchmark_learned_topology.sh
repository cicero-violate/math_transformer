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
TMP_DIR="${TMP_DIR:-runs/benchmarks/learned_topology_tmp}"
mkdir -p "$TMP_DIR"

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

quality_log="$TMP_DIR/quality.log"
hand_dir="$TMP_DIR/hand"
learned_dir="$TMP_DIR/learned"
rm -rf "$hand_dir" "$learned_dir"
mkdir -p "$hand_dir" "$learned_dir"

"$PYTHON" -m src.eval --quality --examples "$EXAMPLES" --checkpoint "$CHECKPOINT" \
  --topology-mode middle_preserving_topk --fixed-k "$HAND_K" --middle-bridge-width 1 \
  --quality-k "$HAND_K" --quality-device "$DEVICE" \
  --learned-scorer-checkpoint "$SCORER" --learned-k "$LEARNED_K" | tee "$quality_log"

"$PYTHON" -m src.eval --benchmark --profile-prepared-block --sizes "$BENCH_N" \
  --node-mode "$BENCH_NODE_MODE" --examples "$EXAMPLES" \
  --topology-mode middle_preserving_topk --fixed-k "$HAND_K" --max-neighbors "$HAND_K" \
  --middle-bridge-width 1 --warmup 3 --iters "$BENCH_STEPS" \
  --benchmark-seed "$BENCH_SEED" "${HAND_TRITON_ARGS[@]}" --save-dir "$hand_dir" >/dev/null

"$PYTHON" -m src.eval --benchmark --profile-prepared-block --sizes "$BENCH_N" \
  --node-mode "$BENCH_NODE_MODE" --examples "$EXAMPLES" \
  --topology-mode learned_topology --learned-scorer-checkpoint "$SCORER" --learned-k "$LEARNED_K" \
  --fixed-k "$LEARNED_K" --max-neighbors "$LEARNED_K" --middle-bridge-width 1 \
  --warmup 3 --iters "$BENCH_STEPS" --benchmark-seed "$BENCH_SEED" \
  "${LEARNED_TRITON_ARGS[@]}" --save-dir "$learned_dir" >/dev/null

"$PYTHON" - "$quality_log" "$hand_dir" "$learned_dir" "$HAND_K" "$LEARNED_K" "$ALLOW_FAIL" <<'PY2'
from __future__ import annotations
import json, re, sys
from pathlib import Path
quality_log, hand_dir, learned_dir, hand_k, learned_k, allow_fail = sys.argv[1:]
hand_k_i = int(hand_k); learned_k_i = int(learned_k); allow = allow_fail == "1"
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
hand_q=pick("topology_only",hand_k_i); learned_q=pick("learned_topology",learned_k_i)
hand_b=latest_json(hand_dir); learned_b=latest_json(learned_dir)
hand_ms=float(hand_b.get("prepared_static_sparse_block_ms") or 0.0); learned_ms=float(learned_b.get("prepared_static_sparse_block_ms") or 0.0)
hand_attn=float(hand_b.get("prepared_static_sparse_attention_ms") or 0.0); learned_attn=float(learned_b.get("prepared_static_sparse_attention_ms") or 0.0)
hand_non=float(hand_b.get("prepared_static_sparse_non_attention_ms") or 0.0); learned_non=float(learned_b.get("prepared_static_sparse_non_attention_ms") or 0.0)
hand_tbd=hand_b.get("triton_block_d"); hand_tbk=hand_b.get("triton_block_k")
learned_tbd=learned_b.get("triton_block_d"); learned_tbk=learned_b.get("triton_block_k")
hand_eff_tbk=hand_b.get("effective_triton_block_k"); learned_eff_tbk=learned_b.get("effective_triton_block_k")
speedup=hand_ms/learned_ms if learned_ms>0 else 0.0
print(f"\nhand_triton_block_d={hand_tbd} hand_triton_block_k={hand_tbk} hand_effective_triton_block_k={hand_eff_tbk}")
print(f"learned_triton_block_d={learned_tbd} learned_triton_block_k={learned_tbk} learned_effective_triton_block_k={learned_eff_tbk}")
print("\nmode                 k    route_acc    block_ms    attn_ms     non_attn_ms hidden_cos    logit_kl    hidden_l1")
print("---------------------------------------------------------------------------------------------------------")
print(f"hand_topology        {hand_k_i:<4d} {hand_q['route_acc']:<12.4f} {hand_ms:<11.3f} {hand_attn:<11.3f} {hand_non:<11.3f} {'-':<13} {'-':<11} {'-':<10}")
print(f"learned_topology     {learned_k_i:<4d} {learned_q['route_acc']:<12.4f} {learned_ms:<11.3f} {learned_attn:<11.3f} {learned_non:<11.3f} {(learned_q.get('hidden_cos') or 0.0):<13.6f} {(learned_q.get('logit_kl') or 0.0):<11.6f} {(learned_q.get('hidden_l1') or 0.0):<10.6f}")
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
speed_ok = learned_ms > 0 and hand_ms > 0 and learned_ms < hand_ms
if quality_ok and speed_ok:
    print(f"acceptance_passed quality_ok={quality_ok} speed_ok={speed_ok}")
else:
    print(f"acceptance_failed quality_ok={quality_ok} speed_ok={speed_ok}", file=sys.stderr)
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
