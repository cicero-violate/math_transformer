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
ALLOW_FAIL="${ALLOW_FAIL:-0}"
TMP_DIR="${TMP_DIR:-runs/benchmarks/learned_topology_tmp}"
mkdir -p "$TMP_DIR"

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
  --middle-bridge-width 1 --warmup 3 --iters "$BENCH_STEPS" --save-dir "$hand_dir" >/dev/null

"$PYTHON" -m src.eval --benchmark --profile-prepared-block --sizes "$BENCH_N" \
  --node-mode "$BENCH_NODE_MODE" --examples "$EXAMPLES" \
  --topology-mode learned_topology --learned-scorer-checkpoint "$SCORER" --learned-k "$LEARNED_K" \
  --fixed-k "$LEARNED_K" --max-neighbors "$LEARNED_K" --middle-bridge-width 1 \
  --warmup 3 --iters "$BENCH_STEPS" --save-dir "$learned_dir" >/dev/null

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
speedup=hand_ms/learned_ms if learned_ms>0 else 0.0
print("\nmode                 k    route_acc    block_ms    hidden_cos    logit_kl    hidden_l1")
print("--------------------------------------------------------------------------------")
print(f"hand_topology        {hand_k_i:<4d} {hand_q['route_acc']:<12.4f} {hand_ms:<11.3f} {'-':<13} {'-':<11} {'-':<10}")
print(f"learned_topology     {learned_k_i:<4d} {learned_q['route_acc']:<12.4f} {learned_ms:<11.3f} {(learned_q.get('hidden_cos') or 0.0):<13.6f} {(learned_q.get('logit_kl') or 0.0):<11.6f} {(learned_q.get('hidden_l1') or 0.0):<10.6f}")
print(f"\nhand_k{hand_k_i}_block_ms={hand_ms:.6f}")
print(f"learned_k{learned_k_i}_block_ms={learned_ms:.6f}")
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
    if not allow:
        raise SystemExit(2)
    print(f"acceptance_allowed_failure quality_ok={quality_ok} speed_ok={speed_ok}")
PY2
