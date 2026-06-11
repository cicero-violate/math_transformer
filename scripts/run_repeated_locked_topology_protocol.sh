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

PROTOCOL_CONFIG="${PROTOCOL_CONFIG:-configs/learned_topology_locked_protocol.json}"
# shellcheck disable=SC1090
eval "$($PYTHON -m src.topology_protocol --config "$PROTOCOL_CONFIG" --emit-shell-env)"

REPEAT_N="${REPEAT_N:-5}"
BASE_BENCH_SEED="${BASE_BENCH_SEED:-$BENCH_SEED}"
HAND_BASELINE_K="${HAND_BASELINE_K:-4}"
CURRENT_CHAMPION_K="${CURRENT_CHAMPION_K:-8}"
POLICY_LEARNED_KS="${POLICY_LEARNED_KS:-4,$CURRENT_CHAMPION_K}"
REPEATED_TMP_DIR="${REPEATED_TMP_DIR:-runs/benchmarks/repeated_locked_protocol_tmp}"
REPEATED_ARTIFACT_JSONL="${REPEATED_ARTIFACT_JSONL:-runs/benchmarks/repeated_locked_protocol_artifacts.jsonl}"
REPEATED_SUMMARY_JSON="${REPEATED_SUMMARY_JSON:-runs/benchmarks/repeated_locked_speed_summary.json}"
REPEATED_SUMMARY_CSV="${REPEATED_SUMMARY_CSV:-runs/benchmarks/repeated_locked_speed_summary.csv}"
REPEATED_REQUIRED_POLICIES="${REPEATED_REQUIRED_POLICIES:-dense_full,hand_k4,learned_k4,current_champion_k8}"
REPEATED_MIN_PASS_RATE="${REPEATED_MIN_PASS_RATE:-0.75}"

mkdir -p "$REPEATED_TMP_DIR" "$(dirname "$REPEATED_ARTIFACT_JSONL")"
: > "$REPEATED_ARTIFACT_JSONL"

echo "repeated_locked_protocol_config=$LOCKED_PROTOCOL_CONFIG"
echo "repeated_locked_protocol_hash=$LOCKED_PROTOCOL_HASH"
echo "repeated_n=$REPEAT_N hand_baseline_k=$HAND_BASELINE_K learned_ks=$POLICY_LEARNED_KS base_seed=$BASE_BENCH_SEED"

total_runs=0
failed_runs=0

IFS=',' read -r -a learned_ks <<< "$POLICY_LEARNED_KS"
for ((i=0; i<REPEAT_N; i++)); do
  seed=$((BASE_BENCH_SEED + i))
  for learned_k_raw in "${learned_ks[@]}"; do
    learned_k="${learned_k_raw//[[:space:]]/}"
    [[ -z "$learned_k" ]] && continue
    run_tmp="$REPEATED_TMP_DIR/run_${i}_learned_k${learned_k}"
    mkdir -p "$run_tmp"
    echo "repeated_run=$i bench_seed=$seed hand_k=$HAND_BASELINE_K learned_k=$learned_k"
    ((total_runs++)) || true
    bench_exit=0
    env HAND_K="$HAND_BASELINE_K" LEARNED_K="$learned_k" BENCH_SEED="$seed" TMP_DIR="$run_tmp" ARTIFACT_JSON="$run_tmp/benchmark_artifact.json" ARTIFACT_JSONL="$REPEATED_ARTIFACT_JSONL" ALLOW_FAIL="${ALLOW_FAIL:-1}" scripts/benchmark_learned_topology.sh || bench_exit=$?
    if [[ $bench_exit -ne 0 ]]; then
      ((failed_runs++)) || true
      echo "repeated_run_failed run=$i learned_k=$learned_k exit_code=$bench_exit"
    fi
  done
done

echo "repeated_protocol_runs_total=$total_runs"
echo "repeated_protocol_runs_failed=$failed_runs"
echo "repeated_protocol_runs_succeeded=$((total_runs - failed_runs))"

# Always run the aggregator — it writes a summary even when some runs failed.
agg_exit=0
"$PYTHON" -m src.locked_speed_aggregator   --artifacts "$REPEATED_ARTIFACT_JSONL"   --json-out "$REPEATED_SUMMARY_JSON"   --csv-out "$REPEATED_SUMMARY_CSV"   --required-policies "$REPEATED_REQUIRED_POLICIES"   --baseline-policy "hand_k4"   --tolerance-ms "$ACCEPTANCE_TOL_MS"   --min-pass-rate "$REPEATED_MIN_PASS_RATE" || agg_exit=$?
echo "repeated_protocol_aggregator_exit=$agg_exit"
echo "repeated_summary_json=$REPEATED_SUMMARY_JSON"
echo "repeated_summary_csv=$REPEATED_SUMMARY_CSV"
if [[ $failed_runs -gt 0 ]]; then
  echo "repeated_protocol_partial_failure=True failed=$failed_runs total=$total_runs"
fi
exit $agg_exit
