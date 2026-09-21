#!/usr/bin/env bash
# BAL-5 Part A phase 3 completion: replay the remaining gate cases.
#
# The first attempt used the strict tool default, which breaks out of the case
# list on the first mismatch. Case 1 matched exactly; case 2 differed in exactly
# one of 400 paired games, which shifted every downstream aggregate and tripped
# the strict equality verdict. Case 3 never ran.
#
# This script finishes the evidence set so the machine verdict rests on all three
# measured cases. Interpretation of the one-game divergence is a separate
# methodology question and is deliberately NOT encoded here: the tool's strict
# comparison output is preserved verbatim in each comparison.json.
#
# Order puts the two cases that still need measurement first, so that the totals
# and the machine verdict have measured data for all three cases.
set -euo pipefail

repo=/root/bal5_4090_migration_20260920/repo
gate_out=/root/bal5_4090_benchmarks/20260921_gate_reuse_v3
python=/root/miniconda3/bin/python

cd "$repo"

case1=training/runs/stage2/bal5/r1/runs/bal5_r1_column_no_tail_warm_fp32lr1e4_seed271828
case2=training/runs/stage2/bal5/r1/runs/bal5_r1_column_serial_attn2_warm_fp32lr1e4_seed271828
case3=training/runs/stage2/bal5/r1/runs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828

# --case is repeatable but the tool stops at the first mismatch, so run one case
# per invocation with its own output root and keep going regardless of verdicts.
run_case() {
  local label=$1
  local case_arg=$2
  local out=$3
  echo "=== gate case: $label ==="
  set +e
  "$python" -B tools/validate_bal5_gate_single_gpu.py \
    --case "$case_arg" \
    --evaluation-parallel-games 8 \
    --evaluation-inference-batch-size 32 \
    --evaluation-inference-batch-timeout-ms 1.0 \
    --output-root "$out"
  local code=$?
  set -e
  echo "=== case $label finished with exit code $code ==="
}

run_case "case3 winning-no-tail g52" "$case3:52" "$gate_out/case3_winning_no_tail_g52"
run_case "case1 column-no-tail g77" "$case1:77" "$gate_out/case1_column_no_tail_g77"
run_case "case2 column-serial-attn2 g77" "$case2:77" "$gate_out/case2_column_serial_attn2_g77"

echo GATE_PHASE_COMPLETE
