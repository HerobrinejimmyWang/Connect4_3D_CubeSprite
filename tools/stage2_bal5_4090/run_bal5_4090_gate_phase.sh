#!/usr/bin/env bash
# BAL-5 Part A phase 3 only: paired gate replay through the role-control reuse path.
#
# Separated from run_bal5_4090_test_queue.sh so a gate-phase failure can be
# re-measured without repeating the ~20 minutes of selfplay and learner points.
# Phases 1 and 2 already succeeded and their numbers stand.
set -euo pipefail

repo=/root/bal5_4090_migration_20260920/repo
gate_out=/root/bal5_4090_benchmarks/20260921_gate_reuse_v2
python=/root/miniconda3/bin/python

cd "$repo"

case1=training/runs/stage2/bal5/r1/runs/bal5_r1_column_no_tail_warm_fp32lr1e4_seed271828
case2=training/runs/stage2/bal5/r1/runs/bal5_r1_column_serial_attn2_warm_fp32lr1e4_seed271828
case3=training/runs/stage2/bal5/r1/runs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828

"$python" -B tools/validate_bal5_gate_single_gpu.py \
  --case "$case1:77" \
  --case "$case2:77" \
  --case "$case3:52" \
  --evaluation-parallel-games 8 \
  --evaluation-inference-batch-size 32 \
  --evaluation-inference-batch-timeout-ms 1.0 \
  --output-root "$gate_out"

echo GATE_PHASE_COMPLETE
