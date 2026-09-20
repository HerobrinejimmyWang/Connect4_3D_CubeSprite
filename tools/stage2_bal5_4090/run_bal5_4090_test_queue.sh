#!/usr/bin/env bash
# BAL-5 Part A: three-phase throughput screen on 1x4090 + 20 vCPU.
#
# Phases measured: selfplay -> learner -> paired_gate, in one launch so the
# whole queue runs unattended.
#
#   selfplay / learner : tools/benchmark_bal5_machine.py at the adapted
#                        production topology (20 actors x 4 lanes).
#   paired_gate        : tools/validate_bal5_gate_single_gpu.py replaying the
#                        three committed gates through the role-control reuse
#                        path (evaluation_parallel_games=8, batch 32, 1 ms).
#
# Reference protocol on connect4_gpu_2608 (git 9fcb1bf, same gate-reuse path):
#   column-no-tail g77        31.6 min
#   column-serial-attn2 g77   34.6 min
#   winning-no-tail g52       66.5 min
#
# Gate outputs go to a fresh root because the validation tool requires a
# non-existent --output-root (the earlier machine benchmark root stays).
set -euo pipefail

repo=/root/bal5_4090_migration_20260920/repo
out=/root/bal5_4090_benchmarks/20260921
gate_out=/root/bal5_4090_benchmarks/20260921_gate_reuse
python=/root/miniconda3/bin/python

mkdir -p "$out"
test ! -e "$out/QUEUE_SUCCESS"
test ! -e "$out/QUEUE_FAILED"

failed() {
  code=$?
  printf 'exit_code=%s\nfailed_utc=%s\n' "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$out/QUEUE_FAILED"
  exit "$code"
}
trap failed ERR

cd "$repo"

echo "=== phase 0: syntax + focused regressions ==="
"$python" -m compileall -q training/v3 connect4_core tools
for focused_test in \
  test_training_v3_data.py \
  test_training_v3_actor_runtime.py \
  test_training_v3_evaluation_runtime.py \
  test_training_v3_formal_runner.py \
  test_training_v3_core.py
do
  echo "--- $focused_test"
  "$python" -m unittest discover -s test -p "$focused_test"
done

case1=training/runs/stage2/bal5/r1/runs/bal5_r1_column_no_tail_warm_fp32lr1e4_seed271828
case2=training/runs/stage2/bal5/r1/runs/bal5_r1_column_serial_attn2_warm_fp32lr1e4_seed271828
case3=training/runs/stage2/bal5/r1/runs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828

echo "=== phase 1+2: selfplay + learner at adapted topology 20x4 ==="
"$python" -B tools/benchmark_bal5_machine.py \
  --case "$case1:77" \
  --case "$case2:77" \
  --case "$case3:52" \
  --adapt-case-index 1 \
  --topologies 16x4,18x4,20x4 \
  --adapt-games 64 \
  --case-games 128 \
  --learner-steps 256 \
  --inference-batch-size 32 \
  --inference-timeout-ms 1.0 \
  --output-root "$out/machine_20x4"

echo "=== phase 3: paired gate replay through the reuse path ==="
"$python" -B tools/validate_bal5_gate_single_gpu.py \
  --case "$case1:77" \
  --case "$case2:77" \
  --case "$case3:52" \
  --evaluation-parallel-games 8 \
  --evaluation-inference-batch-size 32 \
  --evaluation-inference-batch-timeout-ms 1.0 \
  --output-root "$gate_out"

printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$out/QUEUE_SUCCESS"
trap - ERR
echo TEST_QUEUE_COMPLETE
