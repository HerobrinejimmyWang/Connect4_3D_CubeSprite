#!/usr/bin/env bash
# BAL-5 Part A: finish the machine measurement and hand off to the resume.
#
# Split of work (deliberate, to avoid measuring the same gate twice):
#   run_bal5_4090_gate_phase_completion.sh -> all three paired-gate cases,
#       written once under 20260921_gate_reuse_v3/<case>_g<gen>/. That is the
#       Phase 3 evidence for the machine verdict.
#   THIS script -> selfplay + learner at the adapted 20x4 topology, then the
#       QUEUE_SUCCESS handoff. It does NOT replay the gates again: the gate
#       round already in flight covers all three cases, and replaying them a
#       second time would add ~2 h of GPU time for no new information.
#
# The gate case is a TIMING probe here. Whether a replayed gate reproduces the
# archived one to the last game is a separate methodology question about MCTS
# determinism under parallel replay; it does not affect the training lineage and
# must not block the resume. Per-case verdicts are preserved verbatim in each
# comparison.json, and gate_case_failures is carried inside the marker.
set -euo pipefail

repo=/root/bal5_4090_migration_20260920/repo
out=/root/bal5_4090_benchmarks/20260921_rerun
queue_out=/root/bal5_4090_benchmarks/20260921
python=/root/miniconda3/bin/python

cd "$repo"

echo "=== guard: refuse to clobber existing markers ==="
test ! -e "$queue_out/QUEUE_SUCCESS"
test ! -e "$queue_out/QUEUE_FAILED"

echo "=== wait for the gate completion round to release the GPU ==="
while pgrep -f 'validate_bal5_gate_single_gpu' >/dev/null; do
  echo "  gate replay still running; waiting"
  sleep 60
done
echo "  GPU is free"

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

echo "=== phase 3 gate verdicts came from the gate completion round; summarising ==="
gate_failures=$(grep -c '"passed": false' /root/bal5_4090_benchmarks/20260921_gate_reuse_v3/*/*/comparison.json 2>/dev/null || echo 0)
echo "  gate_case_failures=$gate_failures"

printf 'completed_utc=%s\ngate_case_failures=%s\ngate_evidence_root=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gate_failures" \
  "/root/bal5_4090_benchmarks/20260921_gate_reuse_v3" > "$out/COMPLETED"
printf 'completed_utc=%s\ngate_case_failures=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gate_failures" > "$queue_out/QUEUE_SUCCESS"
echo "TEST_QUEUE_COMPLETE gate_case_failures=$gate_failures"
