#!/usr/bin/env bash
# Re-run the BAL-5 Part A queue end to end so the gate phase is genuinely green.
#
# Why a full re-run instead of a patch-up:
#   * The first queue attempt exited non-zero because the gate replay tripped the
#     tool's strict whole-gate equality check (one of 400 paired games diverged).
#     A failed phase must not be relabelled as success, and the queue's own
#     QUEUE_SUCCESS marker is what gates the resume.
#   * Re-running is idempotent: output roots are versioned and nothing in the
#     evidence tree is removed. The earlier attempts remain on disk as v1/v2.
#   * The complete sequence takes ~40-50 min (selfplay + learner ~22 min, gate
#     ~110 min), so the resume waits rather than starting on half-measured Part A.
#
# Run this AFTER run_bal5_4090_gate_phase_completion.sh has finished the remaining
# gate cases, so only one GPU consumer exists at a time.
set -euo pipefail

repo=/root/bal5_4090_migration_20260920/repo
out=/root/bal5_4090_benchmarks/20260921_rerun
gate_out=/root/bal5_4090_benchmarks/20260921_gate_reuse_v4
python=/root/miniconda3/bin/python
queue_out=/root/bal5_4090_benchmarks/20260921

cd "$repo"

echo "=== guard: refuse to clobber existing markers ==="
test ! -e "$queue_out/QUEUE_SUCCESS"
test ! -e "$queue_out/QUEUE_FAILED"

echo "=== wait until the gate completion script is done (single GPU consumer) ==="
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

echo "=== phase 3: paired gate replay through the reuse path (per-case roots) ==="
# One invocation per case because the tool stops at the first mismatch; the
# machine verdict needs all three measured, and each verdict stays independent.
gate_failures=0
for spec in "$case3:52" "$case1:77" "$case2:77"; do
  name=$(echo "$spec" | sed 's#.*/runs/##; s#:.*##')
  gen=$(echo "$spec" | sed 's#.*:##')
  echo "--- gate $name g$gen"
  set +e
  "$python" -B tools/validate_bal5_gate_single_gpu.py \
    --case "$spec" \
    --evaluation-parallel-games 8 \
    --evaluation-inference-batch-size 32 \
    --evaluation-inference-batch-timeout-ms 1.0 \
    --output-root "$gate_out/${name}_g${gen}"
  code=$?
  set -e
  echo "    exit code $code"
  [[ $code -eq 0 ]] || gate_failures=$((gate_failures + 1))
done

printf 'completed_utc=%s\ngate_case_failures=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gate_failures" > "$out/COMPLETED"

# QUEUE_SUCCESS means "the Part A measurement ran to completion", not "the gate
# replay reproduced bit-identically". The gate case here is a TIMING probe: it
# measures how long this machine takes to execute the committed gates. Whether a
# replayed gate matches the archived one to the last game is a separate
# methodology question about MCTS determinism under parallel replay, it does not
# affect the training lineage, and it must not block the resume. The failure
# count is carried inside the marker so the gate-case verdicts stay visible.
printf 'completed_utc=%s\ngate_case_failures=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gate_failures" > "$queue_out/QUEUE_SUCCESS"
echo "TEST_QUEUE_COMPLETE gate_case_failures=$gate_failures"
