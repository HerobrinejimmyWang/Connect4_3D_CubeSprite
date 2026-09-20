#!/usr/bin/env bash
set -euo pipefail

repo=/root/bal5_4090_migration_20260920/repo
bench=/root/bal5_4090_benchmarks/20260921
resume_root=/root/bal5_4090_resume
python=/root/miniconda3/bin/python
base_config=$repo/training/runs/stage2/bal5/r1/configs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828.json
adapted_config=$repo/training/runs/stage2/bal5/r1/configs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828_1x4090.json

mkdir -p "$resume_root"
printf 'status=waiting_for_test_and_resume_snapshot\nstarted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$resume_root/watcher.status"

while true; do
  if [[ -e "$bench/QUEUE_FAILED" ]]; then
    printf 'status=test_failed\ntime_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$resume_root/watcher.status"
    exit 10
  fi
  if [[ -e "$bench/QUEUE_SUCCESS" && -e "$resume_root/RESUME_READY" ]]; then
    break
  fi
  sleep 30
done

cd "$repo"
printf 'status=preparing_resume\ntime_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$resume_root/watcher.status"

"$python" -B tools/prepare_bal5_resume_config.py \
  --base-config "$base_config" \
  --benchmark "$bench/machine/benchmark.json" \
  --output "$adapted_config" \
  > "$resume_root/config_adaptation.json"

"$python" -B -m training.v3 run --config "$adapted_config" \
  > "$resume_root/resume_plan.log" 2>&1

printf 'status=training\nstarted_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$resume_root/watcher.status"

set +e
"$python" -B -m training.v3 run \
  --config "$adapted_config" \
  --resume \
  --execute \
  --max-train-positions 5000000 \
  > "$resume_root/resume_result.log" 2>&1
train_exit=$?
set -e
printf '%s\n' "$train_exit" > "$resume_root/training.exit_code"

set +e
"$python" -B tools/verify_bal5_resume_result.py \
  --log "$resume_root/resume_result.log" \
  --bound 5000000 \
  --status "$resume_root/final_status.json"
verify_exit=$?
set -e

if [[ $train_exit -eq 0 && $verify_exit -eq 0 ]]; then
  printf 'status=complete\ntime_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$resume_root/watcher.status"
  touch "$resume_root/TRAINING_COMPLETE"
  exit 0
fi

printf 'status=needs_attention\ntraining_exit=%s\nverification_exit=%s\ntime_utc=%s\n' \
  "$train_exit" "$verify_exit" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$resume_root/watcher.status"
touch "$resume_root/NEEDS_ATTENTION"
exit 20
