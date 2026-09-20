#!/usr/bin/env bash
# Wait for the BAL-5 Part A test queue and the resume snapshot, then continue the
# winning-no-tail run to the agreed 5M train-position bound.
#
# Automation policy (explicit user decision, 2026-09-21): automated continuation
# is allowed, but every automated decision must leave an auditable receipt under
# /root/bal5_4090_resume/audit/. Three terminal outcomes are distinguished:
#
#   1. stability_pause  -> retried automatically exactly once with
#      --ack-stability-pause-through-train-positions, which records an explicit
#      acknowledgement of a behavioural regression alert. The receipt captures
#      why and with what bound. A second stability pause is NOT auto-acknowledged;
#      it stops for a human.
#   2. archive_required -> stops. Pruning is receipt-gated by contract and there
#      is no verified local receiver for this run, so a human must perform the
#      incremental archive -> verification -> receipt -> prune sequence.
#   3. any other terminal state -> stops and records the parsed status/reason.
#
# A gate inconclusive verdict is never an automated decision here: the runner
# extends the same paired opening evidence instead of accepting a candidate.
set -euo pipefail

repo=/root/bal5_4090_migration_20260920/repo
bench=/root/bal5_4090_benchmarks/20260921
resume_root=/root/bal5_4090_resume
audit=$resume_root/audit
python=/root/miniconda3/bin/python
base_config=$repo/training/runs/stage2/bal5/r1/configs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828.json
adapted_config=$repo/training/runs/stage2/bal5/r1/configs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828_1x4090.json
bound=5000000

mkdir -p "$resume_root" "$audit"

receipt() {
  # receipt <kind> <detail...>
  local kind=$1
  shift
  local stamp path
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  path="$audit/${stamp}-${kind}.json"
  {
    printf '{\n'
    printf '  "schema": "connect4-bal5-4090-resume-receipt-v1",\n'
    printf '  "time_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "kind": "%s",\n' "$kind"
    printf '  "run_id": "bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828",\n'
    printf '  "bound": %s,\n' "$bound"
    printf '  "repo_commit": "%s",\n' "$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf '  "detail": "%s"\n' "$*"
    printf '}\n'
  } > "$path"
  printf '[%s] RECEIPT %s -> %s: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$kind" "$path" "$*" \
    >> "$resume_root/audit.log"
}

printf 'status=waiting_for_test_and_resume_snapshot\nstarted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$resume_root/watcher.status"

while true; do
  if [[ -e "$bench/QUEUE_FAILED" ]]; then
    printf 'status=test_failed\ntime_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$resume_root/watcher.status"
    touch "$resume_root/NEEDS_ATTENTION"
    receipt test_queue_failed "Part A test queue wrote QUEUE_FAILED; resume withheld"
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
  --benchmark "$bench/machine_20x4/benchmark.json" \
  --output "$adapted_config" \
  > "$resume_root/config_adaptation.json"
receipt resume_config_adapted "topology adapted from benchmark; semantic config hash preserved"

printf 'status=training\nstarted_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$resume_root/watcher.status"

attempt=0
while true; do
  attempt=$((attempt + 1))
  command=(
    "$python" -B -m training.v3 run
    --config "$adapted_config"
    --resume
    --execute
    --max-train-positions "$bound"
  )
  if [[ $attempt -gt 1 ]]; then
    command+=(--ack-stability-pause-through-train-positions "$bound")
  fi
  set +e
  "${command[@]}" > "$resume_root/resume_result.log" 2>&1
  train_exit=$?
  set -e
  printf '%s\n' "$train_exit" > "$resume_root/training.exit_code"

  set +e
  "$python" -B tools/verify_bal5_resume_result.py \
    --log "$resume_root/resume_result.log" \
    --bound "$bound" \
    --status "$resume_root/final_status.json" \
    > "$resume_root/verification.log" 2>&1
  verify_exit=$?
  set -e
  verdict=$(cat "$resume_root/verification.log" 2>/dev/null || echo unparsed)

  if [[ $train_exit -eq 0 && $verify_exit -eq 0 ]]; then
    receipt training_complete "reached the ${bound}-position bound; verification=$verdict"
    printf 'status=complete\ntime_utc=%s\nattempts=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt" > "$resume_root/watcher.status"
    touch "$resume_root/TRAINING_COMPLETE"
    exit 0
  fi

  if [[ "$verdict" == stability_pause* ]]; then
    if [[ $attempt -eq 1 ]]; then
      receipt stability_pause_auto_ack \
        "first stability pause auto-acknowledged once with --ack-stability-pause-through-train-positions ${bound}; verification=$verdict; train_exit=$train_exit"
      sleep 60
      continue
    fi
    receipt stability_pause_stop \
      "second stability pause in the same run; refusing a second automated acknowledgement; verification=$verdict; train_exit=$train_exit"
    printf 'status=needs_attention\nreason=stability_pause_twice\ntraining_exit=%s\nverification_exit=%s\ntime_utc=%s\n' \
      "$train_exit" "$verify_exit" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$resume_root/watcher.status"
    touch "$resume_root/NEEDS_ATTENTION"
    exit 20
  fi

  receipt training_stopped_other \
    "terminal state requires a human; verification=$verdict; train_exit=$train_exit; verification_exit=$verify_exit"
  printf 'status=needs_attention\nreason=unexpected_terminal_state\ntraining_exit=%s\nverification_exit=%s\ntime_utc=%s\n' \
    "$train_exit" "$verify_exit" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$resume_root/watcher.status"
  touch "$resume_root/NEEDS_ATTENTION"
  exit 20
done