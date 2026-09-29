#!/usr/bin/env bash
# Planned calibration pause: request ONE supported drain signal and record evidence.
#
# Mechanism (verified in this repo, not assumed):
#   training/v3/formal_runner.py:1470-1472 registers SIGINT/SIGTERM handlers
#   training/v3/formal_runner.py:1557-1559 breaks with stop_reason
#       "drained_after_signal_<N>" at the end of a generation
#   training/v3/formal_runner.py:1563-1583 writes status="stopped_at_safe_boundary"
#       plus formal_loop_state atomically, then returns exit code 0
#   training/v3/formal_runner.py:1594-1596 restores the previous handlers on exit
#
# Therefore SIGTERM never interrupts selfplay/learner/gate mid-flight: the request
# is only observed after the current generation has committed.
#
# Forbidden and not used: SIGKILL, repeated signals, signals to worker children.
set -uo pipefail

RUN=/root/autodl-tmp/Connect4_3D_game_refactor/training/runs/stage2/bal5/r1/runs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828
RESUME=/root/bal5_4090_resume
OUT=$RESUME/planned_pause
mkdir -p "$OUT"

PID=$(pgrep -f 'training\.v3 run' | head -1)
if [[ -z "${PID:-}" ]]; then
  echo "no coordinator process found; nothing to pause" | tee "$OUT/pause_attempt.log"
  exit 3
fi
echo "coordinator PID: $PID" | tee "$OUT/pause_attempt.log"
ps -o pid,ppid,etime,lstart,args -p "$PID" --no-headers >> "$OUT/pause_attempt.log"

# Record the atomic generation state immediately BEFORE the request.
echo "--- pre-request state ---" | tee -a "$OUT/pause_attempt.log"
cat "$RUN/manifests/latest_generation.json" >> "$OUT/pause_attempt.log"

REQ_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
REQ_EPOCH=$(date +%s)
printf 'signal_requested=%s\nsignal=SIGTERM\nsignal_number=15\nrequested_utc=%s\npid=%s\n' \
  "SIGTERM" "$REQ_UTC" "$PID" > "$OUT/pause_request.json"
echo "[$REQ_UTC] sending ONE SIGTERM to $PID (requesting generation-boundary drain)"

kill -TERM "$PID"
SEND_CODE=$?
echo "kill exit code: $SEND_CODE" | tee -a "$OUT/pause_attempt.log"

# Wait for the coordinator to drain and exit. No second signal, ever.
WAITED=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 10
  WAITED=$((WAITED + 10))
  if [[ $WAITED -ge 3600 ]]; then
    echo "coordinator still alive after 3600s; NOT sending another signal" | tee -a "$OUT/pause_attempt.log"
    printf 'status=drain_timeout\nwaited_seconds=%s\n' "$WAITED" > "$OUT/pause_result.json"
    exit 4
  fi
done

EXIT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EXIT_EPOCH=$(date +%s)
DRAIN_SECONDS=$((EXIT_EPOCH - REQ_EPOCH))

{
  printf '{\n'
  printf '  "schema": "connect4-bal5-planned-pause-v1",\n'
  printf '  "signal": "SIGTERM",\n'
  printf '  "signal_number": 15,\n'
  printf '  "pid": %s,\n' "$PID"
  printf '  "requested_utc": "%s",\n' "$REQ_UTC"
  printf '  "exited_utc": "%s",\n' "$EXIT_UTC"
  printf '  "drain_seconds": %s,\n' "$DRAIN_SECONDS"
  printf '  "repeated_signals": 0,\n'
  printf '  "sigkill_used": false\n'
  printf '}\n'
} > "$OUT/pause_result.json"

echo "=== drain complete after ${DRAIN_SECONDS}s ==="
echo "--- stop_reason / status from run_manifest ---"
/root/miniconda3/bin/python -c "
import json
m = json.load(open('$RUN/run_manifest.json'))
fls = m.get('formal_loop_state', {})
print('status           :', m.get('status'))
print('stop_reason      :', m.get('stop_reason'))
print('next_generation  :', fls.get('next_generation'))
print('train_positions  :', fls.get('train_positions_consumed'))
print('accepted_model_id:', fls.get('accepted_model_id'))
print('pending_candidate:', fls.get('pending_candidate'))
print('next_game_id     :', fls.get('next_game_id'))
print('updated_at       :', m.get('updated_at'))
" | tee -a "$OUT/pause_attempt.log"

echo "--- latest_generation pointer AFTER drain ---"
cat "$RUN/manifests/latest_generation.json" | tee -a "$OUT/pause_attempt.log"
echo
echo "--- coordinator lock state ---"
if [[ -e "$RUN/manifests/coordinator.lock" ]]; then
  echo "LOCK FILE STILL PRESENT (expected: released)"
else
  echo "lock file absent (coordinator released it)"
fi
echo "--- remaining training processes ---"
pgrep -af 'training\.v3 run' || echo "(none)"
