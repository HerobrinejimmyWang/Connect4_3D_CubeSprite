#!/usr/bin/env bash
# Planned calibration pause for the connect4_gpu_2608 formal run.
#
# Mechanism (verified in this repo):
#   formal_runner.py:1470-1472  SIGINT/SIGTERM -> drain.handle
#   formal_runner.py:1557-1559  break at the END of a generation with
#                               stop_reason="drained_after_signal_<N>"
#   formal_runner.py:1563-1583  atomic run_manifest write + exit code 0
#   formal_runner.py:1594-1596  previous handlers restored; lock released
#
# The observed driver is a ONE-SHOT script (run_winning_serial_warm_oneshot.sh):
# it launches exactly one coordinator, records the outcome, and exits. It does not
# chain further cases, so stopping the coordinator at a generation boundary is
# sufficient and the driver needs no separate stop. The driver is left to observe
# the coordinator's clean exit so its status artifact stays truthful.
#
# Forbidden and not used: SIGKILL, repeated signals, signals to gate/eval workers,
# signals to the driver.
set -uo pipefail

REPO=/root/autodl-tmp/Connect4_3D_game_refactor
RUN=$REPO/training/runs/stage2/bal5/r1/runs/bal5_r1_winning_serial_attn2_warm_fp32lr1e4_seed271828
OUT=/root/bal5_planned_pause
mkdir -p "$OUT"

PID=$(pgrep -f 'training\.v3 run' | head -1)
if [[ -z "${PID:-}" ]]; then
  echo "no coordinator found" | tee "$OUT/pause_attempt.log"
  exit 3
fi

DRIVER=$(awk '{print $4}' "/proc/$PID/stat" 2>/dev/null)
DRIVER_CMD=$(tr '\0' ' ' < "/proc/$DRIVER/cmdline" 2>/dev/null || echo unknown)

echo "=== process tree BEFORE stop (evidence) ===" | tee "$OUT/pause_attempt.log"
ps -eo pid,ppid,pgid,etime,args | grep -E '[t]raining\.v3 run|[o]neshot|[m]ultiprocessing.spawn' \
  | cut -c1-150 >> "$OUT/pause_attempt.log"
echo "coordinator PID : $PID" | tee -a "$OUT/pause_attempt.log"
echo "driver PID      : $DRIVER ($DRIVER_CMD)" | tee -a "$OUT/pause_attempt.log"

GATE_CASES_BEFORE=$(ls -1 "$RUN/metrics/" 2>/dev/null | grep -c '^gate_g' || echo 0)
echo "gate_g*.json count before: $GATE_CASES_BEFORE" | tee -a "$OUT/pause_attempt.log"
echo "latest_generation before:" | tee -a "$OUT/pause_attempt.log"
cat "$RUN/manifests/latest_generation.json" >> "$OUT/pause_attempt.log"

REQ_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
REQ_EPOCH=$(date +%s)
printf '{\n  "schema": "connect4-bal5-planned-pause-v1",\n  "host": "connect4_gpu_2608",\n  "run_id": "bal5_r1_winning_serial_attn2_warm_fp32lr1e4_seed271828",\n  "signal": "SIGTERM",\n  "signal_number": 15,\n  "target": "coordinator",\n  "pid": %s,\n  "driver_pid": %s,\n  "requested_utc": "%s",\n  "repeated_signals": 0,\n  "sigkill_used": false,\n  "workers_signalled": false\n}\n' \
  "$PID" "$DRIVER" "$REQ_UTC" > "$OUT/pause_request.json"

echo "[$REQ_UTC] sending ONE SIGTERM to coordinator $PID" | tee -a "$OUT/pause_attempt.log"
kill -TERM "$PID"
echo "kill exit code: $?" | tee -a "$OUT/pause_attempt.log"

WAITED=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 10
  WAITED=$((WAITED + 10))
  if [[ $WAITED -ge 7200 ]]; then
    echo "coordinator alive after 7200s; NOT sending another signal" | tee -a "$OUT/pause_attempt.log"
    exit 4
  fi
done

# Let the one-shot driver finish writing its status artifact.
while kill -0 "$DRIVER" 2>/dev/null; do
  sleep 5
  WAITED=$((WAITED + 5))
  [[ $WAITED -ge 7400 ]] && break
done

EXIT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DRAIN=$(( $(date +%s) - REQ_EPOCH ))

{
  printf '{\n'
  printf '  "schema": "connect4-bal5-planned-pause-result-v1",\n'
  printf '  "host": "connect4_gpu_2608",\n'
  printf '  "signal": "SIGTERM",\n'
  printf '  "pid": %s,\n' "$PID"
  printf '  "requested_utc": "%s",\n' "$REQ_UTC"
  printf '  "exited_utc": "%s",\n' "$EXIT_UTC"
  printf '  "drain_seconds": %s,\n' "$DRAIN"
  printf '  "repeated_signals": 0,\n'
  printf '  "sigkill_used": false\n'
  printf '}\n'
} > "$OUT/pause_result.json"

echo "=== drain complete after ${DRAIN}s ==="
echo "--- manifest ---"
/root/miniconda3/bin/python -c "
import json
m=json.load(open('$RUN/run_manifest.json'))
fls=m.get('formal_loop_state',{})
print('status      :', m.get('status'))
print('stop_reason :', m.get('stop_reason'))
print('updated_at  :', m.get('updated_at'))
print('next_generation          :', fls.get('next_generation'))
print('train_positions_consumed :', fls.get('train_positions_consumed'))
print('accepted_model_id        :', fls.get('accepted_model_id'))
print('pending_candidate        :', fls.get('pending_candidate'))
"
echo "--- latest_generation pointer ---"
cat "$RUN/manifests/latest_generation.json"
echo
echo "--- driver status artifact ---"
cat "$REPO/.bal5_staging/bal5_r1_winning_serial_attn2_warm_fp32lr1e4_seed271828.oneshot.status.json" 2>/dev/null || echo "(absent)"
echo "--- coordinator lock ---"
ls -la "$RUN/manifests/coordinator.lock" 2>/dev/null || echo "lock absent (released)"
echo "--- remaining training processes ---"
pgrep -af 'training\.v3 run' || echo "(none)"
echo "--- remaining gate/actor children ---"
pgrep -af 'multiprocessing.spawn' | head -3 || echo "(none)"
