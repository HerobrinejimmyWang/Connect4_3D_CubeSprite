#!/usr/bin/env bash
# BAL-5 16M-class topology calibration for ONE machine. Single queue launch.
#
# Usage:  run_bal5_topology_calibration.sh <profile> <timestamp>
#   profile: 4090      -> 1x RTX 4090 24G, 20-vCPU quota, devices cuda:0
#            3080ti    -> 2x RTX 3080 Ti 12G, 40-vCPU quota, devices cuda:0,cuda:1
#   timestamp: e.g. 20260921T1300Z; output goes to a fresh, non-overwritable dir
#
# Contract (fixed by the requester, not negotiable by this script):
#   * ONE checkpoint: winning-no-tail g57, verified by SHA-256 on both machines
#   * mcts_lanes_per_actor = 4  (lanes=6 is historical reference only)
#   * 256/32 full/fast simulations, openings/seed, temperature/noise, FP32,
#     batch 256, inference_batch_size 32, timeout 1 ms -- all inherited untouched
#   * only the operational topology (actor count, device mapping) is swept
#
# Exit markers: SUCCESS only when every acceptance condition held; otherwise FAILED
# plus a REASON file. Existing results are never deleted or overwritten.
set -uo pipefail

PROFILE=${1:?profile required: 4090|3080ti}
STAMP=${2:?timestamp required, e.g. 20260921T1300Z}

REPO=/root/bal5_calibration/repo
RUN_ROOT=$REPO/training/runs/stage2/bal5/r1/runs
PY=/root/miniconda3/bin/python
ROOT=/root/bal5_calibration/$STAMP

# Per-profile target. Calibration must use the checkpoint the run ACTUALLY paused
# at, so the run's own architecture/state is what gets measured. Both machines
# still verify a SHA-256 before any GPU work.
case "$PROFILE" in
  4090)
    DEVICES="cuda:0"
    SWEEP="20 24 28 32 36 40 48"
    RUN=$RUN_ROOT/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828
    CKPT=$RUN/checkpoints/g000072-s00017611.pt
    CKPT_SHA=22ed45ccd1257f3f18f7f0f9183755a0158b0bf2c54e672d174642bb1b1b791a
    GEN=72
    ;;
  3080ti)
    DEVICES="cuda:0,cuda:1"
    SWEEP="32 40 48 56 64 72"
    RUN=$RUN_ROOT/bal5_r1_winning_serial_attn2_warm_fp32lr1e4_seed271828
    CKPT=$RUN/checkpoints/g000032-s00007875.pt
    CKPT_SHA=c30ae51e3f08561dca220328e061c6df2618500d622c81c178abb651e46a1961
    GEN=32
    ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac
CONFIRM_GAMES=256
CONFIRM_REPS=3

mkdir -p "$ROOT" || exit 2
if [[ -e "$ROOT/SUCCESS" || -e "$ROOT/FAILED" ]]; then
  echo "refusing to reuse an already-finalised output root: $ROOT" >&2; exit 2
fi

fail() {
  local reason=$1
  printf '%s\n' "$reason" > "$ROOT/REASON"
  printf '%s\n' "$reason" > "$ROOT/FAILED"
  echo "CALIBRATION FAILED: $reason"
  exit 1
}

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

########################################
log "profile=$PROFILE devices=$DEVICES stamp=$STAMP"

# Cross-machine gate. The requester's rule is that calibration starts only after
# the winning-no-tail 5M resume reaches its terminal boundary, verification
# completes, and the GPU is idle. That resume runs on exactly ONE host, so:
#   * locally: wait for our own training process to disappear
#   * remotely: if a peer host runs the resume, wait for its terminal MARKER
#     rather than for its process to vanish (a vanished process without the
#     marker means it stopped for a human, which must block calibration).
# A peer that is not reachable over ssh is treated as "no peer": only the host
# that actually runs the resume needs this gate, and the other host does not
# have that ssh alias configured.
PEER_HOST=exp_260921
peer_state() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$PEER_HOST" \
    'if [ -e /root/bal5_4090_resume/TRAINING_COMPLETE ]; then echo DONE;
     elif [ -e /root/bal5_4090_resume/NEEDS_ATTENTION ]; then echo ATTENTION;
     elif pgrep -f "training\.v3 run" >/dev/null; then echo RUNNING;
     else echo UNKNOWN; fi' 2>/dev/null
}

log "waiting for a peer-host resume to finish, if one exists"
while true; do
  STATE=$(peer_state)
  if [[ -z "$STATE" ]]; then
    log "  no reachable peer resume host; skipping the cross-machine gate"
    break
  fi
  case "$STATE" in
    DONE)        log "  peer resume reported complete"; break ;;
    ATTENTION)   fail "peer resume stopped and needs attention; refusing to calibrate" ;;
    *)           log "  peer resume state=$STATE; waiting 120s"; sleep 120 ;;
  esac
done

log "waiting for the local GPU(s) to be idle"
while pgrep -f 'training\.v3 run' >/dev/null; do
  log "  local resume still running; waiting 120s"
  sleep 120
done
while pgrep -f 'validate_bal5_gate_single_gpu' >/dev/null; do
  log "  a gate replay is still running; waiting 60s"
  sleep 60
done
log "GPUs idle; proceeding"

########################################
log "=== precondition verification ==="
[[ -f "$CKPT" ]] || fail "checkpoint missing: $CKPT"
ACTUAL_SHA=$(sha256sum "$CKPT" | cut -d' ' -f1)
[[ "$ACTUAL_SHA" == "$CKPT_SHA" ]] || fail "checkpoint SHA256 mismatch: $ACTUAL_SHA != $CKPT_SHA"
[[ -f "$RUN/manifests/generations/g$(printf '%06d' "$GEN").json" ]] || fail "g$GEN generation manifest missing"
# The paused manifest must record the generation we intend to measure, and the
# run must sit on a safe boundary (no in-flight coordinator).
/root/miniconda3/bin/python - "$RUN/run_manifest.json" "$GEN" <<'PY' || fail "paused manifest is not at the expected safe boundary"
import json, sys
manifest = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
loop = manifest.get("formal_loop_state", {})
status = manifest.get("status")
reason = str(manifest.get("stop_reason") or "")
next_generation = int(loop.get("next_generation", -1))
ok = (
    status == "stopped_at_safe_boundary"
    and reason.startswith("drained_after_signal_")
    and next_generation == expected + 1
)
print("status=%s stop_reason=%s next_generation=%s expected_next=%s" % (
    status, reason, next_generation, expected + 1))
sys.exit(0 if ok else 1)
PY

CFG_SHA=$(sha256sum "$RUN/resolved_config.json" | cut -d' ' -f1)
GIT_COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
GIT_DIRTY=$(git -C "$REPO" status --short 2>/dev/null | tr '\n' ';')

{
  echo "{"
  echo "  \"schema\": \"connect4-bal5-calibration-manifest-v1\","
  echo "  \"profile\": \"$PROFILE\","
  echo "  \"stamp\": \"$STAMP\","
  echo "  \"generated_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"checkpoint\": \"$CKPT\","
  echo "  \"checkpoint_sha256\": \"$ACTUAL_SHA\","
  echo "  \"checkpoint_generation\": $GEN,"
  echo "  \"resolved_config_sha256\": \"$CFG_SHA\","
  echo "  \"git_commit\": \"$GIT_COMMIT\","
  echo "  \"git_dirty\": \"$GIT_DIRTY\","
  echo "  \"devices\": \"$DEVICES\","
  echo "  \"sweep_points\": \"$SWEEP\","
  echo "  \"confirm_games\": $CONFIRM_GAMES,"
  echo "  \"confirm_reps\": $CONFIRM_REPS"
  echo "}"
} > "$ROOT/manifest.json"
log "manifest written (checkpoint sha matches, git $GIT_COMMIT)"

########################################
log "=== shared tool hashes (must match on both machines) ==="
sha256sum "$REPO/tools/benchmark_bal5_topology_point.py" \
         "$REPO/tools/benchmark_bal5_learner_point.py" \
         "$REPO/tools/validate_bal5_gate_single_gpu.py" > "$ROOT/tool_hashes.txt"
cat "$ROOT/tool_hashes.txt"

########################################
log "=== phase 1: self-play 64-game sweep ==="
for ACTORS in $SWEEP; do
  OUT=$ROOT/sweep/actors_$ACTORS
  mkdir -p "$ROOT/sweep"
  [[ -e "$OUT" ]] && fail "sweep output already exists: $OUT"
  log "--- sweep actors=$ACTORS games=64"
  set +e
  "$PY" -B "$REPO/tools/benchmark_bal5_topology_point.py" \
    --run-dir "$RUN" --checkpoint "$CKPT" --expected-checkpoint-sha256 "$CKPT_SHA" \
    --generation $GEN --actors "$ACTORS" --devices "$DEVICES" --games 64 \
    --output-root "$OUT" --label "sweep_actors${ACTORS}" 2>&1 | tee "$OUT.log"
  code=$?
  set -e
  [[ $code -eq 0 ]] || fail "sweep point actors=$ACTORS failed with exit $code (see $OUT.log)"
done

log "=== sweep summary ==="
"$PY" - "$ROOT" <<'PYEOF'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(root, "sweep", "actors_*", "point.json"))):
    d = json.load(open(path))
    rows.append({
        "actors": d["semantics"]["actor_processes"],
        "games_per_second": d["result"]["games_per_second"],
        "simulations_per_second": d["result"]["simulations_per_second"],
        "raw_positions_per_second": d["result"]["raw_positions_per_second"],
        "gpu_util_mean": d["resources"].get("gpu_util_mean"),
        "cpu_util_pct_of_quota": d["resources"].get("cpu_util_percent_of_quota"),
        "cpu_throttled_ratio": d["resources"].get("cpu_throttled_period_ratio"),
        "memory_used_mib_max": d["resources"].get("memory_used_mib_max"),
        "mean_inference_batch": d["result"]["mean_inference_batch"],
    })
best = max(rows, key=lambda r: r["simulations_per_second"])
json.dump({"points": rows, "best": best}, open(os.path.join(root, "sweep_summary.json"), "w"),
          indent=2, sort_keys=True)
print(json.dumps(best, sort_keys=True))
PYEOF

BEST_ACTORS=$("$PY" -c "
import json,sys
print(json.load(open('$ROOT/sweep_summary.json'))['best']['actors'])
")
log "best sweep point: actors=$BEST_ACTORS"

# Extension rule: only if the best point sits on the sweep boundary do we probe
# one step beyond it. Two consecutive non-improvements already end the sweep
# because the best would then not be the boundary maximum.
LAST=$(echo "$SWEEP" | awk '{print $NF}')
if [[ "$BEST_ACTORS" == "$LAST" ]]; then
  EXTRA=$(( LAST + 8 ))
  log "best point is at the boundary; extending once to actors=$EXTRA"
  OUT=$ROOT/sweep/actors_$EXTRA
  mkdir -p "$ROOT/sweep"
  if [[ ! -e "$OUT" ]]; then
    set +e
    "$PY" -B "$REPO/tools/benchmark_bal5_topology_point.py" \
      --run-dir "$RUN" --checkpoint "$CKPT" --expected-checkpoint-sha256 "$CKPT_SHA" \
      --generation $GEN --actors "$EXTRA" --devices "$DEVICES" --games 64 \
      --output-root "$OUT" --label "sweep_actors${EXTRA}_extension" 2>&1 | tee "$OUT.log"
    code=$?
    set -e
    [[ $code -eq 0 ]] || fail "extension point actors=$EXTRA failed with exit $code"
  fi
  "$PY" -c "
import glob, json, os
root='$ROOT'
rows=[]
for path in sorted(glob.glob(os.path.join(root,'sweep','actors_*','point.json'))):
    d=json.load(open(path))
    rows.append({'actors': d['semantics']['actor_processes'],
                 'games_per_second': d['result']['games_per_second'],
                 'simulations_per_second': d['result']['simulations_per_second'],
                 'raw_positions_per_second': d['result']['raw_positions_per_second'],
                 'gpu_util_mean': d['resources'].get('gpu_util_mean'),
                 'cpu_util_pct_of_quota': d['resources'].get('cpu_util_percent_of_quota'),
                 'cpu_throttled_ratio': d['resources'].get('cpu_throttled_period_ratio'),
                 'memory_used_mib_max': d['resources'].get('memory_used_mib_max'),
                 'mean_inference_batch': d['result']['mean_inference_batch']})
best=max(rows,key=lambda r:r['simulations_per_second'])
json.dump({'points':rows,'best':best}, open(os.path.join(root,'sweep_summary.json'),'w'), indent=2, sort_keys=True)
print('best after extension: actors=%s' % best['actors'])
"
  BEST_ACTORS=$("$PY" -c "import json;print(json.load(open('$ROOT/sweep_summary.json'))['best']['actors'])")
fi
printf '%s\n' "$BEST_ACTORS" > "$ROOT/best_sweep_actors.txt"

########################################
log "=== phase 2: confirmation at best + neighbours, $CONFIRM_REPS reps x $CONFIRM_GAMES games (interleaved) ==="
"$PY" -c "
import json
best=int(open('$ROOT/best_sweep_actors.txt').read().strip())
pts=sorted(int(r['actors']) for r in json.load(open('$ROOT/sweep_summary.json'))['points'])
i=pts.index(best)
sel=sorted({pts[j] for j in (i-1,i,i+1) if 0<=j<len(pts)})
open('$ROOT/confirm_actors.txt','w').write(' '.join(str(p) for p in sel))
print('confirm points:', sel)
"
read -r CONFIRM_ACTORS < "$ROOT/confirm_actors.txt"

for REP in $(seq 1 $CONFIRM_REPS); do
  for ACTORS in $CONFIRM_ACTORS; do
    OUT=$ROOT/confirm/actors_${ACTORS}/rep_${REP}
    mkdir -p "$(dirname "$OUT")"
    [[ -e "$OUT" ]] && fail "confirm output already exists: $OUT"
    log "--- confirm actors=$ACTORS rep=$REP games=$CONFIRM_GAMES"
    set +e
    "$PY" -B "$REPO/tools/benchmark_bal5_topology_point.py" \
      --run-dir "$RUN" --checkpoint "$CKPT" --expected-checkpoint-sha256 "$CKPT_SHA" \
      --generation $GEN --actors "$ACTORS" --devices "$DEVICES" --games "$CONFIRM_GAMES" \
      --game-id-base $(( (REP - 1) * CONFIRM_GAMES + 100000 )) \
      --output-root "$OUT" --label "confirm_actors${ACTORS}_rep${REP}" 2>&1 | tee "$OUT.log"
    code=$?
    set -e
    [[ $code -eq 0 ]] || fail "confirm actors=$ACTORS rep=$REP failed with exit $code"
  done
done

"$PY" - "$ROOT" <<'PYEOF'
import glob, json, os, statistics, sys
root = sys.argv[1]
by_actors = {}
for path in glob.glob(os.path.join(root, "confirm", "actors_*", "rep_*", "point.json")):
    d = json.load(open(path))
    by_actors.setdefault(d["semantics"]["actor_processes"], []).append(
        d["result"]["simulations_per_second"])
summary = {}
for actors, values in sorted(by_actors.items()):
    summary[str(actors)] = {
        "reps": len(values),
        "simulations_per_second": values,
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "spread_pct": (max(values) - min(values)) / statistics.fmean(values) * 100.0,
    }
json.dump(summary, open(os.path.join(root, "confirm_summary.json"), "w"), indent=2, sort_keys=True)
for actors, row in summary.items():
    print("actors=%-3s n=%d mean=%.1f sd=%.1f spread=%.2f%%" % (
        actors, row["reps"], row["mean"], row["stdev"], row["spread_pct"]))
PYEOF

########################################
log "=== phase 3: learner, $CONFIRM_REPS x 256 steps FP32 batch 256 ==="
for REP in $(seq 1 $CONFIRM_REPS); do
  OUT=$ROOT/learner/rep_$REP
  mkdir -p "$ROOT/learner"
  [[ -e "$OUT" ]] && fail "learner output already exists: $OUT"
  DEV=${DEVICES%%,*}
  log "--- learner rep=$REP device=$DEV"
  set +e
  "$PY" -B "$REPO/tools/benchmark_bal5_learner_point.py" \
    --run-dir "$RUN" --checkpoint "$CKPT" --expected-checkpoint-sha256 "$CKPT_SHA" \
    --device "$DEV" --steps 256 --repeat-index "$REP" \
    --output-root "$OUT" --label "learner_rep${REP}" 2>&1 | tee "$OUT.log"
  code=$?
  set -e
  [[ $code -eq 0 ]] || fail "learner rep=$REP failed with exit $code"
done

"$PY" - "$ROOT" <<'PYEOF'
import glob, json, os, statistics, sys
root = sys.argv[1]
values = []
for path in sorted(glob.glob(os.path.join(root, "learner", "rep_*", "learner.json"))):
    d = json.load(open(path))
    values.append(d["result"]["positions_per_second"])
payload = {
    "reps": len(values),
    "positions_per_second": values,
    "mean": statistics.fmean(values),
    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    "min": min(values),
    "max": max(values),
    "spread_pct": (max(values) - min(values)) / statistics.fmean(values) * 100.0,
}
json.dump(payload, open(os.path.join(root, "learner_summary.json"), "w"), indent=2, sort_keys=True)
print("learner positions/s mean=%.1f sd=%.1f spread=%.2f%%" % (
    payload["mean"], payload["stdev"], payload["spread_pct"]))
PYEOF

########################################
log "=== phase 4: gate re-verification on the accepted legal setting, 3 cases ==="
GATE_DEVICES="$DEVICES"
[[ "$PROFILE" == "4090" ]] && GATE_DEVICES="cuda:0"
for SPEC in \
  "$RUN/../bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828:52" \
  "$RUN/../bal5_r1_column_no_tail_warm_fp32lr1e4_seed271828:77" \
  "$RUN/../bal5_r1_column_serial_attn2_warm_fp32lr1e4_seed271828:77" ; do
  NAME=$(echo "$SPEC" | sed 's#.*/runs/##; s#:.*##')
  GENN=$(echo "$SPEC" | sed 's#.*:##')
  OUT=$ROOT/gate/${NAME}_g${GENN}
  [[ -e "$OUT" ]] && fail "gate output already exists: $OUT"
  mkdir -p "$ROOT/gate"
  log "--- gate $NAME g$GENN devices=$GATE_DEVICES"
  set +e
  "$PY" -B "$REPO/tools/validate_bal5_gate_single_gpu.py" \
    --case "$SPEC" \
    --evaluation-parallel-games 8 \
    --evaluation-inference-batch-size 32 \
    --evaluation-inference-batch-timeout-ms 1.0 \
    --evaluation-devices "$GATE_DEVICES" \
    --output-root "$OUT" 2>&1 | tee "$OUT.log"
  code=$?
  set -e
  [[ $code -eq 0 ]] || fail "gate $NAME g$GENN failed with exit $code"
done

log "=== phase 5: machine-readable summary ==="
"$PY" - "$ROOT" "$PROFILE" <<'PYEOF'
import glob, json, os, statistics, sys
root, profile = sys.argv[1], sys.argv[2]

def load(path):
    return json.load(open(path))

sweep = load(os.path.join(root, "sweep_summary.json"))
confirm = load(os.path.join(root, "confirm_summary.json"))
learner = load(os.path.join(root, "learner_summary.json"))

gate_rows = []
for path in sorted(glob.glob(os.path.join(root, "gate", "*", "*", "*", "comparison.json"))):
    d = load(path)
    gate_rows.append({
        "run_dir": d["run_dir"],
        "generation": d["generation"],
        "elapsed_seconds": d["elapsed_seconds"],
        "elapsed_minutes": d["elapsed_seconds"] / 60.0,
        "strict_identical": d["passed"],
        "differing_keys": sorted(d.get("differences") or {}),
        "verdict_equal": not any(
            key in (d.get("differences") or {}) for key in ("verdict", "reason")
        ),
        "evaluation_devices": d["single_gpu_runtime"]["evaluation_devices"],
        "evaluation_replicas_per_device": d["single_gpu_runtime"]["evaluation_replicas_per_device"],
    })

best_confirm = max(confirm.items(), key=lambda kv: kv[1]["mean"])
payload = {
    "schema": "connect4-bal5-calibration-summary-v1",
    "profile": profile,
    "generated_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    "manifest": load(os.path.join(root, "manifest.json")),
    "selfplay_sweep": sweep,
    "selfplay_confirmation": confirm,
    "selfplay_selected": {
        "actors": int(best_confirm[0]),
        "basis": "highest mean simulations_per_second over repeated 256-game points",
        "mean": best_confirm[1]["mean"],
        "stdev": best_confirm[1]["stdev"],
        "spread_pct": best_confirm[1]["spread_pct"],
    },
    "learner": learner,
    "gate": gate_rows,
}
json.dump(payload, open(os.path.join(root, "summary.json"), "w"), indent=2, sort_keys=True)
print(json.dumps({
    "profile": profile,
    "selected_actors": payload["selfplay_selected"]["actors"],
    "selfplay_mean_spss": round(best_confirm[1]["mean"], 1),
    "learner_mean_pos_per_s": round(learner["mean"], 1),
    "gate_minutes": [round(row["elapsed_minutes"], 1) for row in gate_rows],
    "gate_strict_identical": [row["strict_identical"] for row in gate_rows],
    "gate_verdict_equal": [row["verdict_equal"] for row in gate_rows],
}, sort_keys=True))
PYEOF

########################################
printf 'profile=%s\nstamp=%s\ncompleted_utc=%s\n' \
  "$PROFILE" "$STAMP" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ROOT/SUCCESS"
log "CALIBRATION SUCCESS -> $ROOT/SUCCESS"
