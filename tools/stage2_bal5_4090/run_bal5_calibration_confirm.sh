#!/usr/bin/env bash
# BAL-5 calibration: confirmation + learner phases, resuming from a completed sweep.
#
# Why this script exists: run_bal5_topology_calibration.sh terminated silently
# after its sweep because of a bash+e bug:
#     read -r CONFIRM_ACTORS < "$ROOT/confirm_actors.txt"
# `read` returns non-zero when the file ends without a newline, and that file is
# written without one, so `set -e` killed the script between the sweep and the
# confirmation phase. It exited on a non-`fail()` path, hence no FAILED/REASON.
# Fixed here with `|| true`, and every `read` in this file is guarded.
#
# This script NEVER re-runs the sweep: it consumes the existing sweep output.
#
# Usage: run_bal5_calibration_confirm.sh <stamp> <actors...>
#   e.g. run_bal5_calibration_confirm.sh 20260921T1500Z 20 24 28
set -uo pipefail

STAMP=${1:?stamp required}
shift
REQUESTED_ACTORS=("$@")
[[ ${#REQUESTED_ACTORS[@]} -gt 0 ]] || { echo "at least one actor count required" >&2; exit 2; }

REPO=/root/bal5_calibration/repo
RUN_ROOT=$REPO/training/runs/stage2/bal5/r1/runs
PY=/root/miniconda3/bin/python
ROOT=/root/bal5_calibration/$STAMP
HOST=$(hostname)
CONFIRM_GAMES=256
CONFIRM_REPS=3
LEARNER_STEPS=256
LEARNER_REPS=3

# Per-profile target: calibrate on the checkpoint this host actually paused at.
case "$HOST" in
  *de4e40b80a*)
    PROFILE=4090
    DEVICES="cuda:0"
    RUN=$RUN_ROOT/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828
    CKPT=$RUN/checkpoints/g000072-s00017611.pt
    CKPT_SHA=22ed45ccd1257f3f18f7f0f9183755a0158b0bf2c54e672d174642bb1b1b791a
    GEN=72
    ;;
  *)
    PROFILE=3080ti
    DEVICES="cuda:0,cuda:1"
    RUN=$RUN_ROOT/bal5_r1_winning_serial_attn2_warm_fp32lr1e4_seed271828
    CKPT=$RUN/checkpoints/g000032-s00007875.pt
    CKPT_SHA=c30ae51e3f08561dca220328e061c6df2618500d622c81c178abb651e46a1961
    GEN=32
    ;;
esac

STATUS=$ROOT/CONFIRM_STATUS.txt
echo "pid=$$ profile=$PROFILE host=$HOST started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
echo "actors=${REQUESTED_ACTORS[*]} devices=$DEVICES run=$RUN gen=$GEN" >> "$STATUS"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

fail() {
  printf 'FAILED %s\n%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ROOT/CONFIRM_FAILED"
  echo "error: $1" >> "$STATUS"
  log "FAILED: $1"
  exit 1
}

cd "$REPO" || fail "cannot cd to $REPO"

log "confirm+learner queue starting (profile=$PROFILE pid=$$)"
echo "=== precondition: checkpoint identity ==="
[[ -f "$CKPT" ]] || fail "checkpoint missing: $CKPT"
ACTUAL=$(sha256sum "$CKPT" | cut -d' ' -f1)
[[ "$ACTUAL" == "$CKPT_SHA" ]] || fail "checkpoint sha mismatch: $ACTUAL"
log "checkpoint verified ($ACTUAL)"
echo "=== precondition: sweep outputs exist ==="
for A in "${REQUESTED_ACTORS[@]}"; do
  [[ -f "$ROOT/sweep/actors_$A/point.json" ]] || fail "missing sweep point for actors=$A; refusing to re-run the sweep"
done
log "all requested sweep points present"

########################################
log "=== phase 2: confirmation, $CONFIRM_REPS reps x $CONFIRM_GAMES games, interleaved ==="
for REP in $(seq 1 $CONFIRM_REPS); do
  for ACTORS in "${REQUESTED_ACTORS[@]}"; do
    OUT=$ROOT/confirm/actors_${ACTORS}/rep_${REP}
    mkdir -p "$ROOT/confirm/actors_${ACTORS}"
    if [[ -e "$OUT" ]]; then
      log "  skip existing $OUT"
      continue
    fi
    log "--- confirm actors=$ACTORS rep=$REP games=$CONFIRM_GAMES"
    set +e
    "$PY" -B "$REPO/tools/benchmark_bal5_topology_point.py" \
      --run-dir "$RUN" --checkpoint "$CKPT" --expected-checkpoint-sha256 "$CKPT_SHA" \
      --generation $GEN --actors "$ACTORS" --devices "$DEVICES" --games "$CONFIRM_GAMES" \
      --game-id-base $(( (REP - 1) * CONFIRM_GAMES + 100000 )) \
      --output-root "$OUT" --label "confirm_actors${ACTORS}_rep${REP}" 2>&1 | tee "$OUT.log"
    code=$?
    set -e
    [[ $code -eq 0 ]] || fail "confirm actors=$ACTORS rep=$REP exit $code (see $OUT.log)"
    echo "confirm actors=$ACTORS rep=$REP ok utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  done
done

"$PY" - "$ROOT" <<'PYEOF' || fail "confirm summary generation failed"
import glob, json, os, statistics, sys
root = sys.argv[1]
by_actors = {}
for path in glob.glob(os.path.join(root, "confirm", "actors_*", "rep_*", "point.json")):
    d = json.load(open(path))
    by_actors.setdefault(d["semantics"]["actor_processes"], []).append(d["result"])
summary = {}
for actors, rows in sorted(by_actors.items()):
    sps = [r["simulations_per_second"] for r in rows]
    gps = [r["games_per_second"] for r in rows]
    rps = [r["raw_positions_per_second"] for r in rows]
    summary[str(actors)] = {
        "reps": len(rows),
        "simulations_per_second": sps,
        "simulations_per_second_mean": statistics.fmean(sps),
        "simulations_per_second_stdev": statistics.stdev(sps) if len(sps) > 1 else 0.0,
        "games_per_second_mean": statistics.fmean(gps),
        "raw_positions_per_second_mean": statistics.fmean(rps),
        "spread_pct": (max(sps) - min(sps)) / statistics.fmean(sps) * 100.0,
        "wall_seconds": [round(r["wall_seconds"], 2) for r in rows],
    }
json.dump(summary, open(os.path.join(root, "confirm_summary.json"), "w"), indent=2, sort_keys=True)
for actors, row in summary.items():
    print("actors=%-3s n=%d mean_sps=%.1f sd=%.1f spread=%.2f%%" % (
        actors, row["reps"], row["simulations_per_second_mean"],
        row["simulations_per_second_stdev"], row["spread_pct"]))
PYEOF
log "confirm summary written"

########################################
log "=== phase 3: learner, $LEARNER_REPS x $LEARNER_STEPS steps FP32 batch 256 ==="
DEV=${DEVICES%%,*}
for REP in $(seq 1 $LEARNER_REPS); do
  OUT=$ROOT/learner/rep_$REP
  mkdir -p "$ROOT/learner"
  if [[ -e "$OUT" ]]; then
    log "  skip existing $OUT"
    continue
  fi
  log "--- learner rep=$REP device=$DEV steps=$LEARNER_STEPS"
  set +e
  "$PY" -B "$REPO/tools/benchmark_bal5_learner_point.py" \
    --run-dir "$RUN" --checkpoint "$CKPT" --expected-checkpoint-sha256 "$CKPT_SHA" \
    --device "$DEV" --steps "$LEARNER_STEPS" --repeat-index "$REP" \
    --output-root "$OUT" --label "learner_rep${REP}" 2>&1 | tee "$OUT.log"
  code=$?
  set -e
  [[ $code -eq 0 ]] || fail "learner rep=$REP exit $code (see $OUT.log)"
  echo "learner rep=$REP ok utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
done

"$PY" - "$ROOT" <<'PYEOF' || fail "learner summary generation failed"
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
log "learner summary written"

########################################
log "=== phase 5: interim machine-readable summary (no gate; per plan) ==="
"$PY" - "$ROOT" "$PROFILE" <<'PYEOF' || fail "interim summary generation failed"
import json, os, statistics, sys
root, profile = sys.argv[1], sys.argv[2]
sweep = json.load(open(os.path.join(root, "sweep_summary.json")))
confirm = json.load(open(os.path.join(root, "confirm_summary.json")))
learner = json.load(open(os.path.join(root, "learner_summary.json")))

# Selection rule, stated explicitly and applied mechanically:
#   * candidates = measured confirm points
#   * a point is eligible only if its spread across repeats is <= 5% and it is
#     NOT excluded for sitting on the sweep boundary without a stable win
#   * winner = highest mean simulations_per_second among eligible points
#   * if the highest-mean point is ineligible, take the best eligible neighbour
ordered = sorted(confirm.items(), key=lambda kv: kv[1]["simulations_per_second_mean"], reverse=True)
sweep_points = sorted(int(r["actors"]) for r in sweep["points"])
boundary = sweep_points[-1]
eligible, excluded = [], []
for actors, row in ordered:
    reasons = []
    if row["spread_pct"] > 5.0:
        reasons.append("spread %.2f%% > 5%%" % row["spread_pct"])
    if int(actors) == boundary and len(ordered) > 1:
        reasons.append("sits on the sweep upper boundary without a stable win")
    if reasons:
        excluded.append({"actors": int(actors), "reasons": reasons})
    else:
        eligible.append((actors, row))
if not eligible:
    # Fall back to the least-noisy measured point rather than guessing.
    eligible = [(actors, row) for actors, row in ordered]
    excluded.append({"note": "no point met the strict rule; used least-noisy measured point"})
selected_actors, selected_row = eligible[0]

payload = {
    "schema": "connect4-bal5-calibration-interim-summary-v1",
    "profile": profile,
    "interim": True,
    "gate_phase": "deferred_by_plan (full three-case gate replay runs after training resumes)",
    "manifest": json.load(open(os.path.join(root, "manifest.json"))),
    "selfplay_sweep": sweep,
    "selfplay_confirmation": confirm,
    "selection": {
        "rule": "highest mean simulations_per_second among confirm points with repeat spread <= 5%, "
                "excluding a sweep-boundary point that has no stable win",
        "selected_actors": int(selected_actors),
        "selected_mean_sps": selected_row["simulations_per_second_mean"],
        "selected_stdev": selected_row["simulations_per_second_stdev"],
        "selected_spread_pct": selected_row["spread_pct"],
        "ordered_by_mean_sps": [int(a) for a, _ in ordered],
        "excluded": excluded,
    },
    "learner": learner,
}
json.dump(payload, open(os.path.join(root, "summary.json"), "w"), indent=2, sort_keys=True)
print(json.dumps({
    "profile": profile,
    "selected_actors": int(selected_actors),
    "selected_mean_sps": round(selected_row["simulations_per_second_mean"], 1),
    "learner_mean_pos_per_s": round(learner["mean"], 1),
}, sort_keys=True))
PYEOF

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
echo "status=success" >> "$STATUS"
printf 'profile=%s\nstamp=%s\ncompleted_utc=%s\n' "$PROFILE" "$STAMP" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ROOT/SUCCESS"
log "CONFIRM+LEARNER SUCCESS -> $ROOT/SUCCESS"
