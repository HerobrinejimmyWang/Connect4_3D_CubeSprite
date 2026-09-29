#!/usr/bin/env bash
# Extra confirmation repeats for the 1x4090 host.
#
# Why: the first confirmation round produced 12-14% spread across 3 repeats, far
# above the <=5% stability bar the selection rule requires. At that noise level the
# 20/24/28 means (3290 / 3354 / 3193 sims/s) are statistically indistinguishable,
# so no topology change is justified and the sweep-boundary/least-noisy fallback
# must not be silently promoted to a production setting.
#
# This adds repeats 4..6 so each point has 6 samples, which tightens the standard
# error by ~sqrt(2) and lets the rule either pick a stable winner honestly or
# report that no point qualifies and keep the incumbent topology.
#
# The existing reps 1..3 are never re-run or deleted.
set -uo pipefail

STAMP=${1:?stamp required}
shift
ACTORS_LIST=("$@")
[[ ${#ACTORS_LIST[@]} -gt 0 ]] || { echo "need actor counts" >&2; exit 2; }

REPO=/root/bal5_calibration/repo
RUN=$REPO/training/runs/stage2/bal5/r1/runs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828
CKPT=$RUN/checkpoints/g000072-s00017611.pt
CKPT_SHA=22ed45ccd1257f3f18f7f0f9183755a0158b0bf2c54e672d174642bb1b1b791a
GEN=72
DEVICES=cuda:0
PY=/root/miniconda3/bin/python
ROOT=/root/bal5_calibration/$STAMP
GAMES=256
EXTRA_REPS="4 5 6"
STATUS=$ROOT/CONFIRM_EXTRA_STATUS.txt

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
fail() { echo "FAILED $1" > "$ROOT/CONFIRM_EXTRA_FAILED"; echo "error: $1" >> "$STATUS"; log "FAILED: $1"; exit 1; }

cd "$REPO" || fail "cd failed"
echo "pid=$$ started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
echo "extra_reps=$EXTRA_REPS actors=${ACTORS_LIST[*]}" >> "$STATUS"

ACTUAL=$(sha256sum "$CKPT" | cut -d' ' -f1)
[[ "$ACTUAL" == "$CKPT_SHA" ]] || fail "checkpoint sha mismatch: $ACTUAL"
log "checkpoint verified; adding reps $EXTRA_REPS for actors ${ACTORS_LIST[*]}"

for REP in $EXTRA_REPS; do
  for ACTORS in "${ACTORS_LIST[@]}"; do
    OUT=$ROOT/confirm/actors_${ACTORS}/rep_${REP}
    mkdir -p "$ROOT/confirm/actors_${ACTORS}"
    if [[ -e "$OUT" ]]; then log "  skip existing $OUT"; continue; fi
    log "--- confirm actors=$ACTORS rep=$REP games=$GAMES"
    set +e
    "$PY" -B "$REPO/tools/benchmark_bal5_topology_point.py" \
      --run-dir "$RUN" --checkpoint "$CKPT" --expected-checkpoint-sha256 "$CKPT_SHA" \
      --generation $GEN --actors "$ACTORS" --devices "$DEVICES" --games "$GAMES" \
      --game-id-base $(( (REP - 1) * GAMES + 100000 )) \
      --output-root "$OUT" --label "confirm_actors${ACTORS}_rep${REP}" 2>&1 | tee "$OUT.log"
    code=$?
    set -e
    [[ $code -eq 0 ]] || fail "confirm actors=$ACTORS rep=$REP exit $code"
    echo "confirm actors=$ACTORS rep=$REP ok utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  done
done

# Regenerate the confirmation summary over ALL reps now on disk.
"$PY" - "$ROOT" <<'PYEOF' || fail "confirm summary regeneration failed"
import glob, json, os, statistics, sys
root = sys.argv[1]
by_actors = {}
for path in glob.glob(os.path.join(root, "confirm", "actors_*", "rep_*", "point.json")):
    d = json.load(open(path))
    by_actors.setdefault(d["semantics"]["actor_processes"], []).append(d["result"])
summary = {}
for actors, rows in sorted(by_actors.items()):
    sps = [r["simulations_per_second"] for r in rows]
    summary[str(actors)] = {
        "reps": len(rows),
        "simulations_per_second": sps,
        "simulations_per_second_mean": statistics.fmean(sps),
        "simulations_per_second_stdev": statistics.stdev(sps) if len(sps) > 1 else 0.0,
        "simulations_per_second_stderr": (statistics.stdev(sps) / len(sps) ** 0.5) if len(sps) > 1 else 0.0,
        "games_per_second_mean": statistics.fmean(r["games_per_second"] for r in rows),
        "raw_positions_per_second_mean": statistics.fmean(r["raw_positions_per_second"] for r in rows),
        "min": min(sps), "max": max(sps),
        "spread_pct": (max(sps) - min(sps)) / statistics.fmean(sps) * 100.0,
    }
json.dump(summary, open(os.path.join(root, "confirm_summary.json"), "w"), indent=2, sort_keys=True)
for actors, row in summary.items():
    print("actors=%-3s n=%d mean=%.1f sd=%.1f stderr=%.1f spread=%.2f%%" % (
        actors, row["reps"], row["simulations_per_second_mean"],
        row["simulations_per_second_stdev"], row["simulations_per_second_stderr"], row["spread_pct"]))
PYEOF

# Rebuild the interim summary with an explicit, honest selection verdict.
"$PY" - "$ROOT" "4090" <<'PYEOF' || fail "summary regeneration failed"
import json, os, sys
root, profile = sys.argv[1], sys.argv[2]
sweep = json.load(open(os.path.join(root, "sweep_summary.json")))
confirm = json.load(open(os.path.join(root, "confirm_summary.json")))
learner = json.load(open(os.path.join(root, "learner_summary.json")))
ordered = sorted(confirm.items(), key=lambda kv: kv[1]["simulations_per_second_mean"], reverse=True)
eligible, excluded = [], []
for actors, row in ordered:
    reasons = []
    if row["spread_pct"] > 5.0:
        reasons.append("spread %.2f%% > 5%% after %d repeats" % (row["spread_pct"], row["reps"]))
    if reasons:
        excluded.append({"actors": int(actors), "reasons": reasons})
    else:
        eligible.append((actors, row))
if eligible:
    # Require the winner to beat the runner-up by more than the combined standard
    # error, otherwise the difference is not resolvable and no change is justified.
    winner_actors, winner = eligible[0]
    second = eligible[1] if len(eligible) > 1 else None
    resolvable = True
    if second is not None:
        gap = winner["simulations_per_second_mean"] - second[1]["simulations_per_second_mean"]
        combined = (winner["simulations_per_second_stderr"] ** 2 + second[1]["simulations_per_second_stderr"] ** 2) ** 0.5
        resolvable = gap > combined
        if not resolvable:
            excluded.append({
                "actors": int(winner_actors),
                "reasons": ["gap %.1f sims/s to runner-up %s is within combined stderr %.1f; not resolvable"
                            % (gap, second[0], combined)],
            })
    selected = int(winner_actors) if resolvable else None
else:
    selected = None

payload = {
    "schema": "connect4-bal5-calibration-interim-summary-v1",
    "profile": profile, "interim": True,
    "gate_phase": "deferred_by_plan (three-case gate replay runs after training resumes)",
    "manifest": json.load(open(os.path.join(root, "manifest.json"))),
    "selfplay_sweep": sweep,
    "selfplay_confirmation": confirm,
    "learner": learner,
    "selection": {
        "rule": ("highest mean simulations_per_second among points with repeat spread <= 5%, "
                 "and only if it beats the runner-up by more than the combined standard error"),
        "selected_actors": selected,
        "verdict": ("stable winner found" if selected is not None else
                    "NO STABLE WINNER - measurements do not resolve the candidate points; "
                    "keep the incumbent topology and do not change operational settings"),
        "ordered_by_mean_sps": [int(a) for a, _ in ordered],
        "excluded": excluded,
    },
}
json.dump(payload, open(os.path.join(root, "summary.json"), "w"), indent=2, sort_keys=True)
print(json.dumps({"profile": profile, "selected_actors": selected,
                  "verdict": payload["selection"]["verdict"]}, sort_keys=True))
PYEOF

echo "status=success" >> "$STATUS"
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
log "EXTRA CONFIRM COMPLETE"
