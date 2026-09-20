#!/usr/bin/env bash
# Assemble the transferred BAL-5 winning-no-tail resume snapshot on the 4090 host.
#
# The 9 GiB snapshot was produced on connect4_gpu_2608 (2x3080Ti) and moved to
# this 1x4090 container as 18 verified parts.  This watcher waits for every part,
# verifies the per-part SHA-256 manifest, concatenates and verifies the full
# archive, extracts it over the repository, and then asserts that the extracted
# generation state is exactly the g57 resume point.
#
# Code provenance is handled out of band: the repository was restored as a real
# git repository from a bundle and the single-GPU CUDA RNG resume compatibility
# patch is committed at 9fcb1bf's descendant d8f2cf80c167ffe11a2ab00759a0e8731d852062.
# This watcher therefore performs no git staging or commit of its own.
set -euo pipefail

data=/root/autodl-tmp
parts=$data/bal5_winning_no_tail_resume_g57_20260921.parts
archive=$data/bal5_winning_no_tail_resume_g57_20260921.tar
archive_sha=$archive.sha256
repo=/root/bal5_4090_migration_20260920/repo
resume=/root/bal5_4090_resume

mkdir -p "$resume"
printf 'status=waiting_for_parts\ntime_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$resume/snapshot.status"

while true; do
  complete=1
  for value in $(seq 0 16); do
    index=$(printf '%03d' "$value")
    path="$parts/part-$index"
    [[ -f "$path" && $(stat -c %s "$path") -eq 536870912 ]] || complete=0
  done
  last="$parts/part-017"
  [[ -f "$last" && $(stat -c %s "$last") -eq 261359616 ]] || complete=0
  if [[ $complete -eq 1 ]]; then
    break
  fi
  sleep 30
done

cd "$parts"
sha256sum -c SHA256SUMS >/dev/null
printf 'status=assembling\ntime_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$resume/snapshot.status"
cat part-* > "$archive"
cd "$data"
sha256sum -c "$(basename "$archive_sha")" >/dev/null

printf 'status=extracting\ntime_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$resume/snapshot.status"
tar -xf "$archive" -C "$repo"

run=$repo/training/runs/stage2/bal5/r1/runs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828
latest=$(basename "$(find "$run/manifests/generations" -maxdepth 1 -name 'g*.json' | sort | tail -n 1)")
[[ "$latest" == g000057.json ]]
test -f "$run/checkpoints/g000057-s00013959.pt"
test -f "$repo/.git/HEAD"

printf 'status=ready\ntime_utc=%s\nlatest_generation=57\nrepo_commit=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git -C "$repo" rev-parse HEAD)" \
  > "$resume/snapshot.status"
touch "$resume/RESUME_READY"
echo RESUME_SNAPSHOT_READY

# Transport artifacts are disposable only after both verification layers and
# materialization have succeeded. Keep the small full-archive checksum.
rm -f -- "$archive"
rm -rf -- "$parts"
