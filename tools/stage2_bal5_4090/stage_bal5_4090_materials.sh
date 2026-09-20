#!/usr/bin/env bash
# Stage the BAL-5 evidence set on connect4_gpu_2608 onto its system disk so that
# the AutoDL image-migration feature can carry it to the replacement host.
#
# The destination is populated once and never overwritten: re-running this
# script against an existing --dest refuses to start.
set -euo pipefail

src=/root/autodl-tmp/Connect4_3D_game_refactor
dest=/root/bal5_4090_migration_20260920

if [[ -e "$dest" ]]; then
  echo "Refusing to overwrite existing destination: $dest" >&2
  exit 2
fi

mkdir -p "$dest/repo"

# Copy the executable project tree, but not generated evidence: the run state is
# staged explicitly below so the evidence boundary stays auditable.
rsync -a \
  --exclude='.git/' \
  --exclude='.tmp/' \
  --exclude='release/' \
  --exclude='training/runs/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$src/" "$dest/repo/"

cd "$src"

# One case per compared architecture, pinned to the generation named in the
# BAL-5 R1 plan and the gate-cache validation.
copy_case() {
  local run_name=$1
  local generation=$2
  local checkpoint_name=$3
  local accepted_name=$4
  local candidate_name=$5
  local run_root="training/runs/stage2/bal5/r1/runs/$run_name"

  mkdir -p "$dest/repo/$run_root"
  cp -a --parents \
    "$run_root/resolved_config.json" \
    "$run_root/run_manifest.json" \
    "$run_root/metrics" \
    "$run_root/replay" \
    "$run_root/samples/g${generation}" \
    "$run_root/manifests/generations/g${generation}.json" \
    "$run_root/manifests/generation_drafts/g${generation}.json" \
    "$run_root/checkpoints/$checkpoint_name" \
    "$run_root/accepted/$accepted_name" \
    "$run_root/rejected/$candidate_name" \
    "$dest/repo"
}

copy_case \
  bal5_r1_column_no_tail_warm_fp32lr1e4_seed271828 \
  000077 \
  g000077-s00018846.pt \
  candidate-g000062-s00015218-d00971827.pt \
  candidate-g000077-s00018846-d01203553.pt

copy_case \
  bal5_r1_column_serial_attn2_warm_fp32lr1e4_seed271828 \
  000077 \
  g000077-s00018864.pt \
  candidate-g000067-s00016432-d01049642.pt \
  candidate-g000077-s00018864-d01204919.pt

copy_case \
  bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828 \
  000052 \
  g000052-s00012744.pt \
  candidate-g000037-s00009072-d00579501.pt \
  candidate-g000052-s00012744-d00813948.pt

# Copy any completed focused gate-replay evidence.
gate_validation=training/runs/stage2/bal5/r1/gate_cache_validation/20260920_after_g57_drain
if [[ -d "$gate_validation" ]]; then
  cp -a --parents "$gate_validation" "$dest/repo"
fi

{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source=$src"
  echo "source_git_commit=$(git -C "$src" rev-parse HEAD)"
  echo "purpose=BAL-5 throughput comparison on 1x4090+20vCPU"
  echo "phases=selfplay,learner,paired_gate"
  echo "case_1=bal5_r1_column_no_tail_warm_fp32lr1e4_seed271828:g77"
  echo "case_2=bal5_r1_column_serial_attn2_warm_fp32lr1e4_seed271828:g77"
  echo "case_3=bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828:g52"
  echo "note=Source run directories were copied, not moved or pruned."
} > "$dest/MIGRATION_INFO.txt"

cd "$dest"
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS >/dev/null
du -sh "$dest"
find "$dest/repo/training/runs/stage2/bal5/r1/runs" -type f | wc -l
echo "STAGING_VERIFIED"
