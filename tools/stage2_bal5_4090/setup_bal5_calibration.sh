#!/usr/bin/env bash
# Prepare the BAL-5 topology-calibration root on ONE machine.
#
# Creates a FRESH git checkout at a pinned commit under /root/bal5_calibration/repo
# so the calibration runs identical tool code on both machines, without touching
# either machine's active training repository or its historical evidence.
#
# Usage: setup_bal5_calibration.sh <base-commit> <expose-runs-from>
#   base-commit:      commit the delta bundle is applied on top of (9fcb1bf...)
#   expose-runs-from: existing repo whose training/runs should be visible to the
#                     calibration checkout (never modified, only read)
set -euo pipefail

BASE_COMMIT=${1:?base commit required}
EXPOSE=${2:?path to the existing repo required}
TARGET_COMMIT=b285d0addb21eb7f7cc8edfe408c7fce8e6f620b
ROOT=/root/bal5_calibration
REPO=$ROOT/repo

if [[ -e "$REPO" ]]; then
  echo "refusing to reuse existing calibration checkout: $REPO" >&2
  echo "remove it deliberately if you really want a fresh prepare" >&2
  exit 2
fi
[[ -f /root/calib_delta.bundle ]] || { echo "missing /root/calib_delta.bundle" >&2; exit 2; }
[[ -d "$EXPOSE/training/runs" ]] || { echo "missing $EXPOSE/training/runs" >&2; exit 2; }

mkdir -p "$ROOT"
git init -q "$REPO"
cd "$REPO"
git config user.name 'calibration'
git config user.email 'calibration@local'
git fetch -q "$EXPOSE/.git" "$BASE_COMMIT:refs/remotes/base/main" 2>/dev/null || true
git fetch -q /root/calib_delta.bundle refs/heads/main:refs/remotes/delta/main
git checkout -q -f -B main refs/remotes/delta/main
git remote remove delta 2>/dev/null || true

HEAD_NOW=$(git rev-parse HEAD)
[[ "$HEAD_NOW" == "$TARGET_COMMIT" ]] || { echo "HEAD $HEAD_NOW != $TARGET_COMMIT" >&2; exit 3; }

# Deploy the three calibration tools at the canonical paths inside this checkout.
cp /root/c_topo.py    "$REPO/tools/benchmark_bal5_topology_point.py"
cp /root/c_learner.py "$REPO/tools/benchmark_bal5_learner_point.py"
cp /root/c_gate.py    "$REPO/tools/validate_bal5_gate_single_gpu.py"

# Make the existing run history visible read-only: the tools only ever read
# resolved_config.json / manifests / checkpoints and always write elsewhere.
mkdir -p "$REPO/training/runs"
if [[ ! -e "$REPO/training/runs/stage2" ]]; then
  ln -s "$EXPOSE/training/runs/stage2" "$REPO/training/runs/stage2"
fi

echo "=== checkout ready ==="
echo "HEAD: $HEAD_NOW"
git status --short | head -10
echo "=== tools present ==="
ls -la "$REPO/tools/benchmark_bal5_topology_point.py" \
       "$REPO/tools/benchmark_bal5_learner_point.py" \
       "$REPO/tools/validate_bal5_gate_single_gpu.py"
echo "=== run history visible ==="
ls -d "$REPO/training/runs/stage2/bal5/r1/runs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828" 2>&1
test -f "$REPO/training/runs/stage2/bal5/r1/runs/bal5_r1_winning_no_tail_warm_fp32lr1e4_seed271828/checkpoints/g000057-s00013959.pt" \
  && echo "g57 checkpoint reachable"
echo "=== python import smoke test ==="
cd "$REPO"
/root/miniconda3/bin/python -c "
import sys; sys.path.insert(0,'.')
sys.path.insert(0,'tools')
from training.v3.config import load_config, config_hash
from training.v3.pipeline import lineage_config_hash
import benchmark_v3_selfplay_topology as b
print('imports OK; ResourceSampler:', hasattr(b,'_ResourceSampler'))
"
echo "=== disk ==="
df -h /root/autodl-tmp 2>/dev/null | tail -n 1 || df -h / | tail -n 1
echo SETUP_READY
