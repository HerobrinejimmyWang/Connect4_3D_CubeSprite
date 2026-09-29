#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/Connect4_3D_game_refactor
name=raw3d_to2d_full_b6c128
base="$repo/training/runs/stage2/fla/b6_raw_full_2m"
config="$base/configs/${name}__standard_late__seed271828.json"
cd "$repo"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
/root/miniconda3/bin/python -B -m training.v3.stage2 train --config "$config"
