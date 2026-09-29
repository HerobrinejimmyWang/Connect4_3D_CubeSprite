#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Connect4_3D_game_refactor
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/root/miniconda3/bin/python -u tools/stage2_fla_raw_search_budget.py positions --positions-per-rule 10
/root/miniconda3/bin/python -u tools/stage2_fla_raw_search_budget.py matches --pairs-per-rule 50
