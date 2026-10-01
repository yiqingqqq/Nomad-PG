#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RVN-Bench-native
RUN=/autodl-fs/data/RVN-Bench-gostanford-mirror-runs/rvn_vint_go_stanford/vint_pg_gostanford_mirror_ft5_2026_09_23_21_36_03
SCREEN_LOG="$ROOT/logs/hm3d_pointnav/mirror_checkpoint_screen.log"

until grep -q "FINISHED_MIRROR_CHECKPOINT_SCREEN" "$SCREEN_LOG"; do
  sleep 30
done

cd "$ROOT"
BEST_EPOCH=$(/root/miniconda3/envs/habitat/bin/python scripts/select_best_mirror_checkpoint.py)
echo "SELECTED_MIRROR_EPOCH=$BEST_EPOCH"
CHECKPOINT="$RUN/$BEST_EPOCH.pth" \
CONTROLLER=trajectory \
LOOKAHEAD_IDX=2 \
  bash scripts/run_hm3d_vint_pg_smoke.sh 100 normal
echo "FINISHED_BEST_MIRROR_100EP epoch=$BEST_EPOCH"
