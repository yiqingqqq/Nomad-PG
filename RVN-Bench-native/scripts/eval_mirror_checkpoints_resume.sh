#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RVN-Bench-native
RUN=/autodl-fs/data/RVN-Bench-gostanford-mirror-runs/rvn_vint_go_stanford/vint_pg_gostanford_mirror_ft5_2026_09_23_21_36_03

cd "$ROOT"
for epoch in 1 2 3 4; do
  result="$ROOT/logs/hm3d_pointnav/vint_pg_${epoch}_trajectory_la2_normal_30ep.json"
  if test -s "$result"; then
    echo "SKIPPING_COMPLETED_MIRROR_EPOCH=$epoch"
    continue
  fi
  echo "RESUMING_MIRROR_EPOCH=$epoch"
  CHECKPOINT="$RUN/$epoch.pth" \
  CONTROLLER=trajectory \
  LOOKAHEAD_IDX=2 \
    bash scripts/run_hm3d_vint_pg_smoke.sh 30 normal
done
echo "FINISHED_MIRROR_CHECKPOINT_SCREEN"
