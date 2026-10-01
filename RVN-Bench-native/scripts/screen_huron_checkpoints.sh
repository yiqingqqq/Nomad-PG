#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RVN-Bench-native
HABITAT=/root/autodl-tmp/habitat-lab
PYTHON=/root/miniconda3/envs/habitat/bin/python
RUN=/root/autodl-tmp/RVN-Bench-gostanford-to-huron-runs/rvn_vint_gostanford_to_huron/vint_pg_gostanford_mirror_to_huron_ft5_2026_09_26_15_15_40
EPISODES="$ROOT/logs/hm3d_pointnav/fixed_mirror_epoch1_100.json.gz"
OUT="$ROOT/logs/hm3d_pointnav/huron_checkpoint_screen"

mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$HABITAT/habitat-lab:$HABITAT/habitat-baselines:$ROOT:$ROOT/models/diffusion_policy:$ROOT/models/gnms_levin/train${PYTHONPATH:+:$PYTHONPATH}"

for epoch in 0 1 2 3 4; do
  "$PYTHON" scripts/eval_vint_pg_hm3d_pointnav.py \
    --checkpoint "$RUN/$epoch.pth" \
    --task-config "$HABITAT/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml" \
    --episodes "$HABITAT/data/datasets/pointnav/hm3d/v1" \
    --episode-file "$EPISODES" \
    --scenes-dir "$HABITAT/data/scene_datasets" \
    --num-episodes 30 --pg-rcs --controller trajectory --lookahead-idx 2 \
    --collision-recovery none --rgb-mode normal \
    --output "$OUT/epoch${epoch}_pure_30ep.json"
done
