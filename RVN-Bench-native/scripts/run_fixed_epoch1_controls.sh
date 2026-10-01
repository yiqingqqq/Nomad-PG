#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RVN-Bench-native
HABITAT=/root/autodl-tmp/habitat-lab
PYTHON=/root/miniconda3/envs/habitat/bin/python
FIXED_EPISODES="$ROOT/logs/hm3d_pointnav/fixed_mirror_epoch1_100.json.gz"
MIRROR_RUN=/autodl-fs/data/RVN-Bench-gostanford-mirror-runs/rvn_vint_go_stanford/vint_pg_gostanford_mirror_ft5_2026_09_23_21_36_03

cd "$ROOT"
export PYTHONPATH="$HABITAT/habitat-lab:$HABITAT/habitat-baselines:$ROOT:$ROOT/models/diffusion_policy:$ROOT/models/gnms_levin/train${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" scripts/eval_vint_pg_hm3d_pointnav.py \
  --checkpoint /root/autodl-tmp/RVN-Bench-baseline-checkpoints/gostandford_only_epoch18.pth \
  --task-config "$HABITAT/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml" \
  --episodes "$HABITAT/data/datasets/pointnav/hm3d/v1" \
  --episode-file "$FIXED_EPISODES" \
  --scenes-dir "$HABITAT/data/scene_datasets" \
  --num-episodes 100 --pg-rcs --controller trajectory --lookahead-idx 2 \
  --rgb-mode normal \
  --output "$ROOT/logs/hm3d_pointnav/fixed_baseline_gostanford_normal_100.json"

"$PYTHON" scripts/eval_vint_pg_hm3d_pointnav.py \
  --checkpoint "$MIRROR_RUN/1.pth" \
  --task-config "$HABITAT/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml" \
  --episodes "$HABITAT/data/datasets/pointnav/hm3d/v1" \
  --episode-file "$FIXED_EPISODES" \
  --scenes-dir "$HABITAT/data/scene_datasets" \
  --num-episodes 100 --pg-rcs --controller trajectory --lookahead-idx 2 \
  --rgb-mode black \
  --output "$ROOT/logs/hm3d_pointnav/fixed_mirror_epoch1_black_100.json"

echo "FINISHED_FIXED_EPOCH1_CONTROLS"
