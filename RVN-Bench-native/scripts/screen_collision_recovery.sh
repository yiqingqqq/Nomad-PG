#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RVN-Bench-native
HABITAT=/root/autodl-tmp/habitat-lab
PYTHON=/root/miniconda3/envs/habitat/bin/python
CHECKPOINT=/autodl-fs/data/RVN-Bench-gostanford-mirror-runs/rvn_vint_go_stanford/vint_pg_gostanford_mirror_ft5_2026_09_23_21_36_03/1.pth
EPISODES="$ROOT/logs/hm3d_pointnav/fixed_mirror_epoch1_100.json.gz"

cd "$ROOT"
export PYTHONPATH="$HABITAT/habitat-lab:$HABITAT/habitat-baselines:$ROOT:$ROOT/models/diffusion_policy:$ROOT/models/gnms_levin/train${PYTHONPATH:+:$PYTHONPATH}"

for turns in 1 3; do
  "$PYTHON" scripts/eval_vint_pg_hm3d_pointnav.py \
    --checkpoint "$CHECKPOINT" \
    --task-config "$HABITAT/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml" \
    --episodes "$HABITAT/data/datasets/pointnav/hm3d/v1" \
    --episode-file "$EPISODES" \
    --scenes-dir "$HABITAT/data/scene_datasets" \
    --num-episodes 30 --pg-rcs --controller trajectory --lookahead-idx 2 \
    --collision-recovery goal_turn --recovery-turn-steps "$turns" \
    --rgb-mode normal \
    --output "$ROOT/logs/hm3d_pointnav/mirror_epoch1_goalturn${turns}_30ep.json"
done
