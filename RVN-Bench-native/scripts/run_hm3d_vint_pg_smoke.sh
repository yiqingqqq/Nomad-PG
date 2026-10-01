#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RVN-Bench-native
HABITAT=/root/autodl-tmp/habitat-lab
PYTHON=/root/miniconda3/envs/habitat/bin/python
NUM_EPISODES="${1:-${NUM_EPISODES:-10}}"
RGB_MODE="${2:-${RGB_MODE:-normal}}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/RVN-Bench-baseline-checkpoints/gostandford_only_epoch18.pth}"
CONTROLLER="${CONTROLLER:-trajectory}"
LOOKAHEAD_IDX="${LOOKAHEAD_IDX:-2}"
CHECKPOINT_TAG="$(basename "$CHECKPOINT" .pth)"

cd "$ROOT"
export PYTHONPATH="$HABITAT/habitat-lab:$HABITAT/habitat-baselines:$ROOT:$ROOT/models/diffusion_policy:$ROOT/models/gnms_levin/train${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" scripts/eval_vint_pg_hm3d_pointnav.py \
  --checkpoint "$CHECKPOINT" \
  --task-config "$HABITAT/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml" \
  --episodes "$HABITAT/data/datasets/pointnav/hm3d/v1" \
  --scenes-dir "$HABITAT/data/scene_datasets" \
  --num-episodes "$NUM_EPISODES" \
  --pg-rcs \
  --controller "$CONTROLLER" \
  --lookahead-idx "$LOOKAHEAD_IDX" \
  --rgb-mode "$RGB_MODE" \
  --output "$ROOT/logs/hm3d_pointnav/vint_pg_${CHECKPOINT_TAG}_${CONTROLLER}_la${LOOKAHEAD_IDX}_${RGB_MODE}_${NUM_EPISODES}ep.json"
