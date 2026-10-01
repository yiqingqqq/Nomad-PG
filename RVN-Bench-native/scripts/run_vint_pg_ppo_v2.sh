#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RVN-Bench-native
export PYTHONPATH=/root/autodl-tmp/RVN-Bench-native:/root/autodl-tmp/RVN-Bench-native/models/diffusion_policy:/root/autodl-tmp/RVN-Bench-native/models/gnms_levin/train
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
run_mode=${1:-train}
common=(
  --checkpoint /root/autodl-fs/RVN-Bench-gostanford-mirror-runs/rvn_vint_go_stanford/vint_pg_gostanford_mirror_ft5_2026_09_23_21_36_03/1.pth
  --task-config /root/autodl-tmp/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml
  --scenes-dir /root/autodl-tmp/habitat-lab/data/scene_datasets
  --curriculum-dir /root/autodl-tmp/RVN-Bench-pointnav-curriculum/train_v2
  --validation-dir /root/autodl-tmp/RVN-Bench-pointnav-curriculum/val
  --batch-size 256 --lr 1e-5 --resume
)
if [[ "$run_mode" == smoke ]]; then
  exec /root/miniconda3/envs/habitat/bin/python -u scripts/train_vint_pg_ppo_v2.py "${common[@]}" \
    --output-dir /root/autodl-tmp/RVN-Bench-interactive-runs/ppo_v2_smoke \
    --smoke --rollout-steps 256 --ppo-epochs 4 --eval-episodes 10 --max-steps 16 --max-epochs 2
elif [[ "$run_mode" == train ]]; then
  exec /root/miniconda3/envs/habitat/bin/python -u scripts/train_vint_pg_ppo_v2.py "${common[@]}" \
    --output-dir /root/autodl-tmp/RVN-Bench-interactive-runs/ppo_v2_frozen_bs256 \
    --rollout-steps 8192 --ppo-epochs 4 --eval-episodes 100 --max-steps 500 \
    --max-epochs 50 --transition-epochs 10 --patience 5
else
  echo "Expected smoke or train" >&2
  exit 2
fi
