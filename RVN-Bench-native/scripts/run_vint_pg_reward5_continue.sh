#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RVN-Bench-native
export PYTHONPATH=/root/autodl-tmp/RVN-Bench-native:/root/autodl-tmp/RVN-Bench-native/models/diffusion_policy:/root/autodl-tmp/RVN-Bench-native/models/gnms_levin/train
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
exec /root/miniconda3/envs/habitat/bin/python -u scripts/train_vint_pg_reward5_continue.py \
  --checkpoint /root/autodl-fs/RVN-Bench-gostanford-mirror-runs/rvn_vint_go_stanford/vint_pg_gostanford_mirror_ft5_2026_09_23_21_36_03/1.pth \
  --task-config /root/autodl-tmp/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml \
  --scenes-dir /root/autodl-tmp/habitat-lab/data/scene_datasets \
  --curriculum-dir /root/autodl-tmp/RVN-Bench-pointnav-curriculum/train_v2 \
  --validation-dir /root/autodl-tmp/RVN-Bench-pointnav-curriculum/val \
  --output-dir /root/autodl-tmp/RVN-Bench-interactive-runs/ppo_reward5_bs256 \
  --success-reward 5 --batch-size 256 --rollout-steps 8192 --ppo-epochs 4 \
  --eval-episodes 100 --max-steps 500 --max-epochs 10 --transition-epochs 10 \
  --lr 1e-5 --patience 3 --seed 270928 --skip-final-test --resume
