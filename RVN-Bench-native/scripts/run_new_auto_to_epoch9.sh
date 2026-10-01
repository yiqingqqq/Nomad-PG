#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
run_dir="$migration_root/runs/nomad_pg_ppo_v2_stage1_new"
checkpoint="$migration_root/runs/stage1/rvn_nomad_pg_stage1_pointnav_local/nomad_pg_stage1_goalmlp_diffusion_2026_09_29_19_39_36/ema_19.pth"

cd "$repo_dir"
"$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh" \
  --checkpoint "$checkpoint" \
  --output-dir "$run_dir" \
  --resume \
  --max-epochs 10

"$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh" \
  --checkpoint "$checkpoint" \
  --output-dir "$run_dir" \
  --collect-only-epoch 10 \
  --policy-state "$run_dir/latest_train.pth" \
  --rollout-output "$run_dir/rollout_epoch_010.pth"
