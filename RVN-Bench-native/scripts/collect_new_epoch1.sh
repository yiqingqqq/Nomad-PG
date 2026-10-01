#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
new_run="$migration_root/runs/nomad_pg_ppo_v2_stage1_new"
new_checkpoint="$migration_root/runs/stage1/rvn_nomad_pg_stage1_pointnav_local/nomad_pg_stage1_goalmlp_diffusion_2026_09_29_19_39_36/ema_19.pth"

cd "$repo_dir"
"$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh" \
  --checkpoint "$new_checkpoint" \
  --output-dir "$new_run" \
  --collect-only-epoch 1 \
  --policy-state "$new_run/latest_train.pth" \
  --rollout-output "$new_run/rollout_epoch_001.pth"
