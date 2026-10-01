#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
run_dir="$migration_root/runs/nomad_pg_ppo_v2"

cd "$repo_dir"
"$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh" \
  --resume \
  --max-epochs 10

"$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh" \
  --collect-only-epoch 10 \
  --policy-state "$run_dir/latest_train.pth" \
  --rollout-output "$run_dir/rollout_epoch_010.pth"
