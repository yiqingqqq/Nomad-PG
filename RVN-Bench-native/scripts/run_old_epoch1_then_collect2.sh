#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
old_run="$migration_root/runs/nomad_pg_ppo_v2"

cd "$repo_dir"
"$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh" \
  --resume \
  --max-epochs 2

"$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh" \
  --collect-only-epoch 2 \
  --policy-state "$old_run/latest_train.pth" \
  --rollout-output "$old_run/rollout_epoch_002.pth"
