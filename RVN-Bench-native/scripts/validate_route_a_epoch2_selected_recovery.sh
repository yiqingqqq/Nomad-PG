#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
output="$migration_root/runs/route_a_epoch2_recovery_screen"
mkdir -p "$output/logs"

exec "$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh" \
  --checkpoint "$migration_root/runs/stage1/rvn_nomad_pg_stage1_pointnav_local/nomad_pg_stage1_goalmlp_diffusion_2026_09_29_19_39_36/ema_19.pth" \
  --output-dir "$output" \
  --eval-only-policy-state "$migration_root/anchors/route_a_epoch002/policy_state.pth" \
  --eval-only-label alternate1_threshold3_full100 \
  --eval-only-split validation \
  --eval-only-rgb-mode normal \
  --eval-episodes 100 \
  --collision-recovery alternate \
  --recovery-turn-steps 1 \
  --recovery-collision-threshold 3 \
  >"$output/logs/alternate1_threshold3_full100.log" 2>&1
