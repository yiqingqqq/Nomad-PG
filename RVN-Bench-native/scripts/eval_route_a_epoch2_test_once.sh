#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
output="$migration_root/runs/route_a_epoch2_frozen_test"
runner="$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh"
checkpoint="$migration_root/runs/stage1/rvn_nomad_pg_stage1_pointnav_local/nomad_pg_stage1_goalmlp_diffusion_2026_09_29_19_39_36/ema_19.pth"
anchor="$migration_root/anchors/route_a_epoch002/policy_state.pth"

if find "$output/eval" -type f -name 'test_*.json' -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing to rerun frozen test: test output already exists in $output" >&2
  exit 2
fi
mkdir -p "$output/logs"

run_condition() {
  local condition="$1"
  local rgb_mode=normal
  local goal_mode=normal
  if [[ "$condition" == black ]]; then
    rgb_mode=black
  elif [[ "$condition" == angle_flip ]]; then
    goal_mode=angle_flip
  fi
  echo "START $condition"
  "$runner" \
    --checkpoint "$checkpoint" \
    --output-dir "$output" \
    --eval-only-policy-state "$anchor" \
    --eval-only-label selected_epoch002 \
    --eval-only-split test \
    --eval-only-rgb-mode "$rgb_mode" \
    --eval-only-goal-mode "$goal_mode" \
    --eval-episodes 100 \
    --collision-recovery alternate \
    --recovery-turn-steps 1 \
    --recovery-collision-threshold 3 \
    >"$output/logs/$condition.log" 2>&1
  echo "DONE $condition"
}

run_condition normal
run_condition black
run_condition angle_flip
