#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
runner="$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh"
checkpoint="$migration_root/runs/stage1/rvn_nomad_pg_stage1_pointnav_local/nomad_pg_stage1_goalmlp_diffusion_2026_09_29_19_39_36/ema_19.pth"
anchor="$migration_root/anchors/route_a_epoch002/policy_state.pth"
output="$migration_root/runs/route_a_epoch2_recovery_screen"
mkdir -p "$output/logs"

configs=(
  "goalturn1_threshold1 goal_turn 1 1"
  "goalturn1_threshold3 goal_turn 1 3"
  "goalturn3_threshold1 goal_turn 3 1"
  "alternate1_threshold3 alternate 1 3"
)

for config in "${configs[@]}"; do
  read -r label mode turns threshold <<<"$config"
  result="$output/eval/$label/validation_normal.json"
  if [[ -s "$result" ]]; then
    echo "SKIP $label (result exists)"
    continue
  fi
  echo "START $label"
  "$runner" \
    --checkpoint "$checkpoint" \
    --output-dir "$output" \
    --eval-only-policy-state "$anchor" \
    --eval-only-label "$label" \
    --eval-only-split validation \
    --eval-only-rgb-mode normal \
    --eval-episodes 30 \
    --collision-recovery "$mode" \
    --recovery-turn-steps "$turns" \
    --recovery-collision-threshold "$threshold" \
    >"$output/logs/$label.log" 2>&1
  echo "DONE $label"
done
