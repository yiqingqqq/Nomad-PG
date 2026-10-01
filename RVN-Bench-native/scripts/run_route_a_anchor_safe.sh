#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
anchor="$migration_root/anchors/route_a_epoch002/policy_state.pth"
checkpoint="$migration_root/runs/stage1/rvn_nomad_pg_stage1_pointnav_local/nomad_pg_stage1_goalmlp_diffusion_2026_09_29_19_39_36/ema_19.pth"
output_dir="$migration_root/runs/nomad_pg_ppo_v3_anchor_safe"

if [[ ! -f "$anchor" ]]; then
  echo "Missing frozen epoch-2 anchor: $anchor" >&2
  exit 2
fi
if [[ -e "$output_dir/latest_train.pth" && "${1:-}" != "--resume" ]]; then
  echo "Refusing to overwrite an existing run. Pass --resume explicitly." >&2
  exit 2
fi

resume_args=()
if [[ "${1:-}" == "--resume" ]]; then
  resume_args=(--resume)
fi

exec "$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh" \
  --checkpoint "$checkpoint" \
  --output-dir "$output_dir" \
  --anchor-policy "$anchor" \
  --lr 2e-6 \
  --ppo-epochs 1 \
  --anchor-kl-coef 1e-4 \
  --anchor-l2-coef 1e-2 \
  --collision-recovery alternate \
  --recovery-turn-steps 1 \
  --recovery-collision-threshold 3 \
  --min-validation-forward-rate 0.15 \
  --max-validation-turn-streak 250 \
  --black-eval-interval 2 \
  --max-epochs 10 \
  "${resume_args[@]}"
