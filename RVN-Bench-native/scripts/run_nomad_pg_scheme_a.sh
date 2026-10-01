#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
mode="${1:-formal}"
anchor="$migration_root/anchors/route_a_epoch002/policy_state.pth"
checkpoint="$migration_root/runs/stage1/rvn_nomad_pg_stage1_pointnav_local/nomad_pg_stage1_goalmlp_diffusion_2026_09_29_19_39_36/ema_19.pth"

case "$mode" in
  smoke)
    output_dir="$migration_root/runs/nomad_pg_ppo_scheme_a_smoke"
    extra=(--rollout-steps 512 --eval-episodes 4 --max-epochs 1)
    ;;
  formal)
    output_dir="$migration_root/runs/nomad_pg_ppo_scheme_a"
    extra=(--rollout-steps 8192 --eval-episodes 100 --max-epochs 10)
    ;;
  *) echo "Usage: $0 [smoke|formal]" >&2; exit 2 ;;
esac

if [[ ! -f "$anchor" ]]; then
  echo "Missing frozen epoch-2 anchor: $anchor" >&2
  exit 2
fi
resume_args=()
if [[ "${2:-}" == "--resume" ]]; then
  resume_args=(--resume)
fi
if [[ -e "$output_dir/config.json" && ${#resume_args[@]} -eq 0 ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 2
fi

exec "$repo_dir/scripts/run_nomad_pg_ppo_v2_local.sh" \
  --checkpoint "$checkpoint" \
  --output-dir "$output_dir" \
  --anchor-policy "$anchor" \
  --train-scheme scheme_a \
  --batch-size 128 \
  --ppo-epochs 1 \
  --actor-lr 2e-6 \
  --critic-lr 1e-4 \
  --diffusion-last-up-lr 1e-6 \
  --diffusion-output-lr 2e-6 \
  --clip 0.2 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --anchor-kl-coef 1e-4 \
  --anchor-l2-coef 1e-2 \
  --collision-recovery alternate \
  --recovery-turn-steps 1 \
  --recovery-collision-threshold 3 \
  --min-validation-sr 0.33 \
  --min-validation-forward-rate 0.15 \
  --max-validation-turn-streak 250 \
  --min-normal-black-sr-gap 0.10 \
  --max-action-kl 0.05 \
  --no-improvement-patience 2 \
  --black-eval-interval 2 \
  --baseline-eval-source "$migration_root/runs/nomad_pg_ppo_v3_anchor_safe/eval/baseline" \
  "${resume_args[@]}" \
  "${extra[@]}"
