#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-$migration_root/.conda-env/bin/python}"
checkpoint="$migration_root/checkpoints/rvn_nomad_gostandford_huron_stage1_resume/nomad_pg_stage1_gostandford_huron_resume_2026_09_24_11_38_06/ema_3.pth"
export PYTHONPATH="$repo_dir:$repo_dir/models/diffusion_policy:$repo_dir/models/gnms_levin/train:$migration_root/habitat-lab/habitat-lab"
export MPLCONFIGDIR="$migration_root/runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR"

cd "$repo_dir"
exec "$python_bin" scripts/train_nomad_pg_ppo_v2.py \
  --checkpoint "$checkpoint" \
  --task-config "$migration_root/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml" \
  --scenes-dir "$migration_root/runtime/habitat-data/scene_datasets" \
  --curriculum-dir "$migration_root/pointnav-curriculum/train_v2" \
  --protocol-dir "$migration_root/protocol" \
  --output-dir "$migration_root/runs/nomad_pg_ppo_v2" \
  "$@"
