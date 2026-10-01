#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
export PYTHONPATH="$repo_dir:$repo_dir/models/diffusion_policy:$repo_dir/models/gnms_levin/train"
export MPLCONFIGDIR="$migration_root/runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$repo_dir/models/gnms_levin/train"
exec "$migration_root/.conda-env/bin/python" train.py --config config/nomad_pg_stage1_pointnav_local.yaml "$@"
