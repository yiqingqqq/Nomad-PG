#!/usr/bin/env bash
# Source this file from RVN-Bench-native: source scripts/activate_nomad_pg_env.sh
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration_root="$(cd "$repo_dir/.." && pwd)"
export PATH="$migration_root/.conda-env/bin:$PATH"
export PYTHONPATH="$repo_dir:$repo_dir/models/diffusion_policy:$repo_dir/models/gnms_levin/train:$migration_root/habitat-lab/habitat-lab"
export MPLCONFIGDIR="$migration_root/runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR"
echo "NoMaD-PG environment: $migration_root/.conda-env"
python --version
