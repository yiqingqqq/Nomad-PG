#!/usr/bin/env bash
# Run RVN gate with configurable checkpoint + spacing.
set -euo pipefail
cd /root/autodl-tmp/RVN-Bench-native
source /root/miniconda3/etc/profile.d/conda.sh
conda activate habitat
export PYTHONPATH="/root/autodl-tmp/RVN-Bench-native/models/gnms_levin/train:/root/autodl-tmp/RVN-Bench-native/models/diffusion_policy:${PYTHONPATH:-}"

CKPT="${1:?checkpoint}"
SPACING="${2:?spacing}"
TAG="${3:?tag}"

python scripts/run_vint_pg_gate.py \
  --checkpoint "${CKPT}" \
  --source_scenario /root/autodl-tmp/RVN-Bench-native/scenarios/seq_point_goal_nav_eval_scenarios/rvn_val_2_32_2502261324.yaml \
  --scenario_dir /root/autodl-tmp/RVN-Bench-native/scenarios/seq_point_goal_nav_eval_scenarios \
  --data_dir /root/autodl-tmp/RVN-Bench-native/data \
  --output_dir /root/autodl-tmp/RVN-Bench-native/logs/eval_results/vint_pg_gostandford_huron_${TAG}_gate \
  --num_episodes 2 \
  --model_waypoint_spacing ${SPACING} \
  --lookahead_idx 2 \
  --controller trajectory \
  --condition normal
