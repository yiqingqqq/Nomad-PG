#!/usr/bin/env bash
# Sweep goal_distance_cap for ep32@0.25 checkpoint.
set -euo pipefail
cd /root/autodl-tmp/RVN-Bench-native
source /root/miniconda3/etc/profile.d/conda.sh
conda activate habitat
export PYTHONPATH="/root/autodl-tmp/RVN-Bench-native/models/gnms_levin/train:/root/autodl-tmp/RVN-Bench-native/models/diffusion_policy:${PYTHONPATH:-}"

CKPT="/root/autodl-tmp/RVN-Bench-gostandford-huron-runs/rvn_vint_gostandford_huron_indoor/vint_pg_gostandford_huron_indoor_2026_09_05_14_17_59/32.pth"
SPACING=0.25

for CAP in 1 2.25 5 10 20; do
  echo "========== CAP=${CAP}m =========="
  python scripts/run_vint_pg_gate.py \
    --checkpoint "${CKPT}" \
    --source_scenario /root/autodl-tmp/RVN-Bench-native/scenarios/seq_point_goal_nav_eval_scenarios/rvn_val_2_32_2502261324.yaml \
    --scenario_dir /root/autodl-tmp/RVN-Bench-native/scenarios/seq_point_goal_nav_eval_scenarios \
    --data_dir /root/autodl-tmp/RVN-Bench-native/data \
    --output_dir /root/autodl-tmp/RVN-Bench-native/logs/eval_results/vint_pg_gostandford_huron_ep32_capsweep \
    --num_episodes 2 \
    --model_waypoint_spacing ${SPACING} \
    --goal_distance_cap ${CAP} \
    --lookahead_idx 2 \
    --controller trajectory \
    --condition normal
done
echo 'CAP SWEEP DONE'
