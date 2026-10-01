#!/usr/bin/env bash
set -euo pipefail

migration_root="/home/yiqing/Robotics/Nav_robot/nomadpg_pointnav_migration_20260928"
repo_dir="$migration_root/RVN-Bench-native"
python_bin="$migration_root/.conda-env/bin/python"
stage1_pid="114999"
run_dir="$migration_root/runs/stage1/rvn_nomad_pg_stage1_pointnav_local/nomad_pg_stage1_goalmlp_diffusion_2026_09_29_19_39_36"
checkpoint="$run_dir/ema_19.pth"
eval_dir="$migration_root/runs/nomad_pg_stage1_new_zero_shot"
ppo_dir="$migration_root/runs/nomad_pg_ppo_v2_stage1_new"
task_config="$migration_root/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml"
scenes_dir="$migration_root/runtime/habitat-data/scene_datasets"
episode_file="$migration_root/protocol/validation.json.gz"

export PYTHONPATH="$repo_dir:$repo_dir/models/diffusion_policy:$repo_dir/models/gnms_levin/train:$migration_root/habitat-lab/habitat-lab"
export MPLCONFIGDIR="$migration_root/runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR" "$eval_dir"

echo "[$(date --iso-8601=seconds)] waiting for Stage 1 PID $stage1_pid"
while kill -0 "$stage1_pid" 2>/dev/null; do
  sleep 30
done

echo "[$(date --iso-8601=seconds)] Stage 1 process ended"
if [[ ! -s "$checkpoint" ]]; then
  echo "missing final checkpoint: $checkpoint" >&2
  exit 1
fi
sha256sum "$checkpoint" > "$eval_dir/checkpoint.sha256"

cd "$repo_dir"
for condition in normal black angle_flip; do
  rgb_mode="normal"
  goal_mode="normal"
  if [[ "$condition" == "black" ]]; then
    rgb_mode="black"
  elif [[ "$condition" == "angle_flip" ]]; then
    goal_mode="angle_flip"
  fi
  echo "[$(date --iso-8601=seconds)] zero-shot condition=$condition"
  "$python_bin" scripts/eval_nomad_pg_hm3d_pointnav_paired.py \
    --checkpoint "$checkpoint" \
    --task-config "$task_config" \
    --episodes "$migration_root/protocol" \
    --episode-file "$episode_file" \
    --scenes-dir "$scenes_dir" \
    --output "$eval_dir/${condition}_100.json" \
    --num-episodes 100 \
    --max-steps 500 \
    --seed 270928 \
    --success-distance 0.2 \
    --model-waypoint-spacing 0.12 \
    --lookahead-idx 2 \
    --controller trajectory \
    --rgb-mode "$rgb_mode" \
    --goal-mode "$goal_mode" \
    --pg-rcs \
    > "$eval_dir/${condition}_100.log" 2>&1
done

echo "[$(date --iso-8601=seconds)] starting PPO v2 from $checkpoint"
exec scripts/run_nomad_pg_ppo_v2_local.sh \
  --checkpoint "$checkpoint" \
  --output-dir "$ppo_dir"
