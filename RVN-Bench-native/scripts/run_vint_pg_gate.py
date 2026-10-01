import argparse
import copy
import os
from collections import Counter

import numpy as np
import torch
import yaml

from agents.vint_agent import VintAgent
import utils.habitat_action_utils as HabitatActionUtils
from evaluator.seq_point_goal_nav_evaluator import SeqPointGoalNavEvaluator
from models.gnms_levin.train.vint_train.models.vint.vint_pointgoal import ViNTPointGoal
from models.gnms_levin.train.vint_train.training.train_eval_loop import load_model

BASE_CONTROLLER = HabitatActionUtils.get_discrete_control_output_from_trajectory_action_wo_stop


def build_agent(
    checkpoint_path: str,
    occlude_rgb: bool,
    lookahead_idx: int,
    controller: str,
    model_waypoint_spacing: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViNTPointGoal(
        context_size=5,
        len_traj_pred=5,
        learn_angle=True,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=512,
        late_fusion=False,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
        pg_rcs=True,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_model(model, "vint_pointgoal", checkpoint)
    model.eval()

    agent = VintAgent(
        model=model,
        device=device,
        pred_horizon=5,
        model_type="vint_pointgoal",
        action_dim=2,
        obs_size=6,
        input_img_size=[96, 96],
        # The network was trained with GoStanford's 0.12 m label spacing.
        # Keep this source-domain normalization at inference; RVN's 0.25 m
        # simulator step does not change the units the checkpoint learned.
        metric_waypoint_spacing=model_waypoint_spacing,
        waypoint_spacing=1,
        normalize=True,
        len_traj_pred=5,
        save_plot=False,
        dataset_name="habitat_hm3d",
        pg_rcs=True,
    )

    original_act = agent.act
    def indexed_controller(action_traj, **kwargs):
        if controller == "yaw":
            return HabitatActionUtils.get_discrete_control_output_from_yaw(
                action_traj, lookahead_point_idx=lookahead_idx
            )
        kwargs["lookahead_point_idx"] = lookahead_idx
        return BASE_CONTROLLER(action_traj, **kwargs)

    HabitatActionUtils.get_discrete_control_output_from_trajectory_action_wo_stop = (
        indexed_controller
    )
    actions = []

    def recorded_act(obs, info):
        model_obs = copy.copy(obs)
        if occlude_rgb:
            model_obs["rgb"] = np.zeros_like(obs["rgb"])
        if args_goal_distance_cap := getattr(agent, "goal_distance_cap", None):
            pointgoal = np.array(obs["pointgoal_with_gps_compass"], copy=True)
            pointgoal[0] = min(pointgoal[0], args_goal_distance_cap)
            model_obs["pointgoal_with_gps_compass"] = pointgoal
        with torch.inference_mode():
            action = original_act(model_obs, info)
        actions.append(action)
        return action

    agent.act = recorded_act
    agent.recorded_actions = actions
    return agent


def create_fixed_scenario(source_path: str, output_dir: str, num_episodes: int):
    scenario_name = f"rvn_val_fixed_{num_episodes}_32_stage1"
    output_path = os.path.join(output_dir, f"{scenario_name}.yaml")
    if os.path.isfile(output_path):
        return scenario_name
    with open(source_path, "r") as stream:
        scenario = yaml.safe_load(stream)
    scenario["episodes"] = scenario["episodes"][:num_episodes]
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as stream:
        yaml.safe_dump(scenario, stream, sort_keys=False)
    return scenario_name


def run_condition(args, scenario_name: str, occlude_rgb: bool):
    rgb_condition = "rgb_occluded" if occlude_rgb else "rgb_normal"
    cap_name = "uncapped" if args.goal_distance_cap is None else f"cap{args.goal_distance_cap:g}m"
    condition = f"{rgb_condition}_{args.controller}_lookahead{args.lookahead_idx}_{cap_name}"
    condition_dir = os.path.join(args.output_dir, condition)
    result_dir = os.path.join(condition_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    agent = build_agent(
        args.checkpoint,
        occlude_rgb,
        args.lookahead_idx,
        args.controller,
        args.model_waypoint_spacing,
    )
    agent.goal_distance_cap = args.goal_distance_cap
    evaluator = SeqPointGoalNavEvaluator(
        scenario_dir=args.scenario_dir,
        scenario_name=scenario_name,
        data_dir=args.data_dir,
        rvn_agent=agent,
    )
    evaluator.evaluate(
        target_model=f"vint_pg_stage1_{condition}",
        max_ep_steps=int(1e6),
        max_wp_steps=100,
        allow_collision=False,
        save_plot=False,
        result_save_dir=result_dir,
        weight_path=args.checkpoint,
        obs_size=1,
        seed=1,
    )

    with open(evaluator._result_file_path, "r") as stream:
        result = yaml.safe_load(stream)
    counts = Counter(agent.recorded_actions)
    result["action_counts"] = dict(counts)
    result["num_unique_actions"] = len(counts)
    result["num_actions"] = len(agent.recorded_actions)
    result["rgb_occluded"] = occlude_rgb
    result["lookahead_idx"] = args.lookahead_idx
    result["controller"] = args.controller
    result["goal_distance_cap_m"] = args.goal_distance_cap
    result["model_waypoint_spacing_m"] = args.model_waypoint_spacing
    with open(evaluator._result_file_path, "w") as stream:
        yaml.safe_dump(result, stream, sort_keys=False)
    print(f"GATE_RESULT condition={condition} file={evaluator._result_file_path}")
    print(
        f"GATE_METRICS condition={condition} sr={result['success_rate']:.6f} "
        f"spl={result['spl']:.6f} unique_actions={len(counts)} action_counts={dict(counts)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_scenario", required=True)
    parser.add_argument("--scenario_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument(
        "--model_waypoint_spacing",
        type=float,
        default=0.12,
        help="Training-data spacing used to normalize PointGoal input (GoStanford: 0.12 m).",
    )
    parser.add_argument("--lookahead_idx", type=int, default=2)
    parser.add_argument(
        "--controller", choices=["trajectory", "yaw"], default="trajectory"
    )
    parser.add_argument("--goal_distance_cap", type=float, default=None)
    parser.add_argument(
        "--condition", choices=["normal", "occluded", "both"], default="both"
    )
    args = parser.parse_args()
    scenario_name = create_fixed_scenario(
        args.source_scenario, args.scenario_dir, args.num_episodes
    )
    if args.condition in ("normal", "both"):
        run_condition(args, scenario_name, occlude_rgb=False)
    if args.condition in ("occluded", "both"):
        run_condition(args, scenario_name, occlude_rgb=True)


if __name__ == "__main__":
    main()
