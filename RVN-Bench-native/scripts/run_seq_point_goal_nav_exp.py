import os
import sys
import argparse
from datetime import datetime

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evaluator.seq_point_goal_nav_evaluator import SeqPointGoalNavEvaluator


def main(args):
    target_model = args.model
    scenario_names = args.scenario_names
    save_plot = args.save_plot
    save_model_plot = args.save_model_plot
    weight_paths = args.weight_paths

    max_ep_iter = 1e6
    max_wp_iter = 100
    obs_size = 1
    curr_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    allow_collision = False

    # =============================== Set Paths ===============================
    curr_path = os.path.dirname(__file__)

    data_dir = os.path.join(curr_path, "../data")
    scenario_dir = os.path.join(
        curr_path, "../scenarios/seq_point_goal_nav_eval_scenarios"
    )
    output_dir = os.path.join(curr_path, "../logs/eval_results/seq_point_goal_nav")

    result_save_dir = os.path.join(output_dir, f"results")

    os.makedirs(result_save_dir, exist_ok=True)

    print(f"data_dir        = {data_dir}")
    print(f"scenario_dir    = {scenario_dir}")
    print(f"scenario_names  = {scenario_names}")
    print(f"output_dir      = {output_dir}")
    print(f"save_plot       = {save_plot}")
    print(f"save_model_plot = {save_model_plot}")
    print(f"weight_paths    = {weight_paths}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for weight_path in weight_paths:
        # =============================== Load Agents ===============================

        if target_model == "nomad" or target_model == "nomad_pg":
            from agents.nomad_agent import load_nomad_model

            if target_model == "nomad":
                config_file_name = "nomad_sq.yaml"
            elif target_model == "nomad_pg":
                config_file_name = "nomad_pg.yaml"

            agent = load_nomad_model(
                save_model_plot,
                None,
                config_file_name=config_file_name,
                weight_path=weight_path,
            )
            obs_size = agent.get_obs_size()
        elif target_model == "nomad_double":
            from agents.double_nomad_agent import load_nomad_model

            agent = load_nomad_model(
                save_model_plot,
                config_file_name="nomad_pg.yaml",
                weight_path_expt=weight_path,
                weight_path_neg="models/pretrained_weights/nomad_pg_neg.pth",
            )
            obs_size = agent.get_obs_size()
        elif target_model == "vint_pg":
            from agents.vint_agent import load_vint_model

            if target_model == "vint_pg":
                config_file_name = "vint_pg.yaml"
            elif target_model == "vint_pg_nmd":
                config_file_name = "vint_pg_nmd.yaml"

            agent = load_vint_model(
                save_model_plot,
                None,
                config_file_name=config_file_name,
                weight_path=weight_path,
            )
        elif target_model == "reslstm_ppo":
            from agents.res_net_policy_agent import ResNetPolicyAgent

            agent = ResNetPolicyAgent(
                weight_path=weight_path,
                device=device,
            )
        elif target_model == "reslstm_ppo_lag":
            from agents.res_net_policy_ppo_lag_agent import ResNetPolicyPpoLagAgent

            agent = ResNetPolicyPpoLagAgent(
                weight_path=weight_path,
                device=device,
            )
        elif target_model == "reslstm_ddppo":
            from agents.ddppo_res_net_lstm_agent import DdppoResNetLstmAgent

            agent = DdppoResNetLstmAgent(
                weight_path=weight_path,
                device=device,
            )
        # =============================== Run Eval ===============================
        for scenario_name in scenario_names:
            weight_name = weight_path.split("/")[-1].split(".")[0]

            if save_model_plot:
                model_plot_save_dir = os.path.join(
                    output_dir,
                    f"debug_model_images/{curr_datetime_str}_{target_model}_{weight_name}_{scenario_name}",
                )
                os.makedirs(model_plot_save_dir, exist_ok=True)
                agent.set_plot_save_dir(model_plot_save_dir)

            plot_save_dir = os.path.join(
                output_dir,
                f"debug_images/{curr_datetime_str}_{target_model}_{weight_name}_{scenario_name}",
            )
            if save_plot:
                os.makedirs(plot_save_dir, exist_ok=True)

            evaluator = SeqPointGoalNavEvaluator(
                scenario_dir=scenario_dir,
                scenario_name=scenario_name,
                data_dir=data_dir,
                rvn_agent=agent,
            )
            evaluator.evaluate(
                target_model=target_model,
                max_ep_steps=max_ep_iter,
                max_wp_steps=max_wp_iter,
                allow_collision=allow_collision,
                save_plot=save_plot,
                result_save_dir=result_save_dir,
                plot_save_dir=plot_save_dir,
                weight_path=weight_path,
                obs_size=obs_size,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model",
        "-m",
        default="reslstm_ppo",
        type=str,
        help="Select the target model to evaluate [nomad, nomad_pg, nomad_double, vint_pg, vint_pg_nmd, reslstm_ddppo, reslstm_ppo, reslstm_ppo_lag]",
    )
    parser.add_argument(
        "--save_plot",
        "-sp",
        default=False,
        type=bool,
        help="Save debug images",
    )
    parser.add_argument(
        "--save_model_plot",
        "-smp",
        default=False,
        type=bool,
        help="Save debug images",
    )
    parser.add_argument(
        "--scenario_names",
        "-sns",
        default=[
            "rvn_val_2_32_2502261324",
            # "rvn_val_20_32_2502261324",
            # "train_2_32_2502261324",
            # "rvn_test_20_32_2508311909",
        ],
        type=str,
        help="Name of the scenario to evaluate",
    )
    parser.add_argument(
        "--weight_paths",
        "-wps",
        default=["agents/weights/resnetlstm_ppo.pth"],
        nargs="+",
        type=str,
        help="Paths to the model weights",
    )
    args = parser.parse_args()

    main(args)
