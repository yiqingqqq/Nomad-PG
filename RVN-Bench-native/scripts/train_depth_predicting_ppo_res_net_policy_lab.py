import os
import sys
import argparse
from datetime import datetime

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from algo.depth_predicting_ppo_res_net_policy import DepthPredictingPpoTrainerResNetPolicyLab
from rvn_envs.vector_env_utils import get_seq_point_goal_nav_venv
from rvn_envs.seq_point_goal_nav_env import SeqPointGoalNavEnv

sys.path.append(os.path.join("..", "Depth-Anything-V2", "metric_depth"))

from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
encoder = 'vits' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model
depth_pseudo_labeling_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
depth_pseudo_labeling_model.load_state_dict(torch.load(f'../Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
depth_pseudo_labeling_model = depth_pseudo_labeling_model.eval()
depth_pseudo_labeling_model = depth_pseudo_labeling_model.to("cuda:0")



def main(args):
    use_wandb = args.wandb
    curr_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "train_reslstm_r4"

    envs = get_seq_point_goal_nav_venv(num_environments=4)

    env_version = SeqPointGoalNavEnv.VERSION

    weight_save_dir = f"logs/weights/rvn_{env_version}/{run_name}/{curr_datetime_str}"
    render_save_dir = (
        f"logs/debug_images/rvn_{env_version}/{run_name}/{curr_datetime_str}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo = DepthPredictingPpoTrainerResNetPolicyLab(
        depth_pseudo_labeling_model = depth_pseudo_labeling_model,
        envs=envs,
        hyper_parameters={
            "save_freq": 32,
            "timesteps_per_batch": 128,
            "max_timesteps_per_episode": 1000,
            "gamma": 0.99,
            "lam": 0.95,
            "lr": 2.5e-4,
            "eps": 1e-5,
            "clip_ratio": 0.2,
            "n_updates_per_iteration": 2,
            "n_mini_batch": 2,
            "value_loss_coeff": 0.5,
            "entropy_loss_coeff": 0.01,
            "normalize_advantage": False,
        },
        depth_loss_coeff = 1.0,
        obs_dim=2,
        action_dim=4,
        hidden_size=512,
        weight_save_dir=weight_save_dir,
        render_save_dir=render_save_dir,
        device=device,
    )
    ppo.learn(
        int(1e9),
        use_wandb=use_wandb,
        # wandb_project=f"rvn-{env.VERSION}",
        wandb_project="rvn_bench",
        run_name=run_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--wandb",
        "-w",
        action="store_true",
        help="Use wandb for logging.",
    )
    args = parser.parse_args()
    main(args)
