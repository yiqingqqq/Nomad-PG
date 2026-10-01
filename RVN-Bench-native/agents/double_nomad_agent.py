import os
import sys
import yaml
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from agents.agent import Agent, ObsType, InfoType


import models.model_utils.sequor_gnm_utils as SequorGnmUtils
import utils.habitat_gnm_utils as HabitatGnmUtils
import utils.habitat_action_utils as HabitatActionUtils


class DoubleNomadAgent(Agent):
    def __init__(
        self,
        model_type="nomad",
        action_dim=2,
        num_samples=8,
        pred_horizon=None,
        device=None,
        ema_model_avg_expt=None,
        noise_scheduler_expt=None,
        ema_model_avg_neg=None,
        noise_scheduler_neg=None,
        obs_size=4,
        input_img_size=None,
        metric_waypoint_spacing: float = 0.25,
        waypoint_spacing: int = 1,
        normalize: bool = True,
        save_plot=False,
        dataset_name: str = "habitat",
        plot_save_dir="",
    ):
        assert model_type in ["nomad", "nomad_pointgoal"]

        self._model_type = model_type

        self._action_dim = action_dim
        self._num_samples = num_samples
        self._pred_horizon = pred_horizon
        self._device = device

        self._ema_model_avg_expt = ema_model_avg_expt
        self._noise_scheduler_expt = noise_scheduler_expt
        self._ema_model_avg_neg = ema_model_avg_neg
        self._noise_scheduler_neg = noise_scheduler_neg

        self._obs_size = obs_size
        self._input_img_size = input_img_size
        # self._obs_image_context = HabitatGnmUtils.ObsImageContext(
        #     context_size=obs_size, input_img_size=input_img_size, device=device
        # )
        self._metric_waypoint_spacing = metric_waypoint_spacing
        self._normalize = normalize
        self._waypoint_spacing = waypoint_spacing

        self._save_plot = save_plot
        self._dataset_name = dataset_name
        self._plot_save_dir = plot_save_dir

        self._num_reset = 0
        self._num_act = 0

        print("DoubleNomadAgent initialized")
        print(f"\taction_dim:\t{self._action_dim}")
        print(f"\tnum_samples:\t{self._num_samples}")
        print(f"\tpred_horizon:\t{self._pred_horizon}")
        print(f"\tdevice:\t\t{self._device}")

    def reset(self):
        self._num_reset += 1
        self._num_act = 0

        # self._obs_image_context.reset()

    def act(
            self,
            obs: ObsType,
            info: InfoType = None,
            rgb_obs_preprocessed=False,
        ) -> str:
        self._num_act += 1

        # self._obs_image_context.add_obs_image(obs["rgb"])
        # obs_input = self._obs_image_context.get_context()
        if not rgb_obs_preprocessed:
            obs_input = HabitatGnmUtils.get_gnm_obs_input_from_rgb_np(
                obs["rgb"], self._input_img_size, self._device
            )
        else:
            obs_input = obs["rgb"]


        goal_gps_compass = obs["pointgoal_with_gps_compass"]
        rcs_input = (
            HabitatGnmUtils.get_gnm_r_c_s_goal_pos_input_from_gps_compass(
                goal_gps_compass,
                self._normalize,
                self._device,
                self._metric_waypoint_spacing,
                self._waypoint_spacing,
                self._pred_horizon,
            )
        )

        goal_pos_input = HabitatGnmUtils.get_gnm_goal_pos_input_from_gps_compass(
            goal_gps_compass,
            self._normalize,
            self._device,
            self._metric_waypoint_spacing,
            self._waypoint_spacing,
        )

        model_output_dict_expt = SequorGnmUtils.pointgoal_model_output(
            self._ema_model_avg_expt,
            self._noise_scheduler_expt,
            obs_input,
            rcs_input,
            self._pred_horizon,
            self._action_dim,
            self._num_samples,
            self._device,
        )

        model_output_dict_neg = SequorGnmUtils.pointgoal_model_output(
            self._ema_model_avg_neg,
            self._noise_scheduler_neg,
            obs_input,
            goal_pos_input,
            self._pred_horizon,
            self._action_dim,
            self._num_samples,
            self._device,
        )

        gc_action_list_expt = (
            model_output_dict_expt["gc_actions"].detach().cpu().numpy()
        )
        gc_action_list_neg = model_output_dict_neg["gc_actions"].detach().cpu().numpy()

        action_traj_idx = self._select_action_by_cor(
            gc_action_list_expt, gc_action_list_neg
        )

        action = HabitatActionUtils.get_discrete_control_output_from_trajectory_action_wo_stop(
            action_traj=gc_action_list_expt[action_traj_idx], lookahead_point_idx=2
        )

        if self._save_plot:
            uc_action_list_expt = (
                model_output_dict_expt["uc_actions"].detach().cpu().numpy()
            )
            plot_title = f"nomad_{self._num_reset}_{self._num_act}"
            SequorGnmUtils.plot_trajs_wo_label(
                gc_actions_list=gc_action_list_expt,
                uc_actions_list=gc_action_list_neg,
                obs_img_to_plot=obs["rgb"],
                goal_img_to_plot=info["goal_rgb"],
                dataset_name=self._dataset_name,
                save_plot=self._save_plot,
                plot_save_path=os.path.join(self._plot_save_dir, f"{plot_title}.png"),
                metric_waypoint_spacing=self._metric_waypoint_spacing,
                action_traj_idx=action_traj_idx,
                fig_title=f"{plot_title}\n{action}",
            )

        return action

    def get_obs_size(self) -> int:
        return self._obs_size

    def get_input_img_size(self) -> tuple:
        return self._input_img_size
    
    def set_plot_save_dir(self, plot_save_dir):
        self._plot_save_dir = plot_save_dir
        os.makedirs(self._plot_save_dir, exist_ok=True)

    def _select_action_by_cor(self, action_list_expt, action_list_neg):
        cor_losses = []
        for action_traj in action_list_expt:
            cor_loss = self._calculate_cor_loss(
                action_traj, action_list_expt, action_list_neg
            )
            cor_losses.append(cor_loss)
        # print(f"cor_losses: {cor_losses}")
        # print(f"min index: {np.argmin(cor_losses)}")
        action_traj_idx = np.argmax(cor_losses)
        # action_traj = action_list_expt[np.argmin(cor_losses)]
        return action_traj_idx

    def _calculate_cor_loss(
        self, target_action, action_list_expt, action_list_neg, alpha=0.5
    ):
        d_s_e = np.sqrt(1 / (len(action_list_expt) - 1)) * np.linalg.norm(
            action_list_expt - target_action
        )

        d_s_n = np.sqrt(1 / (len(action_list_neg) - 1)) * np.linalg.norm(
            action_list_neg - target_action
        )

        expt_term = np.power((1 + d_s_e / alpha), -(alpha + 1) * 0.5)
        neg_term = np.power((1 + d_s_n / alpha), -(alpha + 1) * 0.5)

        cor_loss = expt_term / (expt_term + neg_term)

        return cor_loss


def load_nomad_model(
    save_plot,
    plot_save_dir=None,
    config_file_name="nomad_pg.yaml",
    weight_path_expt: Optional[str] = "models/pretrained_weights/nomad_pg_f3_49.pth",
    weight_path_neg: Optional[
        str
    ] = "models/pretrained_weights/nomad_pg_neg_f_10_49.pth",
):
    assert config_file_name in ["nomad.yaml", "nomad_sq.yaml", "nomad_pg.yaml"]

    # =============================== Load NoMAD ===============================
    action_dim = 2

    with open("models/gnms_levin/train/config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    with open(f"models/gnms_levin/train/config/{config_file_name}", "r") as f:
        user_config = yaml.safe_load(f)

    model_type = user_config["model_type"]

    print(
        f"Loading nomad model: model_type: {model_type} weight_path: {weight_path_expt} {weight_path_neg}"
    )

    config = default_config
    config.update(user_config)

    config["gpu_ids"], device = SequorGnmUtils.set_cuda_visible_devices_and_get_device(
        config
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True
    cudnn.benchmark = True  # good if input sizes don't vary

    ema_model_avg_expt, noise_scheduler_expt = SequorGnmUtils.load_nomad_model(
        model_type=model_type,
        encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
        num_diffusion_iters=config["num_diffusion_iters"],
        action_dim=action_dim,
        weight_path=weight_path_expt,
        gpu_ids=config["gpu_ids"],
        device=device,
    )
    ema_model_avg_neg, noise_scheduler_neg = SequorGnmUtils.load_nomad_model(
        model_type=model_type,
        encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
        num_diffusion_iters=config["num_diffusion_iters"],
        action_dim=action_dim,
        weight_path=weight_path_neg,
        gpu_ids=config["gpu_ids"],
        device=device,
        pg_rcs=False,
    )
    ema_model_avg_expt.eval()
    ema_model_avg_neg.eval()

    return DoubleNomadAgent(
        model_type=model_type,
        action_dim=action_dim,
        num_samples=8,
        pred_horizon=config["len_traj_pred"],
        device=device,
        ema_model_avg_expt=ema_model_avg_expt,
        noise_scheduler_expt=noise_scheduler_expt,
        ema_model_avg_neg=ema_model_avg_neg,
        noise_scheduler_neg=noise_scheduler_neg,
        obs_size=4,
        input_img_size=config["image_size"],
        save_plot=save_plot,
        dataset_name="habitat_hm3d",
        metric_waypoint_spacing=0.25,
        plot_save_dir=plot_save_dir,
    )


def main(args):
    agent = load_nomad_model(
        save_plot=True,
        plot_save_dir="logs/debug_images",
        config_file_name=args.config_file_name,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--config_file_name",
        "-c",
        default="nomad_sq.yaml",
        type=str,
        help="Select the target model config to evaluate [nomad.yaml, nomad_sq.yaml, nomad_pg.yaml]",
    )
    args = parser.parse_args()

    main(args)
