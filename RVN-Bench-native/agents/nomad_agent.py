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
from PIL import Image
from datetime import datetime

class NomadAgent(Agent):
    def __init__(
        self,
        model_type="nomad",
        action_dim=2,
        num_samples=8,
        pred_horizon=None,
        device=None,
        ema_model_avg=None,
        noise_scheduler=None,
        obs_size=4,
        input_img_size=None,
        metric_waypoint_spacing: float = 0.25,
        waypoint_spacing: int = 1,
        normalize: bool = True,
        save_plot=False,
        dataset_name: str = "habitat",
        plot_save_dir="",
        critic=None,
    ):
        assert model_type in ["nomad", "nomad_pointgoal"]

        self._model_type = model_type

        self._action_dim = action_dim
        self._num_samples = num_samples
        self._pred_horizon = pred_horizon
        self._device = device

        self._ema_model_avg = ema_model_avg
        self._noise_scheduler = noise_scheduler

        self._input_img_size = input_img_size
        self._obs_size = obs_size
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

        if critic is not None:
            assert model_type == "nomad_pointgoal", (
                "Critic is only available for nomad_pointgoal model type, "
                f"got {model_type}"
            )
            self._critic = critic

        print("NomadAgent initialized")
        print(f"\taction_dim:\t{self._action_dim}")
        print(f"\tnum_samples:\t{self._num_samples}")
        print(f"\tpred_horizon:\t{self._pred_horizon}")
        print(f"\tdevice:\t\t{self._device}")

    def reset(self):
        self._num_reset += 1
        self._num_act = 0

        # self._obs_image_context.reset()

    
    def get_value_preds(
        self,
        obs: ObsType,
        gps_obs_preprocessed=False,
    ) -> Optional[torch.Tensor]:
        if self._model_type != "nomad_pointgoal":
            raise NotImplementedError(
                "Value estimation is only implemented for nomad_pointgoal model type."
            )
        if self._critic is None:
            raise ValueError(
                "Critic is not set. Please provide a critic model for value estimation."
            )
        # self._obs_image_context.add_obs_image(obs["rgb"])
        if gps_obs_preprocessed:
            goal_pos_input = obs["pointgoal_with_gps_compass"]
        else:
            goal_pos_input = HabitatGnmUtils.get_gnm_goal_pos_input_from_gps_compass(
                obs["pointgoal_with_gps_compass"],
                self._normalize,
                self._device,
                self._metric_waypoint_spacing,
                self._waypoint_spacing,
            )
        value_preds = SequorGnmUtils.pointgoal_model_critic_output(
            self._critic,
            obs["rgb"],
            goal_pos_input,
            self._device,
        )
        
        return value_preds
    
    def get_action_log_probs(
            self,
            obs_input: ObsType,
            chains: torch.Tensor, # [B, K+1, T, 2]
            gps_obs_preprocessed=False,
    ) -> Optional[torch.Tensor]:
        if self._model_type != "nomad_pointgoal":
            raise NotImplementedError(
                "Log probabilities are only implemented for nomad_pointgoal model type."
            )
        if self._critic is None:
            raise ValueError(
                "Critic is not set. Please provide a critic model for log probability estimation."
            )
        # self._obs_image_context.add_obs_image(obs_input["rgb"])
        if gps_obs_preprocessed:
            goal_pos_input = obs_input["pointgoal_with_gps_compass"]
        else:
            goal_pos_input = HabitatGnmUtils.get_gnm_goal_pos_input_from_gps_compass(
                obs_input["pointgoal_with_gps_compass"],
                self._normalize,
                self._device,
                self._metric_waypoint_spacing,
                self._waypoint_spacing,
            )
        logprobs = SequorGnmUtils.pointgoal_model_logprob_output(
            self._ema_model_avg,
            self._noise_scheduler,
            obs_input['rgb'],
            goal_pos_input,
            chains,
            self._device
        )

        return logprobs


    def get_subsample_action_log_probs_and_entropy(
        self,
        obs_input: ObsType,
        chains_prev: torch.Tensor,  # [B, T, 2]
        chains_next: torch.Tensor,  # [B, T, 2]
        prev_to_next_chain_timesteps: torch.Tensor, # [B]
        gps_obs_preprocessed=False
    ):
        if self._model_type != "nomad_pointgoal":
            raise NotImplementedError(
                "Log probabilities are only implemented for nomad_pointgoal model type."
            )
        if self._critic is None:
            raise ValueError(
                "Critic is not set. Please provide a critic model for log probability estimation."
            )
        # self._obs_image_context.add_obs_image(obs_input["rgb"])
        if gps_obs_preprocessed:
            goal_pos_input = obs_input["pointgoal_with_gps_compass"]
        else:
            goal_pos_input = HabitatGnmUtils.get_gnm_goal_pos_input_from_gps_compass(
                obs_input["pointgoal_with_gps_compass"],
                self._normalize,
                self._device,
                self._metric_waypoint_spacing,
                self._waypoint_spacing,
            )
        logprobs, entropy = SequorGnmUtils.pointgoal_model_subsample_logprob_and_entropy_output(
            self._ema_model_avg,
            self._noise_scheduler,
            obs_input['rgb'],
            goal_pos_input,
            chains_prev,
            chains_next,
            prev_to_next_chain_timesteps,
            self._device
        )
        return logprobs, entropy
            


    def act(
        self,
        obs: ObsType,
        info: InfoType = None,
        rgb_obs_preprocessed=False,
        gps_obs_preprocessed=False,
        get_action_as_batch_idx=False,
        return_chain=False # use only for training dppo
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

        if self._model_type == "nomad":
            goal_img_input = HabitatGnmUtils.get_gnm_goal_input_from_rgb_np(
                info["goal_rgb"], self._input_img_size, self._device
            )
            model_output_dict = SequorGnmUtils.model_output(
                self._ema_model_avg,
                self._noise_scheduler,
                obs_input,
                goal_img_input,
                self._pred_horizon,
                self._action_dim,
                self._num_samples,
                self._device,
            )
        elif self._model_type == "nomad_pointgoal":
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
            model_output_dict = SequorGnmUtils.pointgoal_model_output(
                self._ema_model_avg,
                self._noise_scheduler,
                obs_input,
                rcs_input,
                self._pred_horizon,
                self._action_dim,
                self._num_samples,
                self._device, 
                return_chain=return_chain,
            )

        gc_action_list = model_output_dict["gc_actions"].detach().cpu().numpy()
        uc_action_list = model_output_dict["uc_actions"].detach().cpu().numpy()
        action = HabitatActionUtils.get_discrete_control_output_from_trajectory_action_wo_stop(
            action_traj=gc_action_list,
            lookahead_point_idx=2,
            get_action_as_batch_idx=get_action_as_batch_idx,
            num_samples=self._num_samples,
            device=self._device,
        )

        if self._save_plot:
            curr_rgb = obs["rgb"]
            if len(curr_rgb.shape) == 4:
                # If the shape is (num_context+1, H, W, C), remove the first dimension
                curr_rgb = curr_rgb[-1]

            plot_title = f"nomad_{self._num_reset}_{self._num_act}"
            SequorGnmUtils.plot_trajs_wo_label(
                gc_actions_list=gc_action_list,
                uc_actions_list=uc_action_list,
                obs_img_to_plot=curr_rgb,
                goal_img_to_plot=info["goal_rgb"],
                dataset_name=self._dataset_name,
                save_plot=self._save_plot,
                plot_save_path=os.path.join(self._plot_save_dir, f"{plot_title}.png"),
                metric_waypoint_spacing=self._metric_waypoint_spacing,
                fig_title=f"{plot_title}\n{action}",
            )

            SequorGnmUtils.plot_trajs_wo_label(
                gc_actions_list=gc_action_list,
                uc_actions_list=uc_action_list,
                obs_img_to_plot=curr_rgb,
                goal_img_to_plot=info["goal_rgb"],
                dataset_name=self._dataset_name,
                save_plot=self._save_plot,
                plot_save_path=os.path.join(self._plot_save_dir, f"latest.png"),
                metric_waypoint_spacing=self._metric_waypoint_spacing,
                fig_title=f"{plot_title}\n{action}",
            )
        if return_chain:
            return {"actions": action, "model_output_dict": model_output_dict}
        return action

    def get_obs_size(self) -> int:
        return self._obs_size

    def get_input_img_size(self) -> tuple:
        return self._input_img_size

    def get_normalize(self) -> bool:
        return self._normalize

    def get_metric_waypoint_spacing(self) -> float:
        return self._metric_waypoint_spacing

    def get_waypoint_spacing(self) -> int:
        return self._waypoint_spacing

    def set_plot_save_dir(self, plot_save_dir):
        self._plot_save_dir = plot_save_dir
        os.makedirs(self._plot_save_dir, exist_ok=True)


def load_nomad_model(
    save_plot,
    plot_save_dir,
    config_file_name="nomad_sq.yaml",
    weight_path: Optional[str] = None,
    use_critic: bool = False,
):
    assert config_file_name in ["nomad.yaml", "nomad_sq.yaml", "nomad_pg.yaml"]

    # =============================== Load NoMAD ===============================
    action_dim = 2

    with open("models/gnms_levin/train/config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    with open(f"models/gnms_levin/train/config/{config_file_name}", "r") as f:
        user_config = yaml.safe_load(f)

    model_type = user_config["model_type"]

    if weight_path is None:
        if model_type == "nomad":
            weight_path = "models/pretrained_weights/nomad_f2_disc.pth"
        elif model_type == "nomad_pointgoal":
            weight_path = "models/pretrained_weights/nomad_pg_f3_49.pth"

    print(f"Loading nomad model: model_type: {model_type} weight_path: {weight_path}")

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

    ema_model_avg, noise_scheduler = SequorGnmUtils.load_nomad_model(
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
        weight_path=weight_path,
        gpu_ids=config["gpu_ids"],
        device=device,
    )
    # ema_model_avg.eval()
    if use_critic:
        assert model_type == "nomad_pointgoal", "Critic is only available for nomad model type, got {model_type}"
        critic = SequorGnmUtils.load_nomad_critic(
            model_type=model_type,
            encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            weight_path=weight_path,
            gpu_ids=config["gpu_ids"],
            device=device,
        )

    return NomadAgent(
        model_type=model_type,
        action_dim=action_dim,
        num_samples=1,
        pred_horizon=config["len_traj_pred"],
        device=device,
        ema_model_avg=ema_model_avg,
        noise_scheduler=noise_scheduler,
        obs_size=config["context_size"] + 1,
        input_img_size=config["image_size"],
        save_plot=save_plot,
        dataset_name="rvn_bench_eval",
        metric_waypoint_spacing=0.25,
        plot_save_dir=plot_save_dir,
        critic=critic if use_critic else None,
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
