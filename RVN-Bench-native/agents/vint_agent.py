import os
import sys
import yaml
import pdb
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from agents.agent import Agent, ObsType, InfoType


import models.model_utils.sequor_gnm_utils as SequorGnmUtils
import utils.habitat_gnm_utils as HabitatGnmUtils
import utils.habitat_action_utils as HabitatActionUtils

from models.gnms_levin.train.vint_train.training.train_eval_loop import load_model
from models.gnms_levin.train.vint_train.models.vint.vint_pointgoal import ViNTPointGoal


class VintAgent(Agent):
    def __init__(
        self,
        model: torch.nn.Module,
        device=torch.device("cpu"),
        pred_horizon=5,
        model_type="vint_pointgoal",
        action_dim=2,
        obs_size=4,
        input_img_size=[96, 96],
        metric_waypoint_spacing: float = 0.25,
        waypoint_spacing: int = 1,
        normalize: bool = True,
        len_traj_pred: int = 5,
        save_plot=False,
        dataset_name: str = "habitat_hm3d",
        plot_save_dir="",
        pg_rcs=False,
    ):
        assert model_type in ["vint_pointgoal"]

        self._model_type = model_type

        self._action_dim = action_dim
        self._pred_horizon = pred_horizon
        self._device = device

        self._model = model
        self._model.eval()
        self._input_img_size = input_img_size
        self._obs_image_context = HabitatGnmUtils.ObsImageContext(
            context_size=obs_size, input_img_size=input_img_size, device=device
        )
        self._metric_waypoint_spacing = metric_waypoint_spacing
        self._normalize = normalize
        self._len_traj_pred = len_traj_pred
        self._waypoint_spacing = waypoint_spacing

        self._save_plot = save_plot
        self._dataset_name = dataset_name
        self._plot_save_dir = plot_save_dir

        self._num_reset = 0
        self._num_act = 0

        self._pg_rcs = pg_rcs

        print("VintAgent initialized")
        print(f"\taction_dim  : {self._action_dim}")
        print(f"\tpred_horizon: {self._pred_horizon}")
        print(f"\tdevice      : {self._device}")

    def reset(self):
        self._num_reset += 1
        self._num_act = 0

        self._obs_image_context.reset()

    def act(self, obs: ObsType, info: InfoType) -> str:
        self._num_act += 1

        self._obs_image_context.add_obs_image(obs["rgb"])
        obs_input = self._obs_image_context.get_context()

        goal_gps_compass = obs["pointgoal_with_gps_compass"]
        # goal_gps_compass = np.array([2.0, -1.5707])

        if self._model_type == "vint_pointgoal":
            if self._pg_rcs:
                gps_goal_pos_input = (
                    HabitatGnmUtils.get_gnm_r_c_s_goal_pos_input_from_gps_compass(
                        goal_gps_compass,
                        self._normalize,
                        self._device,
                        self._metric_waypoint_spacing,
                        self._waypoint_spacing,
                        self._len_traj_pred,
                    )
                )
            else:
                gps_goal_pos_input = HabitatGnmUtils.get_gnm_goal_pos_input_from_gps_compass(
                    goal_gps_compass,
                    self._normalize,
                    self._device,
                    self._metric_waypoint_spacing,
                    self._waypoint_spacing,
                )
            model_output = self._model(obs_input, gps_goal_pos_input)

        dist_pred, action_pred = model_output
        action_pred = action_pred.detach().cpu().numpy()

        action = HabitatActionUtils.get_discrete_control_output_from_trajectory_action_wo_stop(
            action_traj=action_pred[0], lookahead_point_idx=2
        )

        # action = HabitatActionUtils.get_discrete_control_output_from_yaw(
        #     action_traj=action_pred[0], lookahead_point_idx=0
        # )

        if self._save_plot:
            plot_title = f"{self._model_type}_{self._num_reset}_{self._num_act}"
            SequorGnmUtils.plot_trajs_wo_label(
                gc_actions_list=action_pred,
                uc_actions_list=[],
                obs_img_to_plot=obs["rgb"],
                goal_img_to_plot=info["goal_rgb"],
                dataset_name=self._dataset_name,
                save_plot=self._save_plot,
                plot_save_path=os.path.join(self._plot_save_dir, f"{plot_title}.png"),
                metric_waypoint_spacing=self._metric_waypoint_spacing,
                fig_title=f"{plot_title}\n{action}\n{goal_gps_compass}",
            )

        return action

    def set_plot_save_dir(self, plot_save_dir: str):
        self._plot_save_dir = plot_save_dir
        os.makedirs(self._plot_save_dir, exist_ok=True)


def load_vint_model(
    save_plot,
    plot_save_dir=None,
    config_file_name="vint_pg.yaml",
    weight_path: Optional[str] = None,
):
    assert config_file_name in [
        "vint.yaml",
        "vint_sq.yaml",
        "vint_pg.yaml",
        "vint_pg_nmd.yaml",
    ]

    # =============================== Load NoMAD ===============================
    action_dim = 2

    with open("models/gnms_levin/train/config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    with open(f"models/gnms_levin/train/config/{config_file_name}", "r") as f:
        user_config = yaml.safe_load(f)

    model_type = user_config["model_type"]

    if weight_path is None:
        if model_type == "vint_pointgoal":
            weight_path = "models/pretrained_weights/vint_pg_29.pth"
        else:
            raise NotImplementedError()

    print(f"Loading vint model: model_type: {model_type} weight_path: {weight_path}")

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

    model = ViNTPointGoal(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=config["late_fusion"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        pg_rcs=False,
    ).to(device)

    latest_checkpoint = torch.load(weight_path, weights_only=False)
    load_model(model, config["model_type"], latest_checkpoint)

    return VintAgent(
        model=model,
        device=device,
        pred_horizon=config["len_traj_pred"],
        model_type=model_type,
        action_dim=action_dim,
        obs_size=4,  # config["context_size"], => Fool-proof
        input_img_size=[96, 96],
        metric_waypoint_spacing=0.25,
        waypoint_spacing=1,
        normalize=True,
        len_traj_pred=config["len_traj_pred"],
        save_plot=save_plot,
        dataset_name="habitat_hm3d",
        plot_save_dir=plot_save_dir,
        pg_rcs=False,
    )


def main(args):
    agent = load_vint_model(
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
        default="vint_pg.yaml",
        type=str,
        help="Select the target model config to evaluate [vint.yaml, vint_sq.yaml, vint_pg.yaml, vint_pg_nmd.yaml]",
    )
    args = parser.parse_args()

    main(args)
