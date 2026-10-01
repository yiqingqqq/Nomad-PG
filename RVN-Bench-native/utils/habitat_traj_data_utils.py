import os
import yaml
import pickle
from typing import Tuple, List, TypedDict

import numpy as np
import imageio

from agents.agent import ObsType, InfoType
import utils.geometry_utils as GeometryUtils


class TrainObsType(TypedDict):
    current_rgb_file_path: str
    goal_pose: np.ndarray  # relative pose in [x_g, y_g, yaw_g]


def load_traj_action_labels(trajectory_dir: str, len_traj: int) -> List[int]:
    action_file_dir = os.path.join(trajectory_dir, "discrete_action_data.yaml")
    with open(action_file_dir, "r") as f:
        action_data = yaml.load(f, Loader=yaml.FullLoader)
        action_labels = action_data["actions"][:len_traj]
    return action_labels


def get_noisy_goal_pose(
    last_goal_pose: Tuple[float, float, float], noise_scale: float = 0.1
) -> Tuple[float, float, float]:
    return (
        last_goal_pose[0] + np.random.normal(0, noise_scale),
        last_goal_pose[1] + np.random.normal(0, noise_scale),
        last_goal_pose[2] + np.random.normal(0, noise_scale),
    )


def load_traj_train_obs_info(
    trajectory_dir: str, num_margin_to_goal_idx=1
) -> Tuple[List[TrainObsType], List[InfoType]]:
    traj_obs = []
    traj_info = []

    # load trajectory data to set goal pose
    # load traj_data.pkl
    traj_data_file_dir = os.path.join(trajectory_dir, "traj_data.pkl")
    with open(traj_data_file_dir, "rb") as f:
        traj_data = pickle.load(f)
    positions = traj_data["position"]
    yaws = traj_data["yaw"]

    goal_pose = get_noisy_goal_pose((*positions[-1], yaws[-1]), noise_scale=0.0)

    relative_goal_poses = []
    for idx_path in range(len(positions) - num_margin_to_goal_idx):
        relative_goal_pose = GeometryUtils.transpose_world_pose_to_local_frame(
            (*positions[idx_path], yaws[idx_path]), (*positions[-1], yaws[-1])
        )
        relative_goal_poses.append(relative_goal_pose)

        rgb_obs_file_path = os.path.join(trajectory_dir, f"{idx_path}.jpg")

        obs_on_pose: TrainObsType = {
            "current_rgb_file_path": rgb_obs_file_path,
            "goal_pose": relative_goal_pose,
        }
        info_on_pose = {
            "robot_pose": (*positions[idx_path], yaws[idx_path]),
            "goal_pose_world": goal_pose,
        }

        traj_obs.append(obs_on_pose)
        traj_info.append(info_on_pose)

    return traj_obs, traj_info
