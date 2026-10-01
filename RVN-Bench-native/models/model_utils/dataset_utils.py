import os
import pickle
from typing import Tuple, List

import torch
import numpy as np
from PIL import Image

import models.model_utils.sequor_gnm_utils as SequorGnmUtils


def read_trajectory_data_dir(dataset_dir) -> Tuple[dict, List[Image.Image]]:
    """
    dataset_dir should include the following files:
    traj_data.pkl
    0.jpg, 1.jpg, 2.jpg, ...

    Returns
    -------
    traj_data : dict
        trajectory data with 'position' and 'yaw' keys
    images : list of PIL.Image.Image
        RGB images in the dataset
    """
    dataset_traj_path = os.path.join(dataset_dir, "traj_data.pkl")

    # Read the trajectory data
    with open(dataset_traj_path, "rb") as f:
        traj_data = pickle.load(f)

    # Read the image data
    img_paths = []
    loaded_images = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".jpg") and not file.endswith("rgb.jpg"):
            dataset_traj_path = os.path.join(dataset_dir, file)
            img_paths.append(dataset_traj_path)

    img_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    for path in img_paths:
        img = Image.open(path)
        loaded_images.append(img)

    return traj_data, loaded_images


def get_transformed_obs_img_from_dir(
    dataset_dir, input_img_size, obs_idx, obs_size, goal_idx, device
) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    """
    dataset_dir should include the following files:
    traj_data.pkl
    0.jpg, 1.jpg, 2.jpg, ...

    Returns
    -------
    traj_data : dict
        trajectory data with 'position' and 'yaw' keys
    transformed_obs_images : torch.Tensor
        transformed observation images
    transformed_goal_images : torch.Tensor
        transformed goal image
    """
    traj_data, loaded_images = read_trajectory_data_dir(dataset_dir)

    assert obs_idx - obs_size + 1 >= 0
    assert goal_idx < len(loaded_images)

    obs_img = [
        SequorGnmUtils.resize_and_aspect_crop(img, input_img_size).unsqueeze(0)
        for img in loaded_images[obs_idx - obs_size + 1 : obs_idx + 1]
    ]
    goal_img = SequorGnmUtils.resize_and_aspect_crop(
        loaded_images[goal_idx], input_img_size
    ).unsqueeze(0)

    transform = SequorGnmUtils.get_transform()

    transformed_obs_images = [transform(img) for img in obs_img]
    transformed_goal_images = transform(goal_img)

    transformed_obs_images = torch.cat(transformed_obs_images, dim=1).to(device)
    transformed_goal_images = transformed_goal_images.to(device)

    return (
        traj_data,
        transformed_obs_images,
        transformed_goal_images,
    )


def get_raw_obs_img_from_dir(dataset_dir, obs_idx, goal_idx):
    traj_data, loaded_images = read_trajectory_data_dir(dataset_dir)
    obs_img_to_plot = loaded_images[obs_idx]
    goal_img_to_plot = loaded_images[goal_idx]
    return obs_img_to_plot, goal_img_to_plot


def get_action_and_dist_label(traj_data, obs_idx, goal_idx, pred_horizon):
    """
    Get the action and distance label from the observation and goal.
    """
    obs_position = traj_data["position"][obs_idx]
    obs_yaw = traj_data["yaw"][obs_idx]
    goal_position = traj_data["position"][goal_idx]

    action_label = np.array(
        [
            SequorGnmUtils.transpose_to_robot_frame(obs_position, obs_yaw, pos)
            for pos in traj_data["position"][obs_idx + 1 : obs_idx + 1 + pred_horizon]
        ]
    )
    dist_label = np.linalg.norm(np.array(goal_position) - np.array(obs_position))

    return action_label, dist_label


def get_relative_goal_pose(traj_data, obs_idx, goal_idx) -> np.ndarray:
    """
    Get the relative goal pose from the observation and goal.
    """
    obs_position = traj_data["position"][obs_idx]
    obs_yaw = traj_data["yaw"][obs_idx]
    goal_position = traj_data["position"][goal_idx]

    relative_goal_pose = SequorGnmUtils.transpose_to_robot_frame(
        obs_position, obs_yaw, goal_position
    )

    return np.array(relative_goal_pose)
