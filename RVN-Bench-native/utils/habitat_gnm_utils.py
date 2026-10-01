from typing import List, Tuple

from torchvision.transforms import ToPILImage
import numpy as np
import torch

import models.model_utils.sequor_gnm_utils as SequorGnmUtils
from PIL import Image


def get_gnm_obs_input_from_rgb_img(
    obs_images: List[Image.Image],
    input_img_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Get GNM input from RGB images.

    Parameters
    ----------
    rgb_images : List[Image.Image]
        List of RGB images.
    input_img_size : Tuple[int, int]
        Input image size of the GNM model.
    device : torch.device
        Device.

    Returns
    -------
    torch.Tensor
        GNM obs input.
    """
    resized_obs_images = [
        SequorGnmUtils.resize_and_aspect_crop(img, input_img_size).unsqueeze(0)
        for img in obs_images
    ]
    transform = SequorGnmUtils.get_transform()
    transformed_obs_images = [transform(img) for img in resized_obs_images]
    transformed_obs_images = torch.cat(transformed_obs_images, dim=1).to(device)

    return transformed_obs_images


def get_gnm_obs_input_from_rgb_np(
    obs_rgb_np: np.ndarray,
    input_img_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Get GNM input from RGB images.

    Parameters
    ----------
    rgb_images : np.ndarray
        List of RGB images.
        []
    input_img_size : Tuple[int, int]
        Input image size of the GNM model.
    device : torch.device
        Device.

    Returns
    -------
    torch.Tensor
        GNM obs input.
    """

    if len(obs_rgb_np.shape) == 4:
        obs_images = []
        for i in range(obs_rgb_np.shape[0]):
            obs_images.append(Image.fromarray(obs_rgb_np[i]))

        return get_gnm_obs_input_from_rgb_img(obs_images, input_img_size, device)

    elif len(obs_rgb_np.shape) == 5:
        # TODO: optimize this code
        obs_images_list = []
        for idx_env in range(obs_rgb_np.shape[0]):
            obs_images = []
            for i in range(obs_rgb_np.shape[1]):
                obs_images.append(Image.fromarray(obs_rgb_np[idx_env][i]))
            obs_images_list.append(
                get_gnm_obs_input_from_rgb_img(obs_images, input_img_size, device)
            )
        return torch.cat(obs_images_list, dim=0)

    else:
        raise ValueError(
            f"Unexpected obs['rgb'] shape: {obs_rgb_np.shape}. Expected 4D or 5D tensor."
        )


def get_gnm_goal_input_from_rgb_img(
    goal_image: Image.Image,
    input_img_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Get GNM input from RGB images.

    Parameters
    ----------
    goal_image : Image.Image
        Goal image.
    input_img_size : Tuple[int, int]
        Input image size of the GNM model.
    device : torch.device
        Device.

    Returns
    -------
    torch.Tensor
        GNM goal input.
    """
    resized_goal_image = SequorGnmUtils.resize_and_aspect_crop(
        goal_image, input_img_size
    ).unsqueeze(0)

    transform = SequorGnmUtils.get_transform()
    transformed_goal_images = transform(resized_goal_image)
    transformed_goal_images = transformed_goal_images.to(device)

    return transformed_goal_images


def get_gnm_goal_input_from_rgb_np(
    goal_image_np: np.ndarray,
    input_img_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Get GNM input from RGB images.

    Parameters
    ----------
    goal_image : np.ndarray
        Goal image.
    input_img_size : Tuple[int, int]
        Input image size of the GNM model.
    device : torch.device
        Device.

    Returns
    -------
    torch.Tensor
        GNM goal input.
    """
    goal_image = Image.fromarray(goal_image_np)
    return get_gnm_goal_input_from_rgb_img(goal_image, input_img_size, device)


def get_gnm_goal_pos_input(
    goal_pos: np.ndarray,
    normalize,
    len_traj_pred,
    device,
    metric_waypoint_spacing=None,
    waypoint_spacing=None,
) -> torch.Tensor:

    goal_pos = np.array(goal_pos[:2])

    if normalize:
        goal_pos /= metric_waypoint_spacing * waypoint_spacing
        if len_traj_pred is not None:
            goal_pos /= len_traj_pred

    goal_pos = torch.tensor(goal_pos, device=device, dtype=torch.float).unsqueeze(0)

    return goal_pos


def get_gnm_goal_pos_input_from_gps_compass(
    pointgoal_with_gps_compass: np.ndarray,  # [dist_to_goal, yaw_to_goal]
    normalize,
    device,
    metric_waypoint_spacing=None,
    waypoint_spacing=None,
    len_traj_pred=None,
) -> torch.Tensor:
    if len(pointgoal_with_gps_compass.shape) == 1:

        goal_pose = np.array(
            [
                pointgoal_with_gps_compass[0] * np.cos(pointgoal_with_gps_compass[1]),
                pointgoal_with_gps_compass[0] * np.sin(pointgoal_with_gps_compass[1]),
                0.0,
            ]
        )

        return get_gnm_goal_pos_input(
            goal_pose,
            normalize,
            len_traj_pred,
            device,
            metric_waypoint_spacing=metric_waypoint_spacing,
            waypoint_spacing=waypoint_spacing,
        )

    elif len(pointgoal_with_gps_compass.shape) == 2:
        gnm_goal_poses = []
        for idx_env in range(pointgoal_with_gps_compass.shape[0]):
            goal_pose = np.array(
                [
                    pointgoal_with_gps_compass[idx_env][0]
                    * np.cos(pointgoal_with_gps_compass[idx_env][1]),
                    pointgoal_with_gps_compass[idx_env][0]
                    * np.sin(pointgoal_with_gps_compass[idx_env][1]),
                    0.0,
                ]
            )
            gnm_goal_poses.append(
                get_gnm_goal_pos_input(
                    goal_pose,
                    normalize,
                    len_traj_pred,
                    device,
                    metric_waypoint_spacing=metric_waypoint_spacing,
                    waypoint_spacing=waypoint_spacing,
                )
            )
        return torch.cat(gnm_goal_poses, dim=0)

    else:
        raise ValueError(
            f"Unexpected obs['rgb'] shape: {pointgoal_with_gps_compass.shape}. Expected 1D or 2D array."
        )


def get_gnm_r_c_s_goal_pos_input_from_gps_compass(
    pointgoal_with_gps_compass: np.ndarray,  # [dist_to_goal, yaw_to_goal]
    normalize,
    device,
    metric_waypoint_spacing=None,
    waypoint_spacing=None,
    len_traj_pred=None,
) -> torch.Tensor:
    if len(pointgoal_with_gps_compass.shape) == 1:

        dist_to_goal = pointgoal_with_gps_compass[0]
        if normalize:
            dist_to_goal /= metric_waypoint_spacing * waypoint_spacing
            if len_traj_pred is not None:
                dist_to_goal /= len_traj_pred

        gps_goal = np.array(
            [
                dist_to_goal,
                np.cos(pointgoal_with_gps_compass[1]),
                np.sin(pointgoal_with_gps_compass[1]),
            ]
        )

        return torch.tensor(gps_goal, device=device, dtype=torch.float).unsqueeze(0)

    elif len(pointgoal_with_gps_compass.shape) == 2:
        raise NotImplementedError

    else:
        raise ValueError(
            f"Unexpected obs['rgb'] shape: {pointgoal_with_gps_compass.shape}. Expected 1D or 2D array."
        )


def get_gnm_goal_input_from_rgba_np(
    goal_image_np: np.ndarray,
    input_img_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Get GNM input from RGB images.

    Parameters
    ----------
    goal_image : np.ndarray
        Goal image.
    input_img_size : Tuple[int, int]
        Input image size of the GNM model.
    device : torch.device
        Device.

    Returns
    -------
    torch.Tensor
        GNM goal input.
    """
    goal_image = Image.fromarray(goal_image_np)
    goal_image = goal_image.convert("RGB")
    return get_gnm_goal_input_from_rgb_img(goal_image, input_img_size, device)


class ObsImageContext:
    def __init__(
        self, context_size: int, input_img_size: Tuple[int, int], device: torch.device
    ):
        self._context_size = context_size
        self._input_img_size = input_img_size
        self._device = device

        self._rgb_obs_images = []
        self._context = []

    def get_context(self):
        if len(self._rgb_obs_images) == 0:
            raise ValueError("No images in context")
        return self._context

    def add_obs_image(self, obs_rgb: np.ndarray):
        if len(self._rgb_obs_images) >= self._context_size:
            self._rgb_obs_images.pop(0)

        while len(self._rgb_obs_images) < self._context_size:
            self._rgb_obs_images.append(Image.fromarray(obs_rgb))

        assert len(self._rgb_obs_images) == self._context_size

        self._context = get_gnm_obs_input_from_rgb_img(
            self._rgb_obs_images, self._input_img_size, self._device
        )

    def reset(self):
        self._rgb_obs_images = []
        self._context = []
