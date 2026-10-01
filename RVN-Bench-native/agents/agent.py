from typing import TypedDict

import numpy as np
import torch

DIST_TO_GOAL_THRESHOLD = 0.36


class ObsType(TypedDict):
    """
    Observation for model.
    """

    rgb: np.ndarray
    # goal_pose: np.ndarray  # relative pose in [x_g, y_g, yaw_g]
    pointgoal_with_gps_compass: np.ndarray  # [radius, yaw]


class InfoType(TypedDict):
    """
    Information for NFM model.
    """

    robot_pose: np.ndarray  # [x, y, yaw] in world frame
    goal_pose_world: np.ndarray  # [x_g, y_g, yaw_g] in world frame
    goal_rgb: np.ndarray
    num_wp_reached: int
    first_wp_reached: bool
    collision: bool
    timeout: bool
    travel_distance: float
    l2_dist_to_goal: float


class Agent:
    def __init__(
        self,
    ):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def act(self, obs: ObsType, info: InfoType) -> str:
        raise NotImplementedError


class ActorCriticAgent(Agent):
    def __init__(
        self,
        actor_critic_model,
        weight_path,
        preproc_obs,
        postproc_action,
        obs_dim=2,
        action_dim=3,
        hidden_dim=256,
        device: torch.device = torch.device("cpu"),
    ):
        self._device = device
        if hidden_dim is None:
            self._policy = actor_critic_model(
                obs_dim=obs_dim, action_dim=action_dim
            ).to(device)
        else:
            self._policy = actor_critic_model(
                obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
            ).to(device)
        self._policy.load_state_dict(torch.load(weight_path))
        self._policy.eval()

        self._preproc_obs = preproc_obs
        self._postproc_action = postproc_action

    def reset(self):
        pass

    def act(self, obs: ObsType, info: InfoType):
        rvn_obs = self._preproc_obs(obs)
        rvn_obs = torch.tensor(rvn_obs, dtype=torch.float).to(self._device)
        _, action_probs = self._policy(rvn_obs)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action = self._postproc_action(action.cpu().detach().numpy(), obs)
        return action
