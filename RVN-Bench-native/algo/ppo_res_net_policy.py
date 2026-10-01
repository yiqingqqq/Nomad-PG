import os
import sys
import time
from typing import Optional


import numpy as np
import imageio

import torch
from torch.optim import Adam

from gym.spaces import Box, Discrete
from gym.spaces import Dict as SpaceDict

from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from algo.ppo_vanilla import PpoTrainerDiscrete
from agents.res_net_policy_agent import (
    preproc_rvn_obs_res_net_policy,
    postproc_rvn_action,
)


# Fix PPO to match the performance of the habitat-lab implementation
class PpoTrainerResNetPolicy(PpoTrainerDiscrete):
    def __init__(
        self,
        env,
        hyper_parameters: dict = {},
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        weight_save_dir: str = "weights",
        render_save_dir: str = "images",
        weight_load_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self._render = True
        print(f"Init PpoTrainerResNetPolicy device: {device}")

        self._env = env
        self._preproc_obs = preproc_rvn_obs_res_net_policy
        self._postproc_action = postproc_rvn_action
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_size = hidden_size
        self._weight_save_dir = weight_save_dir
        self._render_save_dir = render_save_dir
        self._device = device

        os.makedirs(weight_save_dir, exist_ok=True)
        os.makedirs(render_save_dir, exist_ok=True)

        observation_spaces = SpaceDict(
            {
                "pointgoal_with_gps_compass": Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
                "rgb": Box(
                    low=0,
                    high=255,
                    shape=(256, 256, 3),
                    dtype=np.uint8,
                ),
            }
        )
        action_spaces = Discrete(action_dim)

        self._init_hyperparmeters(**hyper_parameters, hidden_dim=hidden_size)

        # ALG STEP 1
        # Initialize actor and critic networks
        self._policy = PointNavResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=hidden_size,
            normalize_visual_inputs=True,
            num_recurrent_layers=4,
        ).to(self._device)

        self._optim = Adam(self._policy.parameters(), lr=self._lr)

        # print(self._policy)

        # This logger will help us with printing out summaries of each iteration
        self._logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "batch_rews": [],  # episodic returns in batch
            "batch_num_wp_reached": [],  # number of waypoints reached in batch
            "actor_losses": [],  # losses of actor network in current iteration
        }

    def _rollout(self):
        # Batch data

        batch_obs = {
            "observations": {
                "pointgoal_with_gps_compass": [],  # (timesteps_per_batch, 2)
                "rgb": [],  # (timesteps_per_batch, 256, 256, 3)
            },
            "rnn_hidden_states": [],  # (timesteps_per_batch, num_recurrent_layers, hidden_size)
            "prev_actions": [],  # (timesteps_per_batch, 1)
            "masks": [],  # (timesteps_per_batch, 1)
        }
        batch_acts = []  # (timesteps_per_batch, act_dim)
        batch_log_probs = []  # (timesteps_per_batch)
        batch_reward_to_go = []  # (timesteps_per_batch)

        batch_rewards = []  # (num_episodes, timestpes_per_episode)
        batch_vals = []  # (num_episodes, timestpes_per_episode)
        batch_dones = []  # (num_episodes, timestpes_per_episode)
        batch_lens = []  # (num_episodes)

        # Number of timesteps run so far this batch
        batch_t = 0

        while batch_t < self._timesteps_per_batch:
            ep_rewards = []
            ep_vals = []
            ep_dones = []

            render_episode = (
                self._render
                and (self._logger["i_so_far"] % self._save_freq == 0)
                and len(batch_lens) == 0  # Only the first episode in the batch
            )
            frames = []

            # Reset the environment
            rnn_hidden_states = torch.zeros(
                1,
                self._policy.net.num_recurrent_layers,
                self._hidden_size,
                device=self._device,
            )
            prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self._device)
            not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self._device)

            obs, _ = self._env.reset()
            ep_done = False

            for ep_t in range(self._max_timesteps_per_episode):
                if render_episode:
                    frame = self._env.render()
                    frames.append(frame)

                batch_t += 1

                rvn_obs = self._preproc_obs(obs, self._device)
                batch_input = {
                    "observations": rvn_obs,
                    "rnn_hidden_states": rnn_hidden_states,
                    "prev_actions": prev_actions,
                    "masks": not_done_masks,
                }

                ep_dones.append(ep_done)
                # Collect observation
                for key, value in rvn_obs.items():
                    batch_obs["observations"][key].append(value.squeeze(0))
                batch_obs["rnn_hidden_states"].append(rnn_hidden_states.squeeze(0))
                batch_obs["prev_actions"].append(prev_actions.squeeze(0))
                batch_obs["masks"].append(not_done_masks.squeeze(0))

                action_data = self._policy.act(**batch_input)

                action = action_data.actions[0][0]
                v_obs = action_data.values[0][0]
                action_log_prob = action_data.action_log_probs[0][0]

                obs, reward, terminated, truncated, info = self._env.step(
                    self._postproc_action(action.cpu().detach().numpy(), obs)
                )

                # Collect reward, action, and log_prob
                ep_rewards.append(reward)
                ep_vals.append(v_obs.flatten())

                batch_acts.append(action)
                batch_log_probs.append(action_log_prob)

                rnn_hidden_states = action_data.rnn_hidden_states.clone().detach()
                prev_actions.copy_(action_data.actions)  # type: ignore
                #  Make masks not done till reset (end of episode) will be called
                not_done_masks.fill_(True)

                ep_done = terminated or truncated
                if ep_done:
                    break

            self._logger["batch_num_wp_reached"].append(info["num_wp_reached"])

            self._num_ep += 1
            self._recent_reward_sum = np.sum(ep_rewards)

            # Collect episodic length and rewards
            batch_rewards.append(ep_rewards)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
            batch_lens.append(ep_t + 1)

            if render_episode and len(frames) > 0:
                descript = ""
                if terminated:
                    descript = "_terminated"
                elif truncated:
                    descript = "_truncated"
                imageio.mimsave(
                    os.path.join(
                        self._render_save_dir,
                        f"episode_{self._logger['i_so_far']}{descript}.gif",
                    ),
                    frames,
                )

        batch_obs["observations"]["rgb"] = torch.stack(batch_obs["observations"]["rgb"])
        batch_obs["observations"]["pointgoal_with_gps_compass"] = torch.stack(
            batch_obs["observations"]["pointgoal_with_gps_compass"]
        )
        batch_obs["rnn_hidden_states"] = torch.stack(batch_obs["rnn_hidden_states"])
        batch_obs["prev_actions"] = torch.stack(batch_obs["prev_actions"])
        batch_obs["masks"] = torch.stack(batch_obs["masks"])

        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self._device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(
            self._device
        )

        # ALG STEP 4
        # batch_reward_to_go = self._compute_reward_to_go(batch_rewards)

        self._logger["batch_rews"] = batch_rewards
        self._logger["batch_lens"] = batch_lens

        return (
            batch_obs,
            batch_acts,
            batch_log_probs,
            # batch_reward_to_go,
            batch_rewards,
            batch_vals,
            batch_dones,
            batch_lens,
        )

    def _get_value_and_action_log_probs(self, batch_obs):
        # batch_obs is TensorDict
        action_data: PolicyActionData = self._policy.act(**batch_obs)
        v_values = action_data.values
        action_log_probs = action_data.action_log_probs
        return v_values, action_log_probs

    def _get_value(self, batch_obs):
        """
        Returns
        -------
        V(obs) : torch.tensor
            (batch_size, )
        """
        action_data: PolicyActionData = self._policy.act(**batch_obs)
        V_obs = action_data.values
        return V_obs.squeeze(1)

    def _evaluate_policy(self, batch_obs, batch_acts):
        """
        Returns
        -------
        V(obs) : torch.tensor
            (batch_size, )
        action_log_probs: torch.tensor
            (batch_size, )
        entropy: torch.tensor
            (batch_size, )
        """
        # batch_obs is TensorDict
        features, rnn_hidden_states, _ = self._policy.net(**batch_obs)
        distribution = self._policy.action_distribution(features)

        V_obs = self._policy.critic(features)
        action_log_probs = distribution.log_probs(batch_acts)
        entropy = distribution.entropy()
        return V_obs.squeeze(1), action_log_probs.squeeze(), entropy.squeeze(1)

    def _get_mini_batch_obs(self, batch_obs, minibatch_inds):
        mini_batch_obs = {
            "observations": {
                "rgb": batch_obs["observations"]["rgb"][minibatch_inds],
                "pointgoal_with_gps_compass": batch_obs["observations"][
                    "pointgoal_with_gps_compass"
                ][minibatch_inds],
            },
            "rnn_hidden_states": batch_obs["rnn_hidden_states"][minibatch_inds],
            "prev_actions": batch_obs["prev_actions"][minibatch_inds],
            "masks": batch_obs["masks"][minibatch_inds],
        }
        return mini_batch_obs
