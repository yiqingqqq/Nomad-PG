from email import policy
import os
import sys
import time
from typing import Optional, Tuple


import numpy as np
import wandb

import torch
import torch.nn.functional as F
from torch.optim import Adam

from gym.spaces import Box, Discrete
from gym.spaces import Dict as SpaceDict

from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat import VectorEnv
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from algo.ppo_vanilla import PpoTrainerDiscrete
import algo.utils.ppo_utils as ppo_utils
from agents.res_net_policy_agent import (
    preproc_rvn_obs_res_net_policy,
    postproc_rvn_action,
)
from agents.res_net_policy_ppo_lag_agent import CostHead


class PpoLagTrainerResNetPolicyMultiEnv(PpoTrainerDiscrete):
    def __init__(
        self,
        envs: VectorEnv,
        hyper_parameters: dict = {},
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        weight_save_dir: str = "weights",
        render_save_dir: str = "images",
        weight_load_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        lambda_cost: float = 0.001,
        cost_loss_coeff: float = 0.5,
        cost_limit: float = 0.001,
        use_fixed_dual_lr=True,
        fixed_dual_lr=0.035,
        lr_to_dual_lr: float = 0.01,
    ):
        self._render = True
        print(
            f"Init PpoLagTrainerResNetPolicyMultiEnv device: {device} n_env: {envs.num_envs}"
        )

        self._envs = envs

        self._preproc_obs = preproc_rvn_obs_res_net_policy
        self._postproc_action = postproc_rvn_action
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_size = hidden_size
        self._weight_save_dir = weight_save_dir
        self._render_save_dir = render_save_dir
        self._device = device
        self._n_env = envs.num_envs

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
        self._lambda_cost = lambda_cost
        self._cost_loss_coeff = cost_loss_coeff
        self._cost_limit = cost_limit
        self._use_fixed_dual_lr = use_fixed_dual_lr
        self._fixed_dual_lr = fixed_dual_lr
        self._lr_to_dual_lr = lr_to_dual_lr

        assert self._n_env > self._n_mini_batch
        if self._n_env % self._n_mini_batch != 0:
            print(
                f"Warning: n_env ({self._n_env}) % n_mini_batch ({self._n_mini_batch}) != 0"
            )

        # ALG STEP 1
        # Initialize actor and critic networks
        self._policy = PointNavResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=hidden_size,
            normalize_visual_inputs=True,
            num_recurrent_layers=2,
            rnn_type="LSTM",
            backbone="resnet50",
        )
        self._policy.critic_c = CostHead(input_size=self._policy.net.output_size)
        self._policy.to(self._device)

        self._optim = Adam(self._policy.parameters(), lr=self._lr, eps=self._eps)

        # print(self._policy)

        # This logger will help us with printing out summaries of each iteration
        self._logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "rewards_by_env": [0.0 for _ in range(self._n_env)],  # rewards by env
            "ep_lens_by_env": [0 for _ in range(self._n_env)],  # episode lengths by env
            "batch_rews": [],  # rewards of episodes terminated in current batch
            "batch_ep_lens": [],  # lengths of episodes terminated in batch
            "batch_first_wp_reached": [],  # number of first waypoints reached for terminated ep in batch
            "batch_num_wp_reached": [],  # number of waypoints reached for terminated ep in batch
            "batch_travel_distances": [],  # travel distances for terminated ep in batch
            "batch_l2_dist_to_1st_wp": [],  # l2 distance to first waypoint for terminated ep in batch
            "batch_l2_dist_to_goal": [],  # l2 distance to first waypoint for terminated ep in batch
            "batch_collisions": [],
            "batch_timeouts": [],  # timeouts for terminated ep in batch
            "batch_count": 0,  # number of episodes terminated in batch
            "actor_losses": [],  # losses of actor network in current iteration
            "value_losses": [],  # losses of value network in current iteration
            "cost_losses": [],  # losses of cost network in current iteration
            "dist_entropies": [],  # entropy of action distribution in current iteration
        }

    def learn(
        self,
        total_timesteps: int,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        print(
            f"Learning... Running {self._max_timesteps_per_episode} timesteps per episode, ",
            end="",
        )
        print(
            f"{self._timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps"
        )

        print(f"Using device           : {self._device}")
        print(f"batch_size             : {self._timesteps_per_batch}")
        print(f"gamma                  : {self._gamma}")
        print(f"lam                    : {self._lam}")
        print(f"lr                     : {self._lr}")
        print(f"eps                    : {self._eps}")
        print(f"clip_ratio             : {self._clip_ratio}")
        print(f"n_updates_per_iteration: {self._n_updates_per_iteration}")
        print(f"n_mini_batch           : {self._n_mini_batch}")
        print(f"value_loss_coeff       : {self._value_loss_coeff}")
        print(f"entropy_loss_coeff     : {self._entropy_loss_coeff}")
        print(f"normalize_advantage    : {self._normalize_advantage}")
        print(f"use_clipped_value_loss : {self._use_clipped_value_loss}")
        print(f"hidden_dim             : {self._hidden_dim}")
        print(f"num_envs               : {self._n_env}")
        print(f"lambda_cost            : {self._lambda_cost}")
        print(f"cost_loss_coeff        : {self._cost_loss_coeff}")
        print(f"cost_limit             : {self._cost_limit}")
        print(f"use_fixed_dual_lr      : {self._use_fixed_dual_lr}")
        print(f"fixed_dual_lr          : {self._fixed_dual_lr}")
        print(f"lr_to_dual_lr          : {self._lr_to_dual_lr}")
        

        self._num_ep = 0
        self._use_wandb = use_wandb
        self._best_train_avg_ep_rew = -np.inf

        if wandb_project is None or run_name is None:
            print("wandb_project not specified, set use_wandb to False")
            use_wandb = False
        if use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    "batch_size": self._timesteps_per_batch,
                    "gamma": self._gamma,
                    "lam": self._lam,
                    "lr": self._lr,
                    "eps": self._eps,
                    "clip_ratio": self._clip_ratio,
                    "n_updates_per_iteration": self._n_updates_per_iteration,
                    "n_mini_batch": self._n_mini_batch,
                    "value_loss_coeff": self._value_loss_coeff,
                    "entropy_loss_coeff": self._entropy_loss_coeff,
                    "normalize_advantage": self._normalize_advantage,
                    "hidden_dim": self._hidden_dim,
                    "lambda_cost": self._lambda_cost,
                    "cost_loss_coeff": self._cost_loss_coeff,
                    "cost_limit": self._cost_limit,
                    "use_fixed_dual_lr": self._use_fixed_dual_lr,
                    "fixed_dual_lr": self._fixed_dual_lr,
                    "lr_to_dual_lr": self._lr_to_dual_lr,
                },
            )
            wandb.run.name = run_name
            wandb.run.save()

        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far

        while t_so_far < total_timesteps:  # ALG STEP 2
            # ALG STEP 3
            # Collect set of trajectories D_k = {τ_i}
            # by running policy π_k = π(θ_k) in the environment.
            (
                batch_obs,
                batch_recurrent_hidden_states,
                batch_rewards,
                batch_costs,
                batch_value_preds,
                batch_cost_preds,
                batch_action_log_probs,
                batch_actions,
                batch_prev_actions,
                batch_not_done_masks,
            ) = self._multi_env_rollout()
            print("batch obs gathered from _multi_env_rollout")

            # print("batch_actions:", batch_actions)
            # print("batch_prev_actions:", batch_prev_actions)

            t_so_far += np.sum(self._timesteps_per_batch * self._n_env)
            i_so_far += 1
            # Logging timesteps so far and iterations so far
            self._logger["t_so_far"] = t_so_far
            self._logger["i_so_far"] = i_so_far

            # ALG STEP 5
            # Calculate the adventage
            batch_returns = ppo_utils.compute_gae_returns(
                batch_rewards,
                batch_value_preds,
                batch_not_done_masks,
                gamma=self._gamma,
                lam=self._lam,
            ).detach()
            batch_advantages = (batch_returns - batch_value_preds).detach()

            # Calculate the A_c
            batch_gae_c = ppo_utils.compute_gae_returns(
                batch_costs,
                batch_cost_preds,
                batch_not_done_masks,
                gamma=self._gamma,
                lam=self._lam,
            ).detach()
            batch_advantages_c = (batch_gae_c - batch_cost_preds).detach()

            for idx_update_itr in range(self._n_updates_per_iteration):
                rand_perm = torch.randperm(self._n_env).chunk(self._n_mini_batch)
                dones_cpu = (
                    torch.logical_not(batch_not_done_masks)
                    .cpu()
                    .view(-1, self._n_env)
                    .numpy()
                )

                for inds in rand_perm:
                    curr_slice = (slice(0, self._timesteps_per_batch), inds)
                    mini_batch_obs = {
                        "pointgoal_with_gps_compass": batch_obs[
                            "pointgoal_with_gps_compass"
                        ][curr_slice],
                        "rgb": batch_obs["rgb"][curr_slice],
                    }
                    mini_batch_recurrent_hidden_states = batch_recurrent_hidden_states[
                        curr_slice
                    ]
                    mini_batch_cost = batch_costs[curr_slice]
                    mini_batch_value_preds = batch_value_preds[curr_slice]
                    mini_batch_cost_preds = batch_cost_preds[curr_slice]
                    mini_batch_action_log_probs = batch_action_log_probs[curr_slice]
                    mini_batch_actions = batch_actions[curr_slice]
                    mini_batch_prev_actions = batch_prev_actions[curr_slice]
                    mini_batch_not_done_masks = batch_not_done_masks[curr_slice]
                    mini_batch_returns = batch_returns[curr_slice]
                    mini_batch_gae_c = batch_gae_c[curr_slice]
                    mini_batch_advantages = batch_advantages[curr_slice]
                    mini_batch_advantages_c = batch_advantages_c[curr_slice]
                    mini_batch_rnn_build_seq_info = build_rnn_build_seq_info(
                        device=self._device,
                        build_fn_result=build_pack_info_from_dones(
                            dones_cpu[
                                0 : self._timesteps_per_batch, inds.numpy()
                            ].reshape(-1, len(inds)),
                        ),
                    )

                    mini_batch_obs["rgb"] = mini_batch_obs["rgb"].flatten(0, 1)
                    mini_batch_obs["pointgoal_with_gps_compass"] = mini_batch_obs[
                        "pointgoal_with_gps_compass"
                    ].flatten(0, 1)
                    # ???? Why use [0:1] here?
                    # mini_batch_recurrent_hidden_states = (
                    #     mini_batch_recurrent_hidden_states[0:1].flatten(0, 1)
                    # )
                    mini_batch_recurrent_hidden_states = (
                        mini_batch_recurrent_hidden_states.flatten(0, 1)
                    )
                    mini_batch_cost = mini_batch_cost.flatten(0, 1)
                    mini_batch_value_preds = mini_batch_value_preds.flatten(0, 1)
                    mini_batch_cost_preds = mini_batch_cost_preds.flatten(0, 1)
                    mini_batch_action_log_probs = mini_batch_action_log_probs.flatten(
                        0, 1
                    )
                    mini_batch_actions = mini_batch_actions.flatten(0, 1)
                    mini_batch_prev_actions = mini_batch_prev_actions.flatten(0, 1)
                    mini_batch_not_done_masks = mini_batch_not_done_masks.flatten(0, 1)
                    mini_batch_returns = mini_batch_returns.flatten(0, 1)
                    mini_batch_advantages = mini_batch_advantages.flatten(0, 1)

                    mini_batch_gae_c = mini_batch_gae_c.flatten(0, 1)
                    mini_batch_advantages_c = mini_batch_advantages_c.flatten(0, 1)

                    self._update_w_mini_batch(
                        mini_batch_obs,
                        mini_batch_recurrent_hidden_states,
                        mini_batch_cost,
                        mini_batch_value_preds,
                        mini_batch_cost_preds,
                        mini_batch_action_log_probs,
                        mini_batch_actions,
                        mini_batch_prev_actions,
                        mini_batch_not_done_masks,
                        mini_batch_returns,
                        mini_batch_advantages,
                        mini_batch_gae_c,
                        mini_batch_advantages_c,
                        mini_batch_rnn_build_seq_info,
                    )

            # Print a summary of our training so far
            # Save our model if it's time
            if i_so_far % self._save_freq == 0:
                self._save_model_weights()
            self._log_summary()

    def _update_w_mini_batch(
        self,
        mini_batch_obs,
        mini_batch_recurrent_hidden_states,
        mini_batch_cost,
        mini_batch_value_preds,
        mini_batch_cost_preds,
        mini_batch_action_log_probs,
        mini_batch_actions,
        mini_batch_prev_actions,
        mini_batch_not_done_masks,
        mini_batch_returns,
        mini_batch_advantages,
        mini_batch_gae_c,
        mini_batch_advantages_c,
        mini_batch_rnn_build_seq_info,
    ):

        # print(f"mini_batch_recurrent_hidden_states: {mini_batch_recurrent_hidden_states.shape}")
        # print(f"mini_batch_value_preds     : {mini_batch_value_preds.shape}")
        # print(f"mini_batch_action_log_probs: {mini_batch_action_log_probs.shape}")
        # print(f"mini_batch_actions         : {mini_batch_actions.shape}")
        # print(f"mini_batch_prev_actions    : {mini_batch_prev_actions.shape}")
        # print(f"mini_batch_not_done_masks  : {mini_batch_not_done_masks.shape}")

        # print("mini_batch_cost:", mini_batch_cost)
        # print("mini_batch_cost:", mini_batch_cost.shape)

        # print("mini_batch_advantages:", mini_batch_advantages.shape)
        # print("mini_batch_advantages_c:", mini_batch_advantages_c.shape)

        # print("mini_batch_returns:", mini_batch_returns.shape)
        # print("mini_batch_gae_c:", mini_batch_gae_c.shape)

        mini_batch_advantage_safe = (
            mini_batch_advantages - self._lambda_cost * mini_batch_advantages_c
        )

        value_theta, value_c_theta, action_log_probs_theta, dist_entropy_theta = (
            self._evaluate_actions(
                mini_batch_obs,
                mini_batch_recurrent_hidden_states,
                mini_batch_prev_actions,
                mini_batch_not_done_masks,
                mini_batch_actions,
            )
        )

        ratio = torch.exp(action_log_probs_theta - mini_batch_action_log_probs)

        surr_1 = mini_batch_advantage_safe * ratio
        surr_2 = mini_batch_advantage_safe * (
            torch.clamp(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio)
        )
        action_loss = -torch.min(surr_1, surr_2)

        value_theta = value_theta.float()
        value_c_theta = value_c_theta.float()

        if self._use_clipped_value_loss:
            delta = value_theta.detach() - mini_batch_value_preds
            value_pred_clipped = mini_batch_value_preds + delta.clamp(
                -self._clip_ratio, self._clip_ratio
            )
            value_theta = torch.where(
                delta.abs() < self._clip_ratio,
                value_theta,
                value_pred_clipped,
            )

            delta_c = value_c_theta.detach() - mini_batch_cost_preds
            value_c_pred_clipped = mini_batch_cost_preds + delta_c.clamp(
                -self._clip_ratio, self._clip_ratio
            )
            value_c_theta = torch.where(
                delta_c.abs() < self._clip_ratio,
                value_c_theta,
                value_c_pred_clipped,
            )

        value_loss = 0.5 * F.mse_loss(value_theta, mini_batch_returns, reduction="none")
        cost_loss = 0.5 * F.mse_loss(
            value_c_theta, mini_batch_cost_preds, reduction="none"
        )

        action_loss, value_loss, cost_loss, dist_entropy_theta = map(
            torch.mean,
            (action_loss, value_loss, cost_loss, dist_entropy_theta),
        )

        all_losses = [
            self._value_loss_coeff * value_loss,
            self._cost_loss_coeff * cost_loss,
            action_loss,
            -self._entropy_loss_coeff * dist_entropy_theta,
        ]
        total_loss = torch.stack(all_losses).sum()

        self._optim.zero_grad()
        total_loss.backward()
        self._optim.step()

        print("mini batch cost mean:", torch.mean(mini_batch_cost))
        # update cost lambda
        if self._use_fixed_dual_lr:
            self._dual_lr = self._fixed_dual_lr
        else:
            self._dual_lr = self._lr * self._lr_to_dual_lr
        self._lambda_cost = max(
            0.0,
            self._lambda_cost
            + self._dual_lr * (torch.mean(mini_batch_cost) - self._cost_limit),
        )

        self._logger["actor_losses"].append(action_loss.detach())
        self._logger["value_losses"].append(value_loss.detach())
        self._logger["cost_losses"].append(cost_loss.detach())
        self._logger["dist_entropies"].append(dist_entropy_theta.detach())

    def _multi_env_rollout(
        self,
    ) -> Tuple[
        dict,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_obs = {
            "pointgoal_with_gps_compass": [],  # (n_steps+1, n_env, 2)
            "rgb": [],  # (n_steps+1, n_env, 256, 256, 3)
        }
        batch_recurrent_hidden_states = []  # (n_steps+1, n_env, n_recurrent_layer, 512)
        batch_rewards = []  # (n_steps+1, n_env, 1)
        batch_costs = []  # (n_steps+1, n_env, 1)
        batch_value_preds = []  # (n_steps+1, n_env, 1)
        batch_cost_preds = []  # (n_steps+1, n_env, 1)
        # batch_returns = []  # (n_steps+1, n_env, 1)
        batch_action_log_probs = []  # (n_steps+1, n_env, 1)
        batch_actions = []  # (n_steps+1, n_env, 1)
        batch_prev_actions = []  # (n_steps+1, n_env, 1)
        batch_not_done_masks = []  # (n_steps+1, n_env, 1)

        prev_recurrent_hidden_states = torch.zeros(
            self._n_env,
            self._policy.net.num_recurrent_layers,
            self._hidden_size,
            device=self._device,
        )
        prev_actions = torch.zeros(
            self._n_env, 1, dtype=torch.long, device=self._device
        )
        prev_not_done_masks = torch.zeros(
            self._n_env, 1, dtype=torch.bool, device=self._device
        )

        # ---------- Reset envs ----------
        observations = self._envs.reset()
        prev_obs = self._batch_step_obs(observations)

        for batch_step in range(self._timesteps_per_batch):
            # print(f"batch_step {batch_step} / {self._timesteps_per_batch}")
            # ---------- Get action and step envs ----------
            (
                step_actions,  # (n_env, 1)
                step_recurrent_hidden_states,  # (n_env, n_recurrent_layer, 512)
                step_value_preds,  # (n_env, 1)
                step_cost_preds,  # (n_env, 1)
                step_action_log_probs,  # (n_env, 1)
            ) = self._get_actions_from_obs(
                prev_obs,
                prev_recurrent_hidden_states,
                prev_actions,
                prev_not_done_masks,
            )

            (
                step_obs,  # {"pointgoal_with_gps_compass": (n_env, 2), "rgb": (n_env, 256, 256, 3)}
                step_rewards,  # (n_env, 1)
                step_costs,  # (n_env, 1)
                step_not_done_masks,  # (n_env, 1)
            ) = self._step_venvs(
                step_actions,  # (n_env, 1)
            )

            batch_obs["pointgoal_with_gps_compass"].append(
                prev_obs["pointgoal_with_gps_compass"].clone()
            )
            batch_obs["rgb"].append(prev_obs["rgb"].clone())
            batch_recurrent_hidden_states.append(
                prev_recurrent_hidden_states.detach().clone()
            )
            batch_rewards.append(step_rewards.detach().clone())
            batch_costs.append(step_costs.detach().clone())
            batch_value_preds.append(step_value_preds.detach().clone())
            batch_cost_preds.append(step_cost_preds.detach().clone())
            batch_action_log_probs.append(step_action_log_probs.detach().clone())
            batch_actions.append(step_actions.detach().clone())
            batch_prev_actions.append(prev_actions.detach().clone())
            batch_not_done_masks.append(prev_not_done_masks.detach().clone())

            prev_obs = step_obs
            prev_actions = step_actions
            prev_recurrent_hidden_states = step_recurrent_hidden_states
            prev_not_done_masks = step_not_done_masks

            # If the environment is done, init prev_actions and hidden states.
            for idx_env in range(self._n_env):
                if not step_not_done_masks[idx_env]:
                    prev_actions[idx_env][0] = 0
                    prev_recurrent_hidden_states[idx_env] = torch.zeros(
                        self._policy.net.num_recurrent_layers,
                        self._hidden_size,
                        device=self._device,
                    )

        # ---------- Get the last value prediction ----------
        (
            step_actions,  # (n_env, 1)
            step_recurrent_hidden_states,  # (n_env, n_recurrent_layer, 512)
            step_value_preds,  # (n_env, 1)
            step_cost_preds,  # (n_env, 1)
            step_action_log_probs,  # (n_env, 1)
        ) = self._get_actions_from_obs(
            prev_obs,
            prev_recurrent_hidden_states,
            prev_actions,
            prev_not_done_masks,
        )

        batch_obs["pointgoal_with_gps_compass"].append(
            prev_obs["pointgoal_with_gps_compass"].clone()
        )
        batch_obs["rgb"].append(prev_obs["rgb"].clone())
        batch_recurrent_hidden_states.append(
            prev_recurrent_hidden_states.detach().clone()
        )
        batch_rewards.append(
            torch.zeros(self._n_env, 1, dtype=torch.long, device=self._device)
        )
        batch_costs.append(
            torch.zeros(self._n_env, 1, dtype=torch.long, device=self._device)
        )
        batch_value_preds.append(step_value_preds.detach().clone())
        batch_cost_preds.append(step_cost_preds.detach().clone())
        batch_action_log_probs.append(
            torch.zeros(self._n_env, 1, dtype=torch.long, device=self._device)
        )
        batch_actions.append(
            torch.zeros(self._n_env, 1, dtype=torch.long, device=self._device)
        )
        batch_prev_actions.append(prev_actions.detach().clone())
        batch_not_done_masks.append(prev_not_done_masks)

        # ---------- Convert to tensors ----------
        for k in batch_obs:
            batch_obs[k] = torch.stack(batch_obs[k], dim=0)
        batch_recurrent_hidden_states = torch.stack(
            batch_recurrent_hidden_states, dim=0
        )
        batch_rewards = torch.stack(batch_rewards, dim=0)
        batch_costs = torch.stack(batch_costs, dim=0)
        batch_value_preds = torch.stack(batch_value_preds, dim=0)
        batch_cost_preds = torch.stack(batch_cost_preds, dim=0)
        batch_action_log_probs = torch.stack(batch_action_log_probs, dim=0)
        batch_actions = torch.stack(batch_actions, dim=0)
        batch_prev_actions = torch.stack(batch_prev_actions, dim=0)
        batch_not_done_masks = torch.stack(batch_not_done_masks, dim=0)

        return (
            batch_obs,
            batch_recurrent_hidden_states,
            batch_rewards,
            batch_costs,
            batch_value_preds,
            batch_cost_preds,
            batch_action_log_probs,
            batch_actions,
            batch_prev_actions,
            batch_not_done_masks,
        )

    def _batch_step_obs(self, observations):
        step_obs = {
            "pointgoal_with_gps_compass": [
                obs["pointgoal_with_gps_compass"] for obs in observations
            ],  # (n_env, 2)
            "rgb": [obs["rgb"] for obs in observations],  # (n_env, 256, 256, 3)
        }

        step_obs["pointgoal_with_gps_compass"] = torch.tensor(
            np.array(step_obs["pointgoal_with_gps_compass"]), dtype=torch.float32
        ).to(self._device)
        step_obs["rgb"] = torch.tensor(
            np.array(step_obs["rgb"]), dtype=torch.float32
        ).to(self._device)

        return step_obs

    def _get_actions_from_obs(
        self,
        step_obs,
        prev_recurrent_hidden_states,
        prev_actions,
        not_done_masks,
    ) -> Tuple[
        torch.Tensor,  # step_actions (n_env, 1)
        torch.Tensor,  # step_recurrent_hidden_states (n_env,  n_recurrent_layer, 512)
        torch.Tensor,  # step_value_preds (n_env, 1)
        torch.Tensor,  # step_value_c_preds (n_env, 1)
        torch.Tensor,  # step_action_log_probs (n_env, 1)
    ]:
        """
        Parameters
        ----------
        step_obs: dict
            {
                "pointgoal_with_gps_compass": (n_env, 2),
                "rgb": (n_env, 256, 256, 3)
            }
        """
        features, rnn_hidden_states, _ = self._policy.net(
            observations=step_obs,
            rnn_hidden_states=prev_recurrent_hidden_states,
            prev_actions=prev_actions,
            masks=not_done_masks,
        )
        value = self._policy.critic(features)
        value_c = self._policy.critic_c(features)

        distribution = self._policy.action_distribution(features)
        action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        # batch_input = {
        #     "observations": step_obs,
        #     "rnn_hidden_states": prev_recurrent_hidden_states,
        #     "prev_actions": prev_actions,
        #     "masks": not_done_masks,
        # }
        # action_data = self._policy.act(**batch_input, deterministic=False)
        # print("action_data:", action_data)
        # print("actions shape:", action_data.actions.shape)  # [4, 1]
        # print("values  shape:", action_data.values.shape)  # [4, 1]
        # print("action_log_probs shape:", action_data.action_log_probs.shape)  # [4, 1]
        # print("rnn_hidden_states s:", action_data.rnn_hidden_states.shape)  # [4 4,512]

        return (
            action,
            rnn_hidden_states,
            value,
            value_c,
            action_log_probs,
        )

    def _evaluate_actions(
        self,
        mini_batch_obs,
        mini_batch_recurrent_hidden_states,
        mini_batch_prev_actions,
        mini_batch_not_done_masks,
        mini_batch_actions,
    ):
        features, rnn_hidden_states, aux_loss_state = self._policy.net(
            observations=mini_batch_obs,
            rnn_hidden_states=mini_batch_recurrent_hidden_states,
            prev_actions=mini_batch_prev_actions,
            masks=mini_batch_not_done_masks,
        )

        value = self._policy.critic(features)
        value_c = self._policy.critic_c(features)

        distribution = self._policy.action_distribution(features)
        action_log_probs = distribution.log_probs(mini_batch_actions)

        dist_entropy = distribution.entropy()

        return value, value_c, action_log_probs, dist_entropy

    def _step_venvs(
        self,
        actions: torch.Tensor,  # (n_env, 1)
    ) -> Tuple[
        dict,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        If the environment is done, reset it too.

        Returns
        -------
        s_t+1: dict
        r_t+1: torch.Tensor (n_env, 1)
        notdone_t+1: torch.Tensor (n_env, 1)
        """
        # step
        for index_env in range(self._n_env):
            action = actions[index_env].item()
            self._envs.async_step_at(index_env, action)

        # wait result

        outputs = [
            self._envs.wait_step_at(index_env) for index_env in range(self._n_env)
        ]

        observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

        step_obs = self._batch_step_obs(observations)
        step_rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self._device,
        )
        step_rewards = step_rewards.unsqueeze(1)
        step_costs = torch.tensor(
            [(1.0 if info["collision"] else 0.0) for info in infos],
            dtype=torch.float,
            device=self._device,
        )
        step_costs = step_costs.unsqueeze(1)

        step_not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self._device,
        )

        # Add logs
        for idx_env in range(self._n_env):
            self._logger["rewards_by_env"][idx_env] += step_rewards[idx_env].item()
            self._logger["ep_lens_by_env"][idx_env] += 1

            if not step_not_done_masks[idx_env]:
                self._num_ep += 1
                self._logger["batch_rews"].append(
                    self._logger["rewards_by_env"][idx_env]
                )
                self._logger["batch_num_wp_reached"].append(
                    infos[idx_env]["num_wp_reached"]
                )
                self._logger["batch_first_wp_reached"].append(
                    infos[idx_env]["first_wp_reached"]
                )
                self._logger["batch_l2_dist_to_goal"].append(
                    infos[idx_env]["l2_dist_to_goal"]
                )

                if infos[idx_env]["first_wp_reached"]:
                    self._logger["batch_l2_dist_to_1st_wp"].append(0.0)
                else:
                    self._logger["batch_l2_dist_to_1st_wp"].append(
                        infos[idx_env]["l2_dist_to_goal"]
                    )
                self._logger["batch_collisions"].append(infos[idx_env]["collision"])
                self._logger["batch_travel_distances"].append(
                    infos[idx_env]["travel_distance"]
                )
                self._logger["batch_timeouts"].append(infos[idx_env]["timeout"])
                self._logger["batch_ep_lens"].append(
                    self._logger["ep_lens_by_env"][idx_env]
                )
                self._logger["batch_count"] += 1

                # Reset logs
                self._logger["rewards_by_env"][idx_env] = 0.0
                self._logger["ep_lens_by_env"][idx_env] = 0

        return step_obs, step_rewards, step_costs, step_not_done_masks

    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self._logger["delta_t"]
        self._logger["delta_t"] = time.time_ns()
        delta_t = (self._logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        num_terminated_episodes = self._logger["batch_count"]

        t_so_far = self._logger["t_so_far"]
        i_so_far = self._logger["i_so_far"]
        avg_ep_rews = np.mean(self._logger["batch_rews"])
        avg_ep_lens = np.mean(self._logger["batch_ep_lens"])
        sr_1 = np.mean(self._logger["batch_first_wp_reached"])
        avg_wp_reached = np.mean(self._logger["batch_num_wp_reached"])
        avg_dist_traveled = np.mean(self._logger["batch_travel_distances"])
        avg_dist_to_1st_wp = np.mean(self._logger["batch_l2_dist_to_1st_wp"])
        avg_dist_to_goal = np.mean(self._logger["batch_l2_dist_to_goal"])
        avg_collisions = np.mean(self._logger["batch_collisions"])
        avg_timeouts = np.mean(self._logger["batch_timeouts"])

        avg_actor_loss = np.mean(
            [losses.cpu().float().mean() for losses in self._logger["actor_losses"]]
        )
        avg_value_loss = np.mean(
            [losses.cpu().float().mean() for losses in self._logger["value_losses"]]
        )
        avg_cost_loss = np.mean(
            [losses.cpu().float().mean() for losses in self._logger["cost_losses"]]
        )
        avg_dist_entropy = np.mean(
            [losses.cpu().float().mean() for losses in self._logger["dist_entropies"]]
        )

        if self._use_wandb:
            wandb.log(
                step=t_so_far,
                data={
                    "avg_ep_len": avg_ep_lens,
                    "reward": avg_ep_rews,
                    "avg_actor_loss": avg_actor_loss,
                    "cost_lambda": self._lambda_cost,
                    # habitat-lab metrics
                    "metrics/first_wp_reached": sr_1,
                    "metrics/num_wp_reached": avg_wp_reached,
                    "metrics/travel_distance": avg_dist_traveled,
                    "metrics/l2_dist_to_1st_wp": avg_dist_to_1st_wp,
                    "metrics/l2_dist_to_goal": avg_dist_to_goal,
                    "metrics/collision": avg_collisions,
                    "metrics/timeout": avg_timeouts,
                    "learner/action_loss": avg_actor_loss,
                    "learner/value_loss": avg_value_loss,
                    "learner/cost_loss": avg_cost_loss,
                    "learner/dist_entropy": avg_dist_entropy,
                },
            )

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_value_loss = str(round(avg_value_loss, 5))
        avg_dist_entropy = str(round(avg_dist_entropy, 5))

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"Iteration took  : {delta_t} secs", flush=True)
        print(f"Success Rate 1  : {sr_1}", flush=True)
        print(f"avg wp reached  : {avg_wp_reached}", flush=True)
        print(f"")
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Episodes So Far : {self._num_ep}", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Reward: {avg_ep_rews}", flush=True)
        print(f"Average Action Loss : {avg_actor_loss}", flush=True)
        print(f"Average Value Loss  : {avg_value_loss}", flush=True)
        print(f"Average Dist Entropy: {avg_dist_entropy}", flush=True)
        print(f"")
        print(f"avg dist traveled : {avg_dist_traveled}", flush=True)
        print(f"avg dist to 1st wp: {avg_dist_to_1st_wp}", flush=True)
        print(f"collisions       : {avg_collisions}", flush=True)
        print(f"timeouts         : {avg_timeouts}", flush=True)
        print(f"num terminated eps: {num_terminated_episodes}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self._logger["rewards_by_env"] = [0.0 for _ in range(self._n_env)]
        self._logger["ep_lens_by_env"] = [0 for _ in range(self._n_env)]

        self._logger["batch_rews"] = []
        self._logger["batch_ep_lens"] = []
        self._logger["batch_first_wp_reached"] = []
        self._logger["batch_num_wp_reached"] = []
        self._logger["batch_travel_distances"] = []
        self._logger["batch_l2_dist_to_1st_wp"] = []
        self._logger["batch_l2_dist_to_goal"] = []
        self._logger["batch_collisions"] = []
        self._logger["batch_timeouts"] = []
        self._logger["batch_count"] = 0
        self._logger["actor_losses"] = []
        self._logger["value_losses"] = []
        self._logger["cost_losses"] = []
        self._logger["dist_entropies"] = []
