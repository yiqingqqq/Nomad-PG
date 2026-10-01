import os
import sys
import time
from typing import Optional, Tuple, Union


from PIL import Image
import numpy as np
import wandb
import cv2

import torch
import torch.nn.functional as F
from torch.optim import Adam

from habitat import VectorEnv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import algo.utils.ppo_utils as ppo_utils
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange
import math
# README
# Prev action is probably not needed, but I left it for just in case. Feel free to remove it.
# Don't forget to implement save_model_weights method.
# Curretly, NoMaD outputs path using pure pursuit. Feel free to change it.

class DppoTrainer:
    def __init__(
        self,
        agent,
        envs: VectorEnv,
        preprocess_rgb_obs: bool = True,
        rgb_obs_preprocess_fnc=None,
        rgb_obs_preprocess_fnc_args: dict = {},
        preprocess_gps_obs: bool = True,
        gps_obs_preprocess_fnc=None,
        gps_obs_preprocess_fnc_args: dict = {},
        hyper_parameters: dict = {},
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        weight_save_dir: str = "weights",
        save_render: bool = False,
        render_save_dir: str = "images",
        device: torch.device = torch.device("cpu"),
    ):
        print(f"Init DppoTrainer device: {device} n_env: {envs.num_envs}")

        self._agent = agent
        self._envs = envs

        self._preprocess_rgb_obs = preprocess_rgb_obs
        self._rgb_obs_preprocess_fnc = rgb_obs_preprocess_fnc
        self._rgb_obs_preprocess_fnc_args = rgb_obs_preprocess_fnc_args
        self._preprocess_gps_obs = preprocess_gps_obs
        self._gps_obs_preprocess_fnc = gps_obs_preprocess_fnc
        self._gps_obs_preprocess_fnc_args = gps_obs_preprocess_fnc_args

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_size = hidden_size
        self._weight_save_dir = weight_save_dir
        self._device = device
        self._n_env = envs.num_envs

        os.makedirs(weight_save_dir, exist_ok=True)
        self._save_render = save_render
        self._render_save_dir = render_save_dir
        if self._save_render:
            os.makedirs(render_save_dir, exist_ok=True)
        self._init_hyperparmeters(**hyper_parameters, hidden_dim=hidden_size)

        assert self._n_env >= self._n_mini_batch
        if self._n_env % self._n_mini_batch != 0:
            print(
                f"Warning: n_env ({self._n_env}) % n_mini_batch ({self._n_mini_batch}) != 0"
            )

        # TODO: setup optimizer
        # nomad ema_model_avg: nn.Module
        # nomad noise_scheduler: DDPMScheduler

        # print(f"ema model: {self._agent._ema_model_avg}")
        # print(f"noise scheduler: {self._agent._noise_scheduler}")

        # TODO: change optimizer model parameters!!!
        self._agent_optim = Adam(
            self._agent._ema_model_avg.parameters(), lr=self._lr
        )
        self._critic_optim = Adam(
            self._agent._critic.parameters(), lr=self._critic_lr
        )

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
            "dist_entropies": [],  # entropy of action distribution in current iteration
            "approx_kl": [],  # approximate KL divergence in current iteration
            "ratio": [],  # ratio of action probabilities in current iteration
            "clip_frac": [] # ratio of probabilities the policy gradient was clipped
        }

    def _init_hyperparmeters(
        self,
        save_freq: int = 10,
        timesteps_per_batch: int = 1024,
        max_timesteps_per_episode: int = 1024,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 2.5e-4,
        critic_lr: float = 1e-3,
        eps: float = 1e-5,
        clip_ratio: float = 0.2,
        n_updates_per_iteration: int = 4,
        n_mini_batch: int = 2,
        value_loss_coeff: float = 0.5,
        entropy_loss_coeff: float = 0.01,
        normalize_advantage: bool = False,
        use_clipped_value_loss: bool = True,
        hidden_dim: Optional[Union[int, list]] = None,
        scheduler_timesteps: int = 10,
        gamma_denoising: float = 0.9,
        critic_warmup_steps: int= 10,
        clip_ploss_coef: float = 0.01,
        clip_ploss_coef_base: float = 0.01,
        clip_ploss_coef_rate: float = 1
    ):
        self._save_freq = save_freq

        self._timesteps_per_batch = timesteps_per_batch
        self._max_timesteps_per_episode = max_timesteps_per_episode

        self._gamma = gamma  # Discount factor
        self._lam = lam  # GAE lambda
        self._lr = lr
        self._critic_lr = critic_lr
        self._eps = eps
        self._clip_ratio = clip_ratio
        self._n_updates_per_iteration = n_updates_per_iteration
        self._n_mini_batch = n_mini_batch
        self._value_loss_coeff = value_loss_coeff
        self._entropy_loss_coeff = entropy_loss_coeff
        self._normalize_advantage = normalize_advantage
        self._use_clipped_value_loss = use_clipped_value_loss

        self._hidden_dim = hidden_dim
        self._scheduler_timesteps = scheduler_timesteps
        self._gamma_denoising = gamma_denoising

        self._critic_warmup_steps = critic_warmup_steps

        self._clip_ploss_coef = clip_ploss_coef
        self._clip_ploss_coef_base = clip_ploss_coef_base
        self._clip_ploss_coef_rate = clip_ploss_coef_rate

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
                batch_chains,
                batch_rewards,
                # batch_value_preds,
                # batch_action_log_probs,
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

            batch_value_preds, batch_action_log_probs = self._calculate_value_preds_and_log_probs(
                batch_obs,
                batch_chains,
                batch_not_done_masks
            )


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
            for idx_update_itr in range(self._n_updates_per_iteration):
                rand_perm = torch.randperm(self._n_env).chunk(self._n_mini_batch)
                is_kl_limit_reached = False
                for inds in rand_perm:
                    curr_slice = (slice(0, self._timesteps_per_batch), inds)
                    mini_batch_obs = {
                        "pointgoal_with_gps_compass": batch_obs[
                            "pointgoal_with_gps_compass"
                        ][curr_slice],
                        "rgb": batch_obs["rgb"][curr_slice],
                    }
                    mini_batch_rewards = batch_rewards[curr_slice]
                    mini_batch_value_preds = batch_value_preds[curr_slice]
                    mini_batch_action_log_probs = batch_action_log_probs[curr_slice]
                    mini_batch_actions = batch_actions[curr_slice]
                    mini_batch_prev_actions = batch_prev_actions[curr_slice]
                    mini_batch_not_done_masks = batch_not_done_masks[curr_slice]
                    mini_batch_returns = batch_returns[curr_slice]
                    mini_batch_advantages = batch_advantages[curr_slice]
                    mini_batch_chains = batch_chains[curr_slice]


                    

                    mini_batch_obs["rgb"] = mini_batch_obs["rgb"].flatten(0, 1)
                    mini_batch_obs["pointgoal_with_gps_compass"] = mini_batch_obs[
                        "pointgoal_with_gps_compass"
                    ].flatten(0, 1)
                    mini_batch_rewards = mini_batch_rewards.flatten(0, 1)
                    mini_batch_value_preds = mini_batch_value_preds.flatten(0, 1)
                    mini_batch_action_log_probs = mini_batch_action_log_probs.flatten(
                        0, 1
                    )
                    mini_batch_actions = mini_batch_actions.flatten(0, 1)
                    mini_batch_prev_actions = mini_batch_prev_actions.flatten(0, 1)
                    mini_batch_not_done_masks = mini_batch_not_done_masks.flatten(0, 1)
                    mini_batch_returns = mini_batch_returns.flatten(0, 1)
                    mini_batch_advantages = mini_batch_advantages.flatten(0, 1)
                    mini_batch_chains = mini_batch_chains.flatten(0, 1)  # (n_mini_batch * n_env, K+1, horizon, 2)



                    mini_batch_chains_timesteps = torch.randint(1, 5, (mini_batch_chains.shape[0],)).to(mini_batch_chains.device) # (_n_mini_batch, n_env)
                    # get indices from first index from mini_batch_chains, which is size (B, K+1, horizon, 2)

                    batch_indices = torch.arange(mini_batch_chains.size(0))

                    mini_batch_prev_chains = mini_batch_chains[batch_indices, mini_batch_chains_timesteps, ...]
                    mini_batch_next_chains = mini_batch_chains[batch_indices, mini_batch_chains_timesteps + 1, ...]
                    # mini_batch_prev_chains = mini_batch_chains[:, mini_batch_chains_timesteps, ...]
                    # mini_batch_next_chains = mini_batch_chains[:, mini_batch_chains_timesteps + 1, ...]

                    mini_batch_action_log_probs = mini_batch_action_log_probs[batch_indices, mini_batch_chains_timesteps, ...]

                    mini_batch_chain_timesteps = mini_batch_chains.shape[1] - 2 - mini_batch_chains_timesteps

                    

                    self._update_w_mini_batch(
                        mini_batch_obs,
                        mini_batch_value_preds,
                        mini_batch_action_log_probs,
                        mini_batch_actions,
                        mini_batch_prev_actions,
                        mini_batch_not_done_masks,
                        mini_batch_returns,
                        mini_batch_advantages,
                        mini_batch_prev_chains,
                        mini_batch_next_chains,
                        mini_batch_chain_timesteps,
                        i_so_far
                    )

                    if self._logger["approx_kl"][-1] > 1.0:
                        is_kl_limit_reached = True
                        break
                if is_kl_limit_reached:
                    break



            # Print a summary of our training so far
            # Save our model if it's time
            if i_so_far % self._save_freq == 0 or i_so_far == 1:
                self._save_model_weights()
                self._render_envs_by_gif(batch_obs, batch_actions)
            self._log_summary()

    def _calculate_value_preds_and_log_probs(
        self,
        batch_obs,
        batch_chains,
        batch_not_done_masks,
        batch_slice_size: int = 25,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Calculate value predictions and action log probabilities for the given batch of observations and chains.
        """

        B = batch_obs["pointgoal_with_gps_compass"].shape[0]  # (n_steps+1, n_env, 2)

        # calculate value predictions
        value_obs_inputs = {
            "pointgoal_with_gps_compass": rearrange(batch_obs["pointgoal_with_gps_compass"], "b e ... -> (b e) ..."),
            "rgb": rearrange(batch_obs["rgb"], "b e ... -> (b e) ..."),
        }

        # Prevent OOM errorb
        value_obs_input_slices = [
            ]
        
        num_slices = np.ceil(value_obs_inputs["pointgoal_with_gps_compass"].shape[0] / batch_slice_size).astype(int)
        for i in range(0, num_slices):
            value_obs_input_slices.append(
                {
                    "pointgoal_with_gps_compass": value_obs_inputs["pointgoal_with_gps_compass"][
                        i * batch_slice_size:min(i * batch_slice_size + batch_slice_size, value_obs_inputs["pointgoal_with_gps_compass"].shape[0]),
                    ],
                    "rgb": value_obs_inputs["rgb"][
                        i * batch_slice_size:min(i * batch_slice_size + batch_slice_size, value_obs_inputs["rgb"].shape[0]),
                    ],
                }   
            )

        batch_value_preds = []
        for value_obs_input in value_obs_input_slices:
            value_preds = self._agent.get_value_preds(
                value_obs_input,
                self._preprocess_gps_obs
            )
            value_preds = value_preds.detach()
            batch_value_preds.append(value_preds)   
        
        batch_value_preds = torch.cat(batch_value_preds, dim=0)  # (n_steps+1 *  n_env, 1)
        batch_value_preds = rearrange(
            batch_value_preds, "(b e) c -> b e c", b=B, e=self._n_env
        ) # (n_steps+1, n_env, 1)

        batch_chains = rearrange(
            batch_chains, "b e k h a -> (b e) k h a", b=B, e=self._n_env
        )

        batch_chains_slices = []
        num_chains_slices = np.ceil(batch_chains.shape[0] / batch_slice_size).astype(int)
        for i in range(0, num_chains_slices):
            batch_chains_slices.append(
                batch_chains[
                    i * batch_slice_size:min(i * batch_slice_size + batch_slice_size, batch_chains.shape[0]),
                ]
            )


        # calculate action log probabilities
        #TODO
        action_log_probs = []

        for idx in range(len(batch_chains_slices)):
            action_log_prob = self._agent.get_action_log_probs(
                value_obs_input_slices[idx],
                batch_chains_slices[idx],
                self._preprocess_gps_obs
            )
            action_log_prob = action_log_prob.detach()
            action_log_probs.append(action_log_prob)
        action_log_probs = torch.cat(action_log_probs, dim=0)  # (n_steps+1 * n_env, 1)
        action_log_probs = rearrange(
            action_log_probs, "(b e) ... -> b e ...", b=B, e=self._n_env
        )

        return batch_value_preds, action_log_probs

    def _update_w_mini_batch(
        self,
        mini_batch_obs,
        mini_batch_value_preds,
        mini_batch_action_log_probs,
        mini_batch_actions,
        mini_batch_prev_actions,
        mini_batch_not_done_masks,
        mini_batch_returns,
        mini_batch_advantages,
        mini_batch_prev_chains,
        mini_batch_next_chains,
        mini_batch_chain_timesteps,
        i_so_far
    ):

        # print(f"mini_batch_value_preds     : {mini_batch_value_preds.shape}")
        # print(f"mini_batch_action_log_probs: {mini_batch_action_log_probs.shape}")
        # print(f"mini_batch_actions         : {mini_batch_actions.shape}")
        # print(f"mini_batch_prev_actions    : {mini_batch_prev_actions.shape}")
        # print(f"mini_batch_not_done_masks  : {mini_batch_not_done_masks.shape}")
        # exit()

        value_theta, action_log_probs_theta, dist_entropy_theta = self._evaluate_actions(
                mini_batch_obs,
                mini_batch_prev_chains,
                mini_batch_next_chains,
                mini_batch_chain_timesteps
        )
        mini_batch_advantages = mini_batch_advantages.squeeze(-1)
        mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages.mean()) / (
            mini_batch_advantages.std() + 1e-8
        ) if self._normalize_advantage else mini_batch_advantages


        # apply chain discount
        mini_batch_advantages *= self._gamma_denoising ** mini_batch_chain_timesteps 
        # only use areas which actually affect the policy
        action_log_probs_theta = action_log_probs_theta[:,:4, :]
        mini_batch_action_log_probs = mini_batch_action_log_probs[:,:4, :]
        ratio = torch.exp(action_log_probs_theta.clamp(min=-5, max=2).mean(dim=(-1, -2)) - mini_batch_action_log_probs.clamp(min=-5, max=2).mean(dim=(-1, -2)))


        
        clip_ratio = self._clip_ploss_coef_base + (self._clip_ploss_coef - self._clip_ploss_coef_base) * \
            (torch.exp(self._clip_ploss_coef_rate * (self._scheduler_timesteps - 1 - mini_batch_chain_timesteps) / (self._scheduler_timesteps - 1)) - 1) / (math.exp(self._clip_ploss_coef_rate) - 1)


        


        surr_1 = mini_batch_advantages * ratio
        # surr_2 = mini_batch_advantages * (
        #     torch.clamp(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio)
        # )
        surr_2 = mini_batch_advantages * (
            torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        )
        action_loss = -torch.min(surr_1, surr_2)

        value_theta = value_theta.float()

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

        value_loss = 0.5 * F.mse_loss(value_theta, mini_batch_returns, reduction="none")

        action_loss, value_loss, dist_entropy_theta = map(
            torch.mean,
            (action_loss, value_loss, dist_entropy_theta),
        )

        all_losses = [
            self._value_loss_coeff * value_loss,
            action_loss,
            -self._entropy_loss_coeff * dist_entropy_theta,
        ]
        total_loss = torch.stack(all_losses).sum()

        # TODO: add backward pass
        self._agent_optim.zero_grad()
        self._critic_optim.zero_grad()
        total_loss.backward()
        if i_so_far > self._critic_warmup_steps:
            self._agent_optim.step()
        self._critic_optim.step()

        approx_kl = ((ratio - 1) -torch.log(ratio)).mean()
        clip_frac = ((ratio - 1.0).abs() > self._clip_ploss_coef).float().mean()

        self._logger["actor_losses"].append(action_loss.detach())
        self._logger["value_losses"].append(value_loss.detach())
        self._logger["dist_entropies"].append(dist_entropy_theta.detach())
        self._logger["approx_kl"].append(approx_kl.detach())
        self._logger["clip_frac"].append(clip_frac.detach())
        self._logger["ratio"].append(ratio.detach())

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
    ]:
        """
        Collect a batch of trajectories from multiple environments.
        Returns
        -------
        batch_obs: dict
            {
                "pointgoal_with_gps_compass": (n_steps+1, n_env, 2),
                "rgb": (n_steps+1, n_env, 256, 256, 3)
            }
        batch_chains: torch.Tensor (n_steps+1, n_env, K+1, horizon, 2)
        batch_rewards: torch.Tensor (n_steps+1, n_env, 1)
        # batch_value_preds: torch.Tensor (n_steps+1, n_env, 1)
        # batch_action_log_probs: torch.Tensor (n_steps+1, n_env, 1)
        batch_actions: torch.Tensor (n_steps+1, n_env, 1)
        batch_prev_actions: torch.Tensor (n_steps+1, n_env, 1)
        batch_not_done_masks: torch.Tensor (n_steps+1, n_env, 1)
        """


        batch_obs = {
            "pointgoal_with_gps_compass": [],  # (n_steps+1, n_env, 2)
            "rgb": [],  # (n_steps+1, n_env, 256, 256, 3)
        }
        batch_chains = [] # (n_steps+1, n_env, K+1, horizon, 2)
        batch_rewards = []  # (n_steps+1, n_env, 1)
        # batch_value_preds = []  # (n_steps+1, n_env, 1)
        # batch_returns = []  # (n_steps+1, n_env, 1)
        # batch_action_log_probs = []  # (n_steps+1, n_env, 1)
        batch_actions = []  # (n_steps+1, n_env, 1)
        batch_prev_actions = []  # (n_steps+1, n_env, 1)
        batch_not_done_masks = []  # (n_steps+1, n_env, 1)

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
            # (
            #     step_actions,  # (n_env, 1)
            #     step_value_preds,  # (n_env, 1)
            #     step_action_log_probs,  # (n_env, 1)
            # ) = self._get_actions_from_obs(
            #     prev_obs,
            #     prev_actions,
            #     prev_not_done_masks,
            #     return_chains=True
            # )


        
            (
                step_actions,  # (n_env, 1)
                model_output_dict,  # dict with chains
            ) = self._get_actions_from_obs(
                prev_obs,
                prev_actions,
                prev_not_done_masks,
                return_chains=True
            )

            (
                step_obs,  # {"pointgoal_with_gps_compass": (n_env, 2), "rgb": (n_env, 256, 256, 3)}
                step_rewards,  # (n_env, 1)
                step_not_done_masks,  # (n_env, 1)
            ) = self._step_venvs(
                step_actions,  # (n_env, 1)
            )

            batch_obs["pointgoal_with_gps_compass"].append(
                prev_obs["pointgoal_with_gps_compass"].clone()
            )
            batch_obs["rgb"].append(prev_obs["rgb"].clone())
            batch_chains.append(
                model_output_dict["gc_chain"].detach().clone()
            )  # (n_env, K+1, horizon, 2)
            batch_rewards.append(step_rewards.detach().clone())
            batch_actions.append(step_actions.detach().clone())
            batch_prev_actions.append(prev_actions.detach().clone())
            batch_not_done_masks.append(prev_not_done_masks.detach().clone())

            prev_obs = step_obs
            prev_actions = step_actions
            prev_not_done_masks = step_not_done_masks

            # If the environment is done, init prev_actions and hidden states.
            for idx_env in range(self._n_env):
                if not step_not_done_masks[idx_env]:
                    prev_actions[idx_env][0] = 0

        # ---------- Get the last value prediction ----------
        (
            step_actions,  # (n_env, 1)
            model_output_dict,  # dict with chains
        ) = self._get_actions_from_obs(
            prev_obs,
            prev_actions,
            prev_not_done_masks,
            return_chains=True
        )

        batch_obs["pointgoal_with_gps_compass"].append(
            prev_obs["pointgoal_with_gps_compass"].clone()
        )
        batch_obs["rgb"].append(prev_obs["rgb"].clone())
        batch_chains.append(
            model_output_dict["gc_chain"].detach().clone()
        )
        batch_rewards.append(
            torch.zeros(self._n_env, 1, dtype=torch.long, device=self._device)
        )
        # batch_value_preds.append(step_value_preds.detach().clone())
        # batch_action_log_probs.append(
        #     torch.zeros(self._n_env, 1, dtype=torch.long, device=self._device)
        # )
        batch_actions.append(
            torch.zeros(self._n_env, 1, dtype=torch.long, device=self._device)
        )
        batch_prev_actions.append(prev_actions.detach().clone())
        batch_not_done_masks.append(prev_not_done_masks)

        # ---------- Convert to tensors ----------
        for k in batch_obs:
            batch_obs[k] = torch.stack(batch_obs[k], dim=0)
        batch_chains = torch.stack(batch_chains, dim=0)  # (n_steps+1, n_env, K+1, horizon, 2)
        batch_rewards = torch.stack(batch_rewards, dim=0)
        # batch_value_preds = torch.stack(batch_value_preds, dim=0)
        # batch_action_log_probs = torch.stack(batch_action_log_probs, dim=0)
        batch_actions = torch.stack(batch_actions, dim=0)
        batch_prev_actions = torch.stack(batch_prev_actions, dim=0)
        batch_not_done_masks = torch.stack(batch_not_done_masks, dim=0)

        return (
            batch_obs,
            batch_chains,
            batch_rewards,
            # batch_value_preds,
            # batch_action_log_probs,
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

        if self._preprocess_gps_obs:
            step_obs["pointgoal_with_gps_compass"] = self._gps_obs_preprocess_fnc(
                np.array(step_obs["pointgoal_with_gps_compass"]),
                device=self._device,
                **self._gps_obs_preprocess_fnc_args,
            )
        else:
            step_obs["pointgoal_with_gps_compass"] = torch.tensor(
                np.array(step_obs["pointgoal_with_gps_compass"]), dtype=torch.float32
            ).to(self._device)

        if self._preprocess_rgb_obs:
            step_obs["rgb"] = self._rgb_obs_preprocess_fnc(
                np.array(step_obs["rgb"]),
                device=self._device,
                **self._rgb_obs_preprocess_fnc_args,
            )
        else:
            step_obs["rgb"] = torch.tensor(
                np.array(step_obs["rgb"]), dtype=torch.float32
            ).to(self._device)

        return step_obs

    def _get_actions_from_obs(
        self,
        step_obs,
        prev_actions,
        not_done_masks,
        return_chains: bool = False,
    ) -> Tuple[
        torch.Tensor,  # step_actions (n_env, 1)
        # torch.Tensor,  # step_value_preds (n_env, 1)
        # torch.Tensor,  # step_action_log_probs (n_env, 1)
    ]:
        """
        Parameters
        ----------
        step_obs: dict
            {
                "pointgoal_with_gps_compass": (n_env, 2),
                "rgb": (n_env, obs_size, 256, 256, 3)
            }
        """

        # TOOD: get action from the agent
        # values = torch.zeros(self._n_env, 1, dtype=torch.float, device=self._device)
        # action_log_probs = torch.zeros(
        #     self._n_env, 1, dtype=torch.float, device=self._device
        # )
        if return_chains:
            agent_output = self._agent.act(
                obs=step_obs,
                rgb_obs_preprocessed=self._preprocess_rgb_obs,
                gps_obs_preprocessed=self._preprocess_gps_obs,
                get_action_as_batch_idx=True,
                return_chain=return_chains,  # use only for training dppo
            )
            actions= agent_output["actions"]  # (n_env, 1)
            model_output_dict = agent_output["model_output_dict"]
            return (
                actions,
                model_output_dict
            )
        else:
            # For inference, we don't need chains
            actions = self._agent.act(
                obs=step_obs,
                rgb_obs_preprocessed=self._preprocess_rgb_obs,
                gps_obs_preprocessed=self._preprocess_gps_obs,
                get_action_as_batch_idx=True,
                return_chain=return_chains,  # use only for training dppo
            )
            return (
                actions,
                # values,
                # action_log_probs,
            )
        
        # # print("action_data:", action_data)
        # # print("actions shape:", action_data.actions.shape)  # [4, 1]
        # # print("values  shape:", action_data.values.shape)  # [4, 1]
        # # print("action_log_probs shape:", action_data.action_log_probs.shape)  # [4, 1]

        # return (
        #     actions,
        #     # values,
        #     # action_log_probs,
        # )

    def _evaluate_actions(
        self,
        mini_batch_obs,
        mini_batch_prev_chains,
        mini_batch_next_chains,
        mini_batch_chain_timesteps
    ):
        value_preds = self._agent.get_value_preds(
                mini_batch_obs,
                self._preprocess_gps_obs
            )  # (n_mini_batch * n_env, 1)
        action_log_probs, dist_entropy = self._agent.get_subsample_action_log_probs_and_entropy(
                mini_batch_obs,
                mini_batch_prev_chains,
                mini_batch_next_chains,
                mini_batch_chain_timesteps,
                self._preprocess_gps_obs
            ) # (n_mini_batch, horizon, action_dim), (n_mini_batch, horizon, action_dim)
        
        return (value_preds, action_log_probs, dist_entropy)


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
        return step_obs, step_rewards, step_not_done_masks

    def _save_model_weights(self):
        torch.save(
            self._agent._ema_model_avg.state_dict(),
            os.path.join(self._weight_save_dir, f"agent_{self._logger['i_so_far']}.pth"),
        )

        torch.save(
            self._agent._critic.state_dict(),
            os.path.join(self._weight_save_dir, f"critic_{self._logger['i_so_far']}.pth"),
        )




    def _render_envs(
        self,
        batch_obs: dict,
        batch_actions: torch.Tensor,  # (n_steps+1, n_env, 1)
    ):
        if not self._save_render:
            return

        print("Rendering envs...")
        print(
            f"batch_obs['rgb'].shape: {batch_obs['rgb'].shape}"
        )  # (n_steps+1, n_env, 3 * n_context, 96, 96)
        rgb_tensor = batch_obs["rgb"]
        T, E, C, H, W = rgb_tensor.shape
        n_context = C // 3  # Assuming RGB channels are grouped in sets of 3

        for t in range(T):
            for e in range(E):
                full_img = rgb_tensor[t, e]  # shape: [3*n_context, H, W]
                images = []
                for i in range(n_context):
                    img_tensor = full_img[i * 3 : (i + 1) * 3]  # shape: [3, H, W]
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

                    if img_np.dtype != "uint8":
                        img_np = (img_np * 255).clip(0, 255).astype("uint8")

                    images.append(Image.fromarray(img_np))

                # Create a new canvas to paste side-by-side
                strip = Image.new("RGB", (n_context * W, H))
                for i, img in enumerate(images):
                    strip.paste(img, (i * W, 0))

                fname = os.path.join(
                    self._render_save_dir,
                    f"iter_{self._logger['i_so_far']}_env{e}_{t}_a_{batch_actions[t][e].item()}.png",
                )
                strip.save(fname)
        print("Rendering done.")

    def _render_envs_by_gif(
        self,
        batch_obs: dict,
        batch_actions: torch.Tensor,  # (n_steps+1, n_env, 1)
    ):
        if not self._save_render:
            return

        print("Rendering envs...")
        print(
            f"batch_obs['rgb'].shape: {batch_obs['rgb'].shape}"
        )  # (n_steps+1, n_env, 3 * n_context, 96, 96)

        rgb_tensor = batch_obs["rgb"]
        T, E, C, H, W = rgb_tensor.shape
        n_context = C // 3  # Each context image is RGB (3 channels)

        os.makedirs(self._render_save_dir, exist_ok=True)

        # Collect frames per env
        env_gif_frames = [[] for _ in range(E)]  # List of lists of PIL.Image

        for t in range(T):
            for e in range(E):
                full_img = rgb_tensor[t, e]  # [3*n_context, H, W]
                images = []

                for i in range(n_context):
                    img_tensor = full_img[i * 3 : (i + 1) * 3]  # [3, H, W]
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

                    if img_np.dtype != "uint8":
                        img_np = (img_np * 255).clip(0, 255).astype("uint8")

                    images.append(Image.fromarray(img_np))

                # Create a side-by-side strip
                strip = Image.new("RGB", (n_context * W, H))
                for i, img in enumerate(images):
                    strip.paste(img, (i * W, 0))

                # Optionally annotate with action
                # from PIL import ImageDraw, ImageFont
                # draw = ImageDraw.Draw(strip)
                # draw.text((5, 5), f"A: {batch_actions[t][e].item()}", fill="white")

                env_gif_frames[e].append(strip)

        # Save one gif per environment
        for e, frames in enumerate(env_gif_frames):
            gif_path = os.path.join(
                self._render_save_dir, f"iter_{self._logger['i_so_far']}_env{e}.gif"
            )
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=200,  # ms per frame
                loop=0,
            )

        print("GIF rendering done.")

    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self._logger["delta_t"]
        self._logger["delta_t"] = time.time_ns()
        delta_t = (self._logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        num_terminated_episodes = self._logger["batch_count"]
        # if self._logger["batch_first_wp_reached"] == []:
        #     import ipdb;ipdb.set_trace()
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
        avg_dist_entropy = np.mean(
            [losses.cpu().float().mean() for losses in self._logger["dist_entropies"]]
        )

        avg_kl = np.mean([kl.cpu().float().mean() for kl in self._logger["approx_kl"]])

        avg_ratio = np.mean(
            [ratio.cpu().float().mean() for ratio in self._logger["ratio"]]
        )

        avg_clip_frac = np.mean(
            [clip_frac.cpu().float().mean() for clip_frac in self._logger["clip_frac"]]
        )

        if self._use_wandb:
            wandb.log(
                step=t_so_far,
                data={
                    "avg_ep_len": avg_ep_lens,
                    "reward": avg_ep_rews,
                    "avg_actor_loss": avg_actor_loss,
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
                    "learner/dist_entropy": avg_dist_entropy,
                    "learner/approx_kl": avg_kl,
                    "learner/ratio": avg_ratio,
                    "learner/clip_frac": avg_clip_frac
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
        self._logger["dist_entropies"] = []
        self._logger["approx_kl"] = []
        self._logger["ratio"] = []
        self._logger["clip_frac"] = []
