import os
import sys
import time
from typing import Optional, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.optim import Adam
import gymnasium as gym
import imageio
import wandb

torch.autograd.set_detect_anomaly(True)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.critic = nn.Linear(hidden_dim, 1)
        self.actor = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))

        return self.critic(x), F.softmax(self.actor(x), dim=-1)


def preproc_obs(obs) -> np.ndarray:
    return obs


def postproc_action(action: np.ndarray, obs=None):
    return action


class PpoTrainerDiscrete:
    def __init__(
        self,
        env: gym.Env,
        hyper_parameters: dict = {},
        preproc_obs=preproc_obs,
        postproc_action=postproc_action,
        ActorCriticModel=ActorCritic,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        hidden_dim: Optional[Union[int, list]] = None,
        weight_save_dir: str = "weights",
        render_save_dir: str = "images",
        weight_load_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self._render = True
        print(f"Init PPO Trainer device: {device}")
        if obs_dim is not None:
            self._obs_dim = obs_dim
        else:
            assert type(env.observation_space) == gym.spaces.Box
            print("\tobs_dim shape :", env.observation_space.shape)
            self._obs_dim = env.observation_space.shape[0]

        if action_dim is not None:
            self._action_dim = action_dim
        else:
            assert type(env.action_space) == gym.spaces.Discrete
            print("\taction_space n:", env.action_space.n)
            self._action_dim = env.action_space.n

        self._env = env
        self._preproc_obs = preproc_obs
        self._postproc_action = postproc_action
        self._weight_save_dir = weight_save_dir
        self._render_save_dir = render_save_dir
        self._device = device

        os.makedirs(weight_save_dir, exist_ok=True)
        os.makedirs(render_save_dir, exist_ok=True)

        self._init_hyperparmeters(**hyper_parameters, hidden_dim=hidden_dim)

        # stdev is chosen to be 0.5 arbitrarily
        cov_var_stdev = 0.5
        cov_var = torch.full(size=(self._action_dim,), fill_value=cov_var_stdev)
        self._cov_mat = torch.diag(cov_var)

        # ALG STEP 1
        # Initialize actor and critic networks
        if hidden_dim is not None:
            self._policy = ActorCriticModel(
                obs_dim=self._obs_dim,
                action_dim=self._action_dim,
                hidden_dim=hidden_dim,
            ).to(self._device)
        else:
            self._policy = ActorCriticModel(
                obs_dim=self._obs_dim, action_dim=self._action_dim
            ).to(self._device)

        if weight_load_path is not None:
            self._policy.load_state_dict(torch.load(weight_load_path))

        self._optim = Adam(self._policy.parameters(), lr=self._lr, eps=self._eps)

        print(self._policy)

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
        print(f"hidden_dim             : {self._hidden_dim}")

        self._num_ep = 0
        self._recent_reward_sum = 0.0
        self._use_wandb = use_wandb
        self._best_train_avg_ep_rew = -np.inf

        print("hidden_dim:", self._hidden_dim)

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
                batch_acts,
                batch_log_probs,
                # batch_reward_to_go,
                batch_rewards,
                batch_vals,
                batch_dones,
                batch_lens,
            ) = self._rollout()

            # print(f"batch_rewards len: {len(batch_rewards)}")
            # print(f"batch_vals len   : {len(batch_vals)}")
            # print(f"batch_done len   : {len(batch_dones)}")

            # print(f"batch_rewards[0]len: {len(batch_rewards[0])}")
            # print(f"batch_vals[0]   len: {len(batch_vals[0])}")
            # print(f"batch_done[0]   len: {len(batch_dones[0])}")

            # print(f"batch_rewards[0]: {batch_rewards[0]}")
            # print(f"batch_vals[0]   : {batch_vals[0]}")
            # print(f"batch_done[0]   : {batch_dones[0]}")
            # exit()

            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            # Logging timesteps so far and iterations so far
            self._logger["t_so_far"] = t_so_far
            self._logger["i_so_far"] = i_so_far

            # Adventage
            # A^π(s, a) = Q^π(s, a) - V_{Φk}(s)
            V_obs = self._get_value(batch_obs)

            # ALG STEP 5
            # Calculate the adventage
            # A_k = batch_reward_to_go - V_obs.detach()
            A_k = self._calculate_gae_advantage(batch_rewards, batch_vals, batch_dones)
            batch_reward_to_go = A_k + V_obs.detach()

            # Note that we do V.detach() since V is a tensor with gradient required.
            # However, the advantage will need to be reused each epoch loop,
            # and the computation graph associated with advantage at the k-th iteration
            # will not be useful in multiple epochs of stochastic gradient ascent.

            # Tricks: Normalize the adventage
            # Raw advantage makes PPO training highly unstable
            if self._normalize_advantage:
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            n_step = batch_acts.size(0)
            inds = np.arange(n_step)
            minibatch_size = n_step // self._n_mini_batch + 1

            # ALG STEP 6
            for _ in range(self._n_updates_per_iteration):
                np.random.shuffle(inds)
                for start in range(0, n_step, minibatch_size):
                    end = start + minibatch_size
                    minibatch_inds = inds[start:end]

                    mini_batch_obs = self._get_mini_batch_obs(batch_obs, minibatch_inds)
                    mini_batch_acts = batch_acts[minibatch_inds]
                    mini_batch_log_probs = batch_log_probs[minibatch_inds]
                    mini_batch_A_k = A_k[minibatch_inds]
                    mini_batch_reward_to_go = batch_reward_to_go[minibatch_inds]

                    # Calculate the loss and update the network
                    curr_v, curr_a_log_probs, entropy = self._evaluate_policy(
                        mini_batch_obs, mini_batch_acts
                    )

                    # Calculate the ratio
                    ratio = torch.exp(curr_a_log_probs - mini_batch_log_probs)

                    # Calculate surrogate loss
                    surr1 = ratio * mini_batch_A_k
                    surr2 = (
                        torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
                        * mini_batch_A_k
                    )

                    actor_loss = -(torch.min(surr1, surr2)).mean()
                    critic_loss = torch.nn.MSELoss()(curr_v, mini_batch_reward_to_go)
                    entropy_loss = entropy.mean()

                    # Combine actor and critic losses
                    total_loss = (
                        actor_loss
                        + self._value_loss_coeff * critic_loss
                        - self._entropy_loss_coeff * entropy_loss
                    )  # Scaling critic_loss if needed

                    self._optim.zero_grad()
                    total_loss.backward()
                    self._optim.step()

                    # Log actor loss
                    self._logger["actor_losses"].append(actor_loss.detach())

            # Print a summary of our training so far
            # Save our model if it's time
            if i_so_far % self._save_freq == 0:
                self._save_model_weights()
            self._log_summary()

    def _rollout(self):
        # Batch data
        batch_obs = []  # (timesteps_per_batch, obs_dim)
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
            obs, _ = self._env.reset()
            ep_done = False

            for ep_t in range(self._max_timesteps_per_episode):
                if render_episode:
                    frame = self._env.render()
                    frames.append(frame)

                batch_t += 1

                rvn_obs = self._preproc_obs(obs)

                ep_dones.append(ep_done)
                # Collect observation
                batch_obs.append(rvn_obs)

                rvn_obs = torch.tensor(rvn_obs, dtype=torch.float).to(self._device)
                v_obs, action_probs = self._policy(rvn_obs)
                dist = Categorical(action_probs)
                action = dist.sample()
                action_log_prop = dist.log_prob(action)

                obs, reward, terminated, truncated, info = self._env.step(
                    self._postproc_action(action.cpu().detach().numpy(), obs)
                )

                # Collect reward, action, and log_prob
                ep_rewards.append(reward)
                ep_vals.append(v_obs.flatten())

                batch_acts.append(action)
                batch_log_probs.append(action_log_prop)

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

        batch_obs = np.array(batch_obs)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self._device)
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

    def _init_hyperparmeters(
        self,
        save_freq: int = 10,
        timesteps_per_batch: int = 1024,
        max_timesteps_per_episode: int = 1024,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 2.5e-4,
        eps: float = 1e-5,
        clip_ratio: float = 0.2,
        n_updates_per_iteration: int = 4,
        n_mini_batch: int = 2,
        value_loss_coeff: float = 0.5,
        entropy_loss_coeff: float = 0.01,
        normalize_advantage: bool = False,
        use_clipped_value_loss: bool = True,
        hidden_dim: Optional[Union[int, list]] = None,
    ):
        self._save_freq = save_freq

        self._timesteps_per_batch = timesteps_per_batch
        self._max_timesteps_per_episode = max_timesteps_per_episode

        self._gamma = gamma  # Discount factor
        self._lam = lam  # GAE lambda
        self._lr = lr
        self._eps = eps
        self._clip_ratio = clip_ratio
        self._n_updates_per_iteration = n_updates_per_iteration
        self._n_mini_batch = n_mini_batch
        self._value_loss_coeff = value_loss_coeff
        self._entropy_loss_coeff = entropy_loss_coeff
        self._normalize_advantage = normalize_advantage
        self._use_clipped_value_loss = use_clipped_value_loss

        self._hidden_dim = hidden_dim

    def _compute_reward_to_go(self, batch_rewards):
        """
        Parameters
        ----------
        batch_rewards : list
            (num_episodes, timestpes_per_episode)
        """
        # The rewards-to-go (rtg) per episode per timestep
        # The shape will be (timesetps_per_episode)
        batch_reward_to_go = []

        # Iterate through each episode backwards to maintain same order
        # in batch_reward_to_go

        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(ep_rewards):
                discounted_reward = reward + self._gamma * discounted_reward
                batch_reward_to_go.insert(0, discounted_reward)

        batch_reward_to_go = torch.tensor(batch_reward_to_go, dtype=torch.float).to(
            self._device
        )
        return batch_reward_to_go

    def _calculate_gae_advantage(self, rewards, values, dones):
        batch_advantages = []  # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []  # List to store advantages for the current episode
            last_advantage = 0  # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = (
                        ep_rews[t]
                        + self._gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1])
                        - ep_vals[t]
                    )
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = (
                    delta + self._gamma * self._lam * (1 - ep_dones[t]) * last_advantage
                )
                last_advantage = (
                    advantage  # Update the last advantage for the next timestep
                )
                advantages.insert(
                    0, advantage
                )  # Insert advantage at the beginning of the list

            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float).to(self._device)

    def _get_value(self, batch_obs):
        """
        Returns
        -------
        V(obs) : torch.tensor
            (batch_size, 1)
        """
        V_obs, _ = self._policy(batch_obs)
        return V_obs.squeeze(1)

    def _evaluate_policy(self, batch_obs, batch_acts):
        """
        Returns
        -------
        V(obs) : torch.tensor
            (batch_size, 1)
        action_log_probs: torch.tensor
            (batch_size, )
        entropy: torch.tensor
            (batch_size, )
        """
        V_obs, all_action_log_probs = self._policy(batch_obs)

        dist = Categorical(all_action_log_probs)
        action_log_probs = dist.log_prob(batch_acts)
        entropy = dist.entropy()

        return V_obs.squeeze(1), action_log_probs, entropy

    def _get_mini_batch_obs(self, batch_obs, minibatch_inds):
        return batch_obs[minibatch_inds]

    def _save_model_weights(self):
        avg_ep_rews = np.mean(
            [np.sum(ep_rews) for ep_rews in self._logger["batch_rews"]]
        )
        if avg_ep_rews > self._best_train_avg_ep_rew:
            self._best_train_avg_ep_rew = avg_ep_rews
            print(f"Saving best model with avg_ep_rews {avg_ep_rews}")
            torch.save(
                self._policy.state_dict(),
                os.path.join(
                    self._weight_save_dir,
                    f"ppo_actor_critic_best_{self._num_ep}.pth",
                ),
            )
        torch.save(
            self._policy.state_dict(),
            os.path.join(self._weight_save_dir, f"ppo_vanila_policy_latest.pth"),
        )

    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self._logger["delta_t"]
        self._logger["delta_t"] = time.time_ns()
        delta_t = (self._logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self._logger["t_so_far"]
        i_so_far = self._logger["i_so_far"]
        avg_ep_lens = np.mean(self._logger["batch_lens"])
        avg_ep_rews = np.mean(
            [np.sum(ep_rews) for ep_rews in self._logger["batch_rews"]]
        )
        avg_actor_loss = np.mean(
            [losses.cpu().float().mean() for losses in self._logger["actor_losses"]]
        )
        print("Batnch_num_wp_reached:", self._logger["batch_num_wp_reached"])
        avg_wp_reached = np.mean(self._logger["batch_num_wp_reached"])
        sr_1 = np.mean(
            [
                1 if num_wp_reached >= 1 else 0
                for num_wp_reached in self._logger["batch_num_wp_reached"]
            ]
        )

        if self._use_wandb:
            wandb.log(
                step=self._num_ep,
                data={
                    "episode": self._num_ep,
                    "reward_sum": self._recent_reward_sum,
                    "avg_ep_len": avg_ep_lens,
                    "reward": avg_ep_rews,
                    "avg_actor_loss": avg_actor_loss,
                    "metrics/num_wp_reached": avg_wp_reached,
                    "metrics/first_wp_reached": sr_1,
                },
            )

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss    : {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Episodes So Far : {self._num_ep}", flush=True)
        print(f"Iteration took  : {delta_t} secs", flush=True)
        print(f"Success Rate 1  : {sr_1}", flush=True)
        print(f"avg wp reached  : {avg_wp_reached}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self._logger["batch_lens"] = []
        self._logger["batch_rews"] = []
        self._logger["actor_losses"] = []
        self._logger["batch_num_wp_reached"] = []


if __name__ == "__main__":

    from datetime import datetime

    curr_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_save_dir = f"logs/weights/ppo_vanilla/{curr_datetime_str}"
    render_save_dir = f"logs/debug_images/ppo_vanilla/{curr_datetime_str}"

    def preproc_lunar_obs(obs) -> np.ndarray:
        # normalize obs
        obs[0] /= 2.5
        obs[1] /= 2.5
        obs[2] /= 10.0
        obs[3] /= 10.0
        obs[4] /= 6.2831855
        obs[5] /= 10.0
        return obs

    env = gym.make(
        "LunarLander-v3",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=0.0,
        turbulence_power=1.5,
        render_mode="rgb_array",
    )
    # env=gym.make("Acrobot-v1", render_mode="rgb_array"),
    # env=gym.make("CartPole-v1", render_mode="rgb_array"),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo = PpoTrainerDiscrete(
        env=env,
        hyper_parameters={
            "timesteps_per_batch": 4096,
            "save_freq": 4,
            "lr": 1.0e-3,
            "n_updates_per_iteration": 5,
            "n_mini_batch": 64,
            "entropy_loss_coeff": 0.01,
        },
        preproc_obs=preproc_lunar_obs,
        ActorCriticModel=ActorCritic,
        weight_save_dir=weight_save_dir,
        render_save_dir=render_save_dir,
        hidden_dim=64,
        device=device,
    )
    ppo.learn(
        int(1e8),
        use_wandb=False,
        wandb_project="LunarLander-v3",
        run_name="ppo-gae",
    )
