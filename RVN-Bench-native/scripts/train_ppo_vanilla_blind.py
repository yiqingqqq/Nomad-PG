import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from algo.ppo_vanilla import PpoTrainerDiscrete
from rvn_envs.seq_point_goal_nav_env import SeqPointGoalNavEnv


def preproc_rvn_obs_blind(obs) -> np.ndarray:
    rvn_obs = obs["pointgoal_with_gps_compass"]

    return rvn_obs


def postproc_rvn_action(action: np.ndarray) -> np.ndarray:
    rvn_command = None
    if action == 0:
        rvn_command = "turn_left"
    elif action == 1:
        rvn_command = "move_forward"
    elif action == 2:
        rvn_command = "turn_right"
    return rvn_command


def main():
    curr_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_save_dir = f"logs/weights/rvn_v0/ppo_vanilla/{curr_datetime_str}"
    render_save_dir = f"logs/debug_images/rvn_v0/ppo_vanilla/{curr_datetime_str}"

    env = SeqPointGoalNavEnv()
    ppo = PpoTrainerDiscrete(
        env=env,
        hyper_parameters={
            "save_freq": 10,
            "timesteps_per_batch": 8192,
            "max_timesteps_per_episode": 1000,
            "gamma": 0.99,
            "lr": 0.001,
            "clip_ratio": 0.2,
            "n_updates_per_iteration": 5,
            "critic_loss_weight": 0.5,
        },
        preproc_obs=preproc_rvn_obs_blind,
        postproc_action=postproc_rvn_action,
        obs_dim=2,
        action_dim=3,
        weight_save_dir=weight_save_dir,
        render_save_dir=render_save_dir,
    )
    ppo.learn(
        int(1e8),
        use_wandb=True,
        wandb_project=f"rvn-{env.VERSION}",
        run_name="ppo-vanilla-blind",
    )


if __name__ == "__main__":
    main()
