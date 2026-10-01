import os
import sys

from habitat import VectorEnv


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rvn_envs.seq_point_goal_nav_env import SeqPointGoalNavEnv


class SeqPointGoalNavEnvCreator:
    @classmethod
    def make_rvn_seq_point_goal_nav_env_fn(
        cls,
        config,
    ):
        env = SeqPointGoalNavEnv(
            config_file_path=config["config_file_path"],
            scene_selection_mode=config["scene_selection_mode"],
            num_scene_repeats=config["num_scene_repeats"],
            use_gym_v25_interface=config["use_gym_v25_interface"],
            use_stop_action=config["use_stop_action"],
            obs_size=config["obs_size"],
        )
        env.seed(config["seed"])
        return env


def get_seq_point_goal_nav_venv(
    num_environments: int,
    num_scene_repeats: int = 100,
    use_gym_v25_interface=True,
    use_stop_action: bool = True,
    obs_size: int = 1
):
    args = tuple(
        (
            {
                "config_file_path": "configs/rl_envs/seq_point_goal_nav_env_train.yaml",
                "num_scene_repeats": num_scene_repeats,
                "scene_selection_mode": "random",
                "use_gym_v25_interface": use_gym_v25_interface,
                "use_stop_action": use_stop_action,
                "obs_size": obs_size,  
                "seed": i,
            },
        )
        for i in range(num_environments)
    )

    print("args", args)

    envs = VectorEnv(
        make_env_fn=SeqPointGoalNavEnvCreator.make_rvn_seq_point_goal_nav_env_fn,
        env_fn_args=args,
        workers_ignore_signals=False,
    )
    # print(f"obs   : {envs.observation_spaces[0]}")
    # print(f"action: {envs.action_spaces[0]}")
    # print("envs", envs)

    return envs
