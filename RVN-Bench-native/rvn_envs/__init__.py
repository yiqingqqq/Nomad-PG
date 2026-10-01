from rvn_envs.seq_point_goal_nav_env import SeqPointGoalNavEnv
from gymnasium.envs.registration import register

register(
    id="rvn/SeqPointGoalNav-v0.4",
    entry_point="rvn_envs.seq_point_goal_nav_env:SeqPointGoalNavEnv",
)
