import sys
import os
from typing import Tuple, Optional, Union
from datetime import datetime

import yaml
import numpy as np
import habitat_sim
import quaternion as qt
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import utils.config_utils as ConfigUtils
import utils.geometry_utils as GeometryUtils
import utils.habitat_agent_utils as HabitatAgentUtils
import utils.habitat_task_utils as HabitatTaskUtils

from agents.agent import ObsType, InfoType

import gym


class SeqPointGoalNavEnv(gym.Env):
    _NUM_MAX_PATH_SAMPLING_ATTEMPTS = 1000
    _NUM_MAX_GOAL_SAMPLING_ATTEMPTS = 100000
    _ACTION_SPACE = ["move_forward", "turn_left", "turn_right", "stop"]
    _SET_WAYPOINT_ROT_FROM_PREV_WAYPOINT = True

    VERSION = "v0.4.1-dev"

    def __init__(
        self,
        config_file_path: str = "configs/rl_envs/seq_point_goal_nav_env_miniminival.yaml",
        render_mode: str = "rgb_array",
        scene_selection_mode: str = "sequential",  # ["random", "sequential"]
        num_scene_repeats: int = 32,
        use_gym_v25_interface: bool = False,
        use_stop_action: bool = True,
        obs_size: int = 1,
    ):
        print(
            f"Initializing SeqPointGoalNavEnv {self.VERSION}... config: {config_file_path}, num_scene_repeats: {num_scene_repeats}"
        )
        assert (
            render_mode == "rgb_array"
        ), "Invalid render mode. Choose from ['rgb_array'] ('human' not supported)."

        self._load_env_config(config_file_path)
        self._waypoint_position: Optional[np.ndarray] = None
        self._waypoint_rotation: Optional[qt.quaternion] = None
        self._waypoint_yaw: Optional[float] = None
        self._waypoint_rgba: Optional[np.ndarray] = None

        self._scene_selection_mode = scene_selection_mode
        self._num_scene_repeats = num_scene_repeats
        self._use_gym_v25_interface = use_gym_v25_interface
        self._use_stop_action = use_stop_action
        self._obs_size = obs_size

        self._num_waypoint_reached: int = 0

        self._last_loaded_scene_id: Optional[str] = None
        self._last_loaded_scene_idx = -1
        self._num_scene_repeated = 0

        if obs_size == 1:
            rgb_shape = (self._sim_settings["height"],
                        self._sim_settings["width"],
                        3)
        else:
            rgb_shape = (
                self._obs_size,
                self._sim_settings["height"],
                self._sim_settings["width"],
                3,
            )

        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=rgb_shape,
                    dtype=np.uint8,
                ),
                # "goal_pose": gym.spaces.Box(
                #     low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                # ),
                "pointgoal_with_gps_compass": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(len(self._ACTION_SPACE))
        return

    def seed(self, seed: Optional[int] = None):
        """
        Parameters
        ----------
        seed: Optional[int]
            seed for the environment.
        """
        print(f"Setting seed to {seed}")
        if seed is not None:
            np.random.seed(seed)
        return

    def reset(self, seed: Optional[int] = None) -> Tuple[ObsType, InfoType]:
        """
        Parameters
        ----------
        seed: Optional[int]
            seed for the environment.


        Returns
        -------
        observation, info
        """

        self._num_waypoint_reached = 0
        self._num_step_ep = 0
        self._num_step_wp = 0
        self._last_rgb_obs = None
        self._travel_dist = 0.0
        self._obs_history = []

        if seed is not None:
            self.seed(seed)

        # load scene
        num_trials = 0
        while num_trials < self._NUM_MAX_PATH_SAMPLING_ATTEMPTS:
            scene_id, scene_dataset_config_file = self._select_scene()
            self._load_sim(self._scene_dir, scene_id, scene_dataset_config_file)

            self._set_navmesh_to_sample_mode()
            # sample valid start and goal positions
            start_pos, start_rot, waypoint_pos, waypoint_rot = (
                HabitatTaskUtils.sample_start_and_goal_pose(
                    self._sim,
                    num_trials=self._NUM_MAX_GOAL_SAMPLING_ATTEMPTS,
                    min_path_length=self._min_dist_between_waypoints,
                    max_path_length=self._max_dist_between_waypoints,
                    allow_stairs=self._allow_stairs,
                    randomize_start_rotation=self._randomize_start_rotation,
                    randomize_goal_rotation=self._randomize_waypoint_rotation,
                    try_again_on_failure=False,
                    path_finder_seed=seed,
                )
            )
            if start_pos is not None and waypoint_pos is not None:
                break
            num_trials += 1

        if start_pos is None or waypoint_pos is None:
            raise ValueError("Could not sample valid start and goal positions.")

        self._set_navmesh_to_eval_mode()

        # set agent to start position
        agent = self._sim.initialize_agent(self._sim_settings["default_agent"])
        start_agent_state = habitat_sim.AgentState()
        start_agent_state.position = start_pos
        start_agent_state.rotation = start_rot
        agent.set_state(start_agent_state)

        # get agent state and obs
        agent_position = agent.get_state().position
        agent_yaw_angle = GeometryUtils.get_yaw_from_habitat_qt(
            agent.get_state().rotation
        )
        habitat_sensor_observations = self._sim.get_sensor_observations()

        for _ in range(self._obs_size):
            self._obs_history.append(habitat_sensor_observations)

        # get goal rgba and goal yaw
        waypoint_rgba, waypoint_yaw = HabitatAgentUtils.get_obs_and_yaw_from_pose(
            self._sim, agent, waypoint_pos, waypoint_rot
        )

        obs, info = HabitatAgentUtils.get_obs_info(
            habitat_sensor_observation_history=self._obs_history,
            goal_position=waypoint_pos,
            goal_yaw=waypoint_yaw,
            robot_position=agent_position,
            robot_yaw=agent_yaw_angle,
            goal_rgba=waypoint_rgba,
            num_wp_reached=self._num_waypoint_reached,
            collision=False,
            timeout=False,
            travel_distance=self._travel_dist,
            obs_size=self._obs_size,
        )

        self._last_rgb_obs = obs["rgb"]
        self._last_rgb_goal = info["goal_rgb"]

        # set waypoint position and rotation
        self._waypoint_position = waypoint_pos
        self._waypoint_rotation = waypoint_rot
        self._waypoint_yaw = waypoint_yaw
        self._waypoint_rgba = waypoint_rgba

        if self._use_gym_v25_interface:
            return obs

        return obs, info

    def step(
        self, action: Union[str, int]
    ) -> Tuple[ObsType, float, bool, bool, InfoType]:
        """
        Parameters
        ----------
        action: str | int
            One of ["move_forward", "turn_left", "turn_right", "stop] or index of the action in the action space.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        self._num_step_ep += 1
        self._num_step_wp += 1

        # validate action
        if not isinstance(action, str):
            action = self._ACTION_SPACE[action]

        if action not in self._ACTION_SPACE:
            raise ValueError(f"Invalid action: {action}")

        # save initial state
        initial_agent_position = self._sim.agents[0].get_state().position

        # take action
        habitat_sensor_observations = self._sim.step(action)

        # add obs to history
        if len(self._obs_history) == self._obs_size:
            self._obs_history.pop(0)
        self._obs_history.append(habitat_sensor_observations)

        # get agent state and obs
        final_agent_position = self._sim.agents[0].get_state().position

        self._travel_dist += np.linalg.norm(
            final_agent_position - initial_agent_position
        )

        collision, goal_reached, false_positive = (
            self._check_collision_and_goal_reached(
                action, initial_agent_position, final_agent_position
            )
        )
        reward = self._calculate_reward(
            collision,
            goal_reached,
            false_positive,
            initial_agent_position,
            final_agent_position,
        )

        next_waypoint_unoptainable_flag = False

        if not collision and goal_reached:
            self._num_step_wp = 0
            self._num_waypoint_reached += 1
            next_waypoint_unoptainable_flag = not self._set_next_waypoint()

        terminated, truncated = self._check_terminated_and_truncated(
            collision, false_positive
        )
        if next_waypoint_unoptainable_flag:
            print("could not sample next waypoint, truncated.")
            truncated = True

        # get obs and info
        obs, info = HabitatAgentUtils.get_obs_info(
            habitat_sensor_observation_history=self._obs_history,
            goal_position=self._waypoint_position,
            goal_yaw=self._waypoint_yaw,
            robot_position=final_agent_position,
            robot_yaw=GeometryUtils.get_yaw_from_habitat_qt(
                self._sim.agents[0].get_state().rotation
            ),
            goal_rgba=self._waypoint_rgba,
            num_wp_reached=self._num_waypoint_reached,
            collision=collision,
            timeout=truncated,
            travel_distance=self._travel_dist,
            obs_size=self._obs_size,
        )

        self._last_rgb_obs = obs["rgb"]
        self._last_rgb_goal = info["goal_rgb"]

        if self._use_gym_v25_interface:
            done = terminated or truncated
            return obs, reward, done, info

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        # Add two image (self._last_rgb_obs, self._last_rgb_goal) side by side
        # and return the image
        combined_image = np.concatenate(
            (self._last_rgb_obs, self._last_rgb_goal), axis=1
        )
        return combined_image

    def _calculate_reward(
        self,
        collision,
        goal_reached,
        false_positive,
        initial_agent_position,
        final_agent_position,
    ):
        dist_to_goal_final = np.linalg.norm(
            self._waypoint_position - final_agent_position
        )
        dist_to_goal_initial = np.linalg.norm(
            self._waypoint_position - initial_agent_position
        )
        dist_to_goal_diff = dist_to_goal_initial - dist_to_goal_final

        reward = 0.0
        if collision:
            reward = -0.1
        elif false_positive:
            reward = -0.1
        elif goal_reached:
            reward = 1.0
        else:
            reward += -0.01
            reward += dist_to_goal_diff * 0.1
        return reward

    def _set_next_waypoint(self):
        """
        Sample a new waypoint and set it as the current waypoint.
        """
        next_waypoint_position, next_waypoint_rotation = self._sample_next_waypoint()
        if next_waypoint_position is None:
            return False

        self._set_navmesh_to_sample_mode()

        # get goal rgba and goal yaw
        next_waypoint_rgba, next_waypoint_yaw = (
            HabitatAgentUtils.get_obs_and_yaw_from_pose(
                self._sim,
                self._sim.agents[0],
                next_waypoint_position,
                next_waypoint_rotation,
            )
        )

        self._set_navmesh_to_eval_mode()

        self._waypoint_position = next_waypoint_position
        self._waypoint_rotation = next_waypoint_rotation
        self._waypoint_yaw = next_waypoint_yaw
        self._waypoint_rgba = next_waypoint_rgba

    def _sample_next_waypoint(self) -> Tuple[np.ndarray, qt.quaternion]:
        """
        Sample a new waypoint and set it as the current waypoint.
        Distance between the current waypoint and the new waypoint should be between
        self._min_dist_between_waypoints and self._max_dist_between_waypoints.

        Returns
        -------
        sampled_waypoint_position: np.ndarray
            The sampled waypoint position in the Habitat frame.
        sampled_waypoint_rotation: qt.quaternion
            The sampled waypoint rotation.
        """
        for _ in range(self._NUM_MAX_GOAL_SAMPLING_ATTEMPTS):
            sampled_next_waypoint_position = (
                self._sim.pathfinder.get_random_navigable_point()
            )

            if not self._allow_stairs and (
                sampled_next_waypoint_position[1] != self._waypoint_position[1]
            ):
                continue
            if (
                np.linalg.norm(sampled_next_waypoint_position - self._waypoint_position)
                > self._max_dist_between_waypoints
            ):
                continue

            sampled_path = habitat_sim.ShortestPath()
            sampled_path.requested_start = self._waypoint_position
            sampled_path.requested_end = sampled_next_waypoint_position

            path_found = self._sim.pathfinder.find_path(sampled_path)
            geodesic_distance = sampled_path.geodesic_distance

            if not path_found or not HabitatTaskUtils.validate_path(
                self._waypoint_position,
                sampled_path.points,
                geodesic_distance,
                self._min_dist_between_waypoints,
                self._max_dist_between_waypoints,
                self._allow_stairs,
            ):
                continue

            if self._randomize_waypoint_rotation:
                random_waypoint_yaw = np.random.uniform(0, 2 * np.pi)
                sampled_next_waypoint_rotation = GeometryUtils.get_quaternion_form_yaw(
                    random_waypoint_yaw
                )
            else:
                sampled_next_waypoint_rotation = GeometryUtils.get_rotation_toward_pos(
                    self._waypoint_position, sampled_next_waypoint_position
                )
            return sampled_next_waypoint_position, sampled_next_waypoint_rotation

        print("Could not sample valid waypoint.")
        return None, None

    def _check_terminated_and_truncated(
        self, collision, false_positive
    ) -> Tuple[bool, bool]:
        terminated = False
        truncated = False

        if collision or false_positive:
            terminated = True
        elif self._num_waypoint_reached >= self._max_waypoints_per_episode:
            terminated = True
        elif self._num_step_wp >= self._max_steps_per_waypoint:
            truncated = True

        return terminated, truncated

    def _check_collision_and_goal_reached(
        self,
        action: str,
        initial_agent_position: np.ndarray,
        final_agent_position: np.ndarray,
    ) -> Tuple[bool, bool, bool]:
        """
        Parameters
        ----------
        action: str
            The action taken by the agent.
        initial_agent_position: np.ndarray
            The initial position of the agent in the Habitat frame.
        final_agent_position: np.ndarray
            The final position of the agent in the Habitat frame.

        Returns
        -------
        collision: bool
        goal_reached: bool
        false_positive: bool
            Whether the goal is not reached when the action is "stop".
        """
        goal_reached = False
        false_positive = False
        collision = HabitatAgentUtils.check_collision(
            action,
            initial_agent_position,
            final_agent_position,
            self._move_amount,
            self._coll_move_diff_threshold,
            allow_stairs=self._allow_stairs,
        )

        if collision:
            return collision, goal_reached, false_positive

        if self._use_stop_action:
            if action == "stop":
                dist_to_goal = np.linalg.norm(
                    self._waypoint_position - final_agent_position
                )
                if dist_to_goal < self._dist_to_goal_threshold:
                    goal_reached = True
                else:
                    false_positive = True
        else:
            dist_to_goal = np.linalg.norm(
                self._waypoint_position - final_agent_position
            )
            if dist_to_goal < self._dist_to_goal_threshold:
                goal_reached = True
        return collision, goal_reached, false_positive

    def close(self):
        print("Closing the environment.")
        try:
            self._sim.close()
            print("\tSimulator closed.")
        except:
            pass
        print("Environment closed.")

    def sample_action(self) -> str:
        return np.random.choice(self._ACTION_SPACE)

    def _load_env_config(self, config_file_path):
        with open(config_file_path, "r") as f:
            self._env_config = yaml.safe_load(f)

        self._scene_dir = self._env_config["scene_dir"]
        self._scene_ids = self._env_config["scene_ids"]
        self._scene_dataset_config_files = self._env_config[
            "scene_dataset_config_files"
        ]

        self._allow_stairs = self._env_config["allow_stairs"]
        self._randomize_start_rotation = self._env_config["randomize_start_rotation"]
        self._randomize_waypoint_rotation = self._env_config[
            "randomize_waypoint_rotation"
        ]

        self._max_waypoints_per_episode = self._env_config["max_waypoints_per_episode"]
        self._max_steps_per_waypoint = self._env_config["max_steps_per_waypoint"]

        self._min_dist_between_waypoints = self._env_config[
            "min_dist_between_waypoints"
        ]
        self._max_dist_between_waypoints = self._env_config[
            "max_dist_between_waypoints"
        ]
        self._move_amount = self._env_config["move_amount"]
        self._dist_to_goal_threshold = self._env_config["dist_to_goal_threshold"]
        self._coll_move_diff_threshold = self._env_config["coll_move_diff_threshold"]

        self._margin_to_obst = self._env_config["margin_to_obst"]
        self._robot_radius = self._env_config["robot_radius"]
        self._robot_height = self._env_config["robot_height"]

        self._sim_settings = self._env_config["sim_settings"]

    def _select_scene(self) -> Tuple[str, str]:
        if (
            self._last_loaded_scene_idx != -1
            and self._num_scene_repeated < self._num_scene_repeats
        ):
            # repeat the last scene
            self._num_scene_repeated += 1
            return (
                self._scene_ids[self._last_loaded_scene_idx],
                self._scene_dataset_config_files[self._last_loaded_scene_idx],
            )

        # select new scene
        self._num_scene_repeated = 1

        if self._scene_selection_mode == "random":
            scene_idx = np.random.randint(0, len(self._scene_ids))
        elif self._scene_selection_mode == "sequential":
            scene_idx = (self._last_loaded_scene_idx + 1) % len(self._scene_ids)

        scene_id = self._scene_ids[scene_idx]
        scene_dataset_config_file = self._scene_dataset_config_files[scene_idx]

        self._last_loaded_scene_idx = scene_idx

        return scene_id, scene_dataset_config_file

    def _load_sim(
        self,
        scene_dir: str,
        scene_id: str,
        scene_dataset_config_file: str,
    ):
        if self._last_loaded_scene_id == scene_id:
            return
        print(f"Loading scene: {scene_id}")
        sim_settings = {
            "width": self._sim_settings["width"],
            "height": self._sim_settings["height"],
            "scene_id": os.path.join(scene_dir, scene_id),
            "scene_dataset_config_file": os.path.join(
                scene_dir, scene_dataset_config_file
            ),
            "default_agent": self._sim_settings["default_agent"],
            "sensor_height": self._sim_settings["sensor_height"],
            "color_sensor": True,
            "semantic_sensor": True,
            "depth_sensor": True,
            "seed": 1,
            "enable_physics": False,
        }
        cfg = ConfigUtils.make_cfg(sim_settings)
        try:
            self._sim.close()
        except:
            pass
        with HabitatTaskUtils.suppress_cpp_output():
            self._sim = habitat_sim.Simulator(cfg)
        self._last_loaded_scene_id = scene_id

    def _set_navmesh_to_sample_mode(self):
        navmesh_settings = self._sim.pathfinder.nav_mesh_settings
        navmesh_settings.agent_radius = self._robot_radius + self._margin_to_obst
        navmesh_settings.agent_height = self._robot_height
        navmesh_success = self._sim.recompute_navmesh(
            self._sim.pathfinder, navmesh_settings
        )
        return navmesh_success

    def _set_navmesh_to_eval_mode(self):
        navmesh_settings = self._sim.pathfinder.nav_mesh_settings
        navmesh_settings.agent_radius = self._robot_radius
        navmesh_settings.agent_height = self._robot_height
        navmesh_success = self._sim.recompute_navmesh(
            self._sim.pathfinder, navmesh_settings
        )
        return navmesh_success


def main():
    n_step = 0
    curr_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_save_dir = f"debug_images/{curr_datetime_str}"
    os.makedirs(img_save_dir, exist_ok=True)

    env = SeqPointGoalNavEnv()
    obs, info = env.reset(seed=24)

    current_rgb_image = Image.fromarray(obs["rgb"], "RGB")
    goal_rgb_image = Image.fromarray(info["goal_rgb"], "RGB")
    current_rgb_image.save(os.path.join(img_save_dir, f"current_{n_step}.png"))
    goal_rgb_image.save(os.path.join(img_save_dir, f"goal_{n_step}.png"))

    goal_w = info["goal_pose_world"]
    robot_w = info["robot_pose"]
    print(f"\t\tgoal_w : [{goal_w[0]:.2f}, {goal_w[1]:.2f}, {goal_w[2]:.2f}]")
    print(f"\t\trobot_w: [{robot_w[0]:.2f}, {robot_w[1]:.2f}, {robot_w[2]:.2f}]")

    action_space = env.action_space
    print("action space: ", action_space)
    print("action space size: ", action_space.n)

    for _ in range(100):
        n_step += 1
        # if n_step == 2:
        #     action = "move_forward"
        #     # action = "turn_left"
        # else:
        #     action = "move_forward"
        action = env.action_space.sample()
        while action == 3:
            action = env.action_space.sample()
        print(f"{n_step} - taking action {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\treward: {reward} | terminated: {terminated} | truncated: {truncated}")

        current_rgb_image = Image.fromarray(obs["rgb"], "RGB")
        goal_rgb_image = Image.fromarray(info["goal_rgb"], "RGB")
        current_rgb_image.save(os.path.join(img_save_dir, f"current_{n_step}.png"))
        goal_rgb_image.save(os.path.join(img_save_dir, f"goal_{n_step}.png"))

        goal_w = info["goal_pose_world"]
        robot_w = info["robot_pose"]
        print(f"\t\tgoal_w : [{goal_w[0]:.2f}, {goal_w[1]:.2f}, {goal_w[2]:.2f}]")
        print(f"\t\trobot_w: [{robot_w[0]:.2f}, {robot_w[1]:.2f}, {robot_w[2]:.2f}]")

        if terminated or truncated:
            break

    env.close()
    print("debug images saved at: ", img_save_dir)


if __name__ == "__main__":
    main()
