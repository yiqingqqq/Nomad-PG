import os
from datetime import datetime
from typing import Optional

import yaml
import numpy as np
import torch
import quaternion as qt
import habitat_sim
import matplotlib

matplotlib.use("Agg")


from evaluator.visual_nav_evaluator import VisualNavEvaluator
from evaluator.eval_scenario_types import (
    SeqPointGoalNavEvalEpisode,
    SeqPointGoalNavEvalScenario,
)
from agents.agent import Agent
import utils.habitat_agent_utils as HabitatAgentUtils
import utils.habitat_task_utils as HabitatTaskUtils
import utils.metric_utils as MetricUtils
import utils.plot_utils as PlotUtils
import utils.config_utils as ConfigUtils
import utils.geometry_utils as GeometryUtils

SEED = 1


class SeqPointGoalNavEvaluator(VisualNavEvaluator):
    """
    Evaluator for the local planner.

    Controllers, also known as local planners in ROS 1, are the way we follow the
    globally computed path or complete a local task. The controller will have access
    to a local environment representation to attempt to compute feasible control
    efforts for the base to follow (https://docs.nav2.org/concepts/index.html).

    Local planner outputs control commands given the waypoint not the global goal!

    TODO: Separate env to instance.
    """

    def __init__(
        self,
        scenario_dir: str = "scenarios/seq_point_goal_nav_eval_scenarios/",
        scenario_name: str = "miniminival_2_1_20250216_185507",
        data_dir: str = "data",
        rvn_agent: Agent = None,
    ):
        self._rvn_agent: Agent = None
        self._eval_scenario: SeqPointGoalNavEvalScenario = None

        self.set_scenario(scenario_dir, scenario_name, data_dir)
        self.set_rvn_model(rvn_agent)

    def evaluate(
        self,
        target_model: str = "random",  # name of target model
        max_ep_steps: int = int(1e7),
        max_wp_steps: int = 100,
        dist_to_goal_threshold: float = 0.36,
        move_amount: float = 0.25,
        allow_collision: bool = False,
        result_save_dir: str = "logs/eval_results/seq_point_goal_nav/results",
        save_plot: bool = False,
        plot_save_dir: str = "logs/eval_results/seq_point_goal_nav/debug_images",
        seed: int = SEED,
        weight_path: Optional[str] = None,  # path of the loaded weight of model
        obs_size: int = 1,
    ):
        assert (
            self._rvn_agent is not None
        ), "Agent not set. Set agent using set_rvn_model() method."
        assert (
            self._eval_scenario is not None
        ), "Scenario not set. Set scenario using set_scenario() method."

        self._set_seed(seed)

        eval_start_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(result_save_dir, exist_ok=True)

        self._target_model = target_model
        self._weight_path = weight_path
        self._obs_size = obs_size

        self._save_plot = save_plot
        if self._save_plot:
            self._plot_save_dir = os.path.join(plot_save_dir, eval_start_datetime_str)
            os.makedirs(self._plot_save_dir, exist_ok=True)

        self._max_ep_steps = max_ep_steps
        self._max_wp_steps = max_wp_steps

        self._dist_to_goal_threshold = dist_to_goal_threshold
        self._move_amount = move_amount
        self._allow_collision = allow_collision

        self._init_eval_results(result_save_dir, eval_start_datetime_str)

        self._loaded_scene_id: str = None
        for ep_idx, episode in enumerate(self._eval_scenario["episodes"]):
            self._eval_episode(ep_idx, episode)

        self._calcualte_metrics()
        self._save_eval_results()

    def _set_seed(self, seed):
        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _init_eval_results(
        self,
        result_save_dir,
        eval_start_datetime_str,
    ):
        self._eval_results = {}
        self._eval_results["scenario"] = self._scenario_name
        self._eval_results["model"] = self._target_model
        self._eval_results["weight_path"] = self._weight_path
        self._eval_results["episode_results"] = []
        self._eval_results["max_ep_steps"] = self._max_ep_steps
        self._eval_results["max_wp_steps"] = self._max_wp_steps
        self._eval_results["dist_to_goal_threshold"] = self._dist_to_goal_threshold
        self._eval_results["move_amount"] = self._move_amount
        self._eval_results["allow_collision"] = self._allow_collision
        self._eval_results["num_waypoints_per_ep"] = self._eval_scenario[
            "num_waypoints_per_ep"
        ]

        self._total_steps = 0
        self._total_distance_traveled = 0

        self._num_successes = 0
        self._num_collisions = 0
        self._num_ep_evaluated = 0
        self._num_ep_timeout = 0
        self._num_wp_timeout = 0

        self._result_file_path = os.path.join(
            result_save_dir,
            f"results_{eval_start_datetime_str}_{self._scenario_name}_{self._target_model}.yaml",
        )

    def _eval_episode(self, ep_idx: int, episode: SeqPointGoalNavEvalEpisode):
        if self._loaded_scene_id != episode["scene_id"]:
            self._load_sim_by_scene_id(episode["scene_id"])
        time_ep_start = datetime.now()
        self._rvn_agent.reset()

        start_position = np.array(episode["start_position"])
        start_rotation = qt.quaternion(*episode["start_rotation"])

        distance_traveled_in_ep = 0
        ep_steps = 0
        num_wp_reached = 0
        num_ep_collisions = 0
        ep_timeout = False
        wp_timeout = False
        collision_occured = False
        self._obs_history = []

        # init agent in start pose
        agent = self._sim.initialize_agent(self._sim_settings["default_agent"])
        start_agent_state = habitat_sim.AgentState()
        start_agent_state.position = start_position
        start_agent_state.rotation = start_rotation

        agent.set_state(start_agent_state)
        habitat_sensor_observations = self._sim.get_sensor_observations()

        for _ in range(self._obs_size):
            self._obs_history.append(habitat_sensor_observations)

        prev_agent_position = agent.get_state().position

        # iterate through goals in the episode
        for idx_goal, goal_position in enumerate(episode["waypoint_positions"]):
            goal_position = np.array(goal_position)
            goal_rotation = qt.quaternion(*episode["waypoint_rotations"][idx_goal])
            goal_rgba, goal_yaw = HabitatAgentUtils.get_obs_and_yaw_from_pose(
                self._sim, agent, goal_position, goal_rotation
            )

            for wp_step in range(self._max_wp_steps):
                ep_steps += 1
                self._total_steps += 1

                obs, info = HabitatAgentUtils.get_obs_info(
                    habitat_sensor_observation_history=self._obs_history,
                    goal_position=goal_position,
                    goal_yaw=goal_yaw,
                    robot_position=agent.get_state().position,
                    robot_yaw=GeometryUtils.get_yaw_from_habitat_qt(
                        agent.get_state().rotation
                    ),
                    goal_rgba=goal_rgba,
                    num_wp_reached=num_wp_reached,
                    collision=False,
                    timeout=False,
                    travel_distance=0.0,
                    obs_size=self._obs_size,
                )
                # ==================== get action ====================
                action = self._rvn_agent.act(obs, info)

                # ==================== step ====================
                habitat_sensor_observations = self._sim.step(action)

                # add obs to history
                if len(self._obs_history) == self._obs_size:
                    self._obs_history.pop(0)
                self._obs_history.append(habitat_sensor_observations)

                # ==================== calculate metrics ====================
                agent_position = agent.get_state().position
                collision_occured, wp_reached, distance_moved = self._get_step_result(
                    action, agent_position, prev_agent_position, goal_position
                )
                prev_agent_position = agent_position

                distance_traveled_in_ep += distance_moved
                self._total_distance_traveled += distance_moved

                if self._save_plot:
                    PlotUtils.save_debug_images(
                        self._target_model,
                        ep_idx,
                        ep_steps,
                        obs,
                        info,
                        action,
                        self._plot_save_dir,
                        collision_occured,
                        wp_reached,
                        wp_step == self._max_wp_steps - 1,
                    )

                if collision_occured:
                    self._num_collisions += 1
                    num_ep_collisions += 1
                    if not self._allow_collision:
                        wp_reached = False
                        break

                if wp_reached:
                    num_wp_reached += 1
                    break

                if wp_step == self._max_wp_steps - 1:
                    wp_timeout = True
                    self._num_wp_timeout += 1
                    break

                if ep_steps >= self._max_ep_steps:
                    ep_timeout = True
                    self._num_ep_timeout += 1
                    break

            if not wp_reached:
                break

        if wp_reached:
            self._num_successes += 1

        total_geodesic_distance = sum(episode["geodesic_distances"])
        num_wp = len(episode["waypoint_positions"])

        ep_result = {
            "ep_idx": ep_idx,
            "num_steps": ep_steps,
            "distance_traveled": distance_traveled_in_ep,
            "geodesic_distance": total_geodesic_distance,
            "goal_reached": wp_reached,
            "num_wp_reached": num_wp_reached,
            "num_wp": num_wp,
            "ep_timeout": ep_timeout,
            "wp_timeout": wp_timeout,
            "collision": collision_occured,
            "num_collisions": num_ep_collisions,
        }
        self._eval_results["episode_results"].append(ep_result)
        self._num_ep_evaluated += 1
        time_ep_end = datetime.now()

        print(
            f"Episode {ep_idx}\tcompleted in {ep_steps} steps ({(time_ep_end - time_ep_start).microseconds/1e6:.2f}s)",
            f" success: {wp_reached} ({num_wp_reached}/{num_wp}), timeout: (ep:{ep_timeout} wp: {wp_timeout}), collision: {collision_occured}, dist_travel: {distance_traveled_in_ep:.2f}",
        )

    def _load_sim_by_scene_id(self, scene_id: str):
        self._sim_settings = {
            "width": 256,
            "height": 256,
            "scene_id": os.path.join(self._data_dir, scene_id),
            "scene_dataset_config_file": os.path.join(
                self._data_dir, self._eval_scenario["scene_dataset_config_file"]
            ),
            "default_agent": self._eval_scenario["sim_settings"]["default_agent"],
            "sensor_height": self._eval_scenario["sim_settings"]["sensor_height"],
            "color_sensor": True,
            "semantic_sensor": True,
            "depth_sensor": True,
            "seed": 1,
            "enable_physics": False,
        }
        cfg = ConfigUtils.make_cfg(self._sim_settings)
        try:
            self._sim.close()
        except:
            pass

        with HabitatTaskUtils.suppress_cpp_output():
            self._sim = habitat_sim.Simulator(cfg)
        self._loaded_scene_id = scene_id

    def _calcualte_metrics(self):
        print("\nCalculating metrics...")
        print("\tnum_successes:\t", self._num_successes)
        print("\tnum_collisions:\t", self._num_collisions)
        print("\tnum_timeout:\t", self._num_ep_timeout + self._num_wp_timeout)
        print("\t\tnum_ep_timeout:\t", self._num_ep_timeout)
        print("\t\tnum_wp_timeout:\t", self._num_wp_timeout)

        success_rate = self._num_successes / self._num_ep_evaluated
        success_rate_by_num_wp = MetricUtils.get_success_rate_by_num_waypoints(
            self._eval_results["episode_results"]
        )
        average_wp_reached = MetricUtils.get_average_wp_reached(
            self._eval_results["episode_results"]
        )

        spl = MetricUtils.get_success_weighted_by_path_length(
            self._eval_results["episode_results"]
        )
        collisions_per_km_traveled = (
            self._num_collisions / self._total_distance_traveled * 1000
        )

        self._eval_results["success_rate"] = success_rate
        self._eval_results["success_rate_by_num_wp"] = success_rate_by_num_wp
        self._eval_results["average_wp_reached"] = average_wp_reached
        self._eval_results["spl"] = spl
        self._eval_results["num_successes"] = self._num_successes
        self._eval_results["num_collisions"] = self._num_collisions
        self._eval_results["num_timeout"] = self._num_ep_timeout
        self._eval_results["num_ep_evaluated"] = self._num_ep_evaluated
        self._eval_results["total_steps"] = self._total_steps
        self._eval_results["total_distance_traveled"] = self._total_distance_traveled
        self._eval_results["collisions_per_km_traveled"] = collisions_per_km_traveled

    def _save_eval_results(self):
        with open(self._result_file_path, "w") as file:
            yaml.dump(self._eval_results, file)

        print(
            f"\nEvaluation finished.",
            f"\n\tsr by num wp: \t{self._eval_results['success_rate_by_num_wp']}",
            f"\n\tsr: \t{self._eval_results['success_rate']:.2f}, ",
            f"\n\taverage wp reached: \t{self._eval_results['average_wp_reached']:.3f}, ",
            f"\n\tspl: \t{self._eval_results['spl']:.2f}, ",
            f"\n\tcollisions/km: \t{self._eval_results['collisions_per_km_traveled']:.1f}",
            f"\nresult saved to: \t{self._result_file_path}",
        )
