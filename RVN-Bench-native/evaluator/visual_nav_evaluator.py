import os
from typing import Tuple

import yaml
import numpy as np
import habitat_sim

import utils.config_utils as ConfigUtils


from agents.agent import Agent


class VisualNavEvaluator:
    _COLL_MOVE_DIFF_THRESH = 1e-3

    def __init__(
        self,
        scenario_dir: str = "scenarios",
        scenario_name: str = "scenario_SUHsP6z2gcJ_100",
        data_dir: str = "data",
        rvn_agent: Agent = None,
    ):
        self._local_planner_model = None
        self._eval_scenario = None

        self.set_scenario_and_load_sim(scenario_dir, scenario_name, data_dir)
        self.set_rvn_model(rvn_agent)

        self._move_amount = None
        self._dist_to_goal_threshold = None

    # def set_local_planner_model(self, local_planner_model):
    #     self._local_planner_model = local_planner_model

    def set_scenario_and_load_sim(
        self,
        scenario_dir: str = "scenarios",
        scenario_name: str = "scenario_SUHsP6z2gcJ_100",
        data_dir: str = "data",
    ):
        self.set_scenario(scenario_dir, scenario_name, data_dir)
        self._load_sim()

    def set_rvn_model(self, rvn_agent: Agent):
        self._rvn_agent = rvn_agent

    def set_scenario(
        self,
        scenario_dir: str = "scenarios",
        scenario_name: str = "scenario_SUHsP6z2gcJ_100",
        data_dir: str = "data",
    ):
        self._data_dir = data_dir
        self._scenario_name = scenario_name
        # set scenario
        scenario_path = os.path.join(scenario_dir, f"{scenario_name}.yaml")
        with open(scenario_path, "r") as file:
            self._eval_scenario = yaml.safe_load(file)

    def _load_sim(self):
        self._sim_settings = {
            "width": 256,
            "height": 256,
            "scene_id": os.path.join(self._data_dir, self._eval_scenario["scene_name"]),
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
        self._sim = habitat_sim.Simulator(cfg)

    def _get_step_result(
        self,
        action,
        agent_position,
        prev_agent_position,
        goal_position,
    ) -> Tuple[bool, bool, float]:
        """
        Returns
        -------
        collision: bool
        goal_reached: bool
        distance_moved: float
            The distance moved by the agent in this step.
        """
        collision = False
        goal_reached = False
        distance_moved = float(np.linalg.norm(agent_position - prev_agent_position))
        dist_to_goal = np.linalg.norm(agent_position - goal_position)

        if (
            action == "move_forward"
            and distance_moved < self._move_amount - self._COLL_MOVE_DIFF_THRESH
        ):
            collision = True

        if dist_to_goal < self._dist_to_goal_threshold:
            goal_reached = True

        return collision, goal_reached, distance_moved
