import os
import sys
from typing import List, Union, Tuple
import pickle

import yaml
import numpy as np
import habitat_sim
import cv2
import matplotlib
import quaternion as qt

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import utils.config_utils as ConfigUtils
import utils.habitat_task_utils as HabitatTaskUtils
import utils.habitat_plot_utils as HabitatPlotUtils
import utils.habitat_agent_utils as HabitatAgentUtils
import utils.geometry_utils as GeometryUtils
import utils.plot_utils as PlotUtils

from dataset_collector.dataset_types import (
    ACTION_MAP,
    ACTION_NAMES_TO_IDX,
    OCCUPANCY_MAP_RESOLUTION,
    TRAJECTORY_DATASET_CONFIG_VERSION,
    TrajectoryDataConfigYamlType,
    TrajHabitat,
    DiscreteHabitatActions,
    convert_traj_habitat_to_traj_data_2d,
)
from dataset_collector.pathfinder_dataset_collector import PathfinderDatasetCollector


class PurePursuitLocalPlanner:
    def __init__(
        self,
        move_amount: float,
        rotate_amount: float,
        lookahead_distance: float = 0.5,
    ):
        self._move_amount = move_amount
        self._rotate_amount = rotate_amount
        self._rotate_amount_rad = np.deg2rad(rotate_amount)
        self._lookahead_distance = lookahead_distance

    def get_discrete_action(
        self,
        agent_position: np.ndarray,
        agent_rotation: Union[qt.quaternion, List, np.ndarray],
        trajectory: TrajHabitat,
    ):
        """
        Get discrete action to reach the lookahead point
        """

        agent_yaw = GeometryUtils.get_yaw_from_habitat_qt(agent_rotation)
        return self.get_discrete_action_from_yaw(agent_position, agent_yaw, trajectory)

    def get_discrete_action_from_yaw(
        self,
        agent_position: np.ndarray,
        agent_yaw: float,
        trajectory: TrajHabitat,
    ):
        """
        Get discrete action to reach the lookahead point
        """

        lookahead_point = self._get_lookahead_point(agent_position, trajectory)
        return self._get_action_to_lookahead_point(
            agent_position, agent_yaw, lookahead_point
        )

    def _get_lookahead_point(
        self,
        agent_position: np.ndarray,
        trajectory: TrajHabitat,
    ):
        """
        Get the lookahead point on the trajectory
        1. From the current agent position, find the closest point on the trajectory.
        2. From the closest point, find the point that is lookahead_distance away.
        """
        closest_point_idx = -1
        closest_point_dist = np.inf

        for idx, pos in enumerate(trajectory["positions"]):
            dist = np.linalg.norm(agent_position - pos)
            if dist < closest_point_dist:
                closest_point_idx = idx
                closest_point_dist = dist

        for idx in range(closest_point_idx, len(trajectory["positions"])):
            dist = np.linalg.norm(agent_position - trajectory["positions"][idx])
            if dist > self._lookahead_distance:
                return trajectory["positions"][idx]
        return trajectory["positions"][-1]

    def _get_action_to_lookahead_point(
        self,
        agent_position: np.ndarray,
        agent_yaw: float,
        lookahead_point: np.ndarray,
    ):
        goal_heading = GeometryUtils.get_yaw_toward_pos(agent_position, lookahead_point)
        delta_yaw = GeometryUtils.get_yaw_diff(agent_yaw, goal_heading)

        if delta_yaw > 0.5 * self._rotate_amount_rad:
            return "turn_left"
        elif delta_yaw < -0.5 * self._rotate_amount_rad:
            return "turn_right"
        return "move_forward"


class DiscreteActionDatasetCollector(PathfinderDatasetCollector):
    """
    Collects dataset for discrete action space [move_forward, turn_left, turn_right]
    """

    _DATA_COLLECTOR_NAME = "habitat_discrete_action"

    _GOAL_REACH_THRESHOLD = 0.36
    _COLL_MOVE_THRESHOLD = 0.001  # difference in move amount to consider collision

    _LOOKAHEAD_DISTANCE = 0.25  # [m] distance to look ahead

    _NUM_MAX_ACTIONS_PER_TRAJ = 1000

    _debug_trials = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pure_pursuit_local_planner = PurePursuitLocalPlanner(
            self._move_amount, self._rotate_amount, self._LOOKAHEAD_DISTANCE
        )

    def _collect_data_by_trajectory(
        self, trajectory: TrajHabitat, map_name: str, idx_traj: int
    ) -> bool:
        """
        Collect data from a trajectory and save it to the dataset_save_dir.
        1. save traj_data.pk
        2. save global_occupancy_map.png
        3. save global_occupancy_map.yaml

        dataset_save_dir/
        └──{map_name}_{traj_idx}/
            ├── #.jpg : Image captured from main camera (first element of camera_names).
            ├── #_{camera_name}_{depth}.tiff : Depth image captured from  camera (in mm).
            ├── traj_data.pkl : Pickle file that contains the trajectory
            ├── discrete_action_data.yaml : Pickle file that contains the discrete action.
            ├── data_config.yaml : Dataset configuration [TrajectoryDataConfigYamlType].
            ├── global_occupancy_map.png : Image that contains the global occupancy map.
            └── global_occupancy_map.yaml : Global occupancy map config file.


        discrete_action_data.yaml:
            action_map : Dict[int, str]
                Mapping from discrete action index to action name.
            actions : List[int]
                List of discrete actions.

        Returns:
        --------
        Success : bool
        """

        assert self._recompute_navmesh_with_padding(self._robot_radius)

        self._debug_trials += 1

        # Check if the trajectory is trackable without collision with discrete actions
        # init agent
        self._set_agent_state(trajectory["positions"][0], trajectory["rotations"][0])

        goal_reached = False
        num_action_taken = 0
        self._rgb_obs = []
        self._depth_obs = []
        self._discrete_actions_taken = DiscreteHabitatActions(
            action_map=ACTION_MAP, actions=[]
        )
        self._trajectory_by_actions = TrajHabitat(positions=[], rotations=[], yaws=[])

        self._init_latest_agent_pose_and_get_obs()

        while num_action_taken < self._NUM_MAX_ACTIONS_PER_TRAJ:
            collision, goal_reached = self._move_agent_and_get_obs(trajectory)

            num_action_taken += 1

            if collision:
                goal_reached = False
                break

            if goal_reached:
                break

        if collision:
            return False

        if not goal_reached:
            return False

        assert len(self._trajectory_by_actions["positions"]) == len(self._rgb_obs)

        # Save obs and trajectory data
        self._save_traj_data(
            map_name,
            idx_traj,
            self._trajectory_by_actions,
            self._discrete_actions_taken,
            self._rgb_obs,
            self._depth_obs,
        )

        if self._save_debug_path_img:
            plot_name = f"{map_name}_{idx_traj}_{self._debug_trials}_trajectory_actions"
            PlotUtils.save_discrete_action_debug_plot(
                trajectory["positions"],
                self._trajectory_by_actions["positions"],
                self._debug_path_img_save_dir,
                plot_name,
            )

        return True

    def _init_latest_agent_pose_and_get_obs(self) -> None:
        self._latest_agent_position = self._sim.agents[0].get_state().position
        self._latest_agent_rotation = self._sim.agents[0].get_state().rotation
        self._latest_agent_yaw = GeometryUtils.get_yaw_from_habitat_qt(
            self._sim.agents[0].get_state().rotation
        )
        self._trajectory_by_actions["positions"].append(self._latest_agent_position)
        self._trajectory_by_actions["rotations"].append(self._latest_agent_rotation)
        self._trajectory_by_actions["yaws"].append(self._latest_agent_yaw)

        habitat_sensor_observations = self._sim.get_sensor_observations()
        self._rgb_obs.append(habitat_sensor_observations["color_sensor"])
        self._depth_obs.append(habitat_sensor_observations["depth_sensor"])

    def _move_agent_and_get_obs(self, trajectory: TrajHabitat) -> bool:
        # get action to by lookahead point
        action = self._pure_pursuit_local_planner.get_discrete_action(
            self._latest_agent_position, self._latest_agent_rotation, trajectory
        )
        return self._move_agent_by_action_and_get_obs(trajectory, action)

    def _move_agent_by_action_and_get_obs(
        self, trajectory: TrajHabitat, action: str
    ) -> Tuple[bool, bool]:
        goal_reached = False
        self._discrete_actions_taken["actions"].append(ACTION_NAMES_TO_IDX[action])

        # take action
        habitat_sensor_observations = self._sim.step(action)

        # get agent state and obs
        final_agent_position = self._sim.agents[0].get_state().position
        final_agent_rotation = self._sim.agents[0].get_state().rotation
        final_agent_yaw = GeometryUtils.get_yaw_from_habitat_qt(final_agent_rotation)

        collision = HabitatAgentUtils.check_collision(
            action,
            self._latest_agent_position,
            final_agent_position,
            self._move_amount,
            self._COLL_MOVE_THRESHOLD,
            allow_stairs=self._allow_stairs,
            check_all_action=True,
        )
        if collision:
            return collision, goal_reached

        self._latest_agent_position = final_agent_position
        self._latest_agent_rotation = final_agent_rotation
        self._latest_agent_yaw = final_agent_yaw

        # get pose, and observations
        self._trajectory_by_actions["positions"].append(final_agent_position)
        self._trajectory_by_actions["rotations"].append(final_agent_rotation)
        self._trajectory_by_actions["yaws"].append(final_agent_yaw)

        self._rgb_obs.append(habitat_sensor_observations["color_sensor"])
        self._depth_obs.append(habitat_sensor_observations["depth_sensor"])

        # if goal reached, break
        dist_to_goal = np.linalg.norm(
            final_agent_position - trajectory["positions"][-1]
        )
        if dist_to_goal < self._GOAL_REACH_THRESHOLD:
            goal_reached = True

        return collision, goal_reached

    def _save_traj_data(
        self,
        map_name: str,
        idx_traj: int,
        trajectory_by_actions: TrajHabitat,
        discrete_actions: DiscreteHabitatActions,
        rgb_obs: List[np.ndarray],
        depth_obs: List[np.ndarray],
    ):
        # Save obs and trajectory data
        traj_data_save_dir = os.path.join(
            self._dataset_save_dir, f"{map_name}_{idx_traj}"
        )
        os.makedirs(traj_data_save_dir, exist_ok=True)

        # save traj_data
        traj_data_2d = convert_traj_habitat_to_traj_data_2d(trajectory_by_actions)
        self._save_traj_data_np_pkl(traj_data_save_dir, traj_data_2d)

        with open(os.path.join(traj_data_save_dir, "traj_data_2d.yaml"), "w") as f:
            yaml.dump({"traj_data_2d": traj_data_2d}, f)  # TODO: add 3D traj data
        with open(
            os.path.join(traj_data_save_dir, "discrete_action_data.yaml"), "w"
        ) as f:
            yaml.dump(discrete_actions, f)

        for idx, (rgb, depth) in enumerate(zip(rgb_obs, depth_obs)):
            # save the rgb image to png
            rgb_save_path = os.path.join(traj_data_save_dir, f"{idx}.jpg")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            cv2.imwrite(rgb_save_path, rgb)

            if self._collect_depth:
                # save the depth image to tiff in mm (int)
                depth_save_path = os.path.join(traj_data_save_dir, f"{idx}_depth.tiff")
                depth_mm = (depth * 1000).astype(np.uint16)
                # Convert meters to millimeters
                cv2.imwrite(depth_save_path, depth_mm)

        self._save_data_config_file(traj_data_save_dir, trajectory_by_actions)

    def _interpolate_path_points(
        self,
        sampled_path_points: List[np.ndarray],
        start_yaw: float,
    ) -> TrajHabitat:
        return HabitatTaskUtils.interpolate_path_points(
            sampled_path_points,
            start_yaw,
            self._move_amount * 0.5,
            self._rotate_amount_rad,
        )
