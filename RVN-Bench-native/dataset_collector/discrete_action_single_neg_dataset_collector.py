import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import habitat_sim
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import utils.habitat_agent_utils as HabitatAgentUtils
import utils.geometry_utils as GeometryUtils
import utils.plot_utils as PlotUtils

from dataset_collector.dataset_types import (
    ACTION_MAP,
    ACTION_NAMES_TO_IDX,
    TrajHabitat,
    DiscreteHabitatActions,
)
from dataset_collector.discrete_action_dataset_collector import (
    DiscreteActionDatasetCollector,
)

import utils.habitat_task_utils as HabitatTaskUtils


class DiscreteActionSingleNegDatasetCollector(DiscreteActionDatasetCollector):
    """
    Collects dataset for discrete action space [move_forward, turn_left, turn_right]
    """

    _DATA_COLLECTOR_NAME = "habitat_single_neg_discrete_action"

    def __init__(
        self,
        min_num_pose_before_collision=1,
        max_num_pose_before_collision=8,
        max_num_pose_after_collision=7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._min_num_pose_before_collision = min_num_pose_before_collision
        self._max_num_pose_before_collision = max_num_pose_before_collision
        self._max_num_pose_after_collision = max_num_pose_after_collision

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

        Collection of negative datasaet.
        1. Set padding to 0.
        2. Sample trajectory.
        3. Interpolate and discretized trajectory.
        4. Check if the trajectory is trackable.
        5. If trackable, skip.
        6. Get the index of pose that leads to collision.
        7. Crop the trajectory and save it.
        8. Image after the collision is saved as same as the first image after the collision.

        Returns:
        --------
        Success : bool
        """

        assert self._recompute_navmesh_with_padding(self._robot_radius)

        self._debug_trials += 1

        disc_traj_positions, disc_traj_yaws, disc_actions = self._discretize_trajectory(
            trajectory
        )
        # Check if the start and goal position are reachable
        if not self._sim.pathfinder.is_navigable(trajectory["positions"][0]):
            # print("start position is not navigable")
            return False
        if not self._sim.pathfinder.is_navigable(trajectory["positions"][-1]):
            # print("goal position is not navigable")
            return False

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

        # check if start pos is really valid
        habitat_sensor_observations = self._sim.step("turn_left")
        habitat_sensor_observations = self._sim.step("turn_right")

        final_agent_position = self._sim.agents[0].get_state().position
        start_pos_diff_after_turning = np.linalg.norm(
            final_agent_position - self._latest_agent_position
        )
        if start_pos_diff_after_turning > self._COLL_MOVE_THRESHOLD:
            print(
                f"Start position is actually not navigable {start_pos_diff_after_turning} {self._latest_agent_position}->{final_agent_position}"
            )
            return False
        # print(f"{map_name}_{idx_traj}_{self._debug_trials}_trajectory_actions")

        # while num_action_taken < self._NUM_MAX_ACTIONS_PER_TRAJ:
        for action in disc_actions:
            collision, goal_reached = self._move_agent_by_action_and_get_obs(
                trajectory, action
            )

            num_action_taken += 1

            if collision:
                if (
                    len(self._trajectory_by_actions["positions"])
                    < self._min_num_pose_before_collision
                ):
                    return False

                goal_reached = False
                sensor_observation_on_collision = self._sim.get_sensor_observations()

                self._trajectory_by_actions["positions"].append(
                    disc_traj_positions[num_action_taken]
                )
                self._trajectory_by_actions["yaws"].append(
                    disc_traj_yaws[num_action_taken]
                )
                self._depth_obs.append(sensor_observation_on_collision["depth_sensor"])
                self._rgb_obs.append(sensor_observation_on_collision["color_sensor"])
                break

            if goal_reached:
                break

        if not collision:
            return False

        # print(f"{map_name}_{idx_traj}_{self._debug_trials}")
        # print("\tNum action taken: ", num_action_taken)
        # print("\tlen traj by action positions: ", len(self._trajectory_by_actions["positions"]))
        if (
            len(self._trajectory_by_actions["positions"])
            > self._max_num_pose_before_collision + 1
        ):
            # Crop the trajectory
            self._trajectory_by_actions["positions"] = self._trajectory_by_actions[
                "positions"
            ][-(self._max_num_pose_before_collision + 1) :]
            # self._trajectory_by_actions["rotations"] = self._trajectory_by_actions["rotations"][-(self._max_num_pose_before_collision+1):]
            self._trajectory_by_actions["yaws"] = self._trajectory_by_actions["yaws"][
                -(self._max_num_pose_before_collision + 1) :
            ]
            self._rgb_obs = self._rgb_obs[-(self._max_num_pose_before_collision + 1) :]
            self._depth_obs = self._depth_obs[
                -(self._max_num_pose_before_collision + 1) :
            ]

        # print("\tlen traj by action positions cropped: ", len(self._trajectory_by_actions["positions"]))
        for i in range(self._max_num_pose_after_collision):
            target_pos_in_traj_idx = num_action_taken + 1 + i
            if len(disc_traj_positions) < target_pos_in_traj_idx + 1:
                break

            self._trajectory_by_actions["positions"].append(
                disc_traj_positions[target_pos_in_traj_idx]
            )
            self._trajectory_by_actions["yaws"].append(
                disc_traj_yaws[target_pos_in_traj_idx]
            )
            self._rgb_obs.append(sensor_observation_on_collision["color_sensor"])
            self._depth_obs.append(sensor_observation_on_collision["depth_sensor"])
            self._discrete_actions_taken["actions"].append(
                ACTION_NAMES_TO_IDX[disc_actions[target_pos_in_traj_idx - 1]]
            )

        # print("\tlen traj by action positions added: ", len(self._trajectory_by_actions["positions"]))

        # print(f"disc_traj_positions: {disc_traj_positions}")
        # for i in range(min(20, len(disc_traj_positions))):
        #     print(
        #     f"disc_traj_positions[{i}]: {disc_traj_positions[i]} \t disc_traj_yaws[{i}]: {disc_traj_yaws[i]}"
        # )
        # for pos, yaw in zip(self._trajectory_by_actions["positions"], self._trajectory_by_actions["yaws"]):
        #     print(f"pos: {pos} \tyaw: {yaw}")

        # if idx_traj ==1 and self._debug_trials == 2:
        #     exit()

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

    def _sample_trajectory(self) -> Tuple[List[np.ndarray], TrajHabitat]:
        """
        Set padding to 0.0 for negative dataset collection.

        Returns:
        --------
        sampled_path_points : List[np.ndarray]
            sampled path points.
        interpolated_trajectory : TrajHabitat
            interpolated trajectory.
        """
        assert self._recompute_navmesh_with_padding(0.0)

        sampled_path_points = None
        interpolated_trajectory = None

        sampled_path_points, sampled_geodesic_distances = (
            HabitatTaskUtils.sample_path_points(
                sim=self._sim,
                num_trials=self._num_trials_sample_per_traj,
                min_path_length=self._min_dist_between_waypoints,
                max_path_length=self._max_dist_between_waypoints,
                allow_stairs=self._allow_stairs,
                path_finder_seed=None,
                try_again_on_failure=False,
            )
        )
        if sampled_path_points is None:
            return sampled_path_points, interpolated_trajectory

        if self._set_start_rot_toward_goal:
            start_yaw = GeometryUtils.get_yaw_toward_pos(
                sampled_path_points[0], sampled_path_points[1]
            )
        else:
            start_yaw = np.random.uniform(-np.pi, np.pi)

        interpolated_trajectory = self._interpolate_path_points(
            sampled_path_points,
            start_yaw,
        )
        return sampled_path_points, interpolated_trajectory

    def _discretize_trajectory(
        self,
        trajectory: TrajHabitat,
        max_num_points_on_traj: Optional[int] = None,
        include_start_pose: bool = True,
    ) -> Tuple[List[np.ndarray], List[float], List[str]]:
        """
        Discretize the trajectory by applying the local planner to the trajectory.

        Returns:
        --------
        discretized_trajectory_positions : list[np.ndarray]
            Discretized trajectory positions in the habitat frame.
        discretized_trajectory_yaws : List[float]
            Discretized yaws in radian.
        discretized_trajectory_actions : DiscreteHabitatActions
            Discretized actions.
        """

        latest_pos = trajectory["positions"][0]
        latest_yaw = trajectory["yaws"][0]

        disc_traj_positions = []
        disc_traj_yaws = []

        if include_start_pose:
            disc_traj_positions.append(latest_pos)
            disc_traj_yaws.append(latest_yaw)

        disc_actions = []

        goal_reached = False

        while not goal_reached:
            action = self._pure_pursuit_local_planner.get_discrete_action_from_yaw(
                latest_pos, latest_yaw, trajectory
            )
            latest_pos, latest_yaw = HabitatAgentUtils.predict_next_pose_by_action(
                latest_pos, latest_yaw, action, self._move_amount, self._rotate_amount
            )
            # print(
            #     f"action: {action} \tlatest_yaw: {latest_yaw:.3f} \tlatest_pos: {latest_pos}"
            # )

            disc_actions.append(action)
            disc_traj_positions.append(latest_pos)
            disc_traj_yaws.append(latest_yaw)

            # check goal reached
            dist_to_goal = np.linalg.norm(latest_pos - trajectory["positions"][-1])
            if dist_to_goal < self._dist_to_goal_threshold:
                goal_reached = True

            if (
                max_num_points_on_traj is not None
                and len(disc_traj_positions) >= max_num_points_on_traj
            ):
                break

        return disc_traj_positions, disc_traj_yaws, disc_actions

    def _is_trackable(
        self,
        disc_actions: list[str],
        start_pos: np.ndarray,  # in habitat frame
        start_rot: float,
        goal_pos: np.ndarray,  # in habitat frame
        check_goal_reach: bool = False,
    ):
        """
        Check if the trajectory is trackable.
        Start from the start_pos and start_yaw and apply the actions in disc_actions.
        If the trajectory is trackable, return True. Otherwise, return False.
        """

        self._recompute_navmesh_with_padding(self._robot_radius)

        self._set_agent_state(start_pos, start_rot)

        latest_agent_position = self._sim.agents[0].get_state().position

        for action in disc_actions:
            # move agent by pre-defined action
            self._sim.step(action)

            # get agent state and obs
            final_agent_position = self._sim.agents[0].get_state().position

            collision = HabitatAgentUtils.check_collision(
                action,
                latest_agent_position,
                final_agent_position,
                self._move_amount,
                self._COLL_MOVE_THRESHOLD,
                allow_stairs=self._allow_stairs,
            )
            if collision:
                return False

            latest_agent_position = final_agent_position

        # make sure goal is reached
        if check_goal_reach:
            dist_to_goal = np.linalg.norm(final_agent_position - goal_pos)
            if dist_to_goal < self._GOAL_REACH_THRESHOLD:
                return False

        return True
