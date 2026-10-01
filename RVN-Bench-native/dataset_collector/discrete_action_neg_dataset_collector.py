import os
import sys
from typing import List, Union, Tuple, Optional
import pickle

import yaml
import numpy as np
import habitat_sim
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
    TrajHabitat,
    DiscreteHabitatActions,
    convert_traj_habitat_to_traj_data_2d,
)
from dataset_collector.pathfinder_dataset_collector import PathfinderDatasetCollector
from dataset_collector.discrete_action_dataset_collector import (
    PurePursuitLocalPlanner,
    DiscreteActionDatasetCollector,
)


class DiscreteActionNegDatasetCollector(DiscreteActionDatasetCollector):
    """
    Collects dataset for discrete action space [move_forward, turn_left, turn_right]
    """

    _DATA_COLLECTOR_NAME = "habitat_neg_discrete_action"

    def __init__(
        self, num_neg_exp_traj_per_pos=6, num_pos_in_neg_expt_traj=8, **kwargs
    ):
        super().__init__(**kwargs)

        self._num_pos_in_neg_expt_traj = num_pos_in_neg_expt_traj

        self._neg_or_expt_data_padding_candidates = [
            0.0,
            self._margin_to_obst,
            self._robot_radius,
            self._robot_radius + 0.5 * self._margin_to_obst,
            self._robot_radius + self._margin_to_obst,  # Expt
        ]
        self._expert_data_padding_candidates = [
            self._robot_radius + 4 * self._margin_to_obst,
            self._robot_radius + 2 * self._margin_to_obst,
            self._robot_radius + 1.5 * self._margin_to_obst,
            self._robot_radius + 1.2 * self._margin_to_obst,
            self._robot_radius + 1.1 * self._margin_to_obst,
            self._robot_radius + self._margin_to_obst,  # Add org as expt if none found
        ]
        self._num_neg_exp_traj_per_pos = num_neg_exp_traj_per_pos

    def _collect_data_by_trajectory(
        self, trajectory: TrajHabitat, map_name: str, idx_traj: int
    ) -> bool:
        """
        Collect data from a trajectory and save it to the dataset_save_dir.

        dataset_save_dir/
        ├──{map_name}_{traj_idx}/
        |   ├── #.jpg : Image captured from main camera (first element of camera_names).
        |   ├── #_{camera_name}_{depth}.tiff : Depth image captured from  camera (in mm).
        |   ├── traj_data.pkl : Pickle file that contains the trajectory
        |   ├── data_config.yaml : Dataset configuration [TrajectoryDataConfigYamlType].
        |   ├── global_occupancy_map.png : Image that contains the global occupancy map.
        |   ├── global_occupancy_map.yaml : Global occupancy map config file.
        |   └── ex_neg/
        |       └── generated_ex_neg_traj.pkl
        └──dataset_collector_config.yaml : Dataset collector configuration.

        discrete_action_data.yaml:
            action_map : Dict[int, str]
                Mapping from discrete action index to action name.
            actions : List[int]
                List of discrete actions.

        generated_ex_neg_traj.pkl
            position
            yaw
            generated_local_path_position_and_yaw
            expert_negative_label
            egocentric_goal_position
            number_of_local_paths
            local_path_len
            subgoal_offset

        Parameters:
        -----------
        trajectory : TrajHabitat
            Trajectory to collect data from.
            Must be interpolated.
        map_name : str
            Name of the map.
        idx_traj : int
            Index of the trajectory.

        Returns:
        --------
        Success : bool
        """

        if not super()._collect_data_by_trajectory(trajectory, map_name, idx_traj):
            return False
        self._collect_negative_expert_trajectory(trajectory, map_name, idx_traj)
        return True

    def _collect_negative_expert_trajectory(
        self,
        trajectory: TrajHabitat,
        map_name: str,
        idx_traj: int,
    ) -> bool:
        """
        Collect expert and negative examples and save to /ex_neg/generated_ex_neg_traj_data.pkl

        0. Set goal_pos as the last position in the trajectory.
        1. For each pos_on_traj in discereized trajectory saved (self._trajectory_by_actions):
            1-1. Set start_pos, start_rot, start_yaw as pos_on_traj
            1-2. Plan negative and expert trajectories.
            1-3. Convert the trajectories to the local frame.
        2. Save the expert and negative examples.


        neg_expt_positions_and_yaw have the shape of B x N x L x 3 where
        - B: number of positions on the trajectory
        - N: number of negative or expert examples per pos
        - L: number of points on the trajectory
        - 3: (x, y, yaw)
        """

        goal_pos = self._trajectory_by_actions["positions"][-1]
        local_neg_expt_along_path: List[List[List[Tuple[float, float, float]]]] = []
        local_goal_positions: List[Tuple[float, float]] = []  # (x, y)
        neg_expt_labels: List[List[int]] = []

        goal_pos_world = GeometryUtils.convert_habitat_pos_to_2d_world_pos(goal_pos)

        for idx_pos in range(len(self._trajectory_by_actions["positions"])):
            start_pos = self._trajectory_by_actions["positions"][idx_pos]
            start_rot = self._trajectory_by_actions["rotations"][idx_pos]
            start_yaw = self._trajectory_by_actions["yaws"][idx_pos]

            neg_expt_positions, neg_expt_yaws, neg_expt_label = (
                self._plan_negative_expert_trajectory(
                    trajectory,
                    start_pos,
                    start_rot,
                    start_yaw,
                    goal_pos,
                    idx_pos,
                    map_name,
                    idx_traj,
                )
            )

            if len(neg_expt_positions) == 0:
                break

            start_pose_world = GeometryUtils.convert_habitat_pos_yaw_to_2d_world_pose(
                start_pos, start_yaw
            )
            local_goal_pos = GeometryUtils.transpose_world_pos_to_local_frame(
                start_pose_world, goal_pos_world
            )
            local_goal_positions.append(local_goal_pos)

            # Convert the trajectories to the local frame
            local_neg_expt_positions_and_yaw = (
                GeometryUtils.convert_trajectories_to_local_frame_pos_yaw(
                    neg_expt_positions, neg_expt_yaws, start_pos, start_yaw
                )
            )
            local_neg_expt_along_path.append(local_neg_expt_positions_and_yaw)
            neg_expt_labels.append(neg_expt_label)

        if len(neg_expt_labels) == 0:
            return False

        self._save_negative_expert_trajectory(
            local_neg_expt_along_path,
            neg_expt_labels,
            local_goal_positions,
            map_name,
            idx_traj,
        )

        return True

    def _plan_negative_expert_trajectory(
        self,
        trajectory: TrajHabitat,
        start_pos: np.ndarray,
        start_rot: np.ndarray,
        start_yaw: float,
        goal_pos: np.ndarray,
        idx_pos: int,
        map_name: str,
        idx_traj: int,
    ) -> Tuple[List[List[np.ndarray]], List[List[float]], List[int]]:
        """
        Collect expert and negative examples

        Given neg_or_expt_data_padding_candidates and expert_data_padding_candidates:
        1. For each padding candidate in neg_or_expt_data_padding_candidates:
            1-1. Recompute navmesh with the padding and get the shortest path from start to goal.
            1-2. Discretize the trajectory.
            1-3. Check if the trajectory is trackable.
            1-4. Save the trajectory as negative example if not trackable, expert otherwise.
        2. For each padding candidate in expert_data_padding_candidates:
            2-1. Recompute navmesh with the padding and get the shortest path from start to goal.
            2-2. Discretize the trajectory.
            2-3. Check if the trajectory is trackable. Continue if not trackable.
            2-4. Save the trajectory as expert example. Break if num_exp_data is collected.
        3. Save the expert and negative examples.

        Returns:
        --------
        neg_expt_positions : List[List[np.ndarray]]
            List of positions in the habitat frame.
        neg_expt_yaws : List[List[float]]
            List of yaws in radian.
        neg_expt_label : List[int]
            List of labels. 0: negative, 1: expert
        """

        neg_expt_positions: List[List[np.ndarray]] = []
        neg_expt_yaws: List[List[float]] = []
        neg_expt_label: List[int] = []  # 0: negative, 1: expert

        # Collect negative or expert trajectories from
        # each padding cand in neg_or_expt_data_padding_candidates
        for padding_cand in self._neg_or_expt_data_padding_candidates:
            assert self._recompute_navmesh_with_padding(padding_cand)
            path_found, path_points = self._get_shortest_path(start_pos, goal_pos)
            assert path_found, "Path not found for the smaller margin!"

            traj_w_padding = self._interpolate_path_points(path_points, start_yaw)
            disc_traj_positions, disc_traj_yaws, disc_actions = (
                self._discretize_trajectory(
                    traj_w_padding,
                    self._num_pos_in_neg_expt_traj,
                    include_start_pose=False,
                )
            )

            if len(disc_traj_positions) < self._num_pos_in_neg_expt_traj:
                return [], [], []

            is_negative = not self._is_trackable(
                disc_actions, start_pos, start_rot, goal_pos
            )

            neg_expt_positions.append(disc_traj_positions)
            neg_expt_yaws.append(disc_traj_yaws)
            neg_expt_label.append(0 if is_negative else 1)

        # Try collecting {num_exp_data} more expert example
        num_expt_w_bigger_padding_collected = 0
        num_expt_w_bigger_padding_to_collect = self._num_neg_exp_traj_per_pos - len(
            self._neg_or_expt_data_padding_candidates
        )

        for expert_padding_cand in self._expert_data_padding_candidates:
            if (
                num_expt_w_bigger_padding_collected
                >= num_expt_w_bigger_padding_to_collect
            ):
                break

            if not self._recompute_navmesh_with_padding(expert_padding_cand):
                continue

            path_found, path_points = self._get_shortest_path(start_pos, goal_pos)
            if not path_found:
                continue

            traj_w_padding = self._interpolate_path_points(path_points, start_yaw)
            disc_traj_positions, disc_traj_yaws, disc_actions = (
                self._discretize_trajectory(
                    traj_w_padding,
                    self._num_pos_in_neg_expt_traj,
                    include_start_pose=False,
                )
            )

            if len(disc_traj_positions) < self._num_pos_in_neg_expt_traj:
                return [], [], []

            if not self._is_trackable(disc_actions, start_pos, start_rot, goal_pos):
                print("!!!!!!!!!!!! Expert example not trackable !!!!!!!!!!!!")
                return [], [], []

            neg_expt_positions.append(disc_traj_positions)
            neg_expt_yaws.append(disc_traj_yaws)
            neg_expt_label.append(1)

            num_expt_w_bigger_padding_collected += 1

        if self._save_debug_path_img:
            # PlotUtils.save_neg_exp_traj_debug_plot(
            #     neg_expt_positions,
            #     neg_expt_label,
            #     trajectory["positions"],
            #     self._debug_path_img_save_dir,
            #     f"neg_exp_{map_name}_{idx_traj}_{self._debug_trials}_{idx_pos}",
            # )
            ex_neg_dir = os.path.join(
                self._dataset_save_dir, f"{map_name}_{idx_traj}", "ex_neg"
            )
            os.makedirs(ex_neg_dir, exist_ok=True)
            PlotUtils.save_neg_exp_traj_debug_plot(
                neg_expt_positions,
                neg_expt_label,
                trajectory["positions"],
                ex_neg_dir,
                f"{idx_pos})visualized_ex_neg_traj_data",
            )
            # self._recompute_navmesh_with_padding(0.0)
            # HabitatPlotUtils.plot_paths_on_2d_map(
            #     self._sim,
            #     neg_expt_positions,
            #     path_colors=[
            #         [255, 0, 0] if label == 0 else [0, 0, 255]
            #         for label in neg_expt_label
            #     ],
            #     meters_per_pixel=0.05,
            #     save_plot=True,
            #     plot_save_path=os.path.join(
            #         self._debug_path_img_save_dir,
            #         f"neg_exp_m_{map_name}_{idx_traj}_{self._debug_trials}_{idx_pos}",
            #     ),
            #     height=trajectory["positions"][0][1],
            # )
            pass

        return neg_expt_positions, neg_expt_yaws, neg_expt_label

    def _save_negative_expert_trajectory(
        self,
        neg_expt_positions: List[List[List[Tuple[float, float, float]]]],
        neg_expt_labels: List[List[int]],
        local_goal_positions: List[Tuple[float, float]],
        map_name: str,
        idx_traj: int,
    ):
        """
        Parameters
        ----------
        neg_expt_positions : List[List[List[Tuple[float, float, float]]]]
            List of positions in the local frame.
            B x N x L x 3 where
            - B: number of positions on the trajectory
            - N: number of negative or expert examples per pos
            - L: number of points on the trajectory
            - 3: (x, y, yaw)
        neg_expt_labels : List[List[int]]
            List of labels. 0: negative, 1: expert
            B x N
        local_goal_positions : List[Tuple[float, float]]
            List of goal positions in the local frame.
            B x 2
        """
        # Plot local paths for debugging

        # for idx_batch, local_positions_and_yaw in enumerate(neg_expt_positions):
        #     colors = [
        #         "red" if label == 0 else "blue" for label in neg_expt_labels[idx_batch]
        #     ]
        #     PlotUtils.save_local_paths_and_goal_debug_plot(
        #         local_positions_and_yaw,
        #         colors,
        #         local_goal_positions[idx_batch],
        #         self._debug_path_img_save_dir,
        #         f"local_paths_and_goal_{map_name}_{idx_traj}_{self._debug_trials}_{idx_batch}",
        #     )

        # create /ex_neg directory
        ex_neg_dir = os.path.join(
            self._dataset_save_dir, f"{map_name}_{idx_traj}", "ex_neg"
        )
        os.makedirs(ex_neg_dir, exist_ok=True)

        generated_local_path_position_and_yaw = np.array(neg_expt_positions)
        B, N, L, _ = generated_local_path_position_and_yaw.shape
        traj_data_2d = convert_traj_habitat_to_traj_data_2d(self._trajectory_by_actions)
        subgoal_offset = len(traj_data_2d["position"]) - B

        ex_neg_data = {
            "position": np.array(traj_data_2d["position"]),
            "yaw": np.array(traj_data_2d["yaw"]),
            "generated_local_path_position_and_yaw": generated_local_path_position_and_yaw,
            "expert_negative_label": np.array(neg_expt_labels),
            "egocentric_goal_position": np.array(local_goal_positions),
            "number_of_local_paths": N,
            "local_path_len": L,
            "subgoal_offset": subgoal_offset,
        }

        # save ex_neg_data to pkl
        with open(
            os.path.join(ex_neg_dir, "generated_ex_neg_traj_data.pkl"), "wb"
        ) as f:
            pickle.dump(ex_neg_data, f)

        return

    def _get_shortest_path(
        self, start_pos: np.ndarray, goal_pos: np.ndarray
    ) -> Tuple[bool, List[np.ndarray]]:
        """
        Parameters
        ----------
        start_pos : np.ndarray
            Start position in habitat frame.
        goal_pos : np.ndarray
            Goal position in habitat frame.

        Returns
        -------
        path_found : bool
            True if path is found.
        path_points : List[np.ndarray]
            List of path points.
        """
        sampled_path = habitat_sim.ShortestPath()
        sampled_path.requested_start = start_pos
        sampled_path.requested_end = goal_pos

        path_found = self._sim.pathfinder.find_path(sampled_path)
        path_points = sampled_path.points
        return path_found, path_points

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
