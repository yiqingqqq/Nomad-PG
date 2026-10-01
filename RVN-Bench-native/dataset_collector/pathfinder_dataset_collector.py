import os
import sys
import pickle
from typing import List, Tuple, Optional
from datetime import datetime

import yaml
import numpy as np
import habitat_sim
import cv2
import matplotlib

matplotlib.use("Agg")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset_collector.dataset_types import (
    TrajectoryDataConfigYamlType,
    TrajHabitat,
    TrajData2D,
    convert_traj_habitat_to_traj_data_2d,
    OCCUPANCY_MAP_RESOLUTION,
    TRAJECTORY_DATASET_CONFIG_VERSION,
)
import utils.config_utils as ConfigUtils
import utils.habitat_task_utils as HabitatTaskUtils
import utils.habitat_plot_utils as HabitatPlotUtils
import utils.geometry_utils as GeometryUtils


class PathfinderDatasetCollector:
    _DATA_COLLECTOR_NAME = "habitat_pathfinder"

    def __init__(
        self,
        dataset_save_dir,
        config_path="configs/dataset_collector/pathfinder_dataset_collector.yaml",
        num_trials_traj_planning_per_scene: int = int(1e4),
        num_trials_sample_per_traj: int = int(1e3),
        seed=None,
        save_debug_path_img=False,
        debug_path_img_save_dir="logs/trajectory_datasets",
        collect_depth=False,
    ):
        self._dataset_save_dir = dataset_save_dir
        self._data_collector_conf_path = config_path
        self._num_trials_traj_planning_per_scene = num_trials_traj_planning_per_scene
        self._num_trials_sample_per_traj = num_trials_sample_per_traj
        self._seed = seed
        self._save_debug_path_img = save_debug_path_img
        self._debug_path_img_save_dir = debug_path_img_save_dir
        self._collect_depth = collect_depth

        self._curr_nav_mesh_padding: Optional[float] = None

        os.makedirs(self._dataset_save_dir, exist_ok=True)
        os.makedirs(self._debug_path_img_save_dir, exist_ok=True)

        self._load_config(config_path)
        self._print_config()

        if self._seed is not None:
            self._set_seed(seed)

    def collect_data(self):
        """
        Collect and save the trajectory dataset.
        dataset_save_dir/
        ├──{map_name}_{traj_idx}/
        |   ├── #.jpg : Image captured from main camera (first element of camera_names).
        |   ├── #_{camera_name}_{depth}.tiff : Depth image captured from  camera (in mm).
        |   ├── traj_data.pkl : Pickle file that contains the trajectory
        |   ├── data_config.yaml : Dataset configuration [TrajectoryDataConfigYamlType].
        |   ├── global_occupancy_map.png : Image that contains the global occupancy map.
        |   └── global_occupancy_map.yaml : Global occupancy map config file.
        └──dataset_collector_config.yaml : Dataset collector configuration.
        """
        self._save_data_collector_config()

        time_spent_per_scene = []
        for idx_scene, (scene_id, scene_dataset_config_file) in enumerate(
            zip(self._scene_ids, self._scene_dataset_config_files)
        ):
            print(f"Collect Data {scene_id} [{idx_scene+1}/{len(self._scene_ids)}]...")
            scene_collect_start_time = datetime.now()

            self._collect_data_for_scene(scene_id, scene_dataset_config_file)

            scene_collect_end_time = datetime.now()
            time_spent_per_scene.append(
                (scene_collect_end_time - scene_collect_start_time).total_seconds()
            )

            print(
                f"Data collected in: {time_spent_per_scene[-1]} seconds. ETA: {np.mean(time_spent_per_scene) * (len(self._scene_ids) - idx_scene - 1)/60.0:.1f} mins. runtime: {np.sum(time_spent_per_scene)/60.0:.2f}m"
            )

    def _collect_data_for_scene(self, scene_id, scene_dataset_config_file):
        """
        Parameters
        ----------
        scene_id : str
            Scene ID.
        scene_dataset_config_file : str
            scene_dataset_config_file (json)
        trajectories_for_scene : List[Trajectory]
            Trajectories to collect data from.
        """
        print("\tLoading the scene...")
        self._load_sim(scene_id, scene_dataset_config_file)
        if self._seed is not None:
            self._sim.pathfinder.seed(self._seed)

        map_name = scene_id.split("/")[-1].split(".")[0]

        num_data_collected = 0
        path_heights_by_traj = []

        for idx_trial in range(self._num_trials_traj_planning_per_scene):
            print(
                f"\tCollecting {num_data_collected+1} / {self._n_trajectories_per_scene} \ttrial: {idx_trial}"
            )

            sampled_path_points, interpolated_trajectory = self._sample_trajectory()
            if sampled_path_points is None:
                continue

            # collect data from the trajectory
            if not self._collect_data_by_trajectory(
                interpolated_trajectory, map_name, num_data_collected
            ):
                continue

            path_height = sampled_path_points[0][1]
            path_heights_by_traj.append(path_height)

            if self._save_debug_path_img:
                HabitatPlotUtils.plot_path_on_2d_map(
                    self._sim,
                    sampled_path_points,
                    meters_per_pixel=0.05,
                    save_plot=True,
                    plot_save_path=os.path.join(
                        self._debug_path_img_save_dir,
                        f"path_{map_name}_{num_data_collected}.png",
                    ),
                    height=path_height,
                )

            num_data_collected += 1
            if num_data_collected >= self._n_trajectories_per_scene:
                break

        if num_data_collected < self._n_trajectories_per_scene:
            print("\tNot enough trajectories found.")
            # raise NotImplementedError

        self._save_occupancy_map_in_scene_dirs(map_name, path_heights_by_traj)

    def _recompute_navmesh_with_padding(self, padding: float) -> bool:
        if self._curr_nav_mesh_padding == padding:
            return True

        navmesh_settings = self._sim.pathfinder.nav_mesh_settings
        navmesh_settings.agent_radius = padding
        navmesh_settings.agent_height = self._robot_height
        navmesh_success = self._sim.recompute_navmesh(
            self._sim.pathfinder, navmesh_settings
        )

        if navmesh_success:
            self._curr_nav_mesh_padding = padding

        return navmesh_success

    def _sample_trajectory(self) -> Tuple[List[np.ndarray], TrajHabitat]:
        """
        Returns:
        --------
        sampled_path_points : List[np.ndarray]
            sampled path points.
        interpolated_trajectory : TrajHabitat
            interpolated trajectory.
        """
        assert self._recompute_navmesh_with_padding(
            self._robot_radius + self._margin_to_obst
        )

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

    def _interpolate_path_points(
        self,
        sampled_path_points: List[np.ndarray],
        start_yaw: float,
    ) -> TrajHabitat:
        return HabitatTaskUtils.interpolate_path_points(
            sampled_path_points,
            start_yaw,
            self._move_amount,
            self._rotate_amount_rad,
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
            ├── data_config.yaml : Dataset configuration [TrajectoryDataConfigYamlType].
            ├── global_occupancy_map.png : Image that contains the global occupancy map.
            └── global_occupancy_map.yaml : Global occupancy map config file.

        Returns:
        --------
        Success : bool
        """

        traj_data_save_dir = os.path.join(
            self._dataset_save_dir, f"{map_name}_{idx_traj}"
        )
        os.makedirs(traj_data_save_dir, exist_ok=True)

        # save traj_data
        traj_data_2d = convert_traj_habitat_to_traj_data_2d(trajectory)

        self._save_traj_data_np_pkl(traj_data_save_dir, traj_data_2d)

        with open(os.path.join(traj_data_save_dir, "traj_data_2d.yaml"), "w") as f:
            yaml.dump({"traj_data_2d": traj_data_2d}, f)  # TODO: add 3D traj data

        trajectory_positions = trajectory["positions"]
        trajectory_rotations = trajectory["rotations"]

        for idx, (pos, rot) in enumerate(
            zip(trajectory_positions, trajectory_rotations)
        ):
            # move the agent to the position
            self._set_agent_state(pos, rot)
            habitat_sensor_observations = self._sim.get_sensor_observations()
            rgb = habitat_sensor_observations["color_sensor"]
            depth = habitat_sensor_observations["depth_sensor"]

            # print min max value of depth
            # print(f"depth min: {np.min(depth)}, max: {np.max(depth)}")

            # save the rgb image to png
            rgb_save_path = os.path.join(traj_data_save_dir, f"{idx}.jpg")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            cv2.imwrite(rgb_save_path, rgb)

            if self._collect_depth:
                # save the depth image to tiff in mm (int)
                depth_save_path = os.path.join(traj_data_save_dir, f"{idx}_depth.tiff")
                # Convert meters to millimeters
                depth_mm = (depth * 1000).astype(np.uint16)
                cv2.imwrite(depth_save_path, depth_mm)

        self._save_data_config_file(traj_data_save_dir, trajectory_positions)

        return True

    def _set_agent_state(self, pos: np.ndarray, rot: np.ndarray) -> None:
        """
        Parameters
        ----------
        pos : np.ndarray
            position of the agent in habitat frame.
        rot : np.ndarray
            rotation of the agent in habitat frame.
        """
        agent = self._sim.get_agent(self._sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = pos
        agent_state.rotation = rot
        agent.set_state(agent_state)
        return

    def _save_traj_data_np_pkl(self, traj_data_save_dir: str, traj_data_2d: TrajData2D):
        traj_data_np_pkl = {
            "position": np.array(traj_data_2d["position"]),
            "yaw": np.array(traj_data_2d["yaw"]),
        }
        with open(os.path.join(traj_data_save_dir, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data_np_pkl, f)

    def _save_data_config_file(
        self, traj_data_save_dir: str, trajectory_positions: list
    ):
        # save data_config.yaml
        f_x = self._sim_settings["width"] / 2.0
        f_y = self._sim_settings["height"] / 2.0
        c_x = f_x
        c_y = f_y
        data_config = TrajectoryDataConfigYamlType(
            data_format_version=TRAJECTORY_DATASET_CONFIG_VERSION,
            data_collector_name=self._DATA_COLLECTOR_NAME,
            robot_dimensions=[
                2.0 * self._robot_radius,
                2.0 * self._robot_radius,
                self._robot_height,
            ],
            robot_scale=self._robot_radius,
            camera_names=["front_0"],
            camera_locations={
                "front_0": [0.0, 0.0, self._sim_settings["sensor_height"]]
            },
            camera_orientations={"front_0": [0.0, 0.0, 0.0, 1.0]},
            camera_configs={
                "front_0": {
                    "distortion_model": "plumb_bob",
                    "D": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "K": [[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]],
                }
            },
            data_length=len(trajectory_positions),
            timestep=self._dt,
            robot_velocity=self._move_amount / self._dt,
            max_v=self._move_amount / self._dt,
            max_w=(self._rotate_amount_rad) / self._dt,
        )
        with open(os.path.join(traj_data_save_dir, "data_config.yaml"), "w") as f:
            yaml.dump(data_config, f)

    def _save_occupancy_map_in_scene_dirs(
        self, map_name: str, path_heights_by_traj: List[float]
    ):
        print("\tRecomputing the navmesh for occmap generation...")
        # regererate the occupancy map for occupancy map.
        self._recompute_navmesh_with_padding(0.0)

        # get occupancy map by height
        occupancy_map_by_height = {}
        yaml_config_by_height = {}
        for idx_traj, path_height in enumerate(path_heights_by_traj):
            if path_height in occupancy_map_by_height:
                continue
            occupancy_map, yaml_config = HabitatPlotUtils.get_occupancy_map_and_config(
                self._sim, path_height, OCCUPANCY_MAP_RESOLUTION
            )
            occupancy_map_by_height[path_height] = occupancy_map
            yaml_config_by_height[path_height] = yaml_config

        # save occupancy map and yaml config
        for idx_traj, path_height in enumerate(path_heights_by_traj):
            target_dir = os.path.join(self._dataset_save_dir, f"{map_name}_{idx_traj}")
            map_path = os.path.join(target_dir, "global_occupancy_map.png")
            yaml_path = os.path.join(target_dir, "global_occupancy_map.yaml")

            cv2.imwrite(map_path, occupancy_map_by_height[path_height])
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_config_by_height[path_height], f)

        print("\tOccupancy maps saved.")

    def _save_data_collector_config(self):
        """
        Create `data_save_dir/dataset_collector_config.yaml` file.
        """
        with open(
            os.path.join(self._dataset_save_dir, "dataset_collector_config.yaml"), "w"
        ) as f:
            yaml.dump(self._data_collector_config, f)

    def _set_seed(self, seed):
        np.random.seed(seed)

    def _load_sim(self, scene_id, scene_dataset_config_file):
        self._sim_settings = {
            "width": self._sim_settings["width"],
            "height": self._sim_settings["height"],
            "scene_id": os.path.join(self._scene_dir, scene_id),
            "scene_dataset_config_file": os.path.join(
                self._scene_dir, scene_dataset_config_file
            ),
            "default_agent": self._sim_settings["default_agent"],
            "sensor_height": self._sim_settings["sensor_height"],
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
            self._sim.initialize_agent(self._sim_settings["default_agent"])

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            loaded_conf = yaml.safe_load(f)

        self._n_trajectories_per_scene = loaded_conf["n_trajectories_per_scene"]
        self._allow_stairs = loaded_conf["allow_stairs"]
        self._set_start_rot_toward_goal = loaded_conf["set_start_rot_toward_goal"]

        self._min_dist_between_waypoints = loaded_conf["min_dist_between_waypoints"]
        self._max_dist_between_waypoints = loaded_conf["max_dist_between_waypoints"]
        self._dist_to_goal_threshold = loaded_conf["dist_to_goal_threshold"]

        self._robot_radius = loaded_conf["robot_radius"]
        self._robot_height = loaded_conf["robot_height"]
        self._move_amount = loaded_conf["move_amount"]
        self._rotate_amount = loaded_conf["rotate_amount"]
        self._rotate_amount_rad = self._rotate_amount / 180.0 * np.pi
        self._dt = loaded_conf["dt"]

        self._margin_to_obst = loaded_conf["margin_to_obst"]

        self._sim_settings = loaded_conf["sim_settings"]

        self._scene_dir = loaded_conf["scene_dir"]
        self._scene_ids = loaded_conf["scene_ids"]  # Scene IDs must be unique.
        self._scene_dataset_config_files = loaded_conf["scene_dataset_config_files"]

        assert len(self._scene_ids) == len(
            set(self._scene_ids)
        ), "Scene IDs must be unique."

        self._data_collector_config = loaded_conf
        if self._seed is not None:
            self._data_collector_config["seed"] = self._seed

    def _print_config(self):
        print("PathfinderDatasetCollector initialized.")
        print(f"\tconfig_path                = {self._data_collector_conf_path}")
        print(f"\tdataset_save_dir           = {self._dataset_save_dir}")

        print(f"\tn_trajectories_per_scene   = {self._n_trajectories_per_scene}")
        print(f"\tallow_stairs               = {self._allow_stairs}")
        print(f"\tset_start_rot_toward_goal  = {self._set_start_rot_toward_goal}")

        print(f"\tmin_dist_between_waypoints = {self._min_dist_between_waypoints}")
        print(f"\tmax_dist_between_waypoints = {self._max_dist_between_waypoints}")
        print(f"\tdist_to_goal_threshold     = {self._dist_to_goal_threshold}")

        print(f"\trobot_radius               = {self._robot_radius}")
        print(f"\trobot_height               = {self._robot_height}")
        print(f"\tmove_amount                = {self._move_amount}")
        print(f"\trotate_amount              = {self._rotate_amount}")
        print(f"\trotate_amount_rad          = {self._rotate_amount_rad}")
        print(f"\tdt                         = {self._dt}")

        print(f"\tsim_settings               = {self._sim_settings}")

        print(f"\tscene_dir                  = {self._scene_dir}")
        print(f"\tnum scene_ids              = {len(self._scene_ids)}")
        print(f"\tn scene_dataset_config_files={len(self._scene_dataset_config_files)}")

        if self._seed is not None:
            print(f"\tseed                       = {self._seed}")
