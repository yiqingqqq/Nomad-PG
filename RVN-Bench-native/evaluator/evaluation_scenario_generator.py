import os
import sys

import yaml
import habitat_sim

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import utils.config_utils as ConfigUtils
import utils.habitat_task_utils as HabitatTaskUtils
from evaluator.eval_scenario_types import SeqPointGoalNavEvalScenario


class EvaluationScenarioGenerator:
    def __init__(self):
        self._sim: habitat_sim.Simulator = None
        pass

    def generate_scenario(
        self,
        scenario_save_path,
        data_dir="data",
        randomize_start_rotation=False,
        allow_stairs=False,
        num_ep_to_collect_per_scene=6,
        num_waypoints_per_ep=8,
        margin_to_obst=0.1,
        robot_radius=0.18,
        robot_height=1.0,
        min_path_length=4.0,
        max_path_length=8.0,
        target_scene_dir=None,
        scene_dataset_config_file=None,
        dataset_type="hm3d",
        path_finder_seed=2502032251,
        num_trials_per_scene=int(1e4),
        num_trials_per_ep=int(1e4),
        save_debug_plot=True,
        debug_img_save_dir=None,
    ):
        assert randomize_start_rotation == True, "Randomize start rotation is required"
        assert (
            allow_stairs == False
        ), "Stairs are not supported in this version of the code"
        assert dataset_type == "hm3d", "Only hm3d dataset is supported"

        self._data_dir = data_dir
        self._randomize_start_rotation = randomize_start_rotation
        self._allow_stairs = allow_stairs
        self._num_ep_to_collect_per_scene = num_ep_to_collect_per_scene
        self._num_waypoints_per_ep = num_waypoints_per_ep
        self._margin_to_obst = margin_to_obst
        self._robot_radius = robot_radius
        self._robot_height = robot_height
        self._min_path_length = min_path_length
        self._max_path_length = max_path_length
        self._target_scene_dir = target_scene_dir
        self._scene_dataset_config_file = scene_dataset_config_file
        self._dataset_type = dataset_type
        self._path_finder_seed = path_finder_seed

        self._num_trials_per_scene = num_trials_per_scene
        self._num_trials_per_ep = num_trials_per_ep

        self._scene_dir_names = self._get_scene_dir_names()
        self._eval_scenario = self._init_scenario()

        self._save_debug_plot = save_debug_plot
        self._debug_img_save_dir = debug_img_save_dir
        if self._save_debug_plot:
            print("Not implemented yet")

        # for debugging
        self._num_total_ep_to_collect = (
            len(self._scene_dir_names) * num_ep_to_collect_per_scene
        )

        for scene_dir_name in self._scene_dir_names:
            self._create_episodes_for_scene(scene_dir_name)

        print(
            f"Episode collected: {len(self._eval_scenario['episodes'])} / {self._num_total_ep_to_collect}"
        )
        print(f"Saving scenario to {scenario_save_path}")
        with open(scenario_save_path, "w") as file:
            yaml.dump(self._eval_scenario, file)

    def _get_scene_dir_names(self):
        scene_dir_names = [
            scene_dir_name
            for scene_dir_name in os.listdir(
                os.path.join(self._data_dir, self._target_scene_dir)
            )
            if os.path.isdir(
                os.path.join(self._data_dir, self._target_scene_dir, scene_dir_name)
            )
        ]
        scene_dir_names.sort()
        return scene_dir_names

    def _init_scenario(self):
        eval_scenario: SeqPointGoalNavEvalScenario = {
            "randomize_start_rotation": self._randomize_start_rotation,
            "allow_stairs": self._allow_stairs,
            "num_waypoints_per_ep": self._num_waypoints_per_ep,
            "margin_to_obst": self._margin_to_obst,
            "robot_radius": self._robot_radius,
            "robot_height": self._robot_height,
            "min_path_length": self._min_path_length,
            "max_path_length": self._max_path_length,
            "scene_dataset_config_file": self._scene_dataset_config_file,
            "dataset_type": self._dataset_type,
            "scenario_seed": self._path_finder_seed,
            "sim_settings": {
                "width": 256,
                "height": 256,
                "default_agent": 0,
                "sensor_height": 0.6,
                "color_sensor": True,
            },
            "episodes": [],
        }

        return eval_scenario

    def _create_episodes_for_scene(self, scene_dir_name):
        """
        Create epiosodes for a given scene
        Create `num_ep_to_collect_per_scene` episodes with `num_waypoints_per_ep` waypoints per episode.

        """
        path_finder_seed = self._path_finder_seed
        num_ep_collected = 0

        scene_file_name = f"{scene_dir_name.split('-')[-1]}.basis.glb"

        if self._dataset_type == "hm3d":
            scene_id = os.path.join(
                self._target_scene_dir, scene_dir_name, scene_file_name
            )
        else:
            raise ValueError("Unsupported dataset type")

        self._load_sim(scene_id)
        if not self._recompute_navmesh():
            print("Failed to recompute navmesh. Skipping scene...")
            return

        for iter_trial_scene in range(self._num_trials_per_scene):
            print(
                f"[{len(self._eval_scenario['episodes'])}/{self._num_total_ep_to_collect}]",
                f"\t scene_dir_name: {scene_dir_name} {iter_trial_scene+1}/{self._num_trials_per_scene}",
            )

            (
                start_pos,
                start_rotation,
                waypoint_positions,
                waypoint_rotations,
                geodesic_distances,
                trajectories,
            ) = HabitatTaskUtils.collect_waypoints(
                self._sim,
                self._num_trials_per_ep,
                self._num_waypoints_per_ep,
                self._min_path_length,
                self._max_path_length,
                self._allow_stairs,
                self._randomize_start_rotation,
                path_finder_seed,
            )

            path_finder_seed += 1

            if len(waypoint_positions) < self._num_waypoints_per_ep:
                print("\tNot enough waypoints found. Skipping episode...")
                continue

            # Save the episode
            self._eval_scenario["episodes"].append(
                {
                    "scene_id": scene_id,
                    "path_finder_seed": path_finder_seed,
                    "start_position": start_pos,
                    "start_rotation": start_rotation,
                    "waypoint_positions": waypoint_positions,
                    "waypoint_rotations": waypoint_rotations,
                    "geodesic_distances": geodesic_distances,
                    "trajectories": trajectories,
                }
            )

            num_ep_collected += 1
            if num_ep_collected >= self._num_ep_to_collect_per_scene:
                break

        if num_ep_collected < self._num_ep_to_collect_per_scene:
            print("!!!!!!!!!!!! Not enough episodes found !!!!!!!!!!!!")

    def _load_sim(self, scene_id):

        sim_settings = {
            "width": 256,  # Spatial resolution of the observations
            "height": 256,
            "scene_id": os.path.join(self._data_dir, scene_id),  # Scene path
            "scene_dataset_config_file": os.path.join(
                self._data_dir, self._scene_dataset_config_file
            ),
            "default_agent": 0,
            "sensor_height": 0.6,  # Height of sensors in meters
            "color_sensor": True,  # RGB sensor
            "depth_sensor": True,  # Depth sensor
            "semantic_sensor": True,  # Semantic sensor
            "seed": 1,  # used in the random navigation
            "enable_physics": False,  # kinematics only
        }
        cfg = ConfigUtils.make_cfg(sim_settings)

        try:
            self._sim.close()
        except:
            pass
        with HabitatTaskUtils.suppress_cpp_output():
            self._sim = habitat_sim.Simulator(cfg)

    def _recompute_navmesh(self):
        print(
            f"Recomputing navmesh with robot radius {self._robot_radius} and margin {self._margin_to_obst}"
        )
        navmesh_settings = self._sim.pathfinder.nav_mesh_settings
        navmesh_settings.agent_radius = self._robot_radius + self._margin_to_obst
        navmesh_settings.agent_height = self._robot_height
        navmesh_success = self._sim.recompute_navmesh(
            self._sim.pathfinder, navmesh_settings
        )
        return navmesh_success
