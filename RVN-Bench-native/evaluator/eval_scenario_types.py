from typing import TypedDict, List


class SimSettings(TypedDict):
    color_sensor: bool
    default_agent: int
    height: float
    sensor_height: float
    width: float


class SeqPointGoalNavEvalEpisode(TypedDict):
    geodesic_distances: List[float]
    path_finder_seed: int
    scene_id: str  # relative path to data_dir
    start_position: List[float]
    start_rotation: List[float]  # w, x, y, z
    trajectories: List[List[List[float]]]  # list of trajectories
    waypoint_positions: List[List[float]]
    waypoint_rotations: List[List[float]]  # w, x, y, z


class SeqPointGoalNavEvalScenario(TypedDict):
    allow_stairs: bool
    scene_dataset_config_file: str  # relative path to data_dir
    dataset_type: str
    episodes: List[SeqPointGoalNavEvalEpisode]
    margin_to_obst: float
    robot_radius: float
    robot_height: float
    max_path_length: float
    min_path_length: float
    num_waypoints_per_ep: int
    randomize_start_rotation: bool
    scenario_seed: int  # seed used to generate the scenario
    sim_settings: SimSettings
