from typing import TypedDict, List, Optional

import numpy as np

TRAJECTORY_DATASET_CONFIG_VERSION = "v0.0.2"
OCCUPANCY_MAP_RESOLUTION = 0.05  # [m]
ACTION_MAP = {0: "move_forward", 1: "turn_left", 2: "turn_right"}
ACTION_NAMES_TO_IDX = {"move_forward": 0, "turn_left": 1, "turn_right": 2}


class CameraConfigType(TypedDict):
    """
    Members
    -------
    distortion_model : str
        - "plumb_bob" or "rational_polynomial"
    D : list[float]
        - The distortion parameters, size depending on the distortion model.
        - For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
    K : list[list[float]]
        - Intrinsic camera matrix for the raw (distorted) images.
        - 3 x 3 matrix
        - Projects 3D points in the camera coordinate frame to 2D pixel
        coordinates using the focal lengths (fx, fy) and principal point (cx, cy).
        k = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    """

    distortion_model: str  # "plumb_bob" or "rational_polynomial"
    D: List[float]  # distortion parameters
    K: List[List[float]]  # intrinsic camera matrix


class TrajectoryDataConfigYamlType(TypedDict):
    """
    Type of the dataset collector config yaml file for the each trajectory data.
    """

    data_collector_name: str  # name of the data collector
    data_format_version: str = (
        TRAJECTORY_DATASET_CONFIG_VERSION  # version of the data format
    )
    robot_dimensions: List[float]  # (length, width, height)
    robot_scale: float  # [m] scale of the robot
    camera_names: List[str]  # main camera name should be the first element
    camera_locations: dict[str, List[float]]  # {camera_name: (x, y, z)}
    camera_orientations: dict[str, List[float]]  # (x, y, z, w) quaternion
    camera_configs: dict[str, CameraConfigType]

    data_length: int  # length of the trajectory data
    timestep: float  # in seconds (1/hz)

    robot_velocity: float  # [m/s] robot velocity
    max_v: float  # max linear velocity in m/s
    max_w: float  # max angular velocity in rad/s


class TrajData2D(TypedDict):
    """
    Type of the trajectory data.
    Note that the fields are 'position' and 'yaw' instead of 'positions' and 'rotations'
    to match the GoStanford dataset format.
    """

    position: List[List[float]]  # [[x_0, y_0], [x_1, y_1], ...] in [m].
    yaw: List[float]  # [yaw_0, yaw_1, ...] in [rad].


class TrajData3D(TypedDict):
    """
    Type of the trajectory data.
    """

    positions: List[List[float]]  # [[x_0, y_0, z_0], ...] in world frame (z up) in [m].
    rotations: List[List[float]]  # [rot_0, ...] in [quaternion(w, x, y, z)].


class TrajHabitat(TypedDict):
    """
    Type of the trajectory data in habitat frame.
    """

    positions: List[np.ndarray]  # [pos_0, pos_1, ...] in [m] in habitat frame.
    rotations: List[np.ndarray]  # [rot_0, ...] in [quaternion(w, x, y, z)].
    yaws: List[float]  # [yaw_0, ...] in [rad].


class DiscreteHabitatActions(TypedDict):
    """
    Type of the discrete actions in habitat frame.
    """

    action_map: dict[int, str]  # {action_idx: action_name}
    actions: List[int]  # [action_0, action_1, ...] in [action_idx].


def convert_traj_habitat_to_traj_data_2d(traj_habitat: TrajHabitat) -> TrajData2D:
    """
    Converts a TrajHabitat instance to a TrajData2D instance by extracting
    only the x and y components from positions and using the provided yaws.

    :param traj_habitat: TrajHabitat data to be converted.
    :return: TrajData2D with extracted positions and rotations.
    """

    positions_2d = [
        [float(-pos[2]), float(-pos[0])] for pos in traj_habitat["positions"]
    ]
    yaw_2d = [float(yaw) for yaw in traj_habitat["yaws"]]  # Convert numpy array to list

    return TrajData2D(position=positions_2d, yaw=yaw_2d)


def convert_traj_habitat_to_traj_data_3d(traj_habitat: TrajHabitat) -> TrajData3D:
    """
    Converts a TrajHabitat instance to a TrajData3D instance by extracting
    all the components from positions and rotations.

    :param traj_habitat: TrajHabitat data to be converted.
    :return: TrajData3D with extracted positions and rotations.
    """
    raise NotImplementedError("Implement this function with proper transformation.")
    positions_3d = [
        [float(-pos[2]), float(-pos[0]), float(pos[1])]
        for pos in traj_habitat["positions"]
    ]
    rotations_3d = [rot.tolist() for rot in traj_habitat["rotations"]]
    return TrajData3D(
        positions=positions_3d,
        rotations=rotations_3d,
    )
