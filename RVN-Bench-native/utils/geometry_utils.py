from typing import Optional, Tuple, List

import magnum as mn
import numpy as np
import habitat_sim
import quaternion as qt
from scipy.spatial.transform import Rotation as R

T_HABITAT_TO_WORLD = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])


def get_rotation_toward_pos(
    current_pos: np.ndarray, target_pos: np.ndarray
) -> qt.quaternion:
    """
    Returns the rotation angle to face the target position in habitat frame.

    Args:
        current_pos (np.ndarray): current position in habitat frame
        target_pos (np.ndarray): target position in habitat frame
    """

    tangent = target_pos - current_pos
    tangent_orientation_matrix = mn.Matrix4.look_at(
        current_pos, current_pos + tangent, np.array([0, 1.0, 0.0])
    )
    tangent_orientation_q = mn.Quaternion.from_matrix(
        tangent_orientation_matrix.rotation()
    )
    return habitat_sim.utils.common.quat_from_magnum(tangent_orientation_q)


def get_rotation_toward_pos_in_list(
    current_pos: list[float], target_pos: list[float]
) -> list[float]:
    """
    Returns the rotation angle to face the target position in list.

    Args:
        current_pos (np.ndarray): current position in habitat frame
        target_pos (np.ndarray): target position in habitat frame
    Returns:
        list[float]: [w, x, y, z] in quaternion
    """
    quaternion = get_rotation_toward_pos(np.array(current_pos), np.array(target_pos))
    return [quaternion.w, quaternion.x, quaternion.y, quaternion.z]


def get_world_2d_yaw_from_quaternion(quaternion: qt.quaternion) -> float:
    """
    Returns the yaw angle from a quaternion.
    """
    raise NotImplementedError("Implement this function.")
    R = qt.as_rotation_matrix(quaternion)

    # Apply the transformation
    R_transformed = T_HABITAT_TO_WORLD @ R @ T_HABITAT_TO_WORLD.T

    # Convert back to quaternion
    q_transformed = qt.from_rotation_matrix(R_transformed)

    pass


def get_yaw_from_habitat_qt(q):
    world_dir = quaternion_to_2d_world_direction(q, (1, 0, 0))
    yaw = np.arctan2(world_dir[1], world_dir[0])
    return yaw


def convert_habitat_pos_to_3d_world_pos(habitat_pos: np.ndarray) -> np.ndarray:
    return np.array([-habitat_pos[2], -habitat_pos[0], habitat_pos[1]])


def convert_habitat_pos_to_2d_world_pos(habitat_pos: np.ndarray) -> np.ndarray:
    return np.array([-habitat_pos[2], -habitat_pos[0]])


def convert_habitat_pos_yaw_to_2d_world_pose(
    habitat_pos: np.ndarray, yaw: float
) -> Tuple[float, float, float]:
    """
    Returns
    -------
    2d_world_pose: Tuple[float, float, float]
        The 2d world pose in [x, y, yaw].
    """

    return [-habitat_pos[2], -habitat_pos[0], yaw]


def convert_3d_world_pos_to_habitat_pos(world_pos: np.ndarray) -> np.ndarray:
    return np.array([-world_pos[1], world_pos[2], -world_pos[0]])


def quaternion_to_2d_world_direction(q, reference_vector=(1, 0, 0)):
    # Create rotation object
    rotation = R.from_quat([q.x, q.y, q.z, q.w])

    # Rotate the reference vector
    rotated_vector = rotation.apply(reference_vector)
    return rotated_vector[0], -rotated_vector[2]


def rad_to_mpi_pi(rad: float) -> float:
    """
    Returns the angle in the range of [-pi, pi].
    """
    return (rad + np.pi) % (2 * np.pi) - np.pi


def get_yaw_toward_pos(current_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """
    Returns the yaw angle to face the target position.

    Args:
        current_pos (np.ndarray): current position in habitat frame
        target_pos (np.ndarray): target position in habitat frame

    Returns:
        float: yaw angle in radian in 2d world frame
    """
    curr_pos_2d_world = np.array([-current_pos[2], -current_pos[0]])
    target_pos_2d_world = np.array([-target_pos[2], -target_pos[0]])

    diff = target_pos_2d_world - curr_pos_2d_world
    yaw = np.arctan2(diff[1], diff[0])
    return yaw


def get_quaternion_form_yaw(yaw: float) -> qt.quaternion:
    """
    Returns a quaternion from a yaw angle.
    """
    return qt.from_euler_angles(np.array([0, yaw, 0]))


def get_quaternion_form_yaw_in_list(yaw: float) -> qt.quaternion:
    """
    Returns a quaternion from a yaw angle.
    """
    return qt.from_euler_angles(np.array([0, yaw, 0]))


def get_yaw_diff(yaw1: float, yaw2: float) -> float:
    """
    Returns the difference between two yaw angles.
    """
    return rad_to_mpi_pi(yaw2 - yaw1)


def transpose_world_pose_to_local_frame(
    local_origin: tuple[float, float, float], pose_world: tuple[float, float, float]
) -> tuple[float, float, float]:
    """
    Transpose the pose from world frame to local frame.

    Parameters
    ----------
    local_origin: tuple[float, float, float]
        Origin of the local frame in world frame in [x, y, yaw].
    pose_world: tuple[float, float, float]
        Pose in world frame in [x, y, yaw].

    Returns
    -------
    pose_local: tuple[float, float, float]
        Pose in local frame in [x, y, yaw].
    """
    x_local, y_local = transpose_world_pos_to_local_frame(
        local_origin, (pose_world[0], pose_world[1])
    )
    yaw_local = rad_to_mpi_pi(pose_world[2] - local_origin[2])

    return x_local, y_local, yaw_local


def get_gps_compass_from_relative_pose(
    relative_pose: tuple[float, float, float],
) -> Tuple[float, float]:
    """
    Returns the GPS compass from a relative pose.

    Args:
        relative_pose (np.ndarray): relative pose in [x_g, y_g, yaw_g]

    Returns:
        Tuple[float, float]: (compass_radius, compass_yaw)
    """
    compass_radius = np.linalg.norm(relative_pose[:2])
    compass_yaw = np.arctan2(relative_pose[1], relative_pose[0])
    return compass_radius, compass_yaw


def transpose_world_pos_to_local_frame(
    local_origin: tuple[float, float, float], pos_world: tuple[float, float]
) -> tuple[float, float]:
    """
    Transpose the pose from world frame to local frame.

    Parameters
    ----------
    local_origin: tuple[float, float, float]
        Origin of the local frame in world frame in [x, y, yaw].
    pos_world: tuple[float, float]
        Pose in world frame in [x, y, yaw].

    Returns
    -------
    pose_local: tuple[float, float]
        Pose in local frame in [x, y, yaw].
    """
    x_origin, y_origin, yaw_origin = local_origin
    x_world, y_world = pos_world

    # Translate to the local origin
    dx = x_world - x_origin
    dy = y_world - y_origin

    # Rotate the translated coordinates by the negative of the local origin's yaw
    cos_yaw = np.cos(-yaw_origin)
    sin_yaw = np.sin(-yaw_origin)

    x_local = cos_yaw * dx - sin_yaw * dy
    y_local = sin_yaw * dx + cos_yaw * dy

    return x_local, y_local


def check_path_is_on_plane(path_points: list, height: Optional[float] = None) -> bool:
    """
    Check if the path is on the same plane.
    """
    if height is not None:
        height = path_points[0][1]

    for point in path_points:
        if point[1] != height:
            return False
    return True


def move_straight_toward_target(
    initial_position: np.ndarray, target_position: np.ndarray, move_amount: float
) -> np.ndarray:
    """
    March straight.

    Returns
    -------
    final_position: np.ndarray
        The final position after moving. It can move past the target position.
    """
    move_dir = target_position - initial_position
    move_dir = move_dir / np.linalg.norm(move_dir)
    return initial_position + move_dir * move_amount


def rotate_toward_yaw(
    current_yaw: float, target_yaw: float, rotate_amount: float
) -> float:
    """
    Rotate toward the target yaw. It does not rotate past the target yaw.

    Parameters
    ----------
    current_yaw: float
        Current yaw in radian.
    target_yaw: float
        Target yaw in radian.
    rotate_amount: float
        The amount of rotation in radian. [0, 2*pi]
    """
    diff = rad_to_mpi_pi(target_yaw - current_yaw)

    if np.abs(diff) < rotate_amount:
        return target_yaw

    return rad_to_mpi_pi(current_yaw + np.sign(diff) * rotate_amount)


def convert_trajectories_to_local_frame_pos_yaw(
    neg_expt_positions: List[List[np.ndarray]],  # in habitat frame
    neg_expt_yaws: List[List[float]],
    start_pos: np.ndarray,  # in habitat frame
    start_yaw: np.ndarray,
) -> List[List[Tuple[float, float, float]]]:

    origin_pose = convert_habitat_pos_yaw_to_2d_world_pose(start_pos, start_yaw)

    trajectories_in_local_frame: List[List[Tuple[float, float, float]]] = []

    for idx_traj, trajectory in enumerate(neg_expt_positions):
        trajecty_local = []
        for idx_pos, pose_on_traj in enumerate(trajectory):
            target_pose = convert_habitat_pos_yaw_to_2d_world_pose(
                pose_on_traj, neg_expt_yaws[idx_traj][idx_pos]
            )
            local_pose = transpose_world_pose_to_local_frame(origin_pose, target_pose)
            trajecty_local.append(local_pose)

        trajectories_in_local_frame.append(trajecty_local)

    return trajectories_in_local_frame
