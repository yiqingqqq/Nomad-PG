import os
from typing import Tuple, List, Optional
import contextlib

import numpy as np
import quaternion as qt
import habitat_sim

from dataset_collector.dataset_types import TrajHabitat
import utils.geometry_utils as GeometryUtils


# Context manager to suppress C++ and Python logs
@contextlib.contextmanager
def suppress_cpp_output():
    # Open a null device
    devnull = os.open(os.devnull, os.O_WRONLY)
    # Save the current file descriptors for stdout and stderr
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        # Redirect stdout and stderr to devnull
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Restore original stdout and stderr
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull)


def validate_path(
    start_pos,
    path_points,
    geodesic_distance,
    min_path_length,
    max_path_length,
    allow_stairs,
) -> bool:
    if not allow_stairs and not GeometryUtils.check_path_is_on_plane(
        path_points, start_pos[1]
    ):
        return False

    if geodesic_distance < min_path_length or geodesic_distance > max_path_length:
        return False

    return True


def navigable_point_to_list(point) -> List[float]:
    return [float(val) for val in point]


def collect_waypoints(
    sim: habitat_sim.simulator.Simulator,
    num_trials: int,
    num_waypoints_to_collect: int,
    min_path_length: int,
    max_path_length: int,
    allow_stairs: bool,
    randomize_start_rotation: bool,
    path_finder_seed: Optional[int] = None,
) -> Tuple[
    List[float],
    List[float],
    List[List[float]],
    List[List[float]],
    List[float],
    List[List[List[float]]],
]:
    """
    Collects waypoints for the pathfinder dataset.

    Returns
    -------
    start_pos : List[float]
        start position
    start_rotation : List[float]
        start rotation in habitat frame
    waypoint_positions : List[List[float]]
        list of waypoint positions in habitat frame
    waypoint_rotations : List[List[float]]
        list of waypoint rotations
    geodesic_distances : List[float]
        list of geodesic distances
    trajectories : List[List[List[float]]]
        list of trajectories in habitat frame
    """

    if path_finder_seed is not None:
        sim.pathfinder.seed(path_finder_seed)
        np.random.seed(path_finder_seed)

    start_pos = sim.pathfinder.get_random_navigable_point()
    start_rotation = None
    last_waypoint_pos = start_pos

    waypoint_positions = []
    waypoint_rotations = []
    geodesic_distances = []
    trajectories = []

    for _ in range(num_trials):
        waypoint_pos_sample = sim.pathfinder.get_random_navigable_point()

        # pre-validation
        if not allow_stairs and waypoint_pos_sample[1] != start_pos[1]:
            continue

        if np.linalg.norm(waypoint_pos_sample - last_waypoint_pos) > max_path_length:
            continue

        path_sample = habitat_sim.ShortestPath()
        path_sample.requested_start = last_waypoint_pos
        path_sample.requested_end = waypoint_pos_sample

        path_found = sim.pathfinder.find_path(path_sample)
        path_points = path_sample.points
        geodesic_distance = path_sample.geodesic_distance
        # path_points: list[np.array([p_0, p_1, p_2]), ...]

        if not path_found or not validate_path(
            start_pos,
            path_points,
            geodesic_distance,
            min_path_length,
            max_path_length,
            allow_stairs,
        ):
            continue

        # NOTE: waypoint rotations will be deprecated
        waypoint_positions.append(navigable_point_to_list(waypoint_pos_sample))
        waypoint_rotations.append(
            GeometryUtils.get_rotation_toward_pos_in_list(
                last_waypoint_pos, waypoint_pos_sample
            )
        )
        geodesic_distances.append(geodesic_distance)
        trajectories.append([navigable_point_to_list(point) for point in path_points])

        last_waypoint_pos = waypoint_pos_sample

        if len(waypoint_positions) >= num_waypoints_to_collect:
            break

    if not randomize_start_rotation and len(waypoint_positions) > 0:
        start_rotation = GeometryUtils.get_rotation_toward_pos_in_list(
            last_waypoint_pos, waypoint_positions[0]
        )
    else:
        random_yaw = np.random.uniform(0, 2 * np.pi)
        start_rotation = GeometryUtils.get_quaternion_form_yaw_in_list(random_yaw)

    start_rotation = [
        start_rotation.w,
        start_rotation.x,
        start_rotation.y,
        start_rotation.z,
    ]

    return (
        navigable_point_to_list(start_pos),
        start_rotation,
        waypoint_positions,
        waypoint_rotations,
        geodesic_distances,
        trajectories,
    )


def sample_path_in_sim(
    sim: habitat_sim.simulator.Simulator,
    num_trials: int,
    min_path_length: int,
    max_path_length: int,
    allow_stairs: bool,
    randomize_start_rotation: bool,
    randomize_goal_rotation: bool,
    path_finder_seed: Optional[int] = None,
    try_again_on_failure: bool = True,
) -> Tuple[
    np.ndarray, qt.quaternion, np.ndarray, qt.quaternion, List[np.ndarray], float
]:
    """
    Returns
    -------
    sampled_start_pos: np.ndarray
        The sampled start position in the Habitat frame.
    sampled_start_rotation: qt.quaternion
        The sampled start rotation.
    sampled_goal_pos: np.ndarray
        The sampled goal position in the Habitat frame.
    sampled_goal_rotation: qt.quaternion
        The sampled goal rotation
    sampled_path_points: List[np.ndarray]
        The sampled path points in the Habitat frame.
    geodesic_distance: float
        The geodesic distance of the sampled path.
    """

    if path_finder_seed is not None:
        sim.pathfinder.seed(path_finder_seed)
        np.random.seed(path_finder_seed)

    for _ in range(num_trials):
        sampled_start_pos = sim.pathfinder.get_random_navigable_point()
        sampled_goal_pos = sim.pathfinder.get_random_navigable_point()

        # pre-validation
        if not allow_stairs and sampled_start_pos[1] != sampled_goal_pos[1]:
            continue
        if np.linalg.norm(sampled_start_pos - sampled_goal_pos) > max_path_length:
            continue

        sampled_path = habitat_sim.ShortestPath()
        sampled_path.requested_start = sampled_start_pos
        sampled_path.requested_end = sampled_goal_pos

        path_found = sim.pathfinder.find_path(sampled_path)
        path_points = sampled_path.points
        geodesic_distance = sampled_path.geodesic_distance

        if not path_found or not validate_path(
            sampled_start_pos,
            path_points,
            geodesic_distance,
            min_path_length,
            max_path_length,
            allow_stairs,
        ):
            continue

        if randomize_start_rotation:
            random_start_yaw = np.random.uniform(0, 2 * np.pi)
            sampled_start_rotation = GeometryUtils.get_quaternion_form_yaw(
                random_start_yaw
            )
        else:
            sampled_start_rotation = GeometryUtils.get_rotation_toward_pos(
                sampled_start_pos, sampled_goal_pos
            )

        if randomize_goal_rotation:
            random_goal_yaw = np.random.uniform(0, 2 * np.pi)
            sampled_goal_rotation = GeometryUtils.get_quaternion_form_yaw(
                random_goal_yaw
            )
        else:
            sampled_goal_rotation = GeometryUtils.get_rotation_toward_pos(
                sampled_start_pos, sampled_goal_pos
            )

        return (
            sampled_start_pos,
            sampled_start_rotation,
            sampled_goal_pos,
            sampled_goal_rotation,
            path_points,
            geodesic_distance,
        )

    if try_again_on_failure:
        print("Could not sample valid start and goal pose. Trying again.")
        return sample_path_in_sim(
            sim,
            num_trials,
            min_path_length,
            max_path_length,
            allow_stairs,
            randomize_start_rotation,
            randomize_goal_rotation,
            try_again_on_failure=True,
            path_finder_seed=None,
        )
    else:
        return None, None, None, None, None, None


def sample_path_points(
    sim: habitat_sim.simulator.Simulator,
    num_trials: int,
    min_path_length: int,
    max_path_length: int,
    allow_stairs: bool,
    path_finder_seed: Optional[int] = None,
    try_again_on_failure: bool = True,
) -> Tuple[List[np.ndarray], float]:
    """
    Returns
    -------
    sampled_start_pos: np.ndarray
        The sampled start position in the Habitat frame.
    sampled_start_rotation: qt.quaternion
        The sampled start rotation.
    sampled_goal_pos: np.ndarray
        The sampled goal position in the Habitat frame.
    sampled_goal_rotation: qt.quaternion
        The sampled goal rotation
    """
    _, _, _, _, path_points, geodesic_distance = sample_path_in_sim(
        sim,
        num_trials,
        min_path_length,
        max_path_length,
        allow_stairs,
        True,
        True,
        path_finder_seed,
        try_again_on_failure,
    )

    return path_points, geodesic_distance


def sample_start_and_goal_pose(
    sim: habitat_sim.simulator.Simulator,
    num_trials: int,
    min_path_length: int,
    max_path_length: int,
    allow_stairs: bool,
    randomize_start_rotation: bool,
    randomize_goal_rotation: bool,
    path_finder_seed: Optional[int] = None,
    try_again_on_failure: bool = True,
) -> Tuple[np.ndarray, qt.quaternion, np.ndarray, qt.quaternion]:
    """
    Returns
    -------
    sampled_start_pos: np.ndarray
        The sampled start position in the Habitat frame.
    sampled_start_rotation: qt.quaternion
        The sampled start rotation.
    sampled_goal_pos: np.ndarray
        The sampled goal position in the Habitat frame.
    sampled_goal_rotation: qt.quaternion
        The sampled goal rotation
    """
    start_pos, start_rotation, goal_pos, goal_rotation, _, _ = sample_path_in_sim(
        sim,
        num_trials,
        min_path_length,
        max_path_length,
        allow_stairs,
        randomize_start_rotation,
        randomize_goal_rotation,
        path_finder_seed,
        try_again_on_failure,
    )

    return start_pos, start_rotation, goal_pos, goal_rotation


def interpolate_path_points(
    path_points: List[np.ndarray],
    start_yaw: float,
    move_amount: float,
    rotate_amount: float,
) -> TrajHabitat:
    """
    Parameters
    ----------
    path_points : List[np.ndarray]
        List of path points.
    start_yaw : float
        The start yaw in [-pi, pi] rad.
    move_amount : float
        The amount to move. Equivalent to the v / dt.
    rotate_amount : float
        The amount to rotate. Equivalent to the w / dt.
    """

    interpolated_path_positions = [path_points[0]]
    interpolated_path_yaws = [start_yaw]
    time_left_to_reach = 0.0  # ratio of d_t

    # Rotate toward the first waypoint (path_points[1])
    target_yaw = GeometryUtils.get_yaw_toward_pos(path_points[0], path_points[1])

    interpolated_points, interpolated_yaws, time_left_to_reach = (
        interpolate_yaws_between_points(
            start_yaw,
            target_yaw,
            path_points[0],
            rotate_amount,
            time_left_to_reach,
        )
    )
    interpolated_path_positions.extend(interpolated_points)
    interpolated_path_yaws.extend(interpolated_yaws)

    # Move and rotate toward the rest of the waypoints
    for i in range(len(path_points) - 1):
        initial_position = path_points[i]
        final_position = path_points[i + 1]

        if i == len(path_points) - 2:
            final_yaw = GeometryUtils.get_yaw_toward_pos(
                path_points[-2], path_points[-1]
            )
        else:
            final_yaw = GeometryUtils.get_yaw_toward_pos(
                final_position, path_points[i + 2]
            )

        (
            interpolated_points,
            interpolated_yaws,
            time_left_to_reach,
        ) = interpolate_straight_path_between_points(
            initial_position,
            final_position,
            final_yaw,
            move_amount,
            rotate_amount,
            time_left_to_reach,
        )
        interpolated_path_positions.extend(interpolated_points)
        interpolated_path_yaws.extend(interpolated_yaws)

    if not (
        np.array_equal(interpolated_path_positions[-1], path_points[-1])
        and np.array_equal(interpolated_path_yaws[-1], final_yaw)
    ):
        interpolated_path_positions.append(path_points[-1])
        interpolated_path_yaws.append(final_yaw)

    return TrajHabitat(
        {
            "positions": interpolated_path_positions,
            "rotations": [
                GeometryUtils.get_quaternion_form_yaw(yaw)
                for yaw in interpolated_path_yaws
            ],
            "yaws": interpolated_path_yaws,
        }
    )


def interpolate_yaws_between_points(
    initial_yaw: float,
    final_yaw: float,
    position: np.ndarray,
    rotate_amount: float,
    time_left_to_reach: float,
) -> Tuple[List[np.ndarray], List[float], float]:
    """
    Parameters
    ----------
    initial_yaw : float
        The initial yaw in [-pi, pi] rad.
    final_yaw : float
        The final yaw in [-pi, pi] rad.
    position: np.ndarray
        The position to rotate.
    rotate_amount : float
        The amount to rotate. Equivalent to the w / dt.
    time_left_to_reach : float
        The time left to reach the initial position. It is the ratio of d_t.

    Returns
    -------
    interpolated_positions : List[np.ndarray]
        The interpolated points.
    interpolated_yaws : List[float]
        The interpolated yaws.
    time_left_to_reach : float
        The time left to reach the final position. It is the ratio of d_t.
    """
    assert time_left_to_reach < 1.0 and time_left_to_reach >= 0.0

    interpolated_positions = []
    interpolated_yaws = []

    time_left_to_march = 1.0 - time_left_to_reach

    time_to_rotate_to_target_yaw = (
        abs(GeometryUtils.rad_to_mpi_pi(final_yaw - initial_yaw)) / rotate_amount
    )
    last_yaw = initial_yaw

    # Roatate toward
    while time_to_rotate_to_target_yaw >= time_left_to_march:
        last_yaw = GeometryUtils.rotate_toward_yaw(
            last_yaw, final_yaw, time_left_to_march * rotate_amount
        )
        interpolated_positions.append(position)
        interpolated_yaws.append(last_yaw)

        time_to_rotate_to_target_yaw -= time_left_to_march
        time_left_to_march = 1.0

    time_left_to_march_after_final_yaw = (
        time_left_to_march - time_to_rotate_to_target_yaw
    )
    time_left_to_reach_final_yaw = 1.0 - time_left_to_march_after_final_yaw

    return interpolated_positions, interpolated_yaws, time_left_to_reach_final_yaw


def interpolate_straight_path_between_points(
    initial_position: np.ndarray,
    final_position: np.ndarray,
    final_yaw: float,
    move_amount: float,
    rotate_amount: float,
    time_left_to_reach: float,
) -> Tuple[List[np.ndarray], List[float], float]:
    """
    Parameters
    ----------
    initial_position : np.ndarray
        The initial position.
    final_position : np.ndarray
        The final position.
    final_yaw : float
        The final yaw in [0, 2 * np.pi] rad.
    move_amount : float
        The amount to move. Equivalent to the v / dt.
    rotate_amount : float
        The amount to rotate. Equivalent to the w / dt.
    time_left_to_reach : float
        The time left to reach the initial position. It is the ratio of d_t.

    Returns
    -------
    interpolated_positions : List[np.ndarray]
        The interpolated points.
    interpolated_yaws : List[float]
        The interpolated yaws.
    time_left_to_reach : float
        The time left to reach the final position. It is the ratio of d_t.
    """
    assert time_left_to_reach < 1.0 and time_left_to_reach >= 0.0

    interpolated_positions = []
    interpolated_yaws = []

    time_left_to_march = 1.0 - time_left_to_reach

    yaw_toward_target = GeometryUtils.get_yaw_toward_pos(
        initial_position, final_position
    )

    # Move forward the final position
    time_to_reach_final_pos = (
        np.linalg.norm(final_position - initial_position) / move_amount
    )
    last_position = initial_position

    while time_to_reach_final_pos >= time_left_to_march:
        last_position = GeometryUtils.move_straight_toward_target(
            last_position, final_position, time_left_to_march * move_amount
        )
        interpolated_positions.append(last_position)
        interpolated_yaws.append(yaw_toward_target)

        time_to_reach_final_pos -= time_left_to_march
        time_left_to_march = 1.0

    time_left_to_march -= time_to_reach_final_pos

    time_left_to_reach_to_final_pos = 1.0 - time_left_to_march
    rotation_positions, rotation_yaws, time_left_to_reach_to_final_point = (
        interpolate_yaws_between_points(
            yaw_toward_target,
            final_yaw,
            final_position,
            rotate_amount,
            time_left_to_reach_to_final_pos,
        )
    )
    interpolated_positions.extend(rotation_positions)
    interpolated_yaws.extend(rotation_yaws)

    return interpolated_positions, interpolated_yaws, time_left_to_reach_to_final_point
