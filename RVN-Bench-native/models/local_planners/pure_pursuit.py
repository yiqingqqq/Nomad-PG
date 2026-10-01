import numpy as np


def calculate_action_to_follow_arc(
    waypoint: tuple[float, float],
    linear_velocity: float = 1.0,
    rotate_on_goal_reached=False,
    default_angular_velocity=0.31459,
    goal_reach_threshold=0.2,
    max_angular_velocity=1.0,
    turn_in_place_delta_threshold=np.pi / 4,  # 30 degrees
) -> tuple[float, float]:
    """
    Calculate action (v, w) to follow an arc to the given waypoint.
    1. Calculate the turning radius.
    2. Calculate the direction of the waypoint.
        2-1. If the waypoint is close enough, stop the robot.
        2-2. If the waypoint is in line with the robot, move forward or rotate in place.
        2-3. If the heading to the waypoint is too large, rotate in place.
    3. Calculate the angular velocity.
    4. If the angular velocity is greater than the max angular velocity, limit it.

    Parameters
    ----------
    waypoint : tuple[float, float]
        Waypoint to follow in robot's frame.
    linear_velocity : float, optional
        Linear velocity, by default 1.0


    Returns
    -------
    tuple[float, float]
        Action (v, w) to follow the arc to the waypoint.
    """

    # Calculate turning radius
    dist_i_f = np.linalg.norm((waypoint[0], waypoint[1]))

    if dist_i_f < goal_reach_threshold:
        # If the waypoint is close enough stop the robot
        if rotate_on_goal_reached:
            return (0.0, default_angular_velocity)
        return (0.0, 0.0)

    # Calculate the direction of the waypoint
    delta_i_f = np.arctan2(waypoint[1], waypoint[0])

    # If the waypoint is in line with the robot, move forward or rotate in place
    if abs(np.sin(delta_i_f)) < 1e-12:
        if waypoint[0] < 0:
            return (0.0, 1.0)
        return (linear_velocity, 0.0)

    if delta_i_f > turn_in_place_delta_threshold:
        return (0.0, default_angular_velocity)
    if delta_i_f < -turn_in_place_delta_threshold:
        return (0.0, -default_angular_velocity)

    turning_radius = 0.5 * dist_i_f / np.sin(delta_i_f)
    angular_velocity = linear_velocity / turning_radius

    if abs(angular_velocity) > max_angular_velocity:
        if angular_velocity > 0:
            angular_velocity = max_angular_velocity
        else:
            angular_velocity = -max_angular_velocity
        linear_velocity = angular_velocity * turning_radius

    return (linear_velocity, angular_velocity)


def get_waypoint_from_trajectoy(
    trajectory: list[tuple[float, float]], lookahead_distance: float = 1.0
) -> tuple[float, float]:
    """
    Get the waypoint from the trajectory based on the lookahead distance.
    """
    total_distance = np.linalg.norm(np.array(trajectory[0]))

    for i in range(len(trajectory) - 1):
        total_distance += np.linalg.norm(
            np.array(trajectory[i]) - np.array(trajectory[i + 1])
        )
        if total_distance >= lookahead_distance:
            return trajectory[i]
    return trajectory[-1]


def const_vel_arc_differential_pure_pursuit(
    trajectory: list[tuple[float, float]],
    lookahead_distance: float = 1.0,
    goal_reach_threshold: float = 0.2,
    max_angular_velocity: float = 1.0,
) -> tuple[float, float]:
    """
    Pure pursuit controller with constant velocity.
    1. Get the waypoint from the trajectory based on the lookahead distance.
    2. Calculate the action to follow the arc to the waypoint.

    Parameters
    ----------
    trajectory : list[tuple[float, float]]
        Trajectory to follow in robot's frame.
    lookahead_distance : float, optional
        Lookahead distance, by default 1.0
    goal_reach_threshold : float, optional
        Goal reach threshold, by default 0.2


    Returns
    -------
    tuple[float, float]
        Action (v, w) to follow the arc to the waypoint.
    """
    waypoint = get_waypoint_from_trajectoy(trajectory, lookahead_distance)
    print(waypoint)
    return calculate_action_to_follow_arc(
        waypoint,
        linear_velocity=1.0,
        goal_reach_threshold=goal_reach_threshold,
        max_angular_velocity=max_angular_velocity,
    )
