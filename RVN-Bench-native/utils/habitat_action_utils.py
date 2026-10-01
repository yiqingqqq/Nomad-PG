import math
import numpy as np
import torch


def pure_pursuit_discrete_control(
    delta_lookahead: float,
    rotation_control_angle: float = math.pi / 6,
    get_action_as_batch_idx=False,
) -> str:
    if delta_lookahead > rotation_control_angle * 0.5:
        if get_action_as_batch_idx:
            return 1
        return "turn_left"
    elif delta_lookahead < -rotation_control_angle * 0.5:
        if get_action_as_batch_idx:
            return 2
        return "turn_right"
    else:
        if get_action_as_batch_idx:
            return 0
        return "move_forward"


def get_discrete_control_output_from_yaw(
    action_traj: np.ndarray,
    lookahead_point_idx: int = 4,
    rotation_control_angle: float = math.pi / 6,
    get_action_as_batch_idx: bool = False,
    select_first_trajectory_action: bool = True,
) -> str:
    """
    Returns the discrete control output from the continuous action.

    Returns:
        str: discrete control output [move_forward, turn_left, turn_right, stop]
    """

    if get_action_as_batch_idx:
        raise NotImplementedError()

    if len(action_traj.shape) == 3:
        if select_first_trajectory_action:
            action_traj = action_traj[0]
        else:
            raise NotImplementedError()
    lookahead_point = action_traj[lookahead_point_idx]
    assert (
        len(lookahead_point) == 4
    ), "Lookahead point should be a 4D vector (x, y, cos(theta), sin(theta))"

    lookahead_point_yaw = math.atan2(lookahead_point[3], lookahead_point[2])
    # print(
    #     f"lookahead_point_yaw: {lookahead_point_yaw} \t sin: {lookahead_point[2]} \t cos: {lookahead_point[3]}"
    # )

    return pure_pursuit_discrete_control(
        lookahead_point_yaw,
        rotation_control_angle,
        get_action_as_batch_idx=get_action_as_batch_idx,
    )


def get_discrete_control_output_from_trajectory_action_wo_stop(
    action_traj: np.ndarray,
    num_samples: int = 1,
    lookahead_point_idx: int = 4,
    rotation_control_angle: float = math.pi / 6,
    get_action_as_batch_idx: bool = False,
    select_first_trajectory_action: bool = True,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Returns the discrete control output from the continuous action.

    Returns:
        str: discrete control output [move_forward, turn_left, turn_right, stop]
    """
    if get_action_as_batch_idx:
        if select_first_trajectory_action:
            action_traj = action_traj[
                ::num_samples
            ]  # Select every nth action for batch processing
        else:
            raise NotImplementedError()
        actions = []
        for idx_env in range(action_traj.shape[0]):
            lookahead_point = action_traj[idx_env, lookahead_point_idx]
            delta_lookahead = math.atan2(lookahead_point[1], lookahead_point[0])

            actions.append(
                pure_pursuit_discrete_control(
                    delta_lookahead, rotation_control_angle, get_action_as_batch_idx
                )
            )
        actions = torch.tensor(actions, dtype=torch.int8).unsqueeze(1).to(device)
        return actions
    else:
        if len(action_traj.shape) == 3:
            if select_first_trajectory_action:
                action_traj = action_traj[0]
            else:
                raise NotImplementedError()
        lookahead_point = action_traj[lookahead_point_idx]
        delta_lookahead = math.atan2(lookahead_point[1], lookahead_point[0])

        return pure_pursuit_discrete_control(delta_lookahead, rotation_control_angle)


def get_discrete_control_output_from_trajectory_action(
    action_traj: np.ndarray,
    lookahead_point_idx: int = 4,
    goal_reach_threshold: float = 0.25,
    rotation_control_angle: float = math.pi / 6,
) -> str:
    """
    Returns the discrete control output from the continuous action.

    Returns:
        str: discrete control output [move_forward, turn_left, turn_right, stop]
    """
    lookahead_point = action_traj[lookahead_point_idx]
    print(f"lookahead_point: {lookahead_point}")

    dist_to_last_pos = np.linalg.norm(action_traj[-1])
    delta_lookahead = math.atan2(lookahead_point[1], lookahead_point[0])
    print(f"dist_to_last_pos: {dist_to_last_pos}")
    print(f"delta_lookahead: {delta_lookahead}")

    if dist_to_last_pos < goal_reach_threshold:
        return "stop"

    return pure_pursuit_discrete_control(delta_lookahead, rotation_control_angle)
