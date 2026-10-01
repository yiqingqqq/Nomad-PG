from typing import Tuple, Union

import numpy as np
import habitat_sim
import quaternion as qt

from agents.agent import ObsType, InfoType
import utils.geometry_utils as GeometryUtils


def transpose_habitat_pos_to_2d_frame(
    habitat_sim_position: np.ndarray,  # habitat 3d position
) -> np.ndarray:
    return np.array([-habitat_sim_position[2], -habitat_sim_position[0]])


def get_obs_and_yaw_from_pose(
    sim: habitat_sim.simulator.Simulator,
    agent: habitat_sim.agent.agent.Agent,
    position: np.ndarray,
    rotation: Union[np.ndarray, qt.quaternion],
) -> Tuple[np.ndarray, float]:
    # save initial state
    initial_agent_state = agent.get_state()

    # move agent to target pose
    target_agent_state = habitat_sim.AgentState()
    target_agent_state.position = position
    target_agent_state.rotation = rotation
    agent.set_state(target_agent_state)

    target_obs = sim.get_sensor_observations()
    target_rgba = target_obs["color_sensor"]
    target_yaw = agent.get_state().rotation.angle()

    # reset agent to initial state
    agent.set_state(initial_agent_state)

    return target_rgba, target_yaw


def get_obs_info(
    habitat_sensor_observation_history: list,
    goal_position: np.ndarray,
    goal_yaw: np.ndarray,
    robot_position: np.ndarray,
    robot_yaw: np.ndarray,
    goal_rgba: np.ndarray,
    num_wp_reached: int,
    collision: bool,
    timeout: bool,
    travel_distance: float,
    obs_size: int = 4,  # number of rgb frames in the observation
) -> Tuple[ObsType, InfoType]:
    """
    Parameters
    ----------
    habitat_sensor_observation_history : list
        list of habitat sensor observations
    goal_position : np.ndarray
        goal position in habitat frame
    goal_yaw : np.ndarray
        goal yaw
    robot_position : np.ndarray
        robot position in habitat frame
    robot_yaw : np.ndarray
        robot yaw
    goal_rgba : np.ndarray
        goal rgb image
    """
    assert len(habitat_sensor_observation_history) == obs_size

    robot_2d_world_pos = transpose_habitat_pos_to_2d_frame(robot_position)
    goal_2d_world_pos = transpose_habitat_pos_to_2d_frame(goal_position)

    realtive_goal_pose = GeometryUtils.transpose_world_pose_to_local_frame(
        (robot_2d_world_pos[0], robot_2d_world_pos[1], robot_yaw),
        (goal_2d_world_pos[0], goal_2d_world_pos[1], goal_yaw),
    )

    compass_radius, compass_yaw = GeometryUtils.get_gps_compass_from_relative_pose(
        realtive_goal_pose
    )

    if obs_size == 1:
        obs_rgbs = habitat_sensor_observation_history[0]["color_sensor"][..., :3]
    else:
        obs_rgbs = []
        for i in range(obs_size):
            obs_rgbs.append(
                habitat_sensor_observation_history[i]["color_sensor"][..., :3]
            )
        obs_rgbs = np.array(obs_rgbs, dtype=np.uint8)

    goal_rgb = goal_rgba[..., :3]

    l2_dist_to_goal = np.linalg.norm(robot_2d_world_pos - goal_2d_world_pos)

    obs = {
        "rgb": obs_rgbs,
        # "goal_pose": np.array(realtive_goal_pose),
        "pointgoal_with_gps_compass": np.array([compass_radius, compass_yaw]),
    }
    info = {
        "robot_pose": (robot_2d_world_pos[0], robot_2d_world_pos[1], robot_yaw),
        "goal_pose_world": (goal_2d_world_pos[0], goal_2d_world_pos[1], goal_yaw),
        "goal_rgb": goal_rgb,
        "num_wp_reached": num_wp_reached,
        "first_wp_reached": num_wp_reached > 0,
        "collision": collision,
        "timeout": timeout,
        "travel_distance": travel_distance,
        "l2_dist_to_goal": l2_dist_to_goal,
    }

    return obs, info


def check_collision(
    action: str,
    initial_agent_position: np.ndarray,
    final_agent_position: np.ndarray,
    move_amount: float,
    coll_move_diff_threshold: float,
    allow_stairs: bool = False,
    check_all_action: bool = False,
) -> bool:
    if action != "move_forward":
        if not check_all_action:
            return False

        # assert (
        #     np.linalg.norm(final_agent_position - initial_agent_position)
        #     < coll_move_diff_threshold
        # )
        if (
            np.linalg.norm(final_agent_position - initial_agent_position)
            > coll_move_diff_threshold
        ):
            print("Warning: difference in position for non-move action is too large.")
            return True
        return False

    if not allow_stairs and (initial_agent_position[1] != final_agent_position[1]):
        return True

    distance_moved = float(
        np.linalg.norm(final_agent_position - initial_agent_position)
    )

    if (
        distance_moved < move_amount - coll_move_diff_threshold
        or distance_moved > move_amount + coll_move_diff_threshold
    ):
        return True

    return False


def check_goal_reached(agent_position, goal_position, dist_to_goal_threshold) -> bool:
    dist_to_goal = np.linalg.norm(agent_position - goal_position)
    return dist_to_goal < dist_to_goal_threshold


def predict_next_pose_by_action(
    agent_position: np.ndarray,  # position in habitat frame
    agent_yaw: float,
    action: str,
    move_amount: float,  # move amount in meters
    rotate_amount: float,  # rotate amount in degrees
) -> Tuple[np.ndarray, float]:

    if action == "turn_left":
        agent_yaw += np.deg2rad(rotate_amount)
    elif action == "turn_right":
        agent_yaw -= np.deg2rad(rotate_amount)
    elif action == "move_forward":
        agent_position_world = GeometryUtils.convert_habitat_pos_to_3d_world_pos(
            agent_position
        )
        agent_position_world[0] += move_amount * np.cos(agent_yaw)
        agent_position_world[1] += move_amount * np.sin(agent_yaw)

        agent_position = GeometryUtils.convert_3d_world_pos_to_habitat_pos(
            agent_position_world
        )
    elif action == "stop":
        pass
    else:
        raise ValueError(f"Invalid action: {action}")

    agent_yaw = GeometryUtils.rad_to_mpi_pi(agent_yaw)
    return agent_position, agent_yaw
