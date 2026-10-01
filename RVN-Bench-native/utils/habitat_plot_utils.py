import os
import math
from typing import List, Tuple

import magnum as mn
import numpy as np
import cv2
import yaml
from matplotlib import pyplot as plt
from PIL import Image

import habitat_sim
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import d3_40_colors_rgb


def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None, save_plot=False, plot_save_path=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)

    if save_plot:
        if plot_save_path is None:
            raise ValueError("Please provide a path to save the plot")
        if not os.path.exists(os.path.dirname(plot_save_path)):
            os.makedirs(os.path.dirname(plot_save_path))
        plt.savefig(plot_save_path)
        plt.close()
    else:
        plt.show(block=False)


def plot_path_on_2d_map(
    sim: habitat_sim.Simulator,
    path_points: List[np.ndarray],
    meters_per_pixel: float = 0.025,
    save_plot: bool = False,
    plot_save_path: str = None,
    height=None,
):
    if height is None:
        height = sim.scene_aabb.y().min
    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )

    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
    # convert world trajectory points to maps module grid points
    trajectory = [
        maps.to_grid(
            path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in path_points
    ]
    # remove duplicate points
    trajectory = [trajectory[0]] + [
        trajectory[i]
        for i in range(1, len(trajectory))
        if trajectory[i] != trajectory[i - 1]
    ]

    grid_tangent = mn.Vector2(
        trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
    )
    path_initial_tangent = grid_tangent / grid_tangent.length()
    initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
    # draw the agent and trajectory on the map
    maps.draw_path(top_down_map, trajectory)
    maps.draw_agent(top_down_map, trajectory[0], initial_angle, agent_radius_px=8)
    # print("\nDisplay the map with agent and path overlay:")
    display_map(top_down_map, save_plot=save_plot, plot_save_path=plot_save_path)


def plot_paths_on_2d_map(
    sim: habitat_sim.Simulator,
    paths: List[List[np.ndarray]],
    path_colors: List[Tuple[int, int, int]],
    meters_per_pixel: float = 0.025,
    save_plot: bool = False,
    plot_save_path: str = None,
    height=None,
):
    if height is None:
        height = sim.scene_aabb.y().min
    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )

    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
    # convert world trajectory points to maps module grid points
    trajectories = []

    for path in paths:
        trajectories.append(
            [
                maps.to_grid(
                    path_point[2],
                    path_point[0],
                    grid_dimensions,
                    pathfinder=sim.pathfinder,
                )
                for path_point in path
            ]
        )

    # draw the agent and trajectory on the map
    for idx_path, trajectory in enumerate(trajectories):
        for prev_pt, next_pt in zip(trajectory[:-1], trajectory[1:]):
            # Swapping x y
            cv2.line(
                top_down_map,
                prev_pt[::-1],
                next_pt[::-1],
                color=path_colors[idx_path],
                thickness=1,
            )  # type: ignore
    # print("\nDisplay the map with agent and path overlay:")
    display_map(top_down_map, save_plot=save_plot, plot_save_path=plot_save_path)


def save_occupancy_map(
    sim: habitat_sim.Simulator,
    meters_per_pixel: float = 0.05,
    save_dir: str = None,
    height=None,
):
    occupancy_map_path = os.path.join(save_dir, "global_occupancy_map.png")
    yaml_path = os.path.join(save_dir, "global_occupancy_map.yaml")

    scene_aabb = sim.pathfinder.get_bounds()
    # Get scene bounding box
    min_x, min_y, min_z = scene_aabb[0]  # Bottom-left corner
    max_x, max_y, max_z = scene_aabb[1]  # Top-right corner

    if height is None:
        height = min_y

    map_origin = [
        float(-max_z) - 0.5 * meters_per_pixel,
        float(-max_x) - 0.5 * meters_per_pixel,
        float(height),
    ]

    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )

    # Convert to binary occupancy grid (ROS expects 0 = free, 100 = occupied)
    occupancy_grid = np.zeros_like(top_down_map, dtype=np.uint8)
    occupancy_grid[top_down_map == maps.MAP_INVALID_POINT] = 100  # Unknown
    occupancy_grid[top_down_map == maps.MAP_BORDER_INDICATOR] = 255  # Occupied
    occupancy_grid[top_down_map == maps.MAP_VALID_POINT] = 0  # Free

    occupancy_grid = occupancy_grid.T
    occupancy_grid = np.flip(occupancy_grid, axis=1)
    occupancy_grid = 255 - occupancy_grid

    # Save occupancy map as PNG (invert colors for ROS convention)
    cv2.imwrite(occupancy_map_path, 255 - occupancy_grid)

    # Generate YAML metadata
    yaml_data = {
        "image": "global_occupancy_map.png",
        "resolution": meters_per_pixel,  # Resolution in meters/pixel
        "origin": map_origin,
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }

    # Save YAML file
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file)


def get_occupancy_map_and_config(
    sim: habitat_sim.Simulator,
    height=None,
    meters_per_pixel: float = 0.05,
) -> Tuple[np.ndarray, dict]:
    scene_aabb = sim.pathfinder.get_bounds()
    # Get scene bounding box
    min_x, min_y, min_z = scene_aabb[0]  # Bottom-left corner
    max_x, max_y, max_z = scene_aabb[1]  # Top-right corner

    if height is None:
        height = min_y

    map_origin = [
        float(-max_z) - 0.5 * meters_per_pixel,
        float(-max_x) - 0.5 * meters_per_pixel,
        float(height),
    ]

    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )

    # Convert to binary occupancy grid (ROS expects 0 = free, 100 = occupied)
    occupancy_grid = np.zeros_like(top_down_map, dtype=np.uint8)
    occupancy_grid[top_down_map == maps.MAP_INVALID_POINT] = 100  # Unknown
    occupancy_grid[top_down_map == maps.MAP_BORDER_INDICATOR] = 255  # Occupied
    occupancy_grid[top_down_map == maps.MAP_VALID_POINT] = 0  # Free

    occupancy_grid = occupancy_grid.T
    occupancy_grid = np.flip(occupancy_grid, axis=1)
    occupancy_grid = 255 - occupancy_grid

    # Generate YAML metadata
    yaml_data = {
        "image": "global_occupancy_map.png",
        "resolution": meters_per_pixel,  # Resolution in meters/pixel
        "origin": map_origin,
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    return occupancy_grid, yaml_data
