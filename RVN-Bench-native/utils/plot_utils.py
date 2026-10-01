import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def fig_to_pil(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    return Image.fromarray(arr)


def save_debug_images(
    model_name: str,
    ep_idx: int,
    ep_steps: int,
    obs: dict,  # agent.agent.ObsType
    info: dict,  # agent.agent.InfoType
    action: str,  # habitat action
    save_dir: str,
    collision: bool,
    goal_reached: bool,
    time_out: bool,
    save_goal_rgb: bool = True,
):
    # curr_rgb = obs["rgb"][0]
    curr_rgb = obs["rgb"]
    if len(curr_rgb.shape) == 4:
        # If the shape is (num_context+1, H, W, C), remove the first dimension
        curr_rgb = curr_rgb[-1]

    if save_goal_rgb:
        goal_rgb = info["goal_rgb"]

    result_str = "running"
    if collision:
        result_str = "collision"
    elif goal_reached:
        result_str = "goal_reached"
    elif time_out:
        result_str = "time_out"

    H, W = curr_rgb.shape[:2]

    if save_goal_rgb:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax = [ax]

    ax[0].imshow(curr_rgb)
    if save_goal_rgb:
        ax[0].set_title("rgb")
    ax[0].axis("off")

    if save_goal_rgb:
        ax[1].imshow(goal_rgb)
        ax[1].set_title("goal_rgb")
        ax[1].axis("off")

    plt.suptitle(f"{model_name} ep_{ep_idx} step_{ep_steps}\n{action}\n{result_str}")

    pil_img = fig_to_pil(fig)
    plt.close(fig)

    pil_img.save(os.path.join(save_dir, f"{ep_idx}_{ep_steps}.png"))
    pil_img.save(os.path.join(save_dir, f"latest.png"))
    return


def save_context_images(
    model_name: str,
    ep_idx: int,
    ep_steps: int,
    obs: dict,  # agent.agent.ObsType
    save_dir: str,
):
    curr_rgb = obs["rgb"]
    obs_size = len(curr_rgb.shape)

    fig, ax = plt.subplots(1, obs_size, figsize=(5 * obs_size, 5))

    for i in range(0, obs_size):
        ax[i].imshow(curr_rgb[i])
        ax[i].set_title(f"rgb_{i}")
        ax[i].axis("off")

    plt.suptitle(f"{model_name} ep_{ep_idx} step_{ep_steps}")
    pil_img = fig_to_pil(fig)
    plt.close(fig)

    pil_img.save(os.path.join(save_dir, f"{ep_idx}_{ep_steps}_context.png"))
    return


def save_discrete_action_debug_plot(
    original_traj_pos: List[np.ndarray],  # List of positions in habitat frame
    disc_act_pos: List[np.ndarray],  # List of positions in habitat frame
    plot_save_dir: str,
    plot_name: str = "discrete_action_debug_plot.png",
):
    plt.figure()
    orig_pos = np.array(original_traj_pos)
    disc_act_pos = np.array(disc_act_pos)
    plt.plot(-orig_pos[:, 2], -orig_pos[:, 0], label="Original Trajectory")
    plt.plot(
        -disc_act_pos[:, 2],
        -disc_act_pos[:, 0],
        label="Trajectory by Actions",
        marker="o",
        linestyle="--",
        color="orange",
    )
    plt.scatter(-orig_pos[0, 2], -orig_pos[0, 0], color="green", label="Start")
    plt.scatter(-orig_pos[-1, 2], -orig_pos[-1, 0], color="red", label="Goal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title(plot_name)
    plt.savefig(os.path.join(plot_save_dir, f"{plot_name}.png"))
    plt.close()


def save_neg_exp_traj_debug_plot(
    neg_expt_positions: List[List[np.ndarray]],  # List of positions in habitat frame
    neg_expt_label: List[int],  # 0: negative, 1: expert
    org_traj_pos: List[np.ndarray],
    plot_save_dir: str,
    plot_name: str = "neg_exp_traj_debug_plot.png",
):
    org_traj_pos = np.array(org_traj_pos)

    plt.figure()
    plt.plot(
        -org_traj_pos[:, 2],
        -org_traj_pos[:, 0],
        color="black",
        label="Original Trajectory",
    )

    for idx, positions in enumerate(neg_expt_positions):
        positions = np.array(positions)
        traj_label = "Expert" if neg_expt_label[idx] == 1 else "Negative"
        traj_color = "blue" if neg_expt_label[idx] == 1 else "red"
        traj_marker = "+" if neg_expt_label[idx] == 1 else "x"
        plt.plot(
            -positions[:, 2],
            -positions[:, 0],
            label=traj_label,
            marker=traj_marker,
            linestyle="--",
            color=traj_color,
        )

    plt.scatter(-org_traj_pos[0, 2], -org_traj_pos[0, 0], color="green", label="Start")
    plt.scatter(-org_traj_pos[-1, 2], -org_traj_pos[-1, 0], color="red", label="Goal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title(plot_name)
    plt.savefig(os.path.join(plot_save_dir, f"{plot_name}.png"))
    plt.close()


def save_local_paths_and_goal_debug_plot(
    paths: List[List[Tuple[float, float]]],  # List of positions in world frame
    colors: List[str],
    goal_pos: Tuple[float, float],
    plot_save_dir: str,
    plot_name: str = "paths_and_goal_debug_plot",
):

    for idx, positions in enumerate(paths):
        positions = np.array(positions)
        plt.plot(
            positions[:, 0],
            positions[:, 1],
            linestyle="--",
            color=colors[idx],
        )

    plt.scatter(goal_pos[0], goal_pos[1], color="red", label="Goal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title(plot_name)
    plt.savefig(os.path.join(plot_save_dir, f"{plot_name}.png"))
    plt.close()
