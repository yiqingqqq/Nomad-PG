import os
import sys
from typing import Tuple, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as TF
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel


"""
IMPORT YOUR MODEL HERE
"""

# add python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from diffusion_policy.model.diffusion.conditional_unet1d import (
    ConditionalUnet1D,
)

from models.gnms_levin.train.vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from models.gnms_levin.train.vint_train.models.nomad.nomad_vint import (
    NoMaD_ViNT,
    replace_bn_with_gn,
)
from models.gnms_levin.train.vint_train.models.nomad.nomad_pointgoal import (
    NoMaDPointGoal, NoMaDPointGoalCritic
)
from models.gnms_levin.train.vint_train.models.nomad.nomad_pointgoal_vint import (
    NoMaDPointGoal_ViNT,
)
from models.gnms_levin.train.vint_train.visualizing.action_utils import (
    plot_trajs_and_points,
    plot_trajs_and_points_on_image,
)


RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
DARK_GREEN = np.array([0, 0.5, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])

ACTION_STATS = {
    "min": np.array([-2.5, -4]),  # [min_dx, min_dy]
    "max": np.array([5, 4]),  # [max_dx, max_dy]
}


def get_transform():
    transform = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(transform)


def transpose_to_robot_frame(
    robot_position: Tuple[float, float],
    robot_yaw: float,
    global_position: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Returns
    -------
    tuple[float, float]
        x, y of global in robot frame
    """
    x, y = robot_position
    x_global, y_global = global_position

    x_rel = x_global - x
    y_rel = y_global - y

    x_rel_rot = x_rel * np.cos(robot_yaw) + y_rel * np.sin(robot_yaw)
    y_rel_rot = -x_rel * np.sin(robot_yaw) + y_rel * np.cos(robot_yaw)

    return x_rel_rot, y_rel_rot


def resize_and_aspect_crop(
    img: Image.Image,
    image_resize_size: Tuple[int, int],
    aspect_ratio: float = 4 / 3,
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img


def plot_raw_images(images, traj_data, cols=5):
    # Calculate the number of rows needed
    rows = (len(images) + cols - 1) // cols

    # Create the subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.ravel()  # Flatten to easily iterate over axes

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis("off")  # Hide axes for cleaner look
        # Add the title of the image
        axes[i].set_title(
            f"pos: {traj_data['position'][i][0]:.2f}, {traj_data['position'][i][1]:.2f}\nyaw: {traj_data['yaw'][i]:.2f}"
        )
    # Turn off any remaining empty subplots
    for j in range(i + 1, rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = ndeltas.detach().cpu().numpy()
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return torch.from_numpy(actions).float().to(device)


def get_diffusion_output(
    global_cond: torch.Tensor,
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    pred_horizon: int,
    action_dim: int,
    device: torch.device,
    return_chain: bool = False,
):
    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(global_cond), pred_horizon, action_dim), device=device
    )
    diffusion_output = noisy_diffusion_output

    if return_chain:
        chain = [diffusion_output.clone()]

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=global_cond,
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=diffusion_output
        ).prev_sample

        if return_chain:
            chain.append(diffusion_output.clone())

    if return_chain:
        chain = torch.stack(chain, dim=1)  # (B, T, action_dim)
        diffusion_output = chain[:, -1, :]
        return {"diffusion_output": diffusion_output, "chain": chain}


    return diffusion_output


def get_action_from_visual_encoding(
    obs_cond: torch.Tensor,
    obsgoal_cond: torch.Tensor,
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    pred_horizon: int,
    action_dim: int,
    device: torch.device,
    return_chain: bool = False,
):
    
    """
    return_chain: returns the diffusion output for both unconditioned and goal-conditioned
    """
    unconditioned_diffusion_output = get_diffusion_output(
        obs_cond, model, noise_scheduler, pred_horizon, action_dim, device, return_chain
    )
    goalconditioned_diffusion_output = get_diffusion_output(
        obsgoal_cond, model, noise_scheduler, pred_horizon, action_dim, device, return_chain
    )

    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_distance = model(
        "dist_pred_net", obsgoal_cond=obsgoal_cond
    )

    if return_chain:
        uc_actions = get_action(
            unconditioned_diffusion_output["diffusion_output"], ACTION_STATS
        )
        gc_actions = get_action(
            goalconditioned_diffusion_output["diffusion_output"], ACTION_STATS
        )
        uc_chain = unconditioned_diffusion_output["chain"]
        gc_chain = goalconditioned_diffusion_output["chain"]

        return {
            "uc_actions": uc_actions,
            "gc_actions": gc_actions,
            "gc_distance": gc_distance,
            "uc_chain": uc_chain,
            "gc_chain": gc_chain,
        }

    uc_actions = get_action(unconditioned_diffusion_output, ACTION_STATS)
    gc_actions = get_action(goalconditioned_diffusion_output, ACTION_STATS)
    return {
        "uc_actions": uc_actions,
        "gc_actions": gc_actions,
        "gc_distance": gc_distance,
    }


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    return_chain: bool = False,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=goal_mask,
    )
    # obs_cond = obs_cond.flatten(start_dim=1)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=no_mask,
    )
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    return get_action_from_visual_encoding(
        obs_cond, obsgoal_cond, model, noise_scheduler, pred_horizon, action_dim, device, return_chain
    )


def pointgoal_model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    return_chain: bool = False,
):
    B = batch_obs_images.shape[0]
    goal_mask = torch.ones((B,)).long().to(device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_pos=batch_goal_pos,
        input_goal_mask=goal_mask,
    )
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((B,)).long().to(device)
    obsgoal_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_pos=batch_goal_pos,
        input_goal_mask=no_mask,
    )
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    return get_action_from_visual_encoding(
        obs_cond, obsgoal_cond, model, noise_scheduler, pred_horizon, action_dim, device, return_chain
    )

def pointgoal_model_logprob_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    chains: torch.Tensor,
    device: torch.device,
):
    """
    Input:
        chains: (B, K+1, T, 2)
    -------
    returns: 
        (B, K, T, 2)
    """
    
    B = batch_obs_images.shape[0]
    goal_mask = torch.ones((B,)).long().to(device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_pos=batch_goal_pos,
        input_goal_mask=goal_mask,
    )
    obs_cond = obs_cond.repeat_interleave(1, dim=0)

    chain_prevs = chains[:, :-1, :]  # (B, K, T, 2)
    chain_nexts = chains[:, 1:, :]  # (B, K, T, 2)

    log_probs = []
    for k in noise_scheduler.timesteps[:]:
        noise_pred = model(
            "noise_pred_net",
            sample=chain_prevs[:,k],
            timestep=k.unsqueeze(-1).repeat(chain_prevs.shape[0]).to(device),
            global_cond=obs_cond,
        )
        mean_var_dict = noise_scheduler.return_mean_var(
            model_output=noise_pred,
            timestep=k,
            sample=chain_prevs[:,k],
        )
        mean = mean_var_dict["mean"]
        var = mean_var_dict["variance"].to(device)
        normal = torch.distributions.Normal(mean, var.sqrt())
        log_prob = normal.log_prob(chain_nexts[:,k])
        log_probs.append(log_prob.unsqueeze(1))  # (B, 1, T, 2)
    log_probs = torch.cat(log_probs, dim=1)  # (B, K, T, 2)
    return log_probs


def pointgoal_model_subsample_logprob_and_entropy_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    chains_prev: torch.Tensor,
    chains_next: torch.Tensor,
    prev_to_next_timesteps: torch.Tensor,
    device: torch.device,
):
    B = batch_obs_images.shape[0]
    goal_mask = torch.ones((B,)).long().to(device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_pos=batch_goal_pos,
        input_goal_mask=goal_mask,
    )
    obs_cond = obs_cond.repeat_interleave(1, dim=0)

    means = []
    vars = []
    noise_preds = model(
        "noise_pred_net",
        sample=chains_prev,
        timestep=prev_to_next_timesteps.to(device),
        global_cond=obs_cond,
    )  # (B, T, 2)
    for batch_idx in range(B):
        noise_pred = noise_preds[batch_idx].unsqueeze(0)  # (1, T, 2)
        mean_var_dict = noise_scheduler.return_mean_var(
            model_output=noise_pred,
            timestep=prev_to_next_timesteps[batch_idx],
            sample=chains_prev[batch_idx],
        )
        means.append(mean_var_dict["mean"])
        vars.append(mean_var_dict["variance"].to(device))
    means = torch.cat(means, dim=0)  # (B, T, 2)
    vars = torch.asarray(vars, device=device).unsqueeze(-1).unsqueeze(-1)  # (B, T, 2)
    normal = torch.distributions.Normal(means, vars.sqrt())
    log_probs = normal.log_prob(chains_next)  # (B, T, 2)
    entropies = normal.entropy()  # (B, T, 2)

    return log_probs, entropies
    


def pointgoal_model_critic_output(
    model: nn.Module,
    batch_obs_images: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
):
    B = batch_obs_images.shape[0]
    goal_mask = torch.ones((B,)).long().to(device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_pos=batch_goal_pos,
        input_goal_mask=goal_mask,
    )
    obs_cond = obs_cond.repeat_interleave(1, dim=0)

    no_mask = torch.zeros((B,)).long().to(device)
    obsgoal_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_pos=batch_goal_pos,
        input_goal_mask=no_mask,
    )
    obsgoal_cond = obsgoal_cond.repeat_interleave(1, dim=0)

    return model("dense_network", obs_cond=obs_cond, obsgoal_cond=obsgoal_cond)

def set_cuda_visible_devices_and_get_device(config) -> Tuple[List[int], torch.device]:
    """
    Returns
    -------
    gpu_ids : list of int
        list of GPU ids
    device : torch.device
    """
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            gpu_ids = [0]
        elif type(config["gpu_ids"]) == int:
            gpu_ids = [config["gpu_ids"]]
        else:
            gpu_ids = config["gpu_ids"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpu_ids])
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = gpu_ids[0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    return gpu_ids, device


def get_nomad_model(
    model_type,
    encoding_size,
    context_size,
    mha_num_attention_heads,
    mha_num_attention_layers,
    mha_ff_dim_factor,
    down_dims,
    cond_predict_scale,
    num_diffusion_iters,
    action_dim,
    pg_rcs,
):
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=encoding_size,
        down_dims=down_dims,
        cond_predict_scale=cond_predict_scale,
    )
    dist_pred_network = DenseNetwork(embedding_dim=encoding_size)

    if model_type == "nomad":
        print("Loading NoMaD visual encoder")
        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=encoding_size,
            context_size=context_size,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=mha_ff_dim_factor,
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    elif model_type == "nomad_pointgoal":
        print("Loading NoMaD-PointGoal visual encoder")
        vision_encoder = NoMaDPointGoal_ViNT(
            obs_encoding_size=encoding_size,
            context_size=context_size,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=mha_ff_dim_factor,
            pg_rcs=pg_rcs,
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)

        model = NoMaDPointGoal(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise NotImplementedError

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    return model, noise_scheduler

def get_nomad_critic(
    model_type,
    encoding_size,
    context_size,
    mha_num_attention_heads,
    mha_num_attention_layers,
    mha_ff_dim_factor
):
    value_pred_network = DenseNetwork(embedding_dim=encoding_size)
    if model_type == "nomad_pointgoal":
        print("Loading NoMaD-PointGoal critic visual encoder")
        vision_encoder = NoMaDPointGoal_ViNT(
            obs_encoding_size=encoding_size,
            context_size=context_size,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=mha_ff_dim_factor,
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)

        model = NoMaDPointGoalCritic(
            vision_encoder=vision_encoder,
            dense_network=value_pred_network,
        )
    else:
        raise NotImplementedError(
            f"Model type {model_type} not supported for critic"
        )
    
    return model
    


def load_nomad_model(
    model_type,
    encoding_size,
    context_size,
    mha_num_attention_heads,
    mha_num_attention_layers,
    mha_ff_dim_factor,
    down_dims,
    cond_predict_scale,
    num_diffusion_iters,
    action_dim,
    weight_path,
    gpu_ids,
    device,
    power=0.75,
    pg_rcs=True,
):
    model, noise_scheduler = get_nomad_model(
        model_type=model_type,
        encoding_size=encoding_size,
        context_size=context_size,
        mha_num_attention_heads=mha_num_attention_heads,
        mha_num_attention_layers=mha_num_attention_layers,
        mha_ff_dim_factor=mha_ff_dim_factor,
        down_dims=down_dims,
        cond_predict_scale=cond_predict_scale,
        num_diffusion_iters=num_diffusion_iters,
        action_dim=action_dim,
        pg_rcs=pg_rcs
    )

    latest_checkpoint = torch.load(weight_path)
    model.load_state_dict(latest_checkpoint, strict=False)

    # Multi-GPU
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    ema_model = EMAModel(model=model, power=power)
    ema_model_avg = ema_model.averaged_model
    ema_model_avg = ema_model_avg.train()  # set to train mode
    for param in ema_model_avg.parameters():
        param.requires_grad = True

    return ema_model_avg, noise_scheduler


def load_nomad_critic(
    model_type,
    encoding_size,
    context_size,
    mha_num_attention_heads,
    mha_num_attention_layers,
    mha_ff_dim_factor,
    weight_path,
    gpu_ids,
    device,
):
    model = get_nomad_critic(
        model_type=model_type,
        encoding_size=encoding_size,
        context_size=context_size,
        mha_num_attention_heads=mha_num_attention_heads,
        mha_num_attention_layers=mha_num_attention_layers,
        mha_ff_dim_factor=mha_ff_dim_factor
    )

    latest_checkpoint = torch.load(weight_path)
    model.load_state_dict(latest_checkpoint, strict=False)

    # Multi-GPU
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    return model


def plot_trajs_and_label(
    gc_actions_list,
    uc_actions_list,
    label_waypoints,
    num_samples,
    relative_goal_pos,
    obs_img_to_plot,
    goal_img_to_plot,
    dataset_name,
    save_plot,
    plot_save_path,
    metric_waypoint_spacing=None,
) -> list[np.ndarray]:
    pass

    fig, ax = plt.subplots(1, 3)
    start_pos = np.array([0, 0])

    if metric_waypoint_spacing:
        trajs = [
            *gc_actions_list * metric_waypoint_spacing,
            *uc_actions_list * metric_waypoint_spacing,
            label_waypoints,
        ]
    else:
        trajs = [*gc_actions_list, *uc_actions_list, label_waypoints]

    traj_colors = [GREEN] * num_samples + [RED] * num_samples + [MAGENTA]
    traj_labels = (
        ["goal conditioned"] * num_samples
        + ["unconditioned"] * num_samples
        + ["ground truth"]
    )

    plot_trajs_and_points(
        ax=ax[0],
        list_trajs=trajs,
        list_points=[start_pos, relative_goal_pos],
        traj_colors=traj_colors,
        point_colors=[GREEN, RED],
        traj_labels=traj_labels,
    )
    plot_trajs_and_points_on_image(
        ax=ax[1],
        img=obs_img_to_plot,
        dataset_name=dataset_name,  # NOTE: other data_config has no camera config
        list_trajs=trajs,
        list_points=[start_pos, relative_goal_pos],
        traj_colors=traj_colors,
        point_colors=[GREEN, RED],
    )
    ax[2].imshow(goal_img_to_plot)

    fig.set_size_inches(18.5, 10.5)
    ax[0].set_title(f"Action Prediction")
    ax[1].set_title(f"Observation")
    ax[2].set_title(f"Goal")

    if save_plot:
        dir_name = os.path.dirname(plot_save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(plot_save_path)
    else:
        plt.show()


def plot_trajs_wo_label(
    gc_actions_list,
    uc_actions_list,
    obs_img_to_plot,
    goal_img_to_plot,
    dataset_name,
    save_plot,
    plot_save_path,
    action_traj_idx=0,
    fig_title: str = "",
    metric_waypoint_spacing=None,
) -> list[np.ndarray]:
    pass

    fig, ax = plt.subplots(1, 3)
    start_pos = np.array([0, 0])

    if metric_waypoint_spacing:
        trajs = []
        if len(gc_actions_list) > 0:
            trajs += [
                *gc_actions_list * metric_waypoint_spacing,
            ]
        if len(uc_actions_list) > 0:
            trajs += [
                *uc_actions_list * metric_waypoint_spacing,
            ]
    else:
        trajs = [*gc_actions_list, *uc_actions_list]

    traj_colors = [BLUE] * len(gc_actions_list) + [RED] * len(uc_actions_list)
    traj_colors[action_traj_idx] = GREEN
    traj_labels = ["goal-conditioned"] * len(gc_actions_list) + ["unconditioned"] * len(
        uc_actions_list
    )
    traj_labels[action_traj_idx] = "selected"

    plot_trajs_and_points(
        ax=ax[0],
        list_trajs=trajs,
        list_points=[start_pos],
        traj_colors=traj_colors,
        point_colors=[DARK_GREEN],
        point_labels=["start"],
        traj_labels=traj_labels,
    )
    plot_trajs_and_points_on_image(
        ax=ax[1],
        img=obs_img_to_plot,
        dataset_name=dataset_name,  # NOTE: other data_config has no camera config
        list_trajs=trajs,
        list_points=[],
        traj_colors=traj_colors,
        point_colors=[],
    )
    ax[2].imshow(goal_img_to_plot)

    fig.set_size_inches(18, 6)
    ax[0].set_title(f"Action Prediction")
    ax[1].set_title(f"Observation")
    ax[2].set_title(f"Goal")

    fig.suptitle(fig_title)

    if save_plot:
        dir_name = os.path.dirname(plot_save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(plot_save_path)
        plt.close()
    else:
        plt.show()
