import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

"""
IMPORT YOUR MODEL HERE
"""
# from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT

# from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

from vint_train.models.vint.vint_pointgoal import ViNTPointGoal
from vint_train.models.nomad.nomad_pointgoal import NoMaDPointGoal
from vint_train.models.nomad.nomad_pointgoal_vint import NoMaDPointGoal_ViNT

try:
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
except ModuleNotFoundError:
    ConditionalUnet1D = None


from vint_train.data.vint_dataset import ViNT_Dataset

from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)
from vint_train.training.pointgoal_train_eval_loop import (
    train_eval_loop_pointgoal,
    train_eval_loop_nomad_pointgoal,
)


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                dataset = ViNT_Dataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    context_type=config["context_type"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"],
                    goal_type=config["goal_type"],
                    horizontal_flip_prob=(
                        float(data_config.get("horizontal_flip_prob", 0.0))
                        if data_split_type == "train" else 0.0
                    ),
                )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset

    # Combine all datasets. Opt-in weighting prevents a longer dataset from
    # dominating joint training while preserving the total epoch size.
    dataset_lengths = [len(dataset) for dataset in train_dataset]
    train_dataset = ConcatDataset(train_dataset)
    balanced_sampler = None
    if config.get("balanced_dataset_sampling", False):
        sample_weights = torch.cat(
            [torch.full((length,), 1.0 / length, dtype=torch.double)
             for length in dataset_lengths]
        )
        balanced_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_dataset), replacement=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=balanced_sampler is None,
        sampler=balanced_sampler,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
        pin_memory=True,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )

    # Create the model
    if config["model_type"] == "gnm":
        raise NotImplementedError("GNM not implemented")
        # model = GNM(
        #     config["context_size"],
        #     config["len_traj_pred"],
        #     config["learn_angle"],
        #     config["obs_encoding_size"],
        #     config["goal_encoding_size"],
        # )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "vint_pointgoal":
        model = ViNTPointGoal(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            pg_rcs=config.get("pg_rcs", False),
        )
    elif config["model_type"] == "nomad_pointgoal":
        if ConditionalUnet1D is None:
            raise ModuleNotFoundError(
                "diffusion_policy is required for model_type=nomad_pointgoal"
            )
        vision_encoder = NoMaDPointGoal_ViNT(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            pg_rcs=config.get("pg_rcs", False),
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)

        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = NoMaDPointGoal(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    elif config["model_type"] == "nomad":
        if ConditionalUnet1D is None:
            raise ModuleNotFoundError(
                "diffusion_policy is required for model_type=nomad"
            )
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib":
            raise NotImplementedError("ViB not implemented")
            # vision_encoder = ViB(
            #     obs_encoding_size=config["encoding_size"],
            #     context_size=config["context_size"],
            #     mha_num_attention_heads=config["mha_num_attention_heads"],
            #     mha_num_attention_layers=config["mha_num_attention_layers"],
            #     mha_ff_dim_factor=config["mha_ff_dim_factor"],
            # )
            # vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit":
            raise NotImplementedError("ViT not implemented")
            # vision_encoder = ViT(
            #     obs_encoding_size=config["encoding_size"],
            #     context_size=config["context_size"],
            #     image_size=config["image_size"],
            #     patch_size=config["patch_size"],
            #     mha_num_attention_heads=config["mha_num_attention_heads"],
            #     mha_num_attention_layers=config["mha_num_attention_layers"],
            # )
            # vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if "pretrained_checkpoint" in config:
        print("Loading pretrained model from", config["pretrained_checkpoint"])
        checkpoint = torch.load(
            config["pretrained_checkpoint"], map_location="cpu", weights_only=False
        )
        load_model(model, config["model_type"], checkpoint)
    # ---- Staged training: module freezing / grouped learning rates ----
    # Apply module freezing before the optimizer is built so frozen parameters
    # are excluded (requires_grad=False). Used by the staged HuRoN protocol
    # (Stage 1 freezes the visual encoder + transformer, training only the
    # PointGoal MLP and prediction heads).
    def resolve_module(root, dotted_name):
        module = root
        for part in dotted_name.split("."):
            module = getattr(module, part, None)
            if module is None:
                raise ValueError(
                    f"module path '{dotted_name}' has no component '{part}'"
                )
        return module

    if config.get("train_only_modules"):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name in config["train_only_modules"]:
            module = resolve_module(model, name)
            trainable = 0
            for parameter in module.parameters():
                parameter.requires_grad = True
                trainable += parameter.numel()
            print(f"[train-only] {name}: {trainable/1e6:.2f}M params trainable")

    if config.get("freeze_modules"):
        for name in config["freeze_modules"]:
            module = resolve_module(model, name)
            frozen = 0
            for p in module.parameters():
                p.requires_grad = False
                frozen += p.numel()
            print(f"[freeze] {name}: {frozen/1e6:.2f}M params frozen")

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()

    # Build parameter groups from trainable parameters only. module_lr_multipliers
    # scales the learning rate of specific modules (e.g. a lower LR for the
    # transformer decoder during Stage 2/3 fusion); unlisted modules use the
    # base learning rate.
    param_groups = []
    covered_ids = set()
    for name, mult in config.get("module_lr_multipliers", {}).items():
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"module_lr_multipliers: module '{name}' not found on model")
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr * float(mult)})
            covered_ids.update(id(p) for p in params)
            print(f"[lr-group] {name}: lr={lr * float(mult):.2e}, {len(params)} params")
    remaining = [
        p for p in model.parameters() if p.requires_grad and id(p) not in covered_ids
    ]
    if remaining:
        param_groups.append({"params": remaining, "lr": lr})
    if not param_groups:
        raise ValueError("No trainable parameters: all modules are frozen")

    if config["optimizer"] == "adam":
        optimizer = Adam(param_groups, lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        if "weight_decay" in config:
            print("weight_deacy: ", config["weight_decay"])
            optimizer = AdamW(
                param_groups, lr=lr, weight_decay=config["weight_decay"]
            )
        else:
            optimizer = AdamW(param_groups, lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.0,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        elif config["scheduler"] == "const" and config["warmup"]:
            print("Using Const lr after warmup")
            scheduler = None
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config or "load_checkpoint" in config:
        if "load_checkpoint" in config:
            latest_path = config["load_checkpoint"]
        else:
            load_project_folder = os.path.join("logs", config["load_run"])
            latest_path = os.path.join(load_project_folder, "latest.pth")
        print("Loading training checkpoint from", latest_path)
        latest_checkpoint = torch.load(
            latest_path
        )  # f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, config["model_type"], latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config or "load_checkpoint" in config:
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    if config["model_type"] == "vint" or config["model_type"] == "gnm":
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
    elif config["model_type"] == "vint_pointgoal":
        train_eval_loop_pointgoal(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            len_traj_pred=config["len_traj_pred"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            goal_dropout_prob=float(config.get("goal_dropout_prob", 0.0)),
            goal_angle_noise_std=float(config.get("goal_angle_noise_std", 0.0)),
            goal_distance_noise_std=float(config.get("goal_distance_noise_std", 0.0)),
            vision_aux_weight=float(config.get("vision_aux_weight", 0.0)),
        )
    elif config["model_type"] == "nomad_pointgoal":
        train_eval_loop_nomad_pointgoal(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            len_traj_pred=config["len_traj_pred"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
            max_train_batches=config.get("max_train_batches"),
            max_eval_batches=config.get("max_eval_batches"),
        )
    else:
        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/nomad_sq.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        config.get("log_root", "logs"), config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            # entity="gnmv2", # TODO: change this to your wandb entity
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
