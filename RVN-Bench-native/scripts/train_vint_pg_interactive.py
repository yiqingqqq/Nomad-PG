#!/usr/bin/env python3
"""Interactive PPO adaptation of ViNT-PG on endpoint-only HM3D PointNav.

The EfficientNet observation backbone stays frozen. PPO updates the PointGoal
encoder, fusion decoder and trajectory action head. The policy still predicts
ViNT waypoints; a differentiable categorical controller maps a look-ahead
waypoint angle to forward/left/right probabilities. STOP remains the standard
PointGoal success-radius rule used by the existing evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import habitat
import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch.utils import (
    BlockArgs,
    Conv2dStaticSamePadding,
    GlobalParams,
    MemoryEfficientSwish,
)
from torch.distributions import Categorical
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Dropout,
    Identity,
    LayerNorm,
    Linear,
    ModuleList,
    MultiheadAttention,
    ReLU,
    Sequential,
    TransformerEncoder,
    TransformerEncoderLayer,
    ZeroPad2d,
)
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler.scheduler import GradualWarmupScheduler

import utils.habitat_gnm_utils as habitat_gnm
from vint_train.models.vint.self_attention import MultiLayerDecoder, PositionalEncoding
from vint_train.models.vint.vint_pointgoal import ViNTPointGoal
from vint_train.training.train_eval_loop import load_model


ACTIONS = ("move_forward", "turn_left", "turn_right")


def register_checkpoint_safe_globals() -> None:
    torch.serialization.add_safe_globals(
        [
            ViNTPointGoal, MultiLayerDecoder, PositionalEncoding, EfficientNet,
            MBConvBlock, BlockArgs, Conv2dStaticSamePadding, GlobalParams,
            MemoryEfficientSwish, AdaptiveAvgPool2d, BatchNorm2d, Dropout,
            Identity, LayerNorm, Linear, ModuleList, MultiheadAttention,
            NonDynamicallyQuantizableLinear, ReLU, Sequential,
            TransformerEncoder, TransformerEncoderLayer, ZeroPad2d, AdamW,
            CosineAnnealingLR, GradualWarmupScheduler, OrderedDict, defaultdict,
            dict, set, np.dtype, np.core.multiarray.scalar,
            type(np.dtype(np.float64)), type(np.dtype(np.float32)),
            type(np.dtype(np.int64)), type(np.dtype(np.int32)),
            type(np.dtype(np.bool_)), torch._C._nn.gelu,
        ]
    )


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_env(args, episode_file: str, split: str, shuffle: bool, seed: int):
    config = habitat.get_config(args.task_config)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = split
        config.habitat.dataset.data_path = episode_file
        config.habitat.dataset.scenes_dir = args.scenes_dir
        config.habitat.environment.max_episode_steps = args.max_steps
        config.habitat.environment.iterator_options.shuffle = shuffle
        config.habitat.seed = seed
    return habitat.Env(config=config)


def build_model(args, device: torch.device) -> ViNTPointGoal:
    model = ViNTPointGoal(
        context_size=5,
        len_traj_pred=5,
        learn_angle=True,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=512,
        late_fusion=False,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
        pg_rcs=True,
    ).to(device)
    register_checkpoint_safe_globals()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    load_model(model, "vint_pointgoal", checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.goal_encoder, model.compress_goal_enc, model.decoder, model.action_predictor):
        for parameter in module.parameters():
            parameter.requires_grad = True
    return model


class InteractivePolicy(nn.Module):
    def __init__(self, model: ViNTPointGoal, lookahead_idx: int, temperature: float):
        super().__init__()
        self.model = model
        self.lookahead_idx = lookahead_idx
        self.temperature = temperature
        self.value_head = nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 1))

    @torch.no_grad()
    def encode_observation(self, obs_image: torch.Tensor) -> torch.Tensor:
        images = torch.split(obs_image, 3, dim=1)
        images = torch.cat(images, dim=0)
        encoding = self.model.obs_encoder.extract_features(images)
        encoding = self.model.obs_encoder._avg_pooling(encoding)
        if self.model.obs_encoder._global_params.include_top:
            encoding = encoding.flatten(start_dim=1)
            encoding = self.model.obs_encoder._dropout(encoding)
        encoding = self.model.compress_obs_enc(encoding)
        return encoding.reshape(1, self.model.context_size + 1, self.model.obs_encoding_size)

    def forward_from_tokens(self, obs_tokens: torch.Tensor, goal: torch.Tensor):
        goal_encoding = self.model.compress_goal_enc(self.model.goal_encoder(goal)).unsqueeze(1)
        representation = self.model.decoder(torch.cat((obs_tokens, goal_encoding), dim=1))
        action = self.model.action_predictor(representation)
        action = action.reshape(action.shape[0], self.model.len_trajectory_pred, self.model.num_action_params)
        xy = torch.cumsum(action[:, :, :2], dim=1)
        point = xy[:, self.lookahead_idx]
        angle = torch.atan2(point[:, 1], point[:, 0])
        threshold = math.pi / 12.0
        logits = torch.stack(
            (
                threshold - torch.abs(angle),
                angle - threshold,
                -angle - threshold,
            ),
            dim=1,
        ) / self.temperature
        return logits, self.value_head(representation).squeeze(1)


def pointgoal_tensor(observation: dict, args, device: torch.device) -> torch.Tensor:
    value = habitat_gnm.get_gnm_r_c_s_goal_pos_input_from_gps_compass(
        observation["pointgoal_with_gps_compass"], True, device,
        args.model_waypoint_spacing, 1, 5,
    )
    return value


def perturb_goal(goal: torch.Tensor, args) -> torch.Tensor:
    result = goal.clone()
    if random.random() < args.goal_dropout:
        return torch.zeros_like(result)
    if args.goal_distance_noise > 0:
        result[:, 0] *= torch.exp(torch.randn_like(result[:, 0]) * args.goal_distance_noise)
    if args.goal_angle_noise > 0:
        angle = torch.atan2(result[:, 2], result[:, 1])
        angle += torch.randn_like(angle) * args.goal_angle_noise
        result[:, 1] = torch.cos(angle)
        result[:, 2] = torch.sin(angle)
    return result


def collect_rollout(policy, env, args, device, epoch: int, output_dir: Path):
    storage = {key: [] for key in ("tokens", "goals", "actions", "log_probs", "values", "rewards", "dones")}
    episode_rows = []
    while len(storage["actions"]) < args.rollout_steps:
        observation = env.reset()
        context = habitat_gnm.ObsImageContext(6, [96, 96], device)
        previous_distance = float(env.get_metrics().get("distance_to_goal", observation["pointgoal_with_gps_compass"][0]))
        ep_reward = 0.0
        ep_collisions = 0
        ep_steps = 0
        success = 0.0
        while not env.episode_over and ep_steps < args.max_steps:
            distance = float(observation["pointgoal_with_gps_compass"][0])
            if distance <= args.success_distance:
                env.step({"action": "stop"})
                success = float(env.get_metrics().get("success", 0.0))
                if storage["rewards"]:
                    storage["rewards"][-1] += args.success_reward * success
                    storage["dones"][-1] = True
                ep_reward += args.success_reward * success
                break

            context.add_obs_image(observation["rgb"])
            tokens = policy.encode_observation(context.get_context())
            goal = perturb_goal(pointgoal_tensor(observation, args, device), args)
            with torch.no_grad():
                logits, value = policy.forward_from_tokens(tokens, goal)
                distribution = Categorical(logits=logits)
                action_index = distribution.sample()
                log_probability = distribution.log_prob(action_index)
            observation = env.step({"action": ACTIONS[int(action_index.item())]})
            metrics = env.get_metrics()
            new_distance = float(metrics.get("distance_to_goal", observation["pointgoal_with_gps_compass"][0]))
            collided = bool(env.sim.previous_step_collided)
            if math.isfinite(previous_distance) and math.isfinite(new_distance):
                progress = float(np.clip(previous_distance - new_distance, -1.0, 1.0))
            else:
                progress = 0.0
            reward = (
                args.progress_reward * progress
                + args.step_penalty
                + args.collision_penalty * float(collided)
            )
            if not math.isfinite(reward):
                reward = args.step_penalty + args.collision_penalty * float(collided)
            terminal = bool(env.episode_over)
            storage["tokens"].append(tokens.squeeze(0).cpu().to(torch.float16))
            storage["goals"].append(goal.squeeze(0).cpu())
            storage["actions"].append(int(action_index.item()))
            storage["log_probs"].append(float(log_probability.item()))
            storage["values"].append(float(value.item()))
            storage["rewards"].append(float(reward))
            storage["dones"].append(terminal)
            ep_reward += reward
            ep_collisions += int(collided)
            ep_steps += 1
            previous_distance = new_distance
        if storage["dones"]:
            storage["dones"][-1] = True
        row = {
            "epoch": epoch,
            "episode_index": len(episode_rows),
            "episode_id": str(env.current_episode.episode_id),
            "scene_id": str(env.current_episode.scene_id),
            "success": success,
            "reward": ep_reward,
            "steps": ep_steps,
            "collisions": ep_collisions,
        }
        episode_rows.append(row)
        atomic_json(
            output_dir / "episodes" / f"epoch_{epoch:03d}" / f"{len(episode_rows) - 1:05d}.json",
            row,
        )
        atomic_json(
            output_dir / "live_summary.json",
            {
                "epoch": epoch,
                "episodes_completed": len(episode_rows),
                "samples_collected": len(storage["actions"]),
                "target_samples": args.rollout_steps,
                "mean_reward": float(np.mean([x["reward"] for x in episode_rows])),
                "mean_collisions": float(np.mean([x["collisions"] for x in episode_rows])),
                "success_rate": float(np.mean([x["success"] for x in episode_rows])),
            },
        )
    return storage, episode_rows


def compute_advantages(storage, gamma: float, gae_lambda: float):
    rewards = np.asarray(storage["rewards"], dtype=np.float32)
    values = np.asarray(storage["values"], dtype=np.float32)
    dones = np.asarray(storage["dones"], dtype=np.float32)
    advantages = np.zeros_like(rewards)
    last = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = 0.0 if index == len(rewards) - 1 else values[index + 1]
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        last = delta + gamma * gae_lambda * nonterminal * last
        advantages[index] = last
    returns = advantages + values
    return advantages, returns


def ppo_update(policy, optimizer, storage, args, device):
    tokens = torch.stack(storage["tokens"])
    goals = torch.stack(storage["goals"])
    actions = torch.tensor(storage["actions"], dtype=torch.long)
    old_log_probs = torch.tensor(storage["log_probs"], dtype=torch.float32)
    advantages, returns = compute_advantages(storage, args.gamma, args.gae_lambda)
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise RuntimeError("Non-finite PPO advantages or returns after reward sanitization")
    advantages = torch.from_numpy(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = torch.from_numpy(returns)
    losses = []
    for _ in range(args.ppo_epochs):
        order = torch.randperm(len(actions))
        for start in range(0, len(actions), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_tokens = tokens[indices].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_goals = goals[indices].to(device=device, dtype=torch.float32, non_blocking=True)
            logits, values = policy.forward_from_tokens(batch_tokens, batch_goals)
            if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
                raise RuntimeError("Non-finite policy output before PPO update")
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions[indices].to(device))
            ratio = torch.exp(new_log_probs - old_log_probs[indices].to(device))
            batch_advantages = advantages[indices].to(device)
            clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * batch_advantages
            actor_loss = -torch.min(ratio * batch_advantages, clipped).mean()
            value_loss = nn.functional.mse_loss(values, returns[indices].to(device))
            entropy = distribution.entropy().mean()
            loss = actor_loss + args.value_coef * value_loss - args.entropy_coef * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable = [p for p in policy.parameters() if p.requires_grad]
            gradient_norm = nn.utils.clip_grad_norm_(
                trainable, args.max_grad_norm, error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite PPO gradient norm")
            optimizer.step()
            losses.append((loss.item(), actor_loss.item(), value_loss.item(), entropy.item()))
    return np.asarray(losses).mean(axis=0).tolist()


@torch.no_grad()
def evaluate(policy, args, device, rgb_mode: str):
    env = make_env(args, args.validation_file, "val", False, args.seed)
    rows = []
    try:
        for _ in range(min(args.eval_episodes, len(env.episodes))):
            observation = env.reset()
            context = habitat_gnm.ObsImageContext(6, [96, 96], device)
            collisions = 0
            steps = 0
            while not env.episode_over and steps < args.max_steps:
                if float(observation["pointgoal_with_gps_compass"][0]) <= args.success_distance:
                    observation = env.step({"action": "stop"})
                    break
                rgb = observation["rgb"] if rgb_mode == "normal" else np.zeros_like(observation["rgb"])
                context.add_obs_image(rgb)
                tokens = policy.encode_observation(context.get_context())
                goal = pointgoal_tensor(observation, args, device)
                logits, _ = policy.forward_from_tokens(tokens, goal)
                action_index = int(torch.argmax(logits, dim=1).item())
                observation = env.step({"action": ACTIONS[action_index]})
                collisions += int(bool(env.sim.previous_step_collided))
                steps += 1
            metrics = env.get_metrics()
            rows.append({
                "success": float(metrics.get("success", 0.0)),
                "spl": float(metrics.get("spl", 0.0)),
                "distance": float(metrics.get("distance_to_goal", 0.0)),
                "collisions": collisions,
                "steps": steps,
            })
    finally:
        env.close()
    return {
        "episodes": len(rows),
        "sr": float(np.mean([x["success"] for x in rows])),
        "spl": float(np.mean([x["spl"] for x in rows])),
        "distance": float(np.mean([x["distance"] for x in rows])),
        "collisions": float(np.mean([x["collisions"] for x in rows])),
        "steps": float(np.mean([x["steps"] for x in rows])),
    }


def curriculum_file(args, epoch: int) -> str:
    if epoch < args.direct_epochs:
        return str(Path(args.curriculum_dir) / "direct.json.gz")
    if epoch < args.direct_epochs + args.turn_epochs:
        return str(Path(args.curriculum_dir) / "turn.json.gz")
    return str(Path(args.curriculum_dir) / "mixed.json.gz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--curriculum-dir", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=270927)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--direct-epochs", type=int, default=5)
    parser.add_argument("--turn-epochs", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--goal-dropout", type=float, default=0.30)
    parser.add_argument("--goal-angle-noise", type=float, default=0.15)
    parser.add_argument("--goal-distance-noise", type=float, default=0.10)
    parser.add_argument("--progress-reward", type=float, default=0.20)
    parser.add_argument("--success-reward", type=float, default=2.50)
    parser.add_argument("--collision-penalty", type=float, default=-0.10)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--success-distance", type=float, default=0.20)
    parser.add_argument("--model-waypoint-spacing", type=float, default=0.12)
    parser.add_argument("--lookahead-idx", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.25)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device)
    policy = InteractivePolicy(model, args.lookahead_idx, args.temperature).to(device)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr)
    start_epoch = 0
    best_sr, best_spl, best_collisions = -1.0, -1.0, float("inf")
    stale = 0
    gap_stale = 0
    history = []
    latest = output_dir / "latest_train.pth"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        policy.load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_sr = float(state["best_sr"])
        best_spl = float(state["best_spl"])
        best_collisions = float(state["best_collisions"])
        stale = int(state["stale"])
        gap_stale = int(state.get("gap_stale", 0))
        history = list(state.get("history", []))

    print(json.dumps({
        "device": str(device), "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "visual_backbone_frozen": True,
        "auxiliary_head": False, "expert_trajectories": False,
        "trainable_parameters": sum(p.numel() for p in trainable),
    }, indent=2), flush=True)

    for epoch in range(start_epoch, args.max_epochs):
        episode_file = curriculum_file(args, epoch)
        env = make_env(args, episode_file, "train", True, args.seed + epoch)
        try:
            storage, train_episodes = collect_rollout(
                policy, env, args, device, epoch, output_dir
            )
        finally:
            env.close()
        losses = ppo_update(policy, optimizer, storage, args, device)
        normal = evaluate(policy, args, device, "normal")
        black = evaluate(policy, args, device, "black")
        row = {
            "epoch": epoch, "curriculum_file": episode_file,
            "samples": len(storage["actions"]),
            "train_sr": float(np.mean([x["success"] for x in train_episodes])),
            "train_reward": float(np.mean([x["reward"] for x in train_episodes])),
            "loss": losses[0], "actor_loss": losses[1],
            "value_loss": losses[2], "entropy": losses[3],
            "normal": normal, "black": black,
            "visual_sr_gap": normal["sr"] - black["sr"],
        }
        history.append(row)
        improved = (
            normal["sr"] > best_sr + 1e-9
            or (abs(normal["sr"] - best_sr) <= 1e-9 and normal["spl"] > best_spl + 1e-9)
            or (
                abs(normal["sr"] - best_sr) <= 1e-9
                and abs(normal["spl"] - best_spl) <= 1e-9
                and normal["collisions"] < best_collisions - 1e-9
            )
        )
        if improved:
            best_sr, best_spl, best_collisions = normal["sr"], normal["spl"], normal["collisions"]
            stale = 0
            atomic_torch_save(output_dir / "best_sr.pth", policy.model.state_dict())
            atomic_torch_save(output_dir / "best_spl.pth", policy.model.state_dict())
        elif epoch + 1 > args.warmup_epochs:
            stale += 1
        if epoch + 1 > args.warmup_epochs:
            if abs(row["visual_sr_gap"]) < 0.02:
                gap_stale += 1
            else:
                gap_stale = 0
        state = {
            "epoch": epoch, "policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
            "best_sr": best_sr, "best_spl": best_spl,
            "best_collisions": best_collisions, "stale": stale,
            "gap_stale": gap_stale, "history": history,
        }
        atomic_torch_save(latest, state)
        atomic_json(output_dir / "history.json", {"complete": False, "history": history})
        print(json.dumps(row), flush=True)
        if epoch + 1 >= args.warmup_epochs and (
            stale >= args.patience or gap_stale >= args.patience
        ):
            print(
                f"EARLY_STOP epoch={epoch} stale={stale} gap_stale={gap_stale}",
                flush=True,
            )
            break

    atomic_json(output_dir / "history.json", {"complete": True, "history": history})
    print("FINISHED_INTERACTIVE_TRAINING", flush=True)


if __name__ == "__main__":
    main()
