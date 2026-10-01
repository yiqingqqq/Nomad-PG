#!/usr/bin/env python3
"""Evaluate an existing ViNT-PointGoal checkpoint on standard HM3D PointNav.

Policy inputs are restricted to RGB history and the benchmark-provided
``pointgoal_with_gps_compass`` sensor.  Ground-truth pose, Pathfinder output,
and evaluator measurements are never passed to the policy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import habitat
import numpy as np
import torch
from collections import OrderedDict, defaultdict
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch.utils import (
    BlockArgs,
    Conv2dStaticSamePadding,
    GlobalParams,
    MemoryEfficientSwish,
)
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

from agents.vint_agent import VintAgent
import utils.habitat_action_utils as habitat_actions
from vint_train.models.vint.vint_pointgoal import (
    ViNTPointGoal,
)
from vint_train.models.vint.self_attention import MultiLayerDecoder, PositionalEncoding
from vint_train.training.train_eval_loop import load_model


ALLOWED_POLICY_KEYS = frozenset({"rgb", "pointgoal_with_gps_compass"})


def build_agent(args: argparse.Namespace) -> VintAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        pg_rcs=args.pg_rcs,
    ).to(device)
    torch.serialization.add_safe_globals(
        [
            ViNTPointGoal,
            MultiLayerDecoder,
            PositionalEncoding,
            EfficientNet,
            MBConvBlock,
            BlockArgs,
            Conv2dStaticSamePadding,
            GlobalParams,
            MemoryEfficientSwish,
            AdaptiveAvgPool2d,
            BatchNorm2d,
            Dropout,
            Identity,
            LayerNorm,
            Linear,
            ModuleList,
            MultiheadAttention,
            NonDynamicallyQuantizableLinear,
            ReLU,
            Sequential,
            TransformerEncoder,
            TransformerEncoderLayer,
            ZeroPad2d,
            AdamW,
            CosineAnnealingLR,
            GradualWarmupScheduler,
            OrderedDict,
            defaultdict,
            dict,
            set,
            np.dtype,
            np.core.multiarray.scalar,
            type(np.dtype(np.float64)),
            type(np.dtype(np.float32)),
            type(np.dtype(np.int64)),
            type(np.dtype(np.int32)),
            type(np.dtype(np.bool_)),
            torch._C._nn.gelu,
        ]
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    load_model(model, "vint_pointgoal", checkpoint)
    model.eval()
    agent = VintAgent(
        model=model,
        device=device,
        pred_horizon=5,
        model_type="vint_pointgoal",
        action_dim=2,
        obs_size=6,
        input_img_size=[96, 96],
        metric_waypoint_spacing=args.model_waypoint_spacing,
        waypoint_spacing=1,
        normalize=True,
        len_traj_pred=5,
        save_plot=False,
        dataset_name="habitat_hm3d",
        pg_rcs=args.pg_rcs,
    )
    base_trajectory_controller = (
        habitat_actions.get_discrete_control_output_from_trajectory_action_wo_stop
    )

    def selected_controller(action_traj, **kwargs):
        if args.controller == "yaw":
            return habitat_actions.get_discrete_control_output_from_yaw(
                action_traj, lookahead_point_idx=args.lookahead_idx
            )
        return base_trajectory_controller(
            action_traj, lookahead_point_idx=args.lookahead_idx
        )

    # VintAgent currently hard-codes lookahead=2. Route the call through the
    # explicitly selected evaluation setting without changing shared code.
    habitat_actions.get_discrete_control_output_from_trajectory_action_wo_stop = (
        selected_controller
    )
    return agent


def make_config(args: argparse.Namespace):
    config = habitat.get_config(args.task_config)
    # Assign after Hydra parsing because the official data path contains the
    # literal ``{split}`` placeholder, which is not valid override syntax.
    with habitat.config.read_write(config):
        config.habitat.dataset.split = "val"
        config.habitat.dataset.data_path = args.episode_file or (
            f"{args.episodes}/{{split}}/{{split}}.json.gz"
        )
        config.habitat.dataset.scenes_dir = args.scenes_dir
        config.habitat.environment.max_episode_steps = args.max_steps
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.seed = args.seed
    return config


def policy_observation(observation, rgb_mode: str):
    policy_obs = {key: observation[key] for key in ALLOWED_POLICY_KEYS}
    if set(policy_obs) != ALLOWED_POLICY_KEYS:
        raise RuntimeError(f"Unexpected policy keys: {sorted(policy_obs)}")
    if rgb_mode == "black":
        policy_obs["rgb"] = np.zeros_like(policy_obs["rgb"])
    return policy_obs


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def build_summary(args, results, all_latencies, action_counts):
    return {
        "protocol": "HM3D PointNav-v1 val",
        "checkpoint": os.path.abspath(args.checkpoint),
        "rgb_mode": args.rgb_mode,
        "controller": args.controller,
        "lookahead_idx": args.lookahead_idx,
        "collision_recovery": args.collision_recovery,
        "recovery_turn_steps": args.recovery_turn_steps,
        "recovery_collision_threshold": args.recovery_collision_threshold,
        "requested_episodes": args.num_episodes,
        "num_episodes": len(results),
        "success": float(np.mean([x["success"] for x in results])) if results else 0.0,
        "spl": float(np.mean([x["spl"] for x in results])) if results else 0.0,
        "mean_final_distance": float(np.mean([x["distance_to_goal"] for x in results])) if results else 0.0,
        "mean_collisions": float(np.mean([x["collisions"] for x in results])) if results else 0.0,
        "mean_recovery_actions": float(np.mean([x["recovery_actions"] for x in results])) if results else 0.0,
        "mean_steps": float(np.mean([x["steps"] for x in results])) if results else 0.0,
        "mean_latency_ms": 1000.0 * (sum(all_latencies) / len(all_latencies) if all_latencies else 0.0),
        "p95_latency_ms": 1000.0 * percentile(all_latencies, 95),
        "action_counts": dict(action_counts),
        "policy_inputs": sorted(ALLOWED_POLICY_KEYS),
        "complete": len(results) >= args.num_episodes,
        "episodes": results,
    }


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument(
        "--episode-file",
        help="Optional fixed PointNav JSON/JSON.GZ episode list for paired evaluation.",
    )
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--success-distance", type=float, default=0.2)
    parser.add_argument("--model-waypoint-spacing", type=float, default=0.12)
    parser.add_argument("--lookahead-idx", type=int, default=2)
    parser.add_argument("--controller", choices=("trajectory", "yaw"), default="trajectory")
    parser.add_argument(
        "--collision-recovery",
        choices=("none", "goal_turn", "alternate"),
        default="none",
        help="Controller-only response after a collided forward action.",
    )
    parser.add_argument(
        "--recovery-turn-steps",
        type=int,
        default=0,
        help="Number of discrete turns inserted after a forward collision.",
    )
    parser.add_argument(
        "--recovery-collision-threshold",
        type=int,
        default=1,
        help="Consecutive forward collisions required before recovery turns.",
    )
    parser.add_argument("--rgb-mode", choices=("normal", "black"), default="normal")
    parser.add_argument("--pg-rcs", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed episode rows already present in --output.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = build_agent(args)
    config = make_config(args)

    output = Path(args.output)
    results = []
    all_latencies = []
    action_counts = Counter()
    completed_keys = set()
    if args.resume and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if os.path.abspath(args.checkpoint) != previous.get("checkpoint"):
            raise ValueError("Resume output belongs to a different checkpoint")
        results = list(previous.get("episodes", []))
        completed_keys = {
            (str(row["scene_id"]), str(row["episode_id"])) for row in results
        }
        for row in results:
            all_latencies.extend(
                value / 1000.0 for value in row.get("latencies_ms", [])
            )
            action_counts.update(row.get("action_counts", {}))
        print(f"Resuming with {len(results)} completed episodes", flush=True)

    with habitat.Env(config=config) as env:
        total = min(args.num_episodes, len(env.episodes))
        for episode_index in range(total):
            observation = env.reset()
            agent.reset()
            episode_id = str(env.current_episode.episode_id)
            scene_id = str(env.current_episode.scene_id)
            episode_key = (scene_id, episode_id)
            if episode_key in completed_keys:
                print(
                    f"[{episode_index + 1}/{total}] resume-skip "
                    f"episode={episode_id}",
                    flush=True,
                )
                continue
            latencies = []
            episode_action_counts = Counter()
            collisions = 0
            steps = 0
            recovery_remaining = 0
            alternate_recovery_left = True
            recovery_actions = 0
            forward_collision_streak = 0

            while not env.episode_over:
                pg = observation["pointgoal_with_gps_compass"]
                if float(pg[0]) <= args.success_distance:
                    action = "stop"
                elif recovery_remaining > 0:
                    if args.collision_recovery == "goal_turn":
                        action = "turn_left" if float(pg[1]) >= 0 else "turn_right"
                    elif args.collision_recovery == "alternate":
                        action = "turn_left" if alternate_recovery_left else "turn_right"
                    else:
                        raise RuntimeError("Recovery queue used with recovery disabled")
                    recovery_remaining -= 1
                    recovery_actions += 1
                else:
                    clean_obs = policy_observation(observation, args.rgb_mode)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        action = agent.act(clean_obs, {})
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latencies.append(time.perf_counter() - start)

                action_counts[action] += 1
                episode_action_counts[action] += 1
                observation = env.step({"action": action})
                steps += 1
                collided = bool(env.sim.previous_step_collided)
                collisions += int(collided)
                if action == "move_forward":
                    forward_collision_streak = forward_collision_streak + 1 if collided else 0
                    if (
                        args.collision_recovery != "none"
                        and args.recovery_turn_steps > 0
                        and forward_collision_streak >= args.recovery_collision_threshold
                    ):
                        recovery_remaining = args.recovery_turn_steps
                        alternate_recovery_left = not alternate_recovery_left
                        forward_collision_streak = 0

            metrics = env.get_metrics()
            all_latencies.extend(latencies)
            row = {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "scene_id": scene_id,
                "success": float(metrics.get("success", 0.0)),
                "spl": float(metrics.get("spl", 0.0)),
                "distance_to_goal": float(metrics.get("distance_to_goal", 0.0)),
                "steps": steps,
                "collisions": collisions,
                "recovery_actions": recovery_actions,
                "mean_latency_ms": 1000.0 * (sum(latencies) / len(latencies) if latencies else 0.0),
                "latencies_ms": [1000.0 * value for value in latencies],
                "action_counts": dict(episode_action_counts),
            }
            results.append(row)
            completed_keys.add(episode_key)
            atomic_write_json(
                output,
                build_summary(args, results, all_latencies, action_counts),
            )
            print(
                f"[{episode_index + 1}/{total}] success={row['success']:.0f} "
                f"spl={row['spl']:.3f} distance={row['distance_to_goal']:.3f} "
                f"steps={steps} collisions={collisions}",
                flush=True,
            )

    summary = build_summary(args, results, all_latencies, action_counts)
    summary["complete"] = len(results) >= total
    atomic_write_json(output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
