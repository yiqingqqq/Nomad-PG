#!/usr/bin/env python3
"""Evaluate an existing NoMaD-PointGoal checkpoint on standard HM3D PointNav.

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

from agents.nomad_agent import NomadAgent, load_nomad_model
import utils.habitat_action_utils as habitat_actions


ALLOWED_POLICY_KEYS = frozenset({"rgb", "pointgoal_with_gps_compass"})


def build_agent(args: argparse.Namespace) -> NomadAgent:
    agent = load_nomad_model(
        save_plot=False,
        plot_save_dir=None,
        config_file_name="nomad_pg.yaml",
        weight_path=args.checkpoint,
    )
    # Keep the goal-vector scale explicit in the benchmark record.
    agent._metric_waypoint_spacing = args.model_waypoint_spacing
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

    # NomadAgent currently hard-codes lookahead=2. Route the call through the
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


def policy_observation(observation, rgb_mode: str, rgb_history):
    policy_obs = {key: observation[key] for key in ALLOWED_POLICY_KEYS}
    if set(policy_obs) != ALLOWED_POLICY_KEYS:
        raise RuntimeError(f"Unexpected policy keys: {sorted(policy_obs)}")
    policy_obs["rgb"] = np.stack(rgb_history, axis=0)
    if rgb_mode == "black":
        policy_obs["rgb"] = np.zeros_like(policy_obs["rgb"])
    return policy_obs


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


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
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = build_agent(args)
    config = make_config(args)

    results = []
    all_latencies = []
    action_counts = Counter()
    with habitat.Env(config=config) as env:
        total = min(args.num_episodes, len(env.episodes))
        for episode_index in range(total):
            observation = env.reset()
            agent.reset()
            rgb_history = [observation["rgb"].copy() for _ in range(agent.get_obs_size())]
            episode_id = str(env.current_episode.episode_id)
            scene_id = str(env.current_episode.scene_id)
            latencies = []
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
                    clean_obs = policy_observation(observation, args.rgb_mode, rgb_history)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        action = agent.act(clean_obs, {})
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latencies.append(time.perf_counter() - start)

                action_counts[action] += 1
                observation = env.step({"action": action})
                if len(rgb_history) == agent.get_obs_size():
                    rgb_history.pop(0)
                rgb_history.append(observation["rgb"].copy())
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
            }
            results.append(row)
            print(
                f"[{episode_index + 1}/{total}] success={row['success']:.0f} "
                f"spl={row['spl']:.3f} distance={row['distance_to_goal']:.3f} "
                f"steps={steps} collisions={collisions}",
                flush=True,
            )

    summary = {
        "protocol": "HM3D PointNav-v1 val (NoMaD-PointGoal)",
        "checkpoint": os.path.abspath(args.checkpoint),
        "rgb_mode": args.rgb_mode,
        "controller": args.controller,
        "lookahead_idx": args.lookahead_idx,
        "collision_recovery": args.collision_recovery,
        "recovery_turn_steps": args.recovery_turn_steps,
        "recovery_collision_threshold": args.recovery_collision_threshold,
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
        "episodes": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
