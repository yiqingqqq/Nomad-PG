#!/usr/bin/env python3
"""Audit Habitat PointGoal and ViNT trajectory coordinate conventions."""

from __future__ import annotations

import argparse
import json
import math

import habitat
import numpy as np
import torch

from scripts.eval_vint_pg_hm3d_pointnav import build_agent, make_config
import utils.habitat_gnm_utils as habitat_gnm


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def predict(agent, observation, flip_goal_angle=False):
    agent._obs_image_context.reset()
    agent._obs_image_context.add_obs_image(observation["rgb"])
    obs_input = agent._obs_image_context.get_context()
    pg = np.array(observation["pointgoal_with_gps_compass"], copy=True)
    if flip_goal_angle:
        pg[1] *= -1
    goal_input = habitat_gnm.get_gnm_r_c_s_goal_pos_input_from_gps_compass(
        pg,
        agent._normalize,
        agent._device,
        agent._metric_waypoint_spacing,
        agent._waypoint_spacing,
        agent._len_traj_pred,
    )
    with torch.inference_mode():
        _, action = agent._model(obs_input, goal_input)
    trajectory = action.detach().cpu().numpy()[0]
    waypoint = trajectory[2]
    return float(math.atan2(waypoint[1], waypoint[0])), trajectory.tolist()


def controller_stats(rows, goal, noncenter):
    threshold = math.pi / 12
    stats = {}
    for source, xy in (("position", (0, 1)), ("yaw", (2, 3))):
        for index in range(5):
            angles = np.array(
                [math.atan2(row["trajectory"][index][xy[1]], row["trajectory"][index][xy[0]]) for row in rows]
            )
            stats[f"{source}_lookahead_{index}"] = {
                "left": int(np.sum(angles > threshold)),
                "right": int(np.sum(angles < -threshold)),
                "forward": int(np.sum(np.abs(angles) <= threshold)),
                "sign_agreement_noncenter": float(
                    np.mean(np.sign(goal[noncenter]) == np.sign(angles[noncenter]))
                ),
            }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--num-episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-waypoint-spacing", type=float, default=0.12)
    parser.add_argument("--pg-rcs", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    agent = build_agent(args)
    config = make_config(args)
    rows = []
    turn_probe = None
    with habitat.Env(config=config) as env:
        total = min(args.num_episodes, len(env.episodes))
        for index in range(total):
            obs = env.reset()
            goal_theta = float(obs["pointgoal_with_gps_compass"][1])
            pred_theta, trajectory = predict(agent, obs, False)
            pred_theta_flipped_goal, _ = predict(agent, obs, True)
            rows.append(
                {
                    "episode": str(env.current_episode.episode_id),
                    "goal_theta": goal_theta,
                    "pred_theta": pred_theta,
                    "pred_theta_flipped_goal": pred_theta_flipped_goal,
                    "trajectory": trajectory,
                }
            )
            if turn_probe is None and abs(goal_theta) > 0.35:
                after = env.step({"action": "turn_left"})
                after_theta = float(after["pointgoal_with_gps_compass"][1])
                turn_probe = {
                    "before_goal_theta": goal_theta,
                    "after_left_turn_goal_theta": after_theta,
                    "wrapped_delta": wrap(after_theta - goal_theta),
                }

    goal = np.array([x["goal_theta"] for x in rows])
    pred = np.array([x["pred_theta"] for x in rows])
    pred_flip = np.array([x["pred_theta_flipped_goal"] for x in rows])
    noncenter = np.abs(goal) > math.radians(15)
    summary = {
        "turn_probe": turn_probe,
        "num_episodes": len(rows),
        "goal_left": int(np.sum(goal > math.radians(15))),
        "goal_right": int(np.sum(goal < -math.radians(15))),
        "prediction_left": int(np.sum(pred > math.radians(15))),
        "prediction_right": int(np.sum(pred < -math.radians(15))),
        "prediction_forward": int(np.sum(np.abs(pred) <= math.radians(15))),
        "sign_agreement_noncenter": float(
            np.mean(np.sign(goal[noncenter]) == np.sign(pred[noncenter]))
        ),
        "sign_agreement_if_trajectory_y_flipped": float(
            np.mean(np.sign(goal[noncenter]) == -np.sign(pred[noncenter]))
        ),
        "sign_agreement_with_flipped_goal_input": float(
            np.mean(np.sign(goal[noncenter]) == np.sign(pred_flip[noncenter]))
        ),
        "controller_candidates": controller_stats(rows, goal, noncenter),
        "rows": rows,
    }
    if args.summary_only:
        summary.pop("rows")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
