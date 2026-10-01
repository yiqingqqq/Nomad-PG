#!/usr/bin/env python3
"""Measure left/right balance in the exact GoStanford ViNT-PG sample index."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np


THRESHOLD = math.pi / 12


def local_xy(points, origin, yaw):
    points = np.asarray(points, dtype=np.float64)
    delta = points - np.asarray(origin, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack((delta[:, 0] * c + delta[:, 1] * s,
                     -delta[:, 0] * s + delta[:, 1] * c), axis=1)


def bucket(angles, weights=None):
    angles = np.asarray(angles)
    if weights is None:
        weights = np.ones_like(angles, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    counts = {
        "left": float(weights[angles > THRESHOLD].sum()),
        "right": float(weights[angles < -THRESHOLD].sum()),
        "straight": float(weights[np.abs(angles) <= THRESHOLD].sum()),
    }
    counts["left_fraction"] = counts["left"] / total if total else 0.0
    counts["right_fraction"] = counts["right"] / total if total else 0.0
    counts["straight_fraction"] = counts["straight"] / total if total else 0.0
    return counts


def sign_matrix(position_angles, yaw_angles):
    labels = ("right", "straight", "left")
    def classify(value):
        return 0 if value < -THRESHOLD else (2 if value > THRESHOLD else 1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    for position, yaw in zip(position_angles, yaw_angles):
        matrix[classify(position), classify(yaw)] += 1
    return {
        "rows_position_columns_yaw": labels,
        "matrix": matrix.tolist(),
        "same_bucket_fraction": float(np.trace(matrix) / matrix.sum()),
    }


def audit(data: Path, split: Path):
    names = [x.strip() for x in (split / "traj_names.txt").read_text().splitlines() if x.strip()]
    position_angles = [[] for _ in range(5)]
    yaw_angles = [[] for _ in range(5)]
    position_distances = [[] for _ in range(5)]
    goal_angles, goal_weights = [], []
    endpoint_goal_agreement, endpoint_goal_weights = [], []
    lengths = []
    valid_samples = 0
    invalid = []

    for name in names:
        path = data / name / "traj_data.pkl"
        try:
            with path.open("rb") as stream:
                traj = pickle.load(stream)
            positions = np.asarray(traj["position"], dtype=np.float64)
            yaws = np.asarray(traj["yaw"], dtype=np.float64).reshape(-1)
        except Exception as exc:
            invalid.append({"trajectory": name, "error": str(exc)})
            continue
        lengths.append(len(positions))
        for current in range(5, len(positions) - 5):
            valid_samples += 1
            actions = local_xy(positions[current + 1:current + 6], positions[current], yaws[current])
            endpoint_angle = math.atan2(actions[-1, 1], actions[-1, 0])
            for index, xy in enumerate(actions):
                position_angles[index].append(math.atan2(xy[1], xy[0]))
                position_distances[index].append(float(np.linalg.norm(xy)))
                yaw_delta = yaws[current + index + 1] - yaws[current]
                yaw_angles[index].append(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))

            max_goal = min(20, len(positions) - current - 1)
            goals = local_xy(
                positions[current + 1:current + max_goal + 1],
                positions[current],
                yaws[current],
            )
            weight = 1.0 / max_goal
            for xy in goals:
                goal_angle = math.atan2(xy[1], xy[0])
                goal_angles.append(goal_angle)
                goal_weights.append(weight)
                if abs(goal_angle) > THRESHOLD:
                    endpoint_goal_agreement.append(
                        float(np.sign(goal_angle) == np.sign(endpoint_angle))
                    )
                    endpoint_goal_weights.append(weight)

    return {
        "trajectories": len(names),
        "valid_trajectories": len(lengths),
        "invalid_trajectories": invalid,
        "trajectory_length": {
            "min": int(min(lengths)) if lengths else 0,
            "median": float(np.median(lengths)) if lengths else 0.0,
            "max": int(max(lengths)) if lengths else 0,
        },
        "valid_training_observations": valid_samples,
        "position_angle_by_lookahead": {
            str(i): bucket(values) for i, values in enumerate(position_angles)
        },
        "yaw_angle_by_lookahead": {
            str(i): bucket(values) for i, values in enumerate(yaw_angles)
        },
        "position_distance_m_by_lookahead": {
            str(i): {
                "median": float(np.median(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            } for i, values in enumerate(position_distances)
        },
        "position_yaw_bucket_agreement": {
            str(i): sign_matrix(position_angles[i], yaw_angles[i]) for i in range(5)
        },
        "expected_sampled_goal_angle": bucket(goal_angles, goal_weights),
        "expected_goal_endpoint_sign_agreement_noncenter": float(
            np.average(endpoint_goal_agreement, weights=endpoint_goal_weights)
        ) if endpoint_goal_weights else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        name: audit(args.data, args.splits / name) for name in ("train", "val")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
