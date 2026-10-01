#!/usr/bin/env python3
"""Compare two HM3D PointNav evaluation JSON files episode by episode."""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def key(row):
    return (row["scene_id"], str(row["episode_id"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline, candidate = load(args.baseline), load(args.candidate)
    a, b = baseline["episodes"], candidate["episodes"]
    a_by_key = {key(row): row for row in a}
    b_by_key = {key(row): row for row in b}
    if set(a_by_key) != set(b_by_key):
        raise SystemExit("Episode sets differ; comparison is not paired.")
    rows = []
    for episode_key in sorted(a_by_key):
        old, new = a_by_key[episode_key], b_by_key[episode_key]
        rows.append({
            "scene_id": old["scene_id"],
            "episode_id": old["episode_id"],
            "baseline_success": old["success"],
            "candidate_success": new["success"],
            "baseline_distance": old["distance_to_goal"],
            "candidate_distance": new["distance_to_goal"],
            "baseline_collisions": old["collisions"],
            "candidate_collisions": new["collisions"],
        })
    summary = {
        "paired_episode_count": len(rows),
        "same_episode_set": True,
        "baseline_success": baseline["success"],
        "candidate_success": candidate["success"],
        "new_successes": sum(
            int(not row["baseline_success"] and row["candidate_success"]) for row in rows
        ),
        "lost_successes": sum(
            int(row["baseline_success"] and not row["candidate_success"]) for row in rows
        ),
        "mean_distance_change": sum(
            row["candidate_distance"] - row["baseline_distance"] for row in rows
        ) / len(rows),
        "mean_collision_change": sum(
            row["candidate_collisions"] - row["baseline_collisions"] for row in rows
        ) / len(rows),
        "episodes": rows,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
