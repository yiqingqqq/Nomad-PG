#!/usr/bin/env python3
"""Build reproducible HM3D PointNav endpoint-only curriculum splits.

The output keeps only episode start/goal metadata from the official PointNav
dataset. It never creates or stores a shortest-path/expert action trajectory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from collections import Counter
from pathlib import Path


def episode_geometry(episode: dict) -> tuple[float, float, float]:
    start = episode["start_position"]
    goal = episode["goals"][0]["position"]
    dx, dz = goal[0] - start[0], goal[2] - start[2]
    euclidean = math.hypot(dx, dz)
    geodesic = float(episode.get("info", {}).get("geodesic_distance", euclidean))
    detour_ratio = geodesic / max(euclidean, 1e-6)

    # Habitat rotations are [x, y, z, w]. PointNav starts use yaw-only
    # rotations; rotate the agent's local -Z forward vector into world space.
    _, qy, _, qw = episode["start_rotation"]
    forward_x = -2.0 * qw * qy
    forward_z = -(1.0 - 2.0 * qy * qy)
    cosine = (forward_x * dx + forward_z * dz) / max(euclidean, 1e-6)
    heading_error = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return geodesic, detour_ratio, heading_error


def category(detour_ratio: float, heading_error: float) -> str | None:
    if detour_ratio >= 1.25:
        return "detour"
    if detour_ratio <= 1.20 and heading_error >= 60.0:
        return "turn"
    if detour_ratio <= 1.08 and heading_error <= 30.0:
        return "direct"
    return None


def load_episodes(content_dir: Path) -> list[dict]:
    episodes: list[dict] = []
    for path in sorted(content_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            episodes.extend(json.load(handle).get("episodes", []))
    return episodes


def write_dataset(path: Path, episodes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": episodes}, handle, separators=(",", ":"))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=270927)
    parser.add_argument("--limit-per-category", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    buckets: dict[str, list[dict]] = {name: [] for name in ("direct", "turn", "detour")}
    skipped = 0
    for episode in load_episodes(args.content_dir):
        geodesic, detour_ratio, heading_error = episode_geometry(episode)
        name = category(detour_ratio, heading_error)
        if name is None:
            skipped += 1
            continue
        copied = dict(episode)
        copied["shortest_paths"] = None
        buckets[name].append(copied)

    for rows in buckets.values():
        rng.shuffle(rows)
        if args.limit_per_category > 0:
            del rows[args.limit_per_category :]

    mixed: list[dict] = []
    max_len = max(map(len, buckets.values()), default=0)
    for index in range(max_len):
        for name in ("direct", "turn", "detour"):
            if index < len(buckets[name]):
                mixed.append(buckets[name][index])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        write_dataset(args.output_dir / f"{name}.json.gz", rows)
    write_dataset(args.output_dir / "mixed.json.gz", mixed)

    summary = {
        "seed": args.seed,
        "source": str(args.content_dir),
        "thresholds": {
            "direct": "detour_ratio <= 1.08 and heading_error <= 30 deg",
            "turn": "detour_ratio <= 1.20 and heading_error >= 60 deg",
            "detour": "detour_ratio >= 1.25",
        },
        "counts": {name: len(rows) for name, rows in buckets.items()},
        "mixed": len(mixed),
        "skipped": skipped,
        "contains_expert_trajectories": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
