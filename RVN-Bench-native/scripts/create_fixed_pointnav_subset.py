#!/usr/bin/env python3
"""Create a fixed PointNav episode file from prior evaluation results."""

import argparse
import gzip
import json
from pathlib import Path


def open_json(path, mode):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def normalize_scene_id(scene_id):
    marker = "hm3d/"
    scene_id = str(scene_id)
    return scene_id[scene_id.index(marker):] if marker in scene_id else scene_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_path = Path(args.source)
    with open_json(source_path / "val.json.gz", "rt") as stream:
        dataset = json.load(stream)
    result = json.loads(Path(args.results).read_text())
    requested = [
        (normalize_scene_id(row["scene_id"]), str(row["episode_id"]))
        for row in result["episodes"]
    ]
    available = {}
    for content_path in sorted((source_path / "content").glob("*.json.gz")):
        with open_json(content_path, "rt") as stream:
            content = json.load(stream)
        for row in content["episodes"]:
            available[(normalize_scene_id(row["scene_id"]), str(row["episode_id"]))] = row
    missing = [key for key in requested if key not in available]
    if missing:
        raise SystemExit(f"Missing {len(missing)} requested episodes; first: {missing[0]}")
    dataset["episodes"] = [available[key] for key in requested]
    with open_json(args.output, "wt") as stream:
        json.dump(dataset, stream)
    print(f"wrote {len(dataset['episodes'])} episodes to {args.output}")


if __name__ == "__main__":
    main()
