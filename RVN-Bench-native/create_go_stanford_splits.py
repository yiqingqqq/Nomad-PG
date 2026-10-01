#!/usr/bin/env python3
"""Create deterministic train/validation splits for a GoStanford folder."""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    args = parser.parse_args()

    trajectories = sorted(
        entry.name
        for entry in args.data.iterdir()
        if entry.is_dir() and (entry / "traj_data.pkl").is_file()
    )
    if not trajectories:
        raise SystemExit(f"No trajectories with traj_data.pkl found in {args.data}")

    rng = random.Random(args.seed)
    rng.shuffle(trajectories)
    split = int(len(trajectories) * args.train_fraction)
    train, val = trajectories[:split], trajectories[split:]
    for split_name, names in (("train", train), ("val", val)):
        directory = args.output / split_name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "traj_names.txt").write_text("\n".join(names) + "\n")
    print(f"total={len(trajectories)} train={len(train)} val={len(val)} seed={args.seed}")


if __name__ == "__main__":
    main()
