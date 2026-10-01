#!/usr/bin/env python3
"""Convert HuRoN/SACSoN HDF5 trajectories to the ViNT directory format.

Each output trajectory contains numbered JPEG files plus ``traj_data.pkl``
with the ``position`` and ``yaw`` arrays expected by ViNT_Dataset.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np


def frame_bytes(frame: object) -> bytes:
    """Normalize an HDF5 variable-length JPEG value to bytes."""
    if isinstance(frame, bytes):
        return frame
    if isinstance(frame, np.ndarray):
        return frame.tobytes()
    return bytes(frame)


def convert_split(
    source: h5py.File, split: str, output: Path, limit: int | None, resume: bool
) -> int:
    converted = 0
    for trajectory_name in source[split]:
        if limit is not None and converted >= limit:
            break

        # A complete conversion writes metadata last, after every JPEG.  On a
        # resumed run this makes ``traj_data.pkl`` a safe completion marker.
        destination = output / f"{split}__{trajectory_name}"
        if destination.exists():
            if not resume:
                raise FileExistsError(f"Output trajectory already exists: {destination}")
            if not (destination / "traj_data.pkl").is_file():
                raise RuntimeError(
                    f"Incomplete trajectory directory found: {destination}. "
                    "Inspect it before resuming."
                )
            continue

        trajectory = source[split][trajectory_name]
        frames = trajectory["frames"]
        position = np.asarray(trajectory.attrs["position"], dtype=np.float32)
        yaw = np.asarray(trajectory.attrs["yaw"], dtype=np.float32)
        if len(frames) != len(position) or len(position) != len(yaw):
            raise ValueError(
                f"{split}/{trajectory_name}: frame/position/yaw lengths "
                f"are {len(frames)}/{len(position)}/{len(yaw)}"
            )

        # Prefixing the split prevents accidental name collisions while keeping
        # a flat trajectory-root layout required by data_split.py.
        destination.mkdir(parents=True, exist_ok=False)
        for index, frame in enumerate(frames):
            jpeg = frame_bytes(frame)
            if not jpeg.startswith(b"\xff\xd8"):
                raise ValueError(f"{split}/{trajectory_name}/{index}: not a JPEG frame")
            (destination / f"{index}.jpg").write_bytes(jpeg)
        with (destination / "traj_data.pkl").open("wb") as handle:
            pickle.dump({"position": position, "yaw": yaw}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        converted += 1
        print(f"converted {split}/{trajectory_name}: {len(frames)} frames")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=("train", "test"))
    parser.add_argument("--max-trajectories-per-split", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and not args.resume:
        raise SystemExit(f"Output directory must be empty: {args.output}")
    with h5py.File(args.input, "r") as source:
        total = sum(
            convert_split(source, split, args.output, args.max_trajectories_per_split, args.resume)
            for split in args.splits
        )
    print(f"done: {total} trajectories in {args.output}")


if __name__ == "__main__":
    main()
