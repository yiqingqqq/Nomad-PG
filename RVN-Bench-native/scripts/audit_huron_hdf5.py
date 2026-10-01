#!/usr/bin/env python3
"""Read-only integrity and motion-scale audit for HuRoN/SACSoN HDF5."""

import json

import h5py
import numpy as np


HDF5_PATH = "/root/autodl-tmp/huron_hdf5_audit/sacson.h5"


def main() -> None:
    lengths = []
    step_distances = []
    yaw_deltas = []
    position_shapes = {}
    mismatches = 0
    finite_trajectories = 0
    split_counts = {}

    with h5py.File(HDF5_PATH, "r") as source:
        for split in source:
            split_counts[split] = len(source[split])
            for name in source[split]:
                trajectory = source[split][name]
                position = np.asarray(trajectory.attrs["position"], dtype=np.float32)
                yaw = np.asarray(trajectory.attrs["yaw"], dtype=np.float32)
                frame_count = len(trajectory["frames"])
                lengths.append(frame_count)
                shape = str(position.shape[1:] if position.ndim > 1 else ())
                position_shapes[shape] = position_shapes.get(shape, 0) + 1
                mismatches += int(frame_count != len(position) or len(position) != len(yaw))
                finite_trajectories += int(
                    np.isfinite(position).all() and np.isfinite(yaw).all()
                )
                if len(position) > 1:
                    step_distances.extend(
                        np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1).tolist()
                    )
                    delta = np.diff(yaw)
                    yaw_deltas.extend(
                        np.abs(np.arctan2(np.sin(delta), np.cos(delta))).tolist()
                    )

    result = {
        "split_counts": split_counts,
        "trajectories": len(lengths),
        "frames": int(np.sum(lengths)),
        "length_min_median_mean_p95_max": [
            int(np.min(lengths)),
            float(np.median(lengths)),
            float(np.mean(lengths)),
            float(np.percentile(lengths, 95)),
            int(np.max(lengths)),
        ],
        "position_shapes": position_shapes,
        "length_mismatches": mismatches,
        "all_finite_trajectories": finite_trajectories,
        "step_distance_median_mean_p95": [
            float(np.median(step_distances)),
            float(np.mean(step_distances)),
            float(np.percentile(step_distances, 95)),
        ],
        "yaw_delta_median_mean_p95": [
            float(np.median(yaw_deltas)),
            float(np.mean(yaw_deltas)),
            float(np.percentile(yaw_deltas, 95)),
        ],
        "trajectories_at_least_12_frames": int(np.sum(np.asarray(lengths) >= 12)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
