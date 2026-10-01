#!/usr/bin/env python3
"""One-sample invariant test for GoStanford horizontal mirroring."""

import numpy as np
import torch

from vint_train.data.vint_dataset import ViNT_Dataset


def dataset(flip):
    return ViNT_Dataset(
        data_folder="/root/autodl-tmp/go_stanford",
        data_split_folder="/root/autodl-tmp/go_stanford_splits_seed0/val",
        dataset_name="go_stanford",
        image_size=[96, 96],
        waypoint_spacing=1,
        min_dist_cat=0,
        max_dist_cat=20,
        min_action_distance=0,
        max_action_distance=10,
        negative_mining=True,
        len_traj_pred=5,
        learn_angle=True,
        context_size=5,
        context_type="temporal",
        end_slack=0,
        goals_per_obs=2,
        normalize=True,
        goal_type="image",
        horizontal_flip_prob=flip,
    )


base = dataset(0.0)
np.random.seed(123)
original = base[100]
base.horizontal_flip_prob = 1.0
np.random.seed(123)
flipped = base[100]

checks = {
    "obs_image": torch.allclose(flipped[0], torch.flip(original[0], dims=(-1,))),
    "goal_image": torch.allclose(flipped[1], torch.flip(original[1], dims=(-1,))),
    "action_x": torch.allclose(flipped[2][:, 0], original[2][:, 0]),
    "action_y": torch.allclose(flipped[2][:, 1], -original[2][:, 1]),
    "action_cos": torch.allclose(flipped[2][:, 2], original[2][:, 2]),
    "action_sin": torch.allclose(flipped[2][:, 3], -original[2][:, 3]),
    "goal_x": torch.allclose(flipped[4][0], original[4][0]),
    "goal_y": torch.allclose(flipped[4][1], -original[4][1]),
}
print(checks)
if not all(checks.values()):
    raise SystemExit("mirror invariant test failed")
print("mirror invariant test passed")
