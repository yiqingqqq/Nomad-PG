"""Smoke test for Stage 1 staged training.

Validates, without launching a full run:
1. ViNTPointGoal constructs and loads the pretrained checkpoint.
2. freeze_modules produces exactly the intended trainable set.
3. Optimizer param groups contain only trainable params.
4. A forward/backward pass flows gradients only to unfrozen modules.
5. Both GoStanford and HuRoN datasets load and yield valid samples.
"""
import os
import torch
import numpy as np

from vint_train.models.vint.vint_pointgoal import ViNTPointGoal
from vint_train.training.train_eval_loop import load_model

CKPT = "/root/autodl-fs/RVN-Bench-shared/weights/vint_pg_zero_shot_rcs.pth"
FREEZE = ["obs_encoder", "compress_obs_enc", "decoder"]
# compress_goal_enc is nn.Identity() (no params), so it never appears in the
# trainable set. The PointGoal MLP + heads are the only learnable modules.
TRAIN_EXPECTED = {"goal_encoder", "dist_predictor", "action_predictor"}

print("== construct model ==")
model = ViNTPointGoal(
    context_size=5,
    len_traj_pred=5,
    learn_angle=True,
    obs_encoder="efficientnet-b0",
    obs_encoding_size=512,
    late_fusion=False,
    mha_num_attention_heads=4,
    mha_num_attention_layers=4,
    mha_ff_dim_factor=4,
    pg_rcs=True,
)

print("== load pretrained checkpoint ==")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
load_model(model, "vint_pointgoal", ckpt)

print("== apply freezing ==")
for name in FREEZE:
    module = getattr(model, name)
    n = 0
    for p in module.parameters():
        p.requires_grad = False
        n += p.numel()
    print(f"[freeze] {name}: {n/1e6:.2f}M params")

trainable_modules = {n for n, _ in model.named_modules() if n in TRAIN_EXPECTED}
trainable_param_names = {
    n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
}
print("trainable top-level modules:", sorted(trainable_param_names))
assert trainable_param_names == TRAIN_EXPECTED, (
    f"Unexpected trainable set: {trainable_param_names}"
)
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {total_trainable/1e6:.2f}M / "
      f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M total")

print("== build optimizer (mirror train.py) ==")
lr = 1e-4
param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
optimizer = torch.optim.AdamW(param_groups, lr=lr)
n_opt_tensors = sum(len(g["params"]) for g in optimizer.param_groups)
n_opt = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
print("optimizer param groups:", len(optimizer.param_groups),
      "param tensors:", n_opt_tensors, "elems:", n_opt)
assert n_opt == total_trainable

print("== forward/backward ==")
model = model.cuda()
model.train()
obs = torch.randn(2, 3 * 6, 96, 96).cuda()   # 6 frames x 3 channels
goal = torch.randn(2, 3).cuda()               # [d, cos, sin]
dist_pred, action_pred = model(obs, goal)
print("dist_pred", tuple(dist_pred.shape), "action_pred", tuple(action_pred.shape))
loss = dist_pred.mean() + action_pred.mean()
loss.backward()
frozen_grad = sum(1 for n, p in model.named_parameters() if p.grad is not None and not p.requires_grad)
unfrozen_nograd = sum(1 for n, p in model.named_parameters() if p.requires_grad and p.grad is None)
print("frozen params with grad (should be 0):", frozen_grad)
print("unfrozen params with no grad (should be 0):", unfrozen_nograd)
assert frozen_grad == 0 and unfrozen_nograd == 0

print("== data loading ==")
from vint_train.data.vint_dataset import ViNT_Dataset

def make_dataset(dataset_name, data_folder, split_folder, end_slack, goals_per_obs):
    return ViNT_Dataset(
        data_folder=data_folder,
        data_split_folder=split_folder,
        dataset_name=dataset_name,
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
        end_slack=end_slack,
        goals_per_obs=goals_per_obs,
        normalize=True,
        goal_type="image",
    )

gs = make_dataset("go_stanford", "/root/autodl-tmp/go_stanford",
                  "/root/autodl-tmp/go_stanford_splits_seed0/train", 0, 2)
hr = make_dataset("huron", "/root/autodl-tmp/huron_vint_20260904",
                  "/root/autodl-tmp/RVN-Bench-native/models/gnms_levin/train/vint_train/data/data_splits/huron/train",
                  3, 1)
print("go_stanford len:", len(gs), "huron len:", len(hr))

for nm, ds in [("go_stanford", gs), ("huron", hr)]:
    obs_image, goal_image, actions, distance, goal_pos, dataset_index, action_mask = ds[0]
    print(f"[{nm}] obs_image {tuple(obs_image.shape)} goal_image {tuple(goal_image.shape)} "
          f"actions {tuple(actions.shape)} distance {distance} goal_pos {tuple(goal_pos.shape)}")

print("SMOKE OK")
