# NoMAD-PG old vs Stage-1-new PPO comparison

Updated: 2026-09-30 (Asia/Shanghai)

Unless stated otherwise, relative paths are resolved from the
`RVN-Bench-native` repository root.

This document is the source of truth for the two local HM3D PointNav PPO
experiments. It replaces the outdated settings in `PPO_V2_EXPERIMENT.md` for
this comparison. In particular, these runs use success reward `+5.0`, keep the
whole vision encoder (EfficientNet, PointGoal encoder, and Transformer) frozen
during PPO, and compare two different initialization checkpoints.

## Research question

Does the new 20-epoch supervised Stage 1 initialization improve HM3D PointNav
PPO over the migrated old `ema_3.pth` initialization when every PPO and
evaluation setting is held fixed?

The controlled variable is the PPO initialization:

| Experiment | Initialization | SHA-256 | Output directory |
| --- | --- | --- | --- |
| old | migrated `ema_3.pth` | `fd3d7c3605ce63ebbe278829c94c9b36b4e1bb7bbe1ccaffe351a604de1ed5ea` | `../runs/nomad_pg_ppo_v2` |
| Stage-1-new | local supervised Stage 1 `ema_19.pth` | `f16de52997f010b87c4f7243131390741442afe78c32abf53587a04b1468c07d` | `../runs/nomad_pg_ppo_v2_stage1_new` |

The old checkpoint is retained only as a comparison branch. It must not be
used as the initialization of the Stage-1-new branch.

## Stage-1-new initialization

The new initialization was trained for 20 supervised epochs on a balanced
GoStanford/HuRoN mixture. EfficientNet and the Transformer remained frozen.
Only these modules were trained:

- `vision_encoder.goal_encoder`, base learning rate `1e-4`;
- `noise_pred_net`, learning rate `1e-5` through a `0.1` multiplier.

Other Stage 1 settings include batch size 16, PointGoal masking probability
0.5, AdamW, cosine scheduling, two warmup epochs, gradient clipping at 1.0,
and seed 270928. The exact configuration is
`models/gnms_levin/train/config/nomad_pg_stage1_pointnav_local.yaml`.

## Shared PPO configuration

Both branches use `scripts/train_nomad_pg_ppo_v2.py` with identical settings:

| Setting | Value |
| --- | --- |
| seed | 270928 |
| rollout size | 8,192 transitions per epoch |
| PPO passes | 4 |
| minibatch size | 256 |
| learning rate | `1e-5` |
| PPO clip | 0.2 |
| GAE | gamma 0.99, lambda 0.95 |
| value coefficient | 0.5 |
| entropy coefficient | 0.0 |
| gradient norm limit | 1.0 |
| maximum episode length | 500 steps |
| success distance | 0.2 m |
| validation size | 100 fixed episodes per condition |

PPO freezes `model.vision_encoder` and `dist_pred_net`, forces both into eval
mode, and updates the diffusion `noise_pred_net` plus a newly initialized value
head. Consequently, EfficientNet, the PointGoal MLP, and the Transformer are
all frozen during PPO in both branches. Policy inputs are four RGB frames and
PointGoal `(range, heading)`; geodesic distance is used for reward and metrics,
not as a policy input.

The exact run-time configurations are written to each output directory as
`config.json`.

## Reward

For each movement step:

```text
reward = clip(previous_geodesic_distance - new_geodesic_distance, -0.25, 0.25)
         - 0.01
         - 0.10 * collided_forward
         + 5.0 * successful_stop
```

Non-finite geodesic progress contributes zero progress reward. Success is
awarded only after STOP within 0.2 m.

## Data and fixed evaluation protocol

Training tasks come from `../pointnav-curriculum/train_v2`. The curriculum has
5,000 direct, 5,000 turn, and 5,000 detour candidates and contains no expert
trajectories. Epoch sampling moves over ten epochs from 40/30/30 to 20/20/60
percent for direct/turn/detour.

Validation and test tasks are immutable files under `../protocol`:

- validation: 100 episodes (10 direct, 30 turn, 60 detour);
- test: 100 episodes (10 direct, 30 turn, 60 detour);
- protocol seed: 270928.

Every reported validation number below uses the same 100 episodes. Diffusion
noise is deterministically seeded per episode, making old/new and epoch-to-
epoch evaluation paired. Normal RGB selects checkpoints; black RGB is a visual
dependence diagnostic. Rollout success is not a replacement for validation SR
because rollout tasks and behavior sampling differ by epoch and policy.

Habitat may warn that HM3D semantic annotation files are absent. PointNav here
uses geometry, RGB, PointGoal, and navigation metrics and does not require
semantic labels; these warnings are not treated as failed episodes.

## Collection and PPO ordering

For each branch, the dependency is strict:

```text
policy after epoch N-1
  -> collect rollout_epoch_N (8,192 steps)
  -> PPO update N
  -> fixed normal/black validation
  -> policy after epoch N
```

A rollout may be prefetched only after its source policy exists. A rollout
from an older policy must not be relabeled for a later epoch. To reduce wall
time, the two independent branches are interleaved: one branch can collect
while the other performs PPO/evaluation. Their output directories are separate,
so weights, rollouts, histories, and evaluation JSON files do not overwrite
one another.

Local pipeline launchers:

- `scripts/run_old_epoch1_then_collect2.sh`;
- `scripts/run_new_epoch1_then_collect2.sh`;
- `scripts/collect_new_epoch1.sh` (one-off prefetch used before the new Epoch 1 update).
- `scripts/run_old_auto_to_epoch9.sh` and `scripts/run_new_auto_to_epoch9.sh`
  continuously consume Epoch 2 through Epoch 9 and prefetch Epoch 10.

## Results through Epoch 1

All SR/SPL entries in this table are fixed 100-episode validation results.
Values are percentages.

| Model | Checkpoint stage | Normal SR | Normal SPL | Black SR | Black SPL |
| --- | --- | ---: | ---: | ---: | ---: |
| old | baseline | 20 | 11.56 | 24 | 18.89 |
| Stage-1-new | baseline | 23 | 16.15 | 19 | 16.10 |
| old | PPO Epoch 0 | 23 | 11.63 | 14 | 9.51 |
| Stage-1-new | PPO Epoch 0 | 29 | 17.55 | 10 | 6.63 |
| old | PPO Epoch 1 | 22 | 11.88 | 14 | 9.01 |
| Stage-1-new | PPO Epoch 1 | **33** | **18.97** | 17 | 9.75 |

At Epoch 1, Stage-1-new exceeds old by 11 SR points and 7.09 SPL points on
normal RGB. Its normal-minus-black SR gap is 16 points, versus 8 points for
old. This is consistent with stronger use of visual input, but the comparison
should continue across later epochs and the held-out test protocol before a
final claim is made.

Rollout diagnostics (not fixed validation metrics):

| Model | Epoch | Episodes used to reach 8,192 steps | Rollout success | Rollout SPL |
| --- | ---: | ---: | ---: | ---: |
| old | 0 | 26 | 38.46% | 30.24% |
| old | 1 | 20 | 25.00% | 12.63% |
| Stage-1-new | 0 | 21 | 28.57% | 19.95% |
| Stage-1-new | 1 | 21 | 23.81% | 15.48% |

## Current durable state

As of this update, both branches have completed PPO Epoch 1 and have prefetched
the correct 8,192-step Epoch 2 rollout:

- old: `../runs/nomad_pg_ppo_v2/rollout_epoch_002.pth`;
- Stage-1-new: `../runs/nomad_pg_ppo_v2_stage1_new/rollout_epoch_002.pth`.

There is no active PPO/collection process at this snapshot. Resume each branch
with `--resume --max-epochs 3`; the trainer will consume its existing
`rollout_epoch_002.pth` rather than recollecting it.

Durable records for each branch:

- `config.json`: actual command-line configuration;
- `history.json`: completed PPO updates and validation summaries;
- `latest_train.pth`: policy, optimizer, history, and best score state;
- `best_sr.pth`: best normal-validation checkpoint;
- `eval/epoch_NNN/validation_{normal,black}.json`: summaries and raw paired rows;
- `rollout_epoch_NNN.pth`: prefetched on-policy rollout awaiting update.

## Interpretation rules

1. Select models only by normal-validation SR, then SPL; never by black results.
2. Do not compare rollout SR as if it were fixed validation SR.
3. Do not consume a rollout in a branch other than the one that generated it.
4. Do not report test performance until checkpoint selection is complete.
5. Report raw counts and both SR/SPL; a 100-episode SR changes in one-point steps.
6. Preserve all per-episode JSON rows so gains and regressions can be checked as
   paired outcomes, not only as aggregate means.
