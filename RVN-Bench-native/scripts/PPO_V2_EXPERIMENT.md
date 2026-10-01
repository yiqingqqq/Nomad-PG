# ViNT-PG interactive PPO v2

This experiment supersedes the preliminary PPO run; outputs and code of the
preliminary run remain separate. No claimed performance improvement exists yet.

## Protocol

- Initialize from the earlier GoStanford mirror epoch-1 ViNT-PG baseline.
- No HM3D expert action trajectories; start/goal metadata defines tasks.
- Reward: 2.5 for successful STOP, 0.2 times geodesic progress clipped to
  [-1,1] metres per step, -0.01 per movement decision, -0.1 per collision.
  Non-finite distances provide zero progress reward and increment a counter.
- Geodesic distance is only used for reward/scoring and task categorization,
  never as a policy feature. Policy inputs: six RGB frames and PointGoal.
- Official train scenes supply training tasks. Official val scenes are split
  by scene into this experiment's validation and test groups. These are NOT
  the official hidden benchmark test set; some scenes were used in older work.
- Fixed validation and test lists each have 100 episodes: 10 near-straight,
  30 large-initial-heading, 60 detour-ratio tasks. These are geometric proxy
  categories, not certified doorway or semantic-room labels.
- Train category sampling transitions linearly over ten epochs from
  40/30/30 to 20/20/60 percent. Disable scene grouping; sample distinct scenes
  for successive candidates. Log observed category/scene coverage.

## Model and PPO correctness

- EfficientNet and observation compression parameters AND buffers frozen.
  SHA-256 covers all their state tensors before/after each optimizer round.
- All modules remain in eval mode so dropout and BatchNorm updates are off.
  Autograd still updates PointGoal encoder, decoder, trajectory action head,
  and the PPO value head. No collision prediction auxiliary head.
- Frozen per-frame tokens are cached and stored as float32. Test agreement
  with the original six-frame feature path before training.
- A 15-degree waypoint-heading threshold defines both the original greedy
  controller and the new categorical controller's argmax. Stochastic action
  sampling is used only during PPO collection. STOP remains distance <=0.2m.
- Before every update, all saved action log probabilities are recomputed;
  maximum error must be <=1e-4. Same perturbation-free observations in both.
- A time-limit boundary cuts GAE recursion but bootstraps the next-state
  value. An executed STOP terminates and has no bootstrap.
- At least 8192 steps per round, collected through episode boundaries;
  four PPO passes, exactly 256 samples per optimizer step, shuffled remainder
  shorter than 256 omitted on each pass. Learning rate 1e-5, clip 0.2,
  gamma .99, lambda .95, value coefficient .5, entropy coefficient .01.
- No PointGoal dropout/noise in this first controlled experiment.
- Up to 50 rounds. Early-stop patience 5 begins only after ten transition
  rounds and five further full-course rounds. Rank by validation SR then SPL.
  Visual gap is diagnostic, never an early-stop trigger.

## Evaluation and persistence

- Baseline first: normal / black / shuffled on fixed validation tasks.
  Every epoch: normal / black on the same tasks. Deterministic greedy actions.
- Shuffled condition uses fixed cross-scene initial RGB donors, held constant
  through each episode and reused across models; this is explicitly a
  cross-scene image replacement control, not a permutation of full trajectories.
- End of run: baseline and best-SR model on held-out test tasks for all three
  image modes. Test tasks never select checkpoints or trigger early stopping.
- Report SR, SPL, forward-action collision rate, final distance, progress,
  and per-category results. Raw per-episode rows are retained.
- Each completed collection episode writes atomic JSON and a resumable
  rollout checkpoint including RNG state. Interrupted optimizer work is
  replayed from the last full model/optimizer checkpoint and saved rollout.
  Incomplete current episodes are rerun. Evaluation resumes from completed rows.
- Latest checkpoint contains policy, optimizer, RNG, history, early-stop state.
  Best SR and best SPL exports are selected independently and contain tensors.

## Paths

Remote scripts: `/root/autodl-tmp/RVN-Bench-native/scripts/`

Formal outputs: `/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_v2_frozen_bs256/`

Acceptance outputs: `/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_v2_acceptance/`

Smoke outputs: `/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_v2_smoke/`

The smoke uses separate weights/output and its trained parameters never seed
the formal run. Launch/resume with `bash scripts/run_vint_pg_ppo_v2.sh train`.
