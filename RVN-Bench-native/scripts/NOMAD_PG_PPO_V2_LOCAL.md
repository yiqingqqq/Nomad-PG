# NoMaD-PG PPO v2 (local migration)

The active old-checkpoint versus Stage-1-new controlled comparison, including
its exact settings, results, and interleaved rollout schedule, is documented in
`scripts/NOMAD_PG_OLD_VS_STAGE1_NEW_EXPERIMENT.md`.

The local launcher discovers the migration root from its own location. It uses
the immutable migrated validation/test protocol and the physical HM3D copy via
`runtime/habitat-data/scene_datasets/hm3d`.

Preflight (safe; no Habitat, Torch, or GPU initialization):

```bash
python3 scripts/preflight_nomad_pg_ppo_v2.py --migration-root ..
```

Training is intentionally GPU-only. The reproducible environment is at
`../.conda-env`; the launcher selects it automatically and exposes the migrated
Habitat-Lab source through `PYTHONPATH`. First use a separate smoke output:

Environment versions: Python 3.9, Habitat-Sim 0.3.2 (Bullet), Habitat-Lab
0.3.2 source, and PyTorch 2.7.1 + CUDA 12.8. PyTorch is newer than the remote
2.5.1 build because RTX 5060 (`sm_120`) requires an official CUDA 12.8 build.

For an interactive shell:

```bash
source scripts/activate_nomad_pg_env.sh
```

The launcher does not require prior activation:

```bash
scripts/run_nomad_pg_ppo_v2_local.sh \
  --output-dir ../runs/nomad_pg_ppo_v2_smoke \
  --max-epochs 1 --rollout-steps 64 --eval-episodes 4 --max-steps 30
```

Do not add `--resume` on a fresh run. Add it only when continuing the same
output directory. The trainer refuses to overwrite an existing checkpoint
without that flag.

Design invariants:

- policy inputs are four RGB frames and PointGoal `(range, heading)` only;
- visual encoder and distance head are frozen;
- PPO updates the diffusion noise predictor plus a new value head;
- rollout stores the exact denoising transition and behavior log probability;
- validation and test files are never regenerated;
- no expert action or shortest-path supervision is used.

Default PPO reward uses clipped geodesic progress and is recorded in
`config.json`: `1.0 * clip(progress, -0.25, 0.25) - 0.01` per step,
`-0.10` for a collided forward action, and `+5.0` on success.
