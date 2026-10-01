"""Create a branch from the durable reward5 round-3 state, then train at batch 512."""
import json
import os
from pathlib import Path
import shutil
import sys
import torch

repo = Path('/root/autodl-tmp/RVN-Bench-native')
source = Path('/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_reward5_bs256')
root = Path('/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_reward5_repeatcollision')
branch_checkpoint = source/'round3_train_backup.pth'
assert not root.exists(), 'Refusing to overwrite an existing repeatcollision experiment'
state = torch.load(branch_checkpoint, map_location='cpu', weights_only=True)
assert state['epoch'] == 2 and len(state['history']) == 3
root.mkdir()
for folder in ('protocol', 'eval/baseline'):
    (root/folder).parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source/folder, root/folder)
# This copy cannot contain the later batch256 continuation state.
shutil.copy2(branch_checkpoint, root/'latest_train.pth')
for name in ('best_sr.pth', 'best_spl.pth'):
    temp = root/(name+'.tmp')
    model_state = {k.removeprefix('model.'): v for k, v in state['policy'].items()
                   if k.startswith('model.')}
    torch.save(model_state, temp)
    temp.replace(root/name)
(root/'history.json').write_text(json.dumps(dict(complete=False, history=state['history'],
    branch=dict(source=str(source), checkpoint=str(branch_checkpoint), epoch=2,
                batch_size=256)), indent=2))
(root/'branch.json').write_text(json.dumps(dict(
    source_experiment=str(source), source_checkpoint=str(branch_checkpoint),
    source_epoch=2, copied_baseline=True, batch_size=256, max_epochs=10, patience=3,
    collision_reward='-0.02 - 0.03 * min(k - 1, 6)'), indent=2))
cfg = json.loads((source/'config.json').read_text())
keys = ('checkpoint', 'task_config', 'scenes_dir', 'curriculum_dir', 'validation_dir',
        'seed', 'rollout_steps', 'ppo_epochs', 'eval_episodes', 'max_steps', 'lr',
        'lookahead_idx', 'model_waypoint_spacing', 'success_distance', 'transition_epochs')
cmd = [sys.executable, '-u', str(repo/'scripts/train_vint_pg_repeatcollision.py')]
for key in keys:
    cmd += ['--'+key.replace('_', '-'), str(cfg[key])]
cmd += ['--output-dir', str(root), '--success-reward', '5', '--batch-size', '256',
        '--max-epochs', '10', '--patience', '3', '--skip-final-test', '--resume']
os.chdir(repo)
os.environ['PYTHONPATH'] = ':'.join(map(str, [repo, repo/'models/diffusion_policy', repo/'models/gnms_levin/train']))
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.execv(sys.executable, cmd)
