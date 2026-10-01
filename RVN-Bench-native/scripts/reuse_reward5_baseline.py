"""Verify and reuse matching baseline results, preserving partial rerun output."""
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import time
import torch

base=Path('/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_v2_frozen_bs256')
target=Path('/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_reward5_bs256')
a=json.loads((base/'config.json').read_text())
b=json.loads((target/'config.json').read_text())
for key in ('checkpoint','task_config','scenes_dir','seed','eval_episodes','max_steps','lookahead_idx','model_waypoint_spacing','success_distance','frozen_sha256','controller'):
    assert a[key]==b[key], key
def dataset(p):
    with gzip.open(p,'rt') as f: return json.load(f)
assert dataset(base/'protocol/validation.json.gz')==dataset(target/'protocol/validation.json.gz')
for mode in ('normal','black','shuffled'):
    x=json.loads((base/f'eval/baseline/validation_{mode}.json').read_text())
    assert x['complete'] and len(x['rows'])==100
state=torch.load(target/'latest_train.pth',map_location='cpu',weights_only=True)
assert state['epoch']==-1, 'Training already started; do not interrupt'
proc=Path('/proc/1264')
assert b'train_vint_pg_reward5.py' in (proc/'cmdline').read_bytes()
archive=target/'eval/baseline_partial_before_reuse'
assert not archive.exists()
os.kill(1264,signal.SIGTERM)
for _ in range(100):
    if not proc.exists(): break
    time.sleep(.1)
assert not proc.exists(), 'Process has not stopped'
partial=target/'eval/baseline'
old=json.loads((base/'eval/baseline/validation_normal.json').read_text())['rows']
new=json.loads((partial/'validation_normal.json').read_text())['rows']
assert new==old[:len(new)], 'Partial results differ; stopped, no reuse performed'
partial.rename(archive)
shutil.copytree(base/'eval/baseline',partial)
bank=target/'protocol/validation_rgb_bank.pth'
assert not bank.exists()
shutil.copy2(base/'protocol/validation_rgb_bank.pth',bank)
(target/'baseline_reuse.json').write_text(json.dumps(dict(source=str(base),verified_config=True,
    verified_episodes=True,matching_partial_episodes=len(new),archived_partial=str(archive),timestamp=time.time()),indent=2))
print('BASELINE_REUSED',len(new),flush=True)
