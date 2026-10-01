"""Initialize a separate batch128 experiment with a verified reused baseline."""
import json
import os
from pathlib import Path
import shutil
import sys

repo=Path('/root/autodl-tmp/RVN-Bench-native')
base=Path('/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_v2_frozen_bs256')
root=Path('/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_baseline_bs128')
cfg=json.loads((base/'config.json').read_text())
assert not root.exists(), 'Use explicit resume after initial launch'
for mode in ('normal','black','shuffled'):
    result=json.loads((base/f'eval/baseline/validation_{mode}.json').read_text())
    assert result['complete'] and len(result['rows'])==cfg['eval_episodes']
root.mkdir()
(root/'protocol').mkdir()
for name in ('manifest.json','validation.json.gz','test.json.gz','validation_rgb_bank.pth'):
    shutil.copy2(base/'protocol'/name,root/'protocol'/name)
shutil.copytree(base/'eval/baseline',root/'eval/baseline')
(root/'baseline_reuse.json').write_text(json.dumps(dict(source=str(base),reason='same checkpoint, seed, task and evaluation settings; only minibatch changes'),indent=2))
keys=('checkpoint','task_config','scenes_dir','curriculum_dir','validation_dir','seed','rollout_steps','ppo_epochs','eval_episodes','max_steps','lr','lookahead_idx','model_waypoint_spacing','success_distance','transition_epochs')
cmd=[sys.executable,'-u',str(repo/'scripts/train_vint_pg_bs128.py')]
for key in keys: cmd += ['--'+key.replace('_','-'),str(cfg[key])]
cmd += ['--output-dir',str(root),'--batch-size','128','--success-reward','2.5','--max-epochs','10','--patience','3','--skip-final-test','--resume']
os.chdir(repo)
os.environ['PYTHONPATH']=':'.join(map(str,[repo,repo/'models/diffusion_policy',repo/'models/gnms_levin/train']))
os.environ['MAGNUM_LOG']='quiet'
os.environ['HABITAT_SIM_LOG']='quiet'
os.execv(sys.executable,cmd)
