"""Read-only CPU diagnostics of saved rollouts; writes a separate report."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import train_vint_pg_ppo_v2 as v

torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)
root = Path('/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_v2_frozen_bs256')
args = argparse.Namespace(**json.loads((root/'config.json').read_text()))
policy = v.build_policy(args, torch.device('cpu'))
roll = torch.load(root/'rollout_resume.pth', map_location='cpu', weights_only=True)
saved = torch.load(root/'latest_train.pth', map_location='cpu', weights_only=True)
d = roll['data']
tokens, goals = torch.stack(d['tokens']), torch.stack(d['goals'])
actions = torch.tensor(d['actions'])
adv, returns = v.gae(d)
report = {'rollout_epoch': roll['epoch'], 'checkpoint_epoch': saved['epoch'],
          'samples': len(actions), 'sampled_action_fractions': torch.bincount(actions,minlength=3).div(len(actions)).tolist(),
          'action_order': list(v.ACTIONS), 'mean_reward': float(np.mean(d['rewards'])),
          'behavior_value_explained_variance': float(1-np.var(returns-np.array(d['values']))/np.var(returns)),
          'episodes': roll['rows'], 'models': {}}
ix = torch.linspace(0,len(actions)-1,min(512,len(actions))).long()
selected = tokens[ix]
# Rotate by half the rollout and keep only cross-episode donors.
episode_ids = []
for j, row in enumerate(roll['rows']): episode_ids.extend([j]*row['steps'])
assert len(episode_ids)==len(actions)
donor = (ix+len(actions)//2)%len(actions)
assert all(episode_ids[a]!=episode_ids[b] for a,b in zip(ix.tolist(),donor.tolist()))
black = v.ObservationHistory(policy,torch.device('cpu')).push(np.zeros((480,640,3),dtype=np.uint8))

@torch.no_grad()
def forward(t,g):
    logits=[]; values=[]
    for i in range(0,len(t),64):
        l,z=policy.forward_from_tokens(t[i:i+64],g[i:i+64]);logits.append(l);values.append(z)
    return torch.cat(logits),torch.cat(values)

for label in ('baseline','best_sr','latest'):
    if label=='best_sr': policy.model.load_state_dict(torch.load(root/'best_sr.pth',map_location='cpu',weights_only=True))
    if label=='latest': policy.load_state_dict(saved['policy'])
    normal,_=forward(selected,goals[ix]); p=normal.softmax(-1)
    result={'greedy_action_fractions':torch.bincount(p.argmax(-1),minlength=3).div(len(ix)).tolist(),
            'mean_action_probabilities':p.mean(0).tolist(),'entropy':float(torch.distributions.Categorical(probs=p).entropy().mean()),
            'visual_interventions':{}}
    for name,t in [('black',black.expand(len(ix),-1,-1)),('cross_episode_tokens',tokens[donor])]:
        l,_=forward(t,goals[ix]);q=l.softmax(-1)
        result['visual_interventions'][name]={'greedy_action_change_fraction':float((p.argmax(-1)!=q.argmax(-1)).float().mean()),
                'mean_probability_total_variation':float((p-q).abs().sum(-1).mean()/2),
                'feature_relative_l2':float((selected-t).flatten(1).norm(dim=1).mean()/selected.flatten(1).norm(dim=1).mean())}
    if label=='latest':
        l,z=forward(tokens,goals); dist=torch.distributions.Categorical(logits=l)
        lr=dist.log_prob(actions)-torch.tensor(d['log_probs']); ratio=lr.exp()
        result['ppo_update']={'approx_kl':float(((ratio-1)-lr).mean()),'clip_fraction':float(((ratio-1).abs()>.2).float().mean()),
                'latest_value_explained_variance':float(1-np.var(returns-z.numpy())/np.var(returns))}
    report['models'][label]=result
    print(label,json.dumps(result),flush=True)
v.save_json(root/'diagnostics_cpu.json',report)
print('DIAGNOSTICS_COMPLETE',flush=True)
