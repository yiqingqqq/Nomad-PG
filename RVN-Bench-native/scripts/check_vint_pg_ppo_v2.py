"""Targeted acceptance checks for the frozen ViNT PPO implementation."""
import argparse
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.distributions import Categorical
import train_vint_pg_ppo_v2 as v2

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output-dir',required=True)
args=p.parse_args()
args.lookahead_idx=2
args.batch_size=256
args.ppo_epochs=1
torch.manual_seed(270928)
torch.set_num_threads(2)
torch.backends.mha.set_fastpath_enabled(False)
device=torch.device('cuda')
policy=v2.build_policy(args,device)
policy.train()
assert not any(m.training for m in policy.modules())
before=v2.frozen_hash(policy)
hist=v2.ObservationHistory(policy,device)
max_cache_error=0.
for i in range(3):
    rgb=np.random.default_rng(i).integers(0,256,(128,128,3),dtype=np.uint8)
    tokens=hist.push(rgb)
    full=policy.encode_observation(hist.images.get_context())
    max_cache_error=max(max_cache_error,(full-tokens).abs().max().item())
assert max_cache_error < 1e-4, max_cache_error
goal=torch.tensor([[2.,1.,0.]],device=device)
with torch.no_grad():
    a=policy.forward_from_tokens(tokens,goal)[0]
    b=policy.forward_from_tokens(tokens,goal)[0]
assert torch.equal(a,b)
assert v2.reward_delta(float('inf'),float('inf'),False)==(-.01,1)
assert v2.reward_delta(10.,9.,True)[0] < .2
sample=dict(rewards=[1.,2.],values=[.5,.7],next_values=[.8,0.],boundaries=[True,True])
adv,ret=v2.gae(sample,gamma=.9,lam=.95)
np.testing.assert_allclose(adv,[1+.9*.8-.5,2-.7],rtol=1e-6)

# Collect singleton behavior log probabilities, then recompute/update as 256.
data={k:[] for k in ('tokens','goals','actions','log_probs','values','rewards','next_values','boundaries')}
with torch.no_grad():
    for i in range(256):
        g=goal.clone();g[:,0]+=i/256
        logits,value=policy.forward_from_tokens(tokens,g)
        d=Categorical(logits=logits);action=d.sample()
        for k,v in zip(data,(tokens[0].cpu(),g[0].cpu(),action.item(),d.log_prob(action).item(),
                            value.item(),float(i%2)*.1-.01,0.,True)):
            data[k].append(v)
old_action=policy.model.action_predictor[0].weight.detach().clone()
optim=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=1e-5)
stats=v2.update(policy,optim,data,args,device,-1)
assert not torch.equal(old_action,policy.model.action_predictor[0].weight)
assert before==v2.frozen_hash(policy)
report=dict(passed=True,batch_size=256,frozen_sha256=before,
            frozen_weights_and_buffers_unchanged=True,action_parameters_updated=True,
            repeated_logits_bitwise_equal=True,cache_max_error=max_cache_error,
            truncation_bootstrap_passed=True,nonfinite_reward_guard_passed=True,update=stats)
v2.save_json(Path(args.output_dir)/'acceptance.json',report)
print(report,flush=True)
