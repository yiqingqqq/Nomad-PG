#!/usr/bin/env python3
"""Deterministic frozen-feature ViNT-PG PPO; no expert action supervision.

Uses the legacy module ONLY for model definitions/checkpoint allowlist and
preprocessing, never its rollout/update routines. eval mode does not disable
autograd: the fusion/action/value parameters are explicitly trainable.
"""
import argparse
import gzip
import hashlib
import json
import math
import random
import time
from collections import Counter, deque
from pathlib import Path

import habitat
import numpy as np
import torch
from torch.distributions import Categorical
import train_vint_pg_interactive as legacy

CATEGORIES = ('direct', 'turn', 'detour')
ACTIONS = legacy.ACTIONS


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def live(args, phase, **details):
    save_json(Path(args.output_dir) / 'live_summary.json',
              dict(phase=phase, timestamp=time.time(), **details))


def read_dataset(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)['episodes']


def write_dataset(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with gzip.open(tmp, 'wt') as f:
        json.dump({'episodes': rows}, f, allow_nan=False)
    tmp.replace(path)


def scene_id(row):
    return Path(row['scene_id']).parent.name


def prepare_protocol(args):
    """Official train scenes stay train; split official val by scene for this run."""
    root = Path(args.output_dir) / 'protocol'
    manifest = root / 'manifest.json'
    if manifest.exists():
        return json.loads(manifest.read_text())
    train = {c: read_dataset(Path(args.curriculum_dir) / (c + '.json.gz')) for c in CATEGORIES}
    val = {c: read_dataset(Path(args.validation_dir) / (c + '.json.gz')) for c in CATEGORIES}
    train_scenes = {scene_id(r) for rows in train.values() for r in rows}
    scenes = sorted({scene_id(r) for rows in val.values() for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    val_scenes, test_scenes = set(scenes[::2]), set(scenes[1::2])
    assert not train_scenes & (val_scenes | test_scenes)
    assert not val_scenes & test_scenes
    counts = {}
    for split, selected in [('validation', val_scenes), ('test', test_scenes)]:
        rows = []
        desired = [args.eval_episodes // 10, args.eval_episodes * 3 // 10, 0]
        desired[2] = args.eval_episodes - sum(desired)
        if args.eval_episodes < 10:
            desired = [0, 0, args.eval_episodes]
        for c, count in zip(CATEGORIES, desired):
            pool = [r for r in val[c] if scene_id(r) in selected]
            rng.shuffle(pool)
            if len(pool) < count:
                raise ValueError(f'{split}/{c}: {len(pool)} available, {count} required')
            for r in pool[:count]:
                r = dict(r, shortest_paths=None)
                r['info'] = dict(r.get('info') or {}, curriculum_category=c)
                rows.append(r)
        rng.shuffle(rows)
        write_dataset(root / (split + '.json.gz'), rows)
        counts[split] = dict(Counter(r['info']['curriculum_category'] for r in rows))
    protocol = dict(seed=args.seed, counts=counts, train_scenes=sorted(train_scenes),
                    validation_scenes=sorted(val_scenes), test_scenes=sorted(test_scenes),
                    category_definition='detour ratio >=1.25; turn = initial heading >=60deg, ratio <=1.20; direct ratio <=1.08 and heading <=30deg',
                    test_scope='held out from updates and selection in THIS experiment',
                    expert_trajectories=False, curriculum_source=args.curriculum_dir)
    save_json(manifest, protocol)
    return protocol


def epoch_dataset(args, epoch):
    rng = random.Random(args.seed + epoch + 1000)
    t = min(epoch / max(args.transition_epochs - 1, 1), 1.0)
    weights = [.4 - .2*t, .3 - .1*t, .3 + .3*t]
    # 100 candidates guarantee each epoch can gather >=8192 steps. Repeated
    # cycling is permitted, but the actual scene/category counts are logged.
    labels = [c for c, w in zip(CATEGORIES, weights) for _ in range(round(w*100))]
    rng.shuffle(labels)
    pools = {c: read_dataset(Path(args.curriculum_dir)/(c+'.json.gz')) for c in CATEGORIES}
    selected, used_scenes = [], set()
    for c in labels:
        choices = [r for r in pools[c] if scene_id(r) not in used_scenes]
        if not choices:
            used_scenes.clear()
            choices = pools[c]
        r = dict(rng.choice(choices), shortest_paths=None)
        used_scenes.add(scene_id(r))
        r['info'] = dict(r.get('info') or {}, curriculum_category=c)
        selected.append(r)
    path = Path(args.output_dir)/'protocol'/f'train_epoch_{epoch:03d}.json.gz'
    write_dataset(path, selected)
    return path, weights


def make_env(args, path):
    cfg = habitat.get_config(args.task_config)
    with habitat.config.read_write(cfg):
        cfg.habitat.dataset.data_path = str(path)
        cfg.habitat.dataset.scenes_dir = args.scenes_dir
        cfg.habitat.dataset.split = 'val'
        cfg.habitat.environment.max_episode_steps = args.max_steps
        cfg.habitat.environment.iterator_options.shuffle = False
        cfg.habitat.environment.iterator_options.group_by_scene = False
        cfg.habitat.seed = args.seed
    return habitat.Env(config=cfg)


class Policy(legacy.InteractivePolicy):
    def train(self, mode=True):
        # Deterministic activations in collection, recomputation, and eval;
        # gradients remain enabled for parameters with requires_grad=True.
        super().train(False)
        return self

    def forward_from_tokens(self, tokens, goal):
        g = self.model.compress_goal_enc(self.model.goal_encoder(goal)).unsqueeze(1)
        rep = self.model.decoder(torch.cat((tokens, g), 1))
        raw = self.model.action_predictor(rep).reshape(-1, 5, self.model.num_action_params)
        p = raw[:, :, :2].cumsum(1)[:, self.lookahead_idx]
        # atan2(0,0) has undefined backward. Degenerate predictions default
        # to forward and have zero angle gradient at this singularity.
        degenerate = p.square().sum(-1) < 1e-12
        x = torch.where(degenerate, torch.ones_like(p[:, 0]), p[:, 0])
        y = torch.where(degenerate, torch.zeros_like(p[:, 1]), p[:, 1])
        angle = torch.atan2(y, x)
        h = math.pi / 12
        logits = torch.stack((h-angle.abs(), angle-h, -angle-h), 1)/self.temperature
        value = self.value_head(rep).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise RuntimeError('Non-finite model output; no update performed')
        return logits, value


class ObservationHistory:
    """Cache frame tokens only while the visual encoder is strictly frozen."""
    def __init__(self, policy, device):
        self.policy, self.device = policy, device
        self.images = legacy.habitat_gnm.ObsImageContext(6, [96, 96], device)
        self.tokens = deque(maxlen=6)

    @torch.no_grad()
    def push(self, rgb):
        self.images.add_obs_image(rgb)
        current = self.images.get_context()[:, -3:]
        enc = self.policy.model.obs_encoder
        features = enc._avg_pooling(enc.extract_features(current)).flatten(1)
        features = self.policy.model.compress_obs_enc(enc._dropout(features))
        if not self.tokens:
            self.tokens.extend([features]*6)
        else:
            self.tokens.append(features)
        return torch.stack(list(self.tokens), 1)


def frozen_hash(policy):
    h = hashlib.sha256()
    for name in ('obs_encoder', 'compress_obs_enc'):
        for k, v in getattr(policy.model, name).state_dict().items():
            h.update((name+'.'+k).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=(int(np.random.get_state()[2]),
                np.random.get_state()[1].tolist(), int(np.random.get_state()[3]),
                float(np.random.get_state()[4])), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    pos, keys, has_gauss, cached = state['numpy']
    np.random.set_state(('MT19937', np.array(keys,dtype=np.uint32),pos,has_gauss,cached))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def safe_number(x):
    return float(x) if math.isfinite(float(x)) else None


def reward_delta(before, after, collided):
    valid = math.isfinite(before) and math.isfinite(after)
    progress = float(np.clip(before-after, -1, 1)) if valid else 0.
    return .2*progress - .01 - .1*int(collided), int(not valid)


def collect(policy, args, device, epoch, dataset):
    rollout_path = Path(args.output_dir)/'rollout_resume.pth'
    keys = ('tokens','goals','actions','log_probs','values','rewards','next_values','boundaries')
    data = {k: [] for k in keys}
    rows = []
    if args.resume and rollout_path.exists():
        pending = torch.load(rollout_path,map_location='cpu',weights_only=True)
        if pending['epoch'] == epoch:
            data, rows = pending['data'], pending['rows']
            restore_rng(pending['rng'])
    env = make_env(args, dataset)
    # Resume at an episode boundary; simulator poses need not be serialized.
    all_eps = list(env.episodes)
    offset = len(rows) % len(all_eps)
    env.episodes = all_eps[offset:] + all_eps[:offset]
    if args.resume and rollout_path.exists() and pending['epoch'] == epoch:
        restore_rng(pending['rng'])
    try:
        while len(data['actions']) < args.rollout_steps:
            obs = env.reset()
            hist = ObservationHistory(policy, device)
            tok = hist.push(obs['rgb'])
            goal = legacy.pointgoal_tensor(obs,args,device)
            before = float(env.get_metrics()['distance_to_goal'])
            initial = before
            total_reward, forwards, collisions, invalid = 0., 0, 0, 0
            collision_streak, max_collision_streak, success_reward_events = 0, 0, 0
            success, steps = 0., 0
            while not env.episode_over:
                with torch.no_grad():
                    logits, value = policy.forward_from_tokens(tok,goal)
                    dist = Categorical(logits=logits)
                    act = dist.sample()
                    logp = dist.log_prob(act)
                action = ACTIONS[act.item()]
                nxt = env.step({'action':action})
                steps += 1
                collided = bool(env.sim.previous_step_collided)
                forwards += int(action=='move_forward')
                collisions += int(collided and action=='move_forward')
                collision_streak = collision_streak+1 if collided and action=='move_forward' else 0
                max_collision_streak = max(max_collision_streak, collision_streak)
                after = float(env.get_metrics()['distance_to_goal'])
                reward, invalid_step = reward_delta(before,after,collided)
                invalid += invalid_step
                reached = float(nxt['pointgoal_with_gps_compass'][0]) <= args.success_distance
                stopped = False
                if reached and not env.episode_over:
                    nxt = env.step({'action':'stop'})
                    stopped = True
                    success = float(env.get_metrics()['success'])
                    reward += args.success_reward*success
                    success_reward_events += int(success > 0)
                boundary = bool(env.episode_over)
                nxt_tok = hist.push(nxt['rgb'])
                nxt_goal = legacy.pointgoal_tensor(nxt,args,device)
                with torch.no_grad():
                    # Time limit truncation bootstraps V(next); true STOP does not.
                    nv = 0. if stopped else policy.forward_from_tokens(nxt_tok,nxt_goal)[1].item()
                for k,v in zip(keys,(tok.squeeze(0).cpu(),goal.squeeze(0).cpu(),act.item(),
                                     logp.item(),value.item(),reward,nv,boundary)):
                    data[k].append(v)
                total_reward += reward
                obs,tok,goal,before = nxt,nxt_tok,nxt_goal,after
            row = dict(epoch=epoch,index=len(rows),episode_id=str(env.current_episode.episode_id),
                       scene_id=env.current_episode.scene_id,category=env.current_episode.info['curriculum_category'],
                       steps=steps,success=success,reward=total_reward,forward_actions=forwards,
                       max_consecutive_collision_steps=max_collision_streak,
                       success_reward_events=success_reward_events,success_reward_total=args.success_reward*success_reward_events,
                       collisions=collisions,collision_rate=collisions/max(forwards,1),
                       invalid_distance_steps=invalid,initial_distance=safe_number(initial),
                       final_distance=safe_number(before))
            rows.append(row)
            save_json(Path(args.output_dir)/'episodes'/f'epoch_{epoch:03d}'/f'{len(rows)-1:05d}.json',row)
            legacy.atomic_torch_save(rollout_path,dict(epoch=epoch,data=data,rows=rows,rng=rng_state()))
            live(args,'collect',epoch=epoch,steps=len(data['actions']),target=args.rollout_steps,
                 episodes=len(rows),scenes=len({r['scene_id'] for r in rows}),
                 categories=dict(Counter(r['category'] for r in rows)),last_episode=row)
            print(f'COLLECT epoch={epoch} steps={len(data["actions"])} episodes={len(rows)}',flush=True)
    finally:
        env.close()
    # Replay any interrupted optimizer stage from the same RNG state.
    saved=torch.load(rollout_path,map_location='cpu',weights_only=True)
    restore_rng(saved['rng'])
    return data,rows


def gae(data, gamma=.99, lam=.95):
    adv = np.zeros(len(data['rewards']), dtype=np.float32)
    running=0.
    for i in reversed(range(len(adv))):
        delta=data['rewards'][i]+gamma*data['next_values'][i]-data['values'][i]
        running=delta+gamma*lam*(not data['boundaries'][i])*running
        adv[i]=running
    return adv, adv+np.asarray(data['values'],dtype=np.float32)


def update(policy, optimizer, data, args, device, epoch):
    tokens,goals=torch.stack(data['tokens']),torch.stack(data['goals'])
    actions=torch.tensor(data['actions'])
    old=torch.tensor(data['log_probs'])
    adv,returns=gae(data)
    adv,returns=torch.from_numpy(adv),torch.from_numpy(returns)
    for x in (tokens,goals,old,adv,returns):
        if not torch.isfinite(x).all():
            raise RuntimeError('Non-finite rollout before update')
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    # Check *all* stored actions before changing any weights. Float32 tokens
    # retain exactly the observations used to obtain the behavior log probs.
    max_error=0.
    with torch.no_grad():
        for begin in range(0,len(actions),args.batch_size):
            ix=slice(begin,begin+args.batch_size)
            logits,_=policy.forward_from_tokens(tokens[ix].to(device),goals[ix].to(device))
            lp=Categorical(logits=logits).log_prob(actions[ix].to(device)).cpu()
            max_error=max(max_error,(lp-old[ix]).abs().max().item())
    if max_error>1e-4:
        raise RuntimeError(f'Behavior probability mismatch BEFORE update: {max_error}')
    params=[p for p in policy.parameters() if p.requires_grad]
    losses=[]
    for ppo_pass in range(args.ppo_epochs):
        order=torch.randperm(len(actions))
        # Keep exactly 256 samples per step; a fresh shuffled tail (<256)
        # is omitted each pass, with the full rollout still used for GAE.
        batches=list(order[:len(order)//args.batch_size*args.batch_size].split(args.batch_size))
        if not batches:raise RuntimeError('Rollout smaller than one minibatch')
        for b,ix in enumerate(batches):
            logits,v=policy.forward_from_tokens(tokens[ix].to(device),goals[ix].to(device))
            dist=Categorical(logits=logits)
            logratio=dist.log_prob(actions[ix].to(device))-old[ix].to(device)
            ratio=logratio.exp()
            a=adv[ix].to(device)
            actor=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
            critic=torch.nn.functional.mse_loss(v,returns[ix].to(device))
            entropy=dist.entropy().mean()
            loss=actor+.5*critic-.01*entropy
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,.5,error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in params):
                raise RuntimeError('Non-finite parameter after update; latest remains intact')
            losses.append([loss.item(),actor.item(),critic.item(),entropy.item(),norm.item()])
            live(args,'update',epoch=epoch,ppo_pass=ppo_pass+1,batch=b+1,batches=len(batches),
                 max_preupdate_logprob_error=max_error)
    return dict(mean_losses=np.mean(losses,axis=0).tolist(),max_preupdate_logprob_error=max_error,
                optimizer_steps=len(losses),samples=len(actions),peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20)


def aggregate(rows):
    def mean(k):
        vals=[r[k] for r in rows if r.get(k) is not None]
        return sum(vals)/len(vals) if vals else None
    return dict(episodes=len(rows),sr=mean('success'),spl=mean('spl'),
                mean_distance=mean('final_distance'),mean_steps=mean('steps'),
                mean_collisions=mean('collisions'),
                collision_rate=sum(r['collisions'] for r in rows)/max(sum(r['forward_actions'] for r in rows),1),
                mean_progress=mean('progress'))


@torch.no_grad()
def evaluate(policy,args,device,label,mode,split='validation'):
    policy.eval()
    path=Path(args.output_dir)/'protocol'/(split+'.json.gz')
    out=Path(args.output_dir)/'eval'/label/(split+'_'+mode+'.json')
    rows=json.loads(out.read_text())['rows'] if out.exists() else []
    all_rows=read_dataset(path)
    if len(rows)==len(all_rows): return json.loads(out.read_text())
    env=make_env(args,path)
    env.episodes=list(env.episodes)[len(rows):]
    donor_bank=None
    try:
        if mode=='shuffled':
            # Fixed permutation of initial RGBs, always from another scene;
            # bank and permutation are reused for all model evaluations.
            original=list(env.episodes)
            bank_path=Path(args.output_dir)/'protocol'/(split+'_rgb_bank.pth')
            if bank_path.exists():
                donor_bank=torch.load(bank_path,weights_only=True)
            else:
                bank_env=make_env(args,path)
                images=[]
                try:
                    for i in range(len(all_rows)):
                        images.append(torch.from_numpy(bank_env.reset()['rgb'].copy()))
                        live(args,'prepare_shuffled_rgb',split=split,completed=i+1,total=len(all_rows))
                finally:bank_env.close()
                donor_bank=[]
                for i,r in enumerate(all_rows):
                    j=next((i+d)%len(all_rows) for d in range(1,len(all_rows))
                           if scene_id(all_rows[(i+d)%len(all_rows)])!=scene_id(r))
                    donor_bank.append(images[j])
                legacy.atomic_torch_save(bank_path,donor_bank)
            env.episodes=original
        while len(rows)<len(all_rows):
            obs=env.reset(); hist=ObservationHistory(policy,device)
            start=float(env.get_metrics()['distance_to_goal'])
            steps,collisions,forward=0,0,0
            while not env.episode_over:
                if float(obs['pointgoal_with_gps_compass'][0])<=args.success_distance:
                    obs=env.step({'action':'stop'});steps+=1;break
                rgb=obs['rgb'] if mode=='normal' else (np.zeros_like(obs['rgb']) if mode=='black' else donor_bank[len(rows)].numpy())
                tok=hist.push(rgb);goal=legacy.pointgoal_tensor(obs,args,device)
                logits,_=policy.forward_from_tokens(tok,goal)
                idx=logits.argmax(1).item()
                obs=env.step({'action':ACTIONS[idx]});steps+=1
                forward+=int(idx==0)
                collisions+=int(idx==0 and env.sim.previous_step_collided)
            metrics=env.get_metrics();end=float(metrics['distance_to_goal'])
            row=dict(index=len(rows),episode_id=str(env.current_episode.episode_id),
                     scene_id=env.current_episode.scene_id,category=env.current_episode.info['curriculum_category'],
                     success=float(metrics['success']),spl=float(metrics['spl']),steps=steps,
                     forward_actions=forward,collisions=collisions,initial_distance=safe_number(start),
                     final_distance=safe_number(end),progress=safe_number(start-end))
            rows.append(row)
            payload=dict(complete=len(rows)==len(all_rows),summary=aggregate(rows),rows=rows,
                         by_category={c:aggregate([r for r in rows if r['category']==c]) for c in CATEGORIES})
            save_json(out,payload)
            live(args,'evaluate',label=label,split=split,mode=mode,completed=len(rows),total=len(all_rows),summary=payload['summary'])
            print(f'EVAL {label}/{split}/{mode} {len(rows)}/{len(all_rows)}',flush=True)
    finally: env.close()
    return payload


def build_policy(args,device):
    model=legacy.build_model(args,device)
    # Strictly verify initialization rather than relying on legacy strict=False.
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    source=ckpt.get('model',ckpt)
    if hasattr(source,'module'):source=source.module
    state=source.state_dict() if hasattr(source,'state_dict') else source
    model.load_state_dict(state,strict=True)
    policy=Policy(model,args.lookahead_idx,.25).to(device)
    policy.eval()
    assert not any(p.requires_grad for p in model.obs_encoder.parameters())
    return policy


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','task-config','scenes-dir','curriculum-dir','validation-dir','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--seed',type=int,default=270928)
    p.add_argument('--max-epochs',type=int,default=50)
    p.add_argument('--transition-epochs',type=int,default=10)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--rollout-steps',type=int,default=8192)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--ppo-epochs',type=int,default=4)
    p.add_argument('--eval-episodes',type=int,default=100)
    p.add_argument('--max-steps',type=int,default=500)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--lookahead-idx',type=int,default=2)
    p.add_argument('--model-waypoint-spacing',type=float,default=.12)
    p.add_argument('--success-distance',type=float,default=.2)
    p.add_argument('--success-reward',type=float,default=5.)
    p.add_argument('--skip-final-test',action='store_true')
    args=p.parse_args()
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device=torch.device('cuda');prepare_protocol(args)
    policy=build_policy(args,device);frozen=frozen_hash(policy)
    optim=torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad],lr=args.lr)
    save_json(root/'config.json',dict(vars(args),frozen_sha256=frozen,goal_dropout=0,goal_noise=0,
                                   collision_auxiliary_head=False,value_head=True,controller='angle categorical; argmax threshold 15deg'))
    latest=root/'latest_train.pth'
    history=[];start=0;best_sr=(-1.,-1.);best_spl=-1.;stale=0
    if latest.exists():
        if not args.resume:raise RuntimeError('Output has checkpoint; pass --resume or use new directory')
        state=torch.load(latest,map_location='cpu',weights_only=True)
        if state['frozen_sha256']!=frozen:raise RuntimeError('Initialization mismatch')
        policy.load_state_dict(state['policy']);optim.load_state_dict(state['optimizer'])
        history=state['history'];start=state['epoch']+1
        best_sr=tuple(state['best_sr']);best_spl=state['best_spl'];stale=state['stale']
        restore_rng(state['rng'])
    else:
        legacy.atomic_torch_save(latest,dict(epoch=-1,policy=policy.state_dict(),optimizer=optim.state_dict(),
            history=history,best_sr=best_sr,best_spl=best_spl,stale=0,rng=rng_state(),frozen_sha256=frozen))
    # baseline rows persist on disk; interrupted baseline evaluation is resumable.
    for mode in ('normal','black','shuffled'):
        evaluate(policy,args,device,'baseline',mode)
    if start==0:
        baseline=json.loads((root/'eval/baseline/validation_normal.json').read_text())['summary']
        best_sr=(baseline['sr'],baseline['spl']);best_spl=baseline['spl']
        for name in ('best_sr','best_spl'):
            if not (root/(name+'.pth')).exists():legacy.atomic_torch_save(root/(name+'.pth'),policy.model.state_dict())
    for epoch in range(start,args.max_epochs):
        dataset,weights=epoch_dataset(args,epoch)
        data,rows=collect(policy,args,device,epoch,dataset)
        stats=update(policy,optim,data,args,device,epoch)
        if frozen_hash(policy)!=frozen:raise RuntimeError('Frozen backbone changed')
        normal=evaluate(policy,args,device,f'epoch_{epoch:03d}','normal')['summary']
        black=evaluate(policy,args,device,f'epoch_{epoch:03d}','black')['summary']
        key=(normal['sr'],normal['spl']);improved=key>best_sr
        if improved:
            best_sr=key;legacy.atomic_torch_save(root/'best_sr.pth',policy.model.state_dict())
        if normal['spl']>best_spl:
            best_spl=normal['spl'];legacy.atomic_torch_save(root/'best_spl.pth',policy.model.state_dict())
        # Only start counting after 10-epoch transition +5 full-course epochs.
        stale=0 if improved or epoch<args.transition_epochs+5 else stale+1
        row=dict(epoch=epoch,normal=normal,black=black,update=stats,weights=weights,
                 train_episodes=len(rows),train_scenes=len({r['scene_id'] for r in rows}),
                 categories=dict(Counter(r['category'] for r in rows)),frozen_unchanged=True,stale=stale)
        history.append(row)
        state=dict(epoch=epoch,policy=policy.state_dict(),optimizer=optim.state_dict(),history=history,
                   best_sr=best_sr,best_spl=best_spl,stale=stale,rng=rng_state(),frozen_sha256=frozen)
        legacy.atomic_torch_save(latest,state)
        save_json(root/'history.json',dict(complete=False,history=history))
        print(json.dumps(row),flush=True)
        if stale>=args.patience:break
    if not args.smoke and not args.skip_final_test:
        for label,file in [('baseline_test',args.checkpoint),('best_sr_test',str(root/'best_sr.pth'))]:
            if label=='baseline_test':
                base=build_policy(args,device);policy.model.load_state_dict(base.model.state_dict());del base
            else:policy.model.load_state_dict(torch.load(file,map_location='cpu',weights_only=True))
            for mode in ('normal','black','shuffled'):evaluate(policy,args,device,label,mode,'test')
    if frozen_hash(policy)!=frozen:raise RuntimeError('Final backbone mismatch')
    save_json(root/'history.json',dict(complete=True,history=history))
    live(args,'complete',epochs=len(history),best_sr=best_sr,best_spl=best_spl)
    print('FINISHED_PPO_V2',flush=True)


if __name__=='__main__':main()
