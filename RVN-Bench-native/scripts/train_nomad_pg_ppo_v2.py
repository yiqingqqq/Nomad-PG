#!/usr/bin/env python3
"""NoMaD-PointGoal diffusion PPO on the frozen HM3D PointNav protocol.

The visual encoder and distance head remain frozen. PPO updates the diffusion
noise predictor and a fresh value head from exact stored denoising transitions.
No expert actions, shortest paths, or simulator poses are policy inputs.
"""
import argparse, copy, fcntl, gzip, json, math, random, time
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
from pathlib import Path

import habitat
import numpy as np
import torch

from agents.nomad_agent import load_nomad_model
from models.gnms_levin.train.vint_train.models.nomad.nomad_pointgoal import DenseNetwork
from models.model_utils.sequor_gnm_utils import get_action, get_diffusion_output
from utils.habitat_action_utils import get_discrete_control_output_from_trajectory_action_wo_stop
from utils.habitat_gnm_utils import get_gnm_obs_input_from_rgb_np

CATEGORIES = ("direct", "turn", "detour")


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); tmp.replace(path)


def atomic_save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(obj, tmp); tmp.replace(path)


def read_episodes(path):
    with gzip.open(path, "rt") as f: return json.load(f)["episodes"]


def write_episodes(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt") as f: json.dump({"episodes": rows}, f, allow_nan=False)
    tmp.replace(path)


def epoch_dataset(args, epoch):
    fixed = Path(args.protocol_dir) / f"train_epoch_{epoch:03d}.json.gz"
    if fixed.exists(): return fixed
    out = Path(args.output_dir) / "protocol" / fixed.name
    if out.exists(): return out
    rng = random.Random(args.seed + 1000 + epoch)
    t = min(epoch / max(args.transition_epochs - 1, 1), 1.0)
    weights = (.4 - .2*t, .3 - .1*t, .3 + .3*t)
    pools = {c: read_episodes(Path(args.curriculum_dir) / f"{c}.json.gz") for c in CATEGORIES}
    labels = [c for c, w in zip(CATEGORIES, weights) for _ in range(round(100*w))]
    rng.shuffle(labels); rows = []
    for c in labels:
        row = dict(rng.choice(pools[c]), shortest_paths=None)
        row["info"] = dict(row.get("info") or {}, curriculum_category=c)
        rows.append(row)
    write_episodes(out, rows); return out


def make_env(args, dataset):
    cfg = habitat.get_config(args.task_config)
    with habitat.config.read_write(cfg):
        cfg.habitat.dataset.data_path = str(dataset)
        cfg.habitat.dataset.scenes_dir = args.scenes_dir
        cfg.habitat.dataset.split = "val"
        cfg.habitat.environment.max_episode_steps = args.max_steps
        cfg.habitat.environment.iterator_options.shuffle = False
        cfg.habitat.environment.iterator_options.group_by_scene = False
        cfg.habitat.seed = args.seed
    return habitat.Env(config=cfg)


def rcs(pointgoal, args, device):
    r, theta = float(pointgoal[0]), float(pointgoal[1])
    if args.normalize_goal:
        r /= args.model_waypoint_spacing * args.waypoint_spacing * args.pred_horizon
    return torch.tensor([[r, math.cos(theta), math.sin(theta)]], dtype=torch.float32, device=device)


class ResidualFusionAdapter(torch.nn.Module):
    """Identity-initialized adapter on the frozen Transformer's pooled output."""

    def __init__(self, encoding_size, bottleneck, dropout):
        super().__init__()
        self.down = torch.nn.Linear(encoding_size, bottleneck)
        self.dropout = torch.nn.Dropout(dropout)
        self.up = torch.nn.Linear(bottleneck, encoding_size)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.dropout(torch.nn.functional.gelu(self.down(x))))


class Policy(torch.nn.Module):
    def __init__(self, agent, encoding_size, fusion_adapter_size=0, fusion_adapter_dropout=0.0):
        super().__init__()
        self.model, self.scheduler = agent._ema_model_avg, agent._noise_scheduler
        self.value_head = DenseNetwork(encoding_size)
        self.fusion_adapter = (ResidualFusionAdapter(
            encoding_size, fusion_adapter_size, fusion_adapter_dropout
        ) if fusion_adapter_size > 0 else torch.nn.Identity())
        for p in self.model.vision_encoder.parameters(): p.requires_grad_(False)
        for p in self.model.dist_pred_net.parameters(): p.requires_grad_(False)
        self.model.vision_encoder.eval(); self.model.dist_pred_net.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.vision_encoder.eval(); self.model.dist_pred_net.eval()
        return self

    @torch.no_grad()
    def encode(self, images, goal):
        mask = torch.zeros(images.shape[0], dtype=torch.long, device=images.device)
        return self.model("vision_encoder", obs_img=images, goal_pos=goal, input_goal_mask=mask)

    def actor_condition(self, cond): return self.fusion_adapter(cond)

    def value(self, cond): return self.value_head(cond).squeeze(-1)

    @torch.no_grad()
    def sample(self, cond, args):
        return get_diffusion_output(self.actor_condition(cond), self.model, self.scheduler, args.pred_horizon,
                                    2, cond.device, return_chain=True)

    def transition_stats(self, cond, prev, timestep, noise_pred_net=None):
        if noise_pred_net is None:
            pred = self.model("noise_pred_net", sample=prev, timestep=timestep,
                              global_cond=self.actor_condition(cond))
        else:
            # The anchor is the original frozen epoch-2 actor and therefore sees
            # the unadapted frozen Transformer representation.
            pred = noise_pred_net(sample=prev, timestep=timestep, global_cond=cond)
        means, variances = [], []
        for i in range(prev.shape[0]):
            # Diffusers' public DDPMScheduler.step() samples x_{t-1} but does
            # not expose the transition distribution required by PPO.  Build
            # the same mean and variance here while preserving gradients
            # through the noise predictor.
            t = int(timestep[i].item())
            sample, model_output = prev[i:i+1], pred[i:i+1]
            alpha_prod_t = self.scheduler.alphas_cumprod[t].to(sample.device)
            alpha_prod_t_prev = (self.scheduler.alphas_cumprod[t - 1].to(sample.device)
                                 if t > 0 else self.scheduler.one.to(sample.device))
            beta_prod_t, beta_prod_t_prev = 1 - alpha_prod_t, 1 - alpha_prod_t_prev
            prediction_type = self.scheduler.config.prediction_type
            if prediction_type == "epsilon":
                pred_original = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            elif prediction_type == "sample":
                pred_original = model_output
            elif prediction_type == "v_prediction":
                pred_original = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            else:
                raise ValueError(f"unsupported DDPM prediction_type: {prediction_type}")
            if self.scheduler.config.clip_sample:
                pred_original = pred_original.clamp(-1, 1)
            original_coeff = alpha_prod_t_prev.sqrt() * self.scheduler.betas[t].to(sample.device) / beta_prod_t
            sample_coeff = self.scheduler.alphas[t].to(sample.device).sqrt() * beta_prod_t_prev / beta_prod_t
            means.append(original_coeff * pred_original + sample_coeff * sample)
            variance = self.scheduler._get_variance(t).to(device=sample.device, dtype=sample.dtype)
            # _get_variance returns a standard deviation for fixed_small_log;
            # Normal below always expects an actual variance.
            if self.scheduler.config.variance_type == "fixed_small_log":
                variance = variance.square()
            variances.append(variance.reshape(1, 1, 1))
        return torch.cat(means), torch.cat(variances).clamp_min(1e-12)

    def logprob_entropy(self, cond, prev, nxt, timestep):
        mean, var = self.transition_stats(cond, prev, timestep)
        dist = torch.distributions.Normal(mean, var.sqrt())
        # Legacy DPPO trains the controller-relevant prefix and averages dimensions.
        return dist.log_prob(nxt)[:, :4].mean((1,2)), dist.entropy()[:, :4].mean((1,2))


def observation_input(history, args, device):
    return get_gnm_obs_input_from_rgb_np(np.stack(history), (96, 96), device)


def recovery_action(pointgoal, mode, alternate_left):
    if mode == "goal_turn":
        return "turn_left" if float(pointgoal[1]) >= 0 else "turn_right"
    if mode == "alternate":
        return "turn_left" if alternate_left else "turn_right"
    raise RuntimeError("recovery action requested while collision recovery is disabled")


def update_action_stats(action, counts, turn_streak):
    counts[action] += 1
    if action in ("turn_left", "turn_right"):
        turn_streak += 1
    else:
        turn_streak = 0
    return turn_streak


def action_summary(rows):
    counts = Counter()
    for row in rows:
        counts.update(row.get("action_counts", {}))
    total = sum(counts.values())
    forward = counts.get("move_forward", 0)
    turns = counts.get("turn_left", 0) + counts.get("turn_right", 0)
    return dict(
        action_counts=dict(counts),
        action_steps=total,
        forward_rate=float(forward / total) if total else 0.0,
        turn_rate=float(turns / total) if total else 0.0,
        max_consecutive_turns=max((int(r.get("max_consecutive_turns", 0)) for r in rows), default=0),
        recovery_actions=sum(int(r.get("recovery_actions", 0)) for r in rows),
    )


def collect(policy, args, device, epoch, dataset):
    keys = ("cond","prev","next","timestep","old_logp","value","reward","next_value","boundary")
    data = {k: [] for k in keys}; rows = []
    with make_env(args, dataset) as env:
        while len(data["reward"]) < args.rollout_steps:
            obs = env.reset(); history = deque([obs["rgb"].copy()]*4, maxlen=4)
            total = 0.; steps = policy_steps = collisions = forwards = 0
            recovery_remaining = recovery_actions = 0
            alternate_recovery_left = True
            forward_collision_streak = turn_streak = max_turn_streak = 0
            episode_action_counts = Counter()
            while not env.episode_over and len(data["reward"]) < args.rollout_steps:
                if recovery_remaining > 0:
                    action = recovery_action(
                        obs["pointgoal_with_gps_compass"], args.collision_recovery,
                        alternate_recovery_left,
                    )
                    before = float(env.get_metrics()["distance_to_goal"])
                    new = env.step({"action": action})
                    after = float(env.get_metrics()["distance_to_goal"])
                    collided = bool(env.sim.previous_step_collided)
                    if math.isfinite(before) and math.isfinite(after):
                        progress = float(np.clip(before-after, -args.progress_clip, args.progress_clip))
                    else:
                        progress = 0.0
                    total += args.progress_weight*progress - args.step_penalty
                    steps += 1; collisions += int(collided); recovery_actions += 1
                    recovery_remaining -= 1
                    turn_streak = update_action_stats(action, episode_action_counts, turn_streak)
                    max_turn_streak = max(max_turn_streak, turn_streak)
                    history.append(new["rgb"].copy()); obs = new
                    if env.episode_over and data["boundary"]:
                        # Recovery is outside the learned policy.  If it terminates the
                        # episode, close the preceding learned transition for GAE.
                        data["boundary"][-1] = True
                        data["next_value"][-1] = 0.0
                    continue
                image = observation_input(history, args, device); goal = rcs(obs["pointgoal_with_gps_compass"], args, device)
                cond = policy.encode(image, goal); value = policy.value(cond).item(); sample = policy.sample(cond, args)
                chain = sample["chain"]; hi = min(args.denoise_step_max, chain.shape[1]-2)
                idx = random.randint(args.denoise_step_min, hi)
                prev, nxt = chain[:, idx], chain[:, idx+1]
                timestep = policy.scheduler.timesteps[idx].reshape(1).to(device)
                old_logp = policy.logprob_entropy(cond, prev, nxt, timestep)[0].item()
                traj = get_action(sample["diffusion_output"]).cpu().numpy()
                action = get_discrete_control_output_from_trajectory_action_wo_stop(traj, lookahead_point_idx=args.lookahead_idx)
                before = float(env.get_metrics()["distance_to_goal"]); new = env.step({"action": action})
                collided = bool(env.sim.previous_step_collided); after = float(env.get_metrics()["distance_to_goal"])
                if math.isfinite(before) and math.isfinite(after):
                    progress = float(np.clip(before-after, -args.progress_clip, args.progress_clip))
                else:
                    progress = 0.0
                    print(f"WARNING non-finite distance episode={env.current_episode.episode_id} "
                          f"step={steps} before={before} after={after}; using zero progress",flush=True)
                reward = (args.progress_weight*progress - args.step_penalty
                          - args.collision_penalty*int(collided and action == "move_forward"))
                reached = float(new["pointgoal_with_gps_compass"][0]) <= args.success_distance
                stopped = False
                if reached and not env.episode_over:
                    new = env.step({"action":"stop"}); stopped = True
                    reward += args.success_reward*float(env.get_metrics().get("success", 0.))
                history.append(new["rgb"].copy())
                if stopped: next_value = 0.
                else:
                    ni = observation_input(history,args,device); ng = rcs(new["pointgoal_with_gps_compass"],args,device)
                    next_value = policy.value(policy.encode(ni,ng)).item()
                vals = (cond.squeeze(0).cpu(),prev.squeeze(0).cpu(),nxt.squeeze(0).cpu(),int(timestep.item()),
                        old_logp,value,reward,next_value,bool(env.episode_over))
                for k,v in zip(keys,vals): data[k].append(v)
                total += reward; steps += 1; policy_steps += 1
                forwards += action=="move_forward"; collisions += collided and action=="move_forward"
                turn_streak = update_action_stats(action, episode_action_counts, turn_streak)
                max_turn_streak = max(max_turn_streak, turn_streak)
                if action == "move_forward":
                    forward_collision_streak = forward_collision_streak + 1 if collided else 0
                    if (args.collision_recovery != "none" and args.recovery_turn_steps > 0
                            and forward_collision_streak >= args.recovery_collision_threshold):
                        recovery_remaining = args.recovery_turn_steps
                        alternate_recovery_left = not alternate_recovery_left
                        forward_collision_streak = 0
                obs = new
            metrics=env.get_metrics(); episode_spl=float(metrics.get("spl",0.))
            if not math.isfinite(episode_spl):
                print(f"WARNING non-finite SPL episode={env.current_episode.episode_id} "
                      f"scene={env.current_episode.scene_id}; counting as zero",flush=True)
                episode_spl=0.0
            rows.append(dict(epoch=epoch, episode_id=str(env.current_episode.episode_id),
                scene_id=str(env.current_episode.scene_id), category=(env.current_episode.info or {}).get("curriculum_category"),
                steps=steps, policy_steps=policy_steps, reward=total, collisions=collisions, forward_actions=forwards,
                recovery_actions=recovery_actions, action_counts=dict(episode_action_counts),
                max_consecutive_turns=max_turn_streak,
                success=float(metrics.get("success",0.)), spl=episode_spl))
            print(f"COLLECT epoch={epoch} steps={len(data['reward'])} episodes={len(rows)}", flush=True)
    return data, rows


def advantages(data, gamma=.99, lam=.95):
    adv=np.zeros(len(data["reward"]),np.float32); run=0.
    for i in reversed(range(len(adv))):
        delta=data["reward"][i]+gamma*data["next_value"][i]-data["value"][i]
        run=delta+gamma*lam*(not data["boundary"][i])*run; adv[i]=run
    return adv, adv+np.asarray(data["value"],np.float32)


def update(policy, optimizer, data, args, device, anchor_noise_pred=None, anchor_parameters=None):
    tensors={k:torch.stack(data[k]) for k in ("cond","prev","next")}
    ts=torch.tensor(data["timestep"],dtype=torch.long); old=torch.tensor(data["old_logp"])
    adv,ret=advantages(data,args.gamma,args.gae_lambda); adv=torch.from_numpy(adv); ret=torch.from_numpy(ret)
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8); n=len(old); losses=[]
    with torch.no_grad():
        checks=[]
        for ix in torch.arange(n).split(args.batch_size):
            checks.append(policy.logprob_entropy(tensors["cond"][ix].to(device),
                tensors["prev"][ix].to(device),tensors["next"][ix].to(device),ts[ix].to(device))[0].cpu())
        diff=(torch.cat(checks)-old).abs()
        diagnostic=dict(max=float(diff.max()),p99=float(torch.quantile(diff.float(),.99)),
            mean=float(diff.mean()),over_warn=int((diff>args.logprob_warn_tolerance).sum()),samples=n)
        print("LOGPROB_CHECK "+json.dumps(diagnostic),flush=True)
        if diagnostic["max"]>args.logprob_tolerance:
            raise RuntimeError(f"stored behavior log-probability mismatch: {diagnostic}")
        if diagnostic["max"]>args.logprob_warn_tolerance:
            print(f"WARNING log-probability drift exceeds {args.logprob_warn_tolerance}: {diagnostic}",flush=True)
    for _ in range(args.ppo_epochs):
        for ix in torch.randperm(n).split(args.batch_size):
            cond=tensors["cond"][ix].to(device); prev=tensors["prev"][ix].to(device); nxt=tensors["next"][ix].to(device)
            logp,entropy=policy.logprob_entropy(cond,prev,nxt,ts[ix].to(device))
            # A changed diffusion mean can make Gaussian log-ratios enormous;
            # exponentiating them directly overflows and corrupts the policy.
            log_ratio=(logp-old[ix].to(device)).clamp(-args.max_log_ratio,args.max_log_ratio)
            ratio=log_ratio.exp()
            a=adv[ix].to(device); actor=-torch.minimum(ratio*a,ratio.clamp(1-args.clip,1+args.clip)*a).mean()
            value=policy.value(cond); critic=.5*(value-ret[ix].to(device)).square().mean()
            anchor_kl = torch.zeros((), device=device)
            if anchor_noise_pred is not None and args.anchor_kl_coef > 0:
                current_mean, _ = policy.transition_stats(cond, prev, ts[ix].to(device))
                with torch.no_grad():
                    anchor_mean, anchor_var = policy.transition_stats(
                        cond, prev, ts[ix].to(device), noise_pred_net=anchor_noise_pred
                    )
                anchor_kl = (0.5 * (current_mean-anchor_mean).square()
                             / anchor_var)[:, :4].mean()
            anchor_l2 = torch.zeros((), device=device)
            if anchor_parameters and args.anchor_l2_coef > 0:
                squared = torch.zeros((), device=device); elements = 0
                for name, parameter in policy.model.noise_pred_net.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    squared = squared + (parameter-anchor_parameters[name]).square().sum()
                    elements += parameter.numel()
                anchor_l2 = squared / max(elements, 1)
            loss=(actor+args.value_coef*critic-args.entropy_coef*entropy.mean()
                  +args.anchor_kl_coef*anchor_kl+args.anchor_l2_coef*anchor_l2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite PPO loss: actor={actor.item()} critic={critic.item()}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],
                                           args.max_grad_norm,error_if_nonfinite=True)
            optimizer.step(); losses.append((loss.item(),actor.item(),critic.item(),entropy.mean().item(),
                                             anchor_kl.item(),anchor_l2.item()))
    a=np.asarray(losses); return dict(loss=a[:,0].mean(),actor=a[:,1].mean(),critic=a[:,2].mean(),
        entropy=a[:,3].mean(),anchor_kl=a[:,4].mean(),anchor_l2=a[:,5].mean())


@torch.no_grad()
def evaluate(policy, args, device, label, split="validation", rgb_mode="normal", goal_mode="normal"):
    """Evaluate only on immutable protocol files; no evaluator signal enters policy."""
    rows=[]; dataset=Path(args.protocol_dir)/f"{split}.json.gz"
    with make_env(args,dataset) as env:
        total=min(args.eval_episodes,len(env.episodes))
        for episode_index in range(total):
            # Stable per-episode diffusion noise makes model comparisons paired.
            torch.manual_seed(args.seed+100000+episode_index)
            obs=env.reset(); history=deque([obs["rgb"].copy()]*4,maxlen=4); steps=collisions=0
            recovery_remaining = recovery_actions = 0
            alternate_recovery_left = True
            forward_collision_streak = turn_streak = max_turn_streak = 0
            episode_action_counts = Counter()
            while not env.episode_over:
                pointgoal = np.asarray(obs["pointgoal_with_gps_compass"]).copy()
                if goal_mode == "angle_flip":
                    pointgoal[1] = -pointgoal[1]
                if recovery_remaining > 0:
                    action = recovery_action(pointgoal, args.collision_recovery, alternate_recovery_left)
                    recovery_remaining -= 1; recovery_actions += 1
                else:
                    rgb=np.stack(history)
                    if rgb_mode=="black": rgb=np.zeros_like(rgb)
                    image=get_gnm_obs_input_from_rgb_np(rgb,(96,96),device)
                    goal=rcs(pointgoal,args,device); cond=policy.encode(image,goal)
                    sample=policy.sample(cond,args); traj=get_action(sample["diffusion_output"]).cpu().numpy()
                    action=get_discrete_control_output_from_trajectory_action_wo_stop(traj,lookahead_point_idx=args.lookahead_idx)
                turn_streak = update_action_stats(action, episode_action_counts, turn_streak)
                max_turn_streak = max(max_turn_streak, turn_streak)
                obs=env.step({"action":action}); history.append(obs["rgb"].copy()); steps+=1
                collided = bool(env.sim.previous_step_collided); collisions+=int(collided)
                if action == "move_forward":
                    forward_collision_streak = forward_collision_streak + 1 if collided else 0
                    if (args.collision_recovery != "none" and args.recovery_turn_steps > 0
                            and forward_collision_streak >= args.recovery_collision_threshold):
                        recovery_remaining = args.recovery_turn_steps
                        alternate_recovery_left = not alternate_recovery_left
                        forward_collision_streak = 0
                if float(obs["pointgoal_with_gps_compass"][0])<=args.success_distance and not env.episode_over:
                    obs=env.step({"action":"stop"})
            m=env.get_metrics(); rows.append(dict(episode_id=str(env.current_episode.episode_id),
                scene_id=str(env.current_episode.scene_id),success=float(m.get("success",0.)),
                spl=float(m.get("spl",0.)),distance_to_goal=float(m.get("distance_to_goal",0.)),
                steps=steps,collisions=collisions,recovery_actions=recovery_actions,
                action_counts=dict(episode_action_counts),max_consecutive_turns=max_turn_streak))
    summary=dict(episodes=len(rows),sr=float(np.mean([r["success"] for r in rows])),
                 spl=float(np.mean([r["spl"] for r in rows])),
                 distance=float(np.mean([r["distance_to_goal"] for r in rows])),
                 mean_steps=float(np.mean([r["steps"] for r in rows])),
                 mean_collisions=float(np.mean([r["collisions"] for r in rows])),
                 rgb_mode=rgb_mode,goal_mode=goal_mode,**action_summary(rows))
    condition = "angle_flip" if goal_mode == "angle_flip" else rgb_mode
    atomic_json(Path(args.output_dir)/"eval"/label/f"{split}_{condition}.json",dict(summary=summary,rows=rows))
    return summary


@contextmanager
def validation_guard(args):
    """Serialize fixed validation across independently running PPO jobs."""
    if not args.validation_lock:
        with nullcontext():
            yield
        return
    lock_path = Path(args.validation_lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        print(f"VALIDATION_LOCK waiting path={lock_path}", flush=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print(f"VALIDATION_LOCK acquired path={lock_path}", flush=True)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(f"VALIDATION_LOCK released path={lock_path}", flush=True)


def configure_trainable_modules(policy, args):
    """Apply the full, local-diffusion, or local-diffusion+adapter freeze plan."""
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.value_head.parameters():
        parameter.requires_grad_(True)

    noise = policy.model.noise_pred_net
    if args.train_scheme == "full":
        for parameter in noise.parameters():
            parameter.requires_grad_(True)
    else:
        if not hasattr(noise, "up_modules") or not len(noise.up_modules):
            raise RuntimeError("selective diffusion training requires noise_pred_net.up_modules")
        if not hasattr(noise, "final_conv"):
            raise RuntimeError("selective diffusion training requires noise_pred_net.final_conv")
        for parameter in noise.up_modules[-1].parameters():
            parameter.requires_grad_(True)
        for parameter in noise.final_conv.parameters():
            parameter.requires_grad_(True)
    if args.train_scheme == "scheme_b":
        for parameter in policy.fusion_adapter.parameters():
            parameter.requires_grad_(True)


def trainable_parameter_report(policy):
    groups = {}
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            prefix = ("critic" if name.startswith("value_head.") else
                      "adapter" if name.startswith("fusion_adapter.") else
                      "diffusion_output" if ".final_conv." in name else
                      "diffusion_last_up" if ".up_modules." in name else "other")
            groups[prefix] = groups.get(prefix, 0) + parameter.numel()
    groups["total"] = sum(groups.values())
    return groups


def build_optimizer(policy, args):
    value_parameters = [p for p in policy.value_head.parameters() if p.requires_grad]
    groups = []
    noise = policy.model.noise_pred_net
    if args.train_scheme == "full":
        if args.actor_lr == args.critic_lr:
            groups.append(dict(params=[p for p in policy.parameters() if p.requires_grad],
                               lr=args.actor_lr, name="actor_critic"))
        else:
            groups.append(dict(params=value_parameters, lr=args.critic_lr, name="critic"))
            groups.append(dict(params=[p for p in noise.parameters() if p.requires_grad],
                               lr=args.actor_lr, name="actor"))
    else:
        groups.append(dict(params=value_parameters, lr=args.critic_lr, name="critic"))
        groups.append(dict(params=[p for p in noise.up_modules[-1].parameters() if p.requires_grad],
                           lr=args.diffusion_last_up_lr, name="diffusion_last_up"))
        groups.append(dict(params=[p for p in noise.final_conv.parameters() if p.requires_grad],
                           lr=args.diffusion_output_lr, name="diffusion_output"))
    if args.train_scheme == "scheme_b":
        groups.append(dict(params=[p for p in policy.fusion_adapter.parameters() if p.requires_grad],
                           lr=args.adapter_lr, name="fusion_adapter"))
    if any(not group["params"] for group in groups):
        raise RuntimeError(f"empty optimizer parameter group: {[g['name'] for g in groups if not g['params']]}")
    return torch.optim.AdamW(groups)


def build(args, device):
    agent=load_nomad_model(False,None,"nomad_pg.yaml",args.checkpoint)
    adapter_size = args.adapter_bottleneck if args.train_scheme == "scheme_b" else 0
    policy=Policy(agent,args.encoding_size,adapter_size,args.adapter_dropout).to(device)
    configure_trainable_modules(policy,args)
    policy.train(True)
    return policy


def load_policy_state(path):
    state=torch.load(path,map_location="cpu",weights_only=False)
    return state["policy"] if isinstance(state,dict) and "policy" in state else state


def load_initial_policy(policy, path):
    missing, unexpected = policy.load_state_dict(load_policy_state(path), strict=False)
    allowed_missing = {name for name in policy.state_dict() if name.startswith("fusion_adapter.")}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"incompatible policy state {path}: missing={missing}, unexpected={unexpected}")


def build_anchor_regularizer(policy, anchor_path, device):
    if not anchor_path:
        return None, None
    state=load_policy_state(anchor_path)
    prefix="model.noise_pred_net."
    noise_state={name[len(prefix):]:value for name,value in state.items() if name.startswith(prefix)}
    if not noise_state:
        raise RuntimeError(f"anchor contains no {prefix} parameters: {anchor_path}")
    anchor_noise_pred=copy.deepcopy(policy.model.noise_pred_net).to(device)
    anchor_noise_pred.load_state_dict(noise_state,strict=True); anchor_noise_pred.eval()
    for parameter in anchor_noise_pred.parameters(): parameter.requires_grad_(False)
    anchor_parameters={name:parameter.detach().clone()
                       for name,parameter in anchor_noise_pred.named_parameters()}
    return anchor_noise_pred,anchor_parameters


def main():
    p=argparse.ArgumentParser()
    for n in ("checkpoint","task-config","scenes-dir","curriculum-dir","protocol-dir","output-dir"): p.add_argument("--"+n,required=True)
    p.add_argument("--resume",action="store_true"); p.add_argument("--seed",type=int,default=270928)
    p.add_argument("--collect-only-epoch",type=int,default=None)
    p.add_argument("--policy-state",default=None); p.add_argument("--rollout-output",default=None)
    p.add_argument("--anchor-policy",default=None,
                   help="Frozen full Policy state used for initialization and anti-forgetting regularization.")
    p.add_argument("--eval-only-policy-state",default=None)
    p.add_argument("--eval-only-split",choices=("validation","test"),default="validation")
    p.add_argument("--eval-only-label",default="anchor")
    p.add_argument("--eval-only-rgb-mode",choices=("normal","black"),default="normal")
    p.add_argument("--eval-only-goal-mode",choices=("normal","angle_flip"),default="normal")
    p.add_argument("--max-epochs",type=int,default=50); p.add_argument("--transition-epochs",type=int,default=10)
    p.add_argument("--rollout-steps",type=int,default=8192); p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--ppo-epochs",type=int,default=4); p.add_argument("--max-steps",type=int,default=500)
    p.add_argument("--eval-episodes",type=int,default=100)
    p.add_argument("--lr",type=float,default=1e-5); p.add_argument("--clip",type=float,default=.2)
    p.add_argument("--gamma",type=float,default=.99)
    p.add_argument("--gae-lambda",type=float,default=.95)
    p.add_argument("--actor-lr",type=float,default=None)
    p.add_argument("--critic-lr",type=float,default=None)
    p.add_argument("--diffusion-last-up-lr",type=float,default=1e-6)
    p.add_argument("--diffusion-output-lr",type=float,default=2e-6)
    p.add_argument("--adapter-lr",type=float,default=5e-7)
    p.add_argument("--train-scheme",choices=("full","scheme_a","scheme_b"),default="full")
    p.add_argument("--adapter-bottleneck",type=int,default=32)
    p.add_argument("--adapter-dropout",type=float,default=.05)
    p.add_argument("--max-log-ratio",type=float,default=10.)
    p.add_argument("--value-coef",type=float,default=.5); p.add_argument("--entropy-coef",type=float,default=0.)
    p.add_argument("--max-grad-norm",type=float,default=1.); p.add_argument("--success-distance",type=float,default=.2)
    p.add_argument("--progress-weight",type=float,default=1.0)
    p.add_argument("--progress-clip",type=float,default=.25)
    p.add_argument("--step-penalty",type=float,default=.01)
    p.add_argument("--collision-penalty",type=float,default=.10)
    p.add_argument("--success-reward",type=float,default=5.0)
    p.add_argument("--model-waypoint-spacing",type=float,default=.12); p.add_argument("--waypoint-spacing",type=int,default=1)
    p.add_argument("--pred-horizon",type=int,default=8); p.add_argument("--encoding-size",type=int,default=256)
    p.add_argument("--lookahead-idx",type=int,default=2); p.add_argument("--denoise-step-min",type=int,default=1)
    p.add_argument("--denoise-step-max",type=int,default=4)
    p.add_argument("--logprob-warn-tolerance",type=float,default=1e-3)
    p.add_argument("--logprob-tolerance",type=float,default=1e-2)
    p.add_argument("--anchor-kl-coef",type=float,default=0.0)
    p.add_argument("--anchor-l2-coef",type=float,default=0.0)
    p.add_argument("--collision-recovery",choices=("none","goal_turn","alternate"),default="none")
    p.add_argument("--recovery-turn-steps",type=int,default=0)
    p.add_argument("--recovery-collision-threshold",type=int,default=1)
    p.add_argument("--min-validation-forward-rate",type=float,default=0.0)
    p.add_argument("--max-validation-turn-streak",type=int,default=0)
    p.add_argument("--min-validation-sr",type=float,default=0.0)
    p.add_argument("--min-normal-black-sr-gap",type=float,default=-1.0)
    p.add_argument("--max-action-kl",type=float,default=0.0)
    p.add_argument("--no-improvement-patience",type=int,default=0)
    p.add_argument("--black-eval-interval",type=int,default=1,
                   help="Run black-image validation every N epochs; the final epoch is always evaluated.")
    p.add_argument("--validation-lock",default=None)
    p.add_argument("--baseline-eval-source",default=None,
                   help="Directory containing immutable validation_normal/black.json for the identical anchor.")
    p.add_argument("--no-normalize-goal",dest="normalize_goal",action="store_false"); args=p.parse_args()
    if args.actor_lr is None: args.actor_lr=args.lr
    if args.critic_lr is None: args.critic_lr=args.lr
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required; refusing to start PPO on CPU")
    if args.denoise_step_min<0 or args.denoise_step_min>args.denoise_step_max: raise ValueError("invalid denoise range")
    if args.logprob_warn_tolerance<=0 or args.logprob_warn_tolerance>=args.logprob_tolerance:
        raise ValueError("log-probability tolerances must satisfy 0 < warn < fail")
    if (args.anchor_kl_coef>0 or args.anchor_l2_coef>0) and not args.anchor_policy:
        raise ValueError("anchor regularization requires --anchor-policy")
    if args.collision_recovery=="none" and args.recovery_turn_steps:
        raise ValueError("recovery turn steps require collision recovery")
    if args.train_scheme=="scheme_b" and args.adapter_bottleneck<=0:
        raise ValueError("scheme_b requires a positive adapter bottleneck")
    if args.black_eval_interval<=0:
        raise ValueError("black-eval-interval must be positive")
    root=Path(args.output_dir); root.mkdir(parents=True,exist_ok=True)
    for name in ("manifest.json","validation.json.gz","test.json.gz"):
        if not (Path(args.protocol_dir)/name).is_file(): raise FileNotFoundError(Path(args.protocol_dir)/name)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device("cuda")
    policy=build(args,device)
    if args.eval_only_policy_state:
        load_initial_policy(policy,args.eval_only_policy_state)
        with validation_guard(args):
            summary=evaluate(policy,args,device,args.eval_only_label,split=args.eval_only_split,
                             rgb_mode=args.eval_only_rgb_mode,goal_mode=args.eval_only_goal_mode)
        print(json.dumps(summary,sort_keys=True)); return
    if args.collect_only_epoch is not None:
        if not args.policy_state or not args.rollout_output:
            raise ValueError("collect-only mode requires --policy-state and --rollout-output")
        state=torch.load(args.policy_state,map_location="cpu",weights_only=False)
        policy.load_state_dict(state["policy"])
        data,rows=collect(policy,args,device,args.collect_only_epoch,
                          epoch_dataset(args,args.collect_only_epoch))
        atomic_save(args.rollout_output,dict(epoch=args.collect_only_epoch,data=data,rows=rows,
                    policy_state=str(Path(args.policy_state).resolve())))
        print(f"COLLECT_ONLY saved epoch={args.collect_only_epoch} steps={len(data['reward'])} "
              f"output={args.rollout_output}",flush=True)
        return
    latest=root/"latest_train.pth"; start=0; history=[]; best=(-1.,-1.,float("-inf"),float("-inf")); stale=0
    if args.anchor_policy and not latest.exists():
        load_initial_policy(policy,args.anchor_policy)
    optimizer=build_optimizer(policy,args)
    if latest.exists():
        if not args.resume: raise RuntimeError("output checkpoint exists; pass --resume or choose a new output")
        # This checkpoint is produced locally by this trainer and includes
        # optimizer/history values containing NumPy scalar objects. PyTorch
        # 2.6+ rejects those metadata objects in weights-only mode.
        state=torch.load(latest,map_location="cpu",weights_only=False); policy.load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"]); history=state["history"]; start=state["epoch"]+1
        loaded_best=tuple(state.get("best",best))
        best=loaded_best if len(loaded_best)==4 else (loaded_best[0],loaded_best[1],float("-inf"),float("-inf"))
        stale=int(state.get("stale",0))
    anchor_noise_pred,anchor_parameters=build_anchor_regularizer(policy,args.anchor_policy,device)
    config=dict(vars(args),trainable_parameters=trainable_parameter_report(policy),
                optimizer_groups=[dict(name=g.get("name"),lr=g["lr"],parameters=sum(p.numel() for p in g["params"]))
                                  for g in optimizer.param_groups])
    atomic_json(root/"config.json",config)
    print("TRAINABLE_PARAMETERS "+json.dumps(config["trainable_parameters"]),flush=True)
    print("OPTIMIZER_GROUPS "+json.dumps(config["optimizer_groups"]),flush=True)
    if start==0:
        normal_path=root/"eval"/"baseline"/"validation_normal.json"
        black_path=root/"eval"/"baseline"/"validation_black.json"
        if args.baseline_eval_source:
            source=Path(args.baseline_eval_source)
            for destination,name in ((normal_path,"validation_normal.json"),(black_path,"validation_black.json")):
                source_path=source/name
                if not destination.is_file() and source_path.is_file():
                    atomic_json(destination,json.loads(source_path.read_text()))
                    print(f"BASELINE reused source={source_path} destination={destination}",flush=True)
        if normal_path.is_file() and black_path.is_file():
            baseline=json.loads(normal_path.read_text())["summary"]
            baseline_black=json.loads(black_path.read_text())["summary"]
        else:
            with validation_guard(args):
                baseline=(json.loads(normal_path.read_text())["summary"] if normal_path.is_file()
                          else evaluate(policy,args,device,"baseline"))
                baseline_black=(json.loads(black_path.read_text())["summary"] if black_path.is_file()
                                else evaluate(policy,args,device,"baseline",rgb_mode="black"))
        best=(baseline["sr"],baseline["spl"],-baseline.get("mean_collisions",float("inf")),
              baseline["sr"]-baseline_black["sr"])
        atomic_save(root/"best_sr.pth",policy.state_dict())
    rollout_cache=root/"rollout_latest.pth"; post_update=root/"post_update.pth"
    for epoch in range(start,args.max_epochs):
        prefetched=root/f"rollout_epoch_{epoch:03d}.pth"
        active_rollout=prefetched if prefetched.is_file() else rollout_cache
        if active_rollout.is_file():
            cached=torch.load(active_rollout,map_location="cpu",weights_only=False)
            if cached.get("epoch")!=epoch:
                raise RuntimeError(f"stale rollout cache for epoch {cached.get('epoch')}, expected {epoch}")
            data,rows=cached["data"],cached["rows"]
            print(f"ROLLOUT_CACHE loaded epoch={epoch} steps={len(data['reward'])} path={active_rollout}",flush=True)
        else:
            data,rows=collect(policy,args,device,epoch,epoch_dataset(args,epoch))
            atomic_save(active_rollout,dict(epoch=epoch,data=data,rows=rows))
            print(f"ROLLOUT_CACHE saved epoch={epoch} steps={len(data['reward'])}",flush=True)
        invalid_rewards=sum(not math.isfinite(float(r)) for r in data["reward"])
        if invalid_rewards:
            data["reward"]=[float(r) if math.isfinite(float(r)) else -args.step_penalty for r in data["reward"]]
            print(f"ROLLOUT_CACHE sanitized_nonfinite_rewards={invalid_rewards} "
                  f"replacement={-args.step_penalty}",flush=True)
        # Rebuild per-episode reward totals from the sanitized step rewards and
        # count invalid Habitat metrics as failures instead of emitting NaN.
        offset=0; invalid_episode_metrics=0
        for row in rows:
            end=offset+int(row.get("policy_steps",row["steps"]));
            row["reward"]=float(sum(data["reward"][offset:end])); offset=end
            for key in ("success","spl"):
                if not math.isfinite(float(row[key])):
                    row[key]=0.0; invalid_episode_metrics+=1
        if offset!=len(data["reward"]):
            raise RuntimeError(f"rollout row/step mismatch: rows={offset} data={len(data['reward'])}")
        if invalid_rewards or invalid_episode_metrics:
            atomic_save(active_rollout,dict(epoch=epoch,data=data,rows=rows))
            print(f"ROLLOUT_CACHE repaired invalid_rewards={invalid_rewards} "
                  f"invalid_episode_metrics={invalid_episode_metrics}",flush=True)
        if post_update.is_file():
            state=torch.load(post_update,map_location="cpu",weights_only=False)
            if state.get("epoch")!=epoch:
                raise RuntimeError(f"stale post-update checkpoint for epoch {state.get('epoch')}, expected {epoch}")
            policy.load_state_dict(state["policy"]); optimizer.load_state_dict(state["optimizer"]); stats=state["stats"]
            print(f"POST_UPDATE loaded epoch={epoch}",flush=True)
        else:
            stats=update(policy,optimizer,data,args,device,anchor_noise_pred,anchor_parameters)
            atomic_save(post_update,dict(epoch=epoch,policy=policy.state_dict(),optimizer=optimizer.state_dict(),stats=stats))
            print(f"POST_UPDATE saved epoch={epoch}",flush=True)
        eval_root=root/"eval"/f"epoch_{epoch:03d}"
        normal_path=eval_root/"validation_normal.json"; black_path=eval_root/"validation_black.json"
        run_black = epoch % args.black_eval_interval == 0 or epoch == args.max_epochs - 1
        with validation_guard(args):
            normal=(json.loads(normal_path.read_text())["summary"] if normal_path.is_file()
                    else evaluate(policy,args,device,f"epoch_{epoch:03d}"))
            black=(json.loads(black_path.read_text())["summary"] if black_path.is_file()
                   else evaluate(policy,args,device,f"epoch_{epoch:03d}",rgb_mode="black")) if run_black else None
        sr_gap=normal["sr"]-black["sr"] if black is not None else None
        unsafe=[]
        if args.min_validation_sr > 0 and normal["sr"] < args.min_validation_sr:
            unsafe.append(f"normal_sr={normal['sr']:.6f}<{args.min_validation_sr:.6f}")
        if (args.min_validation_forward_rate > 0
                and normal.get("forward_rate",0.0) < args.min_validation_forward_rate):
            unsafe.append(f"forward_rate={normal.get('forward_rate',0.0):.6f}"
                          f"<{args.min_validation_forward_rate:.6f}")
        if (args.max_validation_turn_streak > 0
                and normal.get("max_consecutive_turns",0) > args.max_validation_turn_streak):
            unsafe.append(f"max_consecutive_turns={normal.get('max_consecutive_turns',0)}"
                          f">{args.max_validation_turn_streak}")
        if (black is not None and args.min_normal_black_sr_gap >= 0
                and sr_gap < args.min_normal_black_sr_gap):
            unsafe.append(f"normal_black_sr_gap={sr_gap:.6f}<{args.min_normal_black_sr_gap:.6f}")
        if args.max_action_kl > 0 and stats.get("anchor_kl",0.0) > args.max_action_kl:
            unsafe.append(f"anchor_kl={stats.get('anchor_kl',0.0):.6f}>{args.max_action_kl:.6f}")
        candidate=(normal["sr"],normal["spl"],-normal.get("mean_collisions",float("inf")),
                   sr_gap if sr_gap is not None else best[3])
        improved_sr_or_spl = normal["sr"] > best[0] or normal["spl"] > best[1]
        stale = 0 if improved_sr_or_spl else stale + 1
        if args.no_improvement_patience > 0 and stale >= args.no_improvement_patience:
            unsafe.append(f"no_sr_or_spl_improvement_epochs={stale}>={args.no_improvement_patience}")
        if unsafe:
            rejected_state={name:value.detach().cpu().clone()
                            for name,value in policy.state_dict().items()}
            policy.load_state_dict(torch.load(root/"best_sr.pth",map_location="cpu",weights_only=True))
            optimizer=build_optimizer(policy,args)
            summary=dict(epoch=epoch,update=stats,episodes=len(rows),steps=len(data["reward"]),
                success=float(np.mean([r["success"] for r in rows])),spl=float(np.mean([r["spl"] for r in rows])),
                validation=normal,black=black,black_evaluated=black is not None,normal_black_sr_gap=sr_gap,
                categories=dict(Counter(r["category"] for r in rows)),
                rollout_actions=action_summary(rows),safety_status="rolled_back",safety_reasons=unsafe,
                timestamp=time.time())
            history.append(summary); atomic_json(root/"history.json",history)
            atomic_save(root/"rejected_update.pth",dict(epoch=epoch,policy=rejected_state,stats=stats,reasons=unsafe))
            atomic_save(root/"rollback_latest.pth",dict(epoch=epoch,policy=policy.state_dict(),
                optimizer=optimizer.state_dict(),history=history,best=best,stale=stale,stop_reason="safety_rule"))
            print("ACTION_SAFETY_ROLLBACK "+json.dumps(unsafe),flush=True)
            break
        summary=dict(epoch=epoch,update=stats,episodes=len(rows),steps=len(data["reward"]),
            success=float(np.mean([r["success"] for r in rows])),spl=float(np.mean([r["spl"] for r in rows])),
            validation=normal,black=black,black_evaluated=black is not None,normal_black_sr_gap=sr_gap,
            categories=dict(Counter(r["category"] for r in rows)),
            rollout_actions=action_summary(rows),safety_status="accepted",timestamp=time.time())
        if candidate>best:
            best=candidate; atomic_save(root/"best_sr.pth",policy.state_dict())
        history.append(summary); atomic_json(root/"history.json",history)
        atomic_save(latest,dict(epoch=epoch,policy=policy.state_dict(),optimizer=optimizer.state_dict(),
                                history=history,best=best,stale=stale))
        active_rollout.unlink(); post_update.unlink()
        print(json.dumps(summary),flush=True)


if __name__ == "__main__": main()
