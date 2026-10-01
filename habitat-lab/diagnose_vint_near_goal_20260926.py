"""Isolated image-goal control check; synthetic nearby views, no training.

Simulator geometry sets up the diagnostic and measures progress only.
The policy receives RGB history and one fixed RGB target, never pose/goal coordinates.
"""
import argparse
import json
import math
from collections import deque
from pathlib import Path

import habitat
import numpy as np
import torch
from PIL import Image
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector

from eval_official_vint_instance_imagenav import load_official_model, NORMALIZE, tf


def preprocess(rgb):
    return NORMALIZE(tf.to_tensor(Image.fromarray(rgb).resize((85, 64))))


def control(waypoint):
    # Official navigate.py unnormalization, followed by pd_controller.py.
    point = waypoint.copy()
    point[:2] *= 0.2 / 4
    dx, dy, hx, hy = point
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        v, w = 0., ((math.atan2(hy, hx) + math.pi) % (2*math.pi) - math.pi) / .25
    elif abs(dx) < 1e-8:
        v, w = 0., np.sign(dy)*math.pi/(2*.25)
    else:
        v, w = dx/.25, math.atan(dy/dx)/.25
    return float(np.clip(v, 0, .2)), float(np.clip(w, -.4, .4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--episodes', type=int, default=1,
                   help='Number of independent Habitat episode starts to diagnose.')
    a = p.parse_args()
    torch.set_num_threads(2)
    model = load_official_model(Path('/autodl-fs/data/RVN-Bench-shared/weights/vint.pth'), torch.device('cpu'))
    cfg = habitat.get_config('benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml', overrides=[
        'habitat.dataset.split=minival', 'habitat.environment.max_episode_steps=500',
        '+habitat/task/actions@habitat.task.actions.velocity_control=velocity_control',
        'habitat.task.actions.velocity_control.lin_vel_range=[0.0,0.2]',
        'habitat.task.actions.velocity_control.ang_vel_range=[-22.9183118052,22.9183118052]',
        'habitat.task.actions.velocity_control.time_step=0.25',
        'habitat.task.actions.velocity_control.min_abs_lin_speed=0.0',
        'habitat.task.actions.velocity_control.min_abs_ang_speed=0.0'])
    results = []
    with habitat.Env(cfg) as env:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        for episode_index in range(a.episodes):
            env.reset()
            state = env.sim.get_agent_state()
            start = state.position.copy()
            original_rotation = state.rotation
            # Select one straight, navmesh-traversable 1 m segment, no route tracking.
            for angle in range(0, 360, 15):
                rot = original_rotation * quat_from_angle_axis(math.radians(angle), np.array([0,1,0]))
                goal = start + quat_rotate_vector(rot, np.array([0.,0.,-1.]))
                endpoint = np.asarray(env.sim.pathfinder.try_step_no_sliding(start, goal))
                if np.linalg.norm(endpoint-goal) < .02:
                    goal = endpoint
                    break
            else:
                print(f'SKIP episode {episode_index}: no unobstructed 1 m segment', flush=True)
                continue
            goal_rgb = env.sim.get_observations_at(goal, rot, keep_agent_at_new_pose=False)['rgb']
            if episode_index == 0:
                Image.fromarray(goal_rgb).save(out.with_suffix('.goal.png'))
            for offset in [0, 20, -20]:
                obs = env.sim.get_observations_at(start, rot * quat_from_angle_axis(math.radians(offset), np.array([0,1,0])), keep_agent_at_new_pose=True)
                if episode_index == 0:
                    Image.fromarray(obs['rgb']).save(out.with_suffix(f'.start_{offset}.png'))
                hist = deque([obs['rgb']]*6, maxlen=6)
                trace = []
                for step in range(a.steps):
                    with torch.inference_mode():
                        dist, wp = model(torch.cat([preprocess(x)[None] for x in hist], dim=1), preprocess(goal_rgb)[None])
                    v,w = control(wp[0,2].numpy())
                    before = env.sim.get_agent_state().position.copy()
                    obs = env.step({'action':'velocity_control','action_args':{'linear_velocity':2*v/.2-1,'angular_velocity':w/.4}})
                    after = env.sim.get_agent_state().position.copy()
                    moved = float(np.linalg.norm(after-before))
                    trace.append({'step':step+1,'v':v,'w':w,'movement_m':moved,'goal_distance_m':float(np.linalg.norm(after-goal)), 'pred_distance':float(dist.item()), 'blocked_proxy':bool(v>.01 and moved<1e-4)})
                    hist.append(obs['rgb'])
                results.append({'episode_index': episode_index, 'offset_deg':offset,'initial_distance_m':float(np.linalg.norm(start-goal)), 'final_distance_m':trace[-1]['goal_distance_m'],'minimum_distance_m':min(t['goal_distance_m'] for t in trace),'path_m':sum(t['movement_m'] for t in trace),'blocked_steps':sum(t['blocked_proxy'] for t in trace),'trace':trace})
                print(json.dumps({k:v for k,v in results[-1].items() if k!='trace'}), flush=True)
                out.write_text(json.dumps({'protocol':'nearby RGB view diagnostic, not InstanceImageNav SR', 'history':'4 Hz synchronized, repeated first frame warmup', 'controller':'official PD law; 4 Hz simulation approximation, ROS controller is 9 Hz', 'cases':results}, indent=2))


if __name__ == '__main__':
    main()
