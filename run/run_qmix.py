import numpy as np
import json
import time
import argparse
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from envs.env import MultiplexEnv
from models.qmix_model import QMixModel
from algs.qmix import QMix
from agents.qmix_agent import QMixAgent
from utils.qmix_buffer import QMixReplayBuffer

# ===== Load configs =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
args, _ = parser.parse_known_args()
dire = args.dire

# ===== Environment =====
env_config_path = f'../configs/{dire}/env_config.json'
qmix_config_path = f'../configs/qmix_config.json'
train_path = f"../configs/{dire}/train_schedule.json"
worker_path = f"../configs/{dire}/worker_config.json"

with open(env_config_path) as f:
    env_cfg = json.load(f)
with open(qmix_config_path) as f:
    cfg = json.load(f)

env = MultiplexEnv(env_config_path,
                   schedule_load_path=train_path,
                   worker_config_load_path=worker_path)

eval_env = MultiplexEnv(env_config_path,
                        schedule_load_path=train_path)
eval_env.worker_config = env.worker_config

num_layers = env_cfg["num_layers"]
num_episodes = cfg["num_episodes"]
steps_per_ep = env_cfg["max_steps"]
eval_interval = cfg["eval_interval"]
eval_episodes = cfg["eval_episodes"]
log_interval = cfg["log_interval"]
num_pad_task = env_cfg["num_pad_tasks"]

# ===== Dim info =====
obs_space = env.observation_space[0]
task_dim = obs_space['task_queue'].shape[1]
load_dim = obs_space['worker_loads'].shape[1]
profile_dim = obs_space['worker_profile'].shape[1]
n_agents = env_cfg["workers_per_layer"][0]
state_dim = (
    env.num_pad_tasks * obs_space['task_queue'].shape[1] +
    n_agents * obs_space['worker_loads'].shape[1] +
    n_agents * obs_space['worker_profile'].shape[1]
)

# ===== Per‑layer structures =====
models, targets, algs, agents, buffers = {}, {}, {}, {}, {}
for lid in range(num_layers):
    model = QMixModel(task_dim, load_dim, profile_dim, num_pad_task, state_dim, n_agents,
                      hidden_dim=cfg["hidden_dim"],
                      q_hidden_dim=cfg["q_hidden_dim"],
                      mixing_hidden_dim=cfg["mixing_hidden_dim"])
    target = QMixModel(task_dim, load_dim, profile_dim, num_pad_task, state_dim, n_agents,
                       hidden_dim=cfg["hidden_dim"],
                       q_hidden_dim=cfg["q_hidden_dim"],
                       mixing_hidden_dim=cfg["mixing_hidden_dim"])
    algs[lid] = QMix(model, target, lr=cfg["initial_lr"], gamma=cfg["gamma"],
                     tau=cfg["target_update_tau"], device=cfg["device"])
    agents[lid] = QMixAgent(model, device=cfg["device"])
    buffers[lid] = QMixReplayBuffer(cfg["buffer_size"])

# ===== Logger =====
writer = SummaryWriter(f"../logs/qmix/{dire}/" + time.strftime("%Y%m%d-%H%M%S"))


def extract_layer_obs(layer_obs):
    tq = layer_obs['task_queue']        # (n_worker, task_dim)
    ld = layer_obs['worker_loads']      # (n_worker, load_dim)
    pf = layer_obs['worker_profile']    # (n_worker, profile_dim)

    state = np.concatenate([tq.flatten(), ld.flatten(), pf.flatten()], axis=0).astype(np.float32)
    return tq, ld, pf, state


def evaluate_policy(agents_dict, env_eval, writer, global_step):
    """Run eval_episodes and log reward / cost / utility / waiting_penalty"""
    tot_reward = tot_cost = tot_util = tot_wp = 0.0
    for _ in range(eval_episodes):
        obs = env_eval.reset()
        done = False
        while not done:
            act = {}
            for lid in range(len(agents_dict)):
                tq, ld, pf, st = extract_layer_obs(obs[lid])
                act[lid] = agents_dict[lid].select_action(tq, ld, pf, st)
            obs, (_, r_det), done, _ = env_eval.step(act)
            for lid in r_det['layer_rewards']:
                lr = r_det['layer_rewards'][lid]
                tot_reward += lr.get('reward', 0.0)
                tot_cost += lr.get('cost', 0.0)
                tot_util += lr.get('utility', 0.0)
                tot_wp += lr.get('waiting_penalty', 0.0)

    writer.add_scalar('eval/reward', tot_reward / eval_episodes, global_step)
    writer.add_scalar('eval/cost', tot_cost / eval_episodes, global_step)
    writer.add_scalar('eval/utility', tot_util / eval_episodes, global_step)
    writer.add_scalar('eval/waiting_penalty', tot_wp / eval_episodes, global_step)


# ===== Training loop =====
for ep in range(num_episodes):
    obs = env.reset()
    ep_reward_layer = defaultdict(float)

    for step in range(steps_per_ep):
        act_dict = {}
        layer_cache = {}  # 缓存当前 obs for transition push

        # --- action selection ---
        for lid in range(num_layers):
            tq, ld, pf, st = extract_layer_obs(obs[lid])
            action = agents[lid].select_action(tq, ld, pf, st)
            act_dict[lid] = action
            layer_cache[lid] = (tq, ld, pf, st)

        # --- env step ---
        next_obs, (_, reward_detail), done, _ = env.step(act_dict)

        # --- store transition & train ---
        for lid in range(num_layers):
            tq, ld, pf, st = layer_cache[lid]
            ntq, nld, npf, nst = extract_layer_obs(next_obs[lid])
            reward = reward_detail['layer_rewards'][lid]['reward']
            ep_reward_layer[lid] += reward
            buffers[lid].push({
                'task_obs': tq,
                'load_obs': ld,
                'profile_obs': pf,
                'state': st,
                'reward': reward,
                'next_task_obs': ntq,
                'next_load_obs': nld,
                'next_profile_obs': npf,
                'next_state': nst,
                'done': float(done)
            })

            if len(buffers[lid]) > cfg['batch_size']:
                for _ in range(cfg['train_per_step']):
                    batch = buffers[lid].sample(cfg['batch_size'])
                    algs[lid].train(batch)

        if done:
            break

    # --- logging ---
    if ep % log_interval == 0:
        total = sum(ep_reward_layer.values())
        print(f"[Episode {ep}] Total reward across layers: {total:.2f}")
        writer.add_scalar('train/episode_reward', total, ep)

    # --- evaluate ---
    if ep % eval_interval == 0:
        evaluate_policy(agents, eval_env, writer, ep)
