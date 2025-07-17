import torch
import numpy as np
import argparse
import json
import time
import random
from torch.utils.tensorboard import SummaryWriter

from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.mappo_model import MAPPOIndustrialModel
from algs.mappo import MAPPO
from agents.mappo_agent import IndustrialAgent
from utils.utils import RunningMeanStd

# ===== Load configurations =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
args, _ = parser.parse_known_args()
dire = args.dire

with open(f'../configs/{dire}/env_config.json') as f:
    env_config = json.load(f)
with open('../configs/ppo_config.json') as f:
    ppo_config = json.load(f)

# ===== Setup environment =====
env_config_path = f'../configs/{dire}/env_config.json'
schedule_path = f"../configs/{dire}/train_schedule.json"
eval_schedule_path = f"../configs/{dire}/eval_schedule.json"
worker_config_path = f"../configs/{dire}/worker_config.json"


mode = "load"
if mode == "save":
    env = MultiplexEnv(env_config_path, schedule_save_path=schedule_path, worker_config_save_path=worker_config_path)
    eval_env = MultiplexEnv(env_config_path, schedule_save_path=eval_schedule_path)
else:  # mode == "load"
    env = MultiplexEnv(env_config_path, schedule_load_path=schedule_path, worker_config_load_path=worker_config_path)
    eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path)

eval_env.worker_config = env.worker_config
eval_env.chain = IndustrialChain(eval_env.worker_config)

# ===== Hyperparameters =====
num_layers = env_config["num_layers"]
num_episodes = ppo_config["num_episodes"]
steps_per_episode = env_config["max_steps"]
update_epochs = ppo_config["update_epochs"]
gamma = ppo_config["gamma"]
lam = ppo_config["lam"]
batch_size = ppo_config["batch_size"]
hidden_dim = ppo_config["hidden_dim"]
device = ppo_config["device"]
log_interval = ppo_config["log_interval"]
eval_interval = ppo_config["eval_interval"]
eval_episodes = ppo_config["eval_episodes"]

# ===== Init per-layer models and agents =====
obs_space = env.observation_space[0]
act_space = env.action_space[0]
n_worker, _ = act_space.shape
n_task_types = len(env_config["task_types"])
profile_dim = 2 * n_task_types

log_dir = f'../logs/mappo/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)

agents, algs, return_rms, buffers = {}, {}, {}, {}

for lid in range(num_layers):
    model = MAPPOIndustrialModel(
        task_input_dim=obs_space['task_queue'].shape[1],
        worker_load_input_dim=obs_space['worker_loads'].shape[1],
        worker_profile_input_dim=profile_dim,
        n_worker=n_worker,
        num_pad_tasks=env.num_pad_tasks,
        hidden_dim=hidden_dim
    )

    alg = MAPPO(
        model,
        clip_param=ppo_config["clip_param"],
        value_loss_coef=ppo_config["value_loss_coef"],
        entropy_coef=ppo_config["entropy_coef"],
        initial_lr=ppo_config["initial_lr"],
        max_grad_norm=ppo_config["max_grad_norm"],
        writer=writer,
        global_step_ref=[0],
        total_training_steps=ppo_config["num_episodes"] * env_config["max_steps"]
    )

    agents[lid] = IndustrialAgent(alg, "mappo", device, env.num_pad_tasks)
    algs[lid] = alg
    return_rms[lid] = RunningMeanStd()
    buffers[lid] = {k: [] for k in [
        'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
        'actions', 'logprobs', 'rewards', 'dones', 'values']
    }


def process_obs(raw_obs, lid):
    obs = raw_obs[lid]
    return obs['task_queue'], obs['worker_loads'], obs['worker_profile']


def evaluate_policy(agent_dict, eval_env, num_episodes, writer, global_step):
    total_reward, total_cost, total_utility, total_wait_penalty = 0, 0, 0, 0
    for _ in range(num_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            actions = {}
            for lid in obs:
                task_obs = obs[lid]['task_queue']
                worker_loads = obs[lid]['worker_loads']
                profile = obs[lid]['worker_profile']
                _, act, _, _ = agent_dict[lid].sample(task_obs, worker_loads, profile)
                actions[lid] = act
            obs, (_, reward_detail), done, _ = eval_env.step(actions)
            for lid, layer_stats in reward_detail['layer_rewards'].items():
                total_reward += layer_stats.get("reward", 0)
                total_cost += layer_stats.get("cost", 0)
                total_utility += layer_stats.get("utility", 0)
                total_wait_penalty += layer_stats.get("waiting_penalty", 0)

        # ===== 统计各种任务状态的数量 =====
        num_total_tasks = 0
        num_waiting_tasks = 0
        num_done_tasks = 0
        num_failed_tasks = 0
        for step_task_list in eval_env.task_schedule.values():
            for task in step_task_list:
                num_total_tasks += 1
                status = task.status
                if status == "waiting":
                    num_waiting_tasks += 1
                elif status == "done":
                    num_done_tasks += 1
                elif status == "failed":
                    num_failed_tasks += 1
        print(f"[Eval Episode {episode}] Total tasks: {num_total_tasks}, Waiting tasks: {num_waiting_tasks}, "
              f"Done tasks: {num_done_tasks}, Failed tasks: {num_failed_tasks}")

    writer.add_scalar("eval/reward", total_reward / num_episodes, global_step)
    writer.add_scalar("eval/cost", total_cost / num_episodes, global_step)
    writer.add_scalar("eval/utility", total_utility / num_episodes, global_step)
    writer.add_scalar("eval/waiting_penalty", total_wait_penalty / num_episodes, global_step)


# ===== Training loop =====
for episode in range(num_episodes):
    obs = env.reset()
    for step in range(steps_per_episode):
        actions = {}
        for lid in range(num_layers):
            task_obs, worker_loads, profile = process_obs(obs, lid)
            value, action, logprob, _ = agents[lid].sample(task_obs, worker_loads, profile)
            actions[lid] = action
            valid_mask = task_obs[:, 3].astype(np.float32)

            buffers[lid]['task_obs'].append(task_obs)
            buffers[lid]['worker_loads'].append(worker_loads)
            buffers[lid]['worker_profile'].append(profile)
            buffers[lid]['valid_mask'].append(valid_mask)
            buffers[lid]['actions'].append(action)
            buffers[lid]['logprobs'].append(logprob)
            buffers[lid]['values'].append(value)

        obs, (_, reward_detail), done, _ = env.step(actions)
        for lid in range(num_layers):
            r = reward_detail['layer_rewards'][lid]['reward']
            buffers[lid]['rewards'].append(r)
            buffers[lid]['dones'].append(done)
        if done:
            break

    # ===== Learn each agent independently =====
    for lid in range(num_layers):
        advs, rets = [], []
        vals = buffers[lid]['values'] + [0]
        gae = 0
        for t in reversed(range(len(buffers[lid]['rewards']))):
            delta = buffers[lid]['rewards'][t] + gamma * vals[t + 1] * (1 - buffers[lid]['dones'][t]) - vals[t]
            gae = delta + gamma * lam * (1 - buffers[lid]['dones'][t]) * gae
            advs.insert(0, gae)
        rets = [a + v for a, v in zip(advs, buffers[lid]['values'])]

        if ppo_config["return_normalization"]:
            return_rms[lid].update(np.array(rets))
            rets = return_rms[lid].normalize(np.array(rets))

        dataset = list(zip(
            buffers[lid]['task_obs'],
            buffers[lid]['worker_loads'],
            buffers[lid]['worker_profile'],
            buffers[lid]['valid_mask'],
            buffers[lid]['actions'],
            buffers[lid]['values'],
            rets,
            buffers[lid]['logprobs'],
            advs
        ))

        for _ in range(update_epochs):
            random.shuffle(dataset)
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                task_batch, load_batch, prof_batch, mask_batch, act_batch, val_batch, ret_batch, logp_batch, adv_batch = zip(*batch)
                algs[lid].learn(
                    torch.tensor(task_batch, dtype=torch.float32),
                    torch.tensor(load_batch, dtype=torch.float32),
                    torch.tensor(prof_batch, dtype=torch.float32),
                    torch.tensor(mask_batch, dtype=torch.float32),
                    torch.tensor(act_batch, dtype=torch.float32),
                    torch.tensor(np.array(val_batch, dtype=np.float32).reshape(-1,)),
                    torch.tensor(np.array(ret_batch, dtype=np.float32).reshape(-1,)),
                    torch.tensor(np.array(logp_batch, dtype=np.float32).reshape(-1,)),
                    torch.tensor(np.array(adv_batch, dtype=np.float32).reshape(-1,)),
                    (episode + 1) * steps_per_episode
                )

    for lid in buffers:
        for k in buffers[lid]:
            buffers[lid][k].clear()

    if episode % eval_interval == 0:
        evaluate_policy(agents, eval_env, eval_episodes, writer, episode * steps_per_episode)
