import torch
import numpy as np
import argparse
import json
import time
import random
from torch.utils.tensorboard import SummaryWriter

from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.mappo_ngu_model import MAPPONGUModel
from algs.mappo_ngu import MAPPONGU
from agents.mappo_ngu_agent import IndustrialAgent
from utils.utils import RunningMeanStd
from explore.ngu import NGUIntrinsicReward


class RewardForwardFilter:
    def __init__(self, gamma):
        self.gamma = gamma
        self.rew = None

    def update(self, r):
        self.rew = r if self.rew is None else self.rew * self.gamma + r
        return self.rew


# ===== Load configurations =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
args, _ = parser.parse_known_args()
dire = args.dire

with open(f'../configs/{dire}/env_config.json') as f:
    env_config = json.load(f)
with open('../configs/mappo_ngu_config.json') as f:
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
int_gamma = ppo_config["int_gamma"]
lam = ppo_config["lam"]
batch_size = ppo_config["batch_size"]
hidden_dim = ppo_config["hidden_dim"]
device = ppo_config["device"]
log_interval = ppo_config["log_interval"]
eval_interval = ppo_config["eval_interval"]
eval_episodes = ppo_config["eval_episodes"]
reset_schedule_interval = ppo_config["reset_schedule_interval"]

# ===== Init per-layer models and agents =====
obs_space = env.observation_space[0]
act_space = env.action_space[0]
n_worker, _ = act_space.shape
n_task_types = len(env_config["task_types"])
profile_dim = 2 * n_task_types

log_dir = f'../logs/mappo-ngu/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)

agents, algs, return_rms, buffers = {}, {}, {}, {}
int_rms, int_filter = {}, {}
ngu_modules = {}
intrinsic_coef = ppo_config.get("intrinsic_coef", 0.1)

for lid in range(num_layers):
    ngu_modules[lid] = NGUIntrinsicReward(
        task_obs_dim=obs_space['task_queue'].shape[1],
        worker_load_dim=obs_space['worker_loads'].shape[1],
        num_pad_tasks=env.num_pad_tasks,
        n_worker=n_worker,
        device=device
    )

    model = MAPPONGUModel(
        task_input_dim=obs_space['task_queue'].shape[1],
        worker_load_input_dim=obs_space['worker_loads'].shape[1],
        worker_profile_input_dim=profile_dim,
        n_worker=n_worker,
        num_pad_tasks=env.num_pad_tasks,
        hidden_dim=hidden_dim
    )

    alg = MAPPONGU(
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
    return_rms[lid] = {
        "ext": RunningMeanStd(),
        "int": RunningMeanStd()
    }
    int_rms[lid] = RunningMeanStd()
    int_filter[lid] = RewardForwardFilter(int_gamma)
    buffers[lid] = {k: [] for k in [
        'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
        'actions', 'logprobs', 'rewards', 'dones', 'values', 'ir', 'er',
        'values_ext', 'values_int']
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
                _, _, act, _, _ = agent_dict[lid].sample(task_obs, worker_loads, profile)
                actions[lid] = act
            obs, (_, reward_detail), done, _ = eval_env.step(actions)
            for lid, layer_stats in reward_detail['layer_rewards'].items():
                total_reward += layer_stats.get("reward", 0)
                total_cost += layer_stats.get("cost", 0)
                total_utility += layer_stats.get("utility", 0)
                total_wait_penalty += layer_stats.get("waiting_penalty", 0)

    writer.add_scalar("eval/reward", total_reward / num_episodes, global_step)
    writer.add_scalar("eval/cost", total_cost / num_episodes, global_step)
    writer.add_scalar("eval/utility", total_utility / num_episodes, global_step)
    writer.add_scalar("eval/waiting_penalty", total_wait_penalty / num_episodes, global_step)


for episode in range(num_episodes):
    global_step = episode * steps_per_episode
    obs = env.reset(with_new_schedule=True) if episode % reset_schedule_interval == 0 else env.reset()

    for lid in range(num_layers):
        ngu_modules[lid].reset_episode()

    for step in range(steps_per_episode):
        actions = {}
        for lid in range(num_layers):
            task_obs, worker_loads, profile = process_obs(obs, lid)
            val_ext, val_int, action, logprob, _ = agents[lid].sample(task_obs, worker_loads, profile)
            actions[lid] = action
            valid_mask = task_obs[:, 3].astype(np.float32)

            buffers[lid]['task_obs'].append(task_obs)
            buffers[lid]['worker_loads'].append(worker_loads)
            buffers[lid]['worker_profile'].append(profile)
            buffers[lid]['valid_mask'].append(valid_mask)
            buffers[lid]['actions'].append(action)
            buffers[lid]['logprobs'].append(logprob)
            buffers[lid]['values_ext'].append(val_ext)
            buffers[lid]['values_int'].append(val_int)

        obs, (_, reward_detail), done, _ = env.step(actions)

        for lid in range(num_layers):
            r = reward_detail['layer_rewards'][lid]['reward']
            task_obs = buffers[lid]['task_obs'][-1]
            worker_loads = buffers[lid]['worker_loads'][-1]

            raw_ir = ngu_modules[lid].compute_bonus(task_obs, worker_loads)
            ngu_modules[lid].update(task_obs, worker_loads)
            cum_ir = int_filter[lid].update(raw_ir)
            int_rms[lid].update(np.array([cum_ir]))
            norm_ir = raw_ir / np.sqrt(int_rms[lid].var + 1e-8)

            total_r = r + intrinsic_coef * norm_ir

            buffers[lid]['rewards'].append(total_r)
            buffers[lid]['dones'].append(done)
            buffers[lid]['ir'].append(norm_ir)
            buffers[lid]['er'].append(r)

        if done:
            break

    # ===== Logging =====
    if episode % log_interval == 0:
        for lid in range(num_layers):
            irs, ers = buffers[lid]['ir'], buffers[lid]['er']
            avg_ir, avg_er = np.mean(irs), np.mean(ers)
            ratio = avg_ir / (avg_er + 1e-8)
            sparsity = np.mean(np.array(ers) != 0)
            writer.add_scalar(f"train/layer_{lid}/avg_intrinsic_reward", avg_ir, global_step)
            writer.add_scalar(f"train/layer_{lid}/avg_extrinsic_reward", avg_er, global_step)
            writer.add_scalar(f"train/layer_{lid}/ir_er_ratio", ratio, global_step)
            writer.add_scalar(f"train/layer_{lid}/extrinsic_sparsity", sparsity, global_step)

    # ===== GAE + Return + Learn per layer =====
    for lid in range(num_layers):
        dones = buffers[lid]['dones']
        vals_ext = buffers[lid]['values_ext'] + [0]
        vals_int = buffers[lid]['values_int'] + [0]
        rews_ext = buffers[lid]['er']
        rews_int = buffers[lid]['ir']

        advs_ext, advs_int = [], []
        gae_ext, gae_int = 0, 0
        for t in reversed(range(len(rews_ext))):
            delta_e = rews_ext[t] + gamma * vals_ext[t+1] * (1 - dones[t]) - vals_ext[t]
            delta_i = rews_int[t] + int_gamma * vals_int[t+1] * (1 - dones[t]) - vals_int[t]
            gae_ext = delta_e + gamma * lam * (1 - dones[t]) * gae_ext
            gae_int = delta_i + int_gamma * lam * (1 - dones[t]) * gae_int
            advs_ext.insert(0, gae_ext)
            advs_int.insert(0, gae_int)

        rets_ext = [a + v for a, v in zip(advs_ext, buffers[lid]['values_ext'])]
        rets_int = [a + v for a, v in zip(advs_int, buffers[lid]['values_int'])]

        if ppo_config["return_normalization"]:
            return_rms[lid]["ext"].update(np.array(rets_ext))
            return_rms[lid]["int"].update(np.array(rets_int))
            rets_ext = return_rms[lid]["ext"].normalize(np.array(rets_ext))
            rets_int = return_rms[lid]["int"].normalize(np.array(rets_int))

        dataset = list(zip(
            buffers[lid]['task_obs'],
            buffers[lid]['worker_loads'],
            buffers[lid]['worker_profile'],
            buffers[lid]['valid_mask'],
            buffers[lid]['actions'],
            buffers[lid]['values_ext'],
            buffers[lid]['values_int'],
            rets_ext,
            rets_int,
            buffers[lid]['logprobs'],
            advs_ext  # ← NGU 中策略训练仍仅使用 ext advantage
        ))

        for _ in range(update_epochs):
            random.shuffle(dataset)
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                task_batch, load_batch, prof_batch, mask_batch, act_batch, val_ext_batch, val_int_batch, ret_ext_batch, ret_int_batch, logp_batch, adv_batch = zip(*batch)
                algs[lid].learn(
                    torch.tensor(np.array(task_batch, dtype=np.float32)),
                    torch.tensor(np.array(load_batch, dtype=np.float32)),
                    torch.tensor(np.array(prof_batch, dtype=np.float32)),
                    torch.tensor(np.array(mask_batch, dtype=np.float32)),
                    torch.tensor(np.array(act_batch, dtype=np.float32)),
                    torch.tensor(np.array(val_ext_batch, dtype=np.float32)),
                    torch.tensor(np.array(val_int_batch, dtype=np.float32)),
                    torch.tensor(np.array(ret_ext_batch, dtype=np.float32)),
                    torch.tensor(np.array(ret_int_batch, dtype=np.float32)),
                    torch.tensor(np.array(logp_batch, dtype=np.float32)),
                    torch.tensor(np.array(adv_batch, dtype=np.float32)),
                    episode * steps_per_episode
                )

    for lid in buffers:
        for k in buffers[lid]:
            buffers[lid][k].clear()

    if episode % eval_interval == 0:
        evaluate_policy(agents, eval_env, eval_episodes, writer, global_step)
