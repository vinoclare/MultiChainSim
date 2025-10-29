# run/run_mappo_emu.py
import torch
import numpy as np
import argparse
import json
import time
import random
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.mappo_model import MAPPOIndustrialModel
from algs.mappo import MAPPO
from algs.happo import HAPPO
from agents.mappo_agent import IndustrialAgent
from utils.utils import RunningMeanStd
from explore.emu import EMUPlugin  # EMU-CHANGE: 插件导入

# ===== Load configurations =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
parser.add_argument("--alg_name", type=str, default="mappo")
parser.add_argument("--num_workers", type=int, default=10, help="Parallel env workers for sampling")
args, _ = parser.parse_known_args()
dire = args.dire
alg_name = args.alg_name.lower()

with open(f'../configs/{dire}/env_config.json') as f:
    env_config = json.load(f)
with open('../configs/ppo_config.json') as f:
    ppo_config = json.load(f)

# EMU-CHANGE: 从 ppo_config 读取 EMU 配置（无命令行覆盖，最小改动）
emu_cfg = ppo_config.get("emu", {}) if isinstance(ppo_config, dict) else {}
EMU_USE = emu_cfg.get("use", True)
EMU_ETA = emu_cfg.get("eta", 0.1)
EMU_EMB = emu_cfg.get("embed_dim", 64)
EMU_K = emu_cfg.get("knn", 32)
EMU_TOPP = emu_cfg.get("top_p", 0.2)
EMU_STEPS = emu_cfg.get("retrain_steps", 8)
EMU_EVERY = emu_cfg.get("update_every", 1)

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
reset_schedule_interval = ppo_config["reset_schedule_interval"]

# ===== Init per-layer models and agents =====
obs_space = env.observation_space[0]
act_space = env.action_space[0]
n_worker, _ = act_space.shape
n_task_types = len(env_config["task_types"])
profile_dim = 2 * n_task_types


def _snapshot_policy_states(algs_dict):
    """将每层的策略参数打包成 CPU 张量，供并发 worker 使用。"""
    return {lid: algs_dict[lid].model.state_dict() for lid in algs_dict}


def _episode_worker(worker_id, dire, alg_name, policy_states, seed=None, with_new_schedule=False):
    """单个进程采样一整条 episode，返回与主进程 buffers 相同键的样本字典（按层返回）。"""
    import json, random, numpy as np, torch
    from envs import IndustrialChain
    from envs.env import MultiplexEnv
    from models.mappo_model import MAPPOIndustrialModel
    from algs.mappo import MAPPO
    from algs.happo import HAPPO
    from agents.mappo_agent import IndustrialAgent

    # 路径按传入的 dire 复用你现有的配置组织
    env_config_path = f'../configs/{dire}/env_config.json'
    schedule_path = f"../configs/{dire}/train_schedule.json"
    worker_cfg_path = f"../configs/{dire}/worker_config.json"

    with open(env_config_path) as f:
        env_cfg = json.load(f)

    # 为可重复性设置种子（不会影响主进程）
    if seed is None:
        seed = np.random.randint(0, 2 ** 31 - 1)
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)

    # 每个 worker 自己建环境与链对象
    env = MultiplexEnv(env_config_path, schedule_load_path=schedule_path, worker_config_load_path=worker_cfg_path)
    env.worker_config = env.worker_config
    env.chain = IndustrialChain(env.worker_config)

    num_layers = env_cfg["num_layers"]
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    n_worker, _ = act_space.shape
    n_task_types = len(env_cfg["task_types"])
    profile_dim = 2 * n_task_types
    hidden_dim = ppo_config["hidden_dim"]

    agents, algs = {}, {}
    for lid in range(num_layers):
        model = MAPPOIndustrialModel(
            task_input_dim=obs_space['task_queue'].shape[1],
            worker_load_input_dim=obs_space['worker_loads'].shape[1],
            worker_profile_input_dim=profile_dim,
            n_worker=n_worker,
            num_pad_tasks=env.num_pad_tasks,
            hidden_dim=hidden_dim
        )
        if alg_name == "mappo":
            alg = MAPPO(model, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.0,
                        initial_lr=3e-4, max_grad_norm=0.5,
                        writer=None, global_step_ref=[0], total_training_steps=1)
        else:
            alg = HAPPO(model, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.0,
                        initial_lr=3e-4, max_grad_norm=0.5,
                        writer=None, global_step_ref=[0], total_training_steps=1)

        # 加载策略快照
        alg.model.load_state_dict(policy_states[lid])
        # 采样放 CPU，避免与主进程 GPU 争用
        agents[lid] = IndustrialAgent(alg, alg_name, device="cpu", num_pad_tasks=env.num_pad_tasks)
        algs[lid] = alg

    # 和主进程 buffers 一致的结构
    local_buffers = {lid: {k: [] for k in [
        'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
        'actions', 'logprobs', 'rewards', 'dones', 'values']} for lid in range(num_layers)
                     }

    obs = env.reset(with_new_schedule=with_new_schedule)
    done = False
    while not done:
        actions = {}
        for lid in range(num_layers):
            task_obs = obs[lid]['task_queue']
            worker_loads = obs[lid]['worker_loads']
            profile = obs[lid]['worker_profile']
            value, action, logprob, _ = agents[lid].sample(task_obs, worker_loads, profile)
            actions[lid] = action
            valid_mask = task_obs[:, 3].astype(np.float32)

            local_buffers[lid]['task_obs'].append(task_obs)
            local_buffers[lid]['worker_loads'].append(worker_loads)
            local_buffers[lid]['worker_profile'].append(profile)
            local_buffers[lid]['valid_mask'].append(valid_mask)
            local_buffers[lid]['actions'].append(action)
            local_buffers[lid]['logprobs'].append(logprob)
            local_buffers[lid]['values'].append(value)

        obs, (_, reward_detail), done, _ = env.step(actions)
        for lid in range(num_layers):
            r = reward_detail['layer_rewards'][lid]['reward']
            local_buffers[lid]['rewards'].append(r)
            local_buffers[lid]['dones'].append(done)

    return local_buffers


def evaluate_policy(agent_dict, eval_env, num_episodes, writer, global_step):
    total_reward, total_cost, total_utility, total_wait_penalty = 0, 0, 0, 0
    for _ in range(num_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            actions = {}
            for lid, agent in agent_dict.items():
                task_obs = obs[lid]['task_queue']
                worker_loads = obs[lid]['worker_loads']
                profile = obs[lid]['worker_profile']
                _, act, _, _ = agent.sample(task_obs, worker_loads, profile)
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


def process_obs(obs, lid):
    """把 env 的观测打平为模型输入（与你原本一致）。"""
    task_obs = obs[lid]['task_queue']
    worker_loads = obs[lid]['worker_loads']
    profile = obs[lid]['worker_profile']
    return task_obs, worker_loads, profile


def main():
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

        if alg_name == "mappo":
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
        else:
            alg = HAPPO(
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

    # EMU-CHANGE: 插件实例化
    emu = EMUPlugin(embed_dim=EMU_EMB, knn=EMU_K, top_p=EMU_TOPP, retrain_steps=EMU_STEPS) if EMU_USE else None

    # ===== Training loop =====
    for episode in range(num_episodes):
        if args.num_workers == 1:
            # ——原有单环境采样路径（与你现有实现一致）——
            if episode % reset_schedule_interval == 0:
                obs = env.reset(with_new_schedule=True)
            else:
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
        else:
            # ——分布式并发采样：N 个进程各跑一条 episode 然后拼回 buffers——
            policy_states = _snapshot_policy_states(algs)
            with_new_schedule = (episode % reset_schedule_interval == 0)

            # 为每个 worker 分配独立随机种子
            seeds = np.random.randint(0, 2 ** 31 - 1, size=args.num_workers).tolist()

            with mp.Pool(processes=args.num_workers) as pool:
                results = pool.starmap(
                    _episode_worker,
                    [(wid, dire, alg_name, policy_states, seeds[wid], with_new_schedule)
                     for wid in range(args.num_workers)]
                )
            # 把每个 worker 的轨迹拼回主 buffers（逐层、逐键拼接）
            for lid in range(num_layers):
                def _cat_list(key):
                    return sum([res[lid][key] for res in results], [])

                for key in ['task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
                            'actions', 'logprobs', 'values', 'rewards', 'dones']:
                    buffers[lid][key].extend(_cat_list(key))

        # EMU-CHANGE: 在计算 GAE/returns 之前做奖励塑形，并更新记忆
        if EMU_USE and emu is not None and (episode % EMU_EVERY == 0):
            bonus = emu.compute_bonus_from_buffers(buffers)  # shape [T]
            T = len(next(iter(buffers.values()))['rewards'])
            if bonus.shape[0] > T:
                bonus = bonus[:T]
            # 对每一层叠加奖励塑形（CTDE：训练期可见全局）
            for lid in range(num_layers):
                for t in range(T):
                    buffers[lid]['rewards'][t] = float(buffers[lid]['rewards'][t]) + float(EMU_ETA) * float(bonus[t]) / float(num_layers)
            # 用本回合数据更新记忆（轻量训练）
            emu.update_memory_from_buffers(buffers, gamma=gamma)
            try:
                writer.add_scalar("emu/bonus_mean", float(np.mean(bonus)) if bonus.size else 0.0,
                                  episode * steps_per_episode)
            except Exception:
                pass

        # ===== Learn each agent independently =====
        for lid in range(num_layers):
            if alg_name.lower() == "mappo":
                advs, rets = [], []
                vals = buffers[lid]['values'] + [0]
                gae = 0
                for t in reversed(range(len(buffers[lid]['rewards']))):
                    delta = buffers[lid]['rewards'][t] + gamma * vals[t + 1] * (1 - buffers[lid]['dones'][t]) - vals[t]
                    gae = delta + gamma * lam * (1 - buffers[lid]['dones'][t]) * gae
                    advs.insert(0, gae)
                rets = [a + v for a, v in zip(advs, buffers[lid]['values'])]
            else:  # happo
                advs = []
                vals = buffers[lid]['values'] + [0.0]
                gae = 0.0
                for t in reversed(range(len(buffers[lid]['rewards']))):
                    delta = (buffers[lid]['rewards'][t] + gamma * vals[t + 1] * (1 - buffers[lid]['dones'][t]) - vals[
                        t])
                    gae = delta + gamma * lam * (1 - buffers[lid]['dones'][t]) * gae
                    advs.insert(0, gae.copy())
                rets = [a + v for a, v in zip(advs, buffers[lid]['values'])]
                advs = np.array(advs, dtype=np.float32)
                rets = np.array(rets, dtype=np.float32)
                advs = np.repeat(advs[:, None], n_worker, axis=1)
                advs = advs.reshape(-1, n_worker)
                rets = rets.reshape(-1)

            if ppo_config["return_normalization"]:
                return_rms[lid].update(np.array(rets))
                rets = return_rms[lid].normalize(np.array(rets))

            dataset = list(zip(
                buffers[lid]['task_obs'],
                buffers[lid]['worker_loads'],
                buffers[lid]['worker_profile'],
                buffers[lid]['valid_mask'],
                buffers[lid]['actions'],
                buffers[lid]['values'],  # 注意：values_old 需要传入
                rets,  # returns
                buffers[lid]['logprobs'],  # log_probs_old
                advs  # advantages
            ))

            for _ in range(update_epochs):
                random.shuffle(dataset)
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i + batch_size]
                    (task_batch, load_batch, prof_batch, mask_batch,
                     act_batch, val_batch, ret_batch, logp_batch, adv_batch) = zip(*batch)

                    algs[lid].learn(
                        torch.tensor(np.array(task_batch, dtype=np.float32)),
                        torch.tensor(np.array(load_batch, dtype=np.float32)),
                        torch.tensor(np.array(prof_batch, dtype=np.float32)),
                        torch.tensor(np.array(mask_batch, dtype=np.float32)),
                        torch.tensor(np.array(act_batch, dtype=np.float32)),
                        torch.tensor(np.array(val_batch, dtype=np.float32)),  # values_old
                        torch.tensor(np.array(ret_batch, dtype=np.float32)),  # returns
                        torch.tensor(np.array(logp_batch, dtype=np.float32)),  # log_probs_old
                        torch.tensor(np.array(adv_batch, dtype=np.float32)),  # advantages
                        episode * steps_per_episode * (args.num_workers if args.num_workers > 1 else 1)  # current_steps
                    )

            # 清空本层缓存
            for k in buffers[lid]:
                buffers[lid][k].clear()

        if episode % eval_interval == 0:
            evaluate_policy(agents, eval_env, eval_episodes, writer, episode * steps_per_episode)


if __name__ == "__main__":
    # Windows/macOS 下多进程需要 spawn；Linux 也可保持一致
    log_dir = f'../logs2/emu/{alg_name}/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    mp.set_start_method("spawn", force=True)
    main()
