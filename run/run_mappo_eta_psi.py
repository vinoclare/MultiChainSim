# run_mappo_eta_psi.py
# -*- coding: utf-8 -*-
"""
基于 ηψ-Learning 的 MAPPO 训练脚本（不改已有文件，新增运行入口）
- 读取 ../configs/{dire}/env_config.json 与 ../configs/ppo_config.json
- 每层一份 EtaPsiModule：在线计算内在奖励 r_int，并做 SR(TD) 更新
- 对 PPO/MAPPO 的 learn 接口零侵入：仅在 reward 处叠加 r_int*int_coef
"""

import os
import json
import time
import math
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# ===== 项目内模块 =====
from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.mappo_model import MAPPOIndustrialModel
from algs.mappo import MAPPO
from agents.mappo_agent import IndustrialAgent
from utils.utils import RunningMeanStd

from explore.eta_psi import EtaPsiModule  # 我们刚新增的探索模块


# -----------------------------
# 工具函数
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_obs(raw_obs: Dict, lid: int):
    """与 run_mappo.py 一致：取出某一层的三块观测。"""
    obs = raw_obs[lid]
    return obs['task_queue'], obs['worker_loads'], obs['worker_profile']


def flatten_state(task_obs: np.ndarray,
                  worker_loads: np.ndarray,
                  worker_profile: np.ndarray) -> np.ndarray:
    """把三块观测展平成一个向量，喂给 EtaPsiModule."""
    return np.concatenate([
        task_obs.reshape(-1).astype(np.float32),
        worker_loads.reshape(-1).astype(np.float32),
        worker_profile.reshape(-1).astype(np.float32)
    ], axis=0)


def build_valid_mask(task_obs: np.ndarray) -> np.ndarray:
    """
    与现有训练一致：你们把 task_obs 第4列作为可用位(valid_mask)=task_obs[:,3]
    """
    return task_obs[:, 3].astype(np.float32)


def evaluate_policy(agent_dict: Dict[int, IndustrialAgent],
                    eval_env: MultiplexEnv,
                    num_episodes: int,
                    writer: SummaryWriter,
                    global_step: int):
    total_reward = 0.0
    total_cost = 0.0
    total_utility = 0.0
    total_wait_penalty = 0.0

    for _ in range(num_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            actions = {}
            for lid in obs:
                task_obs = obs[lid]['task_queue']
                worker_loads = obs[lid]['worker_loads']
                profile = obs[lid]['worker_profile']
                # 评估时直接用 agent.sample 获取均值动作（内部会按你的实现返回 mean+noise 或 mean）
                _, act, _, _ = agent_dict[lid].sample(task_obs, worker_loads, profile)
                actions[lid] = act
            obs, (_, reward_detail), done, _ = eval_env.step(actions)
            for lid, layer_stats in reward_detail['layer_rewards'].items():
                total_reward += layer_stats.get("reward", 0.0)
                total_cost += layer_stats.get("cost", 0.0)
                total_utility += layer_stats.get("utility", 0.0)
                total_wait_penalty += layer_stats.get("waiting_penalty", 0.0)

    writer.add_scalar("eval/reward", total_reward / num_episodes, global_step)
    writer.add_scalar("eval/cost", total_cost / num_episodes, global_step)
    writer.add_scalar("eval/utility", total_utility / num_episodes, global_step)
    writer.add_scalar("eval/waiting_penalty", total_wait_penalty / num_episodes, global_step)


# -----------------------------
# 主程序
# -----------------------------
def main():
    # ===== 解析命令行参数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument("--dire", type=str, default="standard", help="使用的配置目录名，对应 ../configs/{dire}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--int_coef", type=float, default=0.1, help="内在奖励系数 r = r_ext + int_coef * r_int")
    parser.add_argument("--z_dim", type=int, default=64, help="ηψ 嵌入维度")
    parser.add_argument("--sr_lr", type=float, default=3e-4, help="SR 头学习率")
    parser.add_argument("--sr_update_iters", type=int, default=2, help="每次触发 SR 更新的迭代步数")
    parser.add_argument("--device", type=str, default="cuda", help="覆盖 ppo_config.json 的 device（可选）")
    args, _ = parser.parse_known_args()

    set_seed(args.seed)

    # ===== 加载配置 =====
    dire = args.dire
    env_config_path = f'../configs/{dire}/env_config.json'
    schedule_path = f"../configs/{dire}/train_schedule.json"
    eval_schedule_path = f"../configs/{dire}/eval_schedule.json"
    worker_config_path = f"../configs/{dire}/worker_config.json"

    with open(env_config_path, 'r', encoding='utf-8') as f:
        env_config = json.load(f)
    with open('../configs/ppo_config.json', 'r', encoding='utf-8') as f:
        ppo_config = json.load(f)

    # ===== 创建环境（与 run_mappo.py 一致）=====
    mode = "load"
    if mode == "save":
        env = MultiplexEnv(env_config_path, schedule_save_path=schedule_path, worker_config_save_path=worker_config_path)
        eval_env = MultiplexEnv(env_config_path, schedule_save_path=eval_schedule_path)
    else:
        env = MultiplexEnv(env_config_path, schedule_load_path=schedule_path, worker_config_load_path=worker_config_path)
        eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path)

    eval_env.worker_config = env.worker_config
    eval_env.chain = IndustrialChain(eval_env.worker_config)

    # ===== 超参 =====
    num_layers = env_config["num_layers"]
    num_episodes = ppo_config["num_episodes"]
    steps_per_episode = env_config["max_steps"]
    update_epochs = ppo_config["update_epochs"]
    gamma = ppo_config["gamma"]
    lam = ppo_config["lam"]
    batch_size = ppo_config["batch_size"]
    hidden_dim = ppo_config["hidden_dim"]
    device = args.device if args.device is not None else ppo_config["device"]
    log_interval = ppo_config["log_interval"]
    eval_interval = ppo_config["eval_interval"]
    eval_episodes = ppo_config["eval_episodes"]
    reset_schedule_interval = ppo_config["reset_schedule_interval"]

    # ===== 观测/动作维度 =====
    obs_space = env.observation_space[0]        # dict: {'task_queue': (T,ToDim), 'worker_loads': (W,WDim), 'worker_profile': (W,2*Ntypes)}
    act_space = env.action_space[0]             # (n_worker, num_pad_tasks)
    n_worker, n_pad_tasks = act_space.shape
    n_task_types = len(env_config["task_types"])
    profile_dim = 2 * n_task_types

    # ===== 日志 =====
    log_dir = f'../logs/mappo-eta-psi/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    global_step_ref = [0]  # 传给 MAPPO 以计算训练进度

    # ===== 每层：模型、算法、Agent、RMS、缓冲、ηψ模块 =====
    agents: Dict[int, IndustrialAgent] = {}
    algs: Dict[int, MAPPO] = {}
    return_rms: Dict[int, RunningMeanStd] = {}
    buffers: Dict[int, Dict[str, List]] = {}
    eta_psi_mod: Dict[int, EtaPsiModule] = {}
    sr_buffer: Dict[int, Dict[str, List]] = {}
    sr_prev: Dict[int, Dict[str, np.ndarray]] = {}

    # 计算给 EtaPsi 的输入维度（向量观测）
    task_obs_dim = obs_space['task_queue'].shape[1]
    worker_load_dim = obs_space['worker_loads'].shape[1]
    state_vec_dim = (
            task_obs_dim * env.num_pad_tasks  # task_queue.flatten()
            + worker_load_dim * n_worker  # worker_loads.flatten()
            + profile_dim * n_worker  # worker_profile.flatten()
    )
    action_vec_dim = n_worker * n_pad_tasks  # 连续动作 flatten

    for lid in range(num_layers):
        model = MAPPOIndustrialModel(
            task_input_dim=task_obs_dim,
            worker_load_input_dim=worker_load_dim,
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
            device=device,
            writer=writer,
            global_step_ref=global_step_ref,
            total_training_steps=ppo_config["num_episodes"] * env_config["max_steps"]
        )

        agents[lid] = IndustrialAgent(alg, "mappo", device, env.num_pad_tasks)
        algs[lid] = alg
        return_rms[lid] = RunningMeanStd()
        buffers[lid] = {k: [] for k in [
            'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
            'actions', 'logprobs', 'rewards', 'dones', 'values'
        ]}
        eta_psi_mod[lid] = EtaPsiModule(
            state_dim=state_vec_dim,
            action_dim=action_vec_dim,
            z_dim=args.z_dim,
            gamma=gamma,
            device=device,
            lr=args.sr_lr,
            phi_trainable=False
        )
        sr_buffer[lid] = {k: [] for k in ["s", "a", "s_next", "a_next", "mask"]}
        sr_prev[lid] = {}  # 存上一时刻 (s,a)

    # ===== 训练循环 =====
    int_coef = args.int_coef
    for episode in range(num_episodes):
        # 可选：定期重置生产计划
        if episode % reset_schedule_interval == 0:
            obs = env.reset(with_new_schedule=True)
        else:
            obs = env.reset()

        # 回合起始：清 η、清 prev
        for lid in range(num_layers):
            eta_psi_mod[lid].reset_episode()
            sr_prev[lid].clear()

        for step in range(steps_per_episode):
            actions = {}
            r_int_cache = {}

            # ---- 逐层：采样动作 + 计算内在奖励 + 填充 PPO 缓冲 ----
            for lid in range(num_layers):
                task_obs, worker_loads, profile = process_obs(obs, lid)
                value, action, logprob, _ = agents[lid].sample(task_obs, worker_loads, profile)
                actions[lid] = action

                valid_mask = build_valid_mask(task_obs)

                # 存 PPO 缓冲
                buffers[lid]['task_obs'].append(task_obs)
                buffers[lid]['worker_loads'].append(worker_loads)
                buffers[lid]['worker_profile'].append(profile)
                buffers[lid]['valid_mask'].append(valid_mask)
                buffers[lid]['actions'].append(action)
                buffers[lid]['logprobs'].append(logprob)
                buffers[lid]['values'].append(value)

                # ηψ 内在奖励：基于当前 (s_t, a_t)
                s_vec = flatten_state(task_obs, worker_loads, profile)
                a_vec = action.reshape(-1).astype(np.float32)
                r_int = eta_psi_mod[lid].compute_intrinsic(s_vec, a_vec)
                r_int_cache[lid] = r_int

                # SR 训练的 on-policy 配对：用“上一步”推入 (s, a, s_next, a_next)
                if 's' in sr_prev[lid]:
                    sr_buffer[lid]['s'].append(sr_prev[lid]['s'])
                    sr_buffer[lid]['a'].append(sr_prev[lid]['a'])
                    sr_buffer[lid]['s_next'].append(s_vec)
                    sr_buffer[lid]['a_next'].append(a_vec)     # a_{t} 作为 a_{t-1} 的 a_next
                    sr_buffer[lid]['mask'].append(1.0)

                # 更新 prev 为当前步
                sr_prev[lid]['s'] = s_vec
                sr_prev[lid]['a'] = a_vec

            # ---- 环境前进一步 ----
            obs, (_, reward_detail), done, _ = env.step(actions)

            # ---- 写入奖励（叠加内在 r_int）----
            for lid in range(num_layers):
                r_ext = float(reward_detail['layer_rewards'][lid]['reward'])
                buffers[lid]['rewards'].append(r_ext + int_coef * r_int_cache[lid])
                buffers[lid]['dones'].append(done)

            # ---- 若回合结束：把最后一步 prev 补进 SR（mask=0）----
            if done:
                for lid in range(num_layers):
                    if 's' in sr_prev[lid]:
                        # 终止步：仍可使用当前 obs 作为 s_next，但 mask=0 不会用到 ψ(s',a')
                        task_obs_n, worker_loads_n, profile_n = process_obs(obs, lid)
                        s_vec_n = flatten_state(task_obs_n, worker_loads_n, profile_n)
                        sr_buffer[lid]['s'].append(sr_prev[lid]['s'])
                        sr_buffer[lid]['a'].append(sr_prev[lid]['a'])
                        sr_buffer[lid]['s_next'].append(s_vec_n)
                        sr_buffer[lid]['a_next'].append(np.zeros_like(sr_prev[lid]['a'], dtype=np.float32))
                        sr_buffer[lid]['mask'].append(0.0)
                break  # 结束本回合

        # ------- 计算 GAE & Returns；打包数据集；调用 PPO 学习 -------
        for lid in range(num_layers):
            # GAE（与 run_mappo.py 一致）
            advs: List[float] = []
            vals = buffers[lid]['values'] + [0.0]
            gae = 0.0
            T = len(buffers[lid]['rewards'])
            for t in reversed(range(T)):
                delta = buffers[lid]['rewards'][t] + gamma * vals[t + 1] * (1.0 - buffers[lid]['dones'][t]) - vals[t]
                gae = delta + gamma * lam * (1.0 - buffers[lid]['dones'][t]) * gae
                advs.insert(0, gae)
            rets = [a + v for a, v in zip(advs, buffers[lid]['values'])]

            # 构造 dataset（list of tuples）
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

            # PPO 多 epoch、mini-batch
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
                        torch.tensor(np.array(val_batch, dtype=np.float32)).view(-1),
                        torch.tensor(np.array(ret_batch, dtype=np.float32)).view(-1),
                        torch.tensor(np.array(logp_batch, dtype=np.float32)).view(-1),
                        torch.tensor(np.array(adv_batch, dtype=np.float32)).view(-1),
                        episode * steps_per_episode
                    )

        # ------- SR 更新（每回合一次，最小侵入）-------
        for lid in range(num_layers):
            if len(sr_buffer[lid]['s']) > 0:
                batch_dict = {
                    k: torch.tensor(np.array(v, dtype=np.float32))
                    for k, v in sr_buffer[lid].items()
                }
                sr_loss = eta_psi_mod[lid].update_sr(batch_dict, iters=args.sr_update_iters)
                writer.add_scalar(f"eta_psi/sr_loss/layer{lid}", sr_loss, global_step_ref[0])

        # ------- 清理缓冲 -------
        for lid in buffers:
            for k in buffers[lid]:
                buffers[lid][k].clear()
        for lid in sr_buffer:
            for k in sr_buffer[lid]:
                sr_buffer[lid][k].clear()
            sr_prev[lid].clear()

        # ------- 日志与评估 -------
        global_step_ref[0] += steps_per_episode
        if (episode + 1) % log_interval == 0:
            writer.add_scalar("train/episode", episode + 1, global_step_ref[0])
            writer.add_scalar("train/int_coef", int_coef, global_step_ref[0])

        if (episode + 1) % eval_interval == 0:
            evaluate_policy(agents, eval_env, eval_episodes, writer, global_step_ref[0])

    writer.close()


if __name__ == "__main__":
    main()
