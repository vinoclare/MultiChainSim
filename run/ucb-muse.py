import argparse
import json
import os
import time
from collections import deque

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import concurrent.futures as cf

from algs.muse2 import MuSE
from algs.distiller import Distiller
from utils.agent57_scheduler import SoftUCB
from utils.buffer import RolloutBuffer, compute_gae
from utils.utils import RunningMeanStd  # 归一化工具
from agents.agent57_agent import Agent57Agent
from envs import MultiplexEnv


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def run_agent57_multi_layer(env: MultiplexEnv,
                            eval_env: MultiplexEnv,
                            env_config: dict,
                            agent57_config: dict,
                            log_dir: str):
    """
    多层版 Agent57 训练脚本。
    - env: 训练用 MultiplexEnv 实例
    - eval_env: 评估用 MultiplexEnv 实例
    - env_config: 从 env_config.json 读取的字典
    - agent57_config: 从 agent57_config.json 读取的字典
    """

    device = torch.device(agent57_config["device"] if torch.cuda.is_available() else "cpu")

    # 1. 预先读取环境维度信息
    n_layers = env_config["num_layers"]
    n_worker = env_config['workers_per_layer']
    num_pad_tasks = env_config["num_pad_tasks"]
    max_steps = env_config["max_steps"]
    num_layers = len(env_config["workers_per_layer"])

    # 任务维度、负载维度、属性维度（与单层一致）
    n_task_types = len(env_config["task_types"])
    task_dim = 4 + n_task_types
    worker_load_dim = 1 + n_task_types
    worker_profile_dim = 2 * n_task_types

    # 2. 读取多子策略配置
    multi_cfg = agent57_config["multi_strategy"]
    K = multi_cfg["K"]
    betas = multi_cfg["betas"]
    gammas = multi_cfg["gammas"]

    # HiTAC-Muse 超参数迁移
    distill_cfg = agent57_config["distill"]
    device = agent57_config["training"]["device"] if torch.cuda.is_available() else "cpu"
    num_layers = env_config["num_layers"]
    num_workers = env_config["workers_per_layer"]
    num_pad_tasks = env_config["num_pad_tasks"]
    task_types = env_config["task_types"]
    n_task_types = len(task_types)
    profile_dim = 2 * n_task_types
    global_context_dim = 1
    policies_info_dim = agent57_config["hitac"]["policies_info_dim"]

    num_episodes = agent57_config["training"]["num_episodes"]
    eval_interval = agent57_config["training"]["eval_interval"]
    log_interval = agent57_config["training"]["log_interval"]
    eval_episodes = agent57_config["training"]["eval_episodes"]
    distill_interval = agent57_config["scheduler"]["distill_interval"]
    switch_interval = agent57_config["scheduler"]["switch_interval"]
    hitac_update_interval = agent57_config["scheduler"]["hitac_update_interval"]
    reset_schedule_interval = agent57_config["training"]["reset_schedule_interval"]
    neg_interval = agent57_config["scheduler"]["neg_interval"]

    steps_per_episode = env_config["max_steps"]
    K = agent57_config["muse"]["K"]
    neg_policy = agent57_config["distill"]["neg_policy"]
    num_pos_subpolicies = K - 2 if agent57_config["distill"]["neg_policy"] else K
    warmup_ep = agent57_config["distill"]["warmup_ep"]
    min_reward_ratio = agent57_config["distill"]["min_reward_ratio"]
    update_epochs = agent57_config["muse"]["update_epochs"]
    gamma = agent57_config["muse"]["gamma"]
    lam = agent57_config["muse"]["lam"]
    batch_size = agent57_config["muse"]["batch_size"]
    return_norm = agent57_config["muse"]["return_normalization"]
    local_kpi_dim = agent57_config["hitac"]["local_kpi_dim"] + num_pos_subpolicies
    global_kpi_dim = agent57_config["hitac"]["global_kpi_dim"]
    policies_info_dim = agent57_config["hitac"]["policies_info_dim"]
    traj_save_threshold = (num_episodes - 10 * eval_interval) * steps_per_episode

    K = 3
    skip_hitac_train = False

    # 3. 为每一层初始化：Agent、Scheduler、RMS、Buffer
    agents = []
    schedulers = []
    cost_rms_list = []
    util_rms_list = []
    buffers = []
    select_counts = [
        {
            "train": [0 for _ in range(K)],
            "test": [0 for _ in range(K)]
        }
        for _ in range(n_layers)
    ]

    for lid in range(n_layers):
        # a) 初始化 Muse
        agent = MuSE(
            cfg=agent57_config["muse"],
            distill_cfg=agent57_config["distill"],
            obs_shapes={
                "task": task_dim,
                "worker_load": worker_load_dim,
                "worker_profile": worker_profile_dim,
                "n_worker": n_worker[lid],
                "num_pad_tasks": num_pad_tasks,
                "global_context_dim": 1
            },
            device=device,
            total_training_steps=agent57_config["num_episodes"] * max_steps
        )
        agents.append(agent)

        # b) 初始化 SoftUCB 调度器
        sched_cfg = agent57_config["scheduler"]
        scheduler = SoftUCB(K,
                            c=sched_cfg["c"],
                            min_switch_interval=sched_cfg["min_switch_interval"])
        schedulers.append(scheduler)

        # c) 初始化 RunningMeanStd 用于 cost/utility
        # cost_rms_list.append(RunningMeanStd(shape=()))
        # util_rms_list.append(RunningMeanStd(shape=()))

        # d) 初始化空 Buffer
        buffers.append(RolloutBuffer())

    # === 每层 obs 结构描述（供 MuSE init）===
    obs_shapes = []
    for lid in range(num_layers):
        obs_shapes.append({
            "task": 4 + n_task_types,
            "worker_load": 1 + n_task_types,
            "worker_profile": 2 * n_task_types,
            "n_worker": num_workers[lid],
            "num_pad_tasks": num_pad_tasks,
            "global_context_dim": global_context_dim
        })

    act_spaces = [
        (obs_shapes[lid]["n_worker"], obs_shapes[lid]["num_pad_tasks"])
        for lid in range(num_layers)
    ]

    # 初始化 Distiller
    distillers = [
        Distiller(
            obs_spaces=obs_shapes[lid],
            global_context_dim=global_context_dim,
            hidden_dim=distill_cfg["hidden_dim"],
            act_dim=act_spaces[lid],
            K=num_pos_subpolicies,
            loss_type=distill_cfg["loss_type"],
            neg_policy=distill_cfg["neg_policy"],
            device=device,
            sup_coef=distill_cfg["sup_coef"],
            neg_coef=distill_cfg["neg_coef"],
            margin=distill_cfg["margin"],
            std_t=distill_cfg["std_t"]
        )
        for lid in range(num_layers)
    ]

    # 4. TensorBoard Writer
    writer = SummaryWriter(log_dir)
    return_u_rms = {lid: RunningMeanStd() for lid in range(num_layers)}
    return_c_rms = {lid: RunningMeanStd() for lid in range(num_layers)}

    # 5. 主循环：按 Episode 轮训
    for episode in range(agent57_config["num_episodes"]):
        # 每个 episode 开始时，重置环境并清空所有层的 buffer
        raw_obs = env.reset(with_new_schedule=False)
        for buf in buffers:
            buf.clear()
        # 记录每层这轮的累积回报
        episode_returns = [0.0 for _ in range(n_layers)]
        episode_rewards = [0.0 for _ in range(n_layers)]
        # 为每一层在本 Episode 一开始选定一个子策略（不在每个 step 中切换）
        pids = []
        for layer_id in range(n_layers):
            pid = schedulers[layer_id].choose()
            pids.append(pid)
            select_counts[layer_id]["train"][pid] += 1

        if episode % 100 == 0:
            print(f"Step: {episode * max_steps}")
            print(f"Current Pids: {pids}")
            print("Select times:")
            for layer_id in range(n_layers):
                print(f"  Layer {layer_id}: {select_counts[layer_id]['train']}")

        done = False
        step_count = 0

        # 逐步采样直到所有层都 done 或达到最大步数
        while not done and step_count < max_steps:
            step_count += 1
            actions = {}  # 存放每层的动作
            step_data_per_layer = {}  # 缓存各层本 step 的中间数据

            # 5.1 对每一层先推理动作并把临时数据存到对应 buffer
            for layer_id in range(n_layers):
                pid = pids[layer_id]
                beta = betas[pid]

                # a) 从原始 obs 中取出该层的观测
                layer_obs = raw_obs[layer_id]
                task_obs = np.array(layer_obs["task_queue"], dtype=np.float32)  # (T, task_dim)
                worker_loads = np.array(layer_obs["worker_loads"], dtype=np.float32)  # (W, worker_load_dim)
                worker_profiles = np.array(layer_obs["worker_profile"], dtype=np.float32)  # (W, worker_profile_dim)
                gctx = np.array(raw_obs["global_context"], dtype=np.float32)  # (global_context_dim,)
                valid_mask = task_obs[:, -1].astype(np.float32)  # (T,)

                # b) 转为张量并调用 Agent.sample
                task_obs_t = torch.tensor(task_obs, dtype=torch.float32).unsqueeze(0).to(device)
                worker_loads_t = torch.tensor(worker_loads, dtype=torch.float32).unsqueeze(0).to(device)
                worker_profiles_t = torch.tensor(worker_profiles, dtype=torch.float32).unsqueeze(0).to(device)
                gctx_t = torch.tensor(gctx, dtype=torch.float32).unsqueeze(0).to(device)
                valid_mask_t = torch.tensor(valid_mask, dtype=torch.float32).unsqueeze(0).to(device)

                v_u, v_c, action_t, logp_t, _, _ = agents[layer_id].sample(
                    task_obs_t, worker_loads_t, worker_profiles_t, gctx_t, valid_mask_t, pid
                )
                action = action_t.squeeze(0).cpu().numpy()  # (W, T)
                v_u_val = v_u.item()
                v_c_val = v_c.item()
                logp_val = logp_t.item()

                # 保存到临时字典，后续再存入 buffer
                step_data_per_layer[layer_id] = {
                    "task_obs": task_obs,
                    "worker_loads": worker_loads,
                    "worker_profiles": worker_profiles,
                    "global_context": gctx,
                    "valid_mask": valid_mask,
                    "action": action,
                    "v_u": v_u_val,
                    "v_c": v_c_val,
                    "logp": logp_val,
                    "beta": beta
                }
                actions[layer_id] = action

            # 5.2 把所有层的动作一次性传给 env.step
            raw_obs, (total_reward, reward_detail), done, _ = env.step(actions)

            # 5.3 收集每层的 reward、归一化后存入对应 buffer
            for layer_id in range(n_layers):
                data = step_data_per_layer[layer_id]
                beta = data["beta"]

                # 从 reward_detail 中提取该层原始效用和成本
                layer_u = reward_detail["layer_rewards"][layer_id]["utility"]
                layer_c = reward_detail["layer_rewards"][layer_id]["cost"]
                layer_assign = reward_detail["layer_rewards"][layer_id]["assign_bonus"]
                layer_wait = reward_detail["layer_rewards"][layer_id]["wait_penalty"]
                layer_reward = reward_detail["layer_rewards"][layer_id]["reward"]

                # 归一化
                # eps = 1e-8
                # util_rms_list[layer_id].update(np.array([layer_u], dtype=np.float32))
                # u_n = (layer_u - util_rms_list[layer_id].mean) / np.sqrt(util_rms_list[layer_id].var + eps)
                #
                # cost_rms_list[layer_id].update(np.array([layer_c], dtype=np.float32))
                # c_n = (layer_c - cost_rms_list[layer_id].mean) / np.sqrt(cost_rms_list[layer_id].var + eps)

                # 组合即时奖励
                u_total = beta * layer_u + layer_assign
                c_total = (1 - beta) * layer_c + layer_wait
                r_comb = u_total - c_total
                episode_returns[layer_id] += r_comb
                episode_rewards[layer_id] += layer_reward

                # 把本 step 的所有信息存到该层的 buffer
                buffers[layer_id].store(
                    data["task_obs"],
                    data["worker_loads"],
                    data["worker_profiles"],
                    data["global_context"],
                    data["valid_mask"],
                    data["action"],
                    data["v_u"],
                    data["v_c"],
                    data["logp"],
                    u_total,
                    c_total,
                    r_comb,
                    done
                )

        # 5.4 本 Episode 结束：依次对每层进行更新
        current_steps = episode * max_steps
        for layer_id in range(n_layers):
            schedulers[layer_id].increment_episode_count()
            pid = pids[layer_id]
            gamma = gammas[pid]

            # 5.4.1 更新该层的调度器：使用本层累积回报
            schedulers[layer_id].update(pid, episode_returns[layer_id])
            schedulers[layer_id].update_real(pid, episode_rewards[layer_id])

            # 5.4.2 计算该层 GAE，并调用 Agent.learn
            batch = buffers[layer_id].to_tensors(device=device)
            # GAE: 效用和成本分别计算
            ret_u, _ = compute_gae(
                batch["rewards_u"].cpu().numpy(),
                batch["dones"].cpu().numpy(),
                batch["values_u"].cpu().numpy(),
                gamma, agent57_config["lam"]
            )
            ret_c, _ = compute_gae(
                (-batch["rewards_c"].cpu().numpy()),
                batch["dones"].cpu().numpy(),
                (-batch["values_c"].cpu().numpy()),
                gamma, agent57_config["lam"]
            )

            # Normalization
            ret_u_np = np.array(ret_u)
            ret_c_np = np.array(ret_c)
            return_u_rms[layer_id].update(ret_u_np)
            return_c_rms[layer_id].update(ret_c_np)
            ret_u = return_u_rms[layer_id].normalize(ret_u_np)
            ret_c = return_c_rms[layer_id].normalize(ret_c_np)

            returns_u_t = torch.tensor(ret_u, dtype=torch.float32).to(device)
            returns_c_t = torch.tensor(ret_c, dtype=torch.float32).to(device)

            agents[layer_id].learn(
                pid=torch.tensor([pid], device=device),
                task_obs=batch["task_obs"],
                worker_loads=batch["worker_loads"],
                worker_profile=batch["worker_profiles"],
                global_context=batch["global_context"],
                valid_mask=batch["valid_mask"],
                actions=batch["actions"],
                values_u_old=batch["values_u"],
                values_c_old=batch["values_c"],
                returns_u=returns_u_t,
                returns_c=returns_c_t,
                log_probs_old=batch["logps"],
                step=current_steps
            )

        # === 更新 EMA baseline ===
        reward_sum = sum(episode_rewards)
        if episode == 0:
            ema_return = reward_sum
        else:
            ema_return = 0.1 * reward_sum + 0.9 * ema_return

        # === 存入蒸馏 buffer ===
        if reward_sum > (ema_return * min_reward_ratio):
            for lid in range(num_layers):
                batch = buffers[lid].to_tensors(device=device)
                distillers[lid].collect({"task_obs": batch["task_obs"],
                                         "worker_loads": batch["worker_loads"],
                                         "worker_profiles": batch["worker_profiles"],
                                         "global_context": batch["global_context"],
                                         "valid_mask": batch["valid_mask"]},
                                        batch["actions"],
                                        [pids[lid] for _ in range(batch["actions"].shape[0])])

        # 5.5 评估逻辑（每 eval_interval 个 Episode 执行一次）
        if episode % agent57_config["eval_interval"] == 0:
            # 5.5.2 准备统计容器
            eval_reward_sums = {lid: [] for lid in range(n_layers)}
            eval_cost_sums = {lid: [] for lid in range(n_layers)}
            eval_util_sums = {lid: [] for lid in range(n_layers)}
            eval_assign_sums = {lid: [] for lid in range(n_layers)}
            eval_wait_sums = {lid: [] for lid in range(n_layers)}

            # 5.5.3 针对每个 eval_run，运行一次多层并行评估
            for eval_run in range(agent57_config["eval_episodes"]):
                # 重置环境（保持 schedule 不变）
                obs = eval_env.reset(with_new_schedule=False)

                # 各层本回合的累积指标
                episode_u = {lid: 0.0 for lid in range(n_layers)}
                episode_c = {lid: 0.0 for lid in range(n_layers)}
                episode_r = {lid: 0.0 for lid in range(n_layers)}
                episode_a = {lid: 0.0 for lid in range(n_layers)}
                episode_w = {lid: 0.0 for lid in range(n_layers)}

                done = False
                while not done:
                    actions = {}
                    # 5.5.3.1 对所有层并行采集一次动作（主策略生成）
                    for layer_id in range(n_layers):
                        distiller = distillers[layer_id]
                        single = obs[layer_id]
                        task_obs = np.array(single["task_queue"], dtype=np.float32)
                        worker_loads = np.array(single["worker_loads"], dtype=np.float32)
                        worker_profiles = np.array(single["worker_profile"], dtype=np.float32)
                        gctx = np.array(obs["global_context"], dtype=np.float32)
                        valid_mask = task_obs[:, -1].astype(np.float32)

                        task_obs_t = torch.tensor(task_obs, dtype=torch.float32).unsqueeze(0).to(device)
                        worker_loads_t = torch.tensor(worker_loads, dtype=torch.float32).unsqueeze(0).to(device)
                        worker_profiles_t = torch.tensor(worker_profiles, dtype=torch.float32).unsqueeze(0).to(device)
                        gctx_t = torch.tensor(gctx, dtype=torch.float32).unsqueeze(0).to(device)
                        valid_mask_t = torch.tensor(valid_mask, dtype=torch.float32).unsqueeze(0).to(device)

                        # 构造主策略输入字典
                        obs_dict = {
                            "task_obs": task_obs_t,
                            "worker_loads": worker_loads_t,
                            "worker_profiles": worker_profiles_t,
                            "global_context": gctx_t,
                            "valid_mask": valid_mask_t
                        }

                        # 使用 Distiller 主策略预测动作
                        action_t = distiller.predict(obs_dict)
                        actions[layer_id] = action_t.squeeze(0).cpu().numpy()

                    # 5.5.3.2 将所有层的动作一次性喂给 eval_env.step
                    obs, (total_reward, reward_detail), done, _ = eval_env.step(actions)

                    # 5.5.3.3 从 reward_detail 中拆分累加各层指标
                    for layer_id in range(n_layers):
                        layer_reward = reward_detail["layer_rewards"][layer_id]["reward"]
                        layer_u = reward_detail["layer_rewards"][layer_id]["utility"]
                        layer_c = reward_detail["layer_rewards"][layer_id]["cost"]
                        layer_assign = reward_detail["layer_rewards"][layer_id]["assign_bonus"]
                        layer_wait = reward_detail["layer_rewards"][layer_id]["wait_penalty"]

                        episode_u[layer_id] += layer_u
                        episode_c[layer_id] += layer_c
                        episode_a[layer_id] += layer_assign
                        episode_w[layer_id] += layer_wait
                        episode_r[layer_id] += layer_reward

                # 5.5.3.4 本次 eval_run 回合结束后，将各层和“所有层总回报”保存到对应列表
                for layer_id in range(n_layers):
                    eval_reward_sums[layer_id].append(episode_r[layer_id])
                    eval_cost_sums[layer_id].append(episode_c[layer_id])
                    eval_util_sums[layer_id].append(episode_u[layer_id])
                    eval_assign_sums[layer_id].append(episode_a[layer_id])
                    eval_wait_sums[layer_id].append(episode_w[layer_id])

            # 5.5.4 计算各层及“所有层总回报”的平均，并写入 TensorBoard
            total_reward_all = sum([np.mean(eval_reward_sums[lid]) for lid in range(n_layers)])
            total_cost_all = sum([np.mean(eval_cost_sums[lid]) for lid in range(n_layers)])
            total_util_all = sum([np.mean(eval_util_sums[lid]) for lid in range(n_layers)])

            writer.add_scalar("global/eval_avg_reward", total_reward_all, current_steps)
            writer.add_scalar("global/eval_avg_cost", total_cost_all, current_steps)
            writer.add_scalar("global/eval_avg_utility", total_util_all, current_steps)

        # === 蒸馏更新 ===
        if episode % distill_interval == 0 and episode > (warmup_ep * K):
            distill_pids = []
            for layer_id in range(n_layers):
                best_pid = 0
                best_mean = -float('inf')

                # 首先检查：如果所有策略的 deque 都还没有数据，就直接选 pid=0
                all_empty = True
                for i in range(K):
                    if len(schedulers[layer_id].recent_real_returns[i]) > 0:
                        all_empty = False
                        break

                if all_empty:
                    # 这一层还没有任何一次“真回报”上传，就让 greedy_pid = 0
                    greedy_pid = 0
                else:
                    # 否则，遍历每条子策略 i，计算它最近 window_size 次真回报的均值
                    for i in range(K):
                        dq = schedulers[layer_id].recent_real_returns[i]
                        if len(dq) == 0:
                            continue  # 这条策略还没数据，跳过
                        mean_i = sum(dq) / len(dq)
                        if mean_i > best_mean:
                            best_mean = mean_i
                            best_pid = i
                    greedy_pid = best_pid
                distill_pids.append(greedy_pid)
                select_counts[layer_id]["test"][greedy_pid] += 1

            print("Select times (Eval):")
            print(f"Current Pids: {distill_pids}")
            for layer_id in range(n_layers):
                print(f"  Layer {layer_id}: {select_counts[layer_id]['test']}")
            for lid in range(num_layers):
                loss = distillers[lid].bc_update(distill_pids[lid], distill_cfg["batch_size"], distill_cfg["bc_steps"])
                if not skip_hitac_train:  # 非负策略蒸馏时，记录蒸馏 loss
                    writer.add_scalar(f"distill/layer_{lid}_loss", loss, episode)
    writer.close()


def run_one(exp_dir, agent57_cfg, log_dir):
    env_cfg_path = os.path.join("../configs", exp_dir, "env_config.json")
    train_sched_path = os.path.join("../configs", exp_dir, "train_schedule.json")
    eval_sched_path = os.path.join("../configs", exp_dir, "eval_schedule.json")
    worker_cfg_path = os.path.join("../configs", exp_dir, "worker_config.json")

    # 确保四个文件都在
    for f in [env_cfg_path, train_sched_path, eval_sched_path, worker_cfg_path]:
        if not os.path.isfile(f):
            print(f"⚠️  跳过 {exp_dir}（缺少 {os.path.basename(f)}）")
            return

    # 创建环境
    env = MultiplexEnv(
        env_cfg_path,
        schedule_load_path=train_sched_path,
        worker_config_load_path=worker_cfg_path,
    )
    eval_env = MultiplexEnv(
        env_cfg_path,
        schedule_load_path=eval_sched_path,
        worker_config_load_path=worker_cfg_path,
    )

    # 加载配置并训练
    env_cfg = load_config(env_cfg_path)
    agent57_cfg = load_config(agent57_cfg)

    run_agent57_multi_layer(env, eval_env, env_cfg, agent57_cfg, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dire", type=str, default="standard", help="环境配置目录路径")
    parser.add_argument("--cfg", type=str, default="../configs/agent57_config.json", help="Agent57配置文件路径")
    args = parser.parse_args()

    log_dir = f"../logs/agent57/{args.dire}/" + time.strftime("%Y%m%d-%H%M%S")

    os.makedirs(log_dir, exist_ok=True)

    print("\n=== Agent57 单次实验开始 ===\n")
    run_one(args.dire, args.cfg, log_dir)
    print("\n🎉 Agent57 单次实验已完成\n")
