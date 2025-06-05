import os
import json
import time

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.agent57_scheduler import SoftUCB
from utils.buffer import RolloutBuffer, compute_gae
from utils.utils import RunningMeanStd  # 归一化工具
from agents.multi_strategy_agent import MultiStrategyAgent
from envs import MultiplexEnv


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def run_agent57_multi_layer(env: MultiplexEnv,
                            eval_env: MultiplexEnv,
                            env_config: dict,
                            agent57_config: dict):
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

    # 3. 为每一层初始化：Agent、Scheduler、RMS、Buffer
    agents = []
    schedulers = []
    cost_rms_list = []
    util_rms_list = []
    buffers = []

    for lid in range(n_layers):
        # a) 初始化 Agent
        agent = MultiStrategyAgent(
            task_input_dim=task_dim,
            worker_load_input_dim=worker_load_dim,
            worker_profile_input_dim=worker_profile_dim,
            n_worker=n_worker[lid],
            num_pad_tasks=num_pad_tasks,
            global_context_dim=1,
            hidden_dim=agent57_config["hidden_dim"],
            betas=betas,
            gammas=gammas,
            clip_param=agent57_config["clip_param"],
            value_loss_coef=agent57_config["value_loss_coef"],
            entropy_coef=agent57_config["entropy_coef"],
            initial_lr=agent57_config["initial_lr"],
            max_grad_norm=agent57_config["max_grad_norm"],
            lam=agent57_config["lam"],
            device=device
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

    # 4. TensorBoard Writer
    log_dir = '../logs/agent57/' + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

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

        done = False
        step_count = 0

        # 逐步采样直到所有层都 done 或达到最大步数
        while not done and step_count < max_steps:
            step_count += 1
            actions = {}          # 存放每层的动作
            step_data_per_layer = {}  # 缓存各层本 step 的中间数据

            # 5.1 对每一层先推理动作并把临时数据存到对应 buffer
            for layer_id in range(n_layers):
                pid = pids[layer_id]
                beta = betas[pid]

                # a) 从原始 obs 中取出该层的观测
                layer_obs = raw_obs[layer_id]
                task_obs = np.array(layer_obs["task_queue"], dtype=np.float32)        # (T, task_dim)
                worker_loads = np.array(layer_obs["worker_loads"], dtype=np.float32)  # (W, worker_load_dim)
                worker_profiles = np.array(layer_obs["worker_profile"], dtype=np.float32)  # (W, worker_profile_dim)
                gctx = np.array(raw_obs["global_context"], dtype=np.float32)          # (global_context_dim,)
                valid_mask = task_obs[:, -1].astype(np.float32)                        # (T,)

                # b) 转为张量并调用 Agent.sample
                task_obs_t = torch.tensor(task_obs, dtype=torch.float32).unsqueeze(0).to(device)
                worker_loads_t = torch.tensor(worker_loads, dtype=torch.float32).unsqueeze(0).to(device)
                worker_profiles_t = torch.tensor(worker_profiles, dtype=torch.float32).unsqueeze(0).to(device)
                gctx_t = torch.tensor(gctx, dtype=torch.float32).unsqueeze(0).to(device)
                valid_mask_t = torch.tensor(valid_mask, dtype=torch.float32).unsqueeze(0).to(device)

                v_u, v_c, action_t, logp_t, _ = agents[layer_id].sample(
                    task_obs_t, worker_loads_t, worker_profiles_t, gctx_t, valid_mask_t, pid
                )
                action = action_t.squeeze(0).cpu().numpy()   # (W, T)
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

        # ===== 统计各种任务状态的数量 =====
        num_total_tasks = 0
        num_waiting_tasks = 0
        num_done_tasks = 0
        num_failed_tasks = 0
        for step_task_list in env.task_schedule.values():
            for task in step_task_list:
                num_total_tasks += 1
                status = task.status
                if status == "waiting":
                    num_waiting_tasks += 1
                elif status == "done":
                    num_done_tasks += 1
                elif status == "failed":
                    num_failed_tasks += 1
        print(f"[Episode {episode}] Total tasks: {num_total_tasks}, Waiting tasks: {num_waiting_tasks}, "
              f"Done tasks: {num_done_tasks}, Failed tasks: {num_failed_tasks}")

        # 5.4 本 Episode 结束：依次对每层进行更新
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
            ret_u, adv_u = compute_gae(
                batch["rewards_u"].cpu().numpy(),
                batch["dones"].cpu().numpy(),
                batch["values_u"].cpu().numpy(),
                gamma, agent57_config["lam"]
            )
            ret_c, adv_c = compute_gae(
                (-batch["rewards_c"].cpu().numpy()),
                batch["dones"].cpu().numpy(),
                (-batch["values_c"].cpu().numpy()),
                gamma, agent57_config["lam"]
            )
            returns_u_t = torch.tensor(ret_u, dtype=torch.float32).to(device)
            returns_c_t = torch.tensor(ret_c, dtype=torch.float32).to(device)

            policy_loss, value_loss, entropy = agents[layer_id].learn(
                task_obs_batch=batch["task_obs"],
                worker_loads_batch=batch["worker_loads"],
                worker_profiles_batch=batch["worker_profiles"],
                global_context_batch=batch["global_context"],
                valid_mask_batch=batch["valid_mask"],
                actions_batch=batch["actions"],
                values_u_old=batch["values_u"],
                values_c_old=batch["values_c"],
                returns_u=returns_u_t,
                returns_c=returns_c_t,
                log_probs_old=batch["logps"],
                dones_batch=batch["dones"],
                rewards_u_batch=batch["rewards_u"],
                rewards_c_batch=batch["rewards_c"],
                policy_id=pid
            )

            # 5.4.3 记录每层训练日志
            if episode % agent57_config["log_interval"] == 0:
                writer.add_scalar(f"train/layer{layer_id}_episode_return", episode_returns[layer_id], episode)
                writer.add_scalar(f"train/layer{layer_id}_episode_reward", episode_rewards[layer_id], episode)
                writer.add_scalar(f"train/layer{layer_id}_policy_loss", policy_loss, episode)
                writer.add_scalar(f"train/layer{layer_id}_value_loss", value_loss, episode)
                writer.add_scalar(f"train/layer{layer_id}_entropy", entropy, episode)
                writer.add_scalar(f"train/layer{layer_id}_avg_return_pid_{pid}", episode_returns[layer_id], episode)
                writer.add_scalar(f"train/layer{layer_id}_avg_reward_pid_{pid}", episode_rewards[layer_id], episode)

        # 5.5 评估逻辑（每 eval_interval 个 Episode 执行一次）
        if episode % agent57_config["eval_interval"] == 0:
            # 5.5.1 先为各层选取 Greedy 子策略 PID
            greedy_pids = []
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
                greedy_pids.append(greedy_pid)

            # 5.5.2 准备统计容器
            eval_reward_sums = {lid: [] for lid in range(n_layers)}
            eval_cost_sums = {lid: [] for lid in range(n_layers)}
            eval_util_sums = {lid: [] for lid in range(n_layers)}
            eval_assign_sums = {lid: [] for lid in range(n_layers)}
            eval_wait_sums = {lid: [] for lid in range(n_layers)}
            eval_all_layers_sum = []  # 用于存储每次评估中“所有层总回报”

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
                episode_all_sum = 0.0  # 同时累加所有层的回报

                done = False
                while not done:
                    actions = {}
                    # 5.5.3.1 对所有层并行采集一次动作
                    for layer_id in range(n_layers):
                        pid = greedy_pids[layer_id]
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

                        # mean as action
                        # with torch.no_grad():
                        #     mean, _, _, _ = agents[layer_id].model(
                        #         task_obs_t, worker_loads_t, worker_profiles_t,
                        #         gctx_t, valid_mask_t, pid
                        #     )
                        # actions[layer_id] = mean.squeeze(0).cpu().numpy()

                        # sample action
                        with torch.no_grad():
                            v_u, v_c, action_t, logp_t, _ = agents[layer_id].sample(
                                task_obs_t, worker_loads_t, worker_profiles_t, gctx_t, valid_mask_t, pid
                            )
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
                        # 层级总回报的定义（与你 run_ppo.py 一致）
                        episode_r[layer_id] += layer_reward

                        episode_all_sum += layer_reward  # 同步累加到“所有层总回报”

                # 5.5.3.4 本次 eval_run 回合结束后，将各层和“所有层总回报”保存到对应列表
                for layer_id in range(n_layers):
                    eval_reward_sums[layer_id].append(episode_r[layer_id])
                    eval_cost_sums[layer_id].append(episode_c[layer_id])
                    eval_util_sums[layer_id].append(episode_u[layer_id])
                    eval_assign_sums[layer_id].append(episode_a[layer_id])
                    eval_wait_sums[layer_id].append(episode_w[layer_id])
                eval_all_layers_sum.append(episode_all_sum)

            # 5.5.4 计算各层及“所有层总回报”的平均，并写入 TensorBoard
            for layer_id in range(n_layers):
                avg_r = float(np.mean(eval_reward_sums[layer_id]))
                avg_c = float(np.mean(eval_cost_sums[layer_id]))
                avg_u = float(np.mean(eval_util_sums[layer_id]))
                avg_a = float(np.mean(eval_assign_sums[layer_id]))
                avg_w = float(np.mean(eval_wait_sums[layer_id]))

                writer.add_scalar(f"eval/layer{layer_id}_avg_reward", avg_r, episode)
                writer.add_scalar(f"eval/layer{layer_id}_avg_cost", avg_c, episode)
                writer.add_scalar(f"eval/layer{layer_id}_avg_utility", avg_u, episode)
                writer.add_scalar(f"eval/layer{layer_id}_avg_assign_bonus", avg_a, episode)
                writer.add_scalar(f"eval/layer{layer_id}_avg_wait_penalty", avg_w, episode)

            # 新增：记录“所有层总回报”在多个 eval_runs 中的平均
            avg_all_layers = float(np.mean(eval_all_layers_sum))
            writer.add_scalar("global/eval_avg_reward", avg_all_layers, episode)

    writer.close()


if __name__ == "__main__":
    # 配置文件路径
    env_config_path = '../configs/0/env_config_5.json'
    agent57_config_path = '../configs/agent57_config.json'
    train_schedule_path = "../configs/0/train_schedule_5.json"
    eval_schedule_path = "../configs/0/eval_schedule_5.json"
    worker_config_path = "../configs/0/worker_config_5.json"

    # 创建训练与评估环境
    env = MultiplexEnv(env_config_path,
                       schedule_load_path=train_schedule_path,
                       worker_config_load_path=worker_config_path)
    eval_env = MultiplexEnv(env_config_path,
                            schedule_load_path=eval_schedule_path,
                            worker_config_load_path=worker_config_path)

    env_config = load_config(env_config_path)
    agent57_config = load_config(agent57_config_path)

    run_agent57_multi_layer(env, eval_env, env_config, agent57_config)
