import os
import time
import json
import torch
import numpy as np
from multiprocessing import Pipe

from envs.rollout_worker import RolloutWorker
from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.ppo_model import PPOIndustrialModel
from algs.ppo import PPO
from agents.agent import IndustrialAgent
from utils import RunningMeanStd
from torch.utils.tensorboard import SummaryWriter


def process_obs(raw_obs, layer_id):
    layer_obs = raw_obs[layer_id]
    task_obs = layer_obs['task_queue']
    worker_loads = layer_obs['worker_loads']
    worker_profile = layer_obs.get('worker_profile')
    global_context = raw_obs.get('global_context')
    return task_obs, worker_loads, worker_profile, global_context


def compute_gae(rewards, dones, values, gamma, lam):
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def evaluate_policy(agents, eval_env, eval_episodes, writer, global_step):
    reward_sums = {lid: [] for lid in agents}
    assign_bouns_sum = {lid: [] for lid in agents}
    wait_penalty_sum = {lid: [] for lid in agents}
    cost_sums = {lid: [] for lid in agents}
    util_sums = {lid: [] for lid in agents}

    for episode in range(eval_episodes):
        obs = eval_env.reset(with_new_schedule=False)
        episode_reward = {lid: 0.0 for lid in agents}
        episode_assign_bonus = {lid: 0.0 for lid in agents}
        episode_wait_penalty = {lid: 0.0 for lid in agents}
        episode_cost = {lid: 0.0 for lid in agents}
        episode_util = {lid: 0.0 for lid in agents}

        done = False
        while not done:
            actions = {}
            for lid in agents:
                task_obs, worker_loads, worker_profile, global_context = process_obs(obs, lid)
                value, action, logprob, _ = agents[lid].sample(task_obs, worker_loads, worker_profile, global_context)
                actions[lid] = action
            obs, (total_reward, reward_detail), done, _ = eval_env.step(actions)

            for lid in agents:
                layer_reward = reward_detail["layer_rewards"][lid]["reward"]
                layer_assign_bonus = reward_detail["layer_rewards"][lid]["assign_bonus"]
                layer_wait_penalty = reward_detail["layer_rewards"][lid]["wait_penalty"]
                layer_cost = reward_detail["layer_rewards"][lid]["cost"]
                layer_util = reward_detail["layer_rewards"][lid]["utility"]

                episode_reward[lid] += layer_reward
                episode_assign_bonus[lid] += layer_assign_bonus
                episode_wait_penalty[lid] += layer_wait_penalty
                episode_cost[lid] += layer_cost
                episode_util[lid] += layer_util

        for lid in agents:
            reward_sums[lid].append(episode_reward[lid])
            assign_bouns_sum[lid].append(episode_assign_bonus[lid])
            wait_penalty_sum[lid].append(episode_wait_penalty[lid])
            cost_sums[lid].append(episode_cost[lid])
            util_sums[lid].append(episode_util[lid])

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

    # === 写入 TensorBoard ===
    total_reward_all = sum([np.mean(reward_sums[lid]) for lid in agents])
    total_cost_all = sum([np.mean(cost_sums[lid]) for lid in agents])
    total_util_all = sum([np.mean(util_sums[lid]) for lid in agents])
    for lid in agents:
        writer.add_scalar(f"eval/layer_{lid}_avg_reward", np.mean(reward_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_assign_bonus", np.mean(assign_bouns_sum[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_wait_penalty", np.mean(wait_penalty_sum[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_cost", np.mean(cost_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_utility", np.mean(util_sums[lid]), global_step)
        print(f"[Eval] Layer {lid}: reward={np.mean(reward_sums[lid]):.2f}, "
              f"cost={np.mean(cost_sums[lid]):.2f}, utility={np.mean(util_sums[lid]):.2f}")

    # === 写入所有层的总 reward、cost、utility 到 TensorBoard ===
    writer.add_scalar("global/eval_avg_reward", total_reward_all, global_step)
    writer.add_scalar("global/eval_avg_cost", total_cost_all, global_step)
    writer.add_scalar("global/eval_avg_utility", total_util_all, global_step)

    print(f"[Eval Total] reward={total_reward_all:.2f}, cost={total_cost_all:.2f}, utility={total_util_all:.2f}")


def main():
    # ===== 路径配置 =====
    env_config_path = "../configs/0/env_config_5.json"
    ppo_config_path = "../configs/ppo_config.json"
    schedule_path = "../configs/0/train_schedule_5.json"
    eval_schedule_path = "../configs/0/eval_schedule_5.json"
    worker_config_path = "../configs/0/worker_config_5.json"

    # ===== 加载配置 =====
    with open(env_config_path, 'r') as f:
        env_config = json.load(f)
    with open(ppo_config_path, 'r') as f:
        ppo_config = json.load(f)

    num_layers = env_config["num_layers"]
    num_pad_tasks = env_config["num_pad_tasks"]
    n_task_types = len(env_config["task_types"])
    max_steps = env_config["max_steps"]
    reset_schedule_interval = env_config["reset_schedule_interval"]

    mode = env_config["mode"]
    if mode == "save":
        eval_env = MultiplexEnv(env_config_path, schedule_save_path=eval_schedule_path,
                                worker_config_save_path=worker_config_path)
    else:  # mode == "load"
        eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path,
                                worker_config_load_path=worker_config_path)

    # ===== PPO 超参数 =====
    update_epochs = ppo_config["update_epochs"]
    gamma = ppo_config["gamma"]
    lam = ppo_config["lam"]
    batch_size = ppo_config["batch_size"]
    hidden_dim = ppo_config["hidden_dim"]
    eval_interval = ppo_config["eval_interval"]
    eval_episodes = ppo_config["eval_episodes"]
    log_interval = ppo_config["log_interval"]
    num_episodes = ppo_config["num_episodes"]
    num_workers = ppo_config.get("num_workers", 4)  # 默认 4 个 rollout 进程

    # ===== TensorBoard & 全局状态 =====
    log_dir = '../logs/ppo_dist/' + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    global_step = [0]
    return_rms = {lid: RunningMeanStd() for lid in range(num_layers)}

    # ===== 构建 PPO Agent（按层）=====
    agents = {}
    for lid in range(num_layers):
        n_worker = env_config["workers_per_layer"][lid]
        task_input_dim = 4 + n_task_types
        worker_input_dim = 1 + n_task_types
        profile_dim = 2 * n_task_types
        global_context_dim = 1

        model = PPOIndustrialModel(
            task_input_dim=task_input_dim,
            worker_load_input_dim=worker_input_dim,
            worker_profile_input_dim=profile_dim,
            n_worker=n_worker,
            num_pad_tasks=num_pad_tasks,
            global_context_dim=global_context_dim,
            hidden_dim=hidden_dim
        )
        alg = PPO(
            model,
            clip_param=ppo_config["clip_param"],
            value_loss_coef=ppo_config["value_loss_coef"],
            entropy_coef=ppo_config["entropy_coef"],
            initial_lr=ppo_config["initial_lr"],
            max_grad_norm=ppo_config["max_grad_norm"],
            writer=writer,
            global_step_ref=global_step * num_workers,
            total_training_steps=num_episodes * max_steps
        )
        agents[lid] = IndustrialAgent(alg, "ppo", num_pad_tasks)

    # ===== 启动 rollout_worker 子进程 =====
    workers = []
    conns = []
    for wid in range(num_workers):
        parent_conn, child_conn = Pipe()
        worker = RolloutWorker(child_conn, env_config_path, schedule_path, worker_config_path)
        worker.start()
        workers.append(worker)
        conns.append(parent_conn)

    print(f"Started {num_workers} rollout workers.")

    # ===== 初始化经验缓存 =====
    buffers = {lid: {k: [] for k in [
        'task_obs', 'worker_loads', 'worker_profile', 'global_context',
        'valid_mask', 'actions', 'logprobs', 'values', 'rewards', 'dones']}
        for lid in range(num_layers)}

    # ===== 日志缓存 =====
    # reward_buffer = {lid: [] for lid in range(num_layers)}
    # assign_bonus_buffer = {lid: [] for lid in range(num_layers)}
    # wait_penalty_buffer = {lid: [] for lid in range(num_layers)}
    # cost_buffer = {lid: [] for lid in range(num_layers)}
    # util_buffer = {lid: [] for lid in range(num_layers)}

    episode_counter = 0
    step_in_episode = 0

    # ===== 首次 reset 所有 worker，获取初始 obs =====
    step_data_per_worker = []
    for conn in conns:
        conn.send({"cmd": "reset"})
    for conn in conns:
        step_data_per_worker.append(conn.recv())

    # ===== 主训练循环 =====
    while episode_counter < num_episodes:

        actions_per_worker = []

        # === 1. 根据 obs 推理动作 ===
        for step_data in step_data_per_worker:
            obs_dict = step_data["obs"]
            actions = {}

            for lid in range(num_layers):
                obs = obs_dict[lid]
                task_obs = np.array(obs["task_queue"], dtype=np.float32)
                worker_loads = np.array(obs["worker_loads"], dtype=np.float32)
                profile = np.array(obs["worker_profile"], dtype=np.float32)
                gctx = np.array(obs_dict["global_context"], dtype=np.float32)

                value, action, logp, _ = agents[lid].sample(task_obs, worker_loads, profile, gctx)
                valid_mask = task_obs[:, 3].astype(np.float32)

                buffers[lid]['task_obs'].append(task_obs)
                buffers[lid]['worker_loads'].append(worker_loads)
                buffers[lid]['worker_profile'].append(profile)
                buffers[lid]['global_context'].append(gctx)
                buffers[lid]['valid_mask'].append(valid_mask)
                buffers[lid]['actions'].append(action)
                buffers[lid]['logprobs'].append(logp)
                buffers[lid]['values'].append(value)
                buffers[lid]['rewards'].append(step_data["reward"][lid]["reward"])
                buffers[lid]['dones'].append(step_data["done"])
            actions_per_worker.append(actions)

        # === 2. 向所有 worker 发 step 动作 ===
        for conn, action in zip(conns, actions_per_worker):
            conn.send({"cmd": "step", "action": action})

        # === 3. 接收所有 worker 的下一步返回 ===
        step_data_per_worker = [conn.recv() for conn in conns]
        step_in_episode += 1

        # === 4. 是否需要结束 episode？（统一控制）===
        if step_data_per_worker[0]["done"] or step_in_episode >= max_steps:
            episode_counter += 1
            step_in_episode = 0
            print(f"[Episode {episode_counter}] rollout collection complete.")

            for lid in range(num_layers):
                rewards = buffers[lid]['rewards']
                dones = buffers[lid]['dones']
                values = buffers[lid]['values'] + [0]

                advs, rets = [], []
                gae = 0
                for t in reversed(range(len(rewards))):
                    delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
                    gae = delta + gamma * lam * (1 - dones[t]) * gae
                    advs.insert(0, gae)
                    rets.insert(0, gae + values[t])

                if ppo_config.get("return_normalization", False):
                    return_rms[lid].update(np.array(rets))
                    rets = return_rms[lid].normalize(np.array(rets))

                dataset = list(zip(
                    buffers[lid]['task_obs'],
                    buffers[lid]['worker_loads'],
                    buffers[lid]['worker_profile'],
                    buffers[lid]['global_context'],
                    buffers[lid]['valid_mask'],
                    buffers[lid]['actions'],
                    buffers[lid]['values'],
                    rets,
                    buffers[lid]['logprobs'],
                    advs
                ))

                for _ in range(update_epochs):
                    np.random.shuffle(dataset)
                    for i in range(0, len(dataset), batch_size):
                        minibatch = dataset[i:i + batch_size]
                        task_batch, worker_batch, profile_batch, gctx_batch, mask_batch, \
                            act_batch, val_batch, ret_batch, logp_batch, adv_batch = zip(*minibatch)

                        agents[lid].learn(
                            torch.tensor(np.array(task_batch), dtype=torch.float32),
                            torch.tensor(np.array(worker_batch), dtype=torch.float32),
                            torch.tensor(np.array(profile_batch), dtype=torch.float32),
                            torch.tensor(np.array(gctx_batch), dtype=torch.float32),
                            torch.tensor(np.array(mask_batch), dtype=torch.float32),
                            torch.tensor(np.array(act_batch), dtype=torch.float32),
                            torch.tensor(np.array(val_batch), dtype=torch.float32),
                            torch.tensor(np.array(ret_batch), dtype=torch.float32),
                            torch.tensor(np.array(logp_batch), dtype=torch.float32),
                            torch.tensor(np.array(adv_batch), dtype=torch.float32),
                        )

                buffers[lid] = {k: [] for k in buffers[lid]}

            # === 每 N 个 episode 写一次日志 ===
            # if episode_counter % log_interval == 0:
            #     total_episode_reward = 0.0
            #     total_episode_cost = 0.0
            #     total_episode_util = 0.0
            #
            #     for lid in range(num_layers):
            #         avg_reward = np.mean(reward_buffer[lid])
            #         avg_assign = np.mean(assign_bonus_buffer[lid])
            #         avg_wait = np.mean(wait_penalty_buffer[lid])
            #         avg_cost = np.mean(cost_buffer[lid])
            #         avg_util = np.mean(util_buffer[lid])
            #
            #         writer.add_scalar(f"layer_{lid}/avg_episode_reward", avg_reward, episode_counter)
            #         writer.add_scalar(f"layer_{lid}/avg_episode_assign_bonus", avg_assign, episode_counter)
            #         writer.add_scalar(f"layer_{lid}/avg_episode_wait_penalty", avg_wait, episode_counter)
            #         writer.add_scalar(f"layer_{lid}/avg_episode_cost", avg_cost, episode_counter)
            #         writer.add_scalar(f"layer_{lid}/avg_episode_utility", avg_util, episode_counter)
            #
            #         total_episode_reward += avg_reward
            #         total_episode_cost += avg_cost
            #         total_episode_util += avg_util
            #
            #         reward_buffer[lid].clear()
            #         assign_bonus_buffer[lid].clear()
            #         wait_penalty_buffer[lid].clear()
            #         cost_buffer[lid].clear()
            #         util_buffer[lid].clear()
            #
            #     writer.add_scalar("global/episode_total_reward", total_episode_reward, episode_counter)
            #     writer.add_scalar("global/episode_total_cost", total_episode_cost, episode_counter)
            #     writer.add_scalar("global/episode_total_utility", total_episode_util, episode_counter)

            # === 每 eval_interval 执行评估 ===
            if episode_counter % eval_interval == 0:
                evaluate_policy(agents, eval_env, eval_episodes, writer, episode_counter * max_steps)

    # ===== 全部训练完成后，关闭 worker =====
    for conn in conns:
        conn.send({"cmd": "close"})
    for w in workers:
        w.join()
    writer.close()
    print("Training finished. All workers closed.")


if __name__ == "__main__":
    main()
