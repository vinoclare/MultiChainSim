import torch
import numpy as np
import os
import json
import time
from envs import IndustrialChain
from envs.env import MultiplexEnv
from torch.utils.tensorboard import SummaryWriter

# ===== Load configurations =====
env_config_path = '../configs/env_config_simple.json'
with open(env_config_path, 'r') as f:
    env_config = json.load(f)

num_layers = env_config["num_layers"]
max_steps = env_config["max_steps"]
num_episodes = 200  # 可以根据需要设置

# ===== Setup environment =====
env = MultiplexEnv(env_config_path)

log_dir = '../logs/random/' + time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)

# ===== Buffers for statistics =====
reward_buffer = {lid: [] for lid in range(num_layers)}
raw_reward_buffer = {lid: [] for lid in range(num_layers)}
assign_bonus_buffer = {lid: [] for lid in range(num_layers)}
wait_penalty_buffer = {lid: [] for lid in range(num_layers)}
cost_buffer = {lid: [] for lid in range(num_layers)}
util_buffer = {lid: [] for lid in range(num_layers)}

waiting_tasks_all = []
done_tasks_all = []
failed_tasks_all = []
total_tasks_all = []

for episode in range(num_episodes):
    obs = env.reset()
    episode_rewards = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_raw_rewards = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_assign_bonus = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_wait_penalty = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_cost = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_util = {layer_id: 0.0 for layer_id in range(num_layers)}

    for step in range(max_steps):
        actions = {}
        for layer_id in range(num_layers):
            layer_obs = obs[layer_id]
            n_worker = layer_obs["worker_loads"].shape[0]
            n_task = layer_obs["task_queue"].shape[0]
            rand_action = np.random.rand(n_worker, n_task).astype(np.float32)
            actions[layer_id] = rand_action

        obs, reward_dict, done, _ = env.step(actions)

        for layer_id in range(num_layers):
            reward_scalar = reward_dict[1]["layer_rewards"][layer_id]["reward"]
            raw_reward_scalar = reward_dict[1]["layer_rewards"][layer_id]["raw_reward"]
            assign_bonus_scalar = reward_dict[1]["layer_rewards"][layer_id]["assign_bonus"]
            wait_penalty_scalar = reward_dict[1]["layer_rewards"][layer_id]["wait_penalty"]
            cost_scalar = reward_dict[1]['layer_rewards'][layer_id]['cost']
            util_scalar = reward_dict[1]['layer_rewards'][layer_id]['utility']

            episode_rewards[layer_id] += reward_scalar
            episode_raw_rewards[layer_id] += raw_reward_scalar
            episode_assign_bonus[layer_id] += assign_bonus_scalar
            episode_wait_penalty[layer_id] += wait_penalty_scalar
            episode_cost[layer_id] += cost_scalar
            episode_util[layer_id] += util_scalar
        if done:
            break

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

    waiting_tasks_all.append(num_waiting_tasks)
    done_tasks_all.append(num_done_tasks)
    failed_tasks_all.append(num_failed_tasks)
    total_tasks_all.append(num_total_tasks)

    for layer_id in range(num_layers):
        reward_buffer[layer_id].append(episode_rewards[layer_id])
        raw_reward_buffer[layer_id].append(episode_raw_rewards[layer_id])
        assign_bonus_buffer[layer_id].append(episode_assign_bonus[layer_id])
        wait_penalty_buffer[layer_id].append(episode_wait_penalty[layer_id])
        cost_buffer[layer_id].append(episode_cost[layer_id])
        util_buffer[layer_id].append(episode_util[layer_id])

    if (episode + 1) % 10 == 0:
        print(f"[Random Episode {episode}] Waiting: {num_waiting_tasks}, Done: {num_done_tasks}, Failed: {num_failed_tasks}")

print("\n==== Random Policy All-Episode Statistics ====")
for lid in range(num_layers):
    print(f"Layer {lid}:")
    print(f"  Avg Normalized Reward: {np.mean(reward_buffer[lid]):.4f}")
    print(f"  Avg Raw Reward:        {np.mean(raw_reward_buffer[lid]):.4f}")
    print(f"  Avg Assign Bonus:      {np.mean(assign_bonus_buffer[lid]):.4f}")
    print(f"  Avg Wait Penalty:      {np.mean(wait_penalty_buffer[lid]):.4f}")
    print(f"  Avg Cost:              {np.mean(cost_buffer[lid]):.4f}")
    print(f"  Avg Utility:           {np.mean(util_buffer[lid]):.4f}")

print(f"\n==== All Episode Task Stats (sum over all layers) ====")
print(f"Avg Waiting Tasks: {np.mean(waiting_tasks_all):.2f}")
print(f"Avg Done Tasks:    {np.mean(done_tasks_all):.2f}")
print(f"Avg Failed Tasks:  {np.mean(failed_tasks_all):.2f}")
print(f"Avg Total Tasks:   {np.mean(total_tasks_all):.2f}")

print(f"\nStd of Waiting Tasks: {np.std(waiting_tasks_all):.2f}")
print(f"Std of Done Tasks:    {np.std(done_tasks_all):.2f}")
print(f"Std of Failed Tasks:  {np.std(failed_tasks_all):.2f}")
print(f"Std of Total Tasks:   {np.std(total_tasks_all):.2f}")

print("\n== Random Baseline Finished ==")
