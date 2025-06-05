import os
import numpy as np
from envs import MultiplexEnv
from visualization.monitor import plot_task_trajectories

# === 配置参数 ===
episodes = 20
eval_schedule_path = "../configs/5/train_schedule.json"
eval_worker_path = "../configs/5/worker_config.json"
config_path = '../configs/5/env_config.json'

# === 构造环境 ===
env = MultiplexEnv(config_path, schedule_load_path=eval_schedule_path, worker_config_load_path=eval_worker_path)

# === 累计统计信息 ===
total_rewards = []
total_costs = []
total_utils = []
total_failures = []
total_done = []

for ep in range(episodes):
    obs = env.reset(with_new_schedule=False)
    done = False
    total_reward = 0

    while not done:
        action_dict = {}
        for layer_id, layer_obs in obs.items():
            if layer_id == "global_context":
                continue
            num_workers = layer_obs["worker_loads"].shape[0]
            num_tasks = layer_obs["task_queue"].shape[0]
            action = np.random.randint(0, 2, size=(num_workers, num_tasks)).tolist()
            action_dict[layer_id] = action

        obs, reward, done, info = env.step(action_dict)
        total_reward += reward[0] if isinstance(reward, tuple) else reward

    kpi = info["kpi"]
    print(f"[Episode {ep}] Reward: {total_reward:.2f} | Done: {kpi['tasks_done']} | Fail: {kpi['total_failures']} | Cost: {kpi['total_cost']:.2f} | Utility: {kpi['total_utility']:.2f}")

    total_rewards.append(total_reward)
    total_done.append(kpi["tasks_done"])
    total_failures.append(kpi["total_failures"])
    total_costs.append(kpi["total_cost"])
    total_utils.append(kpi["total_utility"])

# === 汇总统计信息 ===
print("\n========== Average over all episodes ==========")
print(f"Avg Reward: {np.mean(total_rewards):.2f}")
print(f"Avg Tasks Done: {np.mean(total_done):.2f}")
print(f"Avg Failures: {np.mean(total_failures):.2f}")
print(f"Avg Cost: {np.mean(total_costs):.2f}")
print(f"Avg Utility: {np.mean(total_utils):.2f}")

# # === 可视化最后一个 episode 的任务轨迹 ===
# plot_task_trajectories(env.chain.finished_tasks)