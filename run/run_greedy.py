import os
import numpy as np
from envs import MultiplexEnv
from visualization.monitor import plot_task_trajectories

eval_schedule_path = "../configs/5/train_schedule.json"
eval_worker_path = "../configs/5/worker_config.json"
config_path = '../configs/5/env_config.json'

env = MultiplexEnv(config_path, schedule_load_path=eval_schedule_path, worker_config_load_path=eval_worker_path)


def greedy_baseline(env, num_episodes: int = 5, mode: str = "utility"):
    assert mode in {"utility", "cost"}, "mode must be 'utility' or 'cost'"

    total_rewards, total_costs, total_utils, total_done, total_fails = [], [], [], [], []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action_dict = {}
            for layer_id, layer in enumerate(env.chain.layers):
                task_queue = layer.task_queue
                num_workers = len(layer.workers)
                num_tasks = len(task_queue)
                layer_action = [[0] * num_tasks for _ in range(num_workers)]

                for t_idx, task in enumerate(task_queue):
                    best_worker_idx = -1
                    best_value = -float("inf") if mode == "utility" else float("inf")

                    for w_idx, worker in enumerate(layer.workers):
                        cap_t = worker.capacity_map.get(task.task_type, 0)
                        used_t = worker.current_load_map.get(task.task_type, 0)
                        total_avail = worker.max_total_load - worker.total_current_load
                        type_avail = cap_t - used_t

                        if type_avail <= 0 or total_avail <= 0:
                            continue

                        val = worker.utility_map.get(task.task_type, 0) if mode == "utility" else worker.cost_map.get(task.task_type, float("inf"))
                        if (mode == "utility" and val > best_value) or (mode == "cost" and val < best_value):
                            best_worker_idx = w_idx
                            best_value = val

                    if best_worker_idx != -1:
                        worker = layer.workers[best_worker_idx]
                        cap_t = worker.capacity_map[task.task_type]
                        used_t = worker.current_load_map[task.task_type]
                        total_avail = worker.max_total_load - worker.total_current_load
                        type_avail = cap_t - used_t
                        assign_amt = min(task.unassigned_amount, type_avail, total_avail)
                        if assign_amt > 0:
                            layer_action[best_worker_idx][t_idx] = assign_amt

                action_dict[layer_id] = layer_action

            obs, reward, done, info = env.step(action_dict)
            total_reward += reward if isinstance(reward, (int, float)) else reward[0]

        kpi = info["kpi"]
        print(f"[Episode {ep}] Reward: {total_reward:.2f} | Done: {kpi['tasks_done']} | Fail: {kpi['total_failures']} | Cost: {kpi['total_cost']:.2f} | Utility: {kpi['total_utility']:.2f}")

        total_rewards.append(total_reward)
        total_done.append(kpi["tasks_done"])
        total_fails.append(kpi["total_failures"])
        total_costs.append(kpi["total_cost"])
        total_utils.append(kpi["total_utility"])

    # === 汇总 ===
    print(f"\n==== Greedy-{mode.capitalize()} Summary over {num_episodes} episodes ====")
    print(f"Avg Reward: {np.mean(total_rewards):.2f}")
    print(f"Avg Done: {np.mean(total_done):.2f}")
    print(f"Avg Fail: {np.mean(total_fails):.2f}")
    print(f"Avg Cost: {np.mean(total_costs):.2f}")
    print(f"Avg Utility: {np.mean(total_utils):.2f}")

    # plot_task_trajectories(env.chain.finished_tasks)
    env.close()


if __name__ == "__main__":
    greedy_baseline(env, num_episodes=20, mode="utility")
    greedy_baseline(env, num_episodes=20, mode="cost")
