import os
import numpy as np
from envs import MultiplexEnv
from visualization.monitor import plot_task_trajectories

eval_schedule_path = "./configs/5/train_schedule.json"
eval_worker_path = "./configs/5/worker_config.json"
config_path = './configs/env_config.json'

env = MultiplexEnv(config_path, schedule_load_path=eval_schedule_path, worker_config_load_path=eval_worker_path)


def round_robin_baseline(env, num_episodes: int = 20):
    num_layers = len(env.chain.layers)
    rr_pointer = [0] * num_layers

    total_rewards = []
    total_costs = []
    total_utils = []
    total_done = []
    total_fails = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action_dict = {}
            for l, layer in enumerate(env.chain.layers):
                task_queue = layer.task_queue
                num_workers = len(layer.workers)
                num_tasks = len(task_queue)
                layer_action = [[0] * num_tasks for _ in range(num_workers)]

                for t_idx, task in enumerate(task_queue):
                    remain = task.unassigned_amount
                    ptr = rr_pointer[l]
                    tried = 0

                    while remain > 0 and tried < num_workers:
                        w_idx = ptr % num_workers
                        worker = layer.workers[w_idx]

                        cap_t = worker.capacity_map.get(task.task_type, 0)
                        used_t = worker.current_load_map.get(task.task_type, 0)
                        type_avail = cap_t - used_t
                        total_avail = worker.max_total_load - worker.total_current_load

                        assign_amt = min(remain, type_avail, total_avail)

                        if assign_amt > 0:
                            layer_action[w_idx][t_idx] += assign_amt
                            remain -= assign_amt

                        ptr += 1
                        tried += 1

                    rr_pointer[l] = ptr % num_workers

                action_dict[l] = layer_action

            obs, reward, done, info = env.step(action_dict)
            total_reward += reward if isinstance(reward, (int, float)) else reward[0]

        kpi = info["kpi"]
        print(f"[Episode {ep}] Reward: {total_reward:.2f} | Done: {kpi['tasks_done']} | Fail: {kpi['total_failures']} | Cost: {kpi['total_cost']:.2f} | Utility: {kpi['total_utility']:.2f}")

        total_rewards.append(total_reward)
        total_done.append(kpi["tasks_done"])
        total_fails.append(kpi["total_failures"])
        total_costs.append(kpi["total_cost"])
        total_utils.append(kpi["total_utility"])

    # === 汇总统计信息 ===
    print("\n========== Average over all episodes ==========")
    print(f"Avg Reward: {np.mean(total_rewards):.2f}")
    print(f"Avg Tasks Done: {np.mean(total_done):.2f}")
    print(f"Avg Failures: {np.mean(total_fails):.2f}")
    print(f"Avg Cost: {np.mean(total_costs):.2f}")
    print(f"Avg Utility: {np.mean(total_utils):.2f}")

    # # 可视化最后一轮轨迹
    # plot_task_trajectories(env.chain.finished_tasks)
    env.close()


if __name__ == "__main__":
    round_robin_baseline(env=env, num_episodes=20)
