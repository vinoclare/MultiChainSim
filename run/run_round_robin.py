import os
from envs import MultiplexEnv
from visualization.monitor import plot_task_trajectories

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "env_config_simple.json")


def round_robin_baseline(config_path: str, num_episodes: int = 1):
    env = MultiplexEnv(config_path=config_path)
    num_layers = len(env.chain.layers)
    rr_pointer = [0] * num_layers

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
                # if l == 1:  # 你也可以打 l==2 查看 Layer 2
                #     print(f"[RR] Action for Layer {l}:")
                #     for i, row in enumerate(layer_action):
                #         print(f"  Worker {i}: {row}")

            obs, reward, done, info = env.step(action_dict)
            env.render()
            total_reward += reward if isinstance(reward, (int, float)) else reward[0]

        print(f"Episode {ep+1} finished. Total Reward = {total_reward:.3f}")
        # plot_task_trajectories(env.chain.finished_tasks)
    env.close()


if __name__ == "__main__":
    round_robin_baseline(config_path=CONFIG_PATH, num_episodes=1)
