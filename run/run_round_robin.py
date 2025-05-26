import numpy as np
from envs.env import MultiplexEnv

CONFIG_PATH = 'configs/env_config_simple.json'
NUM_EPISODES = 200


def round_robin_baseline(config_path: str, num_episodes: int = 1):
    env = MultiplexEnv(config_path=config_path)
    num_layers = len(env.chain.layers)
    max_steps = env.config["max_steps"]

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

    rr_pointer = [0] * num_layers

    for ep in range(num_episodes):
        obs = env.reset()
        episode_rewards = {layer_id: 0.0 for layer_id in range(num_layers)}
        episode_raw_rewards = {layer_id: 0.0 for layer_id in range(num_layers)}
        episode_assign_bonus = {layer_id: 0.0 for layer_id in range(num_layers)}
        episode_wait_penalty = {layer_id: 0.0 for layer_id in range(num_layers)}
        episode_cost = {layer_id: 0.0 for layer_id in range(num_layers)}
        episode_util = {layer_id: 0.0 for layer_id in range(num_layers)}

        for step in range(max_steps):
            action_dict = {}
            for l, layer in enumerate(env.chain.layers):
                task_queue = layer.task_queue
                num_workers = len(layer.workers)
                num_tasks = len(task_queue)
                layer_action = np.zeros((num_workers, num_tasks), dtype=np.float32)

                for t_idx, task in enumerate(task_queue):
                    remain = task.unassigned_amount
                    if remain < 1e-8:
                        continue
                    ptr = rr_pointer[l]
                    tried = 0
                    orig_remain = remain
                    while remain > 1e-6 and tried < num_workers:
                        w_idx = ptr % num_workers
                        worker = layer.workers[w_idx]

                        cap_t = worker.capacity_map.get(task.task_type, 0)
                        used_t = worker.current_load_map.get(task.task_type, 0)
                        type_avail = cap_t - used_t

                        total_avail = worker.max_total_load - worker.total_current_load

                        assign_amt = min(remain, type_avail, total_avail)

                        # 严格按比例scale
                        if assign_amt > 1e-8 and orig_remain > 1e-8:
                            percent = assign_amt / orig_remain
                            layer_action[w_idx][t_idx] += percent
                            remain -= assign_amt

                        ptr += 1
                        tried += 1

                    rr_pointer[l] = ptr % num_workers

                action_dict[l] = layer_action

            obs, reward_dict, done, _ = env.step(action_dict)

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

        if (ep + 1) % 10 == 0:
            print(f"[RR Episode {ep}] Waiting: {num_waiting_tasks}, Done: {num_done_tasks}, Failed: {num_failed_tasks}")

    print("\n==== Round Robin Policy All-Episode Statistics ====")
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

    print("\n== Round Robin Baseline Finished ==")


if __name__ == "__main__":
    round_robin_baseline(config_path=CONFIG_PATH, num_episodes=NUM_EPISODES)
