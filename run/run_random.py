import os
import numpy as np
from envs import MultiplexEnv
from visualization.monitor import plot_task_trajectories


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, "configs", "demo_3x3.json")
    env = MultiplexEnv(config_path=config_path)
    obs = env.reset()

    done = False
    total_reward = 0
    step = 0

    while not done:
        action_dict = {}
        for layer_id, layer_obs in obs.items():
            num_workers = layer_obs["worker_loads"].shape[0]
            num_tasks = layer_obs["task_queue"].shape[0]
            action = np.random.randint(0, 2, size=(num_workers, num_tasks)).tolist()
            action_dict[layer_id] = action

        obs, reward, done, info = env.step(action_dict)
        total_reward += reward[0] if isinstance(reward, tuple) else reward
        print(f"Step {step} | Reward: {reward} | Done: {done}")
        step += 1

    print("Final KPI:", info["kpi"])
    print("Total Reward:", total_reward)
    plot_task_trajectories(env.chain.finished_tasks)
