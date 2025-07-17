import os
import csv
import numpy as np
from pathlib import Path
from envs import MultiplexEnv


def round_robin_baseline(env, num_episodes=10):
    """返回五个标量：avg_reward, avg_cost, avg_utility, reward_std, tasks_done"""
    num_layers = len(env.chain.layers)
    rr_pointer = [0] * num_layers

    ep_rewards, ep_costs, ep_utils, ep_done = [], [], [], []

    for _ in range(num_episodes):
        env.reset()
        done, total_reward = False, 0.0

        while not done:
            action_dict = {}
            for l, layer in enumerate(env.chain.layers):
                tq, n_worker, n_task = layer.task_queue, len(layer.workers), len(layer.task_queue)
                act = [[0] * n_task for _ in range(n_worker)]

                for t_idx, task in enumerate(tq):
                    remain, ptr, tried = task.unassigned_amount, 2, 0
                    while remain > 0 and tried < n_worker:
                        w_idx = ptr % n_worker
                        w = layer.workers[w_idx]
                        cap_t = w.capacity_map.get(task.task_type, 0)
                        used_t = w.current_load_map.get(task.task_type, 0)
                        assign_amt = min(remain,
                                         cap_t - used_t,
                                         w.max_total_load - w.total_current_load)
                        if assign_amt > 0:
                            act[w_idx][t_idx] += assign_amt
                            remain -= assign_amt
                        ptr += 1
                        tried += 1
                    rr_pointer[l] = ptr % n_worker
                action_dict[l] = act

            obs, reward, done, info = env.step(action_dict)
            total_reward += reward if np.isscalar(reward) else reward[0]

        print(f"Episode reward: {total_reward}")
        kpi = info["kpi"]
        ep_rewards.append(total_reward)
        ep_costs.append(kpi["total_cost"])
        ep_utils.append(kpi["total_utility"])
        ep_done.append(kpi["tasks_done"])

    return dict(
        avg_reward=np.mean(ep_rewards),
        avg_cost=np.mean(ep_costs),
        avg_utility=np.mean(ep_utils),
        reward_std=np.std(ep_rewards),
        avg_tasks_done=np.mean(ep_done),
    )


def is_config_leaf(dir_path: Path):
    """判断该目录是否含四个核心 JSON."""
    files = {p.name for p in dir_path.iterdir() if p.is_file()}
    need = {"env_config.json", "worker_config.json",
            "train_schedule.json", "eval_schedule.json"}
    return need.issubset(files)


def save_metric_csv(path: Path, value: float):
    """生成形如 step,value 的单列 CSV 文件."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "value"])
        w.writerow([0, f"{value:.6f}"])


def batch_run(config_root="./configs/task", result_root="./logs/exp2"):
    config_root = Path(config_root).resolve()
    result_root = Path(result_root).resolve()

    # 遍历所有 leaf 配置目录
    for dir_path in config_root.rglob("*"):
        if not dir_path.is_dir() or not is_config_leaf(dir_path):
            continue

        # 读取配置文件
        env_cfg = dir_path / "env_config.json"
        worker_cfg = dir_path / "worker_config.json"
        eval_sch = dir_path / "eval_schedule.json"

        try:
            env = MultiplexEnv(env_cfg,
                               schedule_load_path=eval_sch,
                               worker_config_load_path=worker_cfg)

            metrics = round_robin_baseline(env, num_episodes=10)
            env.close()
        except Exception as e:
            print(f"[{dir_path.relative_to(config_root)}] 运行失败：{e}")
            continue

        # 相对路径，例如 layer/2
        rel_dir = dir_path.relative_to(config_root)

        # rr 结果目录，与 ppo 等算法并列
        algo_dir = result_root / rel_dir / "rr"
        algo_dir.mkdir(parents=True, exist_ok=True)

        save_metric_csv(algo_dir / "eval_avg_reward.csv", metrics["avg_reward"])
        save_metric_csv(algo_dir / "eval_avg_cost.csv", metrics["avg_cost"])
        save_metric_csv(algo_dir / "eval_avg_utility.csv", metrics["avg_utility"])

        print(f"[{rel_dir}] ✅ RR 结果已保存 → {algo_dir}")


if __name__ == "__main__":
    batch_run()
