import os
import json
import time
import random
import numpy as np
from typing import Dict, List
from .core_chain import Task, Worker


def load_env_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 默认值填充
    config.setdefault("task_types", ["A", "B", "C"])
    config.setdefault("task_route", [0, 1, 2])
    config.setdefault("task_amount_range", [1, 5])
    config.setdefault("max_steps", 50)
    config.setdefault("task_arrival_mode", "poisson")
    config.setdefault("task_arrival_rate", 5)
    config.setdefault("task_timeout_range", [30, 80])
    config.setdefault("num_layers", 3)
    config.setdefault("workers_per_layer", [3, 3, 3])
    config.setdefault("worker_cost_range", [1.0, 5.0])
    config.setdefault("worker_utility_range", [0.2, 1.0])
    config.setdefault("worker_task_capacity_range", [3, 10])
    config.setdefault("worker_total_capacity_range", [10, 20])
    config.setdefault("worker_failure_range", [0.05, 0.2])
    config.setdefault("task_schedule_load_path", "")
    config.setdefault("worker_type_probs", [0.2, 0.3, 0.3, 0.2])
    return config


def generate_task_schedule(config: Dict, load_path: str = "", save_path: str = "") -> Dict[int, List[Task]]:
    if load_path and os.path.exists(load_path):
        print(f"[INFO] Loading task schedule from {load_path}")
        with open(load_path, "r") as f:
            raw = json.load(f)
        schedule = {
            int(t): [Task.from_dict(d) for d in task_list]
            for t, task_list in raw.items()
        }
        return schedule

    # ==== 自动生成逻辑 ====
    task_types = config["task_types"]
    task_route = config["task_route"]
    amount_min, amount_max = config["task_amount_range"]
    max_steps = config["max_steps"]
    arrival_mode = config["task_arrival_mode"]
    arrival_rate = config["task_arrival_rate"]
    timeout_min, timeout_max = config["task_timeout_range"]

    schedule = {}
    task_id = 0
    max_task_amount = config["task_amount_range"][1]
    min_exec_efficiency = config["worker_exec_efficiency_range"][0]
    max_exec_time = max_task_amount / min_exec_efficiency
    for t in range(int(max_steps - max_exec_time)):
        if arrival_mode == "poisson":
            num_tasks = np.random.poisson(arrival_rate)
        else:
            num_tasks = 0
        for _ in range(num_tasks):
            task = Task(
                id=task_id,
                arrival_time=t,
                task_type=random.choice(task_types),
                amount=random.randint(amount_min, amount_max),
                route=task_route.copy(),
                timeout=random.randint(timeout_min, timeout_max)
            )
            schedule.setdefault(t, []).append(task)
            task_id += 1

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({
                str(t): [task.to_dict() for task in task_list]
                for t, task_list in schedule.items()
            }, f)
        print(f"[INFO] Saved task schedule to {save_path}")

    return schedule


def generate_worker_layer_config(config: Dict, load_path: str = "", save_path: str = "") -> List[List[dict]]:
    if load_path and os.path.exists(load_path):
        print(f"[INFO] Loading worker config from {load_path}")
        with open(load_path, "r") as f:
            raw = json.load(f)
        # 转换为 List[List[dict]] 格式，并移除多余字段
        raw_sorted = [raw[k] for k in sorted(raw.keys(), key=int)]
        for layer in raw_sorted:
            for worker_cfg in layer:
                worker_cfg.pop("id", None)
        return raw_sorted

    # ==== 自动生成逻辑 ====
    task_types = config["task_types"]
    num_layers = config["num_layers"]
    workers_per_layer = config["workers_per_layer"]
    exec_efficiency_range = config["worker_exec_efficiency_range"]
    cap_range = config["worker_task_capacity_range"]
    total_capacity = random.randint(*config["worker_total_capacity_range"])
    fail_range = config["worker_failure_range"]
    probs = config["worker_type_probs"]

    assert len(probs) == 4 and abs(sum(probs) - 1.0) < 1e-6, \
        "worker_type_probs 必须长度为 4 且总和为 1"

    cost_min, cost_max = config["worker_cost_range"]
    util_min, util_max = config["worker_utility_range"]
    cost_mid = (cost_min + cost_max) / 2
    util_mid = (util_min + util_max) / 2

    worker_layer_config = {}
    for layer_id in range(num_layers):
        layer_cfg = []
        for _ in range(workers_per_layer[layer_id]):
            t = random.choices([0, 1, 2, 3], weights=probs, k=1)[0]
            if t == 0:  # A: high util, low cost
                cost_range = (cost_min, cost_mid)
                util_range = (util_mid, util_max)
            elif t == 1:  # B: high util, high cost
                cost_range = (cost_mid, cost_max)
                util_range = (util_mid, util_max)
            elif t == 2:  # C: low util, low cost
                cost_range = (cost_min, cost_mid)
                util_range = (util_min, util_mid)
            else:  # D: low util, high cost
                cost_range = (cost_mid, cost_max)
                util_range = (util_min, util_mid)

            cost_map = {}
            util_map = {}
            fail_map = {}
            exec_eff_map = {}
            for task_type in task_types:
                cost_map[task_type] = round(random.uniform(*cost_range), 2)
                util_map[task_type] = round(random.uniform(*util_range), 2)
                fail_map[task_type] = round(random.uniform(*fail_range), 3)
                exec_eff_map[task_type] = round(random.uniform(*exec_efficiency_range), 2)
            cap_map = {t: random.randint(*cap_range) for t in task_types}

            worker_cfg = {
                "cost_map": cost_map,
                "utility_map": util_map,
                "capacity_map": cap_map,
                "max_total_load": total_capacity,
                "failure_prob_map": fail_map,
                "exec_efficiency_coef": exec_eff_map
            }
            layer_cfg.append(worker_cfg)
        worker_layer_config[layer_id] = layer_cfg

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(worker_layer_config, f)
        print(f"[INFO] Saved worker config to {save_path}")

    return [worker_layer_config[i] for i in range(num_layers)]
