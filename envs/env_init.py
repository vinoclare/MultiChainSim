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
    config.setdefault("is_save_schedule", True)
    return config


def generate_task_schedule(config: Dict) -> Dict[int, List[Task]]:
    if "task_schedule_load_path" in config and os.path.exists(config["task_schedule_load_path"]):
        print(f"[INFO] Loading task schedule from {config['task_schedule_load_path']}")
        with open(config["task_schedule_load_path"], "r") as f:
            raw = json.load(f)
        schedule = {
            int(t): [Task.from_dict(d) for d in task_list]
            for t, task_list in raw.items()
        }
        return schedule

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
            num_tasks = 0  # future: add other modes
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

    if config.get("is_save_schedule", False):
        os.makedirs("schedules", exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"schedules/{timestamp}.json"
        with open(filename, "w") as f:
            json.dump({
                str(t): [task.to_dict() for task in task_list]
                for t, task_list in schedule.items()
            }, f)
        print(f"[INFO] Saved task schedule to {filename}")
    return schedule


def generate_worker_layer_config(config: Dict) -> List[List[dict]]:
    task_types = config["task_types"]
    num_layers = config["num_layers"]
    workers_per_layer = config["workers_per_layer"]
    exec_efficiency_range = config["worker_exec_efficiency_range"]
    cap_range = config["worker_task_capacity_range"]
    total_capacity = random.randint(*config["worker_total_capacity_range"])
    fail_range = config["worker_failure_range"]

    # ABCD四类采样区间
    # 可根据实际情况调整（推荐不要重叠，便于分析）
    abcd_profile = {
        "A": {"cost": (1.0, 1.5), "utility": (2.5, 3.0)},
        "B": {"cost": (2.5, 3.0), "utility": (2.5, 3.0)},
        "C": {"cost": (1.0, 1.5), "utility": (1.0, 1.5)},
        "D": {"cost": (2.5, 3.0), "utility": (1.0, 1.5)},
    }
    abcd_types = ["A", "B", "C", "D"]

    worker_layer_config = []
    for l in range(num_layers):
        layer_cfg = []
        for _ in range(workers_per_layer[l]):
            cost_map = {}
            util_map = {}
            fail_map = {}
            exec_efficiency_map = {}
            for t in task_types:
                # 随机选一个类型
                label = random.choices(abcd_types, weights=[0.2, 0.3, 0.3, 0.2])[0]

                # 按类型采样
                c_rng = abcd_profile[label]["cost"]
                u_rng = abcd_profile[label]["utility"]

                cost_map[t] = round(random.uniform(*c_rng), 2)
                util_map[t] = round(random.uniform(*u_rng), 2)
                fail_map[t] = round(random.uniform(*fail_range), 3)
                exec_efficiency_map[t] = round(random.uniform(*exec_efficiency_range), 2)
            cap_map = {t: random.randint(*cap_range) for t in task_types}

            worker_cfg = {
                "cost_map": cost_map,
                "utility_map": util_map,
                "capacity_map": cap_map,
                "max_total_load": total_capacity,
                "failure_prob_map": fail_map,
                "exec_efficiency_coef": exec_efficiency_map
            }
            layer_cfg.append(worker_cfg)
        worker_layer_config.append(layer_cfg)
    worker_layer_config = [[
        {
            'cost_map': {'A': 3.0, 'B': 1.0},
            'utility_map': {'A': 3.0, 'B': 1.0},
            'capacity_map': {'A': 50, 'B': 40},
            'max_total_load': 121,
            'failure_prob_map': {'A': 0.0, 'B': 0.0},
            'exec_efficiency_coef': {'A': 1.0, 'B': 1.0}
        },
        {
            'cost_map': {'A': 1.0, 'B': 3.0},
            'utility_map': {'A': 1.0, 'B': 3.0},
            'capacity_map': {'A': 42, 'B': 34},
            'max_total_load': 121,
            'failure_prob_map': {'A': 0.0, 'B': 0.0},
            'exec_efficiency_coef': {'A': 1.0, 'B': 1.0}
        }
    ]]
    return worker_layer_config

