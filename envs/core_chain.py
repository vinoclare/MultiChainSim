from typing import List, Optional, Dict, Tuple
import random


class Task:
    def __init__(self,
                 id: int,
                 arrival_time: int,
                 task_type: str,
                 amount: int,
                 route: Optional[List[int]] = None,
                 timeout: int = 50):
        self.id = id
        self.arrival_time = arrival_time
        self.task_type = task_type
        self.amount = amount
        self.remaining_amount = amount
        self.unassigned_amount = self.amount
        self.route = route or []
        self.current_layer_index = 0

        self.status = "waiting"
        self.start_time = None
        self.finish_time = None
        self.assigned_worker = []
        self.blocking_cost = 0.0
        self.failed = False
        self.trajectory: List[Tuple[int, int]] = []
        self.timeout = timeout

    def to_dict(self):
        return {
            "id": self.id,
            "arrival_time": self.arrival_time,
            "task_type": self.task_type,
            "amount": self.amount,
            "route": self.route,
            "timeout": self.timeout
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            id=d["id"],
            arrival_time=d["arrival_time"],
            task_type=d["task_type"],
            amount=d["amount"],
            route=d["route"],
            timeout=d["timeout"]
        )


class Worker:
    def __init__(self,
                 id: int,
                 layer_id: int,
                 cost_map: Dict[str, float],
                 utility_map: Dict[str, float],
                 capacity_map: Dict[str, float],
                 max_total_load: float,
                 failure_prob_map: Optional[Dict[str, float]] = None,
                 exec_time_coef: Optional[Dict[str, float]] = None):
        self.id = id
        self.layer_id = layer_id

        self.cost_map = cost_map
        self.utility_map = utility_map
        self.capacity_map = capacity_map
        self.max_total_load = max_total_load
        self.current_load_map = {k: 0 for k in capacity_map}
        self.total_current_load = 0
        self.failure_prob_map = failure_prob_map or {k: 0.1 for k in cost_map}
        self.exec_time_coef = exec_time_coef or {k: 1.0 for k in cost_map}

        self.processing_tasks: List[Tuple[Task, float, float, float]] = []  # (task, amount, remaining_time)
        self.task_history: List[Tuple[Task, float]] = []
        self.time = 0

    def can_accept_amount(self, task: Task, amount: float) -> bool:
        type_cap = self.capacity_map.get(task.task_type, 0)
        type_used = self.current_load_map.get(task.task_type, 0)
        return (
            type_used + amount <= type_cap and
            self.total_current_load + amount <= self.max_total_load
        )

    def assign_task(self, task: Task, amount: float, current_time: int) -> bool:
        if amount <= 0 or not self.can_accept_amount(task, amount):
            # print(
            #     f"[ASSIGN FAIL] Layer {self.layer_id} Task {task.id} to Worker {self.id} @ Layer {self.layer_id} — amount={amount}, type_load={self.current_load_map.get(task.task_type, 0)}, type_cap={self.capacity_map.get(task.task_type, 0)}, total_load={self.total_current_load}, current_load={self.total_current_load}, max={self.max_total_load}")
            return False

        exec_time = amount * self.exec_time_coef.get(task.task_type, 1.0)
        unit_per_step = amount / exec_time
        self.processing_tasks.append((task, amount, exec_time, unit_per_step))
        self.current_load_map[task.task_type] += amount
        self.total_current_load += amount

        task.unassigned_amount -= amount
        task.assigned_worker.append(self.id)
        task.start_time = task.start_time or current_time
        return True

    def step(self) -> List[Tuple[Task, float]]:
        finished = []
        new_queue = []

        for task, amount, remaining_time, unit_per_step in self.processing_tasks:
            p = self.failure_prob_map.get(task.task_type, 0.001)
            if random.random() < p:
                task.status = "failed"
                task.failed = True
                task.remaining_amount = 0
                self.task_history.append((task, amount))
                self.current_load_map[task.task_type] -= amount
                self.total_current_load -= amount
                finished.append((task, amount))
                continue

            remaining_time -= 1
            task.remaining_amount -= unit_per_step
            task.remaining_amount = max(task.remaining_amount, 0)
            # print(
            #     f"[Worker {self.id}] executed {unit_per_step:.2f} units of task {task.id}, remaining_amount = {task.remaining_amount:.2f}")
            if remaining_time <= 0:
                self.task_history.append((task, amount))
                self.current_load_map[task.task_type] -= amount
                self.total_current_load -= amount
                finished.append((task, amount))
            else:
                new_queue.append((task, amount, remaining_time, unit_per_step))

        self.processing_tasks = new_queue
        return finished

    def get_cost(self, task: Task, amount: float) -> float:
        return self.cost_map.get(task.task_type, 1.0) * amount

    def get_utility(self, task: Task, amount: float) -> float:
        return self.utility_map.get(task.task_type, 1.0) * amount


class Layer:
    def __init__(self, layer_id: int, worker_configs: List[dict]):
        self.layer_id = layer_id
        self.workers = [Worker(id=i, layer_id=layer_id, **cfg) for i, cfg in enumerate(worker_configs)]
        self.task_queue: List[Task] = []

    def add_task(self, task: Task):
        self.task_queue.append(task)
        # print(f"[Layer {self.layer_id}] 接收到任务 {task.id}")

    def dispatch_tasks(self, actions: List[List[float]], current_time: int) -> List[Task]:
        assigned_tasks = set()

        for worker_id, allocation in enumerate(actions):
            worker = self.workers[worker_id]
            for task_idx, amount in enumerate(allocation):
                if amount <= 0:
                    continue
                if task_idx >= len(self.task_queue):
                    continue

                task = self.task_queue[task_idx]
                if task.status != "waiting":
                    continue

                success = worker.assign_task(task, amount, current_time)
                if success:
                    assigned_tasks.add(task)
                    # print(
                    #     f"[DISPATCH] Layer {self.layer_id} | Assigned {amount} of Task {task.id} to Worker {worker.id}")

        # 保留未完成任务，过滤已完全完成或失败的任务
        self.task_queue = [t for t in self.task_queue if t.unassigned_amount > 0 and not t.failed]
        return list(assigned_tasks)

    def step(self, current_time: int) -> List[Task]:
        finished = []
        for worker in self.workers:
            worker.time = current_time
            finished += worker.step()

        # 检查是否有任务超时失败
        timeout_failed = []
        for task in list(self.task_queue):
            if current_time - task.arrival_time >= task.timeout:
                task.status = "failed"
                task.failed = True
                timeout_failed.append((task, 0))  # 执行量为 0
                self.task_queue.remove(task)
        finished += timeout_failed

        # 收集所有超时失败的任务 ID
        timeout_task_ids = {task.id for task, _ in timeout_failed}

        # 清除所有 worker 中已分配但该任务已超时的条目
        for worker in self.workers:
            worker.processing_tasks = [
                (t, a, r, u)
                for (t, a, r, u) in worker.processing_tasks
                if t.id not in timeout_task_ids
            ]
        return finished

    def get_kpi_snapshot(self) -> Dict:
        all_exec = [(t, a) for w in self.workers for (t, a) in w.task_history if not t.failed]
        if not all_exec:
            return {"avg_cost": 0, "avg_util": 0, "utilization": 0}

        total_cost = sum(w.get_cost(t, a) for w in self.workers for (t, a) in w.task_history if not t.failed)
        total_util = sum(w.get_utility(t, a) for w in self.workers for (t, a) in w.task_history if not t.failed)
        total_load = sum(w.total_current_load for w in self.workers)
        max_load = sum(w.max_total_load for w in self.workers)

        return {
            "avg_cost": total_cost / len(all_exec),
            "avg_util": total_util / len(all_exec),
            "utilization": total_load / max_load if max_load > 0 else 0
        }


class IndustrialChain:
    def __init__(self, worker_layer_configs: List[List[dict]]):
        self.layers: List[Layer] = [Layer(layer_id=i, worker_configs=worker_layer_configs[i])
                                    for i in range(len(worker_layer_configs))]
        self.time = 0
        self.finished_tasks: List[Task] = []

    def insert_tasks(self, task_list: List[Task]):
        for task in task_list:
            layer_id = task.route[task.current_layer_index]
            self.layers[layer_id].add_task(task)
            task.trajectory.append((layer_id, self.time))

    def apply_action(self, action_dict: Dict[int, List[List[float]]]):
        for layer_id, actions in action_dict.items():
            self.layers[layer_id].dispatch_tasks(actions, self.time)

    def step(self):
        for layer in self.layers:
            finished = layer.step(self.time)
            for task, _ in finished:
                if task.failed:
                    self.finished_tasks.append(task)
                    continue
                if task.remaining_amount <= 0:
                    task.status = "done"
                    task.finish_time = self.time + 1
                    task.current_layer_index += 1
                    if task.current_layer_index < len(task.route):
                        next_layer = task.route[task.current_layer_index]
                        task.remaining_amount = task.amount
                        task.unassigned_amount = task.amount
                        task.status = "waiting"
                        task.trajectory.append((next_layer, self.time))
                        self.layers[next_layer].add_task(task)
                        # print(f"[Step {self.time}] Task {task.id} 完成，推进到下一层")
                    else:
                        self.finished_tasks.append(task)
        self.time += 1

    def collect_kpis(self) -> Dict:
        all_tasks = self.finished_tasks
        total_tasks = len(all_tasks)
        failure_base_cost = 5.0
        failure_cost_ratio_range = (0.1, 0.5)

        success_tasks = [t for t in all_tasks if not t.failed]
        fail_tasks = [t for t in all_tasks if t.failed]

        success_cost = sum(t.amount for t in success_tasks)
        fail_cost = 0.0

        for t in fail_tasks:
            cost_ratio = random.uniform(*failure_cost_ratio_range)
            fail_cost += failure_base_cost + cost_ratio * t.amount

        total_cost = success_cost + fail_cost
        total_util = sum(t.amount for t in success_tasks)
        fail_count = len(fail_tasks)

        return {
            "time": self.time,
            "tasks_done": total_tasks,
            "total_failures": fail_count,
            "avg_cost": total_cost / total_tasks if total_tasks else 0,
            "avg_utility": total_util / total_tasks if total_tasks else 0
        }
