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
                 exec_efficiency_coef: Optional[Dict[str, float]] = None):
        self.id = id
        self.layer_id = layer_id

        self.cost_map = cost_map
        self.utility_map = utility_map
        self.capacity_map = capacity_map
        self.max_total_load = max_total_load
        self.current_load_map = {k: 0 for k in capacity_map}
        self.total_current_load = 0
        self.failure_prob_map = failure_prob_map or {k: 0.1 for k in cost_map}
        self.exec_efficiency_coef = exec_efficiency_coef or {k: 1.0 for k in cost_map}

        self.processing_tasks: List[Tuple[Task, float, float, float]] = []  # (task, total_amount, current_amount, unit_per_step)
        self.task_history: List[Tuple[Task, float]] = []
        self.time = 0
        self.failure_exec_cost_base = 12.0

        self.total_cost_map = {t: 0.0 for t in cost_map}
        self.total_util_map = {t: 0.0 for t in utility_map}
        self.amount_map = {t: 0 for t in cost_map}

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

        unit_per_step = self.exec_efficiency_coef.get(task.task_type, 1.0)
        self.processing_tasks.append((task, amount, amount, unit_per_step))
        self.current_load_map[task.task_type] += amount
        self.total_current_load += amount

        task.unassigned_amount -= amount
        task.assigned_worker.append(self.id)
        task.start_time = task.start_time or current_time
        return True

    def step(self) -> List[Tuple[Task, float, float, float]]:
        finished = []
        new_queue = []

        for task, total_amount, current_amount, unit_per_step in self.processing_tasks:
            # task: 当前执行的任务
            # total_amount: 分配给该worker的总任务量
            # current_amount: 当前剩余量
            # unit_per_step: 每步可执行量

            p = self.failure_prob_map.get(task.task_type, 0.001)
            if random.random() < p:
                # 执行失败，强行结束
                task.status = "failed"
                task.failed = True
                task.remaining_amount = 0
                self.task_history.append((task, total_amount))
                self.current_load_map[task.task_type] -= total_amount
                self.total_current_load -= total_amount
                cost = self.failure_exec_cost_base
                utility = 0.0

                self.total_cost_map[task.task_type] += cost
                self.amount_map[task.task_type] += total_amount

                finished.append((task, total_amount, cost, utility))
                continue

            # 成功执行一部分
            performed_amount = min(unit_per_step, current_amount)
            task.remaining_amount -= performed_amount
            current_amount -= performed_amount

            # === 即时 reward（按步产生） ===
            step_cost = self.get_cost(task, performed_amount)
            step_utility = self.get_utility(task, performed_amount)

            self.total_cost_map[task.task_type] += step_cost
            self.total_util_map[task.task_type] += step_utility
            self.amount_map[task.task_type] += performed_amount

            if current_amount <= 0:
                # 当前worker上的这份任务已执行完
                self.task_history.append((task, total_amount))
                self.current_load_map[task.task_type] -= total_amount
                self.total_current_load -= total_amount
                finished.append((task, performed_amount, step_cost, step_utility))
            else:
                # 继续排队
                new_queue.append((task, total_amount, current_amount, unit_per_step))
                finished.append((task, performed_amount, step_cost, step_utility))  # <== 即时回报也记录下来

        self.processing_tasks = new_queue
        return finished

    def get_cost(self, task: Task, amount: float) -> float:
        return self.cost_map.get(task.task_type, 1.0) * amount

    def get_utility(self, task: Task, amount: float) -> float:
        return self.utility_map.get(task.task_type, 1.0) * amount

    def get_profile(self) -> Tuple[List[float], List[float]]:
        """
        直接返回历史平均 cost 和 utility，不再遍历 task_history。
        输出：
          avg_cost_list, avg_util_list
        顺序与 self.cost_map.keys() 保持一致。
        """
        avg_cost = []
        avg_util = []
        for t in self.cost_map:
            amount = self.amount_map.get(t, 0)
            if amount > 0:
                avg_cost.append(self.total_cost_map[t] / amount)
                avg_util.append(self.total_util_map[t] / amount)
            else:
                avg_cost.append(0.0)
                avg_util.append(0.0)
        return avg_cost, avg_util


class Layer:
    def __init__(self, layer_id: int, worker_configs: List[dict]):
        self.layer_id = layer_id
        self.workers = [Worker(id=i, layer_id=layer_id, **cfg) for i, cfg in enumerate(worker_configs)]
        self.task_queue: List[Task] = []

        self.failure_base_cost = 10.0
        self.failure_utility = 0.0

    def add_task(self, task: Task):
        self.task_queue.append(task)

    def dispatch_tasks(self, actions: List[List[float]], current_time: int) -> Tuple[List[Task], float]:
        """
        全局排序分配任务：不再按 worker 顺序，而是将所有 (worker, task, ratio) flatten 后统一排序。

        """
        assigned_tasks = set()
        task_start_unassigned = [t.unassigned_amount for t in self.task_queue]
        total_assign_amount = 0.0

        # 记录所有可分配项 (worker_id, task_idx, ratio)
        dispatch_list = []
        for worker_id, allocation in enumerate(actions):
            for task_idx, ratio in enumerate(allocation):
                if ratio <= 0:
                    continue
                if task_idx >= len(self.task_queue):
                    continue
                dispatch_list.append((worker_id, task_idx, ratio))

        # 按 ratio 降序排序（更强烈意图优先执行）
        dispatch_list.sort(key=lambda x: -x[2])

        for worker_id, task_idx, ratio in dispatch_list:
            worker = self.workers[worker_id]
            task = self.task_queue[task_idx]

            if task.status != "waiting" or task.failed or task.unassigned_amount <= 0:
                continue

            task_type = task.task_type
            # worker 的剩余容量
            cap_left = worker.capacity_map.get(task_type, 0) - worker.current_load_map.get(task_type, 0)
            total_left = worker.max_total_load - worker.total_current_load

            if cap_left <= 0 or total_left <= 0:
                continue

            # 候选分配量 = 比例 × 任务初始未分配量
            ratio = min(max(ratio, 0.0), 1.0)
            proposed_amount = ratio * task_start_unassigned[task_idx]

            # 实际可执行分配量
            assign_amount = min(proposed_amount, task.unassigned_amount, cap_left, total_left)

            if assign_amount <= 0:
                continue

            success = worker.assign_task(task, assign_amount, current_time)
            if success:
                assigned_tasks.add(task)
                total_assign_amount += assign_amount

        # 清除完成、失败或被分配完的任务，只保留 still waiting 的任务
        self.task_queue = [
            t for t in self.task_queue
            if t.status == "waiting" and not t.failed and t.unassigned_amount > 0
        ]
        return list(assigned_tasks), total_assign_amount

    def step(self, current_time: int) -> List[Tuple[Task, float, float, float]]:
        finished: List[Tuple[Task, float, float, float]] = []
        for worker in self.workers:
            worker.time = current_time
            finished += worker.step()

        # 检查是否有任务超时失败
        timeout_failed = []
        for task in list(self.task_queue):
            if current_time - task.arrival_time >= task.timeout:
                task.status = "failed"
                task.failed = True
                timeout_failed.append((task, 0.0, self.failure_base_cost, self.failure_utility))
                self.task_queue.remove(task)
        finished += timeout_failed

        # 收集所有超时失败的任务 ID
        timeout_task_ids = {task.id for task, _, _, _ in timeout_failed}

        # 清除所有 worker 中已分配但该任务已超时的条目
        for worker in self.workers:
            worker.processing_tasks = [
                (t, ta, ca, u)
                for (t, ta, ca, u) in worker.processing_tasks
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
        self.step_kpis = {}
        self.finished_tasks: List[Task] = []

        self.cumulative_kpis = {
            "time": 0,
            "tasks_done": 0,
            "total_failures": 0,
            "total_cost": 0.0,
            "total_utility": 0.0,
            "per_layer": {i: {
                "tasks_done": 0,
                "failures": 0,
                "cost": 0.0,
                "utility": 0.0
            } for i in range(len(worker_layer_configs))}
        }

    def insert_tasks(self, task_list: List[Task]):
        for task in task_list:
            layer_id = task.route[task.current_layer_index]
            self.layers[layer_id].add_task(task)
            task.trajectory.append((layer_id, self.time))

    def apply_action(self, action_dict: Dict[int, List[List[float]]]) -> Dict[int, float]:
        assign_stats = {}
        for layer_id, actions in action_dict.items():
            _, total_assigned = self.layers[layer_id].dispatch_tasks(actions, self.time)
            assign_stats[layer_id] = total_assigned
        return assign_stats

    def step(self):
        self.step_kpis = {}
        all_finished = []

        # 第一步：先让所有 layer 执行 step，收集结果，不做任何任务推进
        for layer_id, layer in enumerate(self.layers):
            finished = layer.step(self.time)
            step_cost = sum(c for _, _, c, _ in finished)
            step_util = sum(u for _, _, _, u in finished)
            self.step_kpis[layer_id] = {
                "step_cost": step_cost,
                "step_utility": step_util,
            }

            self.cumulative_kpis["total_cost"] += step_cost
            self.cumulative_kpis["total_utility"] += step_util
            self.cumulative_kpis["per_layer"][layer_id]["cost"] += step_cost
            self.cumulative_kpis["per_layer"][layer_id]["utility"] += step_util

            for item in finished:
                all_finished.append((layer_id, *item))

        # 第二步：统一处理所有 finished 任务，避免推进后被立即执行
        for layer_id, task, amount, cost, util in all_finished:
            if task.status == "done":
                continue

            self.cumulative_kpis["tasks_done"] += 1
            self.cumulative_kpis["per_layer"][layer_id]["tasks_done"] += 1

            if task.failed:
                self.cumulative_kpis["total_failures"] += 1
                self.cumulative_kpis["per_layer"][layer_id]["failures"] += 1
                self.finished_tasks.append(task)
                continue

            # 这里只处理完成任务（remaining_amount <= 0）
            if task.remaining_amount <= 0:
                task.status = "done"
                task.finish_time = self.time + 1
                task.current_layer_index += 1

                if task.current_layer_index < len(task.route):
                    next_layer = task.route[task.current_layer_index]
                    task.remaining_amount = task.amount
                    task.unassigned_amount = task.amount
                    task.status = "waiting"
                    task.assigned_worker = []
                    task.trajectory.append((next_layer, self.time))
                    self.layers[next_layer].add_task(task)
                else:
                    self.finished_tasks.append(task)

        self.time += 1
        self.cumulative_kpis["time"] = self.time

    def collect_step_kpis(self) -> Dict[int, Dict[str, float]]:
        return self.step_kpis

    def collect_kpis(self) -> Dict:
        return self.cumulative_kpis

    def get_system_load_ratio(self) -> float:
        """
        计算全系统的当前总负载 / 最大总容量，作为 global_context。
        """
        total_load = 0.0
        total_cap = 0.0
        for layer in self.layers:
            for w in layer.workers:
                total_load += w.total_current_load
                total_cap += w.max_total_load
        return (total_load / total_cap) if total_cap > 0 else 0.0
