import time
import gym
from gym import spaces
import numpy as np
from typing import Dict, Any
from .core_chain import IndustrialChain, Task
from .env_init import load_env_config, generate_task_schedule, generate_worker_layer_config


class MultiplexEnv(gym.Env):
    def __init__(self, config_path: str, schedule_save_path: str = None, worker_config_save_path: str = None,
                 schedule_load_path: str = None, worker_config_load_path: str = None):
        super().__init__()
        self.schedule_save_path = schedule_save_path
        self.worker_config_save_path = worker_config_save_path
        self.schedule_load_path = schedule_load_path
        self.worker_config_load_path = worker_config_load_path

        self.config = load_env_config(config_path)
        self.task_schedule = generate_task_schedule(self.config, save_path=self.schedule_save_path, load_path=self.schedule_load_path)
        self.worker_config = generate_worker_layer_config(self.config, save_path=self.worker_config_save_path, load_path=self.worker_config_load_path)
        self.chain = IndustrialChain(self.worker_config)

        self.max_steps = self.config.get("max_steps", 50)
        self.current_step = 0
        self.alpha = self.config.get("alpha", 1.0)
        self.beta = self.config.get("beta", 1.0)

        self.num_layers = self.config["num_layers"]
        self.num_pad_tasks = self.config.get("num_pad_tasks", 10)

        num_task_types = len(self.config["task_types"])

        self.observation_space = spaces.Dict({
            layer_id: spaces.Dict({
                "task_queue": spaces.Box(
                    low=0, high=1,
                    shape=(self.num_pad_tasks, 4 + num_task_types),
                    dtype=np.float32
                ),
                "worker_loads": spaces.Box(
                    low=0, high=np.inf,
                    shape=(len(self.worker_config[layer_id]), num_task_types + 1),
                    dtype=np.float32
                ),
                "worker_profile": spaces.Box(
                    low=0, high=np.inf,
                    shape=(len(self.worker_config[layer_id]), 2 * num_task_types),
                    dtype=np.float32
                ),
                "true_last_action": spaces.Box(
                    low=0, high=1,
                    shape=(len(self.worker_config[layer_id]), self.num_pad_tasks),
                    dtype=np.float32
                ),
            }) for layer_id in range(self.num_layers)
        })

        self.action_space = spaces.Dict({
            layer_id: spaces.Box(low=0, high=1.0, shape=(len(self.worker_config[layer_id]), self.num_pad_tasks), dtype=np.float32)
            for layer_id in range(self.num_layers)
        })

        self.original_schedule_dict = {
            t: [task.to_dict() for task in task_list]
            for t, task_list in self.task_schedule.items()
        }

    def reset(self, with_new_schedule=False, arrival_rate=None):
        self.chain = IndustrialChain(self.worker_config)
        self.current_step = 0

        for layer in self.chain.layers:
            layer.last_actions = np.zeros(
                (len(layer.workers), self.num_pad_tasks), dtype=np.float32)

        if with_new_schedule or arrival_rate is not None:
            self.task_schedule = generate_task_schedule(self.config, arrival_rate=arrival_rate)
        else:
            self.task_schedule = {
                int(t): [Task.from_dict(d) for d in task_list]
                for t, task_list in self.original_schedule_dict.items()
            }
        return self._get_obs()

    def step(self, action_dict: Dict[int, Any]):
        """
        单步：注入任务 -> 应用动作 -> 系统推进 -> 观测/奖励/终止/信息
        奖励从 _get_reward 解包为 (total_reward, reward_detail)，并放入 info。
        """
        # 1) 任务注入
        new_tasks = self.task_schedule.get(self.current_step, [])
        self.chain.insert_tasks(new_tasks)

        # 2) 应用动作并推进系统
        assign_stats = self.chain.apply_action(action_dict)
        self.chain.step()

        # 3) 推进时间步与观测
        self.current_step += 1
        obs = self._get_obs()

        # 4) 奖励（解包：标量 + 明细）
        reward = self._get_reward(assign_stats=assign_stats)

        # 5) 终止判定
        done = self.current_step >= self.max_steps

        # 6) info：透传奖励明细与（可选）终结任务ID
        info = self._get_info()
        info["reward_detail"] = reward[1]
        if hasattr(self.chain, "finalized_this_step"):
            info["finalized_task_ids"] = [getattr(t, "id", None) for t in self.chain.finalized_this_step]

        return obs, reward, done, info

    def _get_obs(self):
        """
        返回完整的 observation dict，结构：
        {
          5: {"task_queue": ..., "worker_loads": ..., "worker_profile": ...},
          1: { ... },
          ...,
          "global_context": np.array([load_ratio], dtype=np.float32)
        }
        """
        obs = {}
        num_task_types = len(self.config["task_types"])

        for layer_id, layer in enumerate(self.chain.layers):
            # —— 构造 task_queue 特征 (num_pad_tasks, 4 + n_task_types) ——
            task_features = np.zeros((self.num_pad_tasks, 4 + num_task_types), dtype=np.float32)
            for i, task in enumerate(layer.task_queue[:self.num_pad_tasks]):
                # 未分配量
                norm_unassigned = task.unassigned_amount / max(self.config["task_amount_range"])
                # 剩余时间
                rem_time = max(0, task.timeout - (self.current_step - task.arrival_time))
                norm_remain = rem_time / max(self.config["task_timeout_range"])
                # 等待时间 = 当前步 - 到达步
                wait_time = self.current_step - task.arrival_time
                norm_wait = wait_time / max(self.config["max_steps"], 1)
                # 类型 one-hot
                one_hot = np.zeros(num_task_types, dtype=np.float32)
                idx = self.config["task_types"].index(task.task_type)
                one_hot[idx] = 1.0
                # 有效标志
                valid_flag = 1.0

                task_features[i] = np.concatenate([
                    [norm_unassigned, norm_remain, norm_wait, valid_flag],
                    one_hot
                ])

            # —— 构造 worker_loads 特征 (n_worker, 2*n_task_type + 2) ——
            loads = []
            for w in layer.workers:
                # 每类任务的负载余量 + 总负载余量
                per_type_remain = []
                for t in self.config["task_types"]:
                    cap = w.capacity_map.get(t, 1)
                    used = w.current_load_map.get(t, 0.0)
                    per_type_remain.append(cap - used)
                # 总负载余量
                tot_remain = w.max_total_load - w.total_current_load

                # 拼成长度 = n_task_type + 1
                loads.append(per_type_remain + [tot_remain])

            worker_loads = np.array(loads, dtype=np.float32)

            # —— 构造 worker_profile 特征 (n_worker, 2*n_task_type) ——
            profiles = []
            for w in layer.workers:
                avg_cost, avg_util = w.get_profile()
                avg_cost = [ac / self.config["worker_cost_range"][1] for ac in avg_cost]
                avg_util = [au / self.config["worker_utility_range"][1] for au in avg_util]
                profiles.append(avg_cost + avg_util)
            worker_profile = np.array(profiles, dtype=np.float32)

            true_last_actions = layer.last_actions.copy()
            obs[layer_id] = {
                "task_queue":   task_features,
                "worker_loads": worker_loads,
                "worker_profile": worker_profile,
                "true_last_actions": true_last_actions
            }
        return obs

    def _get_reward(self, assign_stats=None):
        """
        计算当步总奖励与分层明细。
        约定：core_chain 已按方案A在终结步把各层的 step_cost/step_utility 写入 step_kpis；
             非终结步这些值为0，从而实现稀疏外在回报。
        """
        if assign_stats is None:
            assign_stats = {}

        # 兼容两种取法；缺失时给空 dict
        step_kpis = (
                        self.chain.collect_step_kpis()
                        if hasattr(self.chain, "collect_step_kpis")
                        else getattr(self.chain, "step_kpis", {})
                    ) or {}

        layer_rewards = {}
        total_reward = 0.0
        total_cost = 0.0
        total_util = 0.0
        total_assign_bonus = 0.0
        total_wait_penalty = 0.0

        for layer_id, kpi in step_kpis.items():
            step_cost = float((kpi or {}).get("step_cost", 0.0))
            step_util = float((kpi or {}).get("step_utility", 0.0))

            # 分配奖励与等待惩罚：仅做日志记录，不计入总奖励（与你原逻辑一致）
            assign_bonus = float(assign_stats.get(layer_id, 0.0))
            wait_penalty = 0.0
            if hasattr(self.chain, "layers") and layer_id < len(self.chain.layers):
                q = getattr(self.chain.layers[layer_id], "task_queue", [])
                for task in q:
                    wait_penalty += float(self.current_step - task.arrival_time)

            # 分层奖励（只由 cost/utility 计算）
            r_layer = self.beta * step_util - self.alpha * step_cost

            layer_rewards[int(layer_id)] = {
                "cost": step_cost,
                "utility": step_util,
                "reward": r_layer,
                "assign_bonus": assign_bonus,
                "wait_penalty": wait_penalty,
            }

            total_reward += r_layer
            total_cost += step_cost
            total_util += step_util
            total_assign_bonus += assign_bonus
            total_wait_penalty += wait_penalty

        reward_detail = {
            "mode": getattr(self, "reward_mode", "final_only_per_layer"),
            "alpha": getattr(self, "alpha", None),
            "beta": getattr(self, "beta", None),
            "layer_rewards": layer_rewards,
            "global_summary": {
                "total_cost": total_cost,
                "total_utility": total_util,
                "total_assign_bonus": total_assign_bonus,
                "total_wait_penalty": total_wait_penalty,
                "total_reward": total_reward,
            },
        }

        # 返回 (标量总奖励, 详细字典)
        return float(total_reward), reward_detail

    def _get_info(self):
        # 输出详细KPI与episode参数，方便后续分析
        info = {
            "kpi": self.chain.collect_kpis(),
            "alpha": self.alpha,
            "beta": self.beta,
            "current_step": self.current_step
        }
        return info

    def render(self, mode: str = "human", *, box_width: int = 8, box_gap: int = 2):
        """
        ASCII 面板：
          顶部：t 和累计 Reward
          每层：若干 worker 载荷小方块 + 右侧 processing 任务列表（非等待队列）
        参数：
          mode: "human" 直接打印；"ansi" 返回字符串
          box_width: 每个小方块内部宽度（字符数）
          box_gap: 同层相邻方块之间的空格数
        """

        def _safe_ratio(used: float, total: float) -> float:
            if total is None or total <= 0:
                return 0.0
            r = 0.0 if used is None else float(used) / float(total)
            return max(0.0, min(1.0, r))

        def _cell_lines(ratio: float, w: int):
            # 三行小方块：┌──┐、│██ │、└──┘
            fill = int(round(ratio * w))
            return (
                "┌" + "─" * w + "┐",
                "│" + "█" * fill + " " * (w - fill) + "│",
                "└" + "─" * w + "┘",
            )

        def _collect_processing_task_ids(layer):
            """从各 worker 的 processing_tasks 中收集正在处理的任务 id（去重）。"""
            ids, seen = [], set()
            for w in getattr(layer, "workers", []):
                pt = getattr(w, "processing_tasks", None)
                if not pt:
                    continue
                # 兼容 list/tuple[Task or (Task, amount)] / dict / set 等多种形态
                try:
                    iterable = pt.items() if isinstance(pt, dict) else pt
                except Exception:
                    iterable = []
                for item in iterable:
                    task = item[0] if isinstance(item, (tuple, list)) else item
                    tid = getattr(task, "id", None)
                    if tid is None:
                        # 如果存的是 id 自身
                        if isinstance(task, (int, str)) and task not in seen:
                            seen.add(task);
                            ids.append(task)
                        continue
                    if tid not in seen:
                        seen.add(tid);
                        ids.append(tid)
            return ids

        # —— 顶部信息：时间步与累计奖励（按 alpha/beta 组合）
        ksum = getattr(self.chain, "cumulative_kpis", None) or {}
        total_cost = float(ksum.get("total_cost", 0.0))
        total_util = float(ksum.get("total_utility", 0.0))
        cum_reward = self.beta * total_util - self.alpha * total_cost

        lines = []
        lines.append(f"t={self.current_step}    Reward={cum_reward:.2f}")

        # —— 各层绘制 —— #
        label_w = 3  # 左侧层标签宽度
        gap = " " * max(1, int(box_gap))

        for lid, layer in enumerate(self.chain.layers):
            workers = getattr(layer, "workers", [])
            # 为每个 worker 画一个三行小方块（载荷=used/total）
            top_row, mid_row, bot_row = [], [], []
            for w in workers:
                used = float(getattr(w, "total_current_load", 0.0))
                total = float(getattr(w, "max_total_load", 0.0))
                r = _safe_ratio(used, total)
                t, m, b = _cell_lines(r, box_width)
                top_row.append(t);
                mid_row.append(m);
                bot_row.append(b)

            # 右侧 processing 队列
            proc_ids = _collect_processing_task_ids(layer)
            proc_str = "  →  " + " ".join(f"[t{tid}]" for tid in proc_ids) if proc_ids else "  →  (processing: none)"

            # 组装三行（把层标签放在中间行左侧）
            if workers:
                lines.append(" " * label_w + gap.join(top_row))
                lines.append(f"L{lid}".rjust(label_w) + gap.join(mid_row) + proc_str)
                lines.append(" " * label_w + gap.join(bot_row))
            else:
                # 没 worker 的层也保持占位与 processing 信息
                empty_box = _cell_lines(0.0, box_width)
                lines.append(" " * label_w + empty_box[0])
                lines.append(f"L{lid}".rjust(label_w) + empty_box[1] + proc_str)
                lines.append(" " * label_w + empty_box[2])

            # 层间空白行
            lines.append("")

        out = "\n".join(lines)
        if mode == "ansi":
            return out
        print(out)  # mode == "human"
        time.sleep(0.5)
        return None

