import gym
from gym import spaces
import numpy as np
from typing import Dict, Any
from .core_chain import IndustrialChain, Task
from .env_init import load_env_config, generate_task_schedule, generate_worker_layer_config


class MultiplexEnv(gym.Env):
    def __init__(self, config_path: str):
        super().__init__()

        self.config = load_env_config(config_path)
        self.task_schedule = generate_task_schedule(self.config)
        self.worker_config = generate_worker_layer_config(self.config)
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
                    low=0, high=1,
                    shape=(len(self.worker_config[layer_id]), num_task_types + 1),
                    dtype=np.float32
                ),
                "worker_profile": spaces.Box(
                    low=0, high=np.inf,
                    shape=(len(self.worker_config[layer_id]), 2 * num_task_types),
                    dtype=np.float32
                ),
            }) for layer_id in range(self.num_layers)
        })
        self.observation_space.spaces["global_context"] = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )

        self.action_space = spaces.Dict({
            layer_id: spaces.Box(low=0, high=1.0, shape=(len(self.worker_config[layer_id]), self.num_pad_tasks), dtype=np.float32)
            for layer_id in range(self.num_layers)
        })

        self.original_schedule_dict = {
            t: [task.to_dict() for task in task_list]
            for t, task_list in self.task_schedule.items()
        }

        # == 定义归一化分母 ==
        self.reward_normalizers = {}
        for layer_id, layer in enumerate(self.chain.layers):
            max_assign = sum(w.max_total_load for w in layer.workers)
            max_cost = max_assign * self.config["worker_cost_range"][1]
            max_util = max_assign * self.config["worker_utility_range"][1]

            self.reward_normalizers[layer_id] = {
                "max_assign": max_assign,
                "max_cost": max_cost,
                "max_util": max_util
            }

    def reset(self, with_new_schedule=False):
        self.chain = IndustrialChain(self.worker_config)
        self.current_step = 0
        if with_new_schedule:
            self.alpha = np.random.uniform(0.5, 1.5)
            self.beta = np.random.uniform(0.5, 1.5)
            self.task_schedule = generate_task_schedule(self.config)
            self.original_schedule_dict = {
                t: [task.to_dict() for task in task_list]
                for t, task_list in self.task_schedule.items()
            }
        else:
            self.task_schedule = {
                int(t): [Task.from_dict(d) for d in task_list]
                for t, task_list in self.original_schedule_dict.items()
            }
        return self._get_obs()

    def step(self, action_dict: Dict[int, Any]):
        clipped_action_dict = {}
        for layer_id, act in action_dict.items():
            act = np.clip(act, 0.0, 1.0)
            clipped_action_dict[layer_id] = act

        # 任务注入
        new_tasks = self.task_schedule.get(self.current_step, [])
        self.chain.insert_tasks(new_tasks)
        assign_stats = self.chain.apply_action(clipped_action_dict)
        self.chain.step()
        self.current_step += 1
        obs = self._get_obs()
        reward = self._get_reward(assign_stats=assign_stats)
        done = self.current_step >= self.max_steps
        info = self._get_info()
        return obs, reward, done, info

    def _get_obs(self):
        """
        返回完整的 observation dict，结构：
        {
          0: {"task_queue": ..., "worker_loads": ..., "worker_profile": ...},
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

            obs[layer_id] = {
                "task_queue":   task_features,
                "worker_loads": worker_loads,
                "worker_profile": worker_profile
            }

        # —— 全局上下文 global_context (1,) ——
        load_ratio = self.chain.get_system_load_ratio()
        obs["global_context"] = np.array([load_ratio], dtype=np.float32)

        return obs

    def _get_reward(self, assign_stats=None):
        if assign_stats is None:
            assign_stats = {}
        assign_coef = self.config.get("assign_reward_coef", 0.1)
        wait_penalty_coef = self.config.get("wait_penalty_coef", 0.01)
        max_wait = self.config.get("max_steps", 50)

        step_kpis = self.chain.collect_step_kpis()

        layer_rewards = {}
        total_reward = 0.0
        total_cost = 0.0
        total_util = 0.0

        for layer_id, kpi in step_kpis.items():
            normalizer = self.reward_normalizers[layer_id]

            # 原始值（未归一化）用于日志记录
            raw_cost = kpi.get("step_cost", 0.0)
            raw_util = kpi.get("step_utility", 0.0)
            raw_assign = assign_stats.get(layer_id, 0.0)

            # 归一化后参与 reward 组合的值
            step_cost = raw_cost / (normalizer["max_cost"] + 1e-6)
            step_util = raw_util / (normalizer["max_util"] + 1e-6)
            raw_assign *= assign_coef
            assign_bonus = raw_assign / (normalizer["max_assign"] + 1e-6)

            # 等待惩罚
            raw_wait = 0.0
            for task in self.chain.layers[layer_id].task_queue:
                wait_time = self.current_step - task.arrival_time
                raw_wait += wait_time
            raw_wait *= wait_penalty_coef
            wait_penalty = raw_wait / max_wait

            # 最终组合 reward（用于训练）
            raw_reward = self.beta * raw_util - self.alpha * raw_cost + raw_assign - raw_wait
            reward = self.beta * step_util - self.alpha * step_cost + assign_bonus - wait_penalty

            layer_rewards[layer_id] = {
                "raw_cost": raw_cost,
                "cost": step_cost,
                "utility": step_util,
                "raw_utility": raw_util,
                "raw_assign": raw_assign,
                "assign_bonus": assign_bonus,
                "raw_wait": raw_wait,
                "wait_penalty": wait_penalty,
                "raw_reward": raw_reward,  # 未归一化的 reward，用于日志记录
                "reward": reward  # 归一化之后的 reward，用于 PPO 训练
            }

            total_reward += reward
            total_cost += raw_cost
            total_util += raw_util

        reward_detail = {
            "layer_rewards": layer_rewards,
            "global_summary": {
                "total_cost": total_cost,
                "total_utility": total_util,
                "alpha": self.alpha,
                "beta": self.beta
            }
        }

        return total_reward, reward_detail

    def _get_info(self):
        # 输出详细KPI与episode参数，方便后续分析
        info = {
            "kpi": self.chain.collect_kpis(),
            "alpha": self.alpha,
            "beta": self.beta,
            "current_step": self.current_step
        }
        return info

    def render(self, mode='human'):
        print(f"\n========== Step {self.current_step} ==========")

        # 实时单位时间指标（step_kpis）
        print("Step-wise KPIs:")
        for layer_id, kpi in self.chain.step_kpis.items():
            step_cost = kpi.get("step_cost", 0.0)
            step_util = kpi.get("step_utility", 0.0)
            step_reward = self.beta * step_util - self.alpha * step_cost
            print(f"  Layer {layer_id}: "
                  f"Cost = {step_cost:.2f}, Utility = {step_util:.2f}, Reward = {step_reward:.2f}")

        # 累计指标（cumulative_kpis）
        kpis = self.chain.cumulative_kpis
        print("\nCumulative KPIs:")
        print(f"  Total Time: {kpis['time']}")
        print(f"  Tasks Done: {kpis['tasks_done']}, Total Failures: {kpis['total_failures']}")
        print(f"  Total Cost: {kpis['total_cost']:.2f}, Total Utility: {kpis['total_utility']:.2f}")
        print(f"  Alpha: {self.alpha:.2f}, Beta: {self.beta:.2f}")

        print("\nPer-layer cumulative stats:")
        for layer_id, stats in kpis["per_layer"].items():
            print(f"  Layer {layer_id}: "
                  f"Tasks = {stats['tasks_done']}, Failures = {stats['failures']}, "
                  f"Cost = {stats['cost']:.2f}, Utility = {stats['utility']:.2f}")
