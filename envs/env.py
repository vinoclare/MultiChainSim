import gym
from gym import spaces
import numpy as np
from typing import Dict, Any
from .core_chain import IndustrialChain
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
        self.max_tasks = 100  # 最大任务队列长度

        self.observation_space = spaces.Dict({
            layer_id: spaces.Dict({
                "task_queue": spaces.Box(low=0, high=1, shape=(self.max_tasks, 3), dtype=np.float32),
                "worker_loads": spaces.Box(low=0, high=1, shape=(len(self.worker_config[layer_id]),), dtype=np.float32)
            }) for layer_id in range(self.num_layers)
        })

        self.action_space = spaces.Dict({
            layer_id: spaces.Box(low=0, high=10, shape=(len(self.worker_config[layer_id]), self.max_tasks), dtype=np.int32)
            for layer_id in range(self.num_layers)
        })

    def reset(self, with_new_schedule=False):
        if with_new_schedule:
            self.task_schedule = generate_task_schedule(self.config)
            self.alpha = np.random.uniform(0.5, 1.5)
            self.beta = np.random.uniform(0.5, 1.5)
        self.chain = IndustrialChain(self.worker_config)
        self.current_step = 0
        return self._get_obs()

    def step(self, action_dict: Dict[int, Any]):
        # 任务注入
        new_tasks = self.task_schedule.get(self.current_step, [])
        self.chain.insert_tasks(new_tasks)
        # 动作检查与裁剪（防止越界、非法值）
        checked_action_dict = self._sanitize_action(action_dict)
        self.chain.apply_action(checked_action_dict)
        self.chain.step()
        self.current_step += 1
        obs = self._get_obs()
        reward = self._get_reward()
        done = self.current_step >= self.max_steps
        info = self._get_info()
        return obs, reward, done, info

    def _sanitize_action(self, action_dict: Dict[int, Any]) -> Dict[int, Any]:
        # 动作安全检查，防止越界和非法输入
        checked = {}
        for layer_id, matrix in action_dict.items():
            n_worker = len(self.worker_config[layer_id])
            # 裁剪到预期形状
            mat = np.array(matrix)
            mat = mat[:n_worker, :self.max_tasks]
            mat = np.clip(mat, 0, 10)  # 可自定义最大分配量
            checked[layer_id] = mat.tolist()
        return checked

    def _get_obs(self):
        obs = {}
        for layer_id, layer in enumerate(self.chain.layers):
            task_features = np.zeros((self.max_tasks, 3), dtype=np.float32)
            for i, task in enumerate(layer.task_queue[:self.max_tasks]):
                type_vec = [0.0, 0.0, 0.0]
                # one-hot 任务类型示例
                if task.task_type == "A":
                    type_vec[0] = 1.0
                elif task.task_type == "B":
                    type_vec[1] = 1.0
                elif task.task_type == "C":
                    type_vec[2] = 1.0
                task_features[i] = np.array([
                    task.amount / 10.0,
                    (self.current_step - task.arrival_time) / self.max_steps,
                    type_vec[np.argmax(type_vec)]  # 可替换为 full one-hot
                ])
            worker_loads = np.array([
                w.total_current_load / max(w.max_total_load, 1)
                for w in layer.workers
            ], dtype=np.float32)
            obs[layer_id] = {
                "task_queue": task_features,
                "worker_loads": worker_loads
            }
        return obs

    def _get_reward(self):
        kpi = self.chain.collect_kpis()
        # 结构化reward返回
        reward_detail = {
            "layer_rewards": {},
            "global_summary": {
                "total_cost": kpi["avg_cost"] * kpi["tasks_done"],
                "avg_util": kpi["avg_utility"],
                "total_failures": kpi.get("total_failures", 0)
            }
        }
        # 每层可按需扩展
        for layer_id, layer in enumerate(self.chain.layers):
            layer_kpi = layer.get_kpi_snapshot()
            reward_detail["layer_rewards"][layer_id] = {
                "cost": layer_kpi["avg_cost"],
                "utility": layer_kpi["avg_util"],
                "utilization": layer_kpi["utilization"]
            }
        # 兼容RL接口返回单值reward
        reward_scalar = self.beta * kpi["avg_utility"] - self.alpha * kpi["avg_cost"]
        return reward_scalar, reward_detail

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
        print(f"Step {self.current_step}: {self.chain.collect_kpis()}")
