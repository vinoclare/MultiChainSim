from multiprocessing import Process
import torch
import numpy as np
import json

from envs import IndustrialChain
from envs.env import MultiplexEnv


class RolloutWorker(Process):
    def __init__(self, conn, env_config_path, schedule_path, worker_config_path):
        super().__init__()
        self.conn = conn
        self.env_config_path = env_config_path
        self.schedule_path = schedule_path
        self.worker_config_path = worker_config_path
        self.daemon = True

    def run(self):
        # === 加载配置并初始化环境 ===
        with open(self.env_config_path, 'r') as f:
            env_config = json.load(f)

        self.max_steps = env_config["max_steps"]
        self.num_layers = env_config["num_layers"]

        self.env = MultiplexEnv(
            self.env_config_path,
            schedule_load_path=self.schedule_path,
            worker_config_load_path=self.worker_config_path
        )
        self.step_count = 0

        while True:
            msg = self.conn.recv()
            if msg["cmd"] == "reset":
                self.obs = self.env.reset(with_new_schedule=False)
                self.step_count = 0
                conn_data = {
                    "obs": self.obs,
                    "reward": {lid: {"reward": 0, "assign_bonus": 0, "wait_penalty": 0,
                                     "cost": 0, "utility": 0} for lid in range(self.num_layers)},
                    "done": False
                }
                self.conn.send(conn_data)
            elif msg["cmd"] == "step":
                actions = msg["action"]  # {layer_id: action_array}
                self.obs, (total_reward, reward_detail), done, info = self.env.step(actions)
                self.step_count += 1

                self.conn.send({
                    "obs": self.obs,  # 多层级 obs: Dict[layer_id] → obs_dict
                    "reward": reward_detail["layer_rewards"],  # Dict[layer_id] → reward components
                    "done": done
                })

            elif msg["cmd"] == "close":
                break
