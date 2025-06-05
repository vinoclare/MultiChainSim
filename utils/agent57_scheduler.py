import math
import torch
from collections import deque


class SoftUCB:
    """
    SoftUCB 调度器：在训练阶段选子策略，分两阶段：
      1. 初始轮询阶段（init_phase=True）：依次让每条子策略连续跑 `min_switch_interval` 个 Episode。
      2. 正式 UCB 阶段（init_phase=False）：基于 avg_returns + UCB 探索分数做 Softmax 采样。
    退出初始轮询后，不再回到初始阶段。
    """

    def __init__(
        self,
        K: int,
        c: float = 1.0,
        min_switch_interval: int = 1
    ):
        """
        Args:
            K: 子策略数量
            c: UCB 探索系数
            min_switch_interval: 每条子策略至少连续跑这么多 Episode 才能切换
        """
        self.K = K
        self.c = c
        self.min_switch_interval = min_switch_interval
        self.window_size = 100

        self.recent_returns = [deque(maxlen=self.window_size) for _ in range(K)]
        self.recent_real_returns = [deque(maxlen=self.window_size) for _ in range(K)]

        self.total_pulls = 0  # update_fake 被调用的总次数

        # 两阶段开关
        self.init_phase = True              # True 表示正在初始轮询阶段
        self.init_pid_index = 0             # 初始轮询阶段当前要跑的策略索引
        self.init_pid_counter = 0           # 该策略已连续跑了多少 Episode

        # 正式阶段用
        self.current_pid = None
        self.episodes_since_switch = 0

    def choose(self) -> int:
        """
        根据当前阶段返回要使用的子策略 pid：
          - 若 init_phase=True，则返回 init_pid_index 并让 init_pid_counter++。
            当 init_pid_counter >= min_switch_interval 时，切换到下一个索引；
            如果所有策略都跑完一遍后，切换 init_phase=False，进入正式阶段。
          - 若 init_phase=False，则检查 episodes_since_switch：
              * 如果尚未跑够 min_switch_interval（episodes_since_switch < min_switch_interval），
                则继续返回 current_pid；
              * 否则，基于 UCB+Softmax 重新选一个 pid，重置 episodes_since_switch = 0。
        """
        # —— 初始轮询阶段：让每条 pid 连续跑 min_switch_interval 次 ——
        if self.init_phase:
            pid = self.init_pid_index
            self.init_pid_counter += 1

            # 如果当前 pid 连续跑够指定 Episode，就切换到下一个 pid
            if self.init_pid_counter >= self.min_switch_interval:
                self.init_pid_index += 1
                self.init_pid_counter = 0

                # 如果 init_pid_index 已经越过最后一个，结束初始阶段
                if self.init_pid_index >= self.K:
                    self.init_phase = False
                    self.init_pid_index = None

                    # 进入正式阶段后，立刻让 current_pid 保持 None，
                    # 下一次 choose() 会触发 UCB 采样
                    self.current_pid = None

            return pid

        # —— 正式阶段 ——
        # 如果 current_pid 为空或已跑够 min_switch_interval，则重新 UCB 采样
        if self.current_pid is None or self.episodes_since_switch >= self.min_switch_interval:
            # UCB 采样：计算每条策略的 UCB 分数
            ucb_scores = []
            for i in range(self.K):
                if len(self.recent_returns[i]) == 0:
                    score = float('inf')
                else:
                    mean_i = sum(self.recent_returns[i]) / len(self.recent_returns[i])
                    n_i = len(self.recent_returns[i])
                    score = mean_i + self.c * math.sqrt(math.log(max(1, self.total_pulls)) / n_i)
                ucb_scores.append(score)

            # 对 UCB 分数做 Softmax，得到概率分布
            probs = torch.softmax(torch.tensor(ucb_scores, dtype=torch.float32), dim=0)
            pid = torch.multinomial(probs, 1).item()

            # 切换到新的 pid
            self.current_pid = pid
            self.episodes_since_switch = 0
        else:
            # 继续使用上一轮的 pid
            pid = self.current_pid

        return pid

    def update(self, pid: int, fake_return: float):
        """
        训练阶段调用：更新“假Return”（β 加权后的回报）的滑动平均和计数。
        """
        self.total_pulls += 1
        self.recent_returns[pid].append(fake_return)

    def update_real(self, pid: int, real_return: float):
        """
        训练阶段也调用：更新“真Return”（环境原生回报）的滑动平均，供评估阶段 greedy 选用。
        """
        self.recent_real_returns[pid].append(real_return)

    def increment_episode_count(self):
        """
        在每个 Episode 结束后调用，用于正式阶段持续跟踪
        current_pid 已连续跑了多少 Episode。
        """
        if not self.init_phase and self.current_pid is not None:
            self.episodes_since_switch += 1
