# utils/curriculum.py
from collections import deque
from typing import List, Optional


class CurriculumManager:
    """
    Reinforced Adaptive Stair-case Curriculum (RASCL) 管理器——
    这里只管 **单轴 Poisson λ** 难度。
    """
    def __init__(
        self,
        levels: List[float],          # λ 阶梯，例如 [4, 8, 16, 32, 64]
        burn_in: int = 1000,        # 每阶最少训练步
        worst_buf_size: int = 10     # 是否开“最差实例回放”>0 开启
    ):
        self.levels = levels
        self.cur_idx = 0

        self.burn_in = burn_in
        self.steps_on_level = 0
        self.best_return = -float("inf")

        # —— 最差实例缓存 ——
        self.worst_buf_size = worst_buf_size
        self.worst_buffer = deque(maxlen=worst_buf_size)

    # -------- 接口 --------
    @property
    def level(self) -> float:
        """当前 λ"""
        return self.levels[self.cur_idx]

    def update(self, ep_return: float):
        """每个 episode 结束后调用，用 ep_return 更新状态"""
        self.steps_on_level += 1
        self.best_return = max(self.best_return, ep_return)

        # worst-k 回放：回报落入最低 10% 就塞进缓存
        if ep_return < 0.1 * self.best_return:
            self.worst_buffer.append(self.level)

        # 判断是否晋级
        cond_steps = self.steps_on_level >= self.burn_in
        cond_perf = ep_return >= 0.99 * self.best_return
        if cond_steps and cond_perf and self.cur_idx < len(self.levels) - 1:
            self.cur_idx += 1
            self.steps_on_level = 0
            self.best_return = -float("inf")

    def sample_level(self) -> float:
        """
        根据论文的 worst-k 重放策略，80% 用当前阶梯 λ，
        20% 从最差缓存随机抽一个。
        """
        import random
        if self.worst_buffer and random.random() < 0.2:
            return random.choice(list(self.worst_buffer))
        return self.level

    def is_last_level(self) -> bool:
        return self.cur_idx == len(self.levels) - 1
