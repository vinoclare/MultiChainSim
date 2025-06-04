import math
import torch


class SoftUCB:
    """
    Soft-UCB 调度器：针对 K 条子策略，根据每条策略的平均回报和访问次数，
    计算 UCB 分数后用 softmax 得到概率分布，再采样出下一个要用的 policy_id。
    """

    def __init__(self, K: int, c: float = 1.0, init_round_robin: bool = True):
        """
        Args:
            K (int): 子策略数量
            c (float): UCB 探索系数
            init_round_robin (bool): 是否在前 K 个 Episode 里先轮流选一遍
        """
        self.K = K
        self.c = c
        self.counts = [0] * K            # 每条策略被选中的次数
        self.avg_returns = [0.0] * K     # 每条策略的滑动平均回报
        self.avg_real_returns = [0.0] * K
        self.total_pulls = 0             # 所有策略被选总次数
        self.init_round_robin = init_round_robin

    def choose(self) -> int:
        """
        选出下一个要使用的 policy_id。
        如果 init_round_robin=True 且存在尚未尝试的策略，优先轮流分配；
        否则，用 Soft-UCB 公式算分后采样。
        """
        # 如果还有子策略没跑过，就先让它们轮流跑一次
        if self.init_round_robin and 0 in self.counts:
            return self.counts.index(0)

        # 所有策略至少被尝试过一次时，按 Soft-UCB 采样
        ucb_scores = []
        for i in range(self.K):
            # 计算 UCB 分数: avg_return + c * sqrt( ln(total) / count[i] )
            score = self.avg_returns[i] + self.c * math.sqrt(
                math.log(max(1, self.total_pulls)) / max(1, self.counts[i])
            )
            ucb_scores.append(score)
        probs = torch.softmax(torch.tensor(ucb_scores, dtype=torch.float32), dim=0)
        # 多项式采样，返回一个索引
        pid = torch.multinomial(probs, 1).item()
        return pid

    def update(self, pid: int, episode_return: float):
        """
        更新 policy_id = pid 的子策略统计信息，递增 count 和 total_pulls，
        并用滑动平均更新 avg_returns[pid]。

        Args:
            pid (int): 本轮 Episode 使用的子策略索引
            episode_return (float): 本轮 Episode 的总回报（组合后的）
        """
        self.total_pulls += 1
        self.counts[pid] += 1
        # 滑动平均更新
        n = self.counts[pid]
        self.avg_returns[pid] += (episode_return - self.avg_returns[pid]) / n

    def update_real(self, pid: int, real_return: float):
        """更新 ‘真’Return 的滑动平均，用于评估阶段 greedy 选。"""
        # 只按 pid 对应的一条 avg_real_returns 做增量式滑动平均
        # 注意：不更新 total_pulls 和 counts，这里只要对 avg_real_returns 做平滑即可
        n = self.counts[pid]  # 也可另设 real_counts[pid]，简单起见沿用 counts[pid]
        self.avg_real_returns[pid] += (real_return - self.avg_real_returns[pid]) / n
