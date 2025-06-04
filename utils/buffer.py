import torch
import numpy as np


class RolloutBuffer:
    """
    用于单进程版 Agent-57 的经验缓存。
    每个 Episode 结束时，将序列中的数据整体取出用于 GAE 和 PPO 更新。
    """

    def __init__(self):
        self.clear()

    def clear(self):
        """清空缓存中的列表。"""
        self.obs = []             # List of (task_obs, worker_loads, worker_profiles, global_context, valid_mask)
        self.actions = []         # List of action np arrays (W x T)
        self.values_u = []        # List of v_u (float)
        self.values_c = []        # List of v_c (float)
        self.logps = []           # List of log_prob (float)
        self.rewards_u = []       # List of normalized utility (float)
        self.rewards_c = []       # List of normalized cost (float)
        self.combined_rewards = []# List of combined reward (float)
        self.dones = []           # List of done flags (bool)

    def store(self,
              task_obs,
              worker_loads,
              worker_profiles,
              global_context,
              valid_mask,
              action,
              v_u,
              v_c,
              logp,
              u_n,
              c_n,
              r_comb,
              done):
        """
        将当前 step 的数据以原生 Python 类型或 NumPy 存入列表。
        注意：在主循环里，action、obs 等可以直接以 NumPy 形式传入，
              v_u/v_c/logp 通常是 float，u_n/c_n/r_comb 是 float。
        """
        self.obs.append((task_obs, worker_loads, worker_profiles, global_context, valid_mask))
        self.actions.append(action)
        self.values_u.append(v_u)
        self.values_c.append(v_c)
        self.logps.append(logp)
        self.rewards_u.append(u_n)
        self.rewards_c.append(c_n)
        self.combined_rewards.append(r_comb)
        self.dones.append(done)

    def to_tensors(self, device="cpu"):
        """
        把列表中所有数据转换为 PyTorch 张量。返回一个字典，
        key 包含：
          "obs": tuple of five tensors (task_obs, worker_loads, worker_profiles, global_context, valid_mask)
          "actions": Tensor (L, W, T)
          "values_u": Tensor (L,)
          "values_c": Tensor (L,)
          "logps":    Tensor (L,)
          "rewards_u":Tensor (L,)
          "rewards_c":Tensor (L,)
          "combined": Tensor (L,)
          "dones":    Tensor (L,)
        其中 L = 序列长度；W = n_worker；T = num_pad_tasks
        """
        # 将 obs 中的五个部分分别堆成张量
        task_obs_list, worker_loads_list, worker_profiles_list, global_context_list, valid_mask_list = zip(*self.obs)

        task_obs_tensor = torch.tensor(np.stack(task_obs_list, axis=0), dtype=torch.float32, device=device)
        worker_loads_tensor = torch.tensor(np.stack(worker_loads_list, axis=0), dtype=torch.float32, device=device)
        worker_profiles_tensor = torch.tensor(np.stack(worker_profiles_list, axis=0), dtype=torch.float32, device=device)
        global_context_tensor = torch.tensor(np.stack(global_context_list, axis=0), dtype=torch.float32, device=device)
        valid_mask_tensor = torch.tensor(np.stack(valid_mask_list, axis=0), dtype=torch.float32, device=device)

        actions_tensor = torch.tensor(np.stack(self.actions, axis=0), dtype=torch.float32, device=device)  # (L, W, T)
        values_u_tensor = torch.tensor(self.values_u, dtype=torch.float32, device=device)                  # (L,)
        values_c_tensor = torch.tensor(self.values_c, dtype=torch.float32, device=device)                  # (L,)
        logps_tensor = torch.tensor(self.logps, dtype=torch.float32, device=device)                       # (L,)
        rewards_u_tensor = torch.tensor(self.rewards_u, dtype=torch.float32, device=device)                # (L,)
        rewards_c_tensor = torch.tensor(self.rewards_c, dtype=torch.float32, device=device)                # (L,)
        combined_tensor = torch.tensor(self.combined_rewards, dtype=torch.float32, device=device)          # (L,)
        dones_tensor = torch.tensor(self.dones, dtype=torch.float32, device=device)                        # (L,)

        return {
            "task_obs": task_obs_tensor,
            "worker_loads": worker_loads_tensor,
            "worker_profiles": worker_profiles_tensor,
            "global_context": global_context_tensor,
            "valid_mask": valid_mask_tensor,
            "actions": actions_tensor,
            "values_u": values_u_tensor,
            "values_c": values_c_tensor,
            "logps": logps_tensor,
            "rewards_u": rewards_u_tensor,
            "rewards_c": rewards_c_tensor,
            "combined": combined_tensor,
            "dones": dones_tensor
        }


def compute_gae(rewards, values, gamma, lam):
    """
    计算 GAE（Generalized Advantage Estimation）并返回 (returns, advantages)。
    Args:
        rewards: list 或 1D Tensor, 长度 L，已归一化的 reward 序列
        values:  list 或 1D Tensor, 长度 L，交由模型前向时得到的价值估计
        gamma:   折扣因子
        lam:     GAE 的 λ
    Returns:
        returns:   1D numpy list 或 Tensor, 长度 L
        advantages:1D numpy list 或 Tensor, 长度 L
    """
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.cpu().numpy()
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()

    L = len(rewards)
    advantages = np.zeros(L, dtype=np.float32)
    lastgaelam = 0.0
    # 注意：假设 values 已经包含对下一状态 value 的 bootstrap
    for t in reversed(range(L)):
        if t == L - 1:
            next_nonterminal = 1.0 - 1.0  # 末尾假设 next_value = 0
            next_values = 0.0
        else:
            next_nonterminal = 1.0 - float(0)  # 如果 done，那么用 0，否则 1；单进程版本可简化为 always 1
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam

    returns = advantages + values
    return returns, advantages
