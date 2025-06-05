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


def compute_gae(rewards, dones, values, gamma, lam):
    """
    计算带 dones 信息的 GAE，并返回 (returns, advantages)。

    Args:
        rewards (list or np.ndarray of float): 形状 (T,) 的即时 reward 序列（可正可负）。
        dones   (list or np.ndarray of int):   形状 (T,) 的终止标志序列，done[t] = 1 表示第 t 步结束后进入终止状态。
        values  (list or np.ndarray of float): 形状 (T,) 的 value 函数预测值 V(s_t)。
        gamma   (float): 折扣因子 γ。
        lam     (float):  GAE 参数 λ。

    Returns:
        returns    (list of float): 长度 T 的蒙特卡洛回报（return），等于 advantage + values[t]。
        advantages (list of float): 长度 T 的 Advantage 值，使用 GAE 公式计算。
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0

    # 在末尾 append 一个 0，表示终止后 V(s_{T}) = 0
    # 注意要先把 values 转为列表再拼接，否则 numpy 会报维度错误
    values = list(values) + [0.0]

    # 从后向前递推
    for t in reversed(range(T)):
        # 如果 dones[t] == 1，表示第 t 步结束后环境终止，则 next_nonterminal = 0，让上一个序列与未来隔离
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae

    # 计算 return = advantage + value
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return returns, advantages
