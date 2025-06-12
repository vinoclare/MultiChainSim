import torch
import torch.nn.functional as F
from collections import deque

import random
from torch.distributions import Normal, kl_divergence

from models.ppo_model import PPOIndustrialModel


class Distiller:
    def __init__(
        self,
        obs_spaces,
        global_context_dim,
        hidden_dim,
        act_dim,
        K,
        loss_type="mse",
        neg_policy=False,
        device="cuda",
        buffer_size=10000,
        sup_coef=1.0,
        neg_coef=1.0,
        margin=0.2
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = PPOIndustrialModel(
            task_input_dim=obs_spaces["task"],
            worker_load_input_dim=obs_spaces["worker_load"],
            worker_profile_input_dim=obs_spaces["worker_profile"],
            n_worker=obs_spaces["n_worker"],
            num_pad_tasks=obs_spaces["num_pad_tasks"],
            global_context_dim=global_context_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.act_dim = act_dim
        self.sup_coef = sup_coef
        self.neg_coef = neg_coef
        self.margin = margin

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.K = K
        self.pos_pid_max = K - 3
        self.loss_type = loss_type  # mse or kl
        self.neg_policy = neg_policy

        self.buffers = {pid: deque(maxlen=buffer_size) for pid in range(K)}

    def collect(self, obs_dict, actions, pid):
        """
        收集一批数据，obs_dict 字典含 task_obs、worker_loads 等张量；
        actions: [B, n_worker, num_pad_tasks]
        pid:     [B]，对应每条轨迹来源的子策略 index
        """
        obs_cpu = {k: v.detach().cpu() for k, v in obs_dict.items()}
        for i in range(actions.shape[0]):
            cur_pid = pid[i]
            self.buffers[cur_pid].append({
                "obs": {k: obs_cpu[k][i] for k in obs_cpu},
                "action": actions[i].detach().cpu()
            })

    def _sample_from_buffers(self, cur_pid: int, batch_size: int):
        buf = self.buffers[cur_pid]
        if len(buf) < batch_size:
            return random.sample(buf, len(buf))  # 不足则全取
        return random.sample(buf, batch_size)

    def bc_update(self, cur_pid: int, batch_size: int = 512, steps: int = 50):
        """对主策略进行若干步蒸馏更新"""
        if len(self.buffers[cur_pid]) < batch_size:
            return 0.0   # 数据不足

        total_loss = 0.0
        for _ in range(steps):
            batch = self._sample_from_buffers(cur_pid, batch_size)
            if len(batch) == 0:
                break

            # ---- 构造张量 ----
            obs_keys = ["task_obs", "worker_loads", "worker_profiles",
                        "global_context", "valid_mask"]
            obs_t = {k: torch.stack([torch.tensor(d["obs"][k])
                                     for d in batch]).to(self.device)
                     for k in obs_keys}
            target_actions = torch.stack([torch.tensor(d["action"])
                                          for d in batch]).to(self.device)

            # ---- 主策略前向 ----
            mean_s, std_s, _ = self.model(**obs_t)

            is_pos_policy = cur_pid <= self.pos_pid_max

            loss = torch.zeros(1, requires_grad=True).to(self.device)
            if is_pos_policy:
                if self.loss_type == "mse":
                    loss_pos = F.mse_loss(mean_s, target_actions)
                    loss = self.sup_coef * loss_pos
                else:  # "kl"
                    # teacher 视为 N(mean_t, σ=0.1) 的固定高斯
                    teacher_std = torch.full_like(mean_s, 0.1)
                    dist_t = Normal(target_actions, teacher_std)
                    dist_s = Normal(mean_s, std_s.clamp(min=1e-3))
                    loss_kl = kl_divergence(dist_t, dist_s).mean()
                    loss = self.sup_coef * loss_kl
            elif self.neg_policy:
                diff = F.relu(self.margin
                              - torch.abs(mean_s - target_actions))
                loss_neg = diff.mean()
                loss = self.neg_coef * loss_neg

            # ---- 反向更新 ----
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, steps)

    @torch.no_grad()
    def predict(self, obs_dict):
        inputs = {k: v.to(self.device) for k, v in obs_dict.items()}
        mean, std, _ = self.model(**inputs)
        dist = Normal(mean, std)
        action = dist.sample().clamp(0, 1)
        return action
