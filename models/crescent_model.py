import torch
import torch.nn as nn


class RowWiseEncoder(nn.Module):
    """
    和 mappo_model.py 中完全一致的行编码器：
    对 (B, N, D_in) 做逐行 MLP + LayerNorm，输出 (B, N, D_out)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: (B, N, D_in) => (B, N, D_out)
        B, N, D = x.shape
        x = x.view(B * N, D)
        x = self.mlp(x)
        x = self.ln(x)
        return x.view(B, N, -1)


class CrescentIndustrialModel(nn.Module):
    """
    CReSCENT 用的 Actor-Critic 模型：
    - Actor 结构与 MAPPOIndustrialModel 完全一致，保持策略行为不变
    - Critic 部分拆成两个 head：
        * 外在价值 V_ext(s)  —— 和原 MAPPO critic 等价
        * 内在价值 V_int(s)  —— 专门用于 Intrinsic Reward 的价值估计
    """
    def __init__(
        self,
        task_input_dim: int,            # = 4 + n_task_type
        worker_load_input_dim: int,     # = n_task_type + 1
        worker_profile_input_dim: int,  # = 2 * n_task_type
        n_worker: int,
        num_pad_tasks: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.n_worker = n_worker
        self.num_pad_tasks = num_pad_tasks
        D = hidden_dim

        # ----- Encoders（与 MAPPOIndustrialModel 相同） ----- #
        self.task_encoder = RowWiseEncoder(task_input_dim, D, D)
        self.worker_load_encoder = nn.Sequential(
            nn.Linear(worker_load_input_dim, D),
            nn.ReLU(),
            nn.LayerNorm(D)
        )
        self.worker_profile_encoder = nn.Sequential(
            nn.Linear(worker_profile_input_dim, D),
            nn.ReLU(),
            nn.LayerNorm(D)
        )

        self.fc_worker = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.ReLU(),
            nn.LayerNorm(D)
        )
        self.fc_task = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.LayerNorm(D)
        )

        # ----- Actor（与 MAPPOIndustrialModel 相同） ----- #
        self.fusion_norm = nn.LayerNorm(2 * D)
        self.actor_head = nn.Linear(2 * D, 1)
        self.log_std = nn.Parameter(torch.zeros(1, n_worker, num_pad_tasks))

        # ----- Critic 特征构造（与原来逻辑一致） ----- #
        # t_pool, w_pool, t_pool - w_pool 拼成 3D 向量
        critic_input_dim = 3 * D

        # 外在价值 Critic
        self.critic_norm_ext = nn.LayerNorm(critic_input_dim)
        self.ext_critic = nn.Sequential(
            nn.Linear(critic_input_dim, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

        # 内在价值 Critic（结构对称，但参数独立）
        self.critic_norm_int = nn.LayerNorm(critic_input_dim)
        self.int_critic = nn.Sequential(
            nn.Linear(critic_input_dim, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

    # ========= Actor 部分（完全照搬 MAPPOIndustrialModel） =========
    def forward_actor(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        """
        输入:
          - task_obs:      (B, T, D_task)
          - worker_loads:  (B, W, D_load)
          - worker_profiles:(B, W, D_profile)
          - valid_mask:    (B, T)   表示哪些 task slot 有效
        输出:
          - mean, std:     (B, W, T)
        """
        t_feat = self.fc_task(self.task_encoder(task_obs))          # (B, T, D)
        wl_feat = self.worker_load_encoder(worker_loads)            # (B, W, D)
        wp_feat = self.worker_profile_encoder(worker_profiles)      # (B, W, D)
        w_feat = self.fc_worker(torch.cat([wl_feat, wp_feat], dim=-1))  # (B, W, D)

        # 扩展维度拼 worker-task 对
        w_exp = w_feat.unsqueeze(2).expand(-1, -1, self.num_pad_tasks, -1)  # (B, W, T, D)
        t_exp = t_feat.unsqueeze(1).expand(-1, self.n_worker, -1, -1)      # (B, W, T, D)
        fusion = torch.cat([w_exp, t_exp], dim=-1)                          # (B, W, T, 2D)
        fusion = self.fusion_norm(fusion)

        raw_mean = self.actor_head(fusion).squeeze(-1)  # (B, W, T)
        mean = torch.sigmoid(raw_mean)
        log_std = torch.clamp(self.log_std, min=-4, max=1)
        std = torch.exp(log_std).expand_as(mean)

        if valid_mask is not None:
            # valid_mask: (B, T) -> (B, W, T)
            mask = valid_mask.unsqueeze(1).expand_as(mean)
            mean = mean * mask

        return mean, std

    # ========= Critic 特征构造（供外在 / 内在价值共享） =========
    def _build_critic_features(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        """
        复用一套特征，用于外在价值和内在价值两个 head：
        x = concat(t_pool, w_pool, t_pool - w_pool)
        """
        t_feat = self.fc_task(self.task_encoder(task_obs))      # (B, T, D)
        wl_feat = self.worker_load_encoder(worker_loads)        # (B, W, D)
        wp_feat = self.worker_profile_encoder(worker_profiles)  # (B, W, D)
        w_feat = self.fc_worker(torch.cat([wl_feat, wp_feat], dim=-1))  # (B, W, D)

        if valid_mask is not None:
            # valid_mask: (B, T) -> (B, T, 1)
            mask = valid_mask.unsqueeze(-1)                     # (B, T, 1)
            t_pool = (t_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            t_pool = t_feat.mean(dim=1)                         # (B, D)

        w_pool = w_feat.mean(dim=1)                             # (B, D)

        x = torch.cat([t_pool, w_pool, t_pool - w_pool], dim=-1)  # (B, 3D)
        return x

    # ========= 外在价值 Critic =========
    def forward_critic_ext(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        x = self._build_critic_features(task_obs, worker_loads, worker_profiles, valid_mask)
        x = self.critic_norm_ext(x)
        value = self.ext_critic(x).squeeze(-1)  # (B,)
        return value

    # ========= 内在价值 Critic =========
    def forward_critic_int(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        x = self._build_critic_features(task_obs, worker_loads, worker_profiles, valid_mask)
        x = self.critic_norm_int(x)
        value = self.int_critic(x).squeeze(-1)  # (B,)
        return value

    # ========= 外部调用接口 =========
    def act(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        """
        和 MAPPOIndustrialModel 一致：
        返回 action, log_prob, mean, std
        """
        mean, std = self.forward_actor(task_obs, worker_loads, worker_profiles, valid_mask)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, mean, std

    def get_value(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        """
        外在价值（保持接口兼容原 MAPPO）
        """
        return self.forward_critic_ext(task_obs, worker_loads, worker_profiles, valid_mask)

    def get_int_value(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        """
        内在价值，用于后续：
        - 内在 GAE / returns
        - 基于内在优势的跨层 IR 分配
        """
        return self.forward_critic_int(task_obs, worker_loads, worker_profiles, valid_mask)
