import torch
import torch.nn as nn
import torch.nn.functional as F


class RowWiseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, N, D_in) → (B, N, D_out)
        B, N, D = x.shape
        x = x.view(B * N, D)
        x = self.mlp(x)
        return x.view(B, N, -1)


class PPOIndustrialModel(nn.Module):
    def __init__(
        self,
        task_input_dim: int,
        worker_load_input_dim: int,
        worker_profile_input_dim: int,
        n_worker: int,
        num_pad_tasks: int,
        global_context_dim: int = 1,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.n_worker = n_worker
        self.num_pad_tasks = num_pad_tasks
        D = hidden_dim

        # —— 三路编码器 —— #
        self.task_encoder = RowWiseEncoder(task_input_dim, D, D)
        self.worker_load_encoder = RowWiseEncoder(worker_load_input_dim, D, D)
        self.worker_profile_encoder = RowWiseEncoder(worker_profile_input_dim, D, D)

        # 融合 worker_load + worker_profile → D→D
        self.fc_worker = nn.Linear(2 * D, D)

        # 处理 task 编码
        self.fc_task = nn.Linear(D, D)

        # 处理全局上下文
        self.global_fc = nn.Sequential(
            nn.Linear(global_context_dim, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        # —— Actor —— #
        # 每个 (worker, task) 对应 3 路特征拼接： worker, task, global
        self.actor_head = nn.Linear(3 * D, 1)
        self.std_head = nn.Linear(3 * hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1, n_worker, num_pad_tasks))

        # —— Critic —— #
        # 聚合 task_pool, worker_pool, global → value
        self.shared_critic = nn.Sequential(
            nn.Linear(2 * D + D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

    def forward(
        self,
        task_obs: torch.Tensor,
        worker_loads: torch.Tensor,
        worker_profiles: torch.Tensor,
        global_context: torch.Tensor,
        valid_mask: torch.Tensor = None
    ):
        """
        Inputs:
          task_obs:       (B, num_pad_tasks, task_input_dim)
          worker_loads:   (B, n_worker, worker_load_input_dim)
          worker_profiles:(B, n_worker, worker_profile_input_dim)
          global_context: (B, global_context_dim)
          valid_mask:     (B, num_pad_tasks) 0/1 标志
        Outputs:
          mean: (B, n_worker, num_pad_tasks)
          std:  (B, n_worker, num_pad_tasks)
          value:(B,)
        """

        B = task_obs.size(0)

        # ——— 编码 task ——— #
        t_feat = self.task_encoder(task_obs)    # (B, T, D)
        t_feat = self.fc_task(t_feat)           # (B, T, D)

        # ——— 编码 worker_load & profile ——— #
        wl_feat = self.worker_load_encoder(worker_loads)      # (B, W, D)
        wp_feat = self.worker_profile_encoder(worker_profiles)  # (B, W, D)
        w_cat = torch.cat([wl_feat, wp_feat], dim=-1)         # (B, W, 2D)
        w_feat = self.fc_worker(w_cat)                         # (B, W, D)

        # ——— 编码 global_context ——— #
        g_feat = self.global_fc(global_context)  # (B, D)

        # ——— 构造 Actor 融合特征 ——— #
        # 扩展到 (B, W, T, D)
        w_exp = w_feat.unsqueeze(2).expand(-1, -1, self.num_pad_tasks, -1)
        t_exp = t_feat.unsqueeze(1).expand(-1, self.n_worker, -1, -1)
        g_exp = g_feat.unsqueeze(1).unsqueeze(2).expand(-1, self.n_worker, self.num_pad_tasks, -1)

        fusion = torch.cat([w_exp, t_exp, g_exp], dim=-1)  # (B, W, T, 3D)
        mean = self.actor_head(fusion).squeeze(-1)       # (B, W, T)
        mean = torch.sigmoid(mean)
        raw_std = self.std_head(fusion).squeeze(-1)      # (B, W, T)
        std = F.softplus(raw_std) * 0.5 + 1e-3
        std = torch.clamp(std, min=1e-6, max=2.0)

        # 应用 valid mask（可选）
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(1).expand_as(mean)  # (B, W, T)
            mean = mean * mask

        # ——— Critic 估值 ——— #
        t_pool = t_feat.mean(dim=1)   # (B, D)
        w_pool = w_feat.mean(dim=1)   # (B, D)
        cv_in = torch.cat([t_pool, w_pool, g_feat], dim=-1)  # (B, 2D+D)
        value = self.shared_critic(cv_in).squeeze(-1)        # (B,)

        return mean, std, value
