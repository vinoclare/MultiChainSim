import torch
import torch.nn as nn


class RowWiseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: (B, N, D_in) → (B, N, D_out)
        B, N, D = x.shape
        x = x.view(B * N, D)
        x = self.mlp(x)
        x = self.ln(x)
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
        self.fc_worker = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.ReLU(),
            nn.LayerNorm(D)
        )

        # 处理 task 编码
        self.fc_task = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.LayerNorm(D)
        )

        # 处理全局上下文
        self.global_fc = nn.Sequential(
            nn.Linear(global_context_dim, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        # —— Actor —— #
        self.w_actor = nn.Linear(D, 1, bias=False)
        self.t_actor = nn.Linear(D, 1, bias=False)
        self.g_actor = nn.Linear(D, 1, bias=True)

        self.w_std = nn.Linear(D, 1, bias=False)
        self.t_std = nn.Linear(D, 1, bias=False)
        self.g_std = nn.Linear(D, 1, bias=True)

        # —— Critic —— #
        # 聚合 task_pool, worker_pool, global → value
        self.shared_critic = nn.Sequential(
            nn.Linear(2 * D + D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

        self.fusion_norm = nn.LayerNorm(3 * D)
        self.critic_norm = nn.LayerNorm(3 * D)

        self.worker_pool_query = nn.Parameter(torch.randn(hidden_dim))
        self.task_pool_query = nn.Parameter(torch.randn(hidden_dim))

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

        # ——— Actor ——— #
        w_score = self.w_actor(w_feat).unsqueeze(2)
        t_score = self.t_actor(t_feat).unsqueeze(1)
        g_score = self.g_actor(g_feat).unsqueeze(1).unsqueeze(2)
        mean = w_score + t_score + g_score
        mean = mean.squeeze(-1)

        w_std = self.w_std(w_feat).unsqueeze(2)
        t_std = self.t_std(t_feat).unsqueeze(1)
        g_std = self.g_std(g_feat).unsqueeze(1).unsqueeze(2)
        log_std = w_std + t_std + g_std  # (B, W, T)
        log_std = log_std.clamp(-1.2, 2)
        std = torch.exp(log_std)
        std = std.squeeze(-1)

        # 应用 valid mask（可选）
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(1).expand_as(mean)  # (B, W, T)
            mean = mean * mask
            std = std * mask + (1 - mask) * 1e-6

            attn_logits = (t_feat * self.task_pool_query).sum(-1)  # (B, T)
            attn_logits = attn_logits + (1 - valid_mask) * -1e9
            attn_weight = torch.softmax(attn_logits, dim=1)
        else:
            attn_logits = (t_feat * self.task_pool_query).sum(-1)
            attn_weight = torch.softmax(attn_logits, dim=1)  # (B, T)
        t_pool = (attn_weight.unsqueeze(-1) * t_feat).sum(dim=1)  # (B, D)

        # ——— Critic 估值 ——— #
        worker_attn_logits = (w_feat * self.worker_pool_query).sum(-1)  # (B, W)
        worker_attn_weight = torch.softmax(worker_attn_logits, dim=1)  # (B, W)
        w_pool = (worker_attn_weight.unsqueeze(-1) * w_feat).sum(dim=1)  # (B, D)
        cv_in = torch.cat([t_pool, w_pool, g_feat], dim=-1)  # (B, 2D+D)
        cv_in = self.critic_norm(cv_in)
        value = self.shared_critic(cv_in).squeeze(-1)        # (B,)

        return mean, std, value
