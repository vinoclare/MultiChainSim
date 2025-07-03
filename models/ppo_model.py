import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


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
        hidden_dim: int = 256,
        task_encoder: nn.Module = None,
        worker_load_encoder: nn.Module = None,
        worker_profile_encoder: nn.Module = None,
    ):
        super().__init__()
        self.n_worker = n_worker
        self.num_pad_tasks = num_pad_tasks
        D = hidden_dim

        # —— 三路编码器 —— #
        self.task_encoder = task_encoder or RowWiseEncoder(task_input_dim, D, D)
        self.worker_load_encoder = worker_load_encoder or RowWiseEncoder(worker_load_input_dim, D, D)
        self.worker_profile_encoder = worker_profile_encoder or RowWiseEncoder(worker_profile_input_dim, D, D)

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
        # 每个 (worker, task) 对应 3 路特征拼接： worker, task, global
        self.actor_head = nn.Linear(3 * D, 1)
        self.log_std = nn.Parameter(torch.zeros(1, n_worker, num_pad_tasks))
        # self.log_std.fill_(-2)

        # —— Critic —— #
        # 聚合 task_pool, worker_pool, global → value
        self.shared_critic = nn.Sequential(
            nn.Linear(2 * D + D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

        self.fusion_norm = nn.LayerNorm(3 * D)
        self.critic_norm = nn.LayerNorm(3 * D)

        # self.writer = SummaryWriter()
        self.global_step = 0

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
          valid_mask:     (B, num_pad_tasks) 5/1 标志
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
        fusion = self.fusion_norm(fusion)

        raw_mean = self.actor_head(fusion).squeeze(-1)       # (B, W, T)
        mean = torch.sigmoid(raw_mean)
        log_std = torch.clamp(self.log_std, min=-4, max=1)
        std = torch.exp(log_std).expand_as(mean)  # (B, W, T)

        # 应用 valid mask（可选）
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(1).expand_as(mean)  # (B, W, T)
            mean = mean * mask
            mask_pool = valid_mask.unsqueeze(-1)  # (B, T, 1)
            masked_t_feat = t_feat * mask_pool  # (B, T, D)
            sum_feat = masked_t_feat.sum(dim=1)  # (B, D)
            count = mask_pool.sum(dim=1).clamp(min=1)  # (B, 1)
            t_pool = sum_feat / count  # (B, D)
        else:
            t_pool = t_feat.mean(dim=1)  # fallback

        # ——— Critic 估值 ——— #
        w_pool = w_feat.mean(dim=1)   # (B, D)
        cv_in = torch.cat([t_pool, w_pool, g_feat], dim=-1)  # (B, 2D+D)
        cv_in = self.critic_norm(cv_in)
        value = self.shared_critic(cv_in).squeeze(-1)        # (B,)

        # raw_mean = raw_mean * mask
        # if self.global_step % 100 == 5:
        #     self.writer.add_scalar('mean', raw_mean.sum(), global_step=self.global_step)
        #     self.writer.add_scalar('std', std.sum(), global_step=self.global_step)
        # self.global_step += 1

        return mean, std, value
