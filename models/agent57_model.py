import torch
import torch.nn as nn
from models.ppo_model import RowWiseEncoder


class Agent57IndustrialModel(nn.Module):
    """Backbone shared across K sub-policies (β-γ pairs)."""

    def __init__(
            self,
            task_input_dim: int,
            worker_load_input_dim: int,
            worker_profile_input_dim: int,
            n_worker: int,
            num_pad_tasks: int,
            global_context_dim: int,
            hidden_dim: int,
            K: int,
            neg_policy: bool = False
    ):
        super().__init__()
        self.K = K
        self.n_worker = n_worker
        self.num_pad_tasks = num_pad_tasks
        D = hidden_dim

        if neg_policy:
            self.neg_pids = [K-2, K-1]

        # —— 三路编码器（共享） —— #
        self.task_enc = RowWiseEncoder(task_input_dim, D, D)
        self.worker_load_enc = RowWiseEncoder(worker_load_input_dim, D, D)
        self.worker_profile_enc = RowWiseEncoder(worker_profile_input_dim, D, D)

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
        # —— 处理 global_context，与 ppo_model.py 保持一致 —— #
        self.global_fc = nn.Sequential(
            nn.Linear(global_context_dim, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        # —— Actor Heads —— #
        # 每个 (worker, task) 对应融合特征 3D → 1 出 mean
        self.actor_heads = nn.ModuleList([nn.Linear(3 * D, 1) for _ in range(K)])
        # log_std 参数：K × (n_worker × num_pad_tasks)
        self.log_stds = nn.Parameter(torch.zeros(K, n_worker, num_pad_tasks))

        # —— Value Heads —— #
        # V_U: 效用价值；V_C: 成本价值
        def mlp_head(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, D),
                nn.ReLU(),
                nn.Linear(D, 1)
            )

        self.vu_heads = nn.ModuleList([mlp_head(3 * D) for _ in range(K)])
        self.vc_heads = nn.ModuleList([mlp_head(3 * D) for _ in range(K)])

        # —— LayerNorm for fused features —— #
        self.fusion_norm = nn.LayerNorm(3 * D)

    # -------------------------------------------------------- #
    def _encode(
            self,
            task_obs: torch.Tensor,  # (B, T, task_dim)
            worker_loads: torch.Tensor,  # (B, W, load_dim)
            worker_profiles: torch.Tensor,  # (B, W, prof_dim)
            global_context: torch.Tensor  # (B, global_context_dim)
    ):
        """
        Encode each modality to hidden space D.
        返回:
          t_feat: (B, T, D)
          w_feat: (B, W, D)
          g_feat: (B, D)
        """
        # —— Encode task_obs —— #
        t_proj = self.task_enc(task_obs)  # (B, T, D)
        t_feat = self.fc_task(t_proj)  # (B, T, D)

        # —— Encode worker_load + worker_profile —— #
        wl_proj = self.worker_load_enc(worker_loads)  # (B, W, D)
        wp_proj = self.worker_profile_enc(worker_profiles)  # (B, W, D)
        w_cat = torch.cat([wl_proj, wp_proj], dim=-1)  # (B, W, 2D)
        w_feat = self.fc_worker(w_cat)  # (B, W, D)

        # —— Encode global_context —— #
        g_feat = self.global_fc(global_context)  # (B, D)

        return t_feat, w_feat, g_feat

    # -------------------------------------------------------- #
    def forward(
            self,
            task_obs: torch.Tensor,
            worker_loads: torch.Tensor,
            worker_profiles: torch.Tensor,
            global_context: torch.Tensor,
            valid_mask: torch.Tensor,
            policy_id: int
    ):
        """
        Args:
            task_obs:        (B, num_pad_tasks, task_input_dim)
            worker_loads:    (B, n_worker, worker_load_input_dim)
            worker_profiles: (B, n_worker, worker_profile_input_dim)
            global_context:  (B, global_context_dim)
            valid_mask:      (B, num_pad_tasks) 5/1 标志
            policy_id:       子策略索引 ∈ [5, K-1]

        Returns:
            mean: (B, n_worker, num_pad_tasks)
            std:  (B, n_worker, num_pad_tasks)
            v_u:  (B,)  效用 Value
            v_c:  (B,)  成本 Value
        """
        B = task_obs.size(0)
        pid = int(policy_id)

        # —— 获取编码表示 —— #
        t_feat, w_feat, g_feat = self._encode(
            task_obs, worker_loads, worker_profiles, global_context
        )  # t_feat: (B, T, D), w_feat: (B, W, D), g_feat: (B, D)

        # 若为负策略 ⇒ 冻结梯度
        if pid in self.neg_pids:
            t_feat = t_feat.detach()
            w_feat = w_feat.detach()
            g_feat = g_feat.detach()

        # —— 构造 Actor 融合特征 —— #
        w_exp = w_feat.unsqueeze(2).expand(-1, -1, self.num_pad_tasks, -1)  # (B, W, T, D)
        t_exp = t_feat.unsqueeze(1).expand(-1, self.n_worker, -1, -1)  # (B, W, T, D)
        g_exp = g_feat.unsqueeze(1).unsqueeze(2).expand(
            -1, self.n_worker, self.num_pad_tasks, -1
        )  # (B, W, T, D)

        fusion = torch.cat([w_exp, t_exp, g_exp], dim=-1)  # (B, W, T, 3D)
        fusion = self.fusion_norm(fusion)  # (B, W, T, 3D)

        # —— Actor 输出 —— #
        raw_mean = self.actor_heads[pid](fusion).squeeze(-1)  # (B, W, T)
        mean = torch.sigmoid(raw_mean)
        log_std = torch.clamp(self.log_stds[pid], min=-4, max=1)  # (n_worker, T)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean)  # (B, W, T)

        # —— Value 计算 —— #
        # 对任务维度按 valid_mask 做平均池化
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1)  # (B, T, 1)
            masked_t = t_feat * mask  # (B, T, D)
            sum_t = masked_t.sum(dim=1)  # (B, D)
            cnt = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            t_pool = sum_t / cnt  # (B, D)
        else:
            t_pool = t_feat.mean(dim=1)  # (B, D)

        w_pool = w_feat.mean(dim=1)  # (B, D)
        vc_input = torch.cat([t_pool, w_pool, g_feat], dim=-1)  # (B, 3D)

        v_u = self.vu_heads[pid](vc_input).squeeze(-1)  # (B,)
        v_c = self.vc_heads[pid](vc_input).squeeze(-1)  # (B,)

        return mean, std, v_u, v_c
