import torch
import torch.nn as nn
from models.ppo_model import RowWiseEncoder


class MuseModel(nn.Module):
    """
    每个子策略拥有独立主干网络（编码器 + FC），互不共享。
    仅 Actor / Value 头按索引区分，与旧实现保持一致。
    """

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

        # 记录负向策略 id
        self.neg_policy = neg_policy
        if neg_policy:
            self.neg_pids = [K - 2, K - 1]

        # ========== 独立主干：编码器 ==========
        # Row-wise Encoders
        self.task_encs = nn.ModuleList(
            [RowWiseEncoder(task_input_dim, D, D) for _ in range(K)]
        )
        self.worker_load_encs = nn.ModuleList(
            [RowWiseEncoder(worker_load_input_dim, D, D) for _ in range(K)]
        )
        self.worker_profile_encs = nn.ModuleList(
            [RowWiseEncoder(worker_profile_input_dim, D, D) for _ in range(K)]
        )

        # ========== 独立主干：FC 处理 ==========
        def fc_worker_block():
            return nn.Sequential(
                nn.Linear(2 * D, D),
                nn.ReLU(),
                nn.LayerNorm(D)
            )

        def fc_task_block():
            return nn.Sequential(
                nn.Linear(D, D),
                nn.ReLU(),
                nn.LayerNorm(D)
            )

        def fc_global_block():
            return nn.Sequential(
                nn.Linear(global_context_dim, D),
                nn.ReLU(),
                nn.Linear(D, D)
            )

        self.fc_workers = nn.ModuleList([fc_worker_block() for _ in range(K)])
        self.fc_tasks = nn.ModuleList([fc_task_block() for _ in range(K)])
        self.global_fcs = nn.ModuleList([fc_global_block() for _ in range(K)])

        # ========== Actor / Value 头 ==========
        self.actor_heads = nn.ModuleList([nn.Linear(3 * D, 1) for _ in range(K)])

        # log_std：K × W × T
        self.log_stds = nn.Parameter(torch.zeros(K, n_worker, num_pad_tasks))
        with torch.no_grad():
            self.log_stds.fill_(-0.5)

        # Value heads
        def mlp_head(in_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, D),
                nn.ReLU(),
                nn.Linear(D, 1)
            )

        self.vu_heads = nn.ModuleList([mlp_head(3 * D) for _ in range(K)])
        self.vc_heads = nn.ModuleList([mlp_head(3 * D) for _ in range(K)])

        # LayerNorm for fused features
        self.fusion_norm = nn.LayerNorm(3 * D)

    # -------------------------------------------------------- #
    def _encode(
        self,
        task_obs: torch.Tensor,          # (B, T, task_dim)
        worker_loads: torch.Tensor,      # (B, W, load_dim)
        worker_profiles: torch.Tensor,   # (B, W, prof_dim)
        global_context: torch.Tensor,    # (B, global_dim)
        pid: int
    ):
        """
        返回：
          t_feat: (B, T, D)
          w_feat: (B, W, D)
          g_feat: (B, D)
        """

        # 每个子策略使用自己的一套编码器 & FC
        t_proj = self.task_encs[pid](task_obs)
        t_feat = self.fc_tasks[pid](t_proj)

        wl_proj = self.worker_load_encs[pid](worker_loads)
        wp_proj = self.worker_profile_encs[pid](worker_profiles)
        w_feat = self.fc_workers[pid](torch.cat([wl_proj, wp_proj], dim=-1))

        g_feat = self.global_fcs[pid](global_context)
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
            task_obs:        (B, T, task_dim)
            worker_loads:    (B, W, load_dim)
            worker_profiles: (B, W, prof_dim)
            global_context:  (B, global_dim)
            valid_mask:      (B, T)
            policy_id:       当前子策略索引

        Returns:
            mean: (B, W, T)
            std:  (B, W, T)
            v_u:  (B,)
            v_c:  (B,)
        """
        pid = int(policy_id)

        # 1) 编码
        t_feat, w_feat, g_feat = self._encode(
            task_obs, worker_loads, worker_profiles, global_context, pid
        )

        # 负向策略可选 detach
        if self.neg_policy and pid in self.neg_pids:
            t_feat = t_feat.detach()
            w_feat = w_feat.detach()
            g_feat = g_feat.detach()

        # 2) 构造融合特征
        w_exp = w_feat.unsqueeze(2).expand(-1, -1, self.num_pad_tasks, -1)
        t_exp = t_feat.unsqueeze(1).expand(-1, self.n_worker, -1, -1)
        g_exp = g_feat.unsqueeze(1).unsqueeze(2).expand(
            -1, self.n_worker, self.num_pad_tasks, -1
        )

        fusion = torch.cat([w_exp, t_exp, g_exp], dim=-1)  # (B, W, T, 3D)
        fusion = self.fusion_norm(fusion)

        # 3) Actor
        raw_mean = self.actor_heads[pid](fusion).squeeze(-1)  # (B, W, T)
        mean = torch.sigmoid(raw_mean)
        log_std = torch.clamp(self.log_stds[pid], min=-4, max=1)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean)

        # 4) Value
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1)  # (B, T, 1)
            t_pool = (t_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            t_pool = t_feat.mean(dim=1)

        w_pool = w_feat.mean(dim=1)
        vc_input = torch.cat([t_pool, w_pool, g_feat], dim=-1)  # (B, 3D)

        v_u = self.vu_heads[pid](vc_input).squeeze(-1)
        v_c = self.vc_heads[pid](vc_input).squeeze(-1)

        return mean, std, v_u, v_c
