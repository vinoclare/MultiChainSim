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
        B, N, D = x.shape
        x = x.view(B * N, D)
        x = self.mlp(x)
        x = self.ln(x)
        return x.view(B, N, -1)


class MAPPONGUModel(nn.Module):
    def __init__(
        self,
        task_input_dim: int,
        worker_load_input_dim: int,
        worker_profile_input_dim: int,
        n_worker: int,
        num_pad_tasks: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.n_worker = n_worker
        self.num_pad_tasks = num_pad_tasks
        D = hidden_dim

        self.eps_greedy = 0.20  # ε：以 20% 概率走“均匀随机”分量
        self.uniform_low = 0.0  # 均匀分量的最小值
        self.uniform_high = 1.0  # 均匀分量的最大值

        # ----- Encoders ----- #
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

        # ----- Actor ----- #
        self.fusion_norm = nn.LayerNorm(2 * D)
        self.actor_head = nn.Linear(2 * D, 1)
        self.log_std = nn.Parameter(torch.zeros(1, n_worker, num_pad_tasks))

        # ----- Two Critic Heads (for extrinsic & intrinsic) ----- #
        self.critic_norm = nn.LayerNorm(3 * D)
        self.shared_critic_ext = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )
        self.shared_critic_int = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

    def encode_features(self, task_obs, worker_loads, worker_profiles):
        t_feat = self.fc_task(self.task_encoder(task_obs))  # (B, T, D)
        wl_feat = self.worker_load_encoder(worker_loads)    # (B, W, D)
        wp_feat = self.worker_profile_encoder(worker_profiles)  # (B, W, D)
        w_feat = self.fc_worker(torch.cat([wl_feat, wp_feat], dim=-1))  # (B, W, D)
        return t_feat, w_feat

    def forward_actor(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        t_feat, w_feat = self.encode_features(task_obs, worker_loads, worker_profiles)

        w_exp = w_feat.unsqueeze(2).expand(-1, -1, self.num_pad_tasks, -1)
        t_exp = t_feat.unsqueeze(1).expand(-1, self.n_worker, -1, -1)
        fusion = torch.cat([w_exp, t_exp], dim=-1)
        fusion = self.fusion_norm(fusion)

        raw_mean = self.actor_head(fusion).squeeze(-1)  # (B, W, T)
        mean = torch.sigmoid(raw_mean)
        log_std = torch.clamp(self.log_std, min=-4, max=1)
        std = torch.exp(log_std).expand_as(mean)

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(1).expand_as(mean)
            mean = mean * mask

        return mean, std

    def forward_critic(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        t_feat, w_feat = self.encode_features(task_obs, worker_loads, worker_profiles)

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1)
            t_pool = (t_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            t_pool = t_feat.mean(dim=1)

        w_pool = w_feat.mean(dim=1)
        x = torch.cat([t_pool, w_pool, t_pool - w_pool], dim=-1)
        x = self.critic_norm(x)
        return x

    def get_ext_value(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        x = self.forward_critic(task_obs, worker_loads, worker_profiles, valid_mask)
        return self.shared_critic_ext(x).squeeze(-1)

    def get_int_value(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        x = self.forward_critic(task_obs, worker_loads, worker_profiles, valid_mask)
        return self.shared_critic_int(x).squeeze(-1)

    def get_value(self, *args, **kwargs):
        return self.get_ext_value(*args, **kwargs)

    def act(self, task_obs, worker_loads, worker_profiles, is_predict=False, valid_mask=None):
        mean, std = self.forward_actor(task_obs, worker_loads, worker_profiles, valid_mask)
        dist = torch.distributions.Normal(mean, std)
        if is_predict:
            action = dist.rsample()
            log_prob = dist.log_prob(action)
            return action, log_prob, mean, std

        # === ε-greedy（按元素混合）：以概率 ε 用均匀分量，否则用 Normal 分量 ===
        # 1) 先各自采样
        eps = float(self.eps_greedy)
        a_norm = dist.rsample()
        a_unif = torch.rand_like(mean) * (self.uniform_high - self.uniform_low) + self.uniform_low

        # 2) 按元素 Bernoulli 掩码
        mask = (torch.rand_like(mean) < eps).float()
        action = mask * a_unif + (1.0 - mask) * a_norm

        # 3) 用“混合分布”的密度计算 log_prob（与返回的 action 一致）
        normal_logp = dist.log_prob(action)  # 元素级 log N(a)
        normal_pdf = normal_logp.exp()  # 元素级 N(a)
        uniform_pdf = torch.full_like(normal_pdf, 1.0 / (self.uniform_high - self.uniform_low))
        mix_pdf = (1.0 - eps) * normal_pdf + eps * uniform_pdf  # 元素级混合密度
        log_prob = torch.log(mix_pdf + 1e-8)  # 元素级 log 概率

        return action, log_prob, mean, std
