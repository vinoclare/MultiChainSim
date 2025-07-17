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
        # x: (B, N, D_in) => (B, N, D_out)
        B, N, D = x.shape
        x = x.view(B * N, D)
        x = self.mlp(x)
        x = self.ln(x)
        return x.view(B, N, -1)


class MAPPOIndustrialModel(nn.Module):
    def __init__(
        self,
        task_input_dim: int,           # = 4 + n_task_type
        worker_load_input_dim: int,   # = n_task_type + 1
        worker_profile_input_dim: int,  # = 2 * n_task_type
        n_worker: int,
        num_pad_tasks: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.n_worker = n_worker
        self.num_pad_tasks = num_pad_tasks
        D = hidden_dim

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

        # ----- Centralised Critic ----- #
        self.critic_norm = nn.LayerNorm(3 * D)
        self.shared_critic = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

    def forward_actor(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        t_feat = self.fc_task(self.task_encoder(task_obs))  # (B, T, D)
        wl_feat = self.worker_load_encoder(worker_loads)    # (B, W, D)
        wp_feat = self.worker_profile_encoder(worker_profiles)  # (B, W, D)
        w_feat = self.fc_worker(torch.cat([wl_feat, wp_feat], dim=-1))  # (B, W, D)

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
        t_feat = self.fc_task(self.task_encoder(task_obs))  # (B, T, D)
        wl_feat = self.worker_load_encoder(worker_loads)
        wp_feat = self.worker_profile_encoder(worker_profiles)
        w_feat = self.fc_worker(torch.cat([wl_feat, wp_feat], dim=-1))

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1)
            t_pool = (t_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            t_pool = t_feat.mean(dim=1)

        w_pool = w_feat.mean(dim=1)
        x = torch.cat([t_pool, w_pool, t_pool - w_pool], dim=-1)
        x = self.critic_norm(x)
        value = self.shared_critic(x).squeeze(-1)
        return value

    def act(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        mean, std = self.forward_actor(task_obs, worker_loads, worker_profiles, valid_mask)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, mean, std

    def get_value(self, task_obs, worker_loads, worker_profiles, valid_mask=None):
        return self.forward_critic(task_obs, worker_loads, worker_profiles, valid_mask)
