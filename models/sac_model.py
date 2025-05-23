import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2


class RowWiseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):  # x: (B, N, D)
        B, N, D = x.shape
        x = x.view(B * N, D)
        x = self.mlp(x)
        return x.view(B, N, -1)  # (B, N, output_dim)


class Actor(nn.Module):
    def __init__(self, hidden_dim, n_worker, num_pad_tasks):
        super().__init__()
        self.n_worker = n_worker
        self.num_pad_tasks = num_pad_tasks

        self.worker_fc = nn.Linear(hidden_dim, hidden_dim)
        self.task_fc = nn.Linear(hidden_dim, hidden_dim)

        self.fusion_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(1, n_worker, num_pad_tasks))  # broadcastable

    def forward(self, task_embed, worker_embed, valid_mask=None):
        # task_embed: (B, T, D), worker_embed: (B, W, D)
        B, T, D = task_embed.shape
        W = worker_embed.shape[1]

        t_proj = self.task_fc(task_embed).unsqueeze(1).expand(-1, W, -1, -1)   # (B, W, T, D)
        w_proj = self.worker_fc(worker_embed).unsqueeze(2).expand(-1, -1, T, -1)  # (B, W, T, D)

        joint = torch.cat([t_proj, w_proj], dim=-1)  # (B, W, T, 2D)
        mean = self.fusion_fc(joint).squeeze(-1)     # (B, W, T)
        std = torch.exp(self.log_std).expand_as(mean)

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(1).expand_as(mean)
            mean = mean * mask
            std = std * mask

        return mean, std


class Critic(nn.Module):
    def __init__(self, hidden_dim, n_worker, num_pad_tasks):
        super().__init__()
        self.n_worker = n_worker
        self.num_pad_tasks = num_pad_tasks

        input_dim = 2 * hidden_dim + 1  # concat of task + worker + action

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, task_embed, worker_embed, action):
        # task_embed: (B, T, D), worker_embed: (B, W, D), action: (B, W, T)
        B, T, D = task_embed.shape
        W = worker_embed.shape[1]

        t_proj = task_embed.unsqueeze(1).expand(-1, W, -1, -1)   # (B, W, T, D)
        w_proj = worker_embed.unsqueeze(2).expand(-1, -1, T, -1) # (B, W, T, D)
        act = action.unsqueeze(-1)  # (B, W, T, 1)

        x = torch.cat([t_proj, w_proj, act], dim=-1)  # (B, W, T, 2D+1)
        q1 = self.q1(x).squeeze(-1)  # (B, W, T)
        q2 = self.q2(x).squeeze(-1)

        return q1, q2


class SACIndustrialModel(nn.Module):
    def __init__(self, task_input_dim, worker_input_dim, n_worker, num_pad_tasks, hidden_dim=256):
        super().__init__()
        self.n_worker = n_worker
        self.num_pad_tasks = num_pad_tasks

        self.task_encoder = RowWiseEncoder(task_input_dim, hidden_dim, hidden_dim)
        self.worker_encoder = RowWiseEncoder(worker_input_dim, hidden_dim, hidden_dim)

        self.actor = Actor(hidden_dim, n_worker, num_pad_tasks)
        self.critic = Critic(hidden_dim, n_worker, num_pad_tasks)

    def encode(self, task_obs, worker_obs):
        task_feat = self.task_encoder(task_obs)        # (B, T, D)
        worker_feat = self.worker_encoder(worker_obs)  # (B, W, D)
        return task_feat, worker_feat

    def policy(self, task_obs, worker_obs, valid_mask=None):
        task_feat, worker_feat = self.encode(task_obs, worker_obs)
        return self.actor(task_feat, worker_feat, valid_mask)

    def value(self, task_obs, worker_obs, action):
        task_feat, worker_feat = self.encode(task_obs, worker_obs)
        return self.critic(task_feat, worker_feat, action)

    def get_actor_params(self):
        return self.actor.parameters()

    def get_critic_params(self):
        return self.critic.parameters()
