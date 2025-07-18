import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentQNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_i, act_i):  # (B, obs_dim), (B, act_dim)
        x = torch.cat([obs_i, act_i], dim=-1)
        return self.net(x)  # (B, 1)


class AgentPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_i):  # (B * n_agents, obs_dim)
        x = self.fc(obs_i)
        mean = torch.sigmoid(self.mean_head(x))
        log_std = self.log_std_head(x).clamp(-4, 1)
        std = torch.exp(log_std)
        return mean, std


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim):
        super().__init__()
        self.hyper_w1 = nn.Linear(state_dim, hidden_dim * n_agents)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.n_agents = n_agents

    def forward(self, agent_qs, state):  # agent_qs: (B, n_agents), state: (B, state_dim)
        B = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, -1)  # (B, n_agents, hidden)
        b1 = self.hyper_b1(state).view(B, 1, -1)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # (B, 1, hidden)
        w2 = torch.abs(self.hyper_w2(state)).view(B, -1, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        return q_total.view(B)  # (B,)


class QMixModel(nn.Module):
    def __init__(self, task_input_dim, load_input_dim, profile_input_dim,
                 action_dim, state_dim, n_agents, hidden_dim=64, q_hidden_dim=64, mixing_hidden_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim

        self.task_encoder = nn.Sequential(
            nn.Linear(task_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.load_encoder = nn.Sequential(
            nn.Linear(load_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.profile_encoder = nn.Sequential(
            nn.Linear(profile_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.agent_q = AgentQNet(obs_dim=3 * hidden_dim, action_dim=action_dim, hidden_dim=q_hidden_dim)
        self.agent_policy = AgentPolicy(obs_dim=3 * hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.mixing_net = MixingNetwork(n_agents, state_dim, mixing_hidden_dim)

    def get_actions(self, task_obs, load_obs, profile_obs):
        """
        返回动作分配比例 ∈ [0, 1]（每个 agent 一个分配向量）
        """
        B, n_agents = load_obs.shape[:2]
        tq_feat = self.task_encoder(task_obs.mean(dim=1))
        tq_feat = tq_feat.unsqueeze(1).expand(-1, n_agents, -1)
        load_feat = self.load_encoder(load_obs)
        prof_feat = self.profile_encoder(profile_obs)

        obs_feat = torch.cat([tq_feat, load_feat, prof_feat], dim=-1)  # (B, n_agents, 3D)
        obs_flat = obs_feat.view(B * n_agents, -1)

        mean, std = self.agent_policy(obs_flat)  # (B * n_agents, action_dim)

        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()  # reparameterization

        action = torch.clamp(action, 0.0, 1.0)
        return action.view(B, n_agents, -1)

    def forward(self, task_obs, load_obs, profile_obs, actions, state):
        """
        task_obs:     (B, num_pad_tasks, task_input_dim)
        load_obs:     (B, n_agents, load_input_dim)
        profile_obs:  (B, n_agents, profile_input_dim)
        actions:      (B, n_agents, action_dim)
        state:        (B, state_dim)
        """
        B, n_agents = load_obs.shape[:2]
        assert n_agents == self.n_agents

        # Encode task queue → mean pool → (B, D)
        tq_feat = self.task_encoder(task_obs.mean(dim=1))  # (B, D)
        tq_feat = tq_feat.unsqueeze(1).expand(-1, n_agents, -1)  # (B, n_agents, D)

        load_feat = self.load_encoder(load_obs)            # (B, n_agents, D)
        prof_feat = self.profile_encoder(profile_obs)      # (B, n_agents, D)

        obs_feat = torch.cat([tq_feat, load_feat, prof_feat], dim=-1)  # (B, n_agents, 3D)
        obs_flat = obs_feat.view(B * n_agents, -1)
        act_flat = actions.view(B * n_agents, -1)

        agent_q_flat = self.agent_q(obs_flat, act_flat)  # (B * n_agents, 1)
        agent_qs = agent_q_flat.view(B, n_agents)        # (B, n_agents)

        q_total = self.mixing_net(agent_qs, state)       # (B,)
        return q_total, agent_qs
