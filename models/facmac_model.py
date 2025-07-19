import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


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


class FACMACModel(nn.Module):
    """
    Factorized Actor-Critic with Monotonic Mixing (FACMAC)
    - 支持连续动作 (0-1 clamp)
    """

    def __init__(
        self,
        task_dim: int,
        load_dim: int,
        profile_dim: int,
        action_dim: int,
        state_dim: int,
        n_agents: int,
        hidden_dim: int = 64,
        q_hidden_dim: int = 64,
        mixing_hidden_dim: int = 32
    ):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim

        # === shared encoders（保持与 QMIX 一致，便于复用参数） ===
        self.task_enc = nn.Sequential(
            nn.Linear(task_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.load_enc = nn.Sequential(
            nn.Linear(load_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.prof_enc = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        # === actor ===
        obs_dim = 3 * hidden_dim
        self.actor = AgentPolicy(obs_dim, action_dim, hidden_dim)

        # === critic(s) ===
        self.q1 = AgentQNet(obs_dim, action_dim, q_hidden_dim)
        self.q2 = AgentQNet(obs_dim, action_dim, q_hidden_dim)

        # monotonic mixing network（与 QMIX 相同实现）
        self.mixer = MixingNetwork(n_agents, state_dim, mixing_hidden_dim)

    def _encode_agent_obs(self, task_obs, load_obs, prof_obs):
        """
        task_obs : (B, T, task_dim)   (T 条任务，先 mean-pool)
        load_obs : (B, N, load_dim)   (每个 agent 对应一行)
        prof_obs : (B, N, profile_dim)
        return   : (B, N, 3*hidden_dim)  每个 agent 一个 obs 向量
        """
        B, N = load_obs.size(0), load_obs.size(1)

        task_embed = self.task_enc(task_obs.mean(dim=1)).unsqueeze(1).expand(-1, N, -1)
        load_embed = self.load_enc(load_obs)
        prof_embed = self.prof_enc(prof_obs)
        return torch.cat([task_embed, load_embed, prof_embed], dim=-1)

    def get_actions(
        self,
        task_obs,
        load_obs,
        prof_obs,
        deterministic: bool = False,
        noise_std: float = None,
    ):
        """
        返回动作张量 (B, N, action_dim) ，已 clamp 至 [0,1].
        """
        obs = self._encode_agent_obs(task_obs, load_obs, prof_obs)
        B, N, _ = obs.shape
        mean, log_std = self.actor(obs.view(B * N, -1))
        std = torch.exp(log_std)

        if deterministic:
            act = mean
        else:
            dist = Normal(mean, std)
            act = dist.rsample()  # re-parameterized

            if noise_std is not None and noise_std > 0:
                act = act + noise_std * torch.randn_like(act)

        act = torch.clamp(act, 0.0, 1.0)
        return act.view(B, N, -1)

    def forward(self, task_obs, load_obs, prof_obs, actions, state):
        """
        actions : (B, N, action_dim)
        state   : (B, state_dim)
        返回 tuple:
        (q_tot1, q_tot2)
        """
        B, N = load_obs.size(0), load_obs.size(1)
        obs = self._encode_agent_obs(task_obs, load_obs, prof_obs).view(B * N, -1)
        act = actions.view(B * N, -1)

        # Q1
        q1_agent = self.q1(obs, act).view(B, N)         # (B, N)
        q_tot1 = self.mixer(q1_agent, state)            # (B,)

        q2_agent = self.q2(obs, act).view(B, N)
        q_tot2 = self.mixer(q2_agent, state)
        return q_tot1, q_tot2
