import torch
import torch.nn as nn
from torch.distributions import Normal


class VariBADIndustrialModel(nn.Module):
    """
    对齐 run_varibad.py 当前接口的 VariBAD 模型：
    - raw obs (flatten 后的向量) -> obs embedding
    - belief encoder (GRUCell) 输入: [obs_embed, prev_action, prev_reward, prev_done]
    - policy/value 条件在 (obs_embed, z)
    - decoder 用于 ELBO: (obs_embed, action, z) -> next_obs_raw, reward
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_embed_dim: int = 128,
        belief_hidden: int = 128,
        z_dim: int = 64,
        policy_hidden: int = 256,
        value_hidden: int = 256,
        decoder_hidden: int = 256,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_embed_dim = int(obs_embed_dim)
        self.belief_hidden = int(belief_hidden)
        self.z_dim = int(z_dim)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        # -------------------- Obs encoder: raw_obs -> s_embed --------------------
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.obs_embed_dim),
            nn.ReLU(),
        )

        # -------------------- Belief encoder (GRUCell) --------------------
        # 输入: s_embed + prev_action + prev_reward + prev_done
        self.belief_rnn = nn.GRUCell(
            input_size=self.obs_embed_dim + self.action_dim + 1 + 1,
            hidden_size=self.belief_hidden,
        )
        self.mu_head = nn.Linear(self.belief_hidden, self.z_dim)
        self.logvar_head = nn.Linear(self.belief_hidden, self.z_dim)

        # -------------------- Policy head: (s_embed, z) -> action mean --------------------
        self.policy_fc = nn.Sequential(
            nn.Linear(self.obs_embed_dim + self.z_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ReLU(),
        )
        self.policy_mean = nn.Linear(policy_hidden, self.action_dim)
        self.policy_logstd = nn.Parameter(torch.zeros(self.action_dim))

        # -------------------- Value head: (s_embed, z) -> V --------------------
        self.value_fc = nn.Sequential(
            nn.Linear(self.obs_embed_dim + self.z_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

        # -------------------- Decoder for ELBO: (s_embed, action, z) -> next_obs_raw, reward --------------------
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.obs_embed_dim + self.action_dim + self.z_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
        )
        self.recon_next_obs = nn.Linear(decoder_hidden, self.obs_dim)
        self.recon_reward = nn.Linear(decoder_hidden, 1)

    # -------------------- Core functions --------------------

    def encode_obs(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """
        raw_obs: [B, obs_dim]
        return:  [B, obs_embed_dim]
        """
        return self.obs_encoder(raw_obs)

    def belief_step(
        self,
        s_embed: torch.Tensor,
        a_prev: torch.Tensor,
        r_prev: torch.Tensor,
        done_prev: torch.Tensor,
        h_prev: torch.Tensor,
    ):
        """
        用上一步 (a_prev, r_prev, done_prev) + 当前 s_embed 更新 belief
        输入:
          s_embed:   [B, obs_embed_dim]
          a_prev:    [B, action_dim]
          r_prev:    [B, 1] or [B]
          done_prev: [B, 1] or [B]
          h_prev:    [B, belief_hidden]
        输出:
          h_new, mu, logvar, z
        """
        if r_prev.dim() == 1:
            r_prev = r_prev.unsqueeze(-1)
        if done_prev.dim() == 1:
            done_prev = done_prev.unsqueeze(-1)

        x = torch.cat([s_embed, a_prev, r_prev, done_prev], dim=-1)
        h_new = self.belief_rnn(x, h_prev)

        mu = self.mu_head(h_new)
        logvar = self.logvar_head(h_new)

        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return h_new, mu, logvar, z

    def forward_actor(self, s_embed: torch.Tensor, z: torch.Tensor):
        """
        输出 mean/std（用于构造 Normal 分布）
        """
        x = torch.cat([s_embed, z], dim=-1)
        h = self.policy_fc(x)
        mean = self.policy_mean(h)

        logstd = self.policy_logstd.clamp(self.min_log_std, self.max_log_std)
        logstd = logstd.view(1, -1).expand_as(mean)
        std = torch.exp(logstd)
        return mean, std

    def act(self, s_embed: torch.Tensor, z: torch.Tensor):
        """
        采样动作，返回 action, logp, entropy
        - action:  [B, action_dim]
        - logp:    [B, 1]
        - entropy: [B, 1]
        """
        mean, std = self.forward_actor(s_embed, z)
        dist = Normal(mean, std)
        action = dist.rsample()
        logp = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, logp, entropy

    def get_value(self, s_embed: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        value: [B, 1]
        """
        x = torch.cat([s_embed, z], dim=-1)
        return self.value_fc(x)

    def decode(self, s_embed: torch.Tensor, action: torch.Tensor, z: torch.Tensor):
        """
        decoder 用于 ELBO 重构:
        输入:
          s_embed: [B, obs_embed_dim]
          action:  [B, action_dim]
          z:       [B, z_dim]
        输出:
          next_obs_pred: [B, obs_dim]
          reward_pred:   [B, 1]
        """
        x = torch.cat([s_embed, action, z], dim=-1)
        h = self.decoder_fc(x)
        next_obs_pred = self.recon_next_obs(h)
        reward_pred = self.recon_reward(h)
        return next_obs_pred, reward_pred
