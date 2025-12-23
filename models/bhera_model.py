# models/bhera_model.py
import torch
import torch.nn as nn
from torch.distributions import Normal


class BHERAJointModel(nn.Module):
    """
    BHERA Joint Model (one-file, minimal but complete):
    - raw obs per layer -> obs embedding
    - layer-wise coupling via BiGRU over layer dimension
    - belief inference with:
        slow: windowed causal Transformer (sample z_slow)
        fast: GRUCell step update (sample z_fast_t)
      and q_head: p(sparse) = sigmoid(MLP([z_slow, z_fast]))
    - per-layer policy (Gaussian) conditioned on (h_tilde_l, z)
    - per-layer value ensemble conditioned on (h_tilde_l, z)
    - global n-step return decoder conditioned on (x_pool, a_pool, z)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_layers: int,
        # enc/coupling
        obs_embed_dim: int = 128,
        coupling_hidden: int = 128,
        # belief token/proj
        belief_in_dim: int = 128,          # projected token dim fed to slow/fast
        slow_window_len: int = 16,         # used by agent/buffer; model keeps for reference
        slow_n_layers: int = 2,
        slow_n_heads: int = 4,
        slow_ff_dim: int = 256,
        fast_hidden: int = 128,
        z_slow_dim: int = 32,
        z_fast_dim: int = 32,
        # policy/value
        policy_hidden: int = 256,
        value_hidden: int = 256,
        value_ensemble: int = 5,
        # decoder
        decoder_hidden: int = 256,
        # gaussian std clamp
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_layers = int(num_layers)

        self.obs_embed_dim = int(obs_embed_dim)
        self.coupling_hidden = int(coupling_hidden)

        self.belief_in_dim = int(belief_in_dim)
        self.slow_window_len = int(slow_window_len)
        self.fast_hidden = int(fast_hidden)
        self.z_slow_dim = int(z_slow_dim)
        self.z_fast_dim = int(z_fast_dim)
        self.z_dim = int(z_slow_dim + z_fast_dim)

        self.value_ensemble = int(value_ensemble)

        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        # -------------------- Obs encoder: raw_obs -> h (shared across layers) --------------------
        # raw obs is flattened vector (same as run_varibad.py / varibad_model.py convention)
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.obs_embed_dim),
            nn.ReLU(),
        )

        # -------------------- Layer coupling: BiGRU over layer-dim --------------------
        # Input:  [B, L, D]
        # Output: [B, L, D]
        # Use bidirectional GRU, then project back to D.
        self.layer_bigru = nn.GRU(
            input_size=self.obs_embed_dim,
            hidden_size=self.coupling_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=False,  # we will feed [L, B, D]
        )
        self.layer_couple_proj = nn.Linear(2 * self.coupling_hidden, self.obs_embed_dim)

        # -------------------- Belief token projection --------------------
        # token = [x_pool, a_prev_pool, r_prev, done_prev]
        # token_dim = obs_embed_dim + action_dim + 1 + 1
        self.token_dim = self.obs_embed_dim + self.action_dim + 2
        self.token_proj = nn.Sequential(
            nn.Linear(self.token_dim, self.belief_in_dim),
            nn.ReLU(),
        )

        # -------------------- Slow belief: causal Transformer over time window --------------------
        # We project token to belief_in_dim, then TransformerEncoder, then take last position.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.belief_in_dim,
            nhead=int(slow_n_heads),
            dim_feedforward=int(slow_ff_dim),
            activation="relu",
            batch_first=True,  # [B, K, D]
            norm_first=True,
        )
        self.slow_transformer = nn.TransformerEncoder(enc_layer, num_layers=int(slow_n_layers))
        self.slow_mu = nn.Linear(self.belief_in_dim, self.z_slow_dim)
        self.slow_logvar = nn.Linear(self.belief_in_dim, self.z_slow_dim)

        # -------------------- Fast belief: GRUCell step update --------------------
        self.fast_rnn = nn.GRUCell(
            input_size=self.belief_in_dim,
            hidden_size=self.fast_hidden,
        )
        self.fast_mu = nn.Linear(self.fast_hidden, self.z_fast_dim)
        self.fast_logvar = nn.Linear(self.fast_hidden, self.z_fast_dim)
        self.fast_slow_to_h = nn.Linear(self.z_slow_dim, self.fast_hidden)

        self.fast_prior_net = nn.Sequential(
            nn.Linear(self.z_fast_dim + self.z_slow_dim, max(64, self.z_fast_dim * 2)),
            nn.ReLU(),
            nn.Linear(max(64, self.z_fast_dim * 2), 2 * self.z_fast_dim),
        )

        # -------------------- q head: p(sparse) --------------------
        self.q_head = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # -------------------- Policy: per-layer heads conditioned on (h_tilde_l, z) --------------------
        self.policy_fcs = nn.ModuleList()
        self.policy_means = nn.ModuleList()
        self.policy_logstds = nn.ParameterList()

        for _ in range(self.num_layers):
            fc = nn.Sequential(
                nn.Linear(self.obs_embed_dim + self.z_dim, int(policy_hidden)),
                nn.ReLU(),
                nn.Linear(int(policy_hidden), int(policy_hidden)),
                nn.ReLU(),
            )
            self.policy_fcs.append(fc)
            self.policy_means.append(nn.Linear(int(policy_hidden), self.action_dim))
            self.policy_logstds.append(nn.Parameter(torch.zeros(self.action_dim)))

        # -------------------- Value ensemble: per-layer trunk + N heads --------------------
        self.value_trunks = nn.ModuleList()
        self.value_heads = nn.ModuleList()  # each item is nn.ModuleList([head_i...])

        for _ in range(self.num_layers):
            trunk = nn.Sequential(
                nn.Linear(self.obs_embed_dim + self.z_dim, int(value_hidden)),
                nn.ReLU(),
                nn.Linear(int(value_hidden), int(value_hidden)),
                nn.ReLU(),
            )
            heads = nn.ModuleList([nn.Linear(int(value_hidden), 1) for _ in range(self.value_ensemble)])
            self.value_trunks.append(trunk)
            self.value_heads.append(heads)

        # -------------------- n-step return decoder (global) --------------------
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.obs_embed_dim + self.action_dim + self.z_dim, int(decoder_hidden)),
            nn.ReLU(),
            nn.Linear(int(decoder_hidden), int(decoder_hidden)),
            nn.ReLU(),
            nn.Linear(int(decoder_hidden), 1),
        )

    # -------------------- helpers --------------------

    @staticmethod
    def _reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _causal_mask(K: int, device: torch.device) -> torch.Tensor:
        # Transformer uses additive mask: float('-inf') for disallowed positions
        # shape: [K, K]
        mask = torch.triu(torch.ones((K, K), device=device), diagonal=1)
        mask = mask.masked_fill(mask > 0, float("-inf"))
        return mask

    # -------------------- obs / coupling --------------------

    def encode_obs(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """
        raw_obs: [B, obs_dim]
        return:  [B, obs_embed_dim]
        """
        return self.obs_encoder(raw_obs)

    def encode_obs_stack(self, raw_obs_stack: torch.Tensor) -> torch.Tensor:
        """
        raw_obs_stack: [B, L, obs_dim]
        return:        [B, L, obs_embed_dim]
        """
        B, L, D = raw_obs_stack.shape
        x = raw_obs_stack.reshape(B * L, D)
        h = self.encode_obs(x).view(B, L, -1)
        return h

    def couple_layers(self, h_stack: torch.Tensor) -> torch.Tensor:
        """
        BiGRU coupling over layer dimension.

        h_stack:  [B, L, obs_embed_dim]
        return:   [B, L, obs_embed_dim]
        """
        B, L, D = h_stack.shape
        x = h_stack.permute(1, 0, 2).contiguous()  # [L, B, D]
        y, _ = self.layer_bigru(x)                 # [L, B, 2*coupling_hidden]
        y = self.layer_couple_proj(y)             # [L, B, D]
        out = y.permute(1, 0, 2).contiguous()      # [B, L, D]
        return out

    def pool_layers_mean(self, h_tilde_stack: torch.Tensor) -> torch.Tensor:
        """
        Mean pool over layers.
        h_tilde_stack: [B, L, D]
        return:        [B, D]
        """
        return h_tilde_stack.mean(dim=1)

    # -------------------- belief (slow/fast/q) --------------------

    def fast_prior(self, z_fast_prev: torch.Tensor, z_slow: torch.Tensor):
        """
        Conditional prior: p(z_fast_t | z_fast_{t-1}, z_slow)
        return: mu_p, logvar_p [B, z_fast_dim]
        """
        x = torch.cat([z_fast_prev, z_slow], dim=-1)
        out = self.fast_prior_net(x)
        mu_p, logvar_p = torch.chunk(out, 2, dim=-1)
        logvar_p = logvar_p.clamp(-10.0, 10.0)
        mu_p = torch.nan_to_num(mu_p, nan=0.0, posinf=0.0, neginf=0.0)
        logvar_p = torch.nan_to_num(logvar_p, nan=0.0, posinf=0.0, neginf=0.0)
        return mu_p, logvar_p

    def belief_slow(self, token_window: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        token_window: [B, K, token_dim] (left padded)
        padding_mask: [B, K] bool, True means PAD (optional)
        return: mu, logvar, z each [B, z_slow_dim]
        """
        x = self.token_proj(token_window)  # [B,K,belief_in_dim]
        B, K, D = x.shape

        # causal mask for full length
        if padding_mask is None or (not torch.any(padding_mask)):
            attn_mask = self._causal_mask(K, device=x.device)  # [K,K]
            y = self.slow_transformer(x, mask=attn_mask)  # [B,K,D]
            y_last = y[:, -1, :]
        else:
            # robust path: strip PAD tokens per sample to avoid NaN with mask interaction
            y_last_list = []
            for b in range(B):
                pm = padding_mask[b]  # [K]
                valid_idx = torch.nonzero(~pm, as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    y_last_list.append(torch.zeros((D,), device=x.device, dtype=x.dtype))
                    continue

                xb = x[b:b + 1, valid_idx, :]  # [1,Kb,D]
                Kb = xb.shape[1]
                attn_b = self._causal_mask(Kb, device=x.device)
                yb = self.slow_transformer(xb, mask=attn_b)  # [1,Kb,D]
                y_last_list.append(yb[0, -1, :])

            y_last = torch.stack(y_last_list, dim=0)  # [B,D]

        mu = self.slow_mu(y_last)
        logvar = self.slow_logvar(y_last).clamp(-10.0, 10.0)

        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        logvar = torch.nan_to_num(logvar, nan=0.0, posinf=0.0, neginf=0.0)

        z = self._reparam(mu, logvar)
        return mu, logvar, z

    def belief_fast_step(
            self,
            token_t: torch.Tensor,
            h_prev: torch.Tensor,
            done_prev: torch.Tensor = None,
            z_slow: torch.Tensor = None,
    ):
        """
        token_t:   [B, token_dim]
        h_prev:    [B, fast_hidden]
        done_prev: [B,1]
        z_slow:    [B, z_slow_dim]  (conditions posterior emission)
        """
        if done_prev is not None:
            reset = (1.0 - done_prev.float()).clamp(0.0, 1.0)
            h_prev = h_prev * reset

        x = self.token_proj(token_t)  # [B, belief_in_dim]
        h_new = self.fast_rnn(x, h_prev)

        h_emit = h_new
        if z_slow is not None:
            h_emit = h_emit + self.fast_slow_to_h(z_slow)

        mu = self.fast_mu(h_emit)
        logvar = self.fast_logvar(h_emit).clamp(-10.0, 10.0)

        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        logvar = torch.nan_to_num(logvar, nan=0.0, posinf=0.0, neginf=0.0)

        z = self._reparam(mu, logvar)
        return h_new, mu, logvar, z

    def compute_q(self, z_slow: torch.Tensor, z_fast: torch.Tensor) -> torch.Tensor:
        """
        z_slow: [B, z_slow_dim]
        z_fast: [B, z_fast_dim]
        return:
          q: [B, 1] in (0,1)
        """
        z = torch.cat([z_slow, z_fast], dim=-1)
        q = torch.sigmoid(self.q_head(z))
        return q

    # -------------------- policy / value --------------------

    def forward_actor(self, lid: int, h_tilde: torch.Tensor, z: torch.Tensor):
        """
        lid: layer id
        h_tilde: [B, obs_embed_dim]
        z:       [B, z_dim]
        return: mean/std for Normal
        """
        x = torch.cat([h_tilde, z], dim=-1)
        h = self.policy_fcs[lid](x)
        mean = self.policy_means[lid](h)

        logstd = self.policy_logstds[lid].clamp(self.min_log_std, self.max_log_std)
        logstd = logstd.view(1, -1).expand_as(mean)
        std = torch.exp(logstd)
        return mean, std

    def act(self, lid: int, h_tilde: torch.Tensor, z: torch.Tensor):
        """
        Sample action from layer-specific Gaussian policy.
        Returns:
          action:  [B, action_dim]
          logp:    [B, 1]
          entropy: [B, 1]
        """
        mean, std = self.forward_actor(lid, h_tilde, z)
        dist = Normal(mean, std)
        action = dist.rsample()
        logp = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, logp, entropy

    def get_value_ensemble(self, lid: int, h_tilde: torch.Tensor, z: torch.Tensor):
        """
        Return per-layer ensemble values.
        Returns:
          v_all:  [B, N]
          v_mean: [B, 1]
          u:      [B, 1]  (std over heads)
        """
        x = torch.cat([h_tilde, z], dim=-1)
        trunk = self.value_trunks[lid](x)  # [B, H]
        vs = []
        for head in self.value_heads[lid]:
            vs.append(head(trunk))  # [B,1]
        v_all = torch.cat(vs, dim=-1)  # [B, N]
        v_mean = v_all.mean(dim=-1, keepdim=True)  # [B,1]
        u = v_all.std(dim=-1, keepdim=True)        # [B,1]
        return v_all, v_mean, u

    # -------------------- n-step return decoder --------------------

    def decode_nstep_return(self, x_pool: torch.Tensor, a_pool: torch.Tensor, z: torch.Tensor):
        """
        Predict n-step return target (global decoder).
        Inputs:
          x_pool: [B, obs_embed_dim]   pooled coupled state embedding
          a_pool: [B, action_dim]      pooled action (e.g., mean over layers)
          z:      [B, z_dim]
        Output:
          g_hat:  [B, 1]
        """
        inp = torch.cat([x_pool, a_pool, z], dim=-1)
        return self.decoder_fc(inp)
