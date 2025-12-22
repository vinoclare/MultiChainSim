# algs/wtoe.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def kl_normal(p_mean, p_std, q_mean, q_std, eps=1e-8):
    """
    KL( N(p_mean, p_std) || N(q_mean, q_std) )
    shape broadcastable, returns same broadcast shape.
    """
    p_var = (p_std ** 2).clamp(min=eps)
    q_var = (q_std ** 2).clamp(min=eps)
    return 0.5 * (
        (p_var + (p_mean - q_mean) ** 2) / q_var
        - 1.0
        + torch.log(q_var) - torch.log(p_var)
    )


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WTOEPolicyVAE(nn.Module):
    """
    用“宏观历史 + 状态嵌入”推断一个 step-level latent z，
    再解码出短期推断策略 π_hat(a|s,z) 的 (mean, std)。
    """
    def __init__(self, state_emb_dim, macro_hist_dim, z_dim, n_worker, num_pad_tasks, hidden_dim=256):
        super().__init__()
        self.z_dim = z_dim
        self.n_worker = n_worker
        self.num_pad_tasks = num_pad_tasks
        out_dim = 2 * (n_worker * num_pad_tasks)  # mean + log_std

        self.enc = MLP(state_emb_dim + macro_hist_dim, hidden_dim, 2 * z_dim, num_layers=3)
        self.dec = MLP(state_emb_dim + z_dim, hidden_dim, out_dim, num_layers=3)

    def encode(self, state_emb, macro_hist):
        h = torch.cat([state_emb, macro_hist], dim=-1)
        stats = self.enc(h)
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, state_emb, z, valid_mask=None):
        """
        return mean,std: [B,W,T]
        """
        B = state_emb.shape[0]
        x = torch.cat([state_emb, z], dim=-1)
        out = self.dec(x)  # [B, 2*W*T]
        half = self.n_worker * self.num_pad_tasks
        raw_mean = out[:, :half].view(B, self.n_worker, self.num_pad_tasks)
        raw_logstd = out[:, half:].view(B, self.n_worker, self.num_pad_tasks)

        mean = torch.sigmoid(raw_mean)
        log_std = torch.clamp(raw_logstd, min=-4.0, max=1.0)
        std = torch.exp(log_std)

        if valid_mask is not None:
            # valid_mask: [B,T] -> [B,W,T]
            mask = valid_mask.unsqueeze(1).expand_as(mean)
            mean = mean * mask
        return mean, std

    def forward(self, state_emb, macro_hist, valid_mask=None):
        mu, logvar = self.encode(state_emb, macro_hist)
        z = self.reparam(mu, logvar)
        mean, std = self.decode(state_emb, z, valid_mask=valid_mask)
        return mean, std, mu, logvar


class WTOE:
    """
    PPO/HAPPO 风格的 WToE：
    - actor 仍然是你现有 model.forward_actor 输出的 (mean,std_base)
    - 短期推断策略 π_hat 来自一个小 VAE（监督重构动作 + KL）
    - 当 KL(π_hat || π_actor) 变大 => explore_prob 变大
      => std_scaled = std_base * (1 + explore_scale * explore_prob)
      同时提高熵项权重（更“愿意”随机）
    """
    def __init__(
        self,
        model,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        initial_lr=3e-4,
        vae_lr=3e-4,
        eps=1e-5,
        max_grad_norm=0.5,
        device="cuda",
        inner_k=1,
        use_clipped_value_loss=True,
        norm_adv=True,
        writer=None,
        global_step_ref=None,
        total_training_steps=None,
        # ===== WToE params =====
        macro_hist_dim=None,     # = hist_len * macro_feat_dim
        z_dim=16,
        explore_scale=2.0,
        kl_threshold=0.05,
        kl_beta=10.0,
        entropy_boost=1.0,       # 额外熵权重倍率：1 + entropy_boost * p_exp
        vae_kl_coef=0.1,
        vae_adv_weighted=True,   # 用 advantage 加权重构项（更贴近“短期更优动作”）
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.initial_entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.norm_adv = norm_adv
        self.inner_k = inner_k

        self.writer = writer
        self.global_step_ref = global_step_ref
        self.total_training_steps = total_training_steps

        if macro_hist_dim is None:
            raise ValueError("WTOE requires macro_hist_dim (= hist_len * macro_feat_dim).")

        # 取 critic feature 维度：CrescentIndustrialModel 里是 3*hidden_dim
        # 这里不硬编码，从一次 forward 的输出推断更稳妥，但为了最小侵入：
        # 使用 model._build_critic_features 输出维度作为 state_emb_dim。
        self._macro_hist_dim = macro_hist_dim

        # W/T
        self.n_worker = getattr(model, "n_worker", None)
        self.num_pad_tasks = getattr(model, "num_pad_tasks", None)
        if self.n_worker is None or self.num_pad_tasks is None:
            raise ValueError("model must have attributes n_worker and num_pad_tasks")

        # optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_lr, eps=eps)

        # VAE
        with torch.no_grad():
            # dummy state_emb_dim probe (requires model._build_critic_features)
            # if user model doesn't expose it, we can fallback to a simple pooled embedding
            self._has_critic_feat = hasattr(self.model, "_build_critic_features")

        # state_emb_dim：用 hidden_dim 推断：critic_input_dim = 3*D，D=hidden_dim
        # 但 CrescentIndustrialModel 没暴露 D；所以如果有 _build_critic_features 就动态 probe
        if self._has_critic_feat:
            state_emb_dim = None  # dynamic init later
        else:
            # fallback：不推荐，但给个可运行兜底
            state_emb_dim = 256

        self._vae_state_emb_dim = state_emb_dim
        self.z_dim = z_dim
        self.explore_scale = explore_scale
        self.kl_threshold = kl_threshold
        self.kl_beta = kl_beta
        self.entropy_boost = entropy_boost
        self.vae_kl_coef = vae_kl_coef
        self.vae_adv_weighted = vae_adv_weighted

        self.vae = None
        self.vae_optimizer = None
        self._maybe_init_vae()

        # debug / logging
        self.last_explore_prob = 0.0
        self.last_kl = 0.0

    def _maybe_init_vae(self):
        if self.vae is not None:
            return
        if not hasattr(self.model, "_build_critic_features"):
            # fallback: still create something
            self.vae = WTOEPolicyVAE(
                state_emb_dim=self._vae_state_emb_dim,
                macro_hist_dim=self._macro_hist_dim,
                z_dim=self.z_dim,
                n_worker=self.n_worker,
                num_pad_tasks=self.num_pad_tasks,
                hidden_dim=256,
            ).to(self.device)
            self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=3e-4, eps=1e-5)
            return

        # dynamic probe state_emb_dim with a small fake batch
        # NOTE: 这里不依赖环境，靠输入 tensor shape 即可
        # 但我们无法知道 task_input_dim 等，所以等第一次 learn/sample 时再 probe
        # 因此先占位，第一次用到时再完成 init
        self._vae_pending_init = True

    def _ensure_vae_ready(self, task_obs, worker_loads, worker_profile, valid_mask):
        if self.vae is not None and getattr(self, "_vae_pending_init", False) is False:
            return
        if not hasattr(self.model, "_build_critic_features"):
            self._vae_pending_init = False
            return

        with torch.no_grad():
            state_emb = self.model._build_critic_features(
                task_obs, worker_loads, worker_profile, valid_mask
            )
        state_emb_dim = state_emb.shape[-1]
        self.vae = WTOEPolicyVAE(
            state_emb_dim=state_emb_dim,
            macro_hist_dim=self._macro_hist_dim,
            z_dim=self.z_dim,
            n_worker=self.n_worker,
            num_pad_tasks=self.num_pad_tasks,
            hidden_dim=256,
        ).to(self.device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.vae_optimizer.param_groups[0]["lr"] if self.vae_optimizer else 3e-4, eps=1e-5)
        self._vae_pending_init = False

    @torch.no_grad()
    def _get_state_emb(self, task_obs, worker_loads, worker_profile, valid_mask):
        if hasattr(self.model, "_build_critic_features"):
            return self.model._build_critic_features(task_obs, worker_loads, worker_profile, valid_mask)
        # fallback: simple pooling
        # task_obs: [B,T,D] -> pool
        t_pool = task_obs.mean(dim=1)
        w_pool = worker_loads.mean(dim=1)
        p_pool = worker_profile.mean(dim=1)
        return torch.cat([t_pool, w_pool, p_pool], dim=-1)

    def _compute_explore_prob(self, mean_actor, std_actor, mean_inf, std_inf):
        """
        mean/std: [B,W,T]
        return p_exp: [B,W] and kl_worker: [B,W]
        """
        kl_elem = kl_normal(mean_inf, std_inf, mean_actor, std_actor)  # [B,W,T]
        kl_worker = kl_elem.mean(dim=2)  # [B,W]
        p_exp = torch.sigmoid(self.kl_beta * (kl_worker - self.kl_threshold))
        return p_exp, kl_worker

    # ------------------------------------------------------------------
    # API: sample / predict / value
    # ------------------------------------------------------------------
    def sample(self, task_obs, worker_loads, worker_profile, valid_mask, macro_hist=None):
        """
        macro_hist: [B, macro_hist_dim]
        """
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)
        if macro_hist is None:
            # 没给历史 => 退化成普通 PPO
            macro_hist = torch.zeros((task_obs.shape[0], self._macro_hist_dim), device=self.device, dtype=torch.float32)
        else:
            macro_hist = macro_hist.to(self.device, dtype=torch.float32)

        self._ensure_vae_ready(task_obs, worker_loads, worker_profile, valid_mask)

        mean_actor, std_base = self.model.forward_actor(task_obs, worker_loads, worker_profile, valid_mask)
        state_emb = self._get_state_emb(task_obs, worker_loads, worker_profile, valid_mask)

        mean_inf, std_inf, _, _ = self.vae(state_emb, macro_hist, valid_mask=valid_mask)
        p_exp, kl_worker = self._compute_explore_prob(mean_actor, std_base, mean_inf, std_inf)

        # 状态相关探索：放大 std
        std_scaled = std_base * (1.0 + self.explore_scale * p_exp.unsqueeze(-1))

        dist = Normal(mean_actor, std_scaled)
        action = dist.sample()
        log_prob = dist.log_prob(action)  # [B,W,T]
        entropy = dist.entropy()          # [B,W,T]

        value = self.model.get_value(task_obs, worker_loads, worker_profile, valid_mask)

        # debug
        self.last_explore_prob = float(p_exp.mean().item())
        self.last_kl = float(kl_worker.mean().item())

        return value, action, log_prob.sum(dim=2), entropy.sum(dim=2)

    def predict(self, task_obs, worker_loads, worker_profile, valid_mask):
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)
        mean, _ = self.model.forward_actor(task_obs, worker_loads, worker_profile, valid_mask)
        return mean

    def value(self, task_obs, worker_loads, worker_profile, valid_mask):
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)
        return self.model.get_value(task_obs, worker_loads, worker_profile, valid_mask)

    # ------------------------------------------------------------------
    # learn
    # ------------------------------------------------------------------
    def _vae_update(self, task_obs, worker_loads, worker_profile, valid_mask, macro_hist, actions, advantages):
        """
        VAE: 重构动作 + KL(q(z|.)||N(0,1))
        """
        self.vae.train()
        B = task_obs.shape[0]

        with torch.no_grad():
            state_emb = self._get_state_emb(task_obs, worker_loads, worker_profile, valid_mask)

        mean_inf, std_inf, mu, logvar = self.vae(state_emb, macro_hist, valid_mask=valid_mask)
        dist = Normal(mean_inf, std_inf)
        logp = dist.log_prob(actions)  # [B,W,T]
        logp_sum = logp.sum(dim=(1, 2))  # [B]

        if self.vae_adv_weighted:
            # 用 mean-adv 做权重，强调“短期更优”的动作模式
            w = advantages.mean(dim=1).clamp(min=0.0).detach()  # [B]
            w = w / (w.mean() + 1e-8)
            recon_loss = -(w * logp_sum).mean()
        else:
            recon_loss = -(logp_sum).mean()

        # KL(q||p)
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=-1)  # [B]
        kl_loss = kl.mean()

        loss = recon_loss + self.vae_kl_coef * kl_loss

        self.vae_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)
        self.vae_optimizer.step()

        return float(loss.item()), float(recon_loss.item()), float(kl_loss.item())

    def learn(
        self,
        task_obs,
        worker_loads,
        worker_profile,
        valid_mask,
        macro_hist,     # [B, macro_hist_dim]
        actions,
        values_old,     # [B]
        returns,        # [B]
        log_probs_old,  # [B,W]
        advantages,     # [B,W]
        current_steps,
        lr=None,
    ):
        device = self.device
        task_obs = task_obs.to(device)
        worker_loads = worker_loads.to(device)
        worker_profile = worker_profile.to(device)
        valid_mask = valid_mask.to(device)
        macro_hist = macro_hist.to(device, dtype=torch.float32)

        actions = actions.to(device)
        values_old = values_old.to(device)
        returns = returns.to(device)
        log_probs_old = log_probs_old.to(device)
        advantages = advantages.to(device)

        self._ensure_vae_ready(task_obs, worker_loads, worker_profile, valid_mask)

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # entropy schedule（沿用你 CRESCENT 里的线性衰减）
        if self.total_training_steps is not None and self.total_training_steps > 0:
            frac = float(current_steps) / float(self.total_training_steps)
            frac = max(0.0, min(1.0, frac))
            entropy_coef = max(1e-3, self.initial_entropy_coef * (1.0 - frac))
        else:
            entropy_coef = self.entropy_coef

        if lr is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        # ---- 1) VAE update (once per minibatch) ----
        vae_loss, vae_recon, vae_kl = self._vae_update(
            task_obs, worker_loads, worker_profile, valid_mask,
            macro_hist, actions, advantages
        )

        # ---- 2) PPO/HAPPO update (sequential workers, like your CRESCENT) ----
        self.model.train()
        num_workers = actions.size(1)
        action_loss_total = 0.0
        entropy_total = 0.0
        value_loss_last = 0.0

        for w in range(num_workers):
            for _ in range(self.inner_k):
                mean_actor, std_base = self.model.forward_actor(task_obs, worker_loads, worker_profile, valid_mask)

                with torch.no_grad():
                    state_emb = self._get_state_emb(task_obs, worker_loads, worker_profile, valid_mask)
                    mean_inf, std_inf, _, _ = self.vae(state_emb, macro_hist, valid_mask=valid_mask)
                    p_exp, _ = self._compute_explore_prob(mean_actor, std_base, mean_inf, std_inf)
                    p_exp = p_exp.detach()

                std_scaled = std_base * (1.0 + self.explore_scale * p_exp.unsqueeze(-1))
                dist = Normal(mean_actor, std_scaled)

                log_probs = dist.log_prob(actions).sum(dim=2)  # [B,W]
                entropy_vec = dist.entropy().sum(dim=2)         # [B,W]

                ratio = torch.exp(log_probs[:, w] - log_probs_old[:, w])  # [B]
                adv = advantages[:, w]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
                action_loss = -torch.min(surr1, surr2).mean()

                values = self.model.get_value(task_obs, worker_loads, worker_profile, valid_mask).view(-1)
                if self.use_clipped_value_loss:
                    value_pred_clipped = values_old + torch.clamp(values - values_old, -self.clip_param, self.clip_param)
                    value_losses = (values - returns) ** 2
                    value_losses_clipped = (value_pred_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()
                value_loss_last = float(value_loss.item())

                # WToE: divergence 大时更想探索 => 给熵项加权
                ent_w = (1.0 + self.entropy_boost * p_exp[:, w])
                entropy = (entropy_vec[:, w] * ent_w).mean()

                loss = value_loss * self.value_loss_coef + action_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                action_loss_total += float(action_loss.item())
                entropy_total += float(entropy.item())

        denom = num_workers * max(1, self.inner_k)
        return {
            "value_loss": value_loss_last,
            "policy_loss": action_loss_total / denom,
            "entropy": entropy_total / denom,
            "vae_loss": vae_loss,
            "vae_recon": vae_recon,
            "vae_kl": vae_kl,
        }
