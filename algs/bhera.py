import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from utils.utils_bhera import EMA, compute_nstep_returns, left_pad_last_k


class BHERA:
    """
    BHERA joint learner:
    - 跨层耦合（由 model.couple_layers 决定 BiGRU/Transformer）
    - slow/fast latent 推断（窗口化 slow + 递推 fast）
    - q_t 参与 Regime-balanced（样本权重）
    - ensemble value 给不确定性（可选做可信度调节，这里默认不开）
    - reward decoder 预测 n-step return
    - KL homeostasis：用 EMA 稳态调节 beta（信息容量）
    """

    def __init__(
        self,
        model,
        clip_param=0.1,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        initial_lr=2.5e-4,
        eps=1e-5,
        max_grad_norm=0.5,
        inner_k=1,
        device="cuda",
        use_clipped_value_loss=True,
        norm_adv=True,
        # regime-balanced
        lambda_q=1.0,
        # decoder / KL
        decoder_coef=1.0,
        decoder_sparse_weight=0.0,   # >0：对 |r|>0 加权
        gamma=0.99,
        n_step=5,
        # homeostasis
        beta_init=0.1,
        kl_capacity=1.0,
        beta_lr=1e-3,
        kl_ema_decay=0.99,
        # logging
        writer=None,
        global_step_ref=None,
        total_training_steps=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.clip_param = float(clip_param)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.initial_entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.norm_adv = bool(norm_adv)
        self.inner_k = int(inner_k)

        self.lambda_q = float(lambda_q)

        self.decoder_coef = float(decoder_coef)
        self.decoder_sparse_weight = float(decoder_sparse_weight)
        self.gamma = float(gamma)
        self.n_step = int(n_step)

        self.beta = float(beta_init)
        self.kl_capacity = float(kl_capacity)
        self.beta_lr = float(beta_lr)
        self.kl_ema = EMA(decay=float(kl_ema_decay), init=0.0)

        self.optimizer = optim.Adam(self.model.parameters(), lr=float(initial_lr), eps=float(eps))

        self.writer = writer
        self.global_step_ref = global_step_ref
        self.total_training_steps = total_training_steps if total_training_steps is not None else 1

    # ------------------------- small helpers -------------------------

    @staticmethod
    def _kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(q||p), p=N(0,I): 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)

    @staticmethod
    def _squeeze_last(x: torch.Tensor) -> torch.Tensor:
        # [*,1] -> [*]
        if x is None:
            return x
        if x.dim() >= 1 and x.shape[-1] == 1:
            return x[..., 0]
        return x

    def _entropy_coef(self, current_steps: int) -> float:
        progress = float(current_steps) / float(self.total_training_steps)
        return max(1e-3, self.initial_entropy_coef * (1.0 - progress))

    # ------------------------- model API adapters (robust to naming) -------------------------

    def _call_couple_layers(self, h_btl_d: torch.Tensor) -> torch.Tensor:
        # h: [B,T,L,D] -> [B,T,L,D]
        B, T, L, D = h_btl_d.shape
        x = h_btl_d.reshape(B * T, L, D)
        if hasattr(self.model, "couple_layers"):
            y = self.model.couple_layers(x)
        else:
            # 没耦合模块就原样返回
            y = x
        return y.reshape(B, T, L, D)

    def _call_slow_infer(self, token_win_bk_d: torch.Tensor, padding_mask: torch.Tensor = None):
        return self.model.belief_slow(token_win_bk_d, padding_mask=padding_mask)

    def _call_fast_step(self, token_b_d: torch.Tensor, h_prev: torch.Tensor, done_prev: torch.Tensor):
        # return h_new, mu, logvar, z
        if hasattr(self.model, "fast_step"):
            return self.model.fast_step(token_b_d, h_prev, done_prev)
        if hasattr(self.model, "belief_fast_step"):
            return self.model.belief_fast_step(token_b_d, h_prev, done_prev)
        if hasattr(self.model, "belief_step_fast"):
            return self.model.belief_step_fast(token_b_d, h_prev, done_prev)
        raise AttributeError("BHERA model missing fast step method: fast_step / belief_fast_step / belief_step_fast")

    def _call_q_head(self, z_bt_d: torch.Tensor) -> torch.Tensor:
        # z: [B*T,Z] -> q: [B*T,1]
        if hasattr(self.model, "q_head"):
            q_logits = self.model.q_head(z_bt_d)
            q = torch.sigmoid(q_logits)
        elif hasattr(self.model, "compute_q"):
            q = self.model.compute_q(z_bt_d)
        else:
            # 没 q head 就给 0（等价于不做 regime-balanced）
            q = torch.zeros((z_bt_d.shape[0], 1), dtype=z_bt_d.dtype, device=z_bt_d.device)
        return q

    def _call_forward_actor(self, lid: int, s_flat: torch.Tensor, z_flat: torch.Tensor):
        # return mean, std
        if hasattr(self.model, "forward_actor"):
            try:
                return self.model.forward_actor(lid, s_flat, z_flat)
            except TypeError:
                return self.model.forward_actor(s_flat, z_flat)
        raise AttributeError("BHERA model missing forward_actor")

    def _call_value_ensemble(self, lid: int, s_flat: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        # want: v_all [B*T, N]
        if hasattr(self.model, "get_value_ensemble"):
            out = self.model.get_value_ensemble(lid, s_flat, z_flat)
        elif hasattr(self.model, "value_ensemble"):
            out = self.model.value_ensemble(lid, s_flat, z_flat)
        else:
            raise AttributeError("BHERA model missing value ensemble method: get_value_ensemble / value_ensemble")

        if torch.is_tensor(out):
            v_all = out
        elif isinstance(out, (tuple, list)):
            # 找一个看起来像 [B*T,N] 的 tensor
            cand = None
            for item in out:
                if torch.is_tensor(item) and item.dim() >= 2:
                    cand = item
                    break
            if cand is None:
                raise RuntimeError("get_value_ensemble returned tuple/list but no tensor found")
            v_all = cand
        else:
            raise RuntimeError("get_value_ensemble returned unsupported type")

        # 统一成 [B*T, N]
        if v_all.dim() == 1:
            v_all = v_all.unsqueeze(-1)
        if v_all.dim() == 2:
            return v_all
        if v_all.dim() == 3 and v_all.shape[-1] == 1:
            return v_all[..., 0]
        if v_all.dim() == 3 and v_all.shape[1] == 1:
            return v_all[:, 0, :]
        # 兜底：展平到 2D
        return v_all.reshape(v_all.shape[0], -1)

    def _call_decode_return(self, x_flat: torch.Tensor, a_flat: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        # pred: [B*T,1]
        if hasattr(self.model, "decode_return"):
            pred = self.model.decode_return(x_flat, a_flat, z_flat)
        elif hasattr(self.model, "decode_nstep_return"):
            pred = self.model.decode_nstep_return(x_flat, a_flat, z_flat)
        elif hasattr(self.model, "reward_decoder"):
            pred = self.model.reward_decoder(torch.cat([x_flat, a_flat, z_flat], dim=-1))
        else:
            raise AttributeError("BHERA model missing decoder: decode_return / decode_nstep_return / reward_decoder")

        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        return pred

    # ------------------------- belief rebuild (joint) -------------------------

    def _rebuild_joint_latent(self, obs_raw, actions, rewards, dones):
        """
        obs_raw : [B,T,L,obs_dim]
        actions : [B,T,L,act_dim]
        rewards : [B,T,L,1]
        dones   : [B,T,L,1]
        return:
          h_cpl  : [B,T,L,D]  (耦合后每层表征，给 actor/critic）
          x_pool : [B,T,D]    (跨层 pooled，给 token/decoder）
          a_pool : [B,T,A]    (跨层 pooled action）
          z_seq  : [B,T,Z]
          q_seq  : [B,T,1]
          r_global: [B,T,1]
          kl_slow, kl_fast (scalar tensors)
        """
        B, T, L, _ = obs_raw.shape

        # 1) encode obs per layer
        obs_flat = obs_raw.reshape(B * T * L, -1)
        h = self.model.encode_obs(obs_flat).reshape(B, T, L, -1)  # [B,T,L,D]

        # 2) couple across layers (BiGRU over layer-dim)
        h_cpl = self._call_couple_layers(h)  # [B, T, L, D]

        # 3) pooled features for belief/decoder
        x_pool = h_cpl.mean(dim=2)  # [B,T,D]
        a_pool = actions.mean(dim=2)  # [B,T,A]
        r_global = rewards.sum(dim=2)  # [B,T,1]
        d_global = dones.max(dim=2).values  # [B,T,1]

        # 4) build token_t = [x_t, a_{t-1}, r_{t-1}, d_{t-1}]
        a_prev = torch.zeros_like(a_pool)
        r_prev = torch.zeros_like(r_global)
        d_prev = torch.zeros_like(d_global)
        if T > 1:
            a_prev[:, 1:] = a_pool[:, :-1]
            r_prev[:, 1:] = r_global[:, :-1]
            d_prev[:, 1:] = d_global[:, :-1]
        token = torch.cat([x_pool, a_prev, r_prev, d_prev], dim=-1)  # [B,T,token_dim]

        # 5) slow infer per-timestep with causal window (NO future leakage)
        ks = int(getattr(self.model, "slow_window_len", min(32, T)))
        ks = max(1, min(ks, T))
        Dtok = token.shape[-1]

        # left pad so every timestep t has a full window ending at t
        pad = torch.zeros((B, ks - 1, Dtok), device=token.device, dtype=token.dtype)
        token_pad = torch.cat([pad, token], dim=1)  # [B, T+ks-1, Dtok]

        # windows from token_pad:
        # PyTorch unfold gives [B, T, Dtok, ks] -> permute to [B, T, ks, Dtok]
        wins = token_pad.unfold(dimension=1, size=ks, step=1)  # [B, T, Dtok, ks]
        wins = wins.permute(0, 1, 3, 2).contiguous()  # [B, T, ks, Dtok]
        wins = wins.view(B * T, ks, Dtok)  # [B*T, ks, Dtok]

        # ---- padding_mask per timestep (关键：逐 t 不同) ----
        # 对时刻 t（从 0 开始）：pad_len(t) = max(0, ks-1 - t)
        # mask True 表示 PAD（左侧的补零）
        t_idx = torch.arange(T, device=token.device)  # [T]
        pad_len = (ks - 1 - t_idx).clamp(min=0)  # [T]
        pos = torch.arange(ks, device=token.device)  # [ks]
        mask_tk = (pos.unsqueeze(0) < pad_len.unsqueeze(1))  # [T, ks] bool
        pad_mask = mask_tk.unsqueeze(0).expand(B, T, ks).reshape(B * T, ks)  # [B*T, ks]

        mu_slow_bt, logvar_slow_bt, z_slow_bt = self._call_slow_infer(wins, padding_mask=pad_mask)  # [B*T,Ds]
        kl_slow = self._kl_standard_normal(mu_slow_bt, logvar_slow_bt).mean()
        z_slow = z_slow_bt.view(B, T, -1)  # [B,T,Ds]

        # 6) fast infer recurrently (GRUCell step)
        # init fast hidden
        if hasattr(self.model, "fast_hidden"):
            fast_hidden = int(self.model.fast_hidden)
        elif hasattr(self.model, "belief_fast_hidden"):
            fast_hidden = int(self.model.belief_fast_hidden)
        else:
            # fallback
            fast_hidden = 128

        h_fast = torch.zeros((B, fast_hidden), device=token.device, dtype=token.dtype)
        z_fast_list, mu_fast_list, logvar_fast_list = [], [], []

        for t in range(T):
            token_t = token[:, t, :]  # [B, token_dim]
            done_prev_t = d_prev[:, t, :]  # [B,1]
            h_fast, mu_f, logvar_f, z_f = self._call_fast_step(token_t, h_fast, done_prev_t)
            z_fast_list.append(z_f)
            mu_fast_list.append(mu_f)
            logvar_fast_list.append(logvar_f)

        z_fast = torch.stack(z_fast_list, dim=1)  # [B,T,Df]
        mu_fast = torch.stack(mu_fast_list, dim=1)  # [B,T,Df]
        logvar_fast = torch.stack(logvar_fast_list, dim=1)  # [B,T,Df]
        kl_fast = self._kl_standard_normal(
            mu_fast.reshape(B * T, -1),
            logvar_fast.reshape(B * T, -1)
        ).mean()

        # 7) concat slow+fast -> z_t
        z_seq = torch.cat([z_slow, z_fast], dim=-1)  # [B,T,Z]

        # 8) q_t head
        q_flat = self._call_q_head(z_seq.reshape(B * T, -1))  # [B*T,1]
        q_seq = q_flat.reshape(B, T, 1)

        return h_cpl, x_pool, a_pool, z_seq, q_seq, r_global, kl_slow, kl_fast

    # ------------------------- sampling (optional, for agent) -------------------------

    @torch.no_grad()
    def sample_joint(self, obs_raw_bl, a_prev_pool, r_prev, done_prev, token_window_bk, h_fast_prev):
        """
        采样接口（可给 bhera_agent 用）：
        obs_raw_bl     : [B,L,obs_dim]
        a_prev_pool    : [B,A]
        r_prev         : [B,1]     (global)
        done_prev      : [B,1]     (global)
        token_window_bk: [B,K,token_dim]   (给 slow transformer 的窗口)
        h_fast_prev    : [B,Hf]
        return:
          actions   : [B,L,act_dim]
          logp      : [B,L,1]
          entropy   : [B,L,1]
          v_mean    : [B,L,1]
          q         : [B,1]
          h_fast_new: [B,Hf]
        """
        obs_raw_bl = obs_raw_bl.to(self.device)
        a_prev_pool = a_prev_pool.to(self.device)
        r_prev = r_prev.to(self.device)
        done_prev = done_prev.to(self.device)
        token_window_bk = token_window_bk.to(self.device)
        h_fast_prev = h_fast_prev.to(self.device)

        B, L, _ = obs_raw_bl.shape

        # encode & couple
        obs_flat = obs_raw_bl.reshape(B * L, -1)
        h = self.model.encode_obs(obs_flat).reshape(B, L, -1)  # [B,L,D]
        h_cpl = self._call_couple_layers(h.unsqueeze(1)).squeeze(1)  # [B,L,D]

        x_pool = h_cpl.mean(dim=1)  # [B,D]
        token_t = torch.cat([x_pool, a_prev_pool, r_prev, done_prev], dim=-1)  # [B,token_dim]

        # slow / fast
        mu_s, logvar_s, z_s = self._call_slow_infer(token_window_bk)  # [B,Ds]
        if done_prev.dim() == 2:
            h_fast_prev = h_fast_prev * (1.0 - done_prev)
        else:
            h_fast_prev = h_fast_prev * (1.0 - done_prev.unsqueeze(-1))
        h_fast_new, mu_f, logvar_f, z_f = self._call_fast_step(token_t, h_fast_prev, done_prev)

        z = torch.cat([z_s, z_f], dim=-1)  # [B,Z]
        q = self._call_q_head(z)           # [B,1]

        # per-layer action/value
        actions, logps, ents, vmeans = [], [], [], []
        for lid in range(L):
            mean, std = self._call_forward_actor(lid, h_cpl[:, lid, :], z)
            std = torch.clamp(std, min=1e-6)
            dist = Normal(mean, std)
            a = dist.sample()
            logp = dist.log_prob(a).sum(dim=-1, keepdim=True)
            ent = dist.entropy().sum(dim=-1, keepdim=True)

            v_all = self._call_value_ensemble(lid, h_cpl[:, lid, :], z)  # [B,N]
            v_mean = v_all.mean(dim=-1, keepdim=True)

            actions.append(a)
            logps.append(logp)
            ents.append(ent)
            vmeans.append(v_mean)

        actions = torch.stack(actions, dim=1)  # [B,L,A]
        logps = torch.stack(logps, dim=1)      # [B,L,1]
        ents = torch.stack(ents, dim=1)        # [B,L,1]
        vmeans = torch.stack(vmeans, dim=1)    # [B,L,1]

        return actions, logps, ents, vmeans, q, h_fast_new

    # ------------------------- learn (called by run_bhera.py) -------------------------

    def learn_joint(
        self,
        obs_raw,        # [B,T,L,obs_dim]
        actions,        # [B,T,L,act_dim]
        values_old,     # [B,T,L,1]
        returns,        # [B,T,L,1]
        log_probs_old,  # [B,T,L,1]
        advantages,     # [B,T,L,1]
        rewards,        # [B,T,L,1]
        dones,          # [B,T,L,1]
        current_steps,
        lr=None,
    ):
        obs_raw = obs_raw.to(self.device)
        actions = actions.to(self.device)
        values_old = values_old.to(self.device)
        returns = returns.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        advantages = advantages.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        B, T, L, _ = obs_raw.shape

        last_info = {}

        for _ in range(max(1, self.inner_k)):
            # 1) rebuild joint latent (includes coupling)
            h_cpl, x_pool, a_pool, z_seq, q_seq, r_global, kl_slow, kl_fast = self._rebuild_joint_latent(
                obs_raw, actions, rewards, dones
            )

            # 2) regime-balanced weights (detach q 防止 q 自己“刷权重”作弊)
            q_flat = q_seq.reshape(B * T, 1).detach()
            w_flat = (1.0 + self.lambda_q * q_flat).clamp(min=0.0)  # [B*T,1]

            # 3) decoder target: n-step return on global reward
            target_g = compute_nstep_returns(r_global, gamma=self.gamma, n_step=self.n_step)  # [B,T,1]
            pred_g = self._call_decode_return(
                x_pool.reshape(B * T, -1),
                a_pool.reshape(B * T, -1),
                z_seq.reshape(B * T, -1),
            )  # [B*T,1]

            tgt = target_g.reshape(B * T, 1)
            if self.decoder_sparse_weight > 0:
                q_flat = q_seq.reshape(B * T, 1).detach()
                w_dec = 1.0 + self.decoder_sparse_weight * q_flat
                decoder_loss = (F.smooth_l1_loss(pred_g, tgt, reduction="none") * w_dec).mean()
            else:
                decoder_loss = F.smooth_l1_loss(pred_g, tgt)

            kl_total = kl_slow + kl_fast  # tensor scalar
            # homeostasis update beta (use EMA of detached KL)
            kl_ema_val = self.kl_ema.update(float(kl_total.detach().cpu().item()))
            self.beta = max(0.0, self.beta + self.beta_lr * (kl_ema_val - self.kl_capacity))

            # 4) PPO across layers (shared z_t, per-layer policy/value head)
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_head_loss = 0.0
            total_entropy = 0.0

            entropy_coef = self._entropy_coef(int(current_steps))

            for lid in range(L):
                s_l = h_cpl[:, :, lid, :].reshape(B * T, -1)
                z_flat = z_seq.reshape(B * T, -1)

                act_l = actions[:, :, lid, :].reshape(B * T, -1)
                logp_old_l = self._squeeze_last(log_probs_old[:, :, lid, :]).reshape(B * T)
                v_old_l = self._squeeze_last(values_old[:, :, lid, :]).reshape(B * T)
                ret_l = self._squeeze_last(returns[:, :, lid, :]).reshape(B * T)
                adv_l = self._squeeze_last(advantages[:, :, lid, :]).reshape(B * T)

                if self.norm_adv:
                    adv_l = (adv_l - adv_l.mean()) / (adv_l.std() + 1e-8)

                # regime-balanced weight apply on advantage (weight detached)
                adv_l = adv_l * w_flat.reshape(B * T)

                mean, std = self._call_forward_actor(lid, s_l, z_flat)
                std = torch.clamp(std, min=1e-6)
                dist = Normal(mean, std)

                logp = dist.log_prob(act_l).sum(dim=-1)         # [B*T]
                entropy = dist.entropy().sum(dim=-1).mean()     # scalar

                ratio = torch.exp(logp - logp_old_l)
                surr1 = ratio * adv_l
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_l
                policy_loss = -torch.min(surr1, surr2).mean()

                # ensemble value
                v_all = self._call_value_ensemble(lid, s_l, z_flat)  # [B*T,N]
                v_mean = v_all.mean(dim=-1)  # [B*T]

                if self.use_clipped_value_loss:
                    v_pred_clipped = v_old_l + torch.clamp(v_mean - v_old_l, -self.clip_param, self.clip_param)
                    v_losses = (v_mean - ret_l) ** 2
                    v_losses_clipped = (v_pred_clipped - ret_l) ** 2
                    value_loss = 0.5 * torch.max(v_losses, v_losses_clipped).mean()
                else:
                    value_loss = (0.5 * (ret_l - v_mean) ** 2).mean()

                # 同时让每个 head 都回归 return（否则 head 容易塌成一样）
                mask = (torch.rand_like(v_all) < 0.8).float()  # 80% sample keep
                head_loss = (0.5 * ((v_all - ret_l.unsqueeze(-1)) ** 2) * mask).sum() / (mask.sum() + 1e-6)

                total_policy_loss = total_policy_loss + policy_loss
                total_value_loss = total_value_loss + value_loss
                total_head_loss = total_head_loss + head_loss
                total_entropy = total_entropy + entropy

            total_policy_loss = total_policy_loss / float(L)
            total_value_loss = total_value_loss / float(L)
            total_head_loss = total_head_loss / float(L)
            total_entropy = total_entropy / float(L)

            # total loss (joint)
            loss = (
                total_policy_loss
                + self.value_loss_coef * total_value_loss
                + total_head_loss
                - entropy_coef * total_entropy
                + self.decoder_coef * decoder_loss
                + self.beta * kl_total
            )

            if lr is not None:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = float(lr)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            last_info = {
                "total_loss": float(loss.detach().cpu().item()),
                "policy_loss": float(total_policy_loss.detach().cpu().item()),
                "value_loss": float(total_value_loss.detach().cpu().item()),
                "head_loss": float(total_head_loss.detach().cpu().item()),
                "entropy": float(total_entropy.detach().cpu().item()),
                "entropy_coef": float(entropy_coef),
                "decoder_loss": float(decoder_loss.detach().cpu().item()),
                "kl_slow": float(kl_slow.detach().cpu().item()),
                "kl_fast": float(kl_fast.detach().cpu().item()),
                "kl_total": float(kl_total.detach().cpu().item()),
                "kl_ema": float(self.kl_ema.get()),
                "beta": float(self.beta),
                "q_mean": float(q_seq.detach().mean().cpu().item()),
            }

        # optional tensorboard
        if self.writer is not None and self.global_step_ref is not None:
            gs = int(current_steps)
            for k, v in last_info.items():
                self.writer.add_scalar(f"bhera/{k}", v, gs)

        return last_info
