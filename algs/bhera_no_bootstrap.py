import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from utils.utils_bhera import EMA, compute_nstep_returns, left_pad_last_k, left_pad_sliding_k


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
            decoder_sparse_weight=1.0,  # >0：对 |r|>0 加权
            q_supervised_coef=1.0,
            gamma=0.99,
            n_step=5,
            # homeostasis
            beta_init=0.1,
            kl_capacity=1.0,
            beta_lr=1e-3,
            kl_ema_decay=0.99,
            # ===== 新增：q 条件化熵系数（探索强度调节）=====
            entropy_sparse_boost=2.0,  # alpha_sparse = alpha_dense * boost
            # ===== 新增：bootstrap 不确定性权重 =====
            u_weight_kappa=1.0,  # 越大越抑制高不确定样本
            u_weight_clip_min=0.25,
            u_weight_clip_max=4.0,
            use_bootstrap=True,  # Ablation: disable bootstrap critic (no ensemble uncertainty / head bootstrap loss)
            weight_normalize=True,
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
        self.q_supervised_coef = float(q_supervised_coef)
        self.gamma = float(gamma)
        self.n_step = int(n_step)

        self.beta = float(beta_init)
        self.kl_capacity = float(kl_capacity)
        self.beta_lr = float(beta_lr)
        self.kl_ema = EMA(decay=float(kl_ema_decay), init=0.0)

        # ===== 新增：探索强度调节 =====
        self.entropy_sparse_boost = float(entropy_sparse_boost)

        # ===== 新增：u 加权 =====
        self.u_weight_kappa = float(u_weight_kappa)
        self.u_weight_clip_min = float(u_weight_clip_min)
        self.u_weight_clip_max = float(u_weight_clip_max)
        self.weight_normalize = bool(weight_normalize)

        # ===== bootstrap critic switch =====
        self.use_bootstrap = bool(use_bootstrap)

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

    def _call_fast_step(self, token_b_d: torch.Tensor, h_prev: torch.Tensor, done_prev: torch.Tensor, z_slow=None):
        return self.model.belief_fast_step(token_b_d, h_prev, done_prev, z_slow)

    def _call_q_head(self, z_bt_d: torch.Tensor) -> torch.Tensor:
        q_logits = self.model.q_head(z_bt_d)
        q = torch.sigmoid(q_logits)
        return q

    def _call_forward_actor(self, lid: int, s_flat: torch.Tensor, z_flat: torch.Tensor):
        return self.model.forward_actor(lid, s_flat, z_flat)

    def _call_value_ensemble(self, lid: int, s_flat: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        out = self.model.get_value_ensemble(lid, s_flat, z_flat)

        # 找一个看起来像 [B*T,N] 的 tensor
        cand = None
        for item in out:
            if torch.is_tensor(item) and item.dim() >= 2:
                cand = item
                break
        if cand is None:
            raise RuntimeError("get_value_ensemble returned tuple/list but no tensor found")
        v_all = cand

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
        pred = self.model.decode_nstep_return(x_flat, a_flat, z_flat)
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
        """
        B, T, L, _ = obs_raw.shape
        device = obs_raw.device

        # 1) encode + couple (按时间逐步，维度完全对齐)
        h_list = []
        for t in range(T):
            h_t = self.model.encode_obs_stack(obs_raw[:, t])  # [B,L,D]
            h_t = self.model.couple_layers(h_t)  # [B,L,D]
            h_list.append(h_t)
        h_cpl = torch.stack(h_list, dim=1)  # [B,T,L,D]

        # 2) pool layers -> global token components
        x_pool = h_cpl.mean(dim=2)  # [B,T,D]
        a_pool = actions.mean(dim=2)  # [B,T,act_dim]
        r_global = rewards.mean(dim=2)  # [B,T,1]
        d_global = dones.max(dim=2).values  # [B,T,1]

        # 3) build transition tokens (use prev a/r/d)
        a_prev = torch.zeros_like(a_pool)
        r_prev = torch.zeros_like(r_global)
        d_prev = torch.zeros_like(d_global)
        if T > 1:
            a_prev[:, 1:] = a_pool[:, :-1]
            r_prev[:, 1:] = r_global[:, :-1]
            d_prev[:, 1:] = d_global[:, :-1]

        token = torch.cat([x_pool, a_prev, r_prev, d_prev], dim=-1)  # [B,T,token_dim]

        # 4) slow belief: sliding windows τ_{t-Ks:t}  (fix: not only last-K)
        ks = int(getattr(self.model, "slow_window_len", 16))
        token_win, pad_mask = left_pad_sliding_k(token, ks)  # [B,T,Ks,D], [B,T,Ks]
        token_win_flat = token_win.reshape(B * T, ks, -1)
        pad_flat = pad_mask.reshape(B * T, ks)

        # run slow inference efficiently: split padded vs non-padded windows
        need_pad = torch.any(pad_flat, dim=1)  # [B*T]

        mu_slow_flat = torch.zeros((B * T, self.model.z_slow_dim), device=device)
        logvar_slow_flat = torch.zeros((B * T, self.model.z_slow_dim), device=device)
        z_slow_flat = torch.zeros((B * T, self.model.z_slow_dim), device=device)

        if torch.any(~need_pad):
            mu_np, logvar_np, z_np = self._call_slow_infer(token_win_flat[~need_pad], padding_mask=None)
            mu_slow_flat[~need_pad] = mu_np
            logvar_slow_flat[~need_pad] = logvar_np
            z_slow_flat[~need_pad] = z_np

        if torch.any(need_pad):
            mu_p, logvar_p, z_p = self._call_slow_infer(token_win_flat[need_pad], padding_mask=pad_flat[need_pad])
            mu_slow_flat[need_pad] = mu_p
            logvar_slow_flat[need_pad] = logvar_p
            z_slow_flat[need_pad] = z_p

        z_slow_seq = z_slow_flat.reshape(B, T, -1)  # [B,T,Ds]
        kl_slow = self._kl_standard_normal(mu_slow_flat, logvar_slow_flat).mean()

        # 5) fast belief + conditional prior (condition on per-step z_slow_t)
        h_fast = torch.zeros((B, self.model.fast_hidden), device=device)
        z_prev = torch.zeros((B, self.model.z_fast_dim), device=device)

        kl_fast_terms = []
        z_fast_list = []

        for t in range(T):
            z_slow_t = z_slow_seq[:, t]  # [B,Ds]

            mu_p, logvar_p = self.model.fast_prior(z_prev, z_slow_t)
            h_fast, mu_f, logvar_f, z_f = self._call_fast_step(
                token[:, t], h_fast, d_prev[:, t], z_slow=z_slow_t
            )

            var_q = torch.exp(logvar_f)
            var_p = torch.exp(logvar_p)
            kl_t = 0.5 * (
                    (logvar_p - logvar_f)
                    + (var_q + (mu_f - mu_p) ** 2) / (var_p + 1e-8)
                    - 1.0
            ).sum(dim=-1)

            kl_fast_terms.append(kl_t)
            z_fast_list.append(z_f)
            z_prev = z_f.detach()

        kl_fast = torch.stack(kl_fast_terms, dim=1).mean()
        z_fast = torch.stack(z_fast_list, dim=1)  # [B,T,Df]

        z_seq = torch.cat([z_slow_seq, z_fast], dim=-1)  # [B,T,Ds+Df]
        q_seq = self.model.q_head(z_seq).sigmoid()  # [B,T,1]

        return h_cpl, x_pool, a_pool, z_seq, q_seq, r_global, kl_slow, kl_fast

    # ------------------------- sampling (optional, for agent) -------------------------

    @torch.no_grad()
    def sample_joint(
            self,
            obs_raw_bl,
            a_prev_pool,
            r_prev,
            done_prev,
            token_window_bk,
            h_fast_prev,
            padding_mask=None,
    ):
        """
        采样接口（可给 bhera_agent / run_bhera 用）：
        obs_raw_bl     : [B,L,obs_dim]
        a_prev_pool    : [B,A]
        r_prev         : [B,1]     (global)
        done_prev      : [B,1]     (global)
        token_window_bk: [B,K,token_dim]   (给 slow transformer 的窗口)
        h_fast_prev    : [B,Hf]
        padding_mask   : [B,K] bool, True=PAD (可选)
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
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        B, L, _ = obs_raw_bl.shape

        # encode & couple
        obs_flat = obs_raw_bl.reshape(B * L, -1)
        h = self.model.encode_obs(obs_flat).reshape(B, L, -1)  # [B,L,D]
        h_cpl = self._call_couple_layers(h.unsqueeze(1)).squeeze(1)  # [B,L,D]

        x_pool = h_cpl.mean(dim=1)  # [B,D]
        token_t = torch.cat([x_pool, a_prev_pool, r_prev, done_prev], dim=-1)  # [B,token_dim]

        # slow infer (Transformer) + padding mask（可选）
        if padding_mask is None:
            mu_s, logvar_s, z_s = self._call_slow_infer(token_window_bk)  # [B,Ds]
        else:
            mu_s, logvar_s, z_s = self.model.belief_slow(token_window_bk, padding_mask=padding_mask)  # [B,Ds]

        # fast infer (GRU) + reset on done
        if done_prev.dim() == 2:
            h_fast_prev = h_fast_prev * (1.0 - done_prev)
        else:
            h_fast_prev = h_fast_prev * (1.0 - done_prev.unsqueeze(-1))

        h_fast_new, mu_f, logvar_f, z_f = self._call_fast_step(token_t, h_fast_prev, done_prev, z_s)  # [B,Hf],[B,Df]...

        z = torch.cat([z_s, z_f], dim=-1)  # [B,Z]
        q = self._call_q_head(z)  # [B,1]

        # per-layer action/value
        actions, logps, ents, vmeans = [], [], [], []
        for lid in range(L):
            # 你的 forward_actor 返回的是 (mean, std)
            mean, std = self._call_forward_actor(lid, h_cpl[:, lid, :], z)  # [B,A],[B,A]
            std = torch.clamp(std, min=1e-6)
            dist = Normal(mean, std)

            a = dist.sample()  # [B,A]
            logp = dist.log_prob(a).sum(dim=-1, keepdim=True)  # [B,1]
            ent = dist.entropy().sum(dim=-1, keepdim=True)  # [B,1]

            v_all = self._call_value_ensemble(lid, h_cpl[:, lid, :], z)  # [B,N]
            if v_all.dim() == 2:
                v_mean = v_all.mean(dim=-1, keepdim=True)  # [B,1]
            else:
                # 极端兜底：如果实现返回了 [B,1] 也能跑
                v_mean = v_all

            actions.append(a)
            logps.append(logp)
            ents.append(ent)
            vmeans.append(v_mean)

        actions = torch.stack(actions, dim=1)  # [B,L,A]
        logps = torch.stack(logps, dim=1)  # [B,L,1]
        ents = torch.stack(ents, dim=1)  # [B,L,1]
        vmeans = torch.stack(vmeans, dim=1)  # [B,L,1]

        return actions, logps, ents, vmeans, q, h_fast_new

    # ------------------------- learn (called by run_bhera.py) -------------------------

    def learn_joint(
            self,
            obs_raw,  # [B,T,L,obs_dim]
            actions,  # [B,T,L,act_dim]
            values_old,  # [B,T,L,1]
            returns,  # [B,T,L,1]
            log_probs_old,  # [B,T,L,1]
            advantages,  # [B,T,L,1]
            rewards,  # [B,T,L,1]
            dones,  # [B,T,L,1]
            q_labels=None,  # [B,T,1] or [B,T]  (global sparse/dense label)
            current_steps=None,
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

            # ===== 新增：q_head 监督学习（用 env_config 的 dense/sparse label）=====
            # 只训练 q_head：把 z_seq detach，冻结 encoder / belief 路径，避免影响 PPO 表示学习。
            q_sup_loss = None
            if q_labels is not None and getattr(self, "q_supervised_coef", 0.0) > 0:
                q_lab = q_labels.to(self.device)
                if q_lab.dim() == 2:
                    q_lab = q_lab.unsqueeze(-1)  # [B,T,1]
                q_lab = q_lab.reshape(B * T, 1).float().clamp(0.0, 1.0)

                # 用 logits 做 BCE 更稳
                q_logits = self.model.q_head(z_seq.detach().reshape(B * T, -1))  # [B*T,1]
                q_sup_loss = F.binary_cross_entropy_with_logits(q_logits, q_lab)

            # 2) regime-balanced weight w(q) (detach 防止 q “刷权重”作弊)
            q_flat = q_seq.reshape(B * T).detach()  # [B*T]
            w_q = (1.0 + self.lambda_q * q_flat).clamp(min=0.0)  # [B*T]

            # 3) decoder target: n-step return on global reward
            target_g = compute_nstep_returns(r_global, gamma=self.gamma, n_step=self.n_step)  # [B,T,1]
            pred_g = self._call_decode_return(
                x_pool.reshape(B * T, -1),
                a_pool.reshape(B * T, -1),
                z_seq.reshape(B * T, -1),
            )  # [B*T,1]

            tgt = target_g.reshape(B * T, 1)
            if self.decoder_sparse_weight > 0:
                q_dec = q_seq.reshape(B * T, 1).detach()
                w_sp = 1.0 + self.decoder_sparse_weight * q_dec
                decoder_loss = (F.smooth_l1_loss(pred_g, tgt, reduction="none") * w_sp).mean()
            else:
                decoder_loss = F.smooth_l1_loss(pred_g, tgt)

            kl_total = kl_slow + kl_fast
            kl_ema_val = self.kl_ema.update(float(kl_total.detach().cpu().item()))
            self.beta = max(0.0, self.beta + self.beta_lr * (kl_ema_val - self.kl_capacity))

            # ===== 探索强度调节：alpha = entropy_coef(q) =====
            base_entropy_coef = float(self._entropy_coef(int(current_steps)))
            q_mean = float(q_flat.mean().cpu().item())
            entropy_coef = base_entropy_coef * ((1.0 - q_mean) + q_mean * self.entropy_sparse_boost)
            entropy_coef = max(1e-3, float(entropy_coef))

            # 4) PPO across layers
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_head_loss = 0.0
            total_entropy = 0.0

            total_u_mean = 0.0
            total_wu_mean = 0.0
            total_w_mean = 0.0

            z_flat = z_seq.reshape(B * T, -1)

            for lid in range(L):
                s_l = h_cpl[:, :, lid, :].reshape(B * T, -1)

                act_l = actions[:, :, lid, :].reshape(B * T, -1)
                logp_old_l = self._squeeze_last(log_probs_old[:, :, lid, :]).reshape(B * T)
                v_old_l = self._squeeze_last(values_old[:, :, lid, :]).reshape(B * T)
                ret_l = self._squeeze_last(returns[:, :, lid, :]).reshape(B * T)
                adv_l = self._squeeze_last(advantages[:, :, lid, :]).reshape(B * T)

                if self.norm_adv:
                    adv_l = (adv_l - adv_l.mean()) / (adv_l.std() + 1e-8)

                mean, std = self._call_forward_actor(lid, s_l, z_flat)
                std = torch.clamp(std, min=1e-6)
                dist = Normal(mean, std)

                logp = dist.log_prob(act_l).sum(dim=-1)  # [B*T]
                entropy = dist.entropy().sum(dim=-1).mean()  # scalar

                ratio = torch.exp(logp - logp_old_l)
                surr1 = ratio * adv_l
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_l
                surr = torch.min(surr1, surr2)  # [B*T]

                v_all = self._call_value_ensemble(lid, s_l, z_flat)  # [B*T,N]
                v_mean = v_all.mean(dim=-1)  # [B*T]

                if self.use_bootstrap:
                    u = v_all.var(dim=-1, unbiased=False).detach()  # [B*T]
                    u_mean = u.mean().clamp(min=1e-8)
                    u_norm = u / u_mean  # [B*T]
                    w_u = 1.0 / (1.0 + self.u_weight_kappa * u_norm)  # [B*T]
                    w_u = w_u.clamp(min=self.u_weight_clip_min, max=self.u_weight_clip_max)
                else:
                    # Without bootstrap: do not use ensemble uncertainty for weighting
                    u = torch.zeros_like(v_mean).detach()
                    u_mean = torch.tensor(0.0, device=v_mean.device)
                    w_u = torch.ones_like(v_mean).detach()

                w = (w_q * w_u).clamp(min=0.0)  # [B*T]
                if self.weight_normalize:
                    w = w / (w.mean().detach() + 1e-8)

                policy_loss = -(w * surr).mean()

                if self.use_clipped_value_loss:
                    v_pred_clipped = v_old_l + torch.clamp(v_mean - v_old_l, -self.clip_param, self.clip_param)
                    v_losses = (v_mean - ret_l) ** 2
                    v_losses_clipped = (v_pred_clipped - ret_l) ** 2
                    value_loss = 0.5 * (w * torch.max(v_losses, v_losses_clipped)).mean()
                else:
                    value_loss = 0.5 * (w * (ret_l - v_mean) ** 2).mean()

                if self.use_bootstrap:
                    # Bootstrap critic heads: random mask per-sample/head
                    mask = (torch.rand_like(v_all) < 0.8).float()
                    head_loss = (0.5 * ((v_all - ret_l.unsqueeze(-1)) ** 2) * mask).sum() / (mask.sum() + 1e-6)
                else:
                    # keep type as tensor for logging (.detach())
                    head_loss = torch.zeros((), device=v_mean.device)

                total_policy_loss = total_policy_loss + policy_loss
                total_value_loss = total_value_loss + value_loss
                total_head_loss = total_head_loss + head_loss
                total_entropy = total_entropy + entropy

                total_u_mean += float(u_mean.cpu().item())
                total_wu_mean += float(w_u.mean().cpu().item())
                total_w_mean += float(w.mean().detach().cpu().item())

            total_policy_loss = total_policy_loss / float(L)
            total_value_loss = total_value_loss / float(L)
            total_head_loss = total_head_loss / float(L)
            total_entropy = total_entropy / float(L)

            avg_u_mean = total_u_mean / float(L)
            avg_wu_mean = total_wu_mean / float(L)
            avg_w_mean = total_w_mean / float(L)

            loss = (
                    total_policy_loss
                    + self.value_loss_coef * total_value_loss
                    + total_head_loss
                    + (self.q_supervised_coef * q_sup_loss if q_sup_loss is not None else 0.0)
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
                "entropy_coef_base": float(base_entropy_coef),
                "decoder_loss": float(decoder_loss.detach().cpu().item()),
                "kl_slow": float(kl_slow.detach().cpu().item()),
                "kl_fast": float(kl_fast.detach().cpu().item()),
                "kl_total": float(kl_total.detach().cpu().item()),
                "kl_ema": float(self.kl_ema.get()),
                "beta": float(self.beta),
                "q_mean": float(q_mean),
                "q_sup_loss": float(q_sup_loss.detach().cpu().item()) if q_sup_loss is not None else 0.0,
                "u_mean": float(avg_u_mean),
                "w_u_mean": float(avg_wu_mean),
                "w_mean": float(avg_w_mean),
            }

        if self.writer is not None and self.global_step_ref is not None:
            gs = int(current_steps)
            for k, v in last_info.items():
                self.writer.add_scalar(f"bhera/{k}", v, gs)

        return last_info
