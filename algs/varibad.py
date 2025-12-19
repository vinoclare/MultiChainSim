# algs/varibad.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class VariBAD:
    """
    对齐你现有 MAPPO 风格的 VariBAD：
    - PPO loss（clip policy + clipped value loss + entropy anneal）
    - ELBO loss（recon next_obs + recon reward + KL(q(z|tau)||N(0,I))）
    - belief z_t 在 learn() 中按序列重建（不依赖采样时保存的 mu/logvar）
    """

    def __init__(self,
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
                 elbo_coef=0.1,
                 writer=None,
                 global_step_ref=None,
                 total_training_steps=None):
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

        self.elbo_coef = elbo_coef

        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_lr, eps=eps)

        self.writer = writer
        self.global_step_ref = global_step_ref
        self.total_training_steps = total_training_steps if total_training_steps is not None else 1

    # ------------------------- sampling (used by agent) -------------------------

    def sample(self, s_embed, a_prev, r_prev, done_prev, h_prev):
        """
        s_embed:   [B, obs_embed_dim]  (来自 model.encode_obs(raw_obs))
        a_prev:    [B, action_dim]
        r_prev:    [B, 1]
        done_prev: [B, 1]
        h_prev:    [B, belief_hidden]
        return:
          action [B, action_dim],
          logp   [B, 1],
          entropy[B, 1],
          value  [B, 1],
          h_new, z, mu, logvar
        """
        s_embed = s_embed.to(self.device)
        a_prev = a_prev.to(self.device)
        r_prev = r_prev.to(self.device)
        done_prev = done_prev.to(self.device)
        h_prev = h_prev.to(self.device)

        # 如果上一刻已经 done，则重置 hidden（更鲁棒）
        if done_prev.dim() == 2:
            h_prev = h_prev * (1.0 - done_prev)
        else:
            h_prev = h_prev * (1.0 - done_prev.unsqueeze(-1))

        h_new, mu, logvar, z = self.model.belief_step(s_embed, a_prev, r_prev, done_prev, h_prev)
        action, logp, entropy = self.model.act(s_embed, z)
        value = self.model.get_value(s_embed, z)
        return action, logp, entropy, value, h_new, z, mu, logvar

    # ------------------------- helpers for learn() -------------------------

    @staticmethod
    def _kl_standard_normal(mu, logvar):
        # KL(q||p), p=N(0,I): 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)

    def _rebuild_belief_sequence(self, obs_raw, actions, rewards, dones):
        """
        用序列数据重建每个 t 的 (s_embed_t, z_t, mu_t, logvar_t)
        obs_raw:  [B, T, obs_dim]
        actions:  [B, T, action_dim]
        rewards:  [B, T, 1]
        dones:    [B, T, 1]
        """
        B, T, _ = obs_raw.shape

        # encode obs all at once
        obs_flat = obs_raw.reshape(B * T, -1)
        s_embed_flat = self.model.encode_obs(obs_flat)  # [B*T, obs_embed_dim]
        s_embed = s_embed_flat.view(B, T, -1)

        # init recurrent state and prev feedback
        h = torch.zeros((B, self.model.belief_hidden), dtype=torch.float32, device=self.device)
        a_prev = torch.zeros((B, self.model.action_dim), dtype=torch.float32, device=self.device)
        r_prev = torch.zeros((B, 1), dtype=torch.float32, device=self.device)
        d_prev = torch.zeros((B, 1), dtype=torch.float32, device=self.device)

        z_list, mu_list, logvar_list = [], [], []

        for t in range(T):
            # 若上一刻 done，清空 hidden（兼容不同长度 episode 被裁剪进来）
            h = h * (1.0 - d_prev)

            h, mu, logvar, z = self.model.belief_step(s_embed[:, t], a_prev, r_prev, d_prev, h)

            z_list.append(z)
            mu_list.append(mu)
            logvar_list.append(logvar)

            # 更新 prev 为本步 transition 的信息，供下一步 belief 用
            a_prev = actions[:, t]
            r_prev = rewards[:, t]
            d_prev = dones[:, t]

        z_seq = torch.stack(z_list, dim=1)        # [B, T, z_dim]
        mu_seq = torch.stack(mu_list, dim=1)      # [B, T, z_dim]
        logvar_seq = torch.stack(logvar_list, dim=1)  # [B, T, z_dim]
        return s_embed, z_seq, mu_seq, logvar_seq

    # ------------------------- learn (called by run_varibad.py) -------------------------

    def learn(self,
              obs_raw,          # [B,T,obs_dim]
              next_obs_raw,     # [B,T,obs_dim]
              actions,          # [B,T,action_dim]
              values_old,       # [B,T,1]
              returns,          # [B,T,1]
              log_probs_old,    # [B,T,1]
              advantages,       # [B,T,1]
              rewards,          # [B,T,1]
              dones,            # [B,T,1]
              current_steps,
              lr=None):

        obs_raw = obs_raw.to(self.device)
        next_obs_raw = next_obs_raw.to(self.device)
        actions = actions.to(self.device)
        values_old = values_old.to(self.device)
        returns = returns.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        advantages = advantages.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        B, T, _ = obs_raw.shape

        # 1) 重建 belief 序列（得到 z_t / mu/logvar）
        s_embed, z_seq, mu_seq, logvar_seq = self._rebuild_belief_sequence(
            obs_raw, actions, rewards, dones
        )

        # flatten for PPO computation
        s_flat = s_embed.reshape(B * T, -1)
        z_flat = z_seq.reshape(B * T, -1)
        act_flat = actions.reshape(B * T, -1)
        v_old_flat = values_old.reshape(B * T)
        ret_flat = returns.reshape(B * T)
        logp_old_flat = log_probs_old.reshape(B * T)
        adv_flat = advantages.reshape(B * T)

        # PPO 内层多步更新（对齐 mappo.py 的 inner_k）
        last_info = {}

        for _ in range(self.inner_k):
            mean, std = self.model.forward_actor(s_flat, z_flat)
            dist = Normal(mean, std)

            logp = dist.log_prob(act_flat).sum(dim=-1)   # [B*T]
            entropy = dist.entropy().sum(dim=-1).mean()  # scalar

            if self.norm_adv:
                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

            ratio = torch.exp(logp - logp_old_flat)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_flat
            action_loss = -torch.min(surr1, surr2).mean()

            values = self.model.get_value(s_flat, z_flat).view(-1)  # [B*T]
            if self.use_clipped_value_loss:
                value_pred_clipped = v_old_flat + torch.clamp(values - v_old_flat,
                                                             -self.clip_param, self.clip_param)
                value_losses = (values - ret_flat) ** 2
                value_losses_clipped = (value_pred_clipped - ret_flat) ** 2
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (0.5 * (ret_flat - values) ** 2).mean()

            # entropy coef 退火（对齐 mappo.py）
            progress = float(current_steps) / float(self.total_training_steps)
            entropy_coef = max(1e-3, self.initial_entropy_coef * (1.0 - progress))

            # 2) ELBO loss
            # recon
            next_pred, rew_pred = self.model.decode(s_flat, act_flat, z_flat)
            recon_obs_loss = F.mse_loss(next_pred, next_obs_raw.reshape(B * T, -1))
            recon_rew_loss = F.mse_loss(rew_pred.view(-1), rewards.reshape(B * T))
            recon_loss = recon_obs_loss + recon_rew_loss

            # KL
            kl_per = self._kl_standard_normal(
                mu_seq.reshape(B * T, -1),
                logvar_seq.reshape(B * T, -1),
            )  # [B*T]
            kl_loss = kl_per.mean()

            elbo_loss = recon_loss + kl_loss

            # total
            loss = value_loss * self.value_loss_coef + action_loss - entropy_coef * entropy + self.elbo_coef * elbo_loss

            if lr:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            last_info = {
                "total_loss": float(loss.detach().cpu().item()),
                "policy_loss": float(action_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "entropy": float(entropy.detach().cpu().item()),
                "entropy_coef": float(entropy_coef),
                "recon_loss": float(recon_loss.detach().cpu().item()),
                "kl_loss": float(kl_loss.detach().cpu().item()),
                "elbo_loss": float(elbo_loss.detach().cpu().item()),
            }

        return last_info
