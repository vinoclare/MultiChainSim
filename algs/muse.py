import torch
import torch.nn as nn
from torch.distributions import Normal

from models.agent57_model import Agent57IndustrialModel


class MuSE(nn.Module):
    """
    多子策略管理器（共享一个 Agent57IndustrialModel，多个策略 head）
    """

    def __init__(self, cfg, distill_cfg, obs_shapes, device="cuda",
                 writer=None, total_training_steps=None):
        super().__init__()
        self.K = cfg["K"]
        self.lambda_div = cfg["lambda_div"]
        self.neg_policy = distill_cfg["neg_policy"]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.writer = writer
        self.total_training_steps = total_training_steps

        # === 单个多头模型 ===
        self.model = Agent57IndustrialModel(
            task_input_dim=obs_shapes["task"],
            worker_load_input_dim=obs_shapes["worker_load"],
            worker_profile_input_dim=obs_shapes["worker_profile"],
            n_worker=obs_shapes["n_worker"],
            num_pad_tasks=obs_shapes["num_pad_tasks"],
            global_context_dim=obs_shapes["global_context_dim"],
            hidden_dim=cfg["hidden_dim"],
            K=self.K
        ).to(self.device)

        # with torch.no_grad():
        #     self.model.log_stds.fill_(-0.5)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["initial_lr"])

        # α,β 系数
        self.alphas = torch.tensor(cfg["alphas"], device=self.device)
        self.betas = torch.tensor(cfg["betas"], device=self.device)

        self.value_loss_coef = cfg["value_loss_coef"]
        self.entropy_coef = cfg["entropy_coef"]

        self.adv_norm = cfg["advantage_normalization"]

        # ---------- 采样 ---------- #
    @torch.no_grad()
    def sample(self, *model_inputs):
        """
        model_inputs: 对应模型 forward 的输入
        """
        mean, std, v_u, v_c = self.model(*model_inputs)
        dist = Normal(mean, std)
        action = dist.rsample().clamp(0, 1)
        logp = dist.log_prob(action).sum(dim=[1, 2])
        entropy = dist.entropy().mean(dim=[1, 2])
        action_std = action.std(dim=[1, 2])
        return v_u, v_c, action, logp, entropy, action_std

    # ---------- 训练 ---------- #
    def learn(
            self,
            pid: torch.Tensor,
            task_obs,
            worker_loads,
            worker_profile,
            global_context,
            valid_mask,
            actions,
            returns,         # Tuple: (ret_u, ret_c)
            log_probs_old,
            advantages,
            step
    ):
        k = pid[0].item()

        return self._ppo_update_single_head(
            head_id=k,
            task_obs=task_obs,
            worker_loads=worker_loads,
            worker_profile=worker_profile,
            global_context=global_context,
            valid_mask=valid_mask,
            actions=actions,
            returns=returns,
            log_probs_old=log_probs_old,
            advantages=advantages,
            step=step
        )

    def _ppo_update_single_head(
            self,
            head_id,
            task_obs,
            worker_loads,
            worker_profile,
            global_context,
            valid_mask,
            actions,
            returns,
            log_probs_old,
            advantages,
            step
    ):
        ret_u, ret_c = returns

        mean, std, v_u, v_c = self.model(
            task_obs, worker_loads, worker_profile,
            global_context, valid_mask,
            policy_id=head_id
        )

        dist = Normal(mean, std)
        logp = dist.log_prob(actions).sum(dim=[1, 2])
        entropy = dist.entropy().sum(dim=[1, 2])
        ratio = torch.exp(logp - log_probs_old)

        adv = advantages.clone()
        if self.adv_norm:
            adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv
        pi_loss = -torch.min(surr1, surr2).mean()

        # Value loss（双头）
        value_loss_u = 0.5 * (ret_u - v_u).pow(2).mean()
        value_loss_c = 0.5 * (ret_c - v_c).pow(2).mean()
        value_loss = value_loss_u + value_loss_c

        entropy_loss = -entropy.mean()

        # diversity loss
        if self.neg_policy and head_id > (self.K - 3):
            is_pos_policy = False
        elif self.neg_policy:
            is_pos_policy = True
            num_pos = self.K - 2
        else:
            is_pos_policy = True
            num_pos = self.K

        if is_pos_policy and self.lambda_div > 0:
            mean_j = mean.detach()  # detach 不传梯度给自己
            std_j = std.detach()

            kl_terms = []
            for k in range(num_pos):
                if k == head_id:
                    continue
                with torch.no_grad():
                    mean_k, std_k, *_ = self.model(
                        task_obs, worker_loads, worker_profile,
                        global_context, valid_mask,
                        policy_id=k
                    )
                # KL(π_j || π_k)，高斯 closed-form
                kl = (std_k.log() - std_j.log() +
                      (std_j.pow(2) + (mean_j - mean_k).pow(2)) / (2 * std_k.pow(2)) - 0.5)
                kl_sum = kl.sum(dim=[1, 2])  # 对动作维度求和
                kl_terms.append(kl_sum)
                div_loss = self.lambda_div * torch.stack(kl_terms).mean()
        else:
            div_loss = 0.0

        # entropy_coef decay
        progress = step / self.total_training_steps
        entropy_coef = max(1e-3, self.entropy_coef * (1 - progress))

        loss = pi_loss + self.value_loss_coef * value_loss + entropy_coef * entropy_loss + div_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        return value_loss.item(), pi_loss.item(), entropy.mean().item()
