import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

from models.agent57_model import Agent57IndustrialModel
from models.muse_model import MuseModel


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
        self.model = MuseModel(
            task_input_dim=obs_shapes["task"],
            worker_load_input_dim=obs_shapes["worker_load"],
            worker_profile_input_dim=obs_shapes["worker_profile"],
            n_worker=obs_shapes["n_worker"],
            num_pad_tasks=obs_shapes["num_pad_tasks"],
            global_context_dim=obs_shapes["global_context_dim"],
            hidden_dim=cfg["hidden_dim"],
            K=self.K,
            neg_policy=self.neg_policy
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


class MuSE2(nn.Module):
    def __init__(self, cfg, distill_cfg, alphas, betas, obs_shapes, device="cuda", writer=None, total_training_steps=None):
        super().__init__()
        self.K = cfg["K"]
        self.K_total = self.K + 1
        self.lambda_div = cfg["lambda_div"]
        self.neg_policy = distill_cfg["neg_policy"]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.writer = writer
        self.total_training_steps = total_training_steps

        # === 使用 Agent57IndustrialModel，注意 K 改为 K+1 ===
        self.model = Agent57IndustrialModel(
            task_input_dim=obs_shapes["task"],
            worker_load_input_dim=obs_shapes["worker_load"],
            worker_profile_input_dim=obs_shapes["worker_profile"],
            n_worker=obs_shapes["n_worker"],
            num_pad_tasks=obs_shapes["num_pad_tasks"],
            global_context_dim=obs_shapes["global_context_dim"],
            hidden_dim=cfg["hidden_dim"],
            K=self.K_total,
            neg_policy=self.neg_policy
        ).to(self.device)

        # === 优化器 ===
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["initial_lr"])

        # α, β 系数
        self.alphas = alphas
        self.betas = betas

        self.value_loss_coef = cfg["value_loss_coef"]
        self.entropy_coef = cfg["entropy_coef"]
        self.adv_norm = cfg["advantage_normalization"]

        # === Distiller 相关超参数 ===
        self.sup_coef = distill_cfg.get("sup_coef", 1.0)
        self.neg_coef = distill_cfg.get("neg_coef", 1.0)
        self.margin = distill_cfg.get("margin", 3.0)
        self.var_coef = distill_cfg.get("var_coef", 0.005)
        self.sigma_target = distill_cfg.get("sigma_target", 0.8)
        self.std_t = distill_cfg.get("std_t", 0.2)
        self.loss_type = distill_cfg.get("loss_type", "mse")

        self.pos_pid_max = self.K - 3 if self.neg_policy else self.K

        buffer_size = distill_cfg.get("buffer_size", 10000)
        self.buffers = {pid: deque(maxlen=buffer_size) for pid in range(self.K)}

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
            returns,  # Tuple: (ret_u, ret_c)
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
        if head_id == self.K:  # 主策略，不计算 diversity loss
            div_loss = 0.0
        elif self.neg_policy and head_id > (self.K - 3):  # 负策略，不计算 diversity loss
            div_loss = 0.0
        else:  # 正策略
            if self.neg_policy:
                num_pos = self.K - 2  # 正策略数量（不含主策略，不含两个负策略）
            else:
                num_pos = self.K  # 正策略数量（不含主策略）

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
                kl = (std_k.log() - std.log() +
                      (std.pow(2) + (mean - mean_k).pow(2)) / (2 * std_k.pow(2)) - 0.5)
                kl_sum = kl.sum(dim=[1, 2])
                kl_terms.append(kl_sum)

            if len(kl_terms) > 0:
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

    def collect(self, obs_dict, actions, pid):
        """
        收集一批数据，obs_dict 包含 task_obs、worker_loads、worker_profiles、global_context、valid_mask
        actions: [B, n_worker, num_pad_tasks]
        pid: [B]
        """
        obs_cpu = {k: v.detach().cpu() for k, v in obs_dict.items()}
        for i in range(actions.shape[0]):
            cur_pid = pid[i]
            self.buffers[cur_pid].append({
                "obs": {k: obs_cpu[k][i] for k in obs_cpu},
                "action": actions[i].detach().cpu()
            })

    def _sample_from_buffers(self, cur_pid: int, batch_size: int):
        buf = self.buffers[cur_pid]
        if len(buf) < batch_size:
            return random.sample(buf, len(buf))
        return random.sample(buf, batch_size)

    def bc_update(self, cur_pid: int, batch_size: int = 512, steps: int = 50):
        """
        对主策略 (head_id = K) 执行行为克隆 / 蒸馏更新。
        cur_pid: 被蒸馏的子策略 id（即 buffer 来源的 pid）
        """
        if len(self.buffers[cur_pid]) < batch_size:
            return 0.0  # 数据不足

        # ===== 冻结 encoder 主干 =====
        for name, param in self.model.named_parameters():
            if (
                    "task_enc" in name or
                    "worker_load_enc" in name or
                    "worker_profile_enc" in name or
                    "fc_worker" in name or
                    "fc_task" in name or
                    "global_fc" in name or
                    "fusion_norm" in name
            ):
                param.requires_grad = False

        total_loss = 0.0
        for _ in range(steps):
            batch = self._sample_from_buffers(cur_pid, batch_size)
            if len(batch) == 0:
                break

            # ---- 构造张量 ----
            obs_keys = ["task_obs", "worker_loads", "worker_profiles", "global_context", "valid_mask"]
            obs_t = {k: torch.stack([torch.tensor(d["obs"][k]) for d in batch]).to(self.device)
                     for k in obs_keys}
            target_actions = torch.stack([torch.tensor(d["action"]) for d in batch]).to(self.device)

            # ---- 主策略 head 前向（注意用 self.K） ----
            mean_s, std_s, _, _ = self.model(
                obs_t["task_obs"], obs_t["worker_loads"], obs_t["worker_profiles"],
                obs_t["global_context"], obs_t["valid_mask"],
                policy_id=self.K
            )

            if self.neg_policy:
                is_pos_policy = cur_pid <= self.pos_pid_max
            else:
                is_pos_policy = True

            if is_pos_policy:
                if self.loss_type == "mse":
                    loss_pos = F.mse_loss(mean_s, target_actions)
                    loss = self.sup_coef * loss_pos
                else:  # "kl"
                    teacher_std = torch.full_like(mean_s, self.std_t)
                    dist_t = Normal(target_actions, teacher_std)
                    dist_s = Normal(mean_s, std_s.clamp(min=1e-3))
                    loss_kl = torch.distributions.kl_divergence(dist_t, dist_s).mean()
                    loss = self.sup_coef * loss_kl
            else:
                dist_s = Normal(mean_s, std_s.clamp(min=1e-3))
                logp_neg = dist_s.log_prob(target_actions).sum(-1)
                loss_sp = F.softplus(logp_neg + self.margin).mean()
                loss_var = (std_s - self.sigma_target).clamp(min=0).pow(2).mean()
                loss = self.neg_coef * loss_sp + self.var_coef * loss_var

            # ---- 反向更新 ----
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        # ===== 恢复共享 encoder 主干 =====
        for name, param in self.model.named_parameters():
            if (
                    "task_enc" in name or
                    "worker_load_enc" in name or
                    "worker_profile_enc" in name or
                    "fc_worker" in name or
                    "fc_task" in name or
                    "global_fc" in name or
                    "fusion_norm" in name
            ):
                param.requires_grad = True
        return total_loss / steps
