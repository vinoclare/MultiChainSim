import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

from models.agent57_model import Agent57IndustrialModel


class MultiStrategyAgent:
    def __init__(self,
                 task_input_dim: int,
                 worker_load_input_dim: int,
                 worker_profile_input_dim: int,
                 n_worker: int,
                 num_pad_tasks: int,
                 global_context_dim: int,
                 hidden_dim: int,
                 betas: list,
                 gammas: list,
                 clip_param: float,
                 value_loss_coef: float,
                 entropy_coef: float,
                 initial_lr: float,
                 max_grad_norm: float,
                 lam: float,
                 device: torch.device = torch.device('cpu'),):
        """
        Args:
            betas:       List[float], beta_k 列表，长度 K
            gammas:      List[float], γ_k 列表，长度 K
            其余参数类似 PPO + 模型初始化所需
        """
        self.device = device
        assert len(betas) == len(gammas), "长度必须一致"
        self.K = len(betas)
        self.betas = betas
        self.gammas = gammas
        self.lam = lam

        # —— 初始化共享多头模型 —— #
        self.model = Agent57IndustrialModel(
            task_input_dim=task_input_dim,
            worker_load_input_dim=worker_load_input_dim,
            worker_profile_input_dim=worker_profile_input_dim,
            n_worker=n_worker,
            num_pad_tasks=num_pad_tasks,
            global_context_dim=global_context_dim,
            hidden_dim=hidden_dim,
            K=self.K
        ).to(self.device)

        # —— PPO 优化器（单个）—— #
        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_lr, eps=1e-5)
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.initial_entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # 占位：按步骤更新 entropy_coef（可选）
        self.entropy_coef = entropy_coef
        self.global_step = 0

    def sample(self,
               task_obs: torch.Tensor,
               worker_loads: torch.Tensor,
               worker_profiles: torch.Tensor,
               global_context: torch.Tensor,
               valid_mask: torch.Tensor,
               policy_id: int):
        """
        根据 policy_id，调用模型前向得到 mean/std 和 v_u/v_c，
        然后采样 action、log_prob 与 entropy。

        Returns:
            v_u, v_c:      Tensor (B,)
            action:        Tensor (B, W, T)
            log_prob:      Tensor (B,)
            entropy:       Tensor (B,)
        """
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profiles = worker_profiles.to(self.device)
        global_context = global_context.to(self.device)
        valid_mask = valid_mask.to(self.device)

        with torch.no_grad():
            mean, std, v_u, v_c = self.model(
                task_obs, worker_loads, worker_profiles, global_context, valid_mask, policy_id
            )
            dist = Normal(mean, std)
            action = dist.sample().clamp(0, 1)
            log_prob = dist.log_prob(action).sum(dim=[1, 2])
            entropy = dist.entropy().sum(dim=[1, 2])

        return v_u, v_c, action, log_prob, entropy

    def learn(self,
              task_obs_batch: torch.Tensor,
              worker_loads_batch: torch.Tensor,
              worker_profiles_batch: torch.Tensor,
              global_context_batch: torch.Tensor,
              valid_mask_batch: torch.Tensor,
              actions_batch: torch.Tensor,
              values_u_old: torch.Tensor,
              values_c_old: torch.Tensor,
              returns_u: torch.Tensor,
              returns_c: torch.Tensor,
              log_probs_old: torch.Tensor,
              policy_id: int,
              global_steps: int,
              total_training_steps: int):
        """
        对应单个子策略 policy_id 的小批量数据，计算 Advantage 并执行 PPO 更新：
            A_u = GAE(rewards_u, values_u_old)
            A_c = GAE(-rewards_c, -values_c_old)
            A = beta * A_u + (1-beta) * (-A_c)
            R = beta * returns_u + (1-beta) * (-returns_c)

        然后对 actor 和双价值头计算 PPO 损失。
        """
        beta = self.betas[policy_id]

        # —— 转移到设备 —— #
        task_obs = task_obs_batch.to(self.device)
        worker_loads = worker_loads_batch.to(self.device)
        worker_profiles = worker_profiles_batch.to(self.device)
        global_context = global_context_batch.to(self.device)
        valid_mask = valid_mask_batch.to(self.device)
        actions = actions_batch.to(self.device)
        values_u_old = values_u_old.to(self.device)
        values_c_old = values_c_old.to(self.device)
        returns_u = returns_u.to(self.device)
        returns_c = returns_c.to(self.device)
        log_probs_old = log_probs_old.to(self.device)

        # —— 重新前向，得到新的 mean/std 与 v_u / v_c —— #
        mean, std, v_u_new, v_c_new = self.model(
            task_obs, worker_loads, worker_profiles, global_context, valid_mask, policy_id
        )
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=[1, 2])
        entropy = dist.entropy().sum(dim=[1, 2]).mean()

        # —— 计算 Advantage —— #
        adv = returns_u - values_u_old

        # —— 标准化 Advantage （PPO 推荐） —— #
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # —— 计算 ratio 与策略损失 —— #
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # —— 计算价值损失 —— #
        # 新预测的值
        # 注意：我们希望成本价值也变成“肯定向”序列：用 -v_c_new 作为价值预测
        value_loss_u = 0.5 * (v_u_new - returns_u).pow(2).mean()
        value_loss_c = 0.5 * (v_c_new - returns_c).pow(2).mean()
        value_loss = value_loss_u + value_loss_c

        # entropy_coef decay
        progress = global_steps / total_training_steps
        entropy_coef = max(1e-3, self.entropy_coef * (1 - progress))

        # —— 总损失 —— #
        loss = value_loss * self.value_loss_coef + policy_loss - entropy_coef * entropy

        # —— 优化 —— #
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1

        return policy_loss.item(), value_loss.item(), entropy.item()
