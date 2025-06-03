import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class PPO:
    def __init__(self,
                 model,
                 clip_param=0.1,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 initial_lr=2.5e-4,
                 eps=1e-5,
                 max_grad_norm=0.5,
                 device="cuda",
                 use_clipped_value_loss=True,
                 norm_adv=True,
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_lr, eps=eps)

        self.writer = writer
        self.global_step_ref = global_step_ref
        self.total_training_steps = total_training_steps

        self.train_log_interval = 1000
        self._clip_frac_buffer = []
        self._v_pred_error_buffer = []
        self._action_mean_buffer = []
        self._value_loss_buffer = []
        self._policy_loss_buffer = []
        self._entropy_buffer = []

    def sample(self, task_obs, worker_loads, worker_profile, global_context, valid_mask):
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        global_context = global_context.to(self.device)

        mean, std, value = self.model(task_obs, worker_loads, worker_profile, global_context, valid_mask)
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, 0, 1)
        action_log_probs = dist.log_prob(action).sum(dim=[1, 2])
        entropy = dist.entropy().sum(dim=[1, 2])

        return value, action, action_log_probs, entropy

    def predict(self, task_obs, worker_loads, worker_profile, global_context, valid_mask):
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        global_context = global_context.to(self.device)

        mean, _, _ = self.model(task_obs, worker_loads, worker_profile, global_context, valid_mask)
        return mean  # deterministic policy

    def value(self, task_obs, worker_loads):
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        _, _, value = self.model(task_obs, worker_loads)
        return value

    def learn(self,
              task_obs,
              worker_loads,
              worker_profile,
              global_context,
              valid_mask,
              actions,
              values_old,
              returns,
              log_probs_old,
              advantages,
              lr=None):
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        global_context = global_context.to(self.device)
        valid_mask = valid_mask.to(self.device)
        actions = actions.to(self.device)
        values_old = values_old.to(self.device)
        returns = returns.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        advantages = advantages.to(self.device)

        mean, std, values = self.model(
            task_obs, worker_loads, worker_profile, global_context, valid_mask)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=[1, 2])
        entropy = dist.entropy().sum(dim=[1, 2]).mean()

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        action_loss = -torch.min(surr1, surr2).mean()

        values = values.view(-1)
        if self.use_clipped_value_loss:
            value_pred_clipped = values_old + torch.clamp(
                values - values_old, -self.clip_param, self.clip_param)
            value_losses = (values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (0.5 * (returns - values) ** 2).mean()

        # entropy_coef decay
        progress = self.global_step_ref[0] / self.total_training_steps
        entropy_coef = max(1e-3, self.initial_entropy_coef * (1 - progress))

        loss = value_loss * self.value_loss_coef + action_loss - entropy_coef * entropy

        if lr:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # === TensorBoard Logging ===
        clip_mask = (ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)
        clip_fraction = clip_mask.float().mean().item()
        v_pred_error = (values.detach() - returns).abs().mean().item()

        self._clip_frac_buffer.append(clip_fraction)
        self._v_pred_error_buffer.append(v_pred_error)
        self._action_mean_buffer.append(actions.mean().item())
        self._value_loss_buffer.append(value_loss.item())
        self._policy_loss_buffer.append(action_loss.item())
        self._entropy_buffer.append(entropy.item())

        # === Debug Logging for Diagnosis ===
        # if self.writer is not None and self.global_step_ref is not None:
        #     step = self.global_step_ref[0]
        #     self.global_step_ref[0] += 1
        #
        #     self.writer.add_scalar("debug/reward_mean", returns.mean().item(), step)
        #     self.writer.add_scalar("debug/reward_std", returns.std().item(), step)
        #
        #     self.writer.add_scalar("debug/value_pred_mean", values.mean().item(), step)
        #     self.writer.add_scalar("debug/value_pred_std", values.std().item(), step)
        #     self.writer.add_scalar("debug/value_abs_error", (values - returns).abs().mean().item(), step)
        #
        #     self.writer.add_scalar("debug/advantage_mean", advantages.mean().item(), step)
        #     self.writer.add_scalar("debug/advantage_std", advantages.std().item(), step)
        #     self.writer.add_scalar("debug/advantage_pos_percent", (advantages > 0).float().mean().item(), step)
        #
        #     self.writer.add_scalar("debug/ratio_mean", ratio.mean().item(), step)
        #     self.writer.add_scalar("debug/clip_fraction", clip_fraction, step)
        #     self.writer.add_scalar("debug/action_mean", actions.mean().item(), step)
        #
        #     # 可选：打印每个参数梯度范数（识别梯度断流）
        #     total_norm = 0
        #     for name, param in self.model.named_parameters():
        #         if param.grad is not None:
        #             norm = param.grad.norm().item()
        #             self.writer.add_scalar(f"debug/grad_norm/{name}", norm, step)
        #             total_norm += norm
        #     self.writer.add_scalar("debug/grad_norm/total", total_norm, step)

        return value_loss.item(), action_loss.item(), entropy.item()
