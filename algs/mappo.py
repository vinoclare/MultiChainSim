import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class MAPPO:
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

    def sample(self, task_obs, worker_loads, worker_profile, valid_mask):
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)

        action, log_prob, mean, std = self.model.act(
            task_obs, worker_loads, worker_profile, valid_mask)
        dist = Normal(mean, std)
        value = self.model.get_value(task_obs, worker_loads, worker_profile, valid_mask)
        return value, action, log_prob.sum(dim=[1, 2]), dist.entropy().sum(dim=[1, 2])

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

    def learn(self,
              task_obs,
              worker_loads,
              worker_profile,
              valid_mask,
              actions,
              values_old,
              returns,
              log_probs_old,
              advantages,
              current_steps,
              lr=None):

        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)
        actions = actions.to(self.device)
        values_old = values_old.to(self.device)
        returns = returns.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        advantages = advantages.to(self.device)

        mean, std = self.model.forward_actor(task_obs, worker_loads, worker_profile, valid_mask)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=[1, 2])
        entropy = dist.entropy().sum(dim=[1, 2]).mean()

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        action_loss = -torch.min(surr1, surr2).mean()

        values = self.model.get_value(task_obs, worker_loads, worker_profile, valid_mask).view(-1)
        if self.use_clipped_value_loss:
            value_pred_clipped = values_old + torch.clamp(
                values - values_old, -self.clip_param, self.clip_param)
            value_losses = (values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (0.5 * (returns - values) ** 2).mean()

        progress = current_steps / self.total_training_steps
        entropy_coef = max(1e-3, self.initial_entropy_coef * (1 - progress))

        loss = value_loss * self.value_loss_coef + action_loss - entropy_coef * entropy

        if lr:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), entropy.item()
