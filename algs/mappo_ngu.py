import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class MAPPONGU:
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
        ext_value = self.model.get_ext_value(task_obs, worker_loads, worker_profile, valid_mask)
        int_value = self.model.get_int_value(task_obs, worker_loads, worker_profile, valid_mask)
        return ext_value, int_value, action, log_prob.sum(dim=[1, 2]), dist.entropy().sum(dim=[1, 2])

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
        return (
            self.model.get_ext_value(task_obs, worker_loads, worker_profile, valid_mask),
            self.model.get_int_value(task_obs, worker_loads, worker_profile, valid_mask)
        )

    def learn(self,
              task_obs,
              worker_loads,
              worker_profile,
              valid_mask,
              actions,
              values_old_ext,
              values_old_int,
              returns_ext,
              returns_int,
              log_probs_old,
              advantages,
              current_steps,
              lr=None):

        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)
        actions = actions.to(self.device)
        values_old_ext = values_old_ext.to(self.device)
        values_old_int = values_old_int.to(self.device)
        returns_ext = returns_ext.to(self.device)
        returns_int = returns_int.to(self.device)
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

        # ===== Dual value heads =====
        values_ext = self.model.get_ext_value(task_obs, worker_loads, worker_profile, valid_mask)
        values_int = self.model.get_int_value(task_obs, worker_loads, worker_profile, valid_mask)

        if self.use_clipped_value_loss:
            val_ext_clipped = values_old_ext + torch.clamp(values_ext - values_old_ext, -self.clip_param, self.clip_param)
            val_int_clipped = values_old_int + torch.clamp(values_int - values_old_int, -self.clip_param, self.clip_param)

            loss_ext = torch.max((values_ext - returns_ext)**2, (val_ext_clipped - returns_ext)**2).mean()
            loss_int = torch.max((values_int - returns_int)**2, (val_int_clipped - returns_int)**2).mean()
        else:
            loss_ext = ((values_ext - returns_ext) ** 2).mean()
            loss_int = ((values_int - returns_int) ** 2).mean()

        value_loss = 0.5 * (loss_ext + loss_int)
        # ============================

        progress = current_steps / self.total_training_steps
        entropy_coef = max(1e-3, self.initial_entropy_coef * (1 - progress))

        loss = self.value_loss_coef * value_loss + action_loss - entropy_coef * entropy

        if lr:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), entropy.item()
