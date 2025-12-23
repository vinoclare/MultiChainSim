import torch
import torch.nn as nn
from torch.distributions import Normal

from algs.mappo import MAPPO


class IS(MAPPO):
    """
    Informed Switching (IS) for MAPPO-style policy:
    - exploit mode: use model std as-is
    - explore mode: inflate std by explore_std_scale

    Interface alignment:
      sample(task_obs, worker_loads, worker_profile, valid_mask, mode=0)
      learn(..., current_steps, lr=None, modes=None)

    mode / modes:
      - mode: scalar {0,1} or bool; 0=exploit, 1=explore
      - modes: tensor/list shape [B] (or broadcastable), 0/1 per sample
    """

    def __init__(self, model, explore_std_scale: float = 2.0, **kwargs):
        super().__init__(model, **kwargs)
        if explore_std_scale <= 0:
            raise ValueError(f"explore_std_scale must be positive, got {explore_std_scale}")
        self.explore_std_scale = float(explore_std_scale)

    def _apply_std_scale(self, std: torch.Tensor, mode) -> torch.Tensor:
        """Scale std for explore mode. mode can be scalar or tensor."""
        if mode is None:
            return std

        # Tensor mode: per-sample scaling
        if isinstance(mode, torch.Tensor):
            m = mode.to(std.device).float()
            if m.dim() == 0:
                # scalar tensor
                if float(m.item()) > 0.5:
                    return std * self.explore_std_scale
                return std
            # Expect [B] or [B,1] etc. Make it [B,1,1] for broadcast to [B,W,T]
            while m.dim() < std.dim():
                m = m.unsqueeze(-1)
            # std * (1 + (scale-1)*m)
            return std * (1.0 + (self.explore_std_scale - 1.0) * m)

        # Scalar mode
        try:
            m = int(mode)
        except Exception:
            m = 0
        if m == 1:
            return std * self.explore_std_scale
        return std

    @torch.no_grad()
    def sample(self, task_obs, worker_loads, worker_profile, valid_mask, mode: int = 0):
        """
        Same as MAPPO.sample, but supports mode:
          mode=0 exploit, mode=1 explore (inflate std).
        """
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)

        mean, std = self.model.forward_actor(task_obs, worker_loads, worker_profile, valid_mask)
        std = self._apply_std_scale(std, mode)

        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=[1, 2])
        entropy = dist.entropy().sum(dim=[1, 2])

        value = self.model.get_value(task_obs, worker_loads, worker_profile, valid_mask)
        return value, action, log_prob, entropy

    @torch.no_grad()
    def predict(self, task_obs, worker_loads, worker_profile, valid_mask, mode: int = 0):
        """
        Keep MAPPO.predict behavior for compatibility.
        (We return mean; mode is accepted but ignored by default.)
        """
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)
        mean, _ = self.model.forward_actor(task_obs, worker_loads, worker_profile, valid_mask)
        return mean

    @torch.no_grad()
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
              lr=None,
              modes=None):
        """
        Same positional args as MAPPO.learn, plus optional modes at the end.

        modes:
          - None: all exploit
          - shape [B] or broadcastable; 0 exploit, 1 explore per-sample
        """
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)
        actions = actions.to(self.device)
        values_old = values_old.to(self.device).view(-1)
        returns = returns.to(self.device).view(-1)
        log_probs_old = log_probs_old.to(self.device).view(-1)
        advantages = advantages.to(self.device).view(-1)

        if modes is not None and not isinstance(modes, torch.Tensor):
            modes = torch.tensor(modes, dtype=torch.float32)
        if isinstance(modes, torch.Tensor):
            # make sure batch dimension aligns with B
            if modes.dim() > 0:
                modes = modes.to(self.device)
            else:
                modes = modes.to(self.device)

        inner_k = int(getattr(self, "inner_k", 1))
        clip_param = float(getattr(self, "clip_param", 0.1))
        value_loss_coef = float(getattr(self, "value_loss_coef", 0.5))
        max_grad_norm = float(getattr(self, "max_grad_norm", 0.5))
        use_clipped_value_loss = bool(getattr(self, "use_clipped_value_loss", False))
        norm_adv = bool(getattr(self, "norm_adv", False))

        # entropy schedule params
        total_training_steps = float(getattr(self, "total_training_steps", 1.0))
        initial_entropy_coef = float(getattr(self, "initial_entropy_coef", getattr(self, "entropy_coef", 0.01)))

        for _ in range(inner_k):
            mean, std = self.model.forward_actor(task_obs, worker_loads, worker_profile, valid_mask)
            std = self._apply_std_scale(std, modes)

            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=[1, 2])
            entropy = dist.entropy().sum(dim=[1, 2]).mean()

            adv = advantages
            if norm_adv:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
            action_loss = -torch.min(surr1, surr2).mean()

            values = self.model.get_value(task_obs, worker_loads, worker_profile, valid_mask).view(-1)

            if use_clipped_value_loss:
                value_pred_clipped = values_old + torch.clamp(values - values_old, -clip_param, clip_param)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = 0.5 * ((returns - values) ** 2).mean()

            progress = float(current_steps) / max(total_training_steps, 1.0)
            entropy_coef = max(1e-3, initial_entropy_coef * (1.0 - progress))

            loss = value_loss * value_loss_coef + action_loss - entropy_coef * entropy

            if lr is not None:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = float(lr)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

        # 可选：沿用你原 MAPPO 里的 step 计数/写日志习惯（有则用，无则忽略）
        if hasattr(self, "global_step_ref") and isinstance(self.global_step_ref, list) and self.global_step_ref:
            self.global_step_ref[0] += 1
