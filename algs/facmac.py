import torch, torch.nn as nn, torch.optim as optim
from copy import deepcopy


class FACMAC:
    """TD3-style centralized critic + factorized actors."""

    def __init__(self, model, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.005, device="cuda"):
        self.device = device
        self.model = model.to(device)
        self.target = deepcopy(model).to(device).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        actor_params = list(model.actor.parameters())
        critic_params = [p for n, p in model.named_parameters() if 'actor' not in n]

        self.opt_actor = optim.Adam(actor_params, lr_actor)
        self.opt_critic = optim.Adam(critic_params, lr_critic)

        self.gamma, self.tau = gamma, tau
        self.mse = nn.MSELoss()

    @torch.no_grad()
    def _soft_update(self, tau):
        for p, tp in zip(self.model.parameters(), self.target.parameters()):
            tp.copy_(tau * p + (1 - tau) * tp)

    # ---------- 训练 ----------
    def train(self, batch, policy_delay=2, noise_std=0.2, noise_clip=0.5, step=0):
        toT = lambda x: torch.tensor(x, dtype=torch.float32, device=self.device)
        task, load, prof = toT(batch['task_obs']), toT(batch['load_obs']), toT(batch['profile_obs'])
        n_task, n_load, n_prof = toT(batch['next_task_obs']), toT(batch['next_load_obs']), toT(
            batch['next_profile_obs'])
        act, st, nst = toT(batch['action']), toT(batch['state']), toT(batch['next_state'])
        rew, done = toT(batch['reward']), toT(batch['done'])

        # —— Critic —— --------------------------------------------------------
        with torch.no_grad():
            next_act = self.target.get_actions(n_task, n_load, n_prof, deterministic=False, noise_std=noise_std)
            # clipped noise (TD3)
            noise = torch.randn_like(next_act) * noise_std
            noise = noise.clamp(-noise_clip, noise_clip)
            next_act = (next_act + noise).clamp(0, 1)

            q_next = self.target(n_task, n_load, n_prof, next_act, nst)[0]  # 取 q1
            y = rew + self.gamma * (1 - done) * q_next  # (B,)

        q_curr1, q_curr2 = self.model(task, load, prof, act, st)
        critic_loss = self.mse(q_curr1, y) + self.mse(q_curr2, y)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.opt_critic.param_groups[0]['params'], 10)
        self.opt_critic.step()

        # —— Actor —— ---------------------------------------------------------
        if step % policy_delay == 0:
            # 重新计算当前策略动作
            pi_act = self.model.get_actions(task, load, prof, deterministic=True)
            actor_q = self.model(task, load, prof, pi_act, st)[0]
            actor_loss = -actor_q.mean()

            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            self._soft_update(self.tau)

        return critic_loss.item()
