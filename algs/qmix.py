import torch
import torch.nn as nn
import torch.optim as optim


class QMix:
    def __init__(self,
                 model,
                 target_model,
                 lr=5e-4,
                 gamma=0.99,
                 tau=0.01,
                 device="cuda"):

        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.device = device

        self.gamma = gamma
        self.tau = tau

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.update_target(tau=1.0)  # hard init

    def update_target(self, tau=None):
        tau = self.tau if tau is None else tau
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train(self, batch):
        task_obs = torch.tensor(batch["task_obs"], dtype=torch.float32, device=self.device)  # (B, A, D)
        load_obs = torch.tensor(batch["load_obs"], dtype=torch.float32, device=self.device)
        profile_obs = torch.tensor(batch["profile_obs"], dtype=torch.float32, device=self.device)
        state = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["action"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        next_task_obs = torch.tensor(batch["next_task_obs"], dtype=torch.float32, device=self.device)
        next_load_obs = torch.tensor(batch["next_load_obs"], dtype=torch.float32, device=self.device)
        next_profile_obs = torch.tensor(batch["next_profile_obs"], dtype=torch.float32, device=self.device)
        next_state = torch.tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["done"], dtype=torch.float32, device=self.device)

        q_total, _ = self.model(task_obs, load_obs, profile_obs, actions, state)  # Q_tot(s, a)
        with torch.no_grad():
            next_actions = self.model.get_actions(next_task_obs, next_load_obs, next_profile_obs)
            target_q_total, _ = self.target_model(next_task_obs, next_load_obs, next_profile_obs, next_actions, next_state)
            y = rewards + self.gamma * (1 - dones) * target_q_total

        loss = self.loss_fn(q_total, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

        self.update_target()
        return loss.item()
