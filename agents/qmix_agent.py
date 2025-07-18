import numpy as np
import torch


class QMixAgent:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device

    def select_action(self, task_obs, load_obs, profile_obs, state):
        """
        Args:
            task_obs: (n_worker, task_dim)
            load_obs: (n_worker, load_dim)
            profile_obs: (n_worker, profile_dim)
            state: (state_dim,) or (1, state_dim)
        Returns:
            action: (n_worker, task_dim), float ratio per task
        """
        task_obs = torch.tensor(task_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        load_obs = torch.tensor(load_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        profile_obs = torch.tensor(profile_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        _, agent_qs = self.model(task_obs, load_obs, profile_obs, state)  # (1, n_worker)
        action = torch.sigmoid(agent_qs).squeeze(0).detach().cpu().numpy()  # (n_worker,)
        return action  # can be interpreted as per-worker intensity/ratio
