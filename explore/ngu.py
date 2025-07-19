# explore/ngu.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNDModel(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            t = self.target(x)
        p = self.predictor(x)
        return F.mse_loss(p, t, reduction='none').mean(dim=1)

    def encode(self, x: torch.Tensor):
        with torch.no_grad():
            return self.target(x)


class NGUIntrinsicReward:
    def __init__(self,
                 task_obs_dim,
                 worker_load_dim,
                 num_pad_tasks,
                 n_worker,
                 embed_dim=64,
                 k=10,
                 memory_size=500,
                 device="cuda",
                 rnd_lr=1e-4):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.task_obs_dim = task_obs_dim
        self.worker_load_dim = worker_load_dim
        self.num_pad_tasks = num_pad_tasks
        self.n_worker = n_worker

        input_dim = task_obs_dim * num_pad_tasks + worker_load_dim * n_worker
        self.rnd = RNDModel(input_dim, embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)

        self.episodic_memory = []
        self.max_memory = memory_size
        self.k = k

    def _prepare_obs(self, task_obs_np: np.ndarray, worker_loads_np: np.ndarray) -> torch.Tensor:
        x = np.concatenate([
            task_obs_np.flatten(),
            worker_loads_np.flatten()
        ])
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

    def compute_bonus(self, task_obs_np: np.ndarray, worker_loads_np: np.ndarray) -> float:
        x = self._prepare_obs(task_obs_np, worker_loads_np)
        rnd_error = self.rnd(x).item()
        emb = self.rnd.encode(x).squeeze(0)
        epi_bonus = self._episodic_novelty(emb)
        return rnd_error * epi_bonus

    def update(self, task_obs_np: np.ndarray, worker_loads_np: np.ndarray):
        x = self._prepare_obs(task_obs_np, worker_loads_np)
        target = self.rnd.target(x).detach()
        pred = self.rnd.predictor(x)
        loss = F.mse_loss(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        emb = self.rnd.encode(x).squeeze(0).detach().cpu().numpy()
        self.episodic_memory.append(emb)
        if len(self.episodic_memory) > self.max_memory:
            self.episodic_memory.pop(0)

    def _episodic_novelty(self, emb: torch.Tensor) -> float:
        if len(self.episodic_memory) == 0:
            return 1.0
        mem = torch.tensor(self.episodic_memory, dtype=torch.float32, device=self.device)
        dists = torch.norm(mem - emb, dim=1)
        k = min(self.k, len(dists))
        topk = torch.topk(dists, k, largest=False).values
        return 1.0 / (topk.mean().item() + 1e-5)

    def reset_episode(self):
        self.episodic_memory.clear()
