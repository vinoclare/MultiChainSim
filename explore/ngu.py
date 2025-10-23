import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNDModel(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, embed_dim)
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


class ICM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, embed_dim: int):
        super().__init__()
        # φ(s)
        self.phi = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        # forward: (φ(s_t), a_t) -> φ(s_{t+1})
        self.fwd_head = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        # inverse: (φ(s_t), φ(s_{t+1})) -> a_t
        self.inv_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.phi(s)

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor, s_tp1: torch.Tensor):
        phi_t = self.phi(s_t)
        phi_tp1 = self.phi(s_tp1)
        pred_phi_tp1 = self.fwd_head(torch.cat([phi_t, a_t], dim=-1))
        pred_a = self.inv_head(torch.cat([phi_t, phi_tp1], dim=-1))
        return phi_t, phi_tp1, pred_phi_tp1, pred_a


class NGUIntrinsicReward:
    def __init__(self,
                 task_obs_dim,
                 worker_load_dim,
                 num_pad_tasks,
                 n_worker,
                 embed_dim=32,
                 k=10,
                 memory_size=500,
                 device="cuda",
                 rnd_lr=1e-4,
                 update_proportion=0.25):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.task_obs_dim = task_obs_dim
        self.worker_load_dim = worker_load_dim
        self.num_pad_tasks = num_pad_tasks
        self.n_worker = n_worker
        self.update_proportion = update_proportion

        self.state_dim = worker_load_dim * n_worker
        self.embed_dim = embed_dim
        self.rnd = RNDModel(self.state_dim, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)

        self.episodic_memory = []
        self.max_memory = memory_size
        self.k = k

        self.icm = None
        self.icm_opt = None
        self.icm_beta = 0.2  # forward-loss 权重
        self.icm_lr = 1e-3
        self._prev_state = None
        self._prev_action = None

    def _prepare_obs(self, task_obs_np: np.ndarray, worker_loads_np: np.ndarray) -> torch.Tensor:
        x = np.concatenate([
            worker_loads_np.flatten()
        ])
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

    def set_last_action(self, action_np: np.ndarray):
        a = np.asarray(action_np, dtype=np.float32).flatten()
        a = np.clip(a, -1.0, 1.0)  # 若本来在 [0,1] 可保留；这里做个稳妥 clip
        a_t = torch.tensor(a, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._prev_action = a_t
        # 懒初始化 ICM（需知道 action 维度）
        if self.icm is None:
            self.icm = ICM(state_dim=self.state_dim, action_dim=a_t.shape[1], embed_dim=self.embed_dim).to(self.device)
            self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=self.icm_lr)

    def compute_bonus(self, task_obs_np: np.ndarray, worker_loads_np: np.ndarray) -> float:
        x = self._prepare_obs(task_obs_np, worker_loads_np)
        rnd_error = self.rnd(x).item()
        if self.icm is not None:
            emb = self.icm.encode(x).squeeze(0)
        else:
            emb = self.rnd.encode(x).squeeze(0)
        emb = F.normalize(emb, p=2, dim=0)
        epi_bonus = self._episodic_novelty(emb)
        return rnd_error

    def update(self, task_obs_np: np.ndarray, worker_loads_np: np.ndarray):
        x = self._prepare_obs(task_obs_np, worker_loads_np)

        # RND update
        with torch.no_grad():
            target = self.rnd.target(x)

        pred = self.rnd.predictor(x)

        mask = torch.rand(pred.shape[0], device=self.device) < self.update_proportion
        if mask.sum() == 0:
            return  # 本轮不更新
        loss = F.mse_loss(pred[mask], target[mask])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ICM update
        if self.icm is not None and (self._prev_state is not None) and (self._prev_action is not None):
            s_t = self._prev_state
            a_t = self._prev_action
            s_tp1 = x

            _, phi_tp1, pred_phi_tp1, pred_a = self.icm(s_t, a_t, s_tp1)
            inv_loss = F.mse_loss(pred_a, a_t)
            fwd_loss = F.mse_loss(pred_phi_tp1, phi_tp1)
            icm_loss = (1.0 - self.icm_beta) * inv_loss + self.icm_beta * fwd_loss

            self.icm_opt.zero_grad()
            icm_loss.backward()
            self.icm_opt.step()

        # === 用 φ(s_{t+1}) 更新 episodic memory（若无 ICM，就退回 RND 的编码） ===
        if self.icm is not None:
            emb = self.icm.encode(x).squeeze(0)
        else:
            emb = self.rnd.encode(x).squeeze(0)
        emb = F.normalize(emb, p=2, dim=0).detach().cpu().numpy()
        self.episodic_memory.append(emb)
        if len(self.episodic_memory) > self.max_memory:
            self.episodic_memory.pop(0)

        # 滚动状态
        self._prev_state = x.detach()

    def _episodic_novelty(self, emb: torch.Tensor) -> float:
        if len(self.episodic_memory) == 0:
            return 1.0
        mem_np = np.array(self.episodic_memory, dtype=np.float32)
        mem = torch.tensor(mem_np, device=self.device)  # [M, D]
        emb_n = F.normalize(emb, p=2, dim=0)  # [D]
        mem_n = F.normalize(mem, p=2, dim=1)  # [M, D]
        dists = torch.norm(mem_n - emb_n, dim=1)  # [M]
        k = min(self.k, len(dists))
        topk = torch.topk(dists, k, largest=False).values  # 取k近邻
        return topk.mean().item()

    def reset_episode(self):
        self.episodic_memory.clear()
