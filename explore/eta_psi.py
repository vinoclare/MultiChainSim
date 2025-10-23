from __future__ import annotations
import math
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


class PhiEncoder(nn.Module):
    """轻量状态表征 φ(s)。默认冻结参数，充当随机编码器（稳定、好复现）。"""
    def __init__(self, state_dim: int, z_dim: int = 64, hidden: int = 128, trainable: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
        # 初始化尽量稳定
        nn.init.orthogonal_(self.net[0].weight, gain=math.sqrt(2))
        nn.init.zeros_(self.net[0].bias)
        nn.init.orthogonal_(self.net[2].weight, gain=0.01)
        nn.init.zeros_(self.net[2].bias)

        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B, state_dim) or (state_dim,)
        return: (B, z_dim) >= 0  使用 softplus 让“计数向量”非负
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        z = self.net(s)
        return F.softplus(z)


class SRNet(nn.Module):
    """Successor Representation 头：输入 z(s), a（连续动作），输出 ψ(z,a) >= 0。"""
    def __init__(self, z_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z: (B, z_dim) ; a: (B, action_dim)
        return: (B, z_dim) >= 0
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        x = torch.cat([z, a], dim=-1)
        return F.softplus(self.net(x))


class RunningNorm:
    """在线标准化 r_int，避免训练初期尺度抖动。"""
    def __init__(self, eps: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def normalize(self, x: float) -> float:
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)
        denom = math.sqrt(self.var / max(self.count - 1.0, 1.0)) + 1e-8
        return float((x - self.mean) / denom)


class EtaPsiModule(nn.Module):
    """
    核心模块：
    - 维护 η（本回合访问累积向量，按 φ(s) 累加，非参数）
    - 用 SRNet 近似 ψ(s,a)
    - r_int = H(p)，其中 p = normalize(η + ψ)
    - TD 目标：ψ(s,a) ≈ φ(s') + γ * 1_{not done} * ψ(s', a')
    最小改动接入：仅在采样处调用 compute_intrinsic，并在 PPO/MAPPO 更新前调用 update_sr。
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        z_dim: int = 64,
        gamma: float = 0.99,
        device: str | torch.device = "cpu",
        lr: float = 3e-4,
        phi_trainable: bool = False,
        clip_grad: float = 1.0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.z_dim = z_dim
        self.gamma = gamma
        self.clip_grad = clip_grad

        self.phi = PhiEncoder(state_dim, z_dim=z_dim, trainable=phi_trainable).to(self.device)
        self.sr = SRNet(z_dim=z_dim, action_dim=action_dim).to(self.device)

        # 只优化 requires_grad=True 的参数（默认仅 SR）
        params = [p for p in self.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=lr)

        # 回合态
        self.register_buffer("eta", torch.zeros(z_dim, device=self.device))
        self.rnorm = RunningNorm()

        self.eval()  # 默认推理模式
        self.to(self.device)

    @torch.no_grad()
    def reset_episode(self):
        """回合开始时调用，清空 η。"""
        self.eta.zero_()

    def _entropy_from_counts(self, counts: torch.Tensor) -> torch.Tensor:
        """
        counts: (z_dim,) 非负
        返回香农熵 H(p) = -∑ p log p
        """
        total = counts.sum().clamp_min(1e-8)
        p = counts / total
        H = -(p * p.clamp_min(1e-8).log()).sum()
        return H

    @torch.no_grad()
    def compute_intrinsic(self, obs_t: np.ndarray | torch.Tensor, act_t: np.ndarray | torch.Tensor) -> float:
        """
        给一步 (s_t, a_t) 计算 r_int，并更新 η ← η + φ(s_t)。
        输入：
            obs_t: (state_dim,) 向量观测（已 flatten 即可）
            act_t: (action_dim,) 连续动作（flatten）
        输出：
            r_int (float)：在线标准化后的内在奖励
        """
        s = _to_tensor(obs_t, device=self.device).view(1, -1)
        a = _to_tensor(act_t, device=self.device).view(1, -1)

        z = self.phi(s)                  # (1, z_dim) >= 0
        psi = self.sr(z, a)              # (1, z_dim) >= 0

        counts = self.eta + psi.squeeze(0)   # 预测“过去+未来”的占用
        H = self._entropy_from_counts(counts)
        r_int = self.rnorm.normalize(float(H.item()))

        # 更新 η：只累加“已发生”的 φ(s_t)，不含未来
        self.eta.add_(z.squeeze(0))
        return r_int

    def update_sr(self, batch: Dict[str, Any], iters: int = 1) -> float:
        """
        用 on-policy 批更新 SRNet（最小改动的一阶 TD）：
            ψ(s,a) ≈ φ(s_next) + γ * mask * ψ(s_next, a_next)
        batch 需要字段（numpy 或 tensor 均可）：
            "s": (B, state_dim)
            "a": (B, action_dim)
            "s_next": (B, state_dim)
            "a_next": (B, action_dim)  # 可用策略在 s_next 上的均值动作
            "mask": (B,)  终止为 0.0，非终止为 1.0
        返回：
            最新一次迭代的标量 loss（float）
        """
        self.train()
        last_loss = 0.0
        for _ in range(max(1, iters)):
            s = _to_tensor(batch["s"], device=self.device).view(-1, self.phi.net[0].in_features)
            a = _to_tensor(batch["a"], device=self.device).view(s.size(0), -1)
            sn = _to_tensor(batch["s_next"], device=self.device).view(s.size(0), -1)
            an = _to_tensor(batch.get("a_next", batch["a"]), device=self.device).view(s.size(0), -1)
            mask = _to_tensor(batch["mask"], device=self.device).view(-1, 1)  # (B,1)

            with torch.no_grad():
                zn = self.phi(sn)
                target = zn + self.gamma * mask * self.sr(zn, an)

            z = self.phi(s)              # 若 φ 冻结，不反传到 φ
            pred = self.sr(z, a)

            loss = F.mse_loss(pred, target)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.clip_grad is not None and self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.sr.parameters(), self.clip_grad)
            self.opt.step()

            last_loss = float(loss.item())

        self.eval()
        return last_loss
