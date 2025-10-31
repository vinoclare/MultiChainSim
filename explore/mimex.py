from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["MIMExModule"]


class _TinyReconstructor(nn.Module):
    """极简 MLP：输入为被掩码后的观测向量，输出重建整向量。"""
    def __init__(self, obs_dim: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, obs_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MIMExModule(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        emb_dim: int = 64,
        mask_ratio: float = 0.15,
        lr: float = 1e-3,
        grad_clip: float = 5.0,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        assert 0.0 < mask_ratio < 1.0, "mask_ratio 应在 (0,1) 内"
        self.obs_dim = int(obs_dim)
        self.mask_ratio = float(mask_ratio)
        self.grad_clip = float(grad_clip)
        self.device = torch.device(device)

        self.model = _TinyReconstructor(self.obs_dim, int(emb_dim)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # 计数器仅供必要时监控，不参与逻辑
        self._step = 0

        self.train()  # 始终处于训练模式

    @torch.no_grad()
    def reset(self):
        """与 eta-psi 对齐的占位函数。当前实现无需清内部状态。"""
        self._step = 0

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if t.dim() == 1:
            t = t.unsqueeze(0)  # 形状 [1, obs_dim]
        assert t.shape[-1] == self.obs_dim, f"观测维度不匹配：got {t.shape[-1]}, expect {self.obs_dim}"
        return t.to(self.device, dtype=torch.float32)

    def _random_mask(self) -> torch.Tensor:
        """返回形状 [1, obs_dim] 的 {0,1} 掩码张量，至少掩 1 维。"""
        k = max(1, int(round(self.obs_dim * self.mask_ratio)))
        idx = torch.randperm(self.obs_dim, device=self.device)[:k]
        m = torch.zeros((1, self.obs_dim), device=self.device)
        m[:, idx] = 1.0
        return m

    def update_and_bonus(
        self,
        s: Union[np.ndarray, torch.Tensor],
        sp: Optional[Union[np.ndarray, torch.Tensor]] = None,  # 为对齐 eta-psi 的签名，未使用
    ) -> float:
        """
        对当前状态 s 执行一次“掩码→重建→仅被掩位置 MSE”的训练，并返回该 MSE 作为内在奖励。
        """
        self.model.train()
        x = self._to_tensor(s)               # [1, D]
        mask = self._random_mask()           # [1, D], 1 表示该维被掩

        # 构造被掩输入：掩位直接置 0（不依赖环境尺度，简单稳健）
        x_masked = x * (1.0 - mask)

        # 前向与掩码 MSE
        pred = self.model(x_masked)          # [1, D]
        diff = (pred - x) ** 2               # [1, D]
        masked_mse = (diff * mask).sum() / (mask.sum() + 1e-8)

        # 奖励在更新之前按当前误差评估
        r_int = float(masked_mse.detach().cpu().item())

        # 反向与一步优化
        self.optimizer.zero_grad(set_to_none=True)
        masked_mse.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self._step += 1
        return r_int
