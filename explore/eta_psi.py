import torch
import torch.nn as nn
import torch.nn.functional as F


class EtaPsiModule(nn.Module):
    def __init__(self, obs_dim: int, emb_dim: int = 64, gamma: float = 0.99,
                 lr: float = 1e-3, device: str = "cpu", grad_clip: float = 5.0):
        super().__init__()
        self.device = torch.device(device)
        self.gamma = gamma
        self.grad_clip = grad_clip

        # 观测 -> 表征 e(s)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )

        # e(s) -> ψ(s) 预测头（与 SR 维度一致）
        self.psi_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
        )

        self.to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

        # 折扣前驱累计向量 η
        self.register_buffer("eta_vec", torch.zeros(emb_dim, device=self.device))
        self.register_buffer("ri_ma", torch.zeros(1, device=self.device))  # 指数滑动均值

    @torch.no_grad()
    def reset(self):
        self.eta_vec.zero_()

    def _embed(self, obs_vec: torch.Tensor) -> torch.Tensor:
        # obs_vec: [D] 或 [B, D]
        z = self.encoder(obs_vec)
        # 归一化避免模长漂移
        return F.normalize(z, p=2, dim=-1)

    def update_and_bonus(self, s_vec_np, sp_vec_np) -> float:
        """
        用(s, s')做一次 TD 更新，并返回针对 s 的内在奖励（标量 float）。
        """
        s = torch.as_tensor(s_vec_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, D]
        sp = torch.as_tensor(sp_vec_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        e_s = self._embed(s)  # [1, E]
        e_sp = self._embed(sp)  # [1, E]

        # 目标：ψ(s) ≈ e(s) + γ ψ(s')
        with torch.no_grad():
            psi_sp = self.psi_head(e_sp)  # [1, E]

        target = (e_s + self.gamma * psi_sp).detach()  # [1, E]
        pred = self.psi_head(e_s)  # [1, E]
        loss = F.mse_loss(pred, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.opt.step()

        # 递推 η ← γ·η + e(s)
        with torch.no_grad():
            self.eta_vec.mul_(self.gamma).add_(e_s.squeeze(0))

            counts = self.eta_vec + pred.detach().squeeze(0)  # η + ψ(s)
            # 伪计数 -> 奖励：n^(-1/2)。用均值二范数做标量化更稳定
            norm = counts.pow(2).mean().sqrt()
            r_raw = -torch.log(1e-6 + norm)
            self.ri_ma.mul_(0.99).add_(0.01 * r_raw)
            r_int = r_raw - self.ri_ma.item()

        return float(r_int)
