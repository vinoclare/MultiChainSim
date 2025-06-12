"""
HiTAC  (Hierarchical Transformer-based Adaptive Coordinator)
------------------------------------------------------------
- 输入：全局 KPI token  +  各层最近局部 KPI 序列
- 输出：每层选择子策略的 logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class HiTAC(nn.Module):
    def __init__(
        self,
        local_kpi_dim: int,
        global_kpi_dim: int,
        num_layers: int,
        num_subpolicies: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        transformer_layers: int = 2,
        clip_param: float = 0.2,
        entropy_coef: float = 0.001,
        max_grad_norm: float = 0.5,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.num_subpolicies = num_subpolicies
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # === 编码器 ===
        self.local_embed = nn.Linear(local_kpi_dim, hidden_dim)
        self.global_embed = nn.Linear(global_kpi_dim, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                batch_first=True,
                dim_feedforward=4 * hidden_dim,
                norm_first=True
            ),
            num_layers=transformer_layers
        )

        # === 输出层：每层 → K个子策略的logits ===
        self.head = nn.Linear(hidden_dim, num_subpolicies)

        # === PPO缓存 ===
        self.old_log_probs = None
        self.old_logits = None
        self.old_actions = None
        self.old_advantages = None

    def forward(self, local_kpis: torch.Tensor, global_kpi: torch.Tensor):
        """
        Args:
          local_kpis: (B, L, d_local)    每层的局部KPI序列
          global_kpi: (B, d_global)      全局摘要

        Return:
          logits: (B, L, K) 每层子策略 logits
        """
        B, L, _ = local_kpis.shape
        g_emb = self.global_embed(global_kpi).unsqueeze(1)          # (B, 1, d)
        l_emb = self.local_embed(local_kpis)                        # (B, L, d)
        tokens = torch.cat([g_emb, l_emb], dim=1)                   # (B, L+1, d)
        encoded = self.encoder(tokens)                              # (B, L+1, d)
        h = encoded[:, 1:, :]                                       # 取出每层的表示 (B, L, d)
        logits = self.head(h)                                       # (B, L, K)
        return logits

    def select(self, local_kpis, global_kpi, greedy=False):
        """
        推理接口：为每一层选择子策略
        Return:
          pids: LongTensor[B, L]
        """
        logits = self.forward(local_kpis, global_kpi)               # (B, L, K)
        probs = F.softmax(logits, dim=-1)
        if greedy:
            return torch.argmax(probs, dim=-1)                      # (B, L)
        else:
            dist = Categorical(probs)
            return dist.sample()                                    # (B, L)

    def store_for_update(self, logits, actions, advantages):
        """
        存储用于训练的logits和actions
        """
        self.old_logits = logits.detach()         # (B, L, K)
        self.old_log_probs = F.log_softmax(self.old_logits, dim=-1)
        self.old_actions = actions.detach()       # (B, L)
        self.old_advantages = advantages.detach() # (B,)

    def update(self, local_kpis, global_kpi, actions, advantages, lr=None):
        """
        PPO-style update (单步)
        Args:
          actions: LongTensor[B, L] —— 每层选择的子策略id
          advantages: Tensor[B] —— 环境给出的全局优势
        """
        actions = actions.to(self.device)
        logits = self.forward(local_kpis, global_kpi)              # (B, L, K)
        new_log_probs = F.log_softmax(logits, dim=-1)              # (B, L, K)

        B, L, K = logits.shape
        old_lp = self.old_log_probs.gather(-1, self.old_actions.unsqueeze(-1).to(self.device)).squeeze(-1)  # (B, L)
        new_lp = new_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)               # (B, L)

        # === 计算 ratio + clipped loss ===
        ratio = torch.exp(new_lp - old_lp)         # (B, L)
        adv = advantages.unsqueeze(-1).expand_as(ratio)  # broadcast → (B, L)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = - (new_log_probs * new_log_probs.exp()).sum(dim=-1).mean()

        loss = policy_loss - self.entropy_coef * entropy

        # === 更新 ===
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr or 3e-4)
        if lr:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "hitac/policy_loss": policy_loss.item(),
            "hitac/entropy": entropy.item(),
            "hitac/total_loss": loss.item()
        }
