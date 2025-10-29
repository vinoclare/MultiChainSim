# explore/emu.py
# 简介：
#   EMUPlugin：一个最小依赖的情节记忆插件（Episodic Memory for MARL）。
#   功能：
#     1) 把当前 episode 的跨层观测拼为“训练期全局状态”并做嵌入（dCAE）
#     2) 维护“可取状态”记忆（按 RTG 的 top-p 写入）
#     3) 基于记忆做相似检索，返回逐步 episodic 激励，用于奖励塑形
# 依赖：仅 torch / numpy。与现有 MAPPO/HAPPO/环境完全解耦。

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (256, 128), act=nn.ReLU):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EMUPlugin:
    """
    最小实现版 EMU：
    - 懒初始化 dCAE（Encoder/Decoder + Return 预测头），看到第一批全局状态后确定 state_dim
    - 记忆用 (M x embed_dim) 张量 + (M,) 的可取性分数；KNN 用余弦相似度（无 faiss 依赖）
    - “可取”定义：本回合中 Return-to-Go（折扣和）位于 top-p 分位的状态
    - 训练目标：reconstruction MSE + pred_coef * return_pred MSE（对 RTG 的 z-score）
    """
    def __init__(
        self,
        embed_dim: int = 64,
        knn: int = 32,
        top_p: float = 0.2,
        lr: float = 1e-3,
        recon_coef: float = 1.0,
        pred_coef: float = 0.1,
        retrain_steps: int = 8,
        max_memory: int = 50000,
        device: Optional[str] = None,
        include_keys: Tuple[str, ...] = ("task_obs", "worker_loads", "worker_profile"),
    ):
        self.embed_dim = int(embed_dim)
        self.knn = int(knn)
        self.top_p = float(top_p)
        self.lr = float(lr)
        self.recon_coef = float(recon_coef)
        self.pred_coef = float(pred_coef)
        self.retrain_steps = int(retrain_steps)
        self.max_memory = int(max_memory)
        self.include_keys = include_keys

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 懒初始化
        self.state_dim: Optional[int] = None
        self.encoder: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.ret_head: Optional[nn.Module] = None
        self.optim: Optional[torch.optim.Optimizer] = None

        # 记忆（embedding 与可取性得分），用 torch 张量以便与批量相似度计算
        self.mem_z: Optional[torch.Tensor] = None   # [M, embed_dim]
        self.mem_s: Optional[torch.Tensor] = None   # [M]

        # 运行时归一化（防数值爆炸）：状态在网络内 BatchNorm/激活已足够；返回用 z-score
        self.eps = 1e-8

    # ---------- 对外：从 buffers 构造“训练期全局状态” ----------
    def build_global_states_from_buffers(self, buffers: Dict[int, Dict[str, List]]) -> np.ndarray:
        """
        参数：
            buffers: 形如 {lid: {'task_obs': [T * ndarray], 'worker_loads': [...], 'worker_profile': [...], ...}, ...}
        返回：
            states: shape [T, state_dim]，按时间拼接所有层与所选键，二维向量
        说明：
            - 每个时间步 t，遍历层 lid，再把 include_keys 指定的观测逐个“拉平+拼接”
            - 只依赖 run_mappo.py 中已经存在的三个键：task_obs / worker_loads / worker_profile
        """
        assert len(buffers) > 0, "buffers 为空"
        lids = sorted(buffers.keys())
        T = len(buffers[lids[0]]['rewards'])  # 与 rewards 步数对齐
        chunks: List[np.ndarray] = []

        for t in range(T):
            vec_t: List[np.ndarray] = []
            for lid in lids:
                for k in self.include_keys:
                    # 单步数据：数组或列表，统一转 numpy 并展平
                    v = _to_numpy(buffers[lid][k][t]).reshape(-1)
                    vec_t.append(v.astype(np.float32, copy=False))
            chunks.append(np.concatenate(vec_t, axis=0))

        states = np.stack(chunks, axis=0)  # [T, D]
        return states

    # ---------- 对外：基于记忆返回逐步激励（不更新记忆/不训练） ----------
    def compute_bonus_from_buffers(self, buffers: Dict[int, Dict[str, List]]) -> np.ndarray:
        states = self.build_global_states_from_buffers(buffers)
        z = self._encode_np(states)  # [T, E]
        if self.mem_z is None or self.mem_z.numel() == 0:
            return np.zeros((z.shape[0],), dtype=np.float32)

        # 余弦相似度：先做单位化，再矩阵乘
        z_t = torch.from_numpy(z).to(self.device)                           # [T, E]
        z_t = F.normalize(z_t, dim=1)
        mem = F.normalize(self.mem_z, dim=1)                                # [M, E]
        sims = torch.matmul(z_t, mem.T)                                     # [T, M]

        k = min(self.knn, sims.shape[1])
        if k <= 0:
            return np.zeros((z.shape[0],), dtype=np.float32)

        # top-k 相似度与对应的可取性分数做加权平均
        top_vals, top_idx = torch.topk(sims, k=k, dim=1)                    # [T, k]
        top_scores = self.mem_s[top_idx]                                    # [T, k]
        bonus = (top_vals * top_scores).mean(dim=1)                         # [T]
        # 安全裁剪到 [0, +]，避免极端负值污染（只鼓励“像好状态”的地方）
        bonus = torch.clamp(bonus, min=0.0)
        return bonus.detach().float().cpu().numpy()

    # ---------- 对外：更新记忆并在本回合上微调 dCAE ----------
    def update_memory_from_buffers(self, buffers: Dict[int, Dict[str, List]], gamma: float = 0.99):
        """
        执行流程：
          1) 计算本回合全局状态序列 + 对应 RTG（折扣和）
          2) 训练 dCAE（少量 step）
          3) 选择 RTG 的 top-p 分位状态写入记忆（embedding 与归一化得分）
        """
        states = self.build_global_states_from_buffers(buffers)              # [T, D]
        lids = sorted(buffers.keys())
        T = len(buffers[lids[0]]['rewards'])
        r_mat = np.stack([np.asarray(buffers[lid]['rewards'], dtype=np.float32)[:T] for lid in lids], axis=0)
        rtg = self._discounted_rtg(r_mat.mean(axis=0), gamma=gamma)

        self._ensure_model(states.shape[1])
        self._train_dcae(states, rtg, steps=self.retrain_steps)

        # 写入记忆：选择 top-p RTG 的若干步
        z = self._encode_np(states)                                         # [T, E]
        idx = self._top_p_indices(rtg, self.top_p)
        if idx.size == 0:
            return

        # 可取性分数：把 RTG 做 z-score 后再映射到 [0,1]（sigmoid）
        rtg_t = torch.as_tensor(rtg, dtype=torch.float32, device=self.device)
        rtg_std = torch.clamp(rtg_t.std(), min=self.eps)
        rtg_z = (rtg_t - rtg_t.mean()) / rtg_std
        desirability = torch.sigmoid(rtg_z)                                  # [T]
        add_z = torch.from_numpy(z[idx]).to(self.device)                     # [K, E]
        add_s = desirability[idx]                                            # [K]

        self._append_memory(add_z, add_s)

    # ===================== 内部工具 =====================

    def _discounted_rtg(self, rewards: np.ndarray, gamma: float) -> np.ndarray:
        T = len(rewards)
        out = np.zeros_like(rewards, dtype=np.float32)
        acc = 0.0
        for t in reversed(range(T)):
            acc = rewards[t] + gamma * acc
            out[t] = acc
        return out

    def _top_p_indices(self, arr: np.ndarray, p: float) -> np.ndarray:
        T = arr.shape[0]
        k = max(1, int(math.ceil(T * max(0.0, min(1.0, p)))))
        # 取最大的 k 个
        return np.argpartition(arr, -k)[-k:]

    def _ensure_model(self, state_dim: int):
        if self.state_dim is not None:
            return
        self.state_dim = int(state_dim)
        # 编码器/解码器 + RTG 预测头
        self.encoder = MLP(self.state_dim, self.embed_dim).to(self.device)
        self.decoder = MLP(self.embed_dim, self.state_dim).to(self.device)
        self.ret_head = MLP(self.embed_dim, 1, hidden=(64,)).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.ret_head.parameters()),
            lr=self.lr
        )

    def _encode_np(self, states_np: np.ndarray) -> np.ndarray:
        self._ensure_model(states_np.shape[1])
        with torch.no_grad():
            x = torch.from_numpy(states_np.astype(np.float32)).to(self.device)
            z = self.encoder(x)
        return z.detach().float().cpu().numpy()

    def _train_dcae(self, states_np: np.ndarray, rtg_np: np.ndarray, steps: int = 8, batch_size: int = 256):
        self._ensure_model(states_np.shape[1])
        x_all = torch.from_numpy(states_np.astype(np.float32)).to(self.device)  # [T, D]
        y_all = torch.from_numpy(rtg_np.astype(np.float32)).to(self.device).unsqueeze(1)  # [T, 1]

        T = x_all.shape[0]
        for _ in range(max(1, steps)):
            # 简单随机小批量
            idx = torch.randint(low=0, high=T, size=(min(batch_size, T),), device=self.device)
            x = x_all.index_select(0, idx)
            y = y_all.index_select(0, idx)

            z = self.encoder(x)
            x_hat = self.decoder(z)
            y_hat = self.ret_head(z)

            # RTG 使用 z-score，稳定训练
            y_std = torch.clamp(y.std(), min=self.eps)
            y_z = (y - y.mean()) / y_std

            loss_recon = F.mse_loss(x_hat, x)
            loss_rtg = F.mse_loss(y_hat, y_z)
            loss = self.recon_coef * loss_recon + self.pred_coef * loss_rtg

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.ret_head.parameters()), max_norm=5.0)
            self.optim.step()

    def _append_memory(self, add_z: torch.Tensor, add_s: torch.Tensor):
        """把新 embedding 与可取性分数追加到记忆；若超过上限则 FIFO 丢弃前面的。"""
        if self.mem_z is None:
            self.mem_z = add_z.detach().clone()
            self.mem_s = add_s.detach().clone()
        else:
            self.mem_z = torch.cat([self.mem_z, add_z.detach()], dim=0)
            self.mem_s = torch.cat([self.mem_s, add_s.detach()], dim=0)

        # 裁剪
        M = self.mem_z.shape[0]
        if M > self.max_memory:
            cut = M - self.max_memory
            self.mem_z = self.mem_z[cut:]
            self.mem_s = self.mem_s[cut:]
