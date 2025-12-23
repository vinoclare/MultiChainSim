# utils/utils_bhera.py
# -*- coding: utf-8 -*-
"""
BHERA utilities:
- layer stacking / pooling
- token building for belief inference
- n-step return targets
- KL for diagonal Gaussian posterior vs N(0, I)
- EMA + KL homeostasis (beta update)
- simple window buffer for causal Transformer / windowed inference
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch


Tensor = torch.Tensor


# ---------------------------
# 1) Layer stacking / pooling
# ---------------------------

def stack_by_layer(
    x_dict: Dict[int, Tensor],
    layer_ids: Optional[Sequence[int]] = None,
    dim_layer: int = 1,
) -> Tensor:
    """
    Stack per-layer tensors from dict -> Tensor.

    Args:
        x_dict: {layer_id: Tensor[B, D] or Tensor[B, ...]}
        layer_ids: stacking order; if None, use sorted keys.
        dim_layer: output layer dimension position:
            - dim_layer=1 => [B, L, ...]
            - dim_layer=0 => [L, B, ...]
    Returns:
        x_stack: stacked tensor with explicit layer dimension.
    """
    if layer_ids is None:
        layer_ids = sorted(list(x_dict.keys()))
    xs = [x_dict[lid] for lid in layer_ids]
    x_stack = torch.stack(xs, dim=dim_layer)  # [B, L, ...] if dim_layer=1
    return x_stack


def mean_pool_layers(x_stack: Tensor, dim_layer: int = 1) -> Tensor:
    """
    Mean-pool over layer dimension.
    x_stack: [B, L, D] (dim_layer=1) or [L, B, D] (dim_layer=0)
    """
    return x_stack.mean(dim=dim_layer)


def pool_actions_mean(
    a_dict: Dict[int, Tensor],
    layer_ids: Optional[Sequence[int]] = None,
) -> Tensor:
    """
    Pool per-layer actions by mean. Useful to build a global token.
    a_dict[lid]: Tensor[B, A]
    Returns:
        a_pool: Tensor[B, A]
    """
    a_stack = stack_by_layer(a_dict, layer_ids=layer_ids, dim_layer=1)  # [B, L, A]
    return mean_pool_layers(a_stack, dim_layer=1)  # [B, A]


# ---------------------------
# 2) Token building
# ---------------------------

def _ensure_2d(x: Tensor) -> Tensor:
    """Ensure x is [B, D] or [B, 1] for scalars."""
    if x.dim() == 1:
        return x.unsqueeze(-1)
    return x


def build_belief_token(
    x_pool: Tensor,
    a_prev_pool: Tensor,
    r_prev: Tensor,
    done_prev: Tensor,
) -> Tensor:
    """
    Build a transition token for belief inference.

    Args:
        x_pool:    Tensor[B, Dx]  pooled coupled embedding at time t (or t-1, per your design)
        a_prev_pool: Tensor[B, Da] pooled previous action
        r_prev:    Tensor[B] or Tensor[B,1] previous reward
        done_prev: Tensor[B] or Tensor[B,1] previous done flag (0/1)

    Returns:
        token: Tensor[B, Dx + Da + 1 + 1]
    """
    r_prev = _ensure_2d(r_prev)
    done_prev = _ensure_2d(done_prev)
    return torch.cat([x_pool, a_prev_pool, r_prev, done_prev], dim=-1)


# ---------------------------
# 3) n-step return targets
# ---------------------------

def left_pad_last_k(seq_bt_d: torch.Tensor, k: int) -> torch.Tensor:
    # seq: [B,T,D] -> [B,k,D] 右对齐，不足左侧补 0
    B, T, D = seq_bt_d.shape
    if T >= k:
        return seq_bt_d[:, T - k:T, :]
    pad = torch.zeros((B, k - T, D), dtype=seq_bt_d.dtype, device=seq_bt_d.device)
    return torch.cat([pad, seq_bt_d], dim=1)


def left_pad_sliding_k(seq_bt_d: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build sliding left-padded windows for each timestep.

    Args:
        seq_bt_d: Tensor[B, T, D]
        k: window length

    Returns:
        win: Tensor[B, T, k, D]  (for each t, take seq[max(0,t-k+1):t+1] left padded)
        pad_mask: Tensor[B, T, k] bool, True means PAD
    """
    B, T, D = seq_bt_d.shape
    if k <= 1:
        win = seq_bt_d.unsqueeze(2)  # [B,T,1,D]
        pad_mask = torch.zeros((B, T, 1), dtype=torch.bool, device=seq_bt_d.device)
        return win, pad_mask

    # left pad with (k-1) zeros so that unfold creates left-padded windows
    pad = torch.zeros((B, k - 1, D), dtype=seq_bt_d.dtype, device=seq_bt_d.device)
    seq_pad = torch.cat([pad, seq_bt_d], dim=1)  # [B, T+k-1, D]

    win = seq_pad.unfold(dimension=1, size=k, step=1)  # [B, T, k, D]

    # build pad mask: early timesteps have left padding
    t_idx = torch.arange(T, device=seq_bt_d.device)  # [T]
    pad_cnt = (k - 1 - t_idx).clamp(min=0)  # [T]
    k_idx = torch.arange(k, device=seq_bt_d.device).unsqueeze(0)  # [1,k]
    mask_tk = k_idx < pad_cnt.unsqueeze(1)  # [T,k]
    pad_mask = mask_tk.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B,T,k]
    return win, pad_mask


def compute_nstep_returns(rewards_bt: torch.Tensor, gamma: float, n_step: int) -> torch.Tensor:
    # rewards_bt: [B,T,1] -> [B,T,1]
    B, T, _ = rewards_bt.shape
    out = torch.zeros_like(rewards_bt)
    for t in range(T):
        g = torch.zeros((B, 1), dtype=rewards_bt.dtype, device=rewards_bt.device)
        for k in range(n_step):
            if t + k >= T:
                break
            g = g + (gamma ** k) * rewards_bt[:, t + k, :]
        out[:, t, :] = g
    return out


# ---------------------------
# 4) KL for diagonal Gaussian posterior
# ---------------------------

def kl_diag_gaussian_to_stdnormal(mu: Tensor, logvar: Tensor, reduce: str = "mean") -> Tensor:
    """
    KL( N(mu, sigma^2) || N(0, I) ) for diagonal Gaussian.
    mu/logvar: [B, D]
    reduce:
        - "none": [B]
        - "mean": scalar
        - "sum": scalar
    """
    # kl per dimension: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # [B, D]
    kl_per_sample = kl_per_dim.sum(dim=-1)  # [B]

    if reduce == "none":
        return kl_per_sample
    if reduce == "mean":
        return kl_per_sample.mean()
    if reduce == "sum":
        return kl_per_sample.sum()
    raise ValueError(f"Unknown reduce: {reduce}")


# ---------------------------
# 5) EMA + KL homeostasis
# ---------------------------

@dataclass
class EMA:
    def __init__(self, decay=0.99, init=0.0):
        self.decay = float(decay)
        self.value = float(init)
        self.inited = False

    def update(self, x: float) -> float:
        x = float(x)
        if not self.inited:
            self.value = x
            self.inited = True
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * x
        return self.value

    def get(self) -> float:
        return float(self.value)


@torch.no_grad()
def update_beta_homeostasis(
    beta: Union[float, Tensor],
    kl_ema: Union[float, Tensor],
    capacity: float,
    lr: float,
    min_beta: float = 0.0,
    max_beta: Optional[float] = None,
) -> Union[float, Tensor]:
    """
    Homeostatic update:
        beta <- clamp( beta + lr * (kl_ema - capacity), [min_beta, max_beta] )

    beta can be float or 0-dim tensor. Return keeps the same type as input beta.
    """
    beta_is_tensor = torch.is_tensor(beta)
    if not beta_is_tensor:
        beta_t = torch.tensor(float(beta))
    else:
        beta_t = beta

    if not torch.is_tensor(kl_ema):
        kl_t = torch.tensor(float(kl_ema))
    else:
        kl_t = kl_ema

    beta_new = beta_t + lr * (kl_t - float(capacity))
    if max_beta is None:
        beta_new = torch.clamp(beta_new, min=min_beta)
    else:
        beta_new = torch.clamp(beta_new, min=min_beta, max=float(max_beta))

    if beta_is_tensor:
        return beta_new
    return float(beta_new.item())


# ---------------------------
# 6) Window buffer (for causal Transformer / windowed inference)
# ---------------------------

class WindowBuffer:
    """
    Keep last K tokens (each token is [B, D]) and return a padded window [B, K, D].

    This is meant for:
    - slow Transformer over time with fixed window K_s
    - short-window inference when you don't want to store full episode

    Notes:
    - This buffer stores tensors; you decide whether to detach before append.
    """

    def __init__(self, max_len: int):
        assert max_len >= 1
        self.max_len = int(max_len)
        self._tokens: List[Tensor] = []

    def reset(self):
        self._tokens = []

    def __len__(self) -> int:
        return len(self._tokens)

    def append(self, token: Tensor):
        """
        token: [B, D]
        """
        if token.dim() != 2:
            raise ValueError(f"WindowBuffer expects token [B, D], got shape {tuple(token.shape)}")
        self._tokens.append(token)
        if len(self._tokens) > self.max_len:
            self._tokens = self._tokens[-self.max_len :]

    def get(self, pad_value: float = 0.0) -> Tensor:
        """
        Returns:
            window: [B, K, D] (K = max_len), left-padded if not enough tokens.
        """
        if len(self._tokens) == 0:
            raise RuntimeError("WindowBuffer is empty. Append at least one token before get().")

        B, D = self._tokens[-1].shape
        device = self._tokens[-1].device
        dtype = self._tokens[-1].dtype

        K = self.max_len
        window = torch.full((B, K, D), fill_value=pad_value, device=device, dtype=dtype)

        tokens = self._tokens[-K:]
        t_len = len(tokens)
        # right-align (most recent at end)
        window[:, K - t_len : K, :] = torch.stack(tokens, dim=1)  # [B, t_len, D]
        return window

    def get_with_mask(self, pad_value: float = 0.0):
        """
        Returns:
            window: [B, K, D]
            pad_mask: [B, K] bool, True for PAD positions
        """
        if len(self._tokens) == 0:
            raise RuntimeError("WindowBuffer is empty. Append at least one token before get_with_mask().")

        B, D = self._tokens[-1].shape
        device = self._tokens[-1].device
        dtype = self._tokens[-1].dtype
        K = self.max_len

        window = torch.full((B, K, D), fill_value=pad_value, device=device, dtype=dtype)
        pad_mask = torch.ones((B, K), device=device, dtype=torch.bool)  # True=PAD

        tokens = self._tokens[-K:]
        t_len = len(tokens)
        window[:, K - t_len: K, :] = torch.stack(tokens, dim=1)
        pad_mask[:, K - t_len: K] = False
        return window, pad_mask
