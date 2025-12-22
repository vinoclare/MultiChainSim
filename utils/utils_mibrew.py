# utils/utils_mibrew.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# 复用你现有 BAD(VariBAD) 实现
from models.varibad_model import VariBADIndustrialModel
from algs.varibad import VariBAD


# =========================
# basic helpers
# =========================

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def _safe_to_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float32)
    if x.ndim == 2 and x.shape[1] == 1:
        return x.astype(np.float32)
    # [T, k] -> sum to scalar
    return x.reshape(x.shape[0], -1).sum(axis=1, keepdims=True).astype(np.float32)


def _build_valid_mask(task_obs: np.ndarray, valid_index: int) -> np.ndarray:
    # task_obs: [T, num_pad, task_dim]
    # valid_mask: [T, num_pad]
    v = task_obs[:, :, valid_index]
    return (v > 0).astype(np.float32)


def build_raw_obs_vec_from_arrays(
    task_obs: np.ndarray,
    worker_loads: np.ndarray,
    worker_profile: np.ndarray,
    valid_index: int = 3,
) -> np.ndarray:
    """
    对齐你 run_varibad.py/_build_raw_obs_vec 的 raw_obs：
      raw_obs = [flatten(task_queue), flatten(worker_loads), flatten(profile), flatten(valid_mask)]
    输入（来自 offline npz）：
      task_obs:       [T, num_pad, task_dim]
      worker_loads:   [T, n_worker, load_dim]
      worker_profile: [T, ...]  (会 flatten)
    输出：
      raw_obs: [T, obs_dim_raw]
    """
    T = task_obs.shape[0]
    valid_mask = _build_valid_mask(task_obs, valid_index=valid_index)  # [T, num_pad]

    parts = [
        task_obs.reshape(T, -1).astype(np.float32),
        worker_loads.reshape(T, -1).astype(np.float32),
        worker_profile.reshape(T, -1).astype(np.float32),
        valid_mask.reshape(T, -1).astype(np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_raw_obs_vec_from_env_obs(layer_obs: dict, valid_index: int = 3) -> np.ndarray:
    """
    用于在线 eval：从 env 的 dict obs 构造 raw_obs vec（与 build_raw_obs_vec_from_arrays 一致）
    """
    task_obs = np.asarray(layer_obs["task_queue"], dtype=np.float32)[None, ...]         # [1, num_pad, task_dim]
    worker_loads = np.asarray(layer_obs["worker_loads"], dtype=np.float32)[None, ...]  # [1, n_worker, load_dim]
    worker_profile = np.asarray(layer_obs["worker_profile"], dtype=np.float32)[None, ...]
    raw = build_raw_obs_vec_from_arrays(task_obs, worker_loads, worker_profile, valid_index=valid_index)
    return raw[0]  # [obs_dim_raw]


# =========================
# q inferencers
# =========================

class BaseQInferencer:
    def infer_q_sequence(self, obs_raw: np.ndarray, actions_flat: np.ndarray, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RewardHeuristicQInferencer(BaseQInferencer):
    """
    fallback：不依赖 BAD 的简易 q（保证 pipeline 能跑通）
    q 越大表示“越像 sparse 段”：最近 reward 越接近 0，则 q 越接近 1
    """
    def __init__(self, window: int = 10, thr: float = 1e-3, temp: float = 5e-3):
        self.window = int(window)
        self.thr = float(thr)
        self.temp = float(temp)

    def infer_q_sequence(self, obs_raw, actions_flat, rewards, dones):
        r = np.asarray(rewards).reshape(-1).astype(np.float32)
        T = r.shape[0]
        q = np.zeros((T, 1), dtype=np.float32)
        for t in range(T):
            s = max(0, t - self.window + 1)
            m = float(np.mean(np.abs(r[s:t + 1])))
            q[t, 0] = 1.0 / (1.0 + np.exp((m - self.thr) / max(self.temp, 1e-8)))
        return q


class VariBADQInferencer(BaseQInferencer):
    """
    复用你现有 VariBAD belief 模型来产生标量 q_t（unlabelled regime）。

    核心做法：
      1) belief_step 用 (obs_t, a_{t-1}, r_{t-1}, done_{t-1}) -> z_t
      2) 用 decoder 的 reward head 预测 r_hat_t = g(obs_t, a_{t-1}, z_t)
      3) q_t = sigmoid((thr - |r_hat_t|)/temp)

    注意：这样 q_t 不依赖未知的 a_t，因此部署/评估也能一致计算。
    """
    def __init__(
        self,
        ckpt_path: str,
        obs_dim_raw: int,
        action_dim: int,
        obs_embed_dim: int = 256,
        belief_hidden: int = 128,
        z_dim: int = 64,
        decoder_hidden: int = 256,
        policy_hidden: int = 256,  # 仅用于构造模型形状（虽然我们不用 policy head）
        value_hidden: int = 256,
        device: str = "cpu",
        valid_index: int = 3,
        thr: float = 1e-3,
        temp: float = 5e-3,
        use_abs: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.valid_index = int(valid_index)
        self.thr = float(thr)
        self.temp = float(temp)
        self.use_abs = bool(use_abs)

        self.model = VariBADIndustrialModel(
            obs_dim=obs_dim_raw,
            action_dim=action_dim,
            obs_embed_dim=obs_embed_dim,
            belief_hidden=belief_hidden,
            z_dim=z_dim,
            policy_hidden=policy_hidden,
            value_hidden=value_hidden,
            decoder_hidden=decoder_hidden,
        ).to(self.device)

        sd = torch.load(ckpt_path, map_location="cpu")
        state_dict = self._extract_model_state_dict(sd)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # strict=False 只是为了兼容你可能存了 algo/optim 之类的键；
        # 如果维度不匹配，torch 仍然会报错（这才是我们想要的 fail-fast）。

        self.model.eval()

        # 在线推理缓存（用于 eval）
        self._h = None
        self._prev_a = None
        self._prev_r = None
        self._prev_done = None

    @staticmethod
    def _extract_model_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
        """
        兼容几种常见保存格式：
          - 直接就是 model.state_dict()
          - {"model_state_dict": ...}
          - {"state_dict": ...}
          - {"model": ...}
        """
        if isinstance(ckpt_obj, dict):
            for k in ["model_state_dict", "state_dict", "model"]:
                if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                    return ckpt_obj[k]
            # 直接像 state_dict
            if all(torch.is_tensor(v) for v in ckpt_obj.values()):
                return ckpt_obj
        raise ValueError(
            "Unrecognized BAD ckpt format. Expect a model state_dict or a dict containing "
            "'model_state_dict'/'state_dict'/'model'."
        )

    @torch.no_grad()
    def infer_q_sequence(self, obs_raw: np.ndarray, actions_flat: np.ndarray, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        obs_raw:      [T, obs_dim_raw]
        actions_flat: [T, action_dim]
        rewards:      [T, 1]
        dones:        [T, 1]
        return q:     [T, 1] in (0,1)
        """
        obs_raw_t = torch.tensor(obs_raw, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(actions_flat, dtype=torch.float32, device=self.device)
        rew_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        T = obs_raw_t.shape[0]
        h = torch.zeros((1, self.model.belief_hidden), dtype=torch.float32, device=self.device)
        a_prev = torch.zeros((1, self.model.action_dim), dtype=torch.float32, device=self.device)
        r_prev = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        d_prev = torch.zeros((1, 1), dtype=torch.float32, device=self.device)

        qs = []
        for t in range(T):
            # 跟 VariBAD.sample 一致：如果上一刻 done，把 hidden 清零
            h = h * (1.0 - d_prev)

            s_embed = self.model.encode_obs(obs_raw_t[t:t + 1])  # [1, embed]
            h, mu, logvar, z = self.model.belief_step(s_embed, a_prev, r_prev, d_prev, h)

            # 用“上一动作 a_prev”做 reward 预测（不依赖未知 a_t）
            _, r_hat = self.model.decode(s_embed, a_prev, z)  # r_hat: [1,1]
            x = r_hat.abs() if self.use_abs else r_hat
            q = _sigmoid((self.thr - x) / max(self.temp, 1e-8))  # [1,1]
            qs.append(q)

            # 更新 prev（用数据集里的真实 a_t, r_t, done_t，供下一步 belief 用）
            a_prev = act_t[t:t + 1]
            r_prev = rew_t[t:t + 1]
            d_prev = done_t[t:t + 1]

        q_seq = torch.cat(qs, dim=0).clamp(1e-6, 1.0 - 1e-6)
        return q_seq.detach().cpu().numpy().astype(np.float32)

    # ---------- online eval helper (stateful) ----------

    def reset_online(self):
        self._h = torch.zeros((1, self.model.belief_hidden), dtype=torch.float32, device=self.device)
        self._prev_a = torch.zeros((1, self.model.action_dim), dtype=torch.float32, device=self.device)
        self._prev_r = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        self._prev_done = torch.zeros((1, 1), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def compute_q_from_env_obs(self, layer_obs: dict) -> float:
        """
        先用当前 obs + prev feedback 更新 belief，再输出当前步 q_t
        """
        if self._h is None:
            self.reset_online()

        raw = build_raw_obs_vec_from_env_obs(layer_obs, valid_index=self.valid_index)
        raw_t = torch.tensor(raw[None, :], dtype=torch.float32, device=self.device)

        self._h = self._h * (1.0 - self._prev_done)
        s_embed = self.model.encode_obs(raw_t)
        self._h, mu, logvar, z = self.model.belief_step(s_embed, self._prev_a, self._prev_r, self._prev_done, self._h)

        _, r_hat = self.model.decode(s_embed, self._prev_a, z)
        x = r_hat.abs() if self.use_abs else r_hat
        q = _sigmoid((self.thr - x) / max(self.temp, 1e-8))
        qv = float(q.item())
        qv = max(1e-6, min(1.0 - 1e-6, qv))
        return qv

    def set_prev_feedback(self, action_flat: np.ndarray, reward: float, done: float):
        """
        在 env.step 后调用，把 a_t, r_t, done_t 喂回 belief
        """
        if self._h is None:
            self.reset_online()
        a = np.asarray(action_flat, dtype=np.float32).reshape(1, -1)
        self._prev_a[:] = torch.tensor(a, dtype=torch.float32, device=self.device)
        self._prev_r[:] = float(reward)
        self._prev_done[:] = float(done)


# =========================
# q-cache & dataset wrapper
# =========================

@dataclass
class QCache:
    q: np.ndarray       # [N,1]
    q_next: np.ndarray  # [N,1]
    p_s: float          # mean(q)


def extract_episode_for_q_from_td3bc_npz(
    data: np.lib.npyio.NpzFile,
    num_layers: int,
    q_obs_lid: int = 0,
    q_action_lid: int = 0,
    q_reward_source: str = "l0",
    valid_index: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 run_td3bc.py 的 offline npz 格式抽出用于 q 推理的序列：
      obs_raw:      [T, obs_dim_raw]  (含 valid_mask)
      actions_flat: [T, action_dim]
      rewards:      [T, 1]
      dones:        [T, 1]  (取 l0_dones)
    """
    T = int(data["T"])

    # obs raw（用 q_obs_lid 的 obs 产生）
    p_obs = f"l{q_obs_lid}_"
    task_obs = data[p_obs + "task_obs"]
    worker_loads = data[p_obs + "worker_loads"]
    worker_profile = data[p_obs + "worker_profile"]
    obs_raw = build_raw_obs_vec_from_arrays(task_obs, worker_loads, worker_profile, valid_index=valid_index)

    # action（用 q_action_lid 的动作）
    p_act = f"l{q_action_lid}_"
    actions = data[p_act + "actions"]
    actions_flat = actions.reshape(T, -1).astype(np.float32)

    # reward（默认用某一层 reward，避免跟 belief 训练分布不一致）
    if q_reward_source == "sum":
        r = np.zeros((T, 1), dtype=np.float32)
        for lid in range(num_layers):
            rr = _safe_to_col(data[f"l{lid}_rewards"])
            r += rr
        rewards = r
    else:
        # "l0"/"l1"/...
        lid = int(q_reward_source[1:])
        rewards = _safe_to_col(data[f"l{lid}_rewards"])

    # done（通常一致，取 l0）
    dones = _safe_to_col(data["l0_dones"])
    dones = (dones > 0.5).astype(np.float32)

    return obs_raw, actions_flat, rewards.astype(np.float32), dones.astype(np.float32)


def build_q_cache(
    root_dir: Path,
    num_layers: int,
    inferencer: BaseQInferencer,
    out_path: Path,
    q_obs_lid: int = 0,
    q_action_lid: int = 0,
    q_reward_source: str = "l0",
    valid_index: int = 3,
    force_rebuild: bool = False,
) -> QCache:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force_rebuild:
        d = np.load(str(out_path), allow_pickle=True)
        q = np.asarray(d["q"], dtype=np.float32)
        q_next = np.asarray(d["q_next"], dtype=np.float32)
        p_s = float(d["p_s"])
        return QCache(q=q, q_next=q_next, p_s=p_s)

    files = sorted(root_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {root_dir}")

    q_list, qn_list = [], []

    for f in files:
        ep = np.load(str(f), allow_pickle=True)
        obs_raw, actions_flat, rewards, dones = extract_episode_for_q_from_td3bc_npz(
            ep,
            num_layers=num_layers,
            q_obs_lid=q_obs_lid,
            q_action_lid=q_action_lid,
            q_reward_source=q_reward_source,
            valid_index=valid_index,
        )
        q_seq = inferencer.infer_q_sequence(obs_raw, actions_flat, rewards, dones)
        q_seq = np.asarray(q_seq, dtype=np.float32).reshape(-1, 1)
        q_seq = np.clip(q_seq, 1e-6, 1.0 - 1e-6)

        q_next = np.concatenate([q_seq[1:], q_seq[-1:]], axis=0).astype(np.float32)

        q_list.append(q_seq)
        qn_list.append(q_next)

    q = np.concatenate(q_list, axis=0)
    q_next = np.concatenate(qn_list, axis=0)
    p_s = float(np.mean(q))

    np.savez_compressed(str(out_path), q=q, q_next=q_next, p_s=np.array(p_s, dtype=np.float32))
    return QCache(q=q, q_next=q_next, p_s=p_s)


class OfflineDatasetWithQ:
    """
    不改 run_td3bc.py 的 OfflineDataset，外面包一层加 q/q_next。
    base.sample() 仍然是 (s,a,r,s2,d)
    这里 sample() 返回 (s,a,r,s2,d,q,qn)
    """
    def __init__(self, base_dataset, q: np.ndarray, q_next: np.ndarray, device: torch.device):
        self.base = base_dataset
        self.size = int(base_dataset.size)

        assert q.shape[0] == self.size, f"q size mismatch: {q.shape[0]} vs {self.size}"
        assert q_next.shape[0] == self.size, f"q_next size mismatch: {q_next.shape[0]} vs {self.size}"

        self.q = torch.tensor(q, dtype=torch.float32, device=device)
        self.q_next = torch.tensor(q_next, dtype=torch.float32, device=device)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.q.device)
        s = self.base.states[idx]
        a = self.base.actions[idx]
        r = self.base.rewards[idx]
        s2 = self.base.next_states[idx]
        d = self.base.dones[idx]
        return s, a, r, s2, d, self.q[idx], self.q_next[idx]


def attach_qcache_to_datasets(datasets: Dict[int, Any], qcache: QCache, device: torch.device) -> Dict[int, OfflineDatasetWithQ]:
    return {lid: OfflineDatasetWithQ(ds, qcache.q, qcache.q_next, device=device) for lid, ds in datasets.items()}


def compute_regime_weights(q: torch.Tensor, p_s: float, eps: float = 1e-3) -> torch.Tensor:
    """
    w_t = q/(p_s+eps) + (1-q)/(1-p_s+eps)
    q: [B,1]
    """
    p = float(np.clip(p_s, eps, 1.0 - eps))
    q = q.clamp(min=1e-6, max=1.0 - 1e-6)
    w = q / (p + eps) + (1.0 - q) / ((1.0 - p) + eps)
    return w.detach()
