import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models.varibad_model import VariBADIndustrialModel


# ========== offline npz parsing (match run_td3bc-style dataset) ==========

def _safe_to_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float32)
    if x.ndim == 2 and x.shape[1] == 1:
        return x.astype(np.float32)
    return x.reshape(x.shape[0], -1).sum(axis=1, keepdims=True).astype(np.float32)


def _build_valid_mask(task_obs: np.ndarray, valid_index: int) -> np.ndarray:
    # task_obs: [T, num_pad, task_dim]
    v = task_obs[:, :, valid_index]
    return (v > 0).astype(np.float32)


def build_raw_obs_vec_from_arrays(
    task_obs: np.ndarray,
    worker_loads: np.ndarray,
    worker_profile: np.ndarray,
    valid_index: int = 3,
) -> np.ndarray:
    """
    对齐 run_varibad.py 的 _build_raw_obs_vec：
      raw_obs = [flatten(task_queue), flatten(worker_loads), flatten(profile), flatten(valid_mask)]
    这里只是把它做成 [T, obs_dim_raw] 版本。:contentReference[oaicite:1]{index=1}
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


def extract_episode_arrays(
    npz_path: Path,
    num_layers: int,
    q_obs_lid: int,
    q_action_lid: int,
    q_reward_source: str,
    valid_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 offline npz 抽取一条 episode 的序列，用于 BAD 预训练：
      obs_raw      [T, obs_dim_raw]  (含 valid_mask)
      actions_flat [T, action_dim]
      rewards      [T, 1]
      dones        [T, 1]  (取 l0_dones)
    """
    data = np.load(str(npz_path), allow_pickle=True)
    T = int(data["T"])

    # obs_raw from chosen layer
    p_obs = f"l{q_obs_lid}_"
    task_obs = data[p_obs + "task_obs"]
    worker_loads = data[p_obs + "worker_loads"]
    worker_profile = data[p_obs + "worker_profile"]
    obs_raw = build_raw_obs_vec_from_arrays(task_obs, worker_loads, worker_profile, valid_index=valid_index)

    # action_flat from chosen layer
    p_act = f"l{q_action_lid}_"
    actions = data[p_act + "actions"]
    actions_flat = actions.reshape(T, -1).astype(np.float32)

    # reward
    if q_reward_source == "sum":
        r = np.zeros((T, 1), dtype=np.float32)
        for lid in range(num_layers):
            r += _safe_to_col(data[f"l{lid}_rewards"])
        rewards = r
    else:
        lid = int(q_reward_source[1:])  # "l0" -> 0
        rewards = _safe_to_col(data[f"l{lid}_rewards"])

    # done (use l0)
    dones = _safe_to_col(data["l0_dones"])
    dones = (dones > 0.5).astype(np.float32)

    return obs_raw.astype(np.float32), actions_flat.astype(np.float32), rewards.astype(np.float32), dones.astype(np.float32)


def infer_dims_from_first_npz(
    root_dir: Path,
    num_layers: int,
    q_obs_lid: int,
    q_action_lid: int,
    valid_index: int,
) -> Tuple[int, int]:
    files = sorted(root_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz found in: {root_dir}")

    d = np.load(str(files[0]), allow_pickle=True)

    p = f"l{q_obs_lid}_"
    task_obs0 = d[p + "task_obs"]        # [T, num_pad, task_dim]
    worker_loads0 = d[p + "worker_loads"]  # [T, n_worker, load_dim]
    worker_profile0 = d[p + "worker_profile"]  # [T, ...]
    num_pad = task_obs0.shape[1]
    task_dim = task_obs0.shape[2]
    n_worker = worker_loads0.shape[1]
    load_dim = worker_loads0.shape[2]
    profile_dim_flat = int(np.prod(worker_profile0.shape[1:]))

    # raw_obs dim = flatten(task_queue)+flatten(worker_loads)+flatten(profile)+flatten(valid_mask)
    # 这跟 run_varibad.py 的 _init_worker 计算一致。:contentReference[oaicite:2]{index=2}
    obs_dim_raw = num_pad * task_dim + n_worker * load_dim + profile_dim_flat + num_pad

    pa = f"l{q_action_lid}_"
    actions0 = d[pa + "actions"]
    action_dim = int(np.prod(actions0.shape[1:]))

    return obs_dim_raw, action_dim


# ========== sampling windows ==========

class EpisodeCache:
    """小缓存，避免每个 iter 都重复 load npz。"""
    def __init__(self, capacity: int = 64):
        self.capacity = int(capacity)
        self._dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._keys = []

    def get(self, key: str, loader_fn):
        if key in self._dict:
            return self._dict[key]
        val = loader_fn()
        self._dict[key] = val
        self._keys.append(key)
        if len(self._keys) > self.capacity:
            old = self._keys.pop(0)
            self._dict.pop(old, None)
        return val


def sample_batch_windows(
    files: list,
    cache: EpisodeCache,
    batch_size: int,
    seq_len: int,
    num_layers: int,
    q_obs_lid: int,
    q_action_lid: int,
    q_reward_source: str,
    valid_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 batch windows：
      obs_raw  [B, L, obs_dim]
      actions  [B, L, act_dim]
      rewards  [B, L, 1]
      dones    [B, L, 1]
    """
    B = batch_size
    L = seq_len

    obs_list, act_list, rew_list, done_list = [], [], [], []

    for _ in range(B):
        p = str(files[np.random.randint(0, len(files))])

        def _loader():
            return extract_episode_arrays(
                Path(p),
                num_layers=num_layers,
                q_obs_lid=q_obs_lid,
                q_action_lid=q_action_lid,
                q_reward_source=q_reward_source,
                valid_index=valid_index,
            )

        obs_raw, actions, rewards, dones = cache.get(p, _loader)

        T = obs_raw.shape[0]
        if T < L:
            # 太短就重抽
            continue

        start = np.random.randint(0, T - L + 1)
        obs_list.append(obs_raw[start:start + L])
        act_list.append(actions[start:start + L])
        rew_list.append(rewards[start:start + L])
        done_list.append(dones[start:start + L])

    if len(obs_list) == 0:
        raise RuntimeError("No valid episodes/windows sampled. Maybe seq_len is too large?")

    obs_b = np.stack(obs_list, axis=0)
    act_b = np.stack(act_list, axis=0)
    rew_b = np.stack(rew_list, axis=0)
    done_b = np.stack(done_list, axis=0)
    return obs_b, act_b, rew_b, done_b


# ========== BAD pretrain objective (reward prediction) ==========

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(N(mu, sigma)||N(0,1)) per-sample, summed over z dim:
      0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    """
    return 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar).sum(dim=1, keepdim=True)


def train_step_reward_pred(
    model: VariBADIndustrialModel,
    obs_raw: torch.Tensor,     # [B,L,obs_dim]
    actions: torch.Tensor,     # [B,L,act_dim]
    rewards: torch.Tensor,     # [B,L,1]
    dones: torch.Tensor,       # [B,L,1]
    thr: float,
    temp: float,
    kl_coef: float,
    smooth_coef: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    训练目标：用 belief 产生 z_t，然后用 decoder 预测 reward。
    为了与 MiBReW 的 q 定义一致，这里预测 r_hat_t 用的是 (s_t, a_{t-1}, z_t)。
    """
    B, L, _ = obs_raw.shape

    h = torch.zeros((B, model.belief_hidden), device=obs_raw.device, dtype=torch.float32)
    a_prev = torch.zeros((B, model.action_dim), device=obs_raw.device, dtype=torch.float32)
    r_prev = torch.zeros((B, 1), device=obs_raw.device, dtype=torch.float32)
    d_prev = torch.zeros((B, 1), device=obs_raw.device, dtype=torch.float32)

    loss_mse = 0.0
    loss_kl = 0.0
    loss_smooth = 0.0

    q_prev: Optional[torch.Tensor] = None

    for t in range(L):
        # run_varibad 逻辑：上一刻 done 则 reset belief（用 mask 清零 h）:contentReference[oaicite:3]{index=3}
        h = h * (1.0 - d_prev)

        s_embed = model.encode_obs(obs_raw[:, t, :])
        h, mu, logvar, z = model.belief_step(s_embed, a_prev, r_prev, d_prev, h)

        # 关键：预测当前 reward 用 a_{t-1}（a_prev），这样部署/评估也能一致计算 q
        _, r_hat = model.decode(s_embed, a_prev, z)  # [B,1]

        r_t = rewards[:, t, :]  # [B,1]
        loss_mse = loss_mse + F.mse_loss(r_hat, r_t)

        loss_kl = loss_kl + kl_divergence(mu, logvar).mean()

        # smooth on q (optional)
        q_t = torch.sigmoid((thr - r_hat.abs()) / max(temp, 1e-8))
        if q_prev is not None:
            loss_smooth = loss_smooth + F.mse_loss(q_t, q_prev)
        q_prev = q_t.detach()  # smooth 不需要跨步反传

        # update prev feedback with dataset action/reward/done
        a_prev = actions[:, t, :]
        r_prev = r_t
        d_prev = dones[:, t, :]

    loss_mse = loss_mse / L
    loss_kl = loss_kl / L
    loss_smooth = loss_smooth / max(L - 1, 1)

    loss = loss_mse + kl_coef * loss_kl + smooth_coef * loss_smooth

    info = {
        "loss": float(loss.item()),
        "mse": float(loss_mse.item()),
        "kl": float(loss_kl.item()),
        "smooth": float(loss_smooth.item()),
    }
    return loss, info


# ========== main ==========

def main():
    parser = argparse.ArgumentParser()

    # dataset layout (match run_mibrew)
    parser.add_argument("--dire", type=str, default="standard")
    parser.add_argument("--dataset", type=str, default="expert")
    parser.add_argument("--offline_data_root", type=str, default="../offline_data/crescent")
    parser.add_argument("--num_layers", type=int, default=3)

    # which layer's (obs, action, reward) to train belief on
    parser.add_argument("--q_obs_lid", type=int, default=0)
    parser.add_argument("--q_action_lid", type=int, default=0)
    parser.add_argument("--q_reward_source", type=str, default="l0", choices=["l0", "l1", "l2", "sum"])
    parser.add_argument("--valid_index", type=int, default=3)

    # model hparams (must match your later inference usage)
    parser.add_argument("--obs_embed_dim", type=int, default=256)      # run_varibad uses ppo_config["hidden_dim"] as obs_embed_dim :contentReference[oaicite:4]{index=4}
    parser.add_argument("--belief_hidden", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--decoder_hidden", type=int, default=256)
    parser.add_argument("--policy_hidden", type=int, default=256)
    parser.add_argument("--value_hidden", type=int, default=256)

    # training hparams
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=10.0)

    parser.add_argument("--train_steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)

    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--smooth_coef", type=float, default=0.0)

    # q mapping hyperparams (used in smooth loss, and consistent with run_mibrew q definition)
    parser.add_argument("--q_thr", type=float, default=1e-3)
    parser.add_argument("--q_temp", type=float, default=5e-3)

    # logging & save
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_dir", type=str, default="../logs/bad_pretrain")
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoints/bad")
    parser.add_argument("--save_name", type=str, default="bad.pt")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    root_dir = Path(args.offline_data_root) / args.dire / args.dataset
    files = sorted(root_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files under: {root_dir}")

    obs_dim_raw, action_dim = infer_dims_from_first_npz(
        root_dir=root_dir,
        num_layers=args.num_layers,
        q_obs_lid=args.q_obs_lid,
        q_action_lid=args.q_action_lid,
        valid_index=args.valid_index,
    )

    model = VariBADIndustrialModel(
        obs_dim=obs_dim_raw,
        action_dim=action_dim,
        obs_embed_dim=args.obs_embed_dim,
        belief_hidden=args.belief_hidden,
        z_dim=args.z_dim,
        policy_hidden=args.policy_hidden,
        value_hidden=args.value_hidden,
        decoder_hidden=args.decoder_hidden,
    ).to(device)
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # TensorBoard (optional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        run_name = f"{args.dire}_{args.dataset}_obsL{args.q_obs_lid}_actL{args.q_action_lid}_{time.strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))
    except Exception:
        writer = None

    cache = EpisodeCache(capacity=64)
    t0 = time.time()

    def _log(k: str, v: float, step: int):
        if writer is not None:
            writer.add_scalar(k, v, step)

    for step in range(1, args.train_steps + 1):
        obs_b, act_b, rew_b, done_b = sample_batch_windows(
            files=files,
            cache=cache,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_layers=args.num_layers,
            q_obs_lid=args.q_obs_lid,
            q_action_lid=args.q_action_lid,
            q_reward_source=args.q_reward_source,
            valid_index=args.valid_index,
        )

        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_b, dtype=torch.float32, device=device)
        rew_t = torch.tensor(rew_b, dtype=torch.float32, device=device)
        done_t = torch.tensor(done_b, dtype=torch.float32, device=device)

        loss, info = train_step_reward_pred(
            model=model,
            obs_raw=obs_t,
            actions=act_t,
            rewards=rew_t,
            dones=done_t,
            thr=args.q_thr,
            temp=args.q_temp,
            kl_coef=args.kl_coef,
            smooth_coef=args.smooth_coef,
        )

        optim.zero_grad()
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optim.step()

        if step % args.log_interval == 0:
            elapsed = (time.time() - t0) / 60.0
            print(
                f"[step {step:07d}] "
                f"loss={info['loss']:.4f} mse={info['mse']:.4f} kl={info['kl']:.4f} smooth={info['smooth']:.4f} "
                f"elapsed={elapsed:.1f}min"
            )
            _log("train/loss", info["loss"], step)
            _log("train/mse", info["mse"], step)
            _log("train/kl", info["kl"], step)
            _log("train/smooth", info["smooth"], step)

        if step % args.save_interval == 0 or step == args.train_steps:
            ckpt_dir = Path(args.ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / args.dire / args.save_name

            torch.save(
                {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "obs_dim_raw": int(obs_dim_raw),
                    "action_dim": int(action_dim),
                    "config": vars(args),
                    "note": "MiBReW BAD pretrain (reward prediction on offline dataset)",
                },
                str(ckpt_path),
            )
            print(f"[save] step={step} -> {ckpt_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
