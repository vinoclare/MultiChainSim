import os
import glob
import json
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from envs import IndustrialChain
from envs.env import MultiplexEnv


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_obs(task_q, worker_loads, worker_profile):
    """
    task_q: [30, 7], worker_loads: [8, 4], worker_profile: [8, 6]
    -> obs_vec: [210 + 32 + 48] = [290]
    """
    return np.concatenate(
        [
            task_q.reshape(-1).astype(np.float32),
            worker_loads.reshape(-1).astype(np.float32),
            worker_profile.reshape(-1).astype(np.float32),
        ],
        axis=0,
    )


def to_tanh_action(a01: np.ndarray) -> np.ndarray:
    """[0,1] -> [-1,1]"""
    return (a01 * 2.0 - 1.0).astype(np.float32)


def to_01_action(a_tanh: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]"""
    return (a_tanh + 1.0) * 0.5


# =========================
# Offline Dataset (npz)
# =========================
class OfflineNpzReplay:
    """
    从一个目录中读取 *.npz，合并成离线 replay：
        (s, a, r, s', done)

    约定 npz 内字段：
        l{lid}_task_obs, l{lid}_worker_loads, l{lid}_worker_profile
        l{lid}_next_task_obs, l{lid}_next_worker_loads, l{lid}_next_worker_profile
        l{lid}_actions  (in [0,1], shape [T,8,30])
        l{lid}_rewards  (shape [T])
        l{lid}_dones    (shape [T])
    """

    def __init__(self, offline_dir: str, layer_id: int):
        pattern = os.path.join(offline_dir, "*.npz")
        files = sorted(glob.glob(pattern))
        if not files:
            raise RuntimeError(f"No npz files found under: {offline_dir}")

        s_list, a_list, r_list, sp_list, d_list = [], [], [], [], []

        nz_task = 0
        nz_prof = 0

        for fp in files:
            data = np.load(fp, allow_pickle=True)
            t = data["T"]
            if isinstance(t, np.ndarray):
                T = int(t.item()) if t.ndim == 0 else int(t[0])
            else:
                T = int(t)

            task = data[f"l{layer_id}_task_obs"][:T]
            loads = data[f"l{layer_id}_worker_loads"][:T]
            prof = data[f"l{layer_id}_worker_profile"][:T]

            task_n = data[f"l{layer_id}_next_task_obs"][:T]
            loads_n = data[f"l{layer_id}_next_worker_loads"][:T]
            prof_n = data[f"l{layer_id}_next_worker_profile"][:T]

            act01 = data[f"l{layer_id}_actions"][:T]
            rew = data[f"l{layer_id}_rewards"][:T].astype(np.float32)
            done = data[f"l{layer_id}_dones"][:T].astype(np.float32)

            nz_task += int(np.count_nonzero(task))
            nz_prof += int(np.count_nonzero(prof))

            for tt in range(T):
                s = flatten_obs(task[tt], loads[tt], prof[tt])
                sp = flatten_obs(task_n[tt], loads_n[tt], prof_n[tt])
                a = to_tanh_action(act01[tt].reshape(-1))  # [240] in [-1,1]

                s_list.append(s)
                sp_list.append(sp)
                a_list.append(a)
                r_list.append(rew[tt])
                d_list.append(done[tt])

        self.s = np.stack(s_list, axis=0)  # [N, obs_dim]
        self.a = np.stack(a_list, axis=0)  # [N, act_dim]
        self.r = np.array(r_list, dtype=np.float32)[:, None]  # [N,1]
        self.sp = np.stack(sp_list, axis=0)  # [N, obs_dim]
        self.d = np.array(d_list, dtype=np.float32)[:, None]  # [N,1]

        self.N = self.s.shape[0]
        self.obs_dim = self.s.shape[1]
        self.act_dim = self.a.shape[1]

        print(
            f"[Replay L{layer_id}] dir={offline_dir} files={len(files)} samples={self.N} "
            f"obs_dim={self.obs_dim} act_dim={self.act_dim}"
        )
        print(
            f"[Replay L{layer_id}] nonzero task_queue total={nz_task}, nonzero worker_profile total={nz_prof}"
        )
        if nz_task == 0 or nz_prof == 0:
            print(
                f"[WARN] L{layer_id}: task_queue/profile seems all-zero in loaded data. Offline RL will likely be crippled."
            )

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.N, size=batch_size)
        s = torch.tensor(self.s[idx], device=device)
        a = torch.tensor(self.a[idx], device=device)
        r = torch.tensor(self.r[idx], device=device)
        sp = torch.tensor(self.sp[idx], device=device)
        d = torch.tensor(self.d[idx], device=device)
        return s, a, r, sp, d


class MixedReplay:
    """
    混合 4 份离线数据：
        (hitac_muse, crescent) × (expert, suboptimal)

    采样策略：
        batch 内先按 expert_frac 切 expert/suboptimal，
        再在各自子集中按 *_hitac_ratio 切 hitac/crescent。

    重要：如果某个数据集对应的“使用比例”为 0，则该数据集不会被加载（目录不存在/为空也不会报错）。
    """

    def __init__(
        self,
        hitac_expert_dir: str,
        hitac_subopt_dir: str,
        crescent_expert_dir: str,
        crescent_subopt_dir: str,
        layer_id: int,
        use_hitac_expert: bool = True,
        use_hitac_subopt: bool = True,
        use_crescent_expert: bool = True,
        use_crescent_subopt: bool = True,
    ):
        self.re_h_e = OfflineNpzReplay(hitac_expert_dir, layer_id) if use_hitac_expert else None
        self.re_h_s = OfflineNpzReplay(hitac_subopt_dir, layer_id) if use_hitac_subopt else None
        self.re_c_e = OfflineNpzReplay(crescent_expert_dir, layer_id) if use_crescent_expert else None
        self.re_c_s = OfflineNpzReplay(crescent_subopt_dir, layer_id) if use_crescent_subopt else None

        enabled = [
            ("hitac_expert", self.re_h_e),
            ("hitac_suboptimal", self.re_h_s),
            ("crescent_expert", self.re_c_e),
            ("crescent_suboptimal", self.re_c_s),
        ]

        base = None
        for _, r in enabled:
            if r is not None:
                base = r
                break
        if base is None:
            raise RuntimeError(
                "MixedReplay: all datasets are disabled by ratios. "
                "You must enable at least one dataset (i.e., some ratio > 0)."
            )

        self.obs_dim = base.obs_dim
        self.act_dim = base.act_dim
        for name, r in enabled:
            if r is None:
                continue
            if r.obs_dim != self.obs_dim or r.act_dim != self.act_dim:
                raise ValueError(f"MixedReplay: obs_dim/act_dim mismatch on dataset={name}")

        loaded = [name for name, r in enabled if r is not None]
        skipped = [name for name, r in enabled if r is None]
        print(f"[MixedReplay L{layer_id}] loaded={loaded} skipped={skipped}")

    def sample_mix(
        self,
        batch_size: int,
        expert_frac: float,
        expert_hitac_ratio: float,
        subopt_hitac_ratio: float,
        device: torch.device,
    ):
        n_e = int(round(batch_size * float(expert_frac)))
        n_s = batch_size - n_e

        n_e_h = int(round(n_e * float(expert_hitac_ratio)))
        n_e_c = n_e - n_e_h

        n_s_h = int(round(n_s * float(subopt_hitac_ratio)))
        n_s_c = n_s - n_s_h

        chunks = []
        masks = []

        def _append(name: str, replay: OfflineNpzReplay, n: int, is_expert: bool):
            if n <= 0:
                return
            if replay is None:
                raise RuntimeError(
                    f"MixedReplay.sample_mix: dataset '{name}' is disabled/unavailable, but n={n} samples are requested. "
                    f"Check your ratios (expert_frac/expert_hitac_ratio/subopt_hitac_ratio)."
                )
            s, a, r, sp, d = replay.sample(n, device)
            chunks.append((s, a, r, sp, d))
            masks.append(
                torch.ones((n, 1), device=device) if is_expert else torch.zeros((n, 1), device=device)
            )

        _append("hitac_expert", self.re_h_e, n_e_h, True)
        _append("crescent_expert", self.re_c_e, n_e_c, True)
        _append("hitac_suboptimal", self.re_h_s, n_s_h, False)
        _append("crescent_suboptimal", self.re_c_s, n_s_c, False)

        if not chunks:
            raise RuntimeError(
                "MixedReplay.sample_mix: empty batch (no dataset contributes). "
                "This usually means batch_size=0 or all ratios lead to 0 samples."
            )

        s = torch.cat([c[0] for c in chunks], dim=0)
        a = torch.cat([c[1] for c in chunks], dim=0)
        r = torch.cat([c[2] for c in chunks], dim=0)
        sp = torch.cat([c[3] for c in chunks], dim=0)
        d = torch.cat([c[4] for c in chunks], dim=0)
        expert_mask = torch.cat(masks, dim=0)

        perm = torch.randperm(s.shape[0], device=device)
        return s[perm], a[perm], r[perm], sp[perm], d[perm], expert_mask[perm]


# =========================
# Networks
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TanhGaussianPolicy(nn.Module):
    """
    SAC policy: a ~ tanh(N(mean, std))
    输出动作在 [-1,1]。给环境用时再映射到 [0,1]。
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden, hidden=hidden)
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        h = self.backbone(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        a = torch.tanh(pre_tanh)

        log_prob = (
            -0.5
            * (((pre_tanh - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        ).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return a, log_prob

    def deterministic(self, obs):
        mean, _ = self(obs)
        return torch.tanh(mean)

    def log_prob_action(self, obs, action_tanh):
        """
        计算给定动作 action_tanh (已在[-1,1]) 在当前策略下的 log π(a|s)
        用于 BC：最大似然而不是 MSE(mean, a)
        """
        mean, log_std = self(obs)
        std = log_std.exp()

        a = action_tanh.clamp(-0.999999, 0.999999)
        # atanh(a) = 0.5 * (log(1+a) - log(1-a))
        pre_tanh = 0.5 * (torch.log1p(a) - torch.log1p(-a))

        log_prob = (
            -0.5
            * (((pre_tanh - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        ).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return log_prob


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, 1, hidden=hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)


# =========================
# CQL-SAC Trainer (single layer)
# =========================
class CQLSAC:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
        tau: float,
        cql_alpha: float,
        cql_temp: float,
        cql_n_actions: int,
        cql_rand_std: float,
        device: torch.device,
        target_entropy: float = None,
        auto_alpha: bool = True,
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau

        self.policy = TanhGaussianPolicy(obs_dim, act_dim, hidden=hidden_dim).to(device)
        self.q1 = QNet(obs_dim, act_dim, hidden=hidden_dim).to(device)
        self.q2 = QNet(obs_dim, act_dim, hidden=hidden_dim).to(device)
        self.q1_t = QNet(obs_dim, act_dim, hidden=hidden_dim).to(device)
        self.q2_t = QNet(obs_dim, act_dim, hidden=hidden_dim).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.pi_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)

        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -float(act_dim)
        self.target_entropy = target_entropy

        self.log_alpha = torch.tensor(np.log(0.1), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.cql_alpha = float(cql_alpha)
        self.cql_temp = float(cql_temp)
        self.cql_n_actions = int(cql_n_actions)
        self.cql_rand_std = float(cql_rand_std)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def train_step(
        self,
        replay: MixedReplay,
        batch_size: int,
        expert_frac: float,
        expert_hitac_ratio: float,
        subopt_hitac_ratio: float,
        bc_coef: float,
    ):
        s, a, r, sp, d, expert_mask = replay.sample_mix(
            batch_size=batch_size,
            expert_frac=expert_frac,
            expert_hitac_ratio=expert_hitac_ratio,
            subopt_hitac_ratio=subopt_hitac_ratio,
            device=self.device,
        )

        # -----------------------------
        # Critic target
        # -----------------------------
        with torch.no_grad():
            ap, logp = self.policy.sample(sp)
            q1_tp = self.q1_t(sp, ap)
            q2_tp = self.q2_t(sp, ap)
            q_tp = torch.min(q1_tp, q2_tp) - self.alpha.detach() * logp
            y = r + (1.0 - d) * self.gamma * q_tp

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        bellman1 = F.mse_loss(q1, y)
        bellman2 = F.mse_loss(q2, y)

        # -----------------------------
        # CQL conservative penalty
        # -----------------------------
        B = s.shape[0]
        n = self.cql_n_actions

        # 关键改动：高维动作下，U[-1,1] 太 OOD，容易把 logsumexp 推爆
        # std==0 -> 兼容原行为（仍用U[-1,1]）
        if self.cql_rand_std > 0:
            noise = torch.randn(B, n, self.act_dim, device=self.device) * self.cql_rand_std
            a_base = a[:, None, :].expand(B, n, self.act_dim)
            a_rand = (a_base + noise).clamp(-1.0, 1.0)
        else:
            a_rand = torch.empty(B, n, self.act_dim, device=self.device).uniform_(-1.0, 1.0)

        s_rep = s[:, None, :].repeat(1, n, 1).reshape(B * n, self.obs_dim)
        with torch.no_grad():
            a_pi, _ = self.policy.sample(s_rep)
        a_pi = a_pi.reshape(B, n, self.act_dim)

        s_rep2 = s[:, None, :].repeat(1, n, 1).reshape(B * n, self.obs_dim)
        q1_rand = self.q1(s_rep2, a_rand.reshape(B * n, self.act_dim)).reshape(B, n)
        q2_rand = self.q2(s_rep2, a_rand.reshape(B * n, self.act_dim)).reshape(B, n)
        q1_pi = self.q1(s_rep2, a_pi.reshape(B * n, self.act_dim)).reshape(B, n)
        q2_pi = self.q2(s_rep2, a_pi.reshape(B * n, self.act_dim)).reshape(B, n)

        q1_cat = torch.cat([q1_rand, q1_pi], dim=1)
        q2_cat = torch.cat([q2_rand, q2_pi], dim=1)

        cql1 = (torch.logsumexp(q1_cat / self.cql_temp, dim=1) * self.cql_temp - q1.squeeze(-1)).mean()
        cql2 = (torch.logsumexp(q2_cat / self.cql_temp, dim=1) * self.cql_temp - q2.squeeze(-1)).mean()

        q1_loss = bellman1 + self.cql_alpha * cql1
        q2_loss = bellman2 + self.cql_alpha * cql2

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # -----------------------------
        # Policy update (SAC + expert BC anchor)
        # -----------------------------
        a_new, logp_new = self.policy.sample(s)
        q1_new = self.q1(s, a_new)
        q2_new = self.q2(s, a_new)
        q_new = torch.min(q1_new, q2_new)
        sac_pi_loss = (self.alpha.detach() * logp_new - q_new).mean()

        # 关键改动：BC 不再用 deterministic(mean) 做 MSE
        # 改成 expert-only 的 -log π(a_data|s)，让 stochastic policy 去覆盖数据动作
        bc_loss = torch.tensor(0.0, device=self.device)
        if bc_coef > 0:
            logp_data = self.policy.log_prob_action(s, a)  # [B,1]
            denom = expert_mask.sum().clamp(min=1.0)
            bc_loss = -(logp_data * expert_mask).sum() / denom

        pi_loss = sac_pi_loss + float(bc_coef) * bc_loss

        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        # -----------------------------
        # Alpha update
        # -----------------------------
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # -----------------------------
        # Target update
        # -----------------------------
        self._soft_update(self.q1, self.q1_t)
        self._soft_update(self.q2, self.q2_t)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
            "cql1": float(cql1.item()),
            "cql2": float(cql2.item()),
            "bellman1": float(bellman1.item()),
            "bellman2": float(bellman2.item()),
            "bc_loss": float(bc_loss.item()),
            "expert_frac_in_batch": float(expert_mask.mean().item()),
        }

    def act_env(self, obs_np: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # 你说“mean 训不出来”，所以默认 deterministic=False
        obs = torch.tensor(obs_np[None, :], device=self.device)
        with torch.no_grad():
            if deterministic:
                a = self.policy.deterministic(obs)
            else:
                a, _ = self.policy.sample(obs)
            a01 = to_01_action(a).clamp(0.0, 1.0)
        return a01.squeeze(0).cpu().numpy()


# =========================
# Eval (MultiplexEnv)
# =========================
def evaluate_cql(policies, eval_env: MultiplexEnv, num_episodes: int):
    total_reward, total_cost, total_utility, total_wait_penalty = 0.0, 0.0, 0.0, 0.0

    for _ in range(num_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            actions = {}
            for lid in obs:
                tq = obs[lid]["task_queue"]
                wl = obs[lid]["worker_loads"]
                wp = obs[lid]["worker_profile"]
                s = flatten_obs(tq, wl, wp)

                # 关键改动：eval 用 sample，而不是 deterministic(mean)
                a01_flat = policies[lid].act_env(s, deterministic=False)
                a01 = a01_flat.reshape(eval_env.action_space[lid].shape)
                actions[lid] = a01.astype(np.float32)

            obs, (_, reward_detail), done, _ = eval_env.step(actions)
            for _, layer_stats in reward_detail["layer_rewards"].items():
                total_reward += float(layer_stats.get("reward", 0))
                total_cost += float(layer_stats.get("cost", 0))
                total_utility += float(layer_stats.get("utility", 0))
                total_wait_penalty += float(layer_stats.get("waiting_penalty", 0))

    return {
        "reward": total_reward / num_episodes,
        "cost": total_cost / num_episodes,
        "utility": total_utility / num_episodes,
        "waiting_penalty": total_wait_penalty / num_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dire", type=str, default="standard")

    parser.add_argument("--hitac_root", type=str, default=None, help="默认 ../offline_data/hitac_muse/{dire}")
    parser.add_argument("--crescent_root", type=str, default=None, help="默认 ../offline_data/crescent/{dire}")

    parser.add_argument("--expert_tag", type=str, default="expert")
    parser.add_argument("--subopt_tag", type=str, default="suboptimal")

    parser.add_argument("--expert_frac", type=float, default=1)
    parser.add_argument("--expert_hitac_ratio", type=float, default=0)
    parser.add_argument("--subopt_hitac_ratio", type=float, default=0)

    parser.add_argument("--bc_coef", type=float, default=0.05)

    parser.add_argument("--num_updates", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--cql_alpha", type=float, default=1.0)
    parser.add_argument("--cql_temp", type=float, default=1.0)
    parser.add_argument("--cql_n_actions", type=int, default=10)

    # 新增：高维动作下更稳的 CQL 随机动作
    parser.add_argument("--cql_rand_std", type=float, default=0.1, help=">0: a + N(0,std)；=0: U[-1,1]")

    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--eval_episodes", type=int, default=5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save_dir", type=str, default="../logs/cql")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    env_config_path = f"../configs/{args.dire}/env_config.json"
    schedule_path = f"../configs/{args.dire}/train_schedule.json"
    eval_schedule_path = f"../configs/{args.dire}/eval_schedule.json"
    worker_config_path = f"../configs/{args.dire}/worker_config.json"

    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)

    env = MultiplexEnv(env_config_path, schedule_load_path=schedule_path, worker_config_load_path=worker_config_path)
    eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path)
    eval_env.worker_config = env.worker_config
    eval_env.chain = IndustrialChain(eval_env.worker_config)

    num_layers = int(env_cfg["num_layers"])
    print(f"num_layers={num_layers}")

    hitac_root = args.hitac_root or f"../offline_data/hitac_muse/{args.dire}"
    crescent_root = args.crescent_root or f"../offline_data/crescent/{args.dire}"

    hitac_expert_dir = os.path.join(hitac_root, args.expert_tag)
    hitac_subopt_dir = os.path.join(hitac_root, args.subopt_tag)
    crescent_expert_dir = os.path.join(crescent_root, args.expert_tag)
    crescent_subopt_dir = os.path.join(crescent_root, args.subopt_tag)

    def _assert_01(name: str, x: float) -> float:
        x = float(x)
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {x}")
        return x

    args.expert_frac = _assert_01("expert_frac", args.expert_frac)
    args.expert_hitac_ratio = _assert_01("expert_hitac_ratio", args.expert_hitac_ratio)
    args.subopt_hitac_ratio = _assert_01("subopt_hitac_ratio", args.subopt_hitac_ratio)

    eps = 1e-12
    use_expert = args.expert_frac > eps
    use_subopt = (1.0 - args.expert_frac) > eps

    use_h_e = use_expert and (args.expert_hitac_ratio > eps)
    use_c_e = use_expert and ((1.0 - args.expert_hitac_ratio) > eps)
    use_h_s = use_subopt and (args.subopt_hitac_ratio > eps)
    use_c_s = use_subopt and ((1.0 - args.subopt_hitac_ratio) > eps)

    os.makedirs(args.save_dir, exist_ok=True)
    tb_dir = os.path.join(args.save_dir, args.dire, time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"TensorBoard log_dir: {tb_dir}")

    replays = {}
    trainers = {}
    for lid in range(num_layers):
        replay = MixedReplay(
            hitac_expert_dir=hitac_expert_dir,
            hitac_subopt_dir=hitac_subopt_dir,
            crescent_expert_dir=crescent_expert_dir,
            crescent_subopt_dir=crescent_subopt_dir,
            layer_id=lid,
            use_hitac_expert=use_h_e,
            use_hitac_subopt=use_h_s,
            use_crescent_expert=use_c_e,
            use_crescent_subopt=use_c_s,
        )
        replays[lid] = replay

        trainer = CQLSAC(
            obs_dim=replay.obs_dim,
            act_dim=replay.act_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            cql_alpha=args.cql_alpha,
            cql_temp=args.cql_temp,
            cql_n_actions=args.cql_n_actions,
            cql_rand_std=args.cql_rand_std,
            device=device,
            target_entropy=-float(replay.act_dim),
            auto_alpha=True,
        )
        trainers[lid] = trainer

    for step in range(1, args.num_updates + 1):
        logs = {}
        for lid in range(num_layers):
            logs[lid] = trainers[lid].train_step(
                replays[lid],
                args.batch_size,
                expert_frac=args.expert_frac,
                expert_hitac_ratio=args.expert_hitac_ratio,
                subopt_hitac_ratio=args.subopt_hitac_ratio,
                bc_coef=args.bc_coef,
            )

        if step % 200 == 0:
            l0 = logs[0]
            writer.add_scalar("train/q1_loss", l0["q1_loss"], step)
            writer.add_scalar("train/q2_loss", l0["q2_loss"], step)
            writer.add_scalar("train/pi_loss", l0["pi_loss"], step)
            writer.add_scalar("train/alpha", l0["alpha"], step)
            writer.add_scalar("train/cql1", l0["cql1"], step)
            writer.add_scalar("train/bellman1", l0["bellman1"], step)
            writer.add_scalar("train/bc_loss", l0["bc_loss"], step)
            writer.add_scalar("train/expert_frac_in_batch", l0["expert_frac_in_batch"], step)

            print(
                f"[upd {step}] "
                f"q1={l0['q1_loss']:.4f} q2={l0['q2_loss']:.4f} "
                f"pi={l0['pi_loss']:.4f} alpha={l0['alpha']:.4f} "
                f"cql1={l0['cql1']:.4f} bell1={l0['bellman1']:.4f} "
                f"bc={l0['bc_loss']:.4f} expert_frac={l0['expert_frac_in_batch']:.2f}"
            )

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            stats = evaluate_cql(trainers, eval_env, args.eval_episodes)
            writer.add_scalar("eval/reward", stats["reward"], step)
            writer.add_scalar("eval/cost", stats["cost"], step)
            writer.add_scalar("eval/utility", stats["utility"], step)
            writer.add_scalar("eval/wait_penalty", stats["waiting_penalty"], step)
            print(
                f"[EVAL upd {step}] reward={stats['reward']:.4f} cost={stats['cost']:.4f} "
                f"utility={stats['utility']:.4f} wait_penalty={stats['waiting_penalty']:.4f}"
            )

    for lid in range(num_layers):
        save_path = os.path.join(args.save_dir, f"cql_mix_layer{lid}.pth")
        torch.save(
            {
                "policy": trainers[lid].policy.state_dict(),
                "q1": trainers[lid].q1.state_dict(),
                "q2": trainers[lid].q2.state_dict(),
                "q1_t": trainers[lid].q1_t.state_dict(),
                "q2_t": trainers[lid].q2_t.state_dict(),
                "log_alpha": trainers[lid].log_alpha.detach().cpu().numpy(),
                "obs_dim": trainers[lid].obs_dim,
                "act_dim": trainers[lid].act_dim,
            },
            save_path,
        )
        print(f"Saved: {save_path}")

    stats = evaluate_cql(trainers, eval_env, max(1, args.eval_episodes))
    print(
        f"[FINAL EVAL] reward={stats['reward']:.4f} cost={stats['cost']:.4f} "
        f"utility={stats['utility']:.4f} wait_penalty={stats['waiting_penalty']:.4f}"
    )

    writer.close()


if __name__ == "__main__":
    main()
