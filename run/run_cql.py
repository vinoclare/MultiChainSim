import os
import glob
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # task_q: [30, 7], worker_loads: [8, 4], worker_profile: [8, 6]
    return np.concatenate([
        task_q.reshape(-1).astype(np.float32),
        worker_loads.reshape(-1).astype(np.float32),
        worker_profile.reshape(-1).astype(np.float32),
    ], axis=0)


def to_tanh_action(a01: np.ndarray) -> np.ndarray:
    # [0,1] -> [-1,1]
    return (a01 * 2.0 - 1.0).astype(np.float32)


def to_01_action(a_tanh: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (a_tanh + 1.0) * 0.5


# =========================
# Offline Dataset (npz)
# =========================
class OfflineNpzReplay:
    """
    把 offline_dir 下的 traj_ep*_step*.npz 合并成一个离线 replay buffer（numpy arrays）。
    每个 layer 单独一份数据： (s, a, r, s', done)
    """
    def __init__(self, offline_dir: str, layer_id: int):
        files = sorted(glob.glob(os.path.join(offline_dir, "traj_ep*_step*.npz")))
        if not files:
            raise RuntimeError(f"No npz files found under: {offline_dir}")

        s_list, a_list, r_list, sp_list, d_list = [], [], [], [], []

        # quick sanity counters
        nz_task = 0
        nz_prof = 0

        for fp in files:
            data = np.load(fp, allow_pickle=True)
            T = int(data["T"][0])

            task = data[f"obs_task_queue_l{layer_id}"][:T]           # [T, 30, 7]
            loads = data[f"obs_worker_loads_l{layer_id}"][:T]        # [T, 8, 4]
            prof = data[f"obs_worker_profile_l{layer_id}"][:T]       # [T, 8, 6]

            task_n = data[f"next_task_queue_l{layer_id}"][:T]
            loads_n = data[f"next_worker_loads_l{layer_id}"][:T]
            prof_n = data[f"next_worker_profile_l{layer_id}"][:T]

            act01 = data[f"actions_l{layer_id}"][:T]                 # [T, 8, 30] in [0,1]
            rew = data[f"rewards_l{layer_id}"][:T].astype(np.float32)  # [T]
            done = data[f"dones_l{layer_id}"][:T].astype(np.float32)   # [T]

            nz_task += int(np.count_nonzero(task))
            nz_prof += int(np.count_nonzero(prof))

            for t in range(T):
                s = flatten_obs(task[t], loads[t], prof[t])
                sp = flatten_obs(task_n[t], loads_n[t], prof_n[t])

                a = to_tanh_action(act01[t].reshape(-1))  # [240] in [-1,1]

                s_list.append(s)
                sp_list.append(sp)
                a_list.append(a)
                r_list.append(rew[t])
                d_list.append(done[t])

        self.s = np.stack(s_list, axis=0)      # [N, obs_dim]
        self.a = np.stack(a_list, axis=0)      # [N, act_dim]
        self.r = np.array(r_list, dtype=np.float32)[:, None]  # [N,1]
        self.sp = np.stack(sp_list, axis=0)    # [N, obs_dim]
        self.d = np.array(d_list, dtype=np.float32)[:, None]  # [N,1]

        self.N = self.s.shape[0]
        self.obs_dim = self.s.shape[1]
        self.act_dim = self.a.shape[1]

        print(f"[Replay L{layer_id}] files={len(files)} samples={self.N} obs_dim={self.obs_dim} act_dim={self.act_dim}")
        print(f"[Replay L{layer_id}] nonzero task_queue total={nz_task}, nonzero worker_profile total={nz_prof}")
        if nz_task == 0 or nz_prof == 0:
            print(f"[WARN] L{layer_id}: task_queue/profile seems all-zero in loaded data. "
                  f"Offline RL will likely be crippled unless this is expected.")

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.N, size=batch_size)
        s = torch.tensor(self.s[idx], device=device)
        a = torch.tensor(self.a[idx], device=device)
        r = torch.tensor(self.r[idx], device=device)
        sp = torch.tensor(self.sp[idx], device=device)
        d = torch.tensor(self.d[idx], device=device)
        return s, a, r, sp, d


# =========================
# Networks
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TanhGaussianPolicy(nn.Module):
    """
    SAC policy: a ~ tanh(N(mean, std))
    输出动作在 [-1,1]。给环境用时再映射到 [0,1]。
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, log_std_min=-20, log_std_max=2):
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

        # log_prob correction for tanh
        # log N(pre_tanh | mean, std) - sum log(1 - tanh(x)^2)
        log_prob = (-0.5 * (((pre_tanh - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return a, log_prob

    def deterministic(self, obs):
        mean, _ = self(obs)
        return torch.tanh(mean)


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
            # SAC 常用设置：- |A|
            target_entropy = -float(act_dim)
        self.target_entropy = target_entropy

        self.log_alpha = torch.tensor(np.log(0.1), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # CQL params
        self.cql_alpha = cql_alpha
        self.cql_temp = cql_temp
        self.cql_n_actions = cql_n_actions

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def train_step(self, replay: OfflineNpzReplay, batch_size: int):
        s, a, r, sp, d = replay.sample(batch_size, self.device)

        # -----------------------------
        # Critic target
        # -----------------------------
        with torch.no_grad():
            ap, logp = self.policy.sample(sp)  # [-1,1]
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

        # random actions ~ U[-1,1]
        a_rand = torch.empty(B, n, self.act_dim, device=self.device).uniform_(-1.0, 1.0)

        # policy actions sampled at s
        s_rep = s[:, None, :].repeat(1, n, 1).reshape(B * n, self.obs_dim)
        a_pi, _ = self.policy.sample(s_rep)  # [B*n, act_dim]
        a_pi = a_pi.reshape(B, n, self.act_dim)

        # evaluate Q for sampled actions
        s_rand = s[:, None, :].repeat(1, n, 1).reshape(B * n, self.obs_dim)
        q1_rand = self.q1(s_rand, a_rand.reshape(B * n, self.act_dim)).reshape(B, n)
        q2_rand = self.q2(s_rand, a_rand.reshape(B * n, self.act_dim)).reshape(B, n)

        q1_pi = self.q1(s_rand, a_pi.reshape(B * n, self.act_dim)).reshape(B, n)
        q2_pi = self.q2(s_rand, a_pi.reshape(B * n, self.act_dim)).reshape(B, n)

        q1_cat = torch.cat([q1_rand, q1_pi], dim=1)  # [B, 2n]
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
        # Policy update
        # -----------------------------
        a_new, logp_new = self.policy.sample(s)
        q1_new = self.q1(s, a_new)
        q2_new = self.q2(s, a_new)
        q_new = torch.min(q1_new, q2_new)
        pi_loss = (self.alpha.detach() * logp_new - q_new).mean()

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
        }

    def act_env(self, obs_np: np.ndarray, deterministic: bool = True) -> np.ndarray:
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

                a01_flat = policies[lid].act_env(s, deterministic=True)  # [240] in [0,1]
                a01 = a01_flat.reshape(eval_env.action_space[lid].shape)  # [8,30]
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


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dire", type=str, default="standard")
    parser.add_argument("--offline_dir", type=str, default=None, help="默认 ../offline_data/hitac_muse/{dire}")
    parser.add_argument("--num_updates", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--cql_alpha", type=float, default=1.0)
    parser.add_argument("--cql_temp", type=float, default=1.0)
    parser.add_argument("--cql_n_actions", type=int, default=10)

    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--eval_episodes", type=int, default=5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save_dir", type=str, default="../models/offline_cql")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # ---- env init (follow your run_crescent style) ----
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

    # ---- offline dir ----
    offline_dir = args.offline_dir
    if offline_dir is None:
        offline_dir = f"../offline_data/hitac_muse/{args.dire}"
    print(f"offline_dir={offline_dir}")

    # ---- train per layer ----
    os.makedirs(args.save_dir, exist_ok=True)

    replays = {}
    trainers = {}
    for lid in range(num_layers):
        replay = OfflineNpzReplay(offline_dir, layer_id=lid)
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
            device=device,
            target_entropy=-float(replay.act_dim),
            auto_alpha=True,
        )
        trainers[lid] = trainer

    for step in range(1, args.num_updates + 1):
        logs = {}
        for lid in range(num_layers):
            logs[lid] = trainers[lid].train_step(replays[lid], args.batch_size)

        if step % 200 == 0:
            # 打印 layer0 的几个关键量，避免刷屏
            l0 = logs[0]
            print(f"[upd {step}] "
                  f"q1={l0['q1_loss']:.4f} q2={l0['q2_loss']:.4f} "
                  f"pi={l0['pi_loss']:.4f} alpha={l0['alpha']:.4f} "
                  f"cql1={l0['cql1']:.4f} bell1={l0['bellman1']:.4f}")

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            stats = evaluate_cql(trainers, eval_env, args.eval_episodes)
            print(f"[EVAL upd {step}] reward={stats['reward']:.4f} cost={stats['cost']:.4f} "
                  f"utility={stats['utility']:.4f} wait_penalty={stats['waiting_penalty']:.4f}")

    # ---- save ----
    for lid in range(num_layers):
        save_path = os.path.join(args.save_dir, f"cql_layer{lid}.pth")
        torch.save({
            "policy": trainers[lid].policy.state_dict(),
            "q1": trainers[lid].q1.state_dict(),
            "q2": trainers[lid].q2.state_dict(),
            "q1_t": trainers[lid].q1_t.state_dict(),
            "q2_t": trainers[lid].q2_t.state_dict(),
            "log_alpha": trainers[lid].log_alpha.detach().cpu().numpy(),
            "obs_dim": trainers[lid].obs_dim,
            "act_dim": trainers[lid].act_dim,
        }, save_path)
        print(f"Saved: {save_path}")

    stats = evaluate_cql(trainers, eval_env, max(1, args.eval_episodes))
    print(f"[FINAL EVAL] reward={stats['reward']:.4f} cost={stats['cost']:.4f} "
          f"utility={stats['utility']:.4f} wait_penalty={stats['waiting_penalty']:.4f}")


if __name__ == "__main__":
    main()
