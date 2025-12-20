import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from envs import IndustrialChain
from envs.env import MultiplexEnv


# =========================
# Utils: load offline data (same format as run_crescent_expert_data.py)
# =========================
def _ensure_float32(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        x = np.stack([np.array(t) for t in x], axis=0)
    return np.asarray(x, dtype=np.float32)


def _list_npz_files(root: Path):
    return sorted(root.glob("*.npz"))


def _load_layer_transitions(npz_files, lid: int):
    pre = f"l{lid}_"

    s_task, s_load, s_prof, s_mask = [], [], [], []
    a, r, d = [], [], []
    ns_task, ns_load, ns_prof, ns_mask = [], [], [], []

    for f in npz_files:
        ep = np.load(str(f), allow_pickle=True)

        s_task.append(_ensure_float32(ep[pre + "task_obs"]))
        s_load.append(_ensure_float32(ep[pre + "worker_loads"]))
        s_prof.append(_ensure_float32(ep[pre + "worker_profile"]))
        s_mask.append(_ensure_float32(ep[pre + "valid_mask"]))

        a.append(_ensure_float32(ep[pre + "actions"]))
        r.append(_ensure_float32(ep[pre + "rewards"]).reshape(-1))
        d.append(_ensure_float32(ep[pre + "dones"]).reshape(-1))

        ns_task.append(_ensure_float32(ep[pre + "next_task_obs"]))
        ns_load.append(_ensure_float32(ep[pre + "next_worker_loads"]))
        ns_prof.append(_ensure_float32(ep[pre + "next_worker_profile"]))
        ns_mask.append(_ensure_float32(ep[pre + "next_valid_mask"]))

    data = {
        "task_obs": np.concatenate(s_task, axis=0),
        "worker_loads": np.concatenate(s_load, axis=0),
        "worker_profile": np.concatenate(s_prof, axis=0),
        "valid_mask": np.concatenate(s_mask, axis=0),

        "actions": np.concatenate(a, axis=0),
        "rewards": np.concatenate(r, axis=0),
        "dones": np.concatenate(d, axis=0),

        "next_task_obs": np.concatenate(ns_task, axis=0),
        "next_worker_loads": np.concatenate(ns_load, axis=0),
        "next_worker_profile": np.concatenate(ns_prof, axis=0),
        "next_valid_mask": np.concatenate(ns_mask, axis=0),
    }
    return data


def _sample_batch(layer_data: dict, batch_size: int, action_flat_dim: int, device: torch.device):
    n = layer_data["rewards"].shape[0]
    idx = np.random.randint(0, n, size=batch_size)

    def t(key, reshape_action=False):
        x = layer_data[key][idx]
        if reshape_action:
            x = x.reshape(batch_size, action_flat_dim)
        return torch.tensor(x, dtype=torch.float32, device=device)

    batch = {
        "task_obs": t("task_obs"),
        "worker_loads": t("worker_loads"),
        "worker_profile": t("worker_profile"),
        "valid_mask": t("valid_mask"),

        "actions": t("actions", reshape_action=True),
        "rewards": torch.tensor(layer_data["rewards"][idx], dtype=torch.float32, device=device).view(-1),
        "dones": torch.tensor(layer_data["dones"][idx], dtype=torch.float32, device=device).view(-1),

        "next_task_obs": t("next_task_obs"),
        "next_worker_loads": t("next_worker_loads"),
        "next_worker_profile": t("next_worker_profile"),
        "next_valid_mask": t("next_valid_mask"),
    }
    return batch


# =========================
# Networks: flatten + MLP (same style as IQL)
# =========================
def mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2, out_act=None):
    layers = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim=256, z_dim=256):
        super().__init__()
        self.net = mlp(in_dim, hidden_dim, z_dim, depth=2)

    def forward(self, task_obs, worker_loads, worker_profile, valid_mask):
        b = task_obs.shape[0]
        x = torch.cat([
            task_obs.reshape(b, -1),
            valid_mask.reshape(b, -1),
            worker_loads.reshape(b, -1),
            worker_profile.reshape(b, -1),
        ], dim=-1)
        return self.net(x)


class QNet(nn.Module):
    def __init__(self, encoder: StateEncoder, action_dim: int, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        z_dim = encoder.net[-1].out_features
        self.q = mlp(z_dim + action_dim, hidden_dim, 1, depth=2)

    def forward(self, task_obs, worker_loads, worker_profile, valid_mask, action_flat):
        z = self.encoder(task_obs, worker_loads, worker_profile, valid_mask)
        x = torch.cat([z, action_flat], dim=-1)
        return self.q(x).squeeze(-1)  # [B]


class TanhGaussianPolicy(nn.Module):
    """
    Tanh-squashed Gaussian -> rescale to [low, high]
    We need log_prob(a|s) for AWAC actor update.
    """
    def __init__(self, encoder: StateEncoder, action_dim: int, hidden_dim=256,
                 log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.encoder = encoder
        z_dim = encoder.net[-1].out_features
        self.mu = mlp(z_dim, hidden_dim, action_dim, depth=2)
        self.log_std = mlp(z_dim, hidden_dim, action_dim, depth=2)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def _dist(self, task_obs, worker_loads, worker_profile, valid_mask):
        z = self.encoder(task_obs, worker_loads, worker_profile, valid_mask)
        mu = self.mu(z)
        log_std = self.log_std(z).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, task_obs, worker_loads, worker_profile, valid_mask, action_low, action_high):
        mu, std = self._dist(task_obs, worker_loads, worker_profile, valid_mask)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)

        logp = dist.log_prob(u).sum(dim=-1)
        logp -= torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1)

        a_scaled = action_low + (a + 1.0) * 0.5 * (action_high - action_low)
        return a_scaled, logp

    def log_prob(self, task_obs, worker_loads, worker_profile, valid_mask, action_scaled, action_low, action_high):
        a = 2.0 * (action_scaled - action_low) / (action_high - action_low + 1e-8) - 1.0
        a = a.clamp(-0.999999, 0.999999)
        u = 0.5 * torch.log((1 + a) / (1 - a))

        mu, std = self._dist(task_obs, worker_loads, worker_profile, valid_mask)
        dist = torch.distributions.Normal(mu, std)

        logp = dist.log_prob(u).sum(dim=-1)
        logp -= torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1)
        return logp  # [B]


# =========================
# AWAC core
# =========================
class AWAC:
    """
    Minimal offline AWAC:
      Critic: TD with target networks, target action from current policy
      Actor : weighted BC on dataset actions
              w = exp( (Q(s,a_data) - Q(s,a_pi)) / lambda )
    """
    def __init__(self,
                 q1: QNet, q2: QNet, pi: TanhGaussianPolicy,
                 action_low: torch.Tensor, action_high: torch.Tensor,
                 gamma=0.99, awac_lambda=1.0, max_weight=100.0,
                 lr=3e-4, target_tau=0.005):
        self.q1 = q1
        self.q2 = q2
        self.pi = pi

        import copy
        self.q1_t = copy.deepcopy(self.q1)
        self.q2_t = copy.deepcopy(self.q2)
        for p in self.q1_t.parameters():
            p.requires_grad = False
        for p in self.q2_t.parameters():
            p.requires_grad = False

        self.action_low = action_low  # [1, action_dim]
        self.action_high = action_high

        self.gamma = float(gamma)
        self.awac_lambda = float(awac_lambda)
        self.max_weight = float(max_weight)
        self.target_tau = float(target_tau)

        self.q_optim = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr)

    @torch.no_grad()
    def _soft_update(self):
        tau = self.target_tau
        for p, tp in zip(self.q1.parameters(), self.q1_t.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_t.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)

    def update(self, batch: dict):
        s_task = batch["task_obs"]
        s_load = batch["worker_loads"]
        s_prof = batch["worker_profile"]
        s_mask = batch["valid_mask"]
        a_data = batch["actions"]
        r = batch["rewards"]
        d = batch["dones"]

        ns_task = batch["next_task_obs"]
        ns_load = batch["next_worker_loads"]
        ns_prof = batch["next_worker_profile"]
        ns_mask = batch["next_valid_mask"]

        # ----- Critic update -----
        with torch.no_grad():
            a_next, _ = self.pi.sample(ns_task, ns_load, ns_prof, ns_mask,
                                       action_low=self.action_low, action_high=self.action_high)
            q1_next = self.q1_t(ns_task, ns_load, ns_prof, ns_mask, a_next)
            q2_next = self.q2_t(ns_task, ns_load, ns_prof, ns_mask, a_next)
            q_next = torch.min(q1_next, q2_next)
            y = r + self.gamma * (1.0 - d) * q_next

        q1 = self.q1(s_task, s_load, s_prof, s_mask, a_data)
        q2 = self.q2(s_task, s_load, s_prof, s_mask, a_data)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # ----- Actor update (weighted BC) -----
        with torch.no_grad():
            # advantage baseline: Q(s, a_data) - Q(s, a_pi)
            a_pi, _ = self.pi.sample(s_task, s_load, s_prof, s_mask,
                                     action_low=self.action_low, action_high=self.action_high)
            qd = torch.min(
                self.q1_t(s_task, s_load, s_prof, s_mask, a_data),
                self.q2_t(s_task, s_load, s_prof, s_mask, a_data),
            )
            qp = torch.min(
                self.q1_t(s_task, s_load, s_prof, s_mask, a_pi),
                self.q2_t(s_task, s_load, s_prof, s_mask, a_pi),
            )
            adv = qd - qp
            weights = torch.exp(adv / max(self.awac_lambda, 1e-6)).clamp(max=self.max_weight)

        logp = self.pi.log_prob(
            s_task, s_load, s_prof, s_mask,
            action_scaled=a_data,
            action_low=self.action_low,
            action_high=self.action_high
        )
        pi_loss = -(weights * logp).mean()

        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        self._soft_update()

        info = {
            "q_loss": float(q_loss.detach().cpu().item()),
            "pi_loss": float(pi_loss.detach().cpu().item()),
            "q_mean": float(((q1 + q2) * 0.5).detach().cpu().mean().item()),
            "adv_mean": float(adv.detach().cpu().mean().item()),
            "w_mean": float(weights.detach().cpu().mean().item()),
        }
        return info

    @torch.no_grad()
    def act(self, task_obs, worker_loads, worker_profile, valid_mask, sample=True):
        # keep same "sample always" behavior as your env preference
        if task_obs.dim() == 2:
            task_obs = task_obs.unsqueeze(0)
            worker_loads = worker_loads.unsqueeze(0)
            worker_profile = worker_profile.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)

        a, _ = self.pi.sample(task_obs, worker_loads, worker_profile, valid_mask,
                              action_low=self.action_low, action_high=self.action_high)
        return a


# =========================
# Agent wrapper (same signature as your evaluate loop expects)
# =========================
class AWACAgent:
    def __init__(self, alg: AWAC, action_shape, device: torch.device):
        self.alg = alg
        self.action_shape = tuple(action_shape)
        self.device = device

    def sample(self, task_obs, worker_loads, worker_profile):
        valid_mask = task_obs[:, 3].astype(np.float32)

        t = torch.tensor(task_obs, dtype=torch.float32, device=self.device)
        l = torch.tensor(worker_loads, dtype=torch.float32, device=self.device)
        p = torch.tensor(worker_profile, dtype=torch.float32, device=self.device)
        m = torch.tensor(valid_mask, dtype=torch.float32, device=self.device)

        a = self.alg.act(t, l, p, m, sample=True)
        a_np = a.detach().cpu().numpy().reshape(self.action_shape)

        return 0.0, a_np, 0.0, None


# =========================
# Eval (copy from run_crescent.py style)
# =========================
def evaluate_policy(agent_dict, eval_env, num_episodes, writer, global_step):
    total_reward, total_cost, total_utility, total_wait_penalty = 0, 0, 0, 0
    for _ in range(num_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            actions = {}
            for lid in obs:
                task_obs = obs[lid]['task_queue']
                worker_loads = obs[lid]['worker_loads']
                profile = obs[lid]['worker_profile']
                _, act, _, _ = agent_dict[lid].sample(task_obs, worker_loads, profile)
                actions[lid] = act

            obs, (_, reward_detail), done, _ = eval_env.step(actions)

            for lid, layer_stats in reward_detail['layer_rewards'].items():
                total_reward += layer_stats.get("reward", 0)
                total_cost += layer_stats.get("cost", 0)
                total_utility += layer_stats.get("utility", 0)
                total_wait_penalty += layer_stats.get("waiting_penalty", 0)

    writer.add_scalar("eval/reward", total_reward / num_episodes, global_step)
    writer.add_scalar("eval/cost", total_cost / num_episodes, global_step)
    writer.add_scalar("eval/utility", total_utility / num_episodes, global_step)
    writer.add_scalar("eval/waiting_penalty", total_wait_penalty / num_episodes, global_step)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dire", type=str, default="standard")
    parser.add_argument("--offline_data_root", type=str, default="../offline_data/crescent")
    parser.add_argument("--subdir", type=str, default="expert")
    parser.add_argument("--max_files", type=int, default=0, help="0=all")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--updates", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--awac_lambda", type=float, default=1.0, help="smaller -> more aggressive, larger -> closer to BC")
    parser.add_argument("--max_weight", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target_tau", type=float, default=0.005)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=256)

    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--log_root", type=str, default="../logs/awac")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    dire = args.dire
    env_config_path = f"../configs/{dire}/env_config.json"
    eval_schedule_path = f"../configs/{dire}/eval_schedule.json"
    worker_config_path = f"../configs/{dire}/worker_config.json"

    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)

    # Build env (for space) and eval_env (for interaction)
    env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path,
                       worker_config_load_path=worker_config_path)
    eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path)
    eval_env.worker_config = env.worker_config
    eval_env.chain = IndustrialChain(eval_env.worker_config)

    num_layers = int(env_cfg["num_layers"])

    obs_space = env.observation_space[0]
    act_space = env.action_space[0]

    if not hasattr(act_space, "low") or not hasattr(act_space, "high"):
        raise TypeError(f"AWAC baseline expects continuous Box action space, got: {type(act_space)}")

    action_shape = tuple(act_space.shape)
    action_flat_dim = int(np.prod(action_shape))

    action_low = torch.tensor(np.array(act_space.low, dtype=np.float32).reshape(-1), device=device).view(1, -1)
    action_high = torch.tensor(np.array(act_space.high, dtype=np.float32).reshape(-1), device=device).view(1, -1)

    # Load offline data
    data_dir = Path(args.offline_data_root) / dire / args.subdir
    files = _list_npz_files(data_dir)
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found: {data_dir}")

    print(f"[AWAC] dire={dire} data_dir={data_dir} npz_files={len(files)} layers={num_layers}")
    print(f"[AWAC] act: shape={action_shape} flat_dim={action_flat_dim}")

    layer_data = {}
    for lid in range(num_layers):
        layer_data[lid] = _load_layer_transitions(files, lid)
        print(f"[AWAC] layer {lid} transitions = {layer_data[lid]['rewards'].shape[0]}")

    # Infer encoder input dim from OFFLINE DATA (fixes your previous shape mismatch)
    d0 = layer_data[0]
    task_flat = int(np.prod(d0["task_obs"].shape[1:]))
    mask_flat = int(np.prod(d0["valid_mask"].shape[1:]))
    load_flat = int(np.prod(d0["worker_loads"].shape[1:]))
    prof_flat = int(np.prod(d0["worker_profile"].shape[1:]))
    enc_in_dim = task_flat + mask_flat + load_flat + prof_flat
    print(f"[AWAC] inferred from data: task={task_flat} mask={mask_flat} load={load_flat} prof={prof_flat} enc_in_dim={enc_in_dim}")

    # Build per-layer AWAC
    algs = {}
    agents = {}
    for lid in range(num_layers):
        enc_q1 = StateEncoder(enc_in_dim, hidden_dim=args.hidden_dim, z_dim=args.z_dim).to(device)
        enc_q2 = StateEncoder(enc_in_dim, hidden_dim=args.hidden_dim, z_dim=args.z_dim).to(device)
        enc_pi = StateEncoder(enc_in_dim, hidden_dim=args.hidden_dim, z_dim=args.z_dim).to(device)

        q1 = QNet(enc_q1, action_dim=action_flat_dim, hidden_dim=args.hidden_dim).to(device)
        q2 = QNet(enc_q2, action_dim=action_flat_dim, hidden_dim=args.hidden_dim).to(device)
        pi = TanhGaussianPolicy(enc_pi, action_dim=action_flat_dim, hidden_dim=args.hidden_dim).to(device)

        alg = AWAC(
            q1=q1, q2=q2, pi=pi,
            action_low=action_low, action_high=action_high,
            gamma=args.gamma,
            awac_lambda=args.awac_lambda,
            max_weight=args.max_weight,
            lr=args.lr,
            target_tau=args.target_tau
        )
        algs[lid] = alg
        agents[lid] = AWACAgent(alg, action_shape=action_shape, device=device)

    # Logging
    log_dir = Path(args.log_root) / dire / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(str(log_dir))
    print(f"[AWAC] log_dir = {log_dir}")

    t0 = time.time()
    for step in range(1, args.updates + 1):
        for lid in range(num_layers):
            batch = _sample_batch(layer_data[lid], args.batch_size, action_flat_dim, device)
            info = algs[lid].update(batch)

            if step % 200 == 0:
                writer.add_scalar(f"train/l{lid}_q_loss", info["q_loss"], step)
                writer.add_scalar(f"train/l{lid}_pi_loss", info["pi_loss"], step)
                writer.add_scalar(f"train/l{lid}_q_mean", info["q_mean"], step)
                writer.add_scalar(f"train/l{lid}_adv_mean", info["adv_mean"], step)
                writer.add_scalar(f"train/l{lid}_w_mean", info["w_mean"], step)

        if step % args.eval_interval == 0:
            elapsed = time.time() - t0
            print(f"[AWAC] step={step}/{args.updates} elapsed={elapsed:.1f}s eval...")
            evaluate_policy(agents, eval_env, args.eval_episodes, writer, step)

    writer.close()
    print(f"[AWAC] done. logs at: {log_dir}")


if __name__ == "__main__":
    main()
