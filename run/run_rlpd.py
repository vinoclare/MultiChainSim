# run/run_rlpd.py
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from envs import IndustrialChain
from envs.env import MultiplexEnv


# ======================== 基础网络（复用 TD3BC 样式） ========================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.max_action = float(max_action)

    def forward(self, state):
        # 假定动作被归一化到 [-max_action, max_action]
        return self.max_action * torch.tanh(self.net(state))


class Critic(nn.Module):
    """TD3 双 Q 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


# ======================== Offline Dataset（复用 TD3BC 样式） ========================

class OfflineDataset:
    """
    每一层一个 OfflineDataset：
      states: [N, state_dim]
      actions: [N, action_dim]
      rewards: [N, 1]
      next_states: [N, state_dim]
      dones: [N, 1]
    """
    def __init__(self, states, actions, rewards, next_states, dones, device):
        self.states = torch.tensor(states, dtype=torch.float32, device=device)
        self.actions = torch.tensor(actions, dtype=torch.float32, device=device)
        self.rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        self.next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        self.dones = torch.tensor(dones, dtype=torch.float32, device=device)
        self.size = self.states.shape[0]

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.states.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


def _flatten_state_arrays(task_obs, worker_loads, worker_profile):
    """
    输入：
      task_obs:      [T, N_slots, feat_dim]
      worker_loads:  [T, n_worker, k]
      worker_profile:[T, profile_dim]
    输出：
      states:        [T, state_dim]，简单拼接 (task_flat, load_flat, profile)
    """
    T = task_obs.shape[0]
    task_flat = task_obs.reshape(T, -1).astype(np.float32)
    load_flat = worker_loads.reshape(T, -1).astype(np.float32)
    prof_flat = worker_profile.reshape(T, -1).astype(np.float32)
    return np.concatenate([task_flat, load_flat, prof_flat], axis=1)


def load_offline_data_per_layer(root_dir: Path, num_layers: int, device):
    """
    root_dir: ../offline_data/crescent/{dire}/{dataset} 目录
    返回：
      datasets: {lid: OfflineDataset}
      state_dims: {lid: int}
      action_shapes: {lid: tuple}  # 原始动作形状 (例如 [n_worker, action_dim])
    """
    assert root_dir.is_dir(), f"offline data dir not found: {root_dir}"

    files = sorted(root_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {root_dir}")

    print(f"[load_offline] dir={root_dir}, num_files={len(files)}")

    all_states = {lid: [] for lid in range(num_layers)}
    all_actions = {lid: [] for lid in range(num_layers)}
    all_rewards = {lid: [] for lid in range(num_layers)}
    all_next_states = {lid: [] for lid in range(num_layers)}
    all_dones = {lid: [] for lid in range(num_layers)}

    action_shapes = {}

    for f in files:
        data = np.load(str(f), allow_pickle=True)
        T = int(data["T"])

        for lid in range(num_layers):
            prefix = f"l{lid}_"

            task_obs = data[prefix + "task_obs"]
            worker_loads = data[prefix + "worker_loads"]
            worker_profile = data[prefix + "worker_profile"]
            actions = data[prefix + "actions"]
            rewards = data[prefix + "rewards"]
            dones = data[prefix + "dones"]
            n_task_obs = data[prefix + "next_task_obs"]
            n_worker_loads = data[prefix + "next_worker_loads"]
            n_worker_profile = data[prefix + "next_worker_profile"]

            # sanity check：避免 object dtype
            if task_obs.dtype == object or worker_loads.dtype == object:
                raise ValueError(
                    f"[load_offline] variable-shaped arrays detected in {f}, "
                    f"layer {lid}. _stack_or_array() probably fell back to object dtype."
                )

            assert task_obs.shape[0] == T
            assert worker_loads.shape[0] == T
            assert worker_profile.shape[0] == T
            assert actions.shape[0] == T
            assert rewards.shape[0] == T
            assert dones.shape[0] == T
            assert n_task_obs.shape[0] == T
            assert n_worker_loads.shape[0] == T
            assert n_worker_profile.shape[0] == T

            states = _flatten_state_arrays(task_obs, worker_loads, worker_profile)
            next_states = _flatten_state_arrays(n_task_obs, n_worker_loads, n_worker_profile)

            # 动作 flatten
            if lid not in action_shapes:
                action_shapes[lid] = tuple(actions.shape[1:])
                print(f"[load_offline] layer {lid}: action_shape={action_shapes[lid]}")

            actions_flat = actions.reshape(T, -1).astype(np.float32)
            rewards = rewards.reshape(T, 1).astype(np.float32)
            dones = dones.reshape(T, 1).astype(np.float32)

            all_states[lid].append(states)
            all_actions[lid].append(actions_flat)
            all_rewards[lid].append(rewards)
            all_next_states[lid].append(next_states)
            all_dones[lid].append(dones)

    datasets = {}
    state_dims = {}

    for lid in range(num_layers):
        states = np.concatenate(all_states[lid], axis=0)
        actions = np.concatenate(all_actions[lid], axis=0)
        rewards = np.concatenate(all_rewards[lid], axis=0)
        next_states = np.concatenate(all_next_states[lid], axis=0)
        dones = np.concatenate(all_dones[lid], axis=0)

        datasets[lid] = OfflineDataset(states, actions, rewards, next_states, dones, device)
        state_dims[lid] = states.shape[1]

        print(
            f"[load_offline] layer {lid}: N={datasets[lid].size}, "
            f"state_dim={state_dims[lid]}, action_dim={actions.shape[1]}"
        )

    return datasets, state_dims, action_shapes


def build_state_from_env_obs(layer_obs, expected_dim: int):
    """
    env obs -> 1D state 向量，和离线数据 flatten 方式保持一致
    """
    task_obs = np.asarray(layer_obs["task_queue"], dtype=np.float32).reshape(1, -1)
    worker_loads = np.asarray(layer_obs["worker_loads"], dtype=np.float32).reshape(1, -1)
    worker_profile = np.asarray(layer_obs["worker_profile"], dtype=np.float32).reshape(1, -1)
    state = np.concatenate([task_obs, worker_loads, worker_profile], axis=1)
    if state.shape[1] != expected_dim:
        raise ValueError(
            f"[build_state_from_env_obs] state_dim mismatch: got {state.shape[1]}, expected {expected_dim}"
        )
    return state  # [1, D]


def soft_update(target: nn.Module, src: nn.Module, tau: float):
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), src.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)


# ======================== Online ReplayBuffer（RLPD 需要） ========================

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int, device):
        self.max_size = int(max_size)
        self.device = device

        self.states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, ns, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr, 0] = r
        self.next_states[self.ptr] = ns
        self.dones[self.ptr, 0] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        if self.size <= 0:
            raise RuntimeError("ReplayBuffer is empty.")
        idx = np.random.randint(0, self.size, size=batch_size)

        s = torch.tensor(self.states[idx], dtype=torch.float32, device=self.device)
        a = torch.tensor(self.actions[idx], dtype=torch.float32, device=self.device)
        r = torch.tensor(self.rewards[idx], dtype=torch.float32, device=self.device)
        ns = torch.tensor(self.next_states[idx], dtype=torch.float32, device=self.device)
        d = torch.tensor(self.dones[idx], dtype=torch.float32, device=self.device)
        return s, a, r, ns, d


# ======================== 评估（复用 TD3BC 的风格） ========================

def evaluate_policy(actors, env, num_layers, state_dims, action_shapes,
                    eval_episodes, device, seed=0):
    """
    多层联合评估：所有 layer 的 actor 一起决定动作。
    返回：平均 episode extrinsic return（按 layer_rewards[lid]['reward'] 求和）
    """
    rng = np.random.RandomState(seed)
    total_returns = []

    for ep in range(eval_episodes):
        _ = rng.randint(0, 10**9)  # 保持接口一致，暂不做额外随机扰动
        obs = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            actions = {}
            for lid in range(num_layers):
                layer_obs = obs[lid]
                state_np = build_state_from_env_obs(layer_obs, state_dims[lid])  # [1, D]
                state_t = torch.tensor(state_np, dtype=torch.float32, device=device)
                with torch.no_grad():
                    act_t = actors[lid](state_t)  # [1, A_flat]
                act_np = act_t.cpu().numpy().reshape(action_shapes[lid])
                actions[lid] = act_np

            obs, (_, reward_detail), done, _ = env.step(actions)

            step_ret = 0.0
            for lid in range(num_layers):
                step_ret += float(reward_detail["layer_rewards"][lid]["reward"])
            ep_ret += step_ret

        total_returns.append(ep_ret)

    avg_ret = float(np.mean(total_returns))
    std_ret = float(np.std(total_returns))
    print(f"[eval] episodes={eval_episodes} avg_return={avg_ret:.3f} std={std_ret:.3f}")
    return avg_ret, std_ret


# ======================== RLPD-style TD3 训练 ========================

def train_rlpd(args):
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dire = args.dire
    dataset_tag = args.dataset

    # ---------- 读取环境配置 ----------
    env_config_path = f"../configs/{dire}/env_config.json"
    train_schedule_path = f"../configs/{dire}/train_schedule.json"
    eval_schedule_path = f"../configs/{dire}/eval_schedule.json"
    worker_config_path = f"../configs/{dire}/worker_config.json"

    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)

    num_layers = int(env_cfg["num_layers"])
    max_steps = int(env_cfg["max_steps"])

    # ---------- train env + eval env ----------
    env = MultiplexEnv(
        env_config_path,
        schedule_load_path=train_schedule_path,
        worker_config_load_path=worker_config_path,
    )
    eval_env = MultiplexEnv(
        env_config_path,
        schedule_load_path=eval_schedule_path,
        worker_config_load_path=worker_config_path,
    )
    # 对齐 worker_config / chain（参考 run_varibad 的写法，避免 eval 配置不一致）
    eval_env.worker_config = env.worker_config
    eval_env.chain = IndustrialChain(eval_env.worker_config)

    # ---------- 加载离线数据 ----------
    offline_root = Path(f"../offline_data/crescent/{dire}/{dataset_tag}")
    datasets, state_dims, action_shapes = load_offline_data_per_layer(
        offline_root, num_layers=num_layers, device=device
    )

    # ---------- 初始化网络（每层一个 TD3 agent） ----------
    max_action = float(args.max_action)

    actors, actor_targets = {}, {}
    critics, critic_targets = {}, {}
    actor_optim, critic_optim = {}, {}

    replay_buffers = {}

    for lid in range(num_layers):
        state_dim = state_dims[lid]
        action_dim = int(np.prod(action_shapes[lid]))

        actor = Actor(state_dim, action_dim, hidden_dim=args.hidden_dim, max_action=max_action).to(device)
        actor_target = Actor(state_dim, action_dim, hidden_dim=args.hidden_dim, max_action=max_action).to(device)
        actor_target.load_state_dict(actor.state_dict())

        critic = Critic(state_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
        critic_target = Critic(state_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
        critic_target.load_state_dict(critic.state_dict())

        actors[lid] = actor
        actor_targets[lid] = actor_target
        critics[lid] = critic
        critic_targets[lid] = critic_target

        actor_optim[lid] = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic_optim[lid] = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

        replay_buffers[lid] = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=args.replay_size,
            device=device
        )

    # ---------- TensorBoard ----------
    log_dir = Path("../logs/rlpd") / dire / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[train] TensorBoard log dir: {log_dir}")

    # ---------- batch 里 offline/online 比例（RLPD 核心） ----------
    online_bs = int(round(args.batch_size * args.online_ratio))
    online_bs = max(0, min(args.batch_size, online_bs))
    offline_bs = args.batch_size - online_bs
    if offline_bs <= 0:
        raise ValueError("offline batch size must be > 0. Set --online_ratio < 1.0")

    print(f"[train] batch_size={args.batch_size}, offline_bs={offline_bs}, online_bs={online_bs}")
    print(f"[train] start_steps={args.start_steps}, env_steps={args.env_steps}, utd_ratio={args.utd_ratio}")

    # ---------- 训练主循环：env step + (utd_ratio 次)更新 ----------
    obs = env.reset()
    done = False

    ep_ret = 0.0
    ep_len = 0
    ep_idx = 0

    best_return = -1e18
    best_actor_state_dicts = None
    best_critic_state_dicts = None

    global_update = 0
    last_log_time = time.time()

    # 用于 interval 日志统计
    critic_losses_buf = []
    actor_losses_buf = []
    lambdas_buf = []

    for env_step in range(1, int(args.env_steps) + 1):
        # --- 选动作（每层独立） ---
        actions_env = {}
        cached = {}  # lid -> (s_flat, a_flat)

        for lid in range(num_layers):
            state_np = build_state_from_env_obs(obs[lid], state_dims[lid])  # [1, D]
            s_flat = state_np.reshape(-1)

            action_dim = int(np.prod(action_shapes[lid]))

            if env_step <= args.start_steps:
                a_flat = np.random.uniform(-max_action, max_action, size=(action_dim,)).astype(np.float32)
            else:
                state_t = torch.tensor(state_np, dtype=torch.float32, device=device)
                with torch.no_grad():
                    a_t = actors[lid](state_t).cpu().numpy().reshape(-1).astype(np.float32)
                # TD3 exploration noise
                a_t = a_t + np.random.normal(0, args.expl_noise, size=a_t.shape).astype(np.float32)
                a_flat = np.clip(a_t, -max_action, max_action)

            actions_env[lid] = a_flat.reshape(action_shapes[lid])
            cached[lid] = (s_flat, a_flat)

        # --- 环境推进 ---
        next_obs, (_, reward_detail), done, _ = env.step(actions_env)

        # --- 存入 online replay（每层各存一次） ---
        step_ret = 0.0
        d = float(done)

        for lid in range(num_layers):
            r = float(reward_detail["layer_rewards"][lid]["reward"])
            step_ret += r

            ns_np = build_state_from_env_obs(next_obs[lid], state_dims[lid])
            ns_flat = ns_np.reshape(-1)

            s_flat, a_flat = cached[lid]
            replay_buffers[lid].add(s_flat, a_flat, r, ns_flat, d)

        ep_ret += step_ret
        ep_len += 1
        obs = next_obs

        # --- episode done：记录并 reset ---
        if done:
            writer.add_scalar("train/episode_return", ep_ret, env_step)
            writer.add_scalar("train/episode_len", ep_len, env_step)
            ep_idx += 1

            obs = env.reset()
            done = False
            ep_ret = 0.0
            ep_len = 0

        # --- 更新：RLPD 核心（混 offline + online） ---
        # 需要 online buffer 里有足够样本，否则只能继续采样
        can_update = True
        if online_bs > 0:
            for lid in range(num_layers):
                if replay_buffers[lid].size < max(online_bs, args.min_online_size):
                    can_update = False
                    break

        if env_step > args.start_steps and can_update:
            for _ in range(int(args.utd_ratio)):
                global_update += 1

                # ----- Critic 更新（每层） -----
                for lid in range(num_layers):
                    dataset = datasets[lid]
                    rb = replay_buffers[lid]

                    actor_target = actor_targets[lid]
                    critic = critics[lid]
                    critic_target = critic_targets[lid]

                    # offline
                    s_off, a_off, r_off, ns_off, d_off = dataset.sample(offline_bs)

                    # online
                    if online_bs > 0:
                        s_on, a_on, r_on, ns_on, d_on = rb.sample(online_bs)
                        states = torch.cat([s_off, s_on], dim=0)
                        actions = torch.cat([a_off, a_on], dim=0)
                        rewards = torch.cat([r_off, r_on], dim=0)
                        next_states = torch.cat([ns_off, ns_on], dim=0)
                        dones = torch.cat([d_off, d_on], dim=0)
                    else:
                        states, actions, rewards, next_states, dones = s_off, a_off, r_off, ns_off, d_off

                    with torch.no_grad():
                        noise = (torch.randn_like(actions) * args.policy_noise).clamp(
                            -args.noise_clip, args.noise_clip
                        )
                        next_actions = (actor_target(next_states) + noise).clamp(-max_action, max_action)
                        tq1, tq2 = critic_target(next_states, next_actions)
                        tq = torch.min(tq1, tq2)
                        target = rewards + (1.0 - dones) * args.gamma * tq

                    cq1, cq2 = critic(states, actions)
                    critic_loss = F.mse_loss(cq1, target) + F.mse_loss(cq2, target)

                    critic_optim[lid].zero_grad()
                    critic_loss.backward()
                    critic_optim[lid].step()

                    critic_losses_buf.append(float(critic_loss.item()))

                # ----- Actor + target 更新（延迟） -----
                if global_update % args.policy_delay == 0:
                    for lid in range(num_layers):
                        dataset = datasets[lid]
                        rb = replay_buffers[lid]

                        actor = actors[lid]
                        actor_target = actor_targets[lid]
                        critic = critics[lid]
                        critic_target = critic_targets[lid]

                        # 重新采样一份混合 states，用于 actor 的 RL 项
                        s_off, a_off, _, _, _ = dataset.sample(offline_bs)
                        if online_bs > 0:
                            s_on, _, _, _, _ = rb.sample(online_bs)
                            s_mix = torch.cat([s_off, s_on], dim=0)
                        else:
                            s_mix = s_off

                        pi_mix = actor(s_mix)
                        q_pi = critic.q1_only(s_mix, pi_mix)

                        if args.alpha > 0:
                            # TD3+BC 风格：lambda = alpha / E|Q|
                            lambda_coef = args.alpha / (q_pi.abs().mean().detach() + 1e-6)

                            # BC 只对 offline 部分做（因为 offline 动作是专家/数据集动作）
                            pi_off = actor(s_off)
                            bc_loss = F.mse_loss(pi_off, a_off)

                            actor_loss = (-lambda_coef * q_pi.mean() + bc_loss)
                            lambdas_buf.append(float(lambda_coef.item()))
                        else:
                            actor_loss = -q_pi.mean()
                            lambdas_buf.append(0.0)

                        actor_optim[lid].zero_grad()
                        actor_loss.backward()
                        actor_optim[lid].step()

                        # soft update targets
                        soft_update(critic_target, critic, args.tau)
                        soft_update(actor_target, actor, args.tau)

                        actor_losses_buf.append(float(actor_loss.item()))

        # --- eval ---
        if env_step % args.eval_interval == 0:
            avg_ret, std_ret = evaluate_policy(
                actors, eval_env, num_layers, state_dims, action_shapes,
                args.eval_episodes, device, seed=args.seed
            )
            writer.add_scalar("eval/avg_return", avg_ret, env_step)
            writer.add_scalar("eval/std_return", std_ret, env_step)

            if avg_ret > best_return:
                best_return = avg_ret
                best_actor_state_dicts = {lid: actors[lid].state_dict() for lid in range(num_layers)}
                best_critic_state_dicts = {lid: critics[lid].state_dict() for lid in range(num_layers)}
                print(f"[train] *** new best_return={best_return:.3f} at env_step={env_step}")

        # --- log / print ---
        if env_step % args.log_interval == 0:
            now = time.time()
            elapsed = now - last_log_time
            last_log_time = now

            mean_critic = float(np.mean(critic_losses_buf)) if critic_losses_buf else 0.0
            mean_actor = float(np.mean(actor_losses_buf)) if actor_losses_buf else 0.0
            mean_lambda = float(np.mean(lambdas_buf)) if lambdas_buf else 0.0

            critic_losses_buf.clear()
            actor_losses_buf.clear()
            lambdas_buf.clear()

            # online buffer size（取平均展示）
            rb_sizes = [replay_buffers[lid].size for lid in range(num_layers)]
            rb_mean = float(np.mean(rb_sizes))

            print(
                f"[train] env_step={env_step}/{int(args.env_steps)} "
                f"updates={global_update} "
                f"critic_loss_mean={mean_critic:.4f} "
                f"actor_loss_mean={mean_actor:.4f} "
                f"lambda_mean={mean_lambda:.4f} "
                f"rb_mean_size={rb_mean:.1f} "
                f"elapsed={elapsed:.1f}s"
            )

            writer.add_scalar("train/critic_loss_mean", mean_critic, env_step)
            writer.add_scalar("train/actor_loss_mean", mean_actor, env_step)
            writer.add_scalar("train/lambda_mean", mean_lambda, env_step)
            writer.add_scalar("train/replay_mean_size", rb_mean, env_step)

    # ---------- 保存 best checkpoint ----------
    ckpt_dir = Path(args.ckpt_dir) / dire / dataset_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"rlpd_seed{args.seed}.pt"

    save_obj = {
        "best_return": best_return,
        "args": vars(args),
        "num_layers": num_layers,
        "state_dims": state_dims,
        "action_shapes": action_shapes,
        "actor_state_dicts": best_actor_state_dicts,
        "critic_state_dicts": best_critic_state_dicts,
    }
    torch.save(save_obj, str(ckpt_path))
    print(f"[train] done. best_return={best_return:.3f}, saved to: {ckpt_path}")

    writer.close()


# ======================== main & CLI ========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dire", type=str, default="standard")
    parser.add_argument("--dataset", type=str, default="expert")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    # online 交互步数（RLPD：env_steps + 每步 utd_ratio 次更新）
    parser.add_argument("--env_steps", type=int, default=200_000)
    parser.add_argument("--utd_ratio", type=int, default=5)
    parser.add_argument("--start_steps", type=int, default=2_000, help="前多少 env step 用随机动作填充 online buffer")
    parser.add_argument("--min_online_size", type=int, default=1_000, help="online buffer 至少多少样本后才开始更新")

    # replay buffer
    parser.add_argument("--replay_size", type=int, default=200_000)

    # RLPD 核心：batch 混合比例（online 占比）
    parser.add_argument("--online_ratio", type=float, default=0.5)

    # 网络与优化
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)

    # TD3 超参
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--expl_noise", type=float, default=0.1, help="在线采样时给 actor 动作加的探索噪声 std")

    # 动作范围（通常保持和离线数据/环境一致）
    parser.add_argument("--max_action", type=float, default=1.0)

    # 可选：TD3+BC（alpha=0 表示纯 TD3/RLPD；alpha>0 表示对 offline 部分加 BC 正则）
    parser.add_argument("--alpha", type=float, default=0.0)

    # batch / eval / log
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_interval", type=int, default=5_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=1_000)

    # checkpoint dir
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoints/rlpd")

    args = parser.parse_args()
    train_rlpd(args)


if __name__ == "__main__":
    main()
