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


# ======================== 基础网络 ========================

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
        # 假定动作被归一化到 [-1, 1]，和你 PPO/SAC 风格对齐
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
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def q1_only(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


# ======================== 数据集加载 ========================

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

    # 先为每一层准备列表
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

            # 简单 sanity check
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

            # 状态/下一个状态
            states = _flatten_state_arrays(task_obs, worker_loads, worker_profile)
            next_states = _flatten_state_arrays(n_task_obs, n_worker_loads, n_worker_profile)

            # 动作
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


# ======================== TD3+BC 训练 ========================

def soft_update(target_net, net, tau):
    for tp, p in zip(target_net.parameters(), net.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


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
            f"[build_state_from_env_obs] state_dim mismatch: got {state.shape[1]}, "
            f"expected {expected_dim}"
        )
    return state  # [1, state_dim]


def evaluate_policy(actors, env, num_layers, state_dims, action_shapes,
                    eval_episodes, device, seed=0):
    """
    多层联合评估：所有 layer 的 actor 一起决定动作。
    返回：平均 episode extrinsic return
    """
    rng = np.random.RandomState(seed)
    total_returns = []

    for ep in range(eval_episodes):
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

            # 和 _episode_ext_return 一致：按 layer_reward['reward'] 求和
            step_ret = 0.0
            for lid in range(num_layers):
                step_ret += float(reward_detail["layer_rewards"][lid]["reward"])
            ep_ret += step_ret

        total_returns.append(ep_ret)

    avg_ret = float(np.mean(total_returns))
    std_ret = float(np.std(total_returns))
    print(f"[eval] episodes={eval_episodes} avg_return={avg_ret:.3f} std={std_ret:.3f}")
    return avg_ret, std_ret


def train_td3_bc(args):
    # ---------- 基础配置与 seed ----------
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dire = args.dire
    dataset_tag = args.dataset

    # ---------- 读取环境配置 ----------
    env_config_path = f"../configs/{dire}/env_config.json"
    schedule_path = f"../configs/{dire}/eval_schedule.json"
    worker_config_path = f"../configs/{dire}/worker_config.json"

    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)

    num_layers = int(env_cfg["num_layers"])
    max_steps = int(env_cfg["max_steps"])

    # eval 环境（只用来评估，不参与训练）
    eval_env = MultiplexEnv(
        env_config_path,
        schedule_load_path=schedule_path,
        worker_config_load_path=worker_config_path,
    )
    eval_env.chain = IndustrialChain(eval_env.worker_config)

    # ---------- 载入离线数据 ----------
    offline_root = Path(args.offline_data_root) / dire / dataset_tag
    datasets, state_dims, action_shapes = load_offline_data_per_layer(
        offline_root, num_layers, device
    )

    # ---------- 为每一层创建 Actor / Critic ----------
    actors = {}
    actor_targets = {}
    critics = {}
    critic_targets = {}
    actor_optim = {}
    critic_optim = {}

    max_action = 1.0  # 你如果确认动作范围不是 [-1,1]，可以后面再改

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

    # ---------- TensorBoard 日志 ----------
    log_dir = Path("../logs/td3_bc") / dire / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[train] TensorBoard log dir: {log_dir}")

    # ---------- TD3+BC 训练循环 ----------
    total_updates = int(args.train_steps)
    best_return = -1e18
    best_actor_state_dicts = None
    best_critic_state_dicts = None

    print(f"[train] start TD3+BC offline training, total_updates={total_updates}")
    start_time = time.time()

    for it in range(0, total_updates + 1):
        # ----- 评估 & 写 TensorBoard -----
        if it % args.eval_interval == 0 or it == total_updates:
            avg_ret, std_ret = evaluate_policy(
                actors, eval_env, num_layers, state_dims, action_shapes,
                args.eval_episodes, device, seed=args.seed
            )
            writer.add_scalar("eval/avg_return", avg_ret, it)
            writer.add_scalar("eval/std_return", std_ret, it)

            if avg_ret > best_return:
                best_return = avg_ret
                best_actor_state_dicts = {lid: actors[lid].state_dict() for lid in range(num_layers)}
                best_critic_state_dicts = {lid: critics[lid].state_dict() for lid in range(num_layers)}
                print(f"[train] *** new best_return={best_return:.3f} at it={it}")

        # 每个迭代统计一次所有 layer 的 loss，用于写 TB
        critic_losses_step = []
        actor_losses_step = []
        lambdas_step = []

        # ----- Critic 更新 -----
        for lid in range(num_layers):
            dataset = datasets[lid]
            actor = actors[lid]
            actor_target = actor_targets[lid]
            critic = critics[lid]
            critic_target = critic_targets[lid]

            # 采样 batch
            states, actions, rewards, next_states, dones = dataset.sample(args.batch_size)

            with torch.no_grad():
                # target policy smoothing
                noise = (torch.randn_like(actions) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                )
                next_actions = (actor_target(next_states) + noise).clamp(
                    -max_action, max_action
                )
                target_q1, target_q2 = critic_target(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2)
                target = rewards + (1.0 - dones) * args.gamma * target_q

            current_q1, current_q2 = critic(states, actions)
            critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

            critic_optim[lid].zero_grad()
            critic_loss.backward()
            critic_optim[lid].step()

            critic_losses_step.append(critic_loss.item())

        # ----- Actor + target 更新（延迟） -----
        if it % args.policy_delay == 0:
            for lid in range(num_layers):
                dataset = datasets[lid]
                actor = actors[lid]
                actor_target = actor_targets[lid]
                critic = critics[lid]
                critic_target = critic_targets[lid]

                states, actions, _, _, _ = dataset.sample(args.batch_size)

                pi = actor(states)
                q_pi = critic.q1_only(states, pi)

                # TD3+BC: λ = α / E|Q|
                lambda_coef = args.alpha / (q_pi.abs().mean().detach() + 1e-6)

                bc_loss = F.mse_loss(pi, actions)
                actor_loss = (-lambda_coef * q_pi.mean() + bc_loss)

                actor_optim[lid].zero_grad()
                actor_loss.backward()
                actor_optim[lid].step()

                # soft update target 网络
                soft_update(critic_target, critic, args.tau)
                soft_update(actor_target, actor, args.tau)

                actor_losses_step.append(actor_loss.item())
                lambdas_step.append(lambda_coef.item())

        # ----- 日志打印 & 写入 TensorBoard -----
        if it % args.log_interval == 0:
            elapsed = time.time() - start_time
            mean_critic = float(np.mean(critic_losses_step)) if critic_losses_step else 0.0
            mean_actor = float(np.mean(actor_losses_step)) if actor_losses_step else 0.0
            mean_lambda = float(np.mean(lambdas_step)) if lambdas_step else 0.0

            print(
                f"[train] it={it}/{total_updates} "
                f"critic_loss_mean={mean_critic:.4f} "
                f"actor_loss_mean={mean_actor:.4f} "
                f"lambda_mean={mean_lambda:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

            # 写 TensorBoard
            writer.add_scalar("train/critic_loss_mean", mean_critic, it)
            if actor_losses_step:
                writer.add_scalar("train/actor_loss_mean", mean_actor, it)
                writer.add_scalar("train/lambda_mean", mean_lambda, it)

    # ---------- 保存最优模型 ----------
    if best_actor_state_dicts is None:
        best_actor_state_dicts = {lid: actors[lid].state_dict() for lid in range(num_layers)}
        best_critic_state_dicts = {lid: critics[lid].state_dict() for lid in range(num_layers)}

    ckpt_dir = Path(args.ckpt_root) / dire / dataset_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "td3_bc_best.pt"

    torch.save(
        {
            "actor_state_dicts": best_actor_state_dicts,
            "critic_state_dicts": best_critic_state_dicts,
            "state_dims": state_dims,
            "action_shapes": action_shapes,
            "num_layers": num_layers,
            "config": vars(args),
            "best_return": best_return,
        },
        str(ckpt_path),
    )
    print(f"[train] done. best_return={best_return:.3f}, saved to: {ckpt_path}")

    writer.close()


# ======================== main & CLI ========================

def main():
    parser = argparse.ArgumentParser()
    # 实验目录，比如 standard / easy / hard
    parser.add_argument("--dire", type=str, default="standard")

    # 数据集 tag，比如 expert / medium / random
    parser.add_argument("--dataset", type=str, default="expert")

    # 离线数据根目录，最终目录 = offline_data_root/dire/dataset
    parser.add_argument(
        "--offline_data_root",
        type=str,
        default="../offline_data/crescent",
        help="root dir of offline data, e.g. ../offline_data/crescent",
    )

    # checkpoint 保存目录
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="./checkpoints/td3_bc",
        help="where to save td3+bc checkpoints",
    )

    # TD3+BC 超参数
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="TD3+BC alpha, λ = α / E|Q|",
    )

    parser.add_argument("--train_steps", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--eval_interval", type=int, default=5_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=1_000)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    train_td3_bc(args)


if __name__ == "__main__":
    main()
