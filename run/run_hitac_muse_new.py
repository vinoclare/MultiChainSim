import json
import time
import os
import pickle
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import argparse

from envs import IndustrialChain
from envs.env import MultiplexEnv
from agents.hitac_muse_agent import HiTACMuSEAgent
from utils.utils import RunningMeanStd

# ======== Load Configurations ========
parser = argparse.ArgumentParser()
parser.add_argument('--dire', type=str, default='standard',
                    help='name of sub-folder under ../configs/ 作为本次实验的配置目录')
args, _ = parser.parse_known_args()
dire = args.dire
env_config_path = f'../configs/{dire}/env_config.json'
algo_config_path = f'../configs/hitac_muse_config.json'

with open(env_config_path, 'r') as f:
    env_config = json.load(f)
with open(algo_config_path, 'r') as f:
    algo_config = json.load(f)

# === 路径配置 ===
train_schedule_path = f"../configs/{dire}/train_schedule.json"
eval_schedule_path = f"../configs/{dire}/eval_schedule.json"
worker_config_path = f"../configs/{dire}/worker_config.json"

env = MultiplexEnv(env_config_path, schedule_load_path=train_schedule_path, worker_config_load_path=worker_config_path)
eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path)

eval_env.worker_config = env.worker_config
eval_env.chain = IndustrialChain(eval_env.worker_config)

# === 基本参数 ===
device = algo_config["training"]["device"] if torch.cuda.is_available() else "cpu"
num_layers = env_config["num_layers"]
num_workers = env_config["workers_per_layer"]
num_pad_tasks = env_config["num_pad_tasks"]
task_types = env_config["task_types"]
n_task_types = len(task_types)
profile_dim = 2 * n_task_types
global_context_dim = 1
policies_info_dim = algo_config["hitac"]["policies_info_dim"]
alpha = env_config["alpha"]
beta = env_config["beta"]

# ======= 超参提取 =======
num_episodes = algo_config["training"]["num_episodes"]
eval_interval = algo_config["training"]["eval_interval"]
log_interval = algo_config["training"]["log_interval"]
eval_episodes = algo_config["training"]["eval_episodes"]
distill_interval = algo_config["scheduler"]["distill_interval"]
switch_interval = algo_config["scheduler"]["switch_interval"]
hitac_update_interval = algo_config["scheduler"]["hitac_update_interval"]
reset_schedule_interval = algo_config["training"]["reset_schedule_interval"]
neg_interval = algo_config["scheduler"]["neg_interval"]

steps_per_episode = env_config["max_steps"]
K = algo_config["muse"]["K"]
neg_policy = algo_config["distill"]["neg_policy"]
num_pos_subpolicies = K - 2 if algo_config["distill"]["neg_policy"] else K
warmup_ep = algo_config["distill"]["warmup_ep"]
min_reward_ratio = algo_config["distill"]["min_reward_ratio"]
update_epochs = algo_config["muse"]["update_epochs"]
gamma = algo_config["muse"]["gamma"]
lam = algo_config["muse"]["lam"]
batch_size = algo_config["muse"]["batch_size"]
return_norm = algo_config["muse"]["return_normalization"]
local_kpi_dim = algo_config["hitac"]["local_kpi_dim"] + num_pos_subpolicies
global_kpi_dim = algo_config["hitac"]["global_kpi_dim"]
policies_info_dim = algo_config["hitac"]["policies_info_dim"]
traj_save_threshold = (num_episodes - 10 * eval_interval) * steps_per_episode


# === 每层 obs 结构描述（供 MuSE init）===
obs_shapes = []
for lid in range(num_layers):
    obs_shapes.append({
        "task": 4 + n_task_types,
        "worker_load": 1 + n_task_types,
        "worker_profile": 2 * n_task_types,
        "n_worker": num_workers[lid],
        "num_pad_tasks": num_pad_tasks,
        "global_context_dim": global_context_dim
    })

act_spaces = [
    (obs_shapes[lid]["n_worker"], obs_shapes[lid]["num_pad_tasks"])
    for lid in range(num_layers)
]

# ===== TensorBoard 日志器 =====
log_dir = f'../logs/hitac_muse/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)

# ===== 创建 HiTACMuSEAgent =====
agent = HiTACMuSEAgent(
    muse_cfg=algo_config["muse"],
    hitac_cfg=algo_config["hitac"],
    distill_cfg=algo_config["distill"],
    obs_spaces=obs_shapes,
    act_spaces=act_spaces,
    num_layers=num_layers,
    global_context_dim=global_context_dim,
    device=device,
    writer=writer,
    total_training_steps=steps_per_episode * num_episodes
)

# === KPI缓冲区初始化 ===
kpi_window_size = algo_config["hitac"].get("kpi_window_size", 5)

pol_hist = [
    [
        deque(maxlen=kpi_window_size)   # 每个 deque 里存最近若干 episode 的指标 dict
        for _ in range(num_pos_subpolicies)
    ] for _ in range(num_layers)
]

local_kpi_history = {
    lid: deque(maxlen=kpi_window_size)
    for lid in range(num_layers)
}
global_kpi_history = deque(maxlen=kpi_window_size)

# ===== 每层 buffer，用于 PPO 存储经验 =====
buffers = [
    {"task_obs": [], "worker_loads": [], "worker_profile": [],
     "global_context": [], "valid_mask": [], "actions": [],
     "value_u": [], "value_c": [], "logprobs": [], "rewards": [],
     "reward_u": [], "reward_c": [], "dones": [], "pid": []}
    for _ in range(num_layers)
]

# ===== 初始化 RunningMeanStd 用于归一化 return=====
return_u_rms = {lid: RunningMeanStd() for lid in range(num_layers)}
return_c_rms = {lid: RunningMeanStd() for lid in range(num_layers)}

return_u_rms_main = {lid: RunningMeanStd() for lid in range(num_layers)}
return_c_rms_main = {lid: RunningMeanStd() for lid in range(num_layers)}

# === 初始化 EMA baseline ===
ema_return = 0.0
ema_alpha = 0.1

select_cnt = 0
skip_hitac_train = False


def compute_gae_single_head(rewards, dones, values, next_value, gamma, lam):
    T = len(rewards)
    advs = [0.0] * T
    values_ext = list(values) + [next_value]
    last_gae = 0.0

    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t + 1] * next_nonterminal - values_ext[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advs[t] = last_gae

    returns = [advs[t] + values_ext[t] for t in range(T)]
    return returns, advs


def process_obs(obs, lid, device="cuda"):
    """
    从环境返回的 obs 中提取某一层的 observation 字段，构造 agent.sample() 所需格式。

    返回：
        task_obs: Tensor [B, num_pad_tasks, task_dim]
        worker_loads: Tensor [B, num_worker, load_dim]
        worker_profile: Tensor [B, num_worker, profile_dim]
        global_context: Tensor [B, global_dim]
    """
    layer_obs = obs[lid]  # 取出该层 observation（类型为 dict）

    task_obs = torch.tensor(layer_obs["task_queue"], dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, task_dim]
    worker_loads = torch.tensor(layer_obs["worker_loads"], dtype=torch.float32, device=device).unsqueeze(0)
    worker_profile = torch.tensor(layer_obs["worker_profile"], dtype=torch.float32, device=device).unsqueeze(0)
    global_context = torch.tensor(obs["global_context"], dtype=torch.float32, device=device).unsqueeze(0)

    return task_obs, worker_loads, worker_profile, global_context


def evaluate_policy(agent, eval_env, eval_episodes, writer, global_step, device):
    num_layers = eval_env.num_layers
    reward_sums = {lid: [] for lid in range(num_layers)}
    cost_sums = {lid: [] for lid in range(num_layers)}
    util_sums = {lid: [] for lid in range(num_layers)}
    assign_bonus_sums = {lid: [] for lid in range(num_layers)}
    wait_penalty_sums = {lid: [] for lid in range(num_layers)}

    # === 保存当前eval所有episode的轨迹 ===
    eval_trajectories = []
    is_save_traj = (global_step >= traj_save_threshold)

    for episode in range(eval_episodes):
        obs = eval_env.reset(with_new_schedule=False)
        done = False

        # 每个 episode 累计总指标
        episode_reward = {lid: 0.0 for lid in range(num_layers)}
        episode_cost = {lid: 0.0 for lid in range(num_layers)}
        episode_util = {lid: 0.0 for lid in range(num_layers)}
        episode_assign_bonus = {lid: 0.0 for lid in range(num_layers)}
        episode_wait_penalty = {lid: 0.0 for lid in range(num_layers)}

        # 每个 episode 的 per-layer trajectory buffer
        episode_trajectories = {lid: {"task_obs": [], "worker_loads": [], "worker_profile": [], "global_context": [], "actions": []}
                                for lid in range(num_layers)}

        while not done:
            actions = {}
            for lid in range(num_layers):
                task_obs, worker_loads, worker_profile, global_context = process_obs(obs, lid, device)
                valid_mask = task_obs[:, :, 3]
                obs_dict = {
                    "task_obs": task_obs,
                    "worker_loads": worker_loads,
                    "worker_profiles": worker_profile,
                    "global_context": global_context,
                    "valid_mask": valid_mask,
                }
                _, _, action, _ = agent.main_policy_predict(lid, obs_dict)
                actions[lid] = action.squeeze(0).detach().cpu().numpy()

                # === 保存obs和action到该layer的buffer ===
                if is_save_traj:
                    episode_trajectories[lid]["task_obs"].append(task_obs.squeeze(0).cpu().numpy())
                    episode_trajectories[lid]["worker_loads"].append(worker_loads.squeeze(0).cpu().numpy())
                    episode_trajectories[lid]["worker_profile"].append(worker_profile.squeeze(0).cpu().numpy())
                    episode_trajectories[lid]["global_context"].append(global_context.squeeze(0).cpu().numpy())
                    episode_trajectories[lid]["actions"].append(action.squeeze(0).cpu().numpy())

            obs, (total_reward, reward_detail), done, _ = eval_env.step(actions)

            for lid in range(num_layers):
                r = reward_detail["layer_rewards"][lid]
                episode_reward[lid] += r["reward"]
                episode_cost[lid] += r["cost"]
                episode_util[lid] += r["utility"]
                episode_assign_bonus[lid] += r["assign_bonus"]
                episode_wait_penalty[lid] += r["wait_penalty"]

        # === 一个 episode 完后，再 append 一次 episode-level 总和 ===
        for lid in range(num_layers):
            reward_sums[lid].append(episode_reward[lid])
            cost_sums[lid].append(episode_cost[lid])
            util_sums[lid].append(episode_util[lid])
            assign_bonus_sums[lid].append(episode_assign_bonus[lid])
            wait_penalty_sums[lid].append(episode_wait_penalty[lid])

        if is_save_traj:
            eval_trajectories.append(episode_trajectories)

    # === 在最后10次eval阶段才保存 ===
    if is_save_traj:
        os.makedirs("../logs/eval_trajectories", exist_ok=True)
        with open(f"../logs/eval_trajectories/eval_step_{global_step}.pkl", "wb") as f:
            pickle.dump(eval_trajectories, f)
        print(f"[Eval Save] Trajectories saved at global_step {global_step}")

    # === tensorboard logging (保持原有逻辑) ===
    total_reward_all = sum([np.mean(reward_sums[lid]) for lid in range(num_layers)])
    total_cost_all = sum([np.mean(cost_sums[lid]) for lid in range(num_layers)])
    total_util_all = sum([np.mean(util_sums[lid]) for lid in range(num_layers)])

    for lid in range(num_layers):
        writer.add_scalar(f"eval/layer_{lid}_avg_reward", np.mean(reward_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_cost", np.mean(cost_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_utility", np.mean(util_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_assign_bonus", np.mean(assign_bonus_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_wait_penalty", np.mean(wait_penalty_sums[lid]), global_step)

    writer.add_scalar("global/eval_avg_reward", total_reward_all, global_step)
    writer.add_scalar("global/eval_avg_cost", total_cost_all, global_step)
    writer.add_scalar("global/eval_avg_utility", total_util_all, global_step)

    print(f"[Eval Summary] Total reward={total_reward_all:.2f}, cost={total_cost_all:.2f}, utility={total_util_all:.2f}")


# ======= 训练主循环 =======
global_step = 0
for episode in range(num_episodes):
    # warmup stage：轮询训练每个子策略
    if episode < (warmup_ep * K):
        current_pid_tensor = torch.tensor([episode % K for _ in range(num_layers)], device=device)

    if episode % switch_interval == 0 and episode >= (warmup_ep * K):
        select_cnt += 1

        if select_cnt % neg_interval == 0 and neg_policy:
            skip_hitac_train = True
            current_pid_tensor = torch.tensor([K - 1 for _ in range(num_layers)])
            distill_pid = torch.tensor([K - 1 for _ in range(num_layers)])
        elif select_cnt % neg_interval == 25 and neg_policy:
            skip_hitac_train = True
            current_pid_tensor = torch.tensor([K - 2 for _ in range(num_layers)])
            distill_pid = torch.tensor([K - 2 for _ in range(num_layers)])
        else:
            skip_hitac_train = False

            # === 构造 local KPI ===
            local_kpis_tensor = torch.zeros((1, num_layers, local_kpi_dim), dtype=torch.float32, device=device)
            for lid in range(num_layers):
                if len(local_kpi_history[lid]) > 0:
                    local_kpis_tensor[0, lid] = torch.stack(list(local_kpi_history[lid]), dim=0).mean(dim=0)

            # === 构造 global KPI ===
            if len(global_kpi_history) > 0:
                global_kpi_tensor = torch.stack(list(global_kpi_history), dim=0).mean(dim=0).unsqueeze(0)
            else:
                global_kpi_tensor = torch.zeros((1, 4), dtype=torch.float32, device=device)

            # === 构造 policies_info tensor (去掉 var_r/var_rt) ===
            policies_info_tensor = torch.zeros((1, num_layers, num_pos_subpolicies, 6), dtype=torch.float32,
                                               device=device)
            for lid in range(num_layers):
                for k in range(num_pos_subpolicies):
                    hist = pol_hist[lid][k]
                    if len(hist) == 0:
                        mu_r = mu_c = mu_u = mu_rt = ent = std = 0.0
                    else:
                        mu_r = np.mean([h["r"] for h in hist])
                        mu_c = np.mean([h["c"] for h in hist])
                        mu_u = np.mean([h["u"] for h in hist])
                        mu_rt = np.mean([h["rt"] for h in hist])
                        ent = np.mean([h["ent"] for h in hist])
                        std = np.mean([h["std"] for h in hist])

                    policies_info_tensor[0, lid, k] = torch.tensor(
                        [mu_r, mu_c, mu_u, mu_rt, ent, std], device=device)

            # === 调用 HiTAC 子策略选择 ===
            current_pid_tensor = agent.select_subpolicies(
                local_kpis_tensor, global_kpi_tensor,
                policies_info_tensor, global_step)

            distill_pid = agent.select_subpolicies_distill(
                local_kpis_tensor, global_kpi_tensor,
                policies_info_tensor, global_step)

        for lid in range(num_layers):
            writer.add_scalar(f"layer_{lid}/selected_pid", current_pid_tensor[lid], episode)

    # # 固定子策略0
    # current_pid_tensor = torch.tensor([0 for _ in range(num_layers)])

    # # 轮询子策略
    # current_pid_tensor = torch.tensor([episode % K for _ in range(num_layers)], device=device)

    if (episode + 1) % reset_schedule_interval == 0:
        obs = env.reset(with_new_schedule=True)
        return_u_rms = {lid: RunningMeanStd() for lid in range(num_layers)}
        return_c_rms = {lid: RunningMeanStd() for lid in range(num_layers)}

        return_u_rms_main = {lid: RunningMeanStd() for lid in range(num_layers)}
        return_c_rms_main = {lid: RunningMeanStd() for lid in range(num_layers)}
    else:
        obs = env.reset(with_new_schedule=False)

    episode_stats = {
        lid: {
            "reward": 0.0,
            "cost": 0.0,
            "utility": 0.0,
            "assign_bonus": 0.0,
            "wait_penalty": 0.0,
            "fused_reward": 0.0,
            "ent_sum": 0.0,
            "std_sum": 0.0
        } for lid in range(num_layers)
    }

    for step in range(steps_per_episode):
        # === 构造每层 obs_dict ===
        obs_dicts = {}
        for lid in range(num_layers):
            layer_obs = obs[lid]
            valid_mask = layer_obs["task_queue"][:, 3].astype(np.float32)
            obs_dicts[lid] = {
                "task_obs": torch.tensor(layer_obs["task_queue"], dtype=torch.float32, device=device).unsqueeze(0),
                "worker_loads": torch.tensor(layer_obs["worker_loads"], dtype=torch.float32, device=device).unsqueeze(0),
                "worker_profile": torch.tensor(layer_obs["worker_profile"], dtype=torch.float32,
                                               device=device).unsqueeze(0),
                "global_context": torch.tensor(obs["global_context"], dtype=torch.float32, device=device).unsqueeze(0),
                "valid_mask": torch.tensor(valid_mask, dtype=torch.float32, device=device).unsqueeze(0)
            }

        # === Agent 选择子策略 & 采样动作 & 环境交互 ===
        sample_out = agent.sample(obs_dicts, current_pid_tensor)
        actions = {lid: sample_out[lid]["actions"].squeeze(0).cpu().numpy() for lid in range(num_layers)}
        obs, (total_reward, reward_detail), done, _ = env.step(actions)

        for lid in range(num_layers):
            stats = episode_stats[lid]
            rew = reward_detail["layer_rewards"][lid]["reward"]
            cost = reward_detail["layer_rewards"][lid]["cost"]
            util = reward_detail["layer_rewards"][lid]["utility"]
            ab = reward_detail["layer_rewards"][lid]["assign_bonus"]
            wp = reward_detail["layer_rewards"][lid]["wait_penalty"]

            alpha_k = agent.muses[lid].alphas[current_pid_tensor[lid]].item()
            beta_k = agent.muses[lid].betas[current_pid_tensor[lid]].item()

            total_u = beta_k * util + ab
            total_c = -alpha_k * cost - wp
            fused_reward = total_u + total_c

            # === 累积 episode 指标 ===
            stats["reward"] += rew
            stats["cost"] += cost
            stats["utility"] += util
            stats["assign_bonus"] += ab
            stats["wait_penalty"] += wp
            stats["fused_reward"] += fused_reward
            stats["ent_sum"] += sample_out[lid]["ent"].item()
            stats["std_sum"] += sample_out[lid]["act_std"].item()

            # === 存入 PPO buffer（保留）===
            buffers[lid]["task_obs"].append(obs_dicts[lid]["task_obs"].squeeze(0))
            buffers[lid]["worker_loads"].append(obs_dicts[lid]["worker_loads"].squeeze(0))
            buffers[lid]["worker_profile"].append(obs_dicts[lid]["worker_profile"].squeeze(0))
            buffers[lid]["global_context"].append(obs_dicts[lid]["global_context"].squeeze(0))
            buffers[lid]["valid_mask"].append(obs_dicts[lid]["valid_mask"].squeeze(0))
            buffers[lid]["actions"].append(sample_out[lid]["actions"].squeeze(0))
            buffers[lid]["value_u"].append(sample_out[lid]["v_u"].item())
            buffers[lid]["value_c"].append(sample_out[lid]["v_c"].item())
            buffers[lid]["logprobs"].append(sample_out[lid]["logp"].item())
            buffers[lid]["reward_u"].append(total_u)
            buffers[lid]["reward_c"].append(total_c)
            buffers[lid]["rewards"].append(fused_reward)
            buffers[lid]["dones"].append(done)
            buffers[lid]["pid"].append(sample_out[lid]["pid"].item())

        if done:
            break

    global_step = (episode + 1) * steps_per_episode
    if episode % eval_interval == 0:
        evaluate_policy(agent, eval_env, eval_episodes, writer, global_step, device)

    ppo_stats = {}

    # Train
    for lid in range(num_layers):
        buf = buffers[lid]

        # === Step 1: GAE 计算 ===
        dones = buf["dones"]
        reward_u = buf["reward_u"]
        reward_c = buf["reward_c"]
        value_u = buf["value_u"]
        value_c = buf["value_c"]

        returns_u, _ = compute_gae_single_head(reward_u, dones, value_u, 0.0, gamma, lam)
        returns_c, _ = compute_gae_single_head(reward_c, dones, value_c, 0.0, gamma, lam)

        # === Step 2: return 归一化 ===
        returns_u = np.array(returns_u)
        returns_c = np.array(returns_c)

        if not skip_hitac_train:
            return_u_rms[lid].update(returns_u)
            return_c_rms[lid].update(returns_c)
            returns_u = return_u_rms[lid].normalize(returns_u)
            returns_c = return_c_rms[lid].normalize(returns_c)

        advs_u = returns_u - np.array(value_u)
        advs_c = returns_c - np.array(value_c)
        advs_u = (advs_u - advs_u.mean()) / (advs_u.std() + 1e-8)
        advs_c = (advs_c - advs_c.mean()) / (advs_c.std() + 1e-8)
        advantages = [au + ac for au, ac in zip(advs_u, advs_c)]

        # === Step 3: 构建 batch ===
        batch_data = list(zip(
            buf["pid"], buf["task_obs"], buf["worker_loads"], buf["worker_profile"],
            buf["global_context"], buf["valid_mask"], buf["actions"],
            returns_u, returns_c, buf["logprobs"], advantages
        ))

        # === Step 4: PPO 多 epoch 更新 ===
        for _ in range(update_epochs):
            # np.random.shuffle(batch_data)
            for i in range(0, len(batch_data), batch_size):
                mini = batch_data[i:i + batch_size]
                if len(mini) == 0:
                    continue

                pids, task_obs, worker_loads, profiles, gctx, mask, \
                    acts, ret_u, ret_c, logp_old, advs = zip(*mini)

                mini_batch = {
                    "pid": torch.tensor(pids, dtype=torch.long, device=device),
                    "task_obs": torch.stack([t.to(device) for t in task_obs]),
                    "worker_loads": torch.stack([w.to(device) for w in worker_loads]),
                    "worker_profile": torch.stack([p.to(device) for p in profiles]),
                    "global_context": torch.stack([g.to(device) for g in gctx]),
                    "valid_mask": torch.stack([m.to(device) for m in mask]),
                    "actions": torch.stack([a.to(device) for a in acts]),
                    "returns": (
                        torch.tensor(ret_u, dtype=torch.float32, device=device),
                        torch.tensor(ret_c, dtype=torch.float32, device=device),
                    ),
                    "logp_old": torch.tensor(logp_old, dtype=torch.float32, device=device),
                    "advantages": torch.tensor(advs, dtype=torch.float32, device=device),
                }

                stats = agent.muse_learn(lid, global_step, mini_batch)
                ppo_stats.update(stats)

    # === Step 6: TensorBoard 记录 PPO loss ===
    if episode % log_interval == 0:
        for k, v in ppo_stats.items():
            writer.add_scalar(k, v, episode)

    if current_pid_tensor[0].item() < num_pos_subpolicies:
        for lid in range(num_layers):
            # ---------- update policies_info ----------
            stats = episode_stats[lid]
            cur_pid = current_pid_tensor[lid].item()

            mean_ent = stats["ent_sum"] / steps_per_episode
            mean_std = stats["std_sum"] / steps_per_episode

            pol_hist[lid][cur_pid].append({
                "r": stats["reward"],
                "c": stats["cost"],
                "u": stats["utility"],
                "rt": stats["fused_reward"],
                "ent": mean_ent,
                "std": mean_std
            })

            # === 构造当前 episode 原始 KPI ===
            stats = episode_stats[lid]

            pid_val = current_pid_tensor[lid].item()
            pid_onehot = F.one_hot(torch.tensor(pid_val, device=device), num_classes=num_pos_subpolicies).float()

            local_kpi = torch.cat([
                torch.tensor([
                    stats["reward"],
                    stats["cost"],
                    stats["utility"],
                    stats["assign_bonus"],
                    stats["wait_penalty"]
                ], dtype=torch.float32, device=device),
                pid_onehot
            ])

            local_kpi_history[lid].append(local_kpi)

        # === 构造 global_kpi ===
        reward_sum = sum(stats["reward"] for stats in episode_stats.values())
        cost_sum = sum(stats["cost"] for stats in episode_stats.values())
        util_sum = sum(stats["utility"] for stats in episode_stats.values())
        assign_bonus_sum = sum(stats["assign_bonus"] for stats in episode_stats.values())
        wait_penalty_sum = sum(stats["wait_penalty"] for stats in episode_stats.values())

        # 构造 global KPI tensor（仅保留主要全局指标）
        raw_global_kpi = torch.tensor([
            reward_sum,
            cost_sum,
            util_sum
        ], dtype=torch.float32, device=device)

        global_kpi_history.append(raw_global_kpi)

    if episode % log_interval == 0:
        for lid in range(num_layers):
            stats = episode_stats[lid]
            writer.add_scalar(f"layer_{lid}/reward", stats["reward"], global_step)
            writer.add_scalar(f"layer_{lid}/cost", stats["cost"], global_step)
            writer.add_scalar(f"layer_{lid}/utility", stats["utility"], global_step)
            writer.add_scalar(f"layer_{lid}/assign_bonus", stats["assign_bonus"], global_step)
            writer.add_scalar(f"layer_{lid}/wait_penalty", stats["wait_penalty"], global_step)

        writer.add_scalar("global/episode_total_cost", cost_sum, global_step)
        writer.add_scalar("global/episode_total_reward", reward_sum, global_step)
        writer.add_scalar("global/episode_total_utility", util_sum, global_step)
        writer.add_scalar("global/episode_assign_bonus", assign_bonus_sum, global_step)
        writer.add_scalar("global/episode_wait_penalty", wait_penalty_sum, global_step)

    # === 更新 EMA baseline ===
    if episode == 0:
        ema_return = reward_sum
    else:
        ema_return = ema_alpha * reward_sum + (1 - ema_alpha) * ema_return

    # === 存入蒸馏 buffer ===
    if reward_sum > (ema_return * min_reward_ratio):
        for lid in range(num_layers):
            agent.distill_collect(lid,
                                  {"task_obs": torch.stack(buffers[lid]["task_obs"], dim=0),
                                   "worker_loads": torch.stack(buffers[lid]["worker_loads"], dim=0),
                                   "worker_profiles": torch.stack(buffers[lid]["worker_profile"], dim=0),
                                   "global_context": torch.stack(buffers[lid]["global_context"], dim=0),
                                   "valid_mask": torch.stack(buffers[lid]["valid_mask"], dim=0)},
                                  torch.stack(buffers[lid]["actions"], dim=0),
                                  buffers[lid]["pid"])

            buffers[lid] = {k: [] for k in buffers[lid]}

    # === HiTAC PPO 更新 ===
    if episode % hitac_update_interval == 0 and episode >= (warmup_ep * K) and not skip_hitac_train:
        # ---- 构造 kpi 张量 ----
        local_kpis_tensor = torch.zeros((1, num_layers, local_kpi_dim), dtype=torch.float32, device=device)
        for lid in range(num_layers):
            local_kpis_tensor[0, lid] = local_kpi_history[lid][-1]

        global_kpi_tensor = global_kpi_history[-1].unsqueeze(0)

        # ---- 构造 policies_info 张量 ----
        policies_info_tensor = torch.zeros((1, num_layers, num_pos_subpolicies, policies_info_dim), dtype=torch.float32, device=device)
        for lid in range(num_layers):
            for k in range(num_pos_subpolicies):
                hist = pol_hist[lid][k]
                if len(hist) == 0:
                    mu_r = mu_c = mu_u = mu_rt = ent = std = 0.0
                else:
                    last = hist[-1]
                    mu_r = last["r"]
                    mu_c = last["c"]
                    mu_u = last["u"]
                    mu_rt = last["rt"]
                    ent = last["ent"]
                    std = last["std"]

                policies_info_tensor[0, lid, k] = torch.tensor(
                    [mu_r, mu_c, mu_u, mu_rt, ent, std], device=device)

        # === 计算 episode-level return ===
        layer_returns = [episode_stats[lid]["reward"] for lid in range(num_layers)]
        returns_L = torch.tensor([layer_returns], dtype=torch.float32, device=device)

        hitac_stats = agent.hitac_update(local_kpis_tensor,
                                         global_kpi_tensor,
                                         policies_info_tensor,
                                         returns_L,  # shape (L,)
                                         global_step)

        # === TensorBoard 记录 ===
        if episode % log_interval == 0:
            for k, v in hitac_stats.items():
                writer.add_scalar(k, v, episode)

    # BC Update
    if episode % distill_interval == 0 and episode > (warmup_ep * K):
        for lid in range(num_layers):
            loss = agent.distill_update(lid, distill_pid[lid].item())
            if not skip_hitac_train:  # 非负策略蒸馏时，记录蒸馏 loss
                writer.add_scalar(f"distill/layer_{lid}_loss", loss, episode)

    # Online Correction
    obs_rollout = env.reset(with_new_schedule=False)
    done = False
    distill_buffer = {}
    for lid in range(num_layers):
        distill_buffer[lid] = {
            "task_obs": [],
            "worker_loads": [],
            "worker_profile": [],
            "global_context": [],
            "valid_mask": [],
            "actions": [],
            "rewards": [],
            "rewards_u": [],
            "rewards_c": [],
            "values_u": [],
            "values_c": [],
            "log_probs": [],
            "dones": [],
        }

    while not done:
        actions = {}
        obs_dicts = {}

        for lid in range(num_layers):
            layer_obs = obs_rollout[lid]
            valid_mask = layer_obs["task_queue"][:, -1].astype(np.float32)

            obs_dict = {
                "task_obs": torch.tensor(layer_obs["task_queue"], dtype=torch.float32, device=device).unsqueeze(0),
                "worker_loads": torch.tensor(layer_obs["worker_loads"], dtype=torch.float32,
                                             device=device).unsqueeze(0),
                "worker_profiles": torch.tensor(layer_obs["worker_profile"], dtype=torch.float32,
                                                device=device).unsqueeze(0),
                "global_context": torch.tensor(obs_rollout["global_context"], dtype=torch.float32,
                                               device=device).unsqueeze(0),
                "valid_mask": torch.tensor(valid_mask, dtype=torch.float32, device=device).unsqueeze(0)
            }

            obs_dicts[lid] = obs_dict
            v_u, v_c, action_t, logp_t = agent.distillers[lid].predict(obs_dict)
            actions[lid] = action_t.squeeze(0).cpu().numpy()

            distill_buffer[lid]["task_obs"].append(obs_dict["task_obs"].squeeze(0))
            distill_buffer[lid]["worker_loads"].append(obs_dict["worker_loads"].squeeze(0))
            distill_buffer[lid]["worker_profile"].append(obs_dict["worker_profiles"].squeeze(0))
            distill_buffer[lid]["global_context"].append(obs_dict["global_context"].squeeze(0))
            distill_buffer[lid]["valid_mask"].append(obs_dict["valid_mask"].squeeze(0))
            distill_buffer[lid]["actions"].append(action_t.squeeze(0))
            distill_buffer[lid]["values_u"].append(v_u.item())
            distill_buffer[lid]["values_c"].append(v_c.item())
            distill_buffer[lid]["log_probs"].append(logp_t.item())

        # 环境一步
        obs_next, (total_reward, reward_detail), done, _ = env.step(actions)

        # 存 rewards、done
        for lid in range(num_layers):
            rew = reward_detail["layer_rewards"][lid]["reward"]
            cost = reward_detail["layer_rewards"][lid]["cost"]
            util = reward_detail["layer_rewards"][lid]["utility"]
            ab = reward_detail["layer_rewards"][lid]["assign_bonus"]
            wp = reward_detail["layer_rewards"][lid]["wait_penalty"]

            total_u = beta * util + ab
            total_c = alpha * cost + wp

            distill_buffer[lid]["rewards"].append(rew)
            distill_buffer[lid]["rewards_u"].append(total_u)
            distill_buffer[lid]["rewards_c"].append(total_c)
            distill_buffer[lid]["dones"].append(done)

        obs_rollout = obs_next

    # RL update
    for lid in range(num_layers):
        buffer = distill_buffer[lid]
        reward_u_list = buffer["rewards_u"]
        reward_c_list = buffer["rewards_c"]
        value_u_list = buffer["values_u"]
        value_c_list = buffer["values_c"]
        dones_list = buffer["dones"]

        neg_reward_c_list = [-r for r in reward_c_list]
        neg_value_c_list = [-v for v in value_c_list]

        returns_u, _ = compute_gae_single_head(reward_u_list, dones_list, value_u_list, 0.0, gamma, lam)
        returns_c, _ = compute_gae_single_head(neg_reward_c_list, dones_list, neg_value_c_list, 0.0, gamma, lam)

        returns_u = np.array(returns_u)
        returns_c = np.array(returns_c)

        return_u_rms_main[lid].update(returns_u)
        return_c_rms_main[lid].update(returns_c)
        returns_u = return_u_rms_main[lid].normalize(returns_u)
        returns_c = return_c_rms_main[lid].normalize(returns_c)

        returns_u_t = torch.tensor(returns_u, dtype=torch.float32, device=device)
        returns_c_t = torch.tensor(returns_c, dtype=torch.float32, device=device)

        values_u_old = torch.tensor(value_u_list, dtype=torch.float32, device=device)
        values_c_old = torch.tensor(value_c_list, dtype=torch.float32, device=device)
        log_probs_old = torch.tensor(buffer["log_probs"], dtype=torch.float32, device=device)

        for _ in range(4):
            policy_loss, value_loss, entropy = agent.distillers[lid].online_correction_update(
                task_obs_batch=torch.stack(buffer["task_obs"]),
                worker_loads_batch=torch.stack(buffer["worker_loads"]),
                worker_profiles_batch=torch.stack(buffer["worker_profile"]),
                global_context_batch=torch.stack(buffer["global_context"]),
                valid_mask_batch=torch.stack(buffer["valid_mask"]),
                actions_batch=torch.stack(buffer["actions"]),
                values_u_old=values_u_old,
                values_c_old=values_c_old,
                returns_u=returns_u_t,
                returns_c=returns_c_t,
                log_probs_old=log_probs_old,
                global_steps=global_step,
                total_training_steps=num_episodes * steps_per_episode
            )
