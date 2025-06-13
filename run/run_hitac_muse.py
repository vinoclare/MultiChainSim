import json
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque

from envs import IndustrialChain
from envs.env import MultiplexEnv
from agents.hitac_muse_agent import HiTACMuSEAgent
from utils.utils import RunningMeanStd

# ======== Load Configurations ========
num_layers = 2
env_config_path = f'../configs/{num_layers}/env_config.json'
algo_config_path = f'../configs/hitac_muse_config.json'

with open(env_config_path, 'r') as f:
    env_config = json.load(f)
with open(algo_config_path, 'r') as f:
    algo_config = json.load(f)

# === 路径配置 ===
train_schedule_path = f"../configs/{num_layers}/train_schedule.json"
eval_schedule_path = f"../configs/{num_layers}/eval_schedule.json"
worker_config_path = f"../configs/{num_layers}/worker_config.json"

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

# ======= 超参提取 =======
num_episodes = algo_config["training"]["num_episodes"]
eval_interval = algo_config["training"]["eval_interval"]
log_interval = algo_config["training"]["log_interval"]
eval_episodes = algo_config["training"]["eval_episodes"]
distill_interval = algo_config["scheduler"]["distill_interval"]
switch_interval = algo_config["scheduler"]["switch_interval"]
hitac_update_interval = algo_config["scheduler"]["hitac_update_interval"]

steps_per_episode = env_config["max_steps"]
K = algo_config["muse"]["K"]
warmup_ep = algo_config["distill"]["warmup_ep"]
min_reward_ratio = algo_config["distill"]["min_reward_ratio"]
update_epochs = algo_config["muse"]["update_epochs"]
gamma = algo_config["muse"]["gamma"]
lam = algo_config["muse"]["lam"]
batch_size = algo_config["muse"]["batch_size"]
return_norm = algo_config["muse"]["return_normalization"]

current_pid_tensor = None

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
log_dir = f'../logs/hitac_muse/{num_layers}/' + time.strftime("%Y%m%d-%H%M%S")
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

ema_beta = 0.9
pol_stats = [
    [
        {k: 0.0 for k in ["avg_reward", "avg_cost", "avg_util",
                          "avg_return", "var_rew", "var_ret"]}
        for _ in range(K)
    ] for _ in range(num_layers)
]
pol_cnt = [[0 for _ in range(K)] for _ in range(num_layers)]

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

# === 初始化 EMA baseline ===
ema_return = 0.0
ema_alpha = 0.1


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

    task_obs = torch.tensor(layer_obs["task_queue"], dtype=torch.float32, device=device).unsqueeze(
        0)  # [1, N, task_dim]
    worker_loads = torch.tensor(layer_obs["worker_loads"], dtype=torch.float32, device=device).unsqueeze(0)
    worker_profile = torch.tensor(layer_obs["worker_profile"], dtype=torch.float32, device=device).unsqueeze(0)
    global_context = torch.tensor(obs["global_context"], dtype=torch.float32, device=device).unsqueeze(0)

    return task_obs, worker_loads, worker_profile, global_context


def evaluate_policy(agent, eval_env, eval_episodes, writer, global_step, log_interval, device):
    num_layers = eval_env.num_layers
    reward_sums = {lid: [] for lid in range(num_layers)}
    assign_bonus_sums = {lid: [] for lid in range(num_layers)}
    wait_penalty_sums = {lid: [] for lid in range(num_layers)}
    cost_sums = {lid: [] for lid in range(num_layers)}
    util_sums = {lid: [] for lid in range(num_layers)}

    for episode in range(eval_episodes):
        obs = eval_env.reset(with_new_schedule=False)
        episode_reward = {lid: 0.0 for lid in range(num_layers)}
        episode_assign_bonus = {lid: 0.0 for lid in range(num_layers)}
        episode_wait_penalty = {lid: 0.0 for lid in range(num_layers)}
        episode_cost = {lid: 0.0 for lid in range(num_layers)}
        episode_util = {lid: 0.0 for lid in range(num_layers)}

        done = False
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
                action = agent.main_policy_predict(lid, obs_dict)
                actions[lid] = action.squeeze(0).detach().cpu().numpy()

            obs, (total_reward, reward_detail), done, _ = eval_env.step(actions)

            for lid in range(num_layers):
                r = reward_detail["layer_rewards"][lid]
                episode_reward[lid] += r["reward"]
                episode_assign_bonus[lid] += r["assign_bonus"]
                episode_wait_penalty[lid] += r["wait_penalty"]
                episode_cost[lid] += r["cost"]
                episode_util[lid] += r["utility"]

        for lid in range(num_layers):
            reward_sums[lid].append(episode_reward[lid])
            assign_bonus_sums[lid].append(episode_assign_bonus[lid])
            wait_penalty_sums[lid].append(episode_wait_penalty[lid])
            cost_sums[lid].append(episode_cost[lid])
            util_sums[lid].append(episode_util[lid])

        # 打印任务状态
        num_total_tasks = 0
        num_waiting_tasks = 0
        num_done_tasks = 0
        num_failed_tasks = 0
        for step_task_list in eval_env.task_schedule.values():
            for task in step_task_list:
                num_total_tasks += 1
                status = task.status
                if status == "waiting":
                    num_waiting_tasks += 1
                elif status == "done":
                    num_done_tasks += 1
                elif status == "failed":
                    num_failed_tasks += 1
        print(f"[Eval Episode {episode}] Total tasks: {num_total_tasks}, Waiting tasks: {num_waiting_tasks}, "
              f"Done tasks: {num_done_tasks}, Failed tasks: {num_failed_tasks}")

    # === TensorBoard logging ===
    total_reward_all = sum([np.mean(reward_sums[lid]) for lid in range(num_layers)])
    total_cost_all = sum([np.mean(cost_sums[lid]) for lid in range(num_layers)])
    total_util_all = sum([np.mean(util_sums[lid]) for lid in range(num_layers)])

    if global_step % log_interval == 0:
        for lid in range(num_layers):
            writer.add_scalar(f"eval/layer_{lid}_avg_reward", np.mean(reward_sums[lid]), global_step)
            writer.add_scalar(f"eval/layer_{lid}_avg_assign_bonus", np.mean(assign_bonus_sums[lid]), global_step)
            writer.add_scalar(f"eval/layer_{lid}_avg_wait_penalty", np.mean(wait_penalty_sums[lid]), global_step)
            writer.add_scalar(f"eval/layer_{lid}_avg_cost", np.mean(cost_sums[lid]), global_step)
            writer.add_scalar(f"eval/layer_{lid}_avg_utility", np.mean(util_sums[lid]), global_step)

        writer.add_scalar("global/eval_avg_reward", total_reward_all, global_step)
        writer.add_scalar("global/eval_avg_cost", total_cost_all, global_step)
        writer.add_scalar("global/eval_avg_utility", total_util_all, global_step)

    print(
        f"[Eval Summary] Total reward={total_reward_all:.2f}, cost={total_cost_all:.2f}, utility={total_util_all:.2f}")


# ======= 训练主循环 =======
for episode in range(num_episodes):
    ep_sums = [
        [dict(r=0.0, c=0.0, u=0.0, rt=0.0, r2=0.0, rt2=0.0, n=0)
         for _ in range(K)]
        for _ in range(num_layers)
    ]

    # warmup stage：轮询训练每个子策略
    if episode < (warmup_ep * K):
        current_pid_tensor = torch.tensor([episode % K for _ in range(num_layers)], device=device)

    if episode % switch_interval == 0 and episode >= (warmup_ep * K):
        # === 构造滑动平均 KPI（用于 HiTAC select）===
        local_kpis_tensor = torch.zeros((1, num_layers, 8), dtype=torch.float32, device=device)
        for lid in range(num_layers):
            if len(local_kpi_history[lid]) > 0:
                local_kpis_tensor[0, lid] = torch.stack(list(local_kpi_history[lid]), dim=0).mean(dim=0)

        if len(global_kpi_history) > 0:
            global_kpi_tensor = torch.stack(list(global_kpi_history), dim=0).mean(dim=0).unsqueeze(0)
        else:
            global_kpi_tensor = torch.zeros((1, 4), dtype=torch.float32, device=device)

        # ---- 构造 policies_info 张量 ----
        policies_info_tensor = torch.zeros((1, num_layers, K, 6), dtype=torch.float32, device=device)

        for lid in range(num_layers):
            for k in range(K):
                cnt = max(pol_cnt[lid][k], 1)
                st = pol_stats[lid][k]
                mu_r = st["avg_reward"] / cnt
                mu_c = st["avg_cost"] / cnt
                mu_u = st["avg_util"] / cnt
                mu_rt = st["avg_return"] / cnt
                var_rt = max(st["var_ret"] / cnt - mu_rt ** 2, 1e-6)
                var_r = max(st["var_rew"] / cnt - mu_r ** 2, 1e-6)
                policies_info_tensor[0, lid, k] = torch.tensor(
                                [mu_r, mu_c, mu_u, mu_rt, var_rt, var_r], device=device)

        current_pid_tensor = agent.select_subpolicies(
                local_kpis_tensor, global_kpi_tensor,
                policies_info_tensor, episode * steps_per_episode)

        for lid in range(num_layers):
            writer.add_scalar(f"layer_{lid}/selected_pid", current_pid_tensor[lid], episode)

    # # 固定子策略0
    # current_pid_tensor = torch.tensor([0 for _ in range(num_layers)])

    # # 轮询子策略
    # current_pid_tensor = torch.tensor([episode % K for _ in range(num_layers)], device=device)

    obs = env.reset(with_new_schedule=False)
    episode_reward = {lid: 0.0 for lid in range(num_layers)}
    episode_cost = {lid: 0.0 for lid in range(num_layers)}
    episode_util = {lid: 0.0 for lid in range(num_layers)}
    episode_ab = {lid: 0.0 for lid in range(num_layers)}
    episode_wp = {lid: 0.0 for lid in range(num_layers)}

    episodes_reward_deque = deque(maxlen=log_interval)
    episodes_cost_deque = deque(maxlen=log_interval)
    episodes_util_deque = deque(maxlen=log_interval)
    episodes_ab_deque = deque(maxlen=log_interval)
    episodes_wp_deque = deque(maxlen=log_interval)

    for step in range(steps_per_episode):
        # === 构造每层 obs_dict ===
        obs_dicts = {}
        for lid in range(num_layers):
            layer_obs = obs[lid]
            valid_mask = layer_obs["task_queue"][:, 3].astype(np.float32)
            obs_dicts[lid] = {
                "task_obs": torch.tensor(layer_obs["task_queue"], dtype=torch.float32, device=device).unsqueeze(0),
                "worker_loads": torch.tensor(layer_obs["worker_loads"], dtype=torch.float32, device=device).unsqueeze(
                    0),
                "worker_profile": torch.tensor(layer_obs["worker_profile"], dtype=torch.float32,
                                               device=device).unsqueeze(0),
                "global_context": torch.tensor(obs["global_context"], dtype=torch.float32, device=device).unsqueeze(0),
                "valid_mask": torch.tensor(valid_mask, dtype=torch.float32, device=device).unsqueeze(0)
            }

        # === Agent 选择子策略 & 采样动作 ===
        sample_out = agent.sample(obs_dicts, current_pid_tensor)

        actions = {lid: sample_out[lid]["actions"].squeeze(0).cpu().numpy() for lid in range(num_layers)}
        sample_vu = {lid: sample_out[lid]["v_u"].item() for lid in range(num_layers)}
        sample_vc = {lid: sample_out[lid]["v_c"].item() for lid in range(num_layers)}
        pid = torch.stack([sample_out[lid]["pid"] for lid in range(num_layers)], dim=0)  # (L)

        # === 与环境交互 ===
        obs, (total_reward, reward_detail), done, _ = env.step(actions)

        for lid in range(num_layers):
            alpha_k = agent.muses[lid].alphas[current_pid_tensor[lid]].item()
            beta_k = agent.muses[lid].betas[current_pid_tensor[lid]].item()

            # === 奖励信息 ===
            rew = reward_detail["layer_rewards"][lid]["reward"]
            cost = reward_detail["layer_rewards"][lid]["cost"]
            util = reward_detail["layer_rewards"][lid]["utility"]
            ab = reward_detail["layer_rewards"][lid]["assign_bonus"]
            wp = reward_detail["layer_rewards"][lid]["wait_penalty"]
            total_u = beta_k * util + ab
            total_c = -alpha_k * cost - wp
            fused_reward = total_u + total_c

            episode_reward[lid] += rew
            episode_cost[lid] += cost
            episode_util[lid] += util
            episode_ab[lid] += ab
            episode_wp[lid] += wp

            # ---- 统计到 ep_sums ----
            cur_pid = current_pid_tensor[lid].item()
            bucket = ep_sums[lid][cur_pid]
            bucket["r"] += rew
            bucket["c"] += cost
            bucket["u"] += util
            bucket["rt"] += fused_reward
            bucket["r2"] += rew * rew
            bucket["rt2"] += fused_reward * fused_reward
            bucket["n"] += 1

            # === 存入 PPO buffer ===
            buffers[lid]["task_obs"].append(obs_dicts[lid]["task_obs"].squeeze(0))
            buffers[lid]["worker_loads"].append(obs_dicts[lid]["worker_loads"].squeeze(0))
            buffers[lid]["worker_profile"].append(obs_dicts[lid]["worker_profile"].squeeze(0))
            buffers[lid]["global_context"].append(obs_dicts[lid]["global_context"].squeeze(0))
            buffers[lid]["valid_mask"].append(obs_dicts[lid]["valid_mask"].squeeze(0))
            buffers[lid]["actions"].append(sample_out[lid]["actions"].squeeze(0))
            buffers[lid]["value_u"].append(sample_vu[lid])
            buffers[lid]["value_c"].append(sample_vc[lid])
            buffers[lid]["logprobs"].append(sample_out[lid]["logp"].item())
            buffers[lid]["reward_u"].append(total_u)
            buffers[lid]["reward_c"].append(total_c)
            buffers[lid]["rewards"].append(fused_reward)
            buffers[lid]["dones"].append(done)
            buffers[lid]["pid"].append(sample_out[lid]["pid"].item())

        if done:
            break

    global_step = (episode + 1) * steps_per_episode
    if (episode % eval_interval == 0) and (episode > warmup_ep * K):
        evaluate_policy(agent, eval_env, eval_episodes, writer, global_step, log_interval, device)

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

        returns_u, advs_u = compute_gae_single_head(reward_u, dones, value_u, 0.0, gamma, lam)
        returns_c, advs_c = compute_gae_single_head(reward_c, dones, value_c, 0.0, gamma, lam)

        advantages = [au + ac for au, ac in zip(advs_u, advs_c)]

        # === Step 2: 可选 return 归一化 ===
        if return_norm:
            ret_u_np = np.array(returns_u)
            ret_c_np = np.array(returns_c)
            return_u_rms[lid].update(ret_u_np)
            return_c_rms[lid].update(ret_c_np)
            returns_u = return_u_rms[lid].normalize(ret_u_np)
            returns_c = return_c_rms[lid].normalize(ret_c_np)

            advs_u = returns_u - np.array(value_u)
            advs_c = returns_c - np.array(value_c)
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

    # ===== 统计各种任务状态的数量 =====
    num_total_tasks = 0
    num_waiting_tasks = 0
    num_done_tasks = 0
    num_failed_tasks = 0
    for step_task_list in env.task_schedule.values():
        for task in step_task_list:
            num_total_tasks += 1
            status = task.status
            if status == "waiting":
                num_waiting_tasks += 1
            elif status == "done":
                num_done_tasks += 1
            elif status == "failed":
                num_failed_tasks += 1
    global_done_rate = num_done_tasks / num_total_tasks
    print(f"[Episode {episode}] Total tasks: {num_total_tasks}, Waiting tasks: {num_waiting_tasks}, "
          f"Done tasks: {num_done_tasks}, Failed tasks: {num_failed_tasks}")

    # ---------- 更新 EMA 统计 ----------
    for lid in range(num_layers):
        for k in range(K):
            n = ep_sums[lid][k]["n"]
            if n == 0:
                continue

            coef = 1.0 - ema_beta
            pol_cnt[lid][k] = ema_beta * pol_cnt[lid][k] + coef

            mean_r = ep_sums[lid][k]["r"] / n
            mean_c = ep_sums[lid][k]["c"] / n
            mean_u = ep_sums[lid][k]["u"] / n
            mean_rt = ep_sums[lid][k]["rt"] / n
            var_r = ep_sums[lid][k]["r2"] / n - mean_r ** 2
            var_rt = ep_sums[lid][k]["rt2"] / n - mean_rt ** 2

            s = pol_stats[lid][k]
            s["avg_reward"] = ema_beta * s["avg_reward"] + coef * mean_r
            s["avg_cost"] = ema_beta * s["avg_cost"] + coef * mean_c
            s["avg_util"] = ema_beta * s["avg_util"] + coef * mean_u
            s["avg_return"] = ema_beta * s["avg_return"] + coef * mean_rt
            s["var_rew"] = ema_beta * s["var_rew"] + coef * var_r
            s["var_ret"] = ema_beta * s["var_ret"] + coef * var_rt

    # === 构造当前 episode 原始 KPI（用于 hitac_update）===
    raw_local_kpis = torch.zeros((1, num_layers, 8), dtype=torch.float32, device=device)
    for lid in range(num_layers):
        reward = episode_reward[lid]
        cost = episode_cost[lid]
        util = episode_util[lid]
        ab = episode_ab[lid]
        wp = episode_wp[lid]
        pid_val = current_pid_tensor[lid].item()
        done_rate, task_load_ratio = env.chain.layers[lid].get_kpi_snapshot()

        local_kpi = torch.tensor([
            reward, cost, util, ab, wp,
            done_rate, task_load_ratio, pid_val
        ], dtype=torch.float32, device=device)

        raw_local_kpis[0, lid] = local_kpi

        local_kpi_history[lid].append(local_kpi)

    # === 构造 global_kpi（用于 update + 缓存） ===
    reward_sum = sum(episode_reward.values())
    cost_sum = sum(episode_cost.values())
    util_sum = sum(episode_util.values())
    wait_penalty_sum = sum(episode_wp.values())
    assign_bonus_sum = sum(episode_ab.values())

    episodes_reward_deque.append(reward_sum)
    episodes_cost_deque.append(cost_sum)
    episodes_util_deque.append(util_sum)
    episodes_wp_deque.append(wait_penalty_sum)
    episodes_ab_deque.append(assign_bonus_sum)

    raw_global_kpi = torch.tensor([
        reward_sum, cost_sum, util_sum, global_done_rate
    ], dtype=torch.float32, device=device).unsqueeze(0)

    global_kpi_history.append(raw_global_kpi.squeeze(0))

    # === Logging episode score ===
    if episode % log_interval == 0:
        for lid in range(num_layers):
            writer.add_scalar(f"layer_{lid}/reward", episode_reward[lid], global_step)
            writer.add_scalar(f"layer_{lid}/cost", episode_cost[lid], global_step)
            writer.add_scalar(f"layer_{lid}/utility", episode_util[lid], global_step)
            writer.add_scalar(f"layer_{lid}/assign_bonus", episode_ab[lid], global_step)
            writer.add_scalar(f"layer_{lid}/wait_penalty", episode_wp[lid], global_step)
        writer.add_scalar("global/episode_total_cost", np.mean(episodes_cost_deque), global_step)
        writer.add_scalar("global/episode_total_reward", np.mean(episodes_reward_deque), global_step)
        writer.add_scalar("global/episode_total_utility", np.mean(episodes_util_deque), global_step)
        writer.add_scalar("global/episode_assign_bonus", np.mean(episodes_ab_deque), global_step)
        writer.add_scalar("global/episode_wait_penalty", np.mean(episodes_wp_deque), global_step)

    # === 存入蒸馏 buffer ===
    if reward_sum > np.mean(episodes_reward_deque) * min_reward_ratio:
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
    if episode % hitac_update_interval and episode >= (warmup_ep * K) == 0:
        # ---- 构造 policies_info 张量 ----
        policies_info_tensor = torch.zeros((1, num_layers, K, 6), dtype=torch.float32, device=device)

        for lid in range(num_layers):
            for k in range(K):
                cnt = max(pol_cnt[lid][k], 1)
                st = pol_stats[lid][k]
                mu_r = st["avg_reward"] / cnt
                mu_c = st["avg_cost"] / cnt
                mu_u = st["avg_util"] / cnt
                mu_rt = st["avg_return"] / cnt
                var_rt = max(st["var_ret"] / cnt - mu_rt ** 2, 1e-6)
                var_r = max(st["var_rew"] / cnt - mu_r ** 2, 1e-6)
                policies_info_tensor[0, lid, k] = torch.tensor(
                    [mu_r, mu_c, mu_u, mu_rt, var_rt, var_r], device=device)

        # === 计算 episode-level return ===
        layer_returns = []
        for lid in range(num_layers):
            pid_k = current_pid_tensor[lid].item()
            alpha_k = agent.muses[lid].alphas[pid_k].item()
            beta_k = agent.muses[lid].betas[pid_k].item()

            total_u = beta_k * episode_util[lid] + episode_ab[lid]
            total_c = -alpha_k * episode_cost[lid] - episode_wp[lid]
            layer_returns.append(total_u + total_c)  # scalar

        returns_L = torch.tensor(layer_returns, dtype=torch.float32, device=device)  # shape (L,)

        hitac_stats = agent.hitac_update(raw_local_kpis,
                                         raw_global_kpi,
                                         policies_info_tensor,
                                         returns_L,  # shape (L,)
                                         global_step)

        # === TensorBoard 记录 ===
        if episode % log_interval == 0:
            for k, v in hitac_stats.items():
                writer.add_scalar(k, v, episode)

    # === 蒸馏更新 ===
    if episode % distill_interval == 0 and episode >= (warmup_ep * K):
        for lid in range(num_layers):
            loss = agent.distill_update(lid, current_pid_tensor[lid].item())
            writer.add_scalar(f"distill/layer_{lid}_loss", loss, episode)
