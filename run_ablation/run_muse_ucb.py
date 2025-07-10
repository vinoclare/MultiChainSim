import json
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import os
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

        # # 打印任务状态
        # num_total_tasks = 0
        # num_waiting_tasks = 0
        # num_done_tasks = 0
        # num_failed_tasks = 0
        # for step_task_list in eval_env.task_schedule.values():
        #     for task in step_task_list:
        #         num_total_tasks += 1
        #         status = task.status
        #         if status == "waiting":
        #             num_waiting_tasks += 1
        #         elif status == "done":
        #             num_done_tasks += 1
        #         elif status == "failed":
        #             num_failed_tasks += 1
        # print(f"[Eval Episode {episode}] Total tasks: {num_total_tasks}, Waiting tasks: {num_waiting_tasks}, "
        #       f"Done tasks: {num_done_tasks}, Failed tasks: {num_failed_tasks}")

    # === TensorBoard logging ===
    total_reward_all = sum([np.mean(reward_sums[lid]) for lid in range(num_layers)])
    total_cost_all = sum([np.mean(cost_sums[lid]) for lid in range(num_layers)])
    total_util_all = sum([np.mean(util_sums[lid]) for lid in range(num_layers)])
    episode_total_rewards = [
        sum([reward_sums[lid][i] for lid in range(num_layers)])
        for i in range(len(reward_sums[0]))
    ]
    global_reward_std = np.std(episode_total_rewards)

    for lid in range(num_layers):
        writer.add_scalar(f"eval/layer_{lid}_avg_reward", np.mean(reward_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_assign_bonus", np.mean(assign_bonus_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_wait_penalty", np.mean(wait_penalty_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_cost", np.mean(cost_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_utility", np.mean(util_sums[lid]), global_step)

    writer.add_scalar("global/eval_avg_reward", total_reward_all, global_step)
    writer.add_scalar("global/eval_avg_cost", total_cost_all, global_step)
    writer.add_scalar("global/eval_avg_utility", total_util_all, global_step)
    writer.add_scalar("global/eval_reward_std", global_reward_std, global_step)

    print(f"[Eval Summary] Total reward={total_reward_all:.2f}, cost={total_cost_all:.2f}, utility={total_util_all:.2f}")


# ======= 训练主循环 =======
for episode in range(num_episodes):
    ep_sums = [
        [dict(r=0.0, c=0.0, u=0.0, rt=0.0, r2=0.0, rt2=0.0, n=0, ent_sum=0.0, std_sum=0.0)
         for _ in range(K)]
        for _ in range(num_layers)
    ]

    # warmup stage：轮询训练每个子策略
    if episode < (warmup_ep * K):
        current_pid_tensor = torch.tensor([episode % K for _ in range(num_layers)], device=device)

    if episode % switch_interval == 0 and episode >= (warmup_ep * K):
        select_cnt += 1

        # ===== 负策略固定选择 =====
        if select_cnt % neg_interval == 0 and neg_policy:
            skip_hitac_train = True
            current_pid_tensor = torch.tensor([K - 1 for _ in range(num_layers)])
            distill_pid = torch.tensor([K - 1 for _ in range(num_layers)])

        elif select_cnt % neg_interval == 1 and neg_policy:
            skip_hitac_train = True
            current_pid_tensor = torch.tensor([K - 2 for _ in range(num_layers)])
            distill_pid = torch.tensor([K - 2 for _ in range(num_layers)])

        # ===== 替换原 HiTAC：使用基于 reward 的 UCB 选择 =====
        else:
            skip_hitac_train = True  # 注意：不训练 HiTAC

            current_pid_tensor = torch.zeros(num_layers, dtype=torch.long)
            distill_pid = torch.zeros(num_layers, dtype=torch.long)

            for lid in range(num_layers):
                total_counts = sum([len(pol_hist[lid][k]) for k in range(num_pos_subpolicies)])
                total_counts = max(total_counts, 1)

                ucb_scores = []
                for k in range(num_pos_subpolicies):
                    hist = pol_hist[lid][k]
                    n_k = len(hist)
                    if n_k == 0:
                        ucb = float("inf")  # 鼓励探索未选过的
                    else:
                        avg_reward = np.mean([h["r"] for h in hist])
                        r_std = np.std([h["r"] for h in hist]) + 1e-6
                        bonus = 0.5 * r_std * np.sqrt(np.log(total_counts) / (n_k + 1e-6))
                        ucb = avg_reward + bonus
                    ucb_scores.append(ucb)

                best_k = int(np.argmax(ucb_scores))
                current_pid_tensor[lid] = best_k
                distill_pid[lid] = best_k

        for lid in range(num_layers):
            writer.add_scalar(f"layer_{lid}/selected_pid", current_pid_tensor[lid], episode)

    # # 固定子策略0
    # current_pid_tensor = torch.tensor([0 for _ in range(num_layers)])

    # # 轮询子策略
    # current_pid_tensor = torch.tensor([episode % K for _ in range(num_layers)], device=device)

    if (episode + 1) % reset_schedule_interval == 0:
        obs = env.reset(with_new_schedule=True)
    else:
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
            bucket["ent_sum"] += sample_out[lid]["ent"].item()
            bucket["std_sum"] += sample_out[lid]["act_std"].item()
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

    # # ===== 统计各种任务状态的数量 =====
    # num_total_tasks = 0
    # num_waiting_tasks = 0
    # num_done_tasks = 0
    # num_failed_tasks = 0
    # for step_task_list in env.task_schedule.values():
    #     for task in step_task_list:
    #         num_total_tasks += 1
    #         status = task.status
    #         if status == "waiting":
    #             num_waiting_tasks += 1
    #         elif status == "done":
    #             num_done_tasks += 1
    #         elif status == "failed":
    #             num_failed_tasks += 1
    # global_done_rate = num_done_tasks / num_total_tasks
    # print(f"[Episode {episode}] Total tasks: {num_total_tasks}, Waiting tasks: {num_waiting_tasks}, "
    #       f"Done tasks: {num_done_tasks}, Failed tasks: {num_failed_tasks}")

    # ---------- update policies_info ----------
    for lid in range(num_layers):
        for k in range(num_pos_subpolicies):
            n = ep_sums[lid][k]["n"]
            if n == 0:
                continue

            mean_r = ep_sums[lid][k]["r"] / n
            mean_c = ep_sums[lid][k]["c"] / n
            mean_u = ep_sums[lid][k]["u"] / n
            mean_rt = ep_sums[lid][k]["rt"] / n
            var_r = max(ep_sums[lid][k]["r2"] / n - mean_r ** 2, 1e-6)
            var_rt = max(ep_sums[lid][k]["rt2"] / n - mean_rt ** 2, 1e-6)
            mean_ent = ep_sums[lid][k]["ent_sum"] / n
            mean_std = ep_sums[lid][k]["std_sum"] / n

            pol_hist[lid][k].append({
                "r": mean_r,
                "c": mean_c,
                "u": mean_u,
                "rt": mean_rt,
                "var_r": var_r,
                "var_rt": var_rt,
                "ent": mean_ent,
                "std": mean_std
            })

    # === 构造当前 episode 原始 KPI（用于 hitac_update）===
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
        reward_sum, cost_sum, util_sum
    ], dtype=torch.float32, device=device)

    global_kpi_history.append(raw_global_kpi)

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
    if episode % hitac_update_interval == 0 and episode >= (warmup_ep * K) and not skip_hitac_train:
        # ---- 构造 kpi 张量 ----
        local_kpis_tensor = torch.zeros((1, num_layers, 8), dtype=torch.float32, device=device)
        for lid in range(num_layers):
            if len(local_kpi_history[lid]) > 0:
                local_kpis_tensor[0, lid] = torch.stack(list(local_kpi_history[lid]), dim=0).mean(dim=0)

        if len(global_kpi_history) > 0:
            global_kpi_tensor = torch.stack(list(global_kpi_history), dim=0).mean(dim=0).unsqueeze(0)
        else:
            global_kpi_tensor = torch.zeros((1, 4), dtype=torch.float32, device=device)

        # ---- 构造 policies_info 张量 ----
        policies_info_tensor = torch.zeros((1, num_layers, num_pos_subpolicies, policies_info_dim), dtype=torch.float32, device=device)
        for lid in range(num_layers):
            for k in range(num_pos_subpolicies):
                hist = pol_hist[lid][k]
                if len(hist) == 0:
                    mu_r = mu_c = mu_u = mu_rt = ent = std = 0.0
                    var_r = var_rt = 1e-6
                else:
                    mu_r = np.mean([h["r"] for h in hist])
                    mu_c = np.mean([h["c"] for h in hist])
                    mu_u = np.mean([h["u"] for h in hist])
                    mu_rt = np.mean([h["rt"] for h in hist])
                    var_r = max(np.mean([h["var_r"] for h in hist]), 1e-6)
                    var_rt = max(np.mean([h["var_rt"] for h in hist]), 1e-6)
                    ent = np.mean([h["ent"] for h in hist])
                    std = np.mean([h["std"] for h in hist])

                policies_info_tensor[0, lid, k] = torch.tensor(
                    [mu_r, mu_c, mu_u, mu_rt, var_rt, var_r, ent, std], device=device)

        # === 计算 episode-level return ===
        layer_returns = [episode_reward[lid] for lid in range(num_layers)]
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
