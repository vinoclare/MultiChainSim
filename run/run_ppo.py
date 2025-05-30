import torch
import numpy as np

from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.ppo_model import PPOIndustrialModel
from algs.ppo import PPO
from agents.agent import IndustrialAgent
import json
import time
import random
from torch.utils.tensorboard import SummaryWriter

from utils import RunningMeanStd

# ===== Load configurations =====
env_config_path = '../configs/env_config_simple.json'
ppo_config_path = '../configs/ppo_config.json'
with open(env_config_path, 'r') as f:
    env_config = json.load(f)

with open(ppo_config_path, 'r') as f:
    ppo_config = json.load(f)

# ===== Setup environment =====
env = MultiplexEnv(env_config_path)
eval_env = MultiplexEnv(env_config_path)
eval_env.worker_config = env.worker_config
eval_env.chain = IndustrialChain(eval_env.worker_config)
num_layers = env_config["num_layers"]
max_steps = env_config["max_steps"]

# ===== Hyperparameters =====
num_episodes = ppo_config["num_episodes"]
steps_per_episode = env_config["max_steps"]
update_epochs = ppo_config["update_epochs"]
gamma = ppo_config["gamma"]
lam = ppo_config["lam"]
batch_size = ppo_config["batch_size"]
hidden_dim = ppo_config["hidden_dim"]
eval_interval = ppo_config["eval_interval"]
eval_episodes = ppo_config["eval_episodes"]
log_interval = ppo_config["log_interval"]
reset_schedule_every = env_config["reset_schedule_every"]

agents = {}
buffers = {}
global_step = [0]

log_dir = '../logs/ppo/' + time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)
return_rms = {lid: RunningMeanStd() for lid in range(num_layers)}

# ===== Initialize agents and buffers for each layer =====
for layer_id in range(num_layers):
    obs_space = env.observation_space[layer_id]
    act_space = env.action_space[layer_id]
    n_worker, _ = act_space.shape

    task_input_dim = obs_space['task_queue'].shape[1]
    worker_input_dim = obs_space['worker_loads'].shape[1]
    n_task_types = len(env_config["task_types"])
    profile_dim = 2 * n_task_types
    global_context_dim = 1

    model = PPOIndustrialModel(
        task_input_dim=task_input_dim,
        worker_load_input_dim=worker_input_dim,
        worker_profile_input_dim=profile_dim,
        n_worker=n_worker,
        num_pad_tasks=env.num_pad_tasks,
        global_context_dim=global_context_dim,
        hidden_dim=hidden_dim
    )

    alg = PPO(
        model,
        clip_param=ppo_config["clip_param"],
        value_loss_coef=ppo_config["value_loss_coef"],
        entropy_coef=ppo_config["entropy_coef"],
        initial_lr=ppo_config["initial_lr"],
        max_grad_norm=ppo_config["max_grad_norm"],
        writer=writer,
        global_step_ref=global_step,
        total_training_steps=ppo_config["num_episodes"] * env_config["max_steps"]
    )

    agents[layer_id] = IndustrialAgent(alg, "ppo", env.num_pad_tasks)

    buffers[layer_id] = {
        'task_obs': [],
        'worker_loads': [],
        'worker_profile': [],
        'global_context': [],
        'valid_mask': [],
        'actions': [],
        'logprobs': [],
        'rewards': [],
        'dones': [],
        'values': []
    }


def process_obs(raw_obs, layer_id):
    layer_obs = raw_obs[layer_id]
    task_obs = layer_obs['task_queue']
    worker_loads = layer_obs['worker_loads']
    worker_profile = layer_obs.get('worker_profile')  # 新增
    global_context = raw_obs.get('global_context')  # 新增
    return task_obs, worker_loads, worker_profile, global_context


def compute_gae(rewards, dones, values, last_value, gamma, lam):
    advantages = []
    gae = 0
    values_ext = values + [last_value]
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def evaluate_policy(agents, eval_env, eval_episodes, writer, global_step):
    """
    评估当前 agents 策略在 eval_env 上的表现（无探索）。
    会记录每层的 avg reward, cost, utility 到 TensorBoard。

    Args:
        agents (Dict[int, PPOAgent]): 每一层的 agent。
        eval_env (IndustrialEnv): 用于测试泛化性能的环境。
        eval_episodes (int): 评估次数。
        writer (SummaryWriter): TensorBoard writer。
        global_step (int): 当前训练步数（用于写入 tag）。
    """
    reward_sums = {lid: [] for lid in agents}
    assign_bouns_sum = {lid: [] for lid in agents}
    wait_penalty_sum = {lid: [] for lid in agents}
    cost_sums = {lid: [] for lid in agents}
    util_sums = {lid: [] for lid in agents}

    num_total_tasks = 0
    num_waiting_tasks = 0
    num_done_tasks = 0
    num_failed_tasks = 0

    for _ in range(eval_episodes):
        obs = eval_env.reset(with_new_schedule=False)
        episode_reward = {lid: 0.0 for lid in agents}
        episode_assign_bonus = {lid: 0.0 for lid in agents}
        episode_wait_penalty = {lid: 0.0 for lid in agents}
        episode_cost = {lid: 0.0 for lid in agents}
        episode_util = {lid: 0.0 for lid in agents}

        done = False
        while not done:
            actions = {}
            for lid in agents:
                task_obs, worker_loads, worker_profile, global_context = process_obs(obs, lid)
                action = agents[lid].predict(task_obs, worker_loads, worker_profile, global_context)
                actions[lid] = action

            obs, reward_dict, done, _ = eval_env.step(actions)

            for lid in agents:
                layer_reward = reward_dict[1]["layer_rewards"][lid]["raw_reward"]
                layer_assign_bonus = reward_dict[1]["layer_rewards"][lid]["assign_bonus"]
                layer_wait_penalty = reward_dict[1]["layer_rewards"][lid]["wait_penalty"]
                layer_cost = reward_dict[1]["layer_rewards"][lid]["cost"]
                layer_util = reward_dict[1]["layer_rewards"][lid]["utility"]

                episode_reward[lid] += layer_reward
                episode_assign_bonus[lid] += layer_assign_bonus
                episode_wait_penalty[lid] += layer_wait_penalty
                episode_cost[lid] += layer_cost
                episode_util[lid] += layer_util

        for lid in agents:
            reward_sums[lid].append(episode_reward[lid])
            assign_bouns_sum[lid].append(episode_assign_bonus[lid])
            wait_penalty_sum[lid].append(episode_wait_penalty[lid])
            cost_sums[lid].append(episode_cost[lid])
            util_sums[lid].append(episode_util[lid])

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

    print(f"[Eval total tasks: {num_total_tasks / eval_episodes}, Waiting tasks: {num_waiting_tasks / eval_episodes}, "
          f"Done tasks: {num_done_tasks / eval_episodes}, Failed tasks: {num_failed_tasks / eval_episodes}")

    # === 写入 TensorBoard ===
    for lid in agents:
        writer.add_scalar(f"eval/layer_{lid}_avg_reward", np.mean(reward_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_assign_bonus", np.mean(assign_bouns_sum[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_wait_penalty", np.mean(wait_penalty_sum[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_cost", np.mean(cost_sums[lid]), global_step)
        writer.add_scalar(f"eval/layer_{lid}_avg_utility", np.mean(util_sums[lid]), global_step)
        print(f"[Eval] Layer {lid}: reward={np.mean(reward_sums[lid]):.2f}, "
              f"cost={np.mean(cost_sums[lid]):.2f}, utility={np.mean(util_sums[lid]):.2f}")


# ===== Training loop =====
# 用于 tensorboard 记录的奖励、代价、效用等
reward_buffer = {lid: [] for lid in range(num_layers)}
raw_reward_buffer = {lid: [] for lid in range(num_layers)}
assign_bonus_buffer = {lid: [] for lid in range(num_layers)}
raw_assign_bonus_buffer = {lid: [] for lid in range(num_layers)}
wait_penalty_buffer = {lid: [] for lid in range(num_layers)}
raw_wait_penalty_buffer = {lid: [] for lid in range(num_layers)}
cost_buffer = {lid: [] for lid in range(num_layers)}
raw_cost_buffer = {lid: [] for lid in range(num_layers)}
util_buffer = {lid: [] for lid in range(num_layers)}
raw_util_buffer = {lid: [] for lid in range(num_layers)}

for episode in range(num_episodes):
    if episode % eval_interval == 0:
        evaluate_policy(agents, eval_env, eval_episodes, writer, episode * max_steps)

    if (episode + 1) % reset_schedule_every == 0:
        obs = env.reset(with_new_schedule=True)
    else:
        obs = env.reset(with_new_schedule=False)
    episode_rewards = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_raw_rewards = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_assign_bonus = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_raw_assign_bonus = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_wait_penalty = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_raw_wait_penalty = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_cost = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_raw_cost = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_util = {layer_id: 0.0 for layer_id in range(num_layers)}
    episode_raw_util = {layer_id: 0.0 for layer_id in range(num_layers)}

    for step in range(steps_per_episode):
        actions = {}

        for layer_id in range(num_layers):
            task_obs, worker_loads, worker_profile, global_context = process_obs(obs, layer_id)
            value, action, logprob, _, raw_action = agents[layer_id].sample(
                                task_obs, worker_loads, worker_profile, global_context)

            actions[layer_id] = action
            valid_mask = task_obs[:, 3].astype(np.float32)

            buffers[layer_id]['task_obs'].append(task_obs)
            buffers[layer_id]['worker_loads'].append(worker_loads)
            buffers[layer_id]['worker_profile'].append(worker_profile)
            buffers[layer_id]['global_context'].append(global_context)
            buffers[layer_id]['valid_mask'].append(valid_mask)
            buffers[layer_id]['actions'].append(raw_action)
            buffers[layer_id]['logprobs'].append(logprob)
            buffers[layer_id]['values'].append(value)

        obs, reward_dict, done, _ = env.step(actions)

        for layer_id in range(num_layers):
            reward_scalar = reward_dict[1]["layer_rewards"][layer_id]["reward"]
            raw_reward_scalar = reward_dict[1]["layer_rewards"][layer_id]["raw_reward"]
            assign_bonus_scalar = reward_dict[1]["layer_rewards"][layer_id]["assign_bonus"]
            raw_assign_bonus_scalar = reward_dict[1]["layer_rewards"][layer_id]["raw_assign"]
            wait_penalty_scalar = reward_dict[1]["layer_rewards"][layer_id]["wait_penalty"]
            raw_wait_penalty_scalar = reward_dict[1]["layer_rewards"][layer_id]["raw_wait"]
            cost_scalar = reward_dict[1]['layer_rewards'][layer_id]['cost']
            raw_cost_scalar = reward_dict[1]['layer_rewards'][layer_id]['raw_cost']
            util_scalar = reward_dict[1]['layer_rewards'][layer_id]['utility']
            raw_util_scalar = reward_dict[1]['layer_rewards'][layer_id]['raw_utility']

            buffers[layer_id]['rewards'].append(reward_scalar)
            buffers[layer_id]['dones'].append(done)
            episode_rewards[layer_id] += reward_scalar
            episode_raw_rewards[layer_id] += raw_reward_scalar
            episode_assign_bonus[layer_id] += assign_bonus_scalar
            episode_raw_assign_bonus[layer_id] += raw_assign_bonus_scalar
            episode_wait_penalty[layer_id] += wait_penalty_scalar
            episode_raw_wait_penalty[layer_id] += raw_wait_penalty_scalar
            episode_cost[layer_id] += cost_scalar
            episode_raw_cost[layer_id] += raw_cost_scalar
            episode_util[layer_id] += util_scalar
            episode_raw_util[layer_id] += raw_util_scalar
        if done:
            break

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
    print(f"[Episode {episode}] Total tasks: {num_total_tasks}, Waiting tasks: {num_waiting_tasks}, "
          f"Done tasks: {num_done_tasks}, Failed tasks: {num_failed_tasks}")

    # ===== 记录每层的奖励、代价、效用等 =====
    for layer_id in range(num_layers):
        reward_buffer[layer_id].append(episode_rewards[layer_id])
        raw_reward_buffer[layer_id].append(episode_raw_rewards[layer_id])
        assign_bonus_buffer[layer_id].append(episode_assign_bonus[layer_id])
        raw_assign_bonus_buffer[layer_id].append(episode_raw_assign_bonus[layer_id])
        wait_penalty_buffer[layer_id].append(episode_wait_penalty[layer_id])
        raw_wait_penalty_buffer[layer_id].append(episode_raw_wait_penalty[layer_id])
        cost_buffer[layer_id].append(episode_cost[layer_id])
        raw_cost_buffer[layer_id].append(episode_raw_cost[layer_id])
        util_buffer[layer_id].append(episode_util[layer_id])
        raw_util_buffer[layer_id].append(episode_raw_util[layer_id])

    # ===== 每 N 个 episode 写一次 TensorBoard 均值 =====
    if (episode + 1) % log_interval == 0:
        for layer_id in range(num_layers):
            writer.add_scalar(f"layer_{layer_id}/avg_episode_reward",
                              np.mean(reward_buffer[layer_id]), episode)
            writer.add_scalar(f"layer_{layer_id}/avg_raw_reward",
                              np.mean(raw_reward_buffer[layer_id]), episode)
            writer.add_scalar(f"layer_{layer_id}/avg_episode_assign_bonus",
                              np.mean(assign_bonus_buffer[layer_id]), episode)
            writer.add_scalar(f"layer_{layer_id}/avg_raw_assign_bonus",
                              np.mean(raw_assign_bonus_buffer[layer_id]), episode)
            writer.add_scalar(f"layer_{layer_id}/avg_episode_wait_penalty",
                              np.mean(wait_penalty_buffer[layer_id]), episode)
            writer.add_scalar(f"layer_{layer_id}/avg_raw_wait_penalty",
                              np.mean(raw_wait_penalty_buffer[layer_id]), episode)
            writer.add_scalar(f"layer_{layer_id}/avg_episode_cost",
                              np.mean(cost_buffer[layer_id]), episode)
            writer.add_scalar(f"layer_{layer_id}/avg_raw_cost",
                              np.mean(raw_cost_buffer[layer_id]), episode)
            writer.add_scalar(f"layer_{layer_id}/avg_episode_utility",
                              np.mean(util_buffer[layer_id]), episode)
            writer.add_scalar(f"layer_{layer_id}/avg_raw_utility",
                              np.mean(raw_util_buffer[layer_id]), episode)
            reward_buffer[layer_id].clear()
            raw_reward_buffer[layer_id].clear()
            assign_bonus_buffer[layer_id].clear()
            raw_assign_bonus_buffer[layer_id].clear()
            wait_penalty_buffer[layer_id].clear()
            raw_wait_penalty_buffer[layer_id].clear()
            cost_buffer[layer_id].clear()
            raw_cost_buffer[layer_id].clear()
            util_buffer[layer_id].clear()
            raw_util_buffer[layer_id].clear()

    # ===== GAE & PPO updates per layer =====
    for layer_id in range(num_layers):
        task_obs = torch.tensor(buffers[layer_id]['task_obs'][-1], dtype=torch.float32).unsqueeze(0)
        worker_loads = torch.tensor(buffers[layer_id]['worker_loads'][-1], dtype=torch.float32).unsqueeze(0)
        worker_profile = torch.tensor(buffers[layer_id]['worker_profile'][-1], dtype=torch.float32).unsqueeze(0)
        global_context = torch.tensor(buffers[layer_id]['global_context'][-1], dtype=torch.float32).unsqueeze(0)
        valid_mask = torch.tensor(buffers[layer_id]['valid_mask'][-1], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, _, last_value = agents[layer_id].alg.model(
                task_obs, worker_loads, worker_profile, global_context, valid_mask
            )
        last_value = last_value.item()

        advs, rets = compute_gae(
            buffers[layer_id]['rewards'],
            buffers[layer_id]['dones'],
            buffers[layer_id]['values'],
            last_value, gamma, lam
        )

        if ppo_config["return_normalization"]:
            rets_np = np.array(rets)
            return_rms[layer_id].update(rets_np)
            rets = return_rms[layer_id].normalize(rets_np)

        dataset = list(zip(
            buffers[layer_id]['task_obs'],
            buffers[layer_id]['worker_loads'],
            buffers[layer_id]['worker_profile'],
            buffers[layer_id]['global_context'],
            buffers[layer_id]['valid_mask'],
            buffers[layer_id]['actions'],
            buffers[layer_id]['values'],
            rets,
            buffers[layer_id]['logprobs'],
            advs
        ))

        for _ in range(update_epochs):
            random.shuffle(dataset)
            for i in range(0, len(dataset), batch_size):
                minibatch = dataset[i:i+batch_size]
                task_batch, worker_batch, profile_batch, gctx_batch, mask_batch, \
                    act_batch, val_batch, ret_batch, logp_batch, adv_batch = zip(*minibatch)

                agents[layer_id].learn(
                    torch.tensor(np.array(task_batch), dtype=torch.float32),
                    torch.tensor(np.array(worker_batch), dtype=torch.float32),
                    torch.tensor(np.array(profile_batch), dtype=torch.float32),
                    torch.tensor(np.array(gctx_batch), dtype=torch.float32),
                    torch.tensor(np.array(mask_batch), dtype=torch.float32),
                    torch.tensor(np.array(act_batch), dtype=torch.float32),
                    torch.tensor(np.array(val_batch), dtype=torch.float32),
                    torch.tensor(np.array(ret_batch), dtype=torch.float32),
                    torch.tensor(np.array(logp_batch), dtype=torch.float32),
                    torch.tensor(np.array(adv_batch), dtype=torch.float32),
                )

        buffers[layer_id] = {k: [] for k in buffers[layer_id]}

    print(f"[Episode {episode}] Rewards per layer: {episode_rewards}")
