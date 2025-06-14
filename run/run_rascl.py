import json
import os
import time
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import concurrent.futures as cf, multiprocessing as mp

from envs.core_chain import IndustrialChain
from envs.env import MultiplexEnv
from utils.curriculum import CurriculumManager
from utils.utils import RunningMeanStd

from models.ppo_model import PPOIndustrialModel
from algs.ppo import PPO
from agents.agent import IndustrialAgent


def run_once(exp_dir, log_dir):
    # ========================== 读取配置 ==========================
    ppo_config_path = '../configs/ppo_config.json'

    env_config_path = os.path.join(exp_dir, "env_config.json")
    train_sched_path = os.path.join(exp_dir, "train_schedule.json")
    eval_sched_path = os.path.join(exp_dir, "eval_schedule.json")
    worker_cfg_path = os.path.join(exp_dir, "worker_config.json")
    with open(env_config_path, 'r') as f:
        env_cfg = json.load(f)
    with open(ppo_config_path, 'r') as f:
        ppo_cfg = json.load(f)

    num_layers = env_cfg["num_layers"]

    # —— 课程难度阶梯（Poisson λ） ——
    lambda_levels = [1, 1.5, 2.5, 3.5]
    cm = CurriculumManager(lambda_levels,
                           burn_in=1000,
                           worst_buf_size=10)

    # ========================== 环境初始化 =========================
    env = MultiplexEnv(env_config_path, schedule_load_path=train_sched_path,
                       worker_config_load_path=worker_cfg_path)
    eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_sched_path)
    # 保持两环境 worker一致
    eval_env.worker_config = env.worker_config
    eval_env.chain = IndustrialChain(eval_env.worker_config)

    max_steps = env_cfg["max_steps"]
    device = ppo_cfg["device"]

    agents, buffers = {}, {}
    return_rms = {lid: RunningMeanStd() for lid in range(num_layers)}

    # —— TensorBoard ——
    writer = SummaryWriter(log_dir)

    for lid in range(num_layers):
        obs_space = env.observation_space[lid]
        act_space = env.action_space[lid]
        n_worker, _ = act_space.shape
        task_dim = obs_space['task_queue'].shape[1]
        load_dim = obs_space['worker_loads'].shape[1]
        n_types = len(env_cfg["task_types"])
        profile_dim, gctx_dim = 2 * n_types, 1
        model = PPOIndustrialModel(task_dim, load_dim, profile_dim,
                                   n_worker, env.num_pad_tasks,
                                   gctx_dim, ppo_cfg["hidden_dim"])
        alg = PPO(model,
                  clip_param=ppo_cfg["clip_param"],
                  value_loss_coef=ppo_cfg["value_loss_coef"],
                  entropy_coef=ppo_cfg["entropy_coef"],
                  initial_lr=ppo_cfg["initial_lr"],
                  max_grad_norm=ppo_cfg["max_grad_norm"],
                  writer=writer, global_step_ref=[0],
                  total_training_steps=ppo_cfg["num_episodes"] * max_steps)
        agents[lid] = IndustrialAgent(alg, "ppo", device, env.num_pad_tasks)
        # 简化 buffer 结构
        buffers[lid] = {k: [] for k in
                        ['task_obs', 'worker_loads', 'worker_profile', 'global_context',
                         'valid_mask', 'actions', 'logprobs', 'rewards', 'dones', 'values']}

    # ========================== 工具函数 ==========================
    def process_obs(raw, lid):
        lo = raw[lid]
        return lo['task_queue'], lo['worker_loads'], lo['worker_profile'], raw['global_context']

    def compute_gae(rews, dones, vals, gamma, lam):
        advs, gae = [], 0
        vals = vals + [0]
        for t in reversed(range(len(rews))):
            delta = rews[t] + gamma * vals[t + 1] * (1 - dones[t]) - vals[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advs.insert(0, gae)
        returns = [a + v for a, v in zip(advs, vals[:-1])]
        return advs, returns

    def evaluate_policy(agents, eval_env, eval_episodes, writer, global_step):
        reward_sums = {lid: [] for lid in agents}
        assign_bonus_sums = {lid: [] for lid in agents}
        wait_penalty_sums = {lid: [] for lid in agents}
        cost_sums = {lid: [] for lid in agents}
        util_sums = {lid: [] for lid in agents}

        for episode in range(eval_episodes):
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
                    value, action, logprob, _ = agents[lid].sample(task_obs, worker_loads, worker_profile,
                                                                   global_context)
                    actions[lid] = action

                obs, (total_reward, reward_detail), done, _ = eval_env.step(actions)

                for lid in agents:
                    r_info = reward_detail["layer_rewards"][lid]
                    episode_reward[lid] += r_info["reward"]
                    episode_assign_bonus[lid] += r_info["assign_bonus"]
                    episode_wait_penalty[lid] += r_info["wait_penalty"]
                    episode_cost[lid] += r_info["cost"]
                    episode_util[lid] += r_info["utility"]

            for lid in agents:
                reward_sums[lid].append(episode_reward[lid])
                assign_bonus_sums[lid].append(episode_assign_bonus[lid])
                wait_penalty_sums[lid].append(episode_wait_penalty[lid])
                cost_sums[lid].append(episode_cost[lid])
                util_sums[lid].append(episode_util[lid])

            # 统计任务状态
            n_total, n_wait, n_done, n_fail = 0, 0, 0, 0
            for step_task_list in eval_env.task_schedule.values():
                for task in step_task_list:
                    n_total += 1
                    if task.status == "waiting":
                        n_wait += 1
                    elif task.status == "done":
                        n_done += 1
                    elif task.status == "failed":
                        n_fail += 1
            print(
                f"[Eval Episode {episode}] Total tasks: {n_total}, Waiting: {n_wait}, Done: {n_done}, Failed: {n_fail}")

        # ===== TensorBoard log =====
        total_reward_all = sum(np.mean(reward_sums[lid]) for lid in agents)
        total_cost_all = sum(np.mean(cost_sums[lid]) for lid in agents)
        total_util_all = sum(np.mean(util_sums[lid]) for lid in agents)

        log_interval = ppo_cfg["log_interval"]  # 引用配置字典
        if global_step % log_interval == 0:
            for lid in agents:
                writer.add_scalar(f"eval/layer_{lid}_avg_reward", np.mean(reward_sums[lid]), global_step)
                writer.add_scalar(f"eval/layer_{lid}_avg_assign_bonus", np.mean(assign_bonus_sums[lid]), global_step)
                writer.add_scalar(f"eval/layer_{lid}_avg_wait_penalty", np.mean(wait_penalty_sums[lid]), global_step)
                writer.add_scalar(f"eval/layer_{lid}_avg_cost", np.mean(cost_sums[lid]), global_step)
                writer.add_scalar(f"eval/layer_{lid}_avg_utility", np.mean(util_sums[lid]), global_step)
                print(f"[Eval] Layer {lid}: reward={np.mean(reward_sums[lid]):.2f}, "
                      f"cost={np.mean(cost_sums[lid]):.2f}, utility={np.mean(util_sums[lid]):.2f}")

            writer.add_scalar("global/eval_avg_reward", total_reward_all, global_step)
            writer.add_scalar("global/eval_avg_cost", total_cost_all, global_step)
            writer.add_scalar("global/eval_avg_utility", total_util_all, global_step)

        print(f"[Eval Total] reward={total_reward_all:.2f}, cost={total_cost_all:.2f}, utility={total_util_all:.2f}")

    # ========================== 主训练循环 ==========================
    global_step = 0  # step 计数器
    num_episodes = ppo_cfg["num_episodes"]
    gamma, lam = ppo_cfg["gamma"], ppo_cfg["lam"]
    update_epochs, batch_size = ppo_cfg["update_epochs"], ppo_cfg["batch_size"]
    log_intv, eval_intv = ppo_cfg["log_interval"], ppo_cfg["eval_interval"]
    eval_eps = ppo_cfg["eval_episodes"]

    for ep in range(num_episodes):
        # === 1. 获取当前课程难度（Poisson λ） ===
        cur_lambda = cm.sample_level()
        obs = env.reset(with_new_schedule=True, arrival_rate=cur_lambda)

        # === 2. 初始化 Episode 内统计 ===
        ep_reward_layer = {lid: 0.0 for lid in range(num_layers)}
        done = False
        step = 0

        # === 3. 清空 buffer ===
        for lid in range(num_layers):
            buffers[lid] = {k: [] for k in buffers[lid]}  # 保留结构，只清空内容

        while not done and step < max_steps:
            acts = {}

            for lid in range(num_layers):
                task, load, prof, gctx = process_obs(obs, lid)
                val, act, logp, _ = agents[lid].sample(task, load, prof, gctx)

                # === 存入 buffer ===
                buffers[lid]['task_obs'].append(task)
                buffers[lid]['worker_loads'].append(load)
                buffers[lid]['worker_profile'].append(prof)
                buffers[lid]['global_context'].append(gctx)
                buffers[lid]['valid_mask'].append(task[:, 3].astype(np.float32))
                buffers[lid]['actions'].append(act)
                buffers[lid]['logprobs'].append(logp)
                buffers[lid]['values'].append(val)

                acts[lid] = act

            obs, (tot_r, r_detail), done, _ = env.step(acts)

            for lid in range(num_layers):
                r = r_detail["layer_rewards"][lid]["reward"]
                buffers[lid]['rewards'].append(r)
                buffers[lid]['dones'].append(done)
                ep_reward_layer[lid] += r

            global_step += 1
            step += 1

        # === 4. 每层分别进行 GAE + PPO 更新 ===
        current_steps = (ep + 1) * max_steps

        for layer_id in range(num_layers):
            # 4.1. GAE
            advs, rets = compute_gae(
                buffers[layer_id]['rewards'],
                buffers[layer_id]['dones'],
                buffers[layer_id]['values'],
                gamma, lam
            )

            # 4.2. 可选 return normalization
            if ppo_cfg["return_normalization"]:
                rets_np = np.array(rets)
                return_rms[layer_id].update(rets_np)
                rets = return_rms[layer_id].normalize(rets_np)

            # 4.3. 构建训练数据集（手动打包）
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

            # 4.4. Minibatch PPO 更新
            for _ in range(update_epochs):
                random.shuffle(dataset)
                for i in range(0, len(dataset), batch_size):
                    minibatch = dataset[i:i + batch_size]
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
                        current_steps
                    )

            # 4.5. 清空 buffer
            buffers[layer_id] = {k: [] for k in buffers[layer_id]}

        # === 5. 更新课程 ===
        ep_total_return = sum(ep_reward_layer.values())
        cm.update(ep_total_return)

        # === 6. TensorBoard 日志 ===
        writer.add_scalar("train/episode_total_return", ep_total_return, ep)
        writer.add_scalar("curriculum/lambda", cur_lambda, ep)

        for lid in range(num_layers):
            writer.add_scalar(f"train/layer_{lid}_reward", ep_reward_layer[lid], ep)

        # === 7. 控制台输出 ===
        if ep % log_intv == 0:
            print(f"[Ep {ep}] λ={cur_lambda:.1f}, return={ep_total_return:.2f}, "
                  f"layer-rewards: {[round(ep_reward_layer[l], 1) for l in range(num_layers)]}, "
                  f"→ next λ: {cm.level:.1f}")

        # === 8. 周期性评估 ===
        if ep % eval_intv == 0:
            eval_env.reset()
            evaluate_policy(
                agents=agents,
                eval_env=eval_env,
                eval_episodes=eval_eps,
                writer=writer,
                global_step=ep
            )


if __name__ == "__main__":
    CFG_ROOT = "../configs"
    CATS = ["task", "layer", "worker", "step"]       # 四大类实验
    REPEAT = 1                                       # 每份配置重复次数

    need_files = {"env_config.json", "train_schedule.json",
                  "eval_schedule.json", "worker_config.json"}

    n_workers = 12

    print("\n===== RASCL 批量实验开始 =====\n")

    tasks = []
    for cat in CATS:
        cat_dir = os.path.join(CFG_ROOT, cat)
        if not os.path.isdir(cat_dir):
            continue

        for exp_name in sorted(os.listdir(cat_dir)):
            exp_dir = os.path.join(cat_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue
            # 检查必须文件是否齐全
            if not need_files.issubset(set(os.listdir(exp_dir))):
                print(f"⚠️  跳过 {cat}/{exp_name} —— 文件不完整")
                continue

            for k in range(REPEAT):
                log_dir = f"logs/rascl/{cat}/{exp_name}/" + time.strftime("%Y%m%d-%H%M%S")
                tasks.append((exp_dir, log_dir))

    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(run_once, exp_dir, run_idx)
                   for (exp_dir, run_idx) in tasks]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                fut.result()  # 捕获子进程异常
            except Exception as e:
                print(f"❌ 任务出错：{e}")
            print(f"✔️  已完成 {i}/{len(tasks)}")

    print("🎉 全部 RASCL 实验已结束\n")

