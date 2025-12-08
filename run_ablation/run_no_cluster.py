import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import time
import random
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.crescent_model import CrescentIndustrialModel
from algs.crescent import CRESCENT
from agents.mappo_agent import IndustrialAgent
from utils.utils import RunningMeanStd

# ===== Load configurations =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
parser.add_argument("--alg_name", type=str, default="crescent")
parser.add_argument("--num_workers", type=int, default=10, help="Parallel env workers for sampling")
parser.add_argument("--mode", type=str, default="load", help="save or load configs")

args, _ = parser.parse_known_args()
dire = args.dire
alg_name = args.alg_name.lower()

with open(f'../configs/{dire}/env_config.json') as f:
    env_config = json.load(f)
with open('../configs/ppo_config.json') as f:
    ppo_config = json.load(f)

# ===== Setup environment =====
env_config_path = f'../configs/{dire}/env_config.json'
schedule_path = f"../configs/{dire}/train_schedule.json"
eval_schedule_path = f"../configs/{dire}/eval_schedule.json"
worker_config_path = f"../configs/{dire}/worker_config.json"

# 子进程中的全局对象（仅采样用）
g_env = None
g_agents = None
g_algs = None
g_num_layers = None
g_obs_space = None
g_profile_dim = None
g_n_worker = None
g_num_pad = None
g_max_steps = None
g_macro_feat_dim = None  # 真实宏观特征维度


def _snapshot_policy_states(algs_dict):
    """把每层当前策略权重做成只含张量的 state_dict，便于跨进程传递。"""
    return {lid: algs_dict[lid].model.state_dict() for lid in algs_dict}


def _episode_worker(policy_states, with_new_schedule, seed, worker_id):
    global g_env, g_agents, g_algs, g_num_layers, g_max_steps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载主进程广播的策略快照（只读）
    for lid in range(g_num_layers):
        g_algs[lid].model.load_state_dict(policy_states[lid])

    # 采样一条 episode
    buffers_local = {
        lid: {k: [] for k in [
            'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
            'actions', 'logprobs',
            'rewards', 'ext_rewards', 'int_rewards',
            'dones', 'values',
            'macro_feat', 'episode_ids', 'step_ids'
        ]}
        for lid in range(g_num_layers)
    }

    obs = g_env.reset(with_new_schedule=with_new_schedule)
    done = False
    step_idx = 0
    # 在当前 outer-episode 内，每个 worker 的 episode_id 可以设成 worker_id
    episode_id = worker_id

    while not done:
        actions = {}
        for lid in range(g_num_layers):
            task_obs = obs[lid]['task_queue']
            worker_loads = obs[lid]['worker_loads']
            profile = obs[lid]['worker_profile']
            value, action, logprob, _ = g_agents[lid].sample(task_obs, worker_loads, profile)
            actions[lid] = action
            valid_mask = task_obs[:, 3].astype(np.float32)

            buffers_local[lid]['task_obs'].append(task_obs)
            buffers_local[lid]['worker_loads'].append(worker_loads)
            buffers_local[lid]['worker_profile'].append(profile)
            buffers_local[lid]['valid_mask'].append(valid_mask)
            buffers_local[lid]['actions'].append(action)
            buffers_local[lid]['logprobs'].append(logprob)
            buffers_local[lid]['values'].append(value)
            buffers_local[lid]['episode_ids'].append(episode_id)
            buffers_local[lid]['step_ids'].append(step_idx)

        # 构造宏观结构特征（真实维度）
        raw_macro = build_struct_macro_feature(
            obs=obs,
            num_layers=g_num_layers,
            step_idx=step_idx,
            max_steps=g_max_steps
        )

        for lid in range(g_num_layers):
            buffers_local[lid]['macro_feat'].append(raw_macro)

        obs, (_, reward_detail), done, _ = g_env.step(actions)
        for lid in range(g_num_layers):
            r = reward_detail['layer_rewards'][lid]['reward']
            buffers_local[lid]['rewards'].append(r)
            buffers_local[lid]['ext_rewards'].append(r)
            buffers_local[lid]['dones'].append(done)

        step_idx += 1

    return buffers_local


def _init_worker(dire, env_config_path, schedule_path, worker_config_path,
                 alg_name, hidden_dim, macro_feat_dim):
    """每个子进程启动时调用一次：创建本进程持久复用的 env / agents（仅推理）"""
    global g_env, g_agents, g_algs, g_num_layers, g_obs_space, g_profile_dim
    global g_n_worker, g_num_pad, g_max_steps, g_macro_feat_dim

    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)

    # 每个子进程只创建一次环境，以后每个 episode 复用
    g_env = MultiplexEnv(env_config_path,
                         schedule_load_path=schedule_path,
                         worker_config_load_path=worker_config_path)
    g_env.chain = IndustrialChain(g_env.worker_config)

    g_num_layers = env_cfg["num_layers"]
    g_max_steps = env_cfg["max_steps"]
    g_macro_feat_dim = macro_feat_dim

    obs_space = g_env.observation_space[0]
    act_space = g_env.action_space[0]
    g_n_worker, _ = act_space.shape
    n_task_types = len(env_cfg["task_types"])
    g_profile_dim = 2 * n_task_types
    g_num_pad = g_env.num_pad_tasks
    g_obs_space = obs_space

    g_agents, g_algs = {}, {}
    for lid in range(g_num_layers):
        model = CrescentIndustrialModel(
            task_input_dim=obs_space['task_queue'].shape[1],
            worker_load_input_dim=obs_space['worker_loads'].shape[1],
            worker_profile_input_dim=g_profile_dim,
            n_worker=g_n_worker,
            num_pad_tasks=g_num_pad,
            hidden_dim=hidden_dim
        )

        # worker 端只负责采样，不训练 encoder / 对比学习
        alg = CRESCENT(
            model,
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.0,
            initial_lr=3e-4,
            max_grad_norm=0.5,
            writer=None,
            global_step_ref=[0],
            total_training_steps=1,
            device="cpu",
            macro_feat_dim=macro_feat_dim,
            use_contrastive=False,
            train_contrastive=False,
        )

        g_algs[lid] = alg
        g_agents[lid] = IndustrialAgent(alg, alg_name, device="cpu", num_pad_tasks=g_num_pad)


def process_obs(raw_obs, lid):
    obs = raw_obs[lid]
    return obs['task_queue'], obs['worker_loads'], obs['worker_profile']


def build_struct_macro_feature(obs, num_layers, step_idx, max_steps):
    """
    用 env/obs 构造一维宏观结构特征向量（长度为 F）。
    """
    feats = []

    total_valid = 0.0
    total_slots = 0.0

    for lid in range(num_layers):
        layer_obs = obs[lid]
        task_obs = layer_obs['task_queue']              # [N_slots, feat_dim]
        valid_mask = task_obs[:, 3].astype(np.float32)  # 第 4 维是“是否有任务”
        num_valid = float(valid_mask.sum())
        num_slots = float(valid_mask.shape[0])

        # 每层 backlog 比例
        backlog_ratio = num_valid / (num_slots + 1e-8)
        feats.append(backlog_ratio)

        total_valid += num_valid
        total_slots += num_slots

        # 工人负载结构：对 worker_loads 做简单统计
        worker_loads = layer_obs['worker_loads']        # [n_worker, k]
        wl = worker_loads.reshape(-1).astype(np.float32)
        if wl.size > 0:
            feats.append(float(wl.mean()))
            feats.append(float(wl.max()))
            feats.append(float(wl.std()))
        else:
            feats.extend([0.0, 0.0, 0.0])

    # 全局 backlog 比例
    if total_slots > 0.0:
        global_backlog_ratio = total_valid / (total_slots + 1e-8)
    else:
        global_backlog_ratio = 0.0
    feats.append(float(global_backlog_ratio))

    # 时间相位：当前 step 在整条 episode 中的位置
    if max_steps > 0:
        time_frac = float(step_idx) / float(max_steps)
    else:
        time_frac = 0.0
    feats.append(time_frac)

    return np.array(feats, dtype=np.float32)  # [F]


def compute_contrastive_intrinsic(z_seq: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """
    使用结构表示 z_seq 构造一个 InfoNCE 风格的逐步对比“损失”，作为全局内在奖励。
    当前消融中不再使用聚类和 pseudo-count，只用 z 本身的对比难度作为 IR 来源。
    """
    T = z_seq.shape[0]
    if T <= 1:
        return np.zeros(T, dtype=np.float32)

    z = torch.tensor(z_seq, dtype=torch.float32)

    # 简单做个标准化和 L2 归一化，避免数值太飘
    z = z - z.mean(dim=0, keepdim=True)
    z = F.normalize(z, p=2, dim=1)

    # 相似度矩阵 [T, T]
    sim = torch.matmul(z, z.t())
    sim = sim / tau

    # 不和自己比，mask 掉对角线
    mask = torch.eye(T, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask, -1e9)

    # 构造正样本下标：t>0 的正样本用 t-1，t=0 的正样本用 1（如果存在）
    labels = torch.arange(T, dtype=torch.long, device=sim.device)
    if T >= 2:
        labels[0] = 1
        labels[1:] = torch.arange(T - 1, dtype=torch.long, device=sim.device)
    else:
        labels[0] = 0

    # InfoNCE 风格的逐步损失（不做 batch mean，保留每个 time step 的 loss）
    loss_per_step = F.cross_entropy(sim, labels, reduction="none")  # [T]

    r_int = loss_per_step.detach().cpu().numpy().astype(np.float32)

    # 做一个简单的标准化和裁剪，避免 reward 尺度失控
    mean = r_int.mean()
    std = r_int.std() + 1e-8
    r_int = (r_int - mean) / std
    r_int = np.clip(r_int, -3.0, 3.0)

    # shift 到非负区间，方便和外在奖励叠加
    r_int = r_int - r_int.min()

    return r_int.astype(np.float32)


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


def main():
    mode = args.mode
    if mode == "save":
        env = MultiplexEnv(env_config_path, schedule_save_path=schedule_path,
                           worker_config_save_path=worker_config_path)
        eval_env = MultiplexEnv(env_config_path, schedule_save_path=eval_schedule_path)
    else:  # mode == "load"
        env = MultiplexEnv(env_config_path, schedule_load_path=schedule_path,
                           worker_config_load_path=worker_config_path)
        eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path)

    eval_env.worker_config = env.worker_config
    eval_env.chain = IndustrialChain(eval_env.worker_config)

    # ===== Hyperparameters =====
    num_layers = env_config["num_layers"]
    num_episodes = ppo_config["num_episodes"]
    steps_per_episode = env_config["max_steps"]
    update_epochs = ppo_config["update_epochs"]
    gamma = ppo_config["gamma"]
    lam = ppo_config["lam"]
    batch_size = ppo_config["batch_size"]
    hidden_dim = ppo_config["hidden_dim"]
    device = ppo_config["device"]
    log_interval = ppo_config["log_interval"]
    eval_interval = ppo_config["eval_interval"] / args.num_workers
    eval_episodes = ppo_config["eval_episodes"]
    reset_schedule_interval = ppo_config["reset_schedule_interval"]

    # 预先计算真实宏观特征维度
    probe_obs = env.reset(with_new_schedule=True)
    raw_macro_probe = build_struct_macro_feature(
        obs=probe_obs,
        num_layers=num_layers,
        step_idx=0,
        max_steps=steps_per_episode
    )
    macro_feat_dim = int(raw_macro_probe.shape[0])

    # ===== Init per-layer models and agents =====
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    n_worker, _ = act_space.shape
    n_task_types = len(env_config["task_types"])
    profile_dim = 2 * n_task_types

    agents, algs, return_rms, buffers = {}, {}, {}, {}

    for lid in range(num_layers):
        model = CrescentIndustrialModel(
            task_input_dim=obs_space['task_queue'].shape[1],
            worker_load_input_dim=obs_space['worker_loads'].shape[1],
            worker_profile_input_dim=profile_dim,
            n_worker=n_worker,
            num_pad_tasks=env.num_pad_tasks,
            hidden_dim=hidden_dim
        )

        alg = CRESCENT(
            model,
            clip_param=ppo_config["clip_param"],
            value_loss_coef=ppo_config["value_loss_coef"],
            entropy_coef=ppo_config["entropy_coef"],
            initial_lr=ppo_config["initial_lr"],
            max_grad_norm=ppo_config["max_grad_norm"],
            writer=writer,
            global_step_ref=[0],
            total_training_steps=ppo_config["num_episodes"] * env_config["max_steps"],
            macro_feat_dim=macro_feat_dim,
            use_contrastive=True,
            train_contrastive=True,
        )

        agents[lid] = IndustrialAgent(alg, alg_name, device, env.num_pad_tasks)
        algs[lid] = alg
        return_rms[lid] = RunningMeanStd()
        buffers[lid] = {k: [] for k in [
            'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
            'actions', 'logprobs',
            'rewards', 'ext_rewards', 'int_rewards',
            'dones', 'values',
            'macro_feat', 'episode_ids', 'step_ids'
        ]}

    # 不再使用结构聚类器，这个消融里 IR 改为对比“损失感”信号

    pool = mp.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(dire, env_config_path, schedule_path, worker_config_path,
                  alg_name, hidden_dim, macro_feat_dim)
    ) if args.num_workers > 1 else None

    # ===== Training loop =====
    for episode in range(int(num_episodes / args.num_workers)):
        if args.num_workers == 1:
            if episode % reset_schedule_interval == 0:
                obs = env.reset(with_new_schedule=True)
            else:
                obs = env.reset()

            # 单进程：一轮 outer-episode 里只有 1 条轨迹，episode_id 可以直接用 episode
            for step in range(steps_per_episode):
                actions = {}
                for lid in range(num_layers):
                    task_obs, worker_loads, profile = process_obs(obs, lid)
                    value, action, logprob, _ = agents[lid].sample(task_obs, worker_loads, profile)
                    actions[lid] = action
                    valid_mask = task_obs[:, 3].astype(np.float32)

                    buffers[lid]['task_obs'].append(task_obs)
                    buffers[lid]['worker_loads'].append(worker_loads)
                    buffers[lid]['worker_profile'].append(profile)
                    buffers[lid]['valid_mask'].append(valid_mask)
                    buffers[lid]['actions'].append(action)
                    buffers[lid]['logprobs'].append(logprob)
                    buffers[lid]['values'].append(value)
                    buffers[lid]['episode_ids'].append(episode)
                    buffers[lid]['step_ids'].append(step)

                # 构造宏观结构特征（真实维度）
                raw_macro = build_struct_macro_feature(
                    obs=obs,
                    num_layers=num_layers,
                    step_idx=step,
                    max_steps=steps_per_episode
                )

                for lid in range(num_layers):
                    buffers[lid]['macro_feat'].append(raw_macro)

                obs, (_, reward_detail), done, _ = env.step(actions)
                for lid in range(num_layers):
                    r = reward_detail['layer_rewards'][lid]['reward']
                    buffers[lid]['rewards'].append(r)
                    buffers[lid]['ext_rewards'].append(r)
                    buffers[lid]['dones'].append(done)
                if done:
                    break
        else:
            # 多进程并行采样：每个 worker 各跑一条 episode，然后合并
            policy_states = _snapshot_policy_states(algs)
            with_new_schedule = (episode % reset_schedule_interval == 0)

            seeds = np.random.randint(0, 2 ** 31 - 1, size=args.num_workers).tolist()

            results = pool.starmap(
                _episode_worker,
                [(policy_states, with_new_schedule, int(seeds[wid]), wid)
                 for wid in range(args.num_workers)]
            )

            # 汇总子进程采样结果
            for lid in range(num_layers):
                for k in buffers[lid]:
                    buffers[lid][k].clear()

                def _cat_list(key):
                    return sum([res[lid][key] for res in results], [])

                for key in ['task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
                            'actions', 'logprobs', 'values',
                            'rewards', 'ext_rewards', 'int_rewards',
                            'dones', 'macro_feat', 'episode_ids', 'step_ids']:
                    buffers[lid][key].extend(_cat_list(key))

        # ========= 对比损失 IR + Per-Layer IR 跨层 credit assignment =========
        macro_seq = np.array(buffers[0]['macro_feat'], dtype=np.float32)  # [T, macro_feat_dim]
        z_seq = algs[0].encode_macro_for_cluster(macro_seq)               # [T, repr_dim]

        # global IR：直接用 z_seq 的对比损失构造，不再做聚类
        r_int_seq = compute_contrastive_intrinsic(z_seq)  # [T]

        T = len(buffers[0]['rewards'])
        assert T == len(r_int_seq), f"IR 长度 {len(r_int_seq)} 和 reward 序列 {T} 不一致"

        # 1）基于内在 value 头计算每层的重要性 I_l
        layer_importances = []
        with torch.no_grad():
            for lid in range(num_layers):
                task_arr = np.array(buffers[lid]['task_obs'], dtype=np.float32)
                load_arr = np.array(buffers[lid]['worker_loads'], dtype=np.float32)
                prof_arr = np.array(buffers[lid]['worker_profile'], dtype=np.float32)
                mask_arr = np.array(buffers[lid]['valid_mask'], dtype=np.float32)
                done_arr = np.array(buffers[lid]['dones'], dtype=np.float32)

                task_t = torch.tensor(task_arr)
                load_t = torch.tensor(load_arr)
                prof_t = torch.tensor(prof_arr)
                mask_t = torch.tensor(mask_arr)

                v_int = algs[lid].int_value(task_t, load_t, prof_t, mask_t)  # [T]
                v_int = v_int.detach().cpu().numpy().astype(np.float32)

                deltas = []
                for t in range(T):
                    v_now = v_int[t]
                    v_next = v_int[t + 1] if t < T - 1 else 0.0
                    done_flag = done_arr[t]
                    delta = r_int_seq[t] + gamma * v_next * (1.0 - done_flag) - v_now
                    deltas.append(delta)
                deltas = np.array(deltas, dtype=np.float32)

                I_l = float(np.mean(np.abs(deltas)))
                layer_importances.append(I_l)

        layer_importances = np.array(layer_importances, dtype=np.float32)
        layer_importances = np.nan_to_num(layer_importances, nan=0.0, posinf=0.0, neginf=0.0)

        if layer_importances.sum() <= 1e-8:
            weights = np.ones(num_layers, dtype=np.float32) / float(num_layers)
        else:
            weights = layer_importances / (layer_importances.sum() + 1e-8)

        # 2）按权重分配 global IR，得到 Per-Layer IR
        for lid in range(num_layers):
            assert len(buffers[lid]['int_rewards']) == 0
            w_l = float(weights[lid])
            for t in range(T):
                r_int = float(r_int_seq[t] * w_l)
                buffers[lid]['int_rewards'].append(r_int)
                buffers[lid]['rewards'][t] += r_int

        # ===== Learn each agent independently =====
        for lid in range(num_layers):
            # GAE（多智能体 advantage）
            advs = []
            vals = buffers[lid]['values'] + [0.0]
            gae = 0.0
            for t in reversed(range(len(buffers[lid]['rewards']))):
                delta = (buffers[lid]['rewards'][t] +
                         gamma * vals[t + 1] * (1 - buffers[lid]['dones'][t]) -
                         vals[t])
                gae = delta + gamma * lam * (1 - buffers[lid]['dones'][t]) * gae
                advs.insert(0, gae.copy())
            rets = [a + v for a, v in zip(advs, buffers[lid]['values'])]

            advs = np.array(advs, dtype=np.float32)
            rets = np.array(rets, dtype=np.float32)
            advs = np.repeat(advs[:, None], n_worker, axis=1)
            advs = advs.reshape(-1, n_worker)
            rets = rets.reshape(-1)

            if ppo_config["return_normalization"]:
                return_rms[lid].update(np.array(rets))
                rets = return_rms[lid].normalize(np.array(rets))

            # 内在奖励折扣回报（用于训练内在价值头）
            int_rews = buffers[lid]['int_rewards']
            dones = buffers[lid]['dones']
            assert len(int_rews) == len(dones) == len(buffers[lid]['rewards'])
            int_rets = []
            int_gae = 0.0
            for t in reversed(range(len(int_rews))):
                int_gae = int_rews[t] + gamma * int_gae * (1 - dones[t])
                int_rets.insert(0, float(int_gae))
            int_rets = np.array(int_rets, dtype=np.float32)

            # CRESCENT：喂入 macro_feat + episode_ids + step_ids + int_returns
            dataset = list(zip(
                buffers[lid]['task_obs'],
                buffers[lid]['worker_loads'],
                buffers[lid]['worker_profile'],
                buffers[lid]['valid_mask'],
                buffers[lid]['macro_feat'],
                buffers[lid]['episode_ids'],
                buffers[lid]['step_ids'],
                buffers[lid]['actions'],
                buffers[lid]['values'],
                rets,
                int_rets,
                buffers[lid]['logprobs'],
                advs
            ))

            for _ in range(update_epochs):
                random.shuffle(dataset)
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i + batch_size]
                    (task_batch, load_batch, prof_batch, mask_batch,
                     macro_batch, ep_id_batch, step_id_batch,
                     act_batch, val_batch, ret_batch,
                     int_ret_batch,
                     logp_batch, adv_batch) = zip(*batch)

                    task_batch_t = torch.tensor(np.array(task_batch, dtype=np.float32))
                    load_batch_t = torch.tensor(np.array(load_batch, dtype=np.float32))
                    prof_batch_t = torch.tensor(np.array(prof_batch, dtype=np.float32))
                    mask_batch_t = torch.tensor(np.array(mask_batch, dtype=np.float32))
                    macro_batch_t = torch.tensor(np.array(macro_batch, dtype=np.float32))
                    act_batch_t = torch.tensor(np.array(act_batch, dtype=np.float32))
                    val_batch_t = torch.tensor(np.array(val_batch, dtype=np.float32))
                    ret_batch_t = torch.tensor(np.array(ret_batch, dtype=np.float32))
                    int_ret_batch_t = torch.tensor(
                        np.array(int_ret_batch, dtype=np.float32)
                    )
                    logp_batch_t = torch.tensor(np.array(logp_batch, dtype=np.float32))
                    adv_batch_t = torch.tensor(np.array(adv_batch, dtype=np.float32))

                    ep_id_batch_np = np.array(ep_id_batch, dtype=np.int64)
                    step_id_batch_np = np.array(step_id_batch, dtype=np.int64)

                    current_steps = episode * steps_per_episode * (
                        args.num_workers if args.num_workers > 1 else 1
                    )

                    algs[lid].learn(
                        task_batch_t,
                        load_batch_t,
                        prof_batch_t,
                        mask_batch_t,
                        macro_batch_t,
                        ep_id_batch_np,
                        step_id_batch_np,
                        act_batch_t,
                        val_batch_t,
                        ret_batch_t,
                        logp_batch_t,
                        adv_batch_t,
                        current_steps,
                        int_returns=int_ret_batch_t
                    )

        # 清空 buffers 进入下个周期
        for lid in buffers:
            for k in buffers[lid]:
                buffers[lid][k].clear()

        if episode % eval_interval == 0:
            evaluate_policy(agents, eval_env, eval_episodes, writer,
                            episode * steps_per_episode * args.num_workers)

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    log_dir = f'../logs2/ablations/no_cluster/' + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    mp.set_start_method("spawn", force=True)
    main()
