import torch
import numpy as np
import argparse
import json
import time
import random
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.mappo_model import MAPPOIndustrialModel
from algs.crescent import CRESCENT
from agents.mappo_agent import IndustrialAgent
from utils.utils import RunningMeanStd
# NEW: 结构聚类 + 内在奖励
from explore.crescent_cluster import CrescentClusterer


# ===== Load configurations =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
parser.add_argument("--num_workers", type=int, default=10, help="Parallel env workers for sampling")

args, _ = parser.parse_known_args()
dire = args.dire
alg_name = 'crescent'

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


def _snapshot_policy_states(algs_dict):
    """把每层当前策略权重做成只含张量的 state_dict，便于跨进程传递。"""
    return {lid: algs_dict[lid].model.state_dict() for lid in algs_dict}


def _episode_worker(policy_states, with_new_schedule, seed):
    global g_env, g_agents, g_algs, g_num_layers, g_max_steps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载主进程广播的策略快照（只读）
    for lid in range(g_num_layers):
        g_algs[lid].model.load_state_dict(policy_states[lid])

    # 采样一条 episode
    buffers_local = {lid: {k: [] for k in [
        'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
        'actions', 'logprobs', 'rewards', 'dones', 'values', 'macro_feat']} for lid in range(g_num_layers)
    }

    obs = g_env.reset(with_new_schedule=with_new_schedule)
    done = False
    prev_layer_tasks_done = np.zeros(g_num_layers, dtype=np.float32)
    step_idx = 0
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

        macro_feat, curr_done = build_macro_feature(g_env, prev_layer_tasks_done)
        prev_layer_tasks_done = curr_done
        for lid in range(g_num_layers):
            buffers_local[lid]['macro_feat'].append(macro_feat)

        obs, (_, reward_detail), done, _ = g_env.step(actions)
        for lid in range(g_num_layers):
            r = reward_detail['layer_rewards'][lid]['reward']
            buffers_local[lid]['rewards'].append(r)
            buffers_local[lid]['dones'].append(done)

        step_idx += 1
    return buffers_local


def _init_worker(dire, env_config_path, schedule_path, worker_config_path, alg_name, hidden_dim):
    """每个子进程启动时调用一次：创建本进程持久复用的 env / agents（仅推理）"""
    global g_env, g_agents, g_algs, g_num_layers, g_obs_space, g_profile_dim, g_n_worker, g_num_pad, g_max_steps

    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)

    # 每个子进程只创建一次环境，以后每个 episode 复用
    g_env = MultiplexEnv(env_config_path,
                         schedule_load_path=schedule_path,
                         worker_config_load_path=worker_config_path)
    g_env.chain = IndustrialChain(g_env.worker_config)

    g_num_layers = env_cfg["num_layers"]
    obs_space = g_env.observation_space[0]
    act_space = g_env.action_space[0]
    g_n_worker, _ = act_space.shape
    n_task_types = len(env_cfg["task_types"])
    g_profile_dim = 2 * n_task_types
    g_num_pad = g_env.num_pad_tasks
    g_obs_space = obs_space
    g_max_steps = env_cfg["max_steps"]

    # 预先算出 macro_feat_dim，供子进程里的 CRESCENT 使用
    macro_feat_example, _ = build_macro_feature(g_env, prev_layer_tasks_done=None)
    macro_feat_dim = macro_feat_example.shape[0]

    g_agents, g_algs = {}, {}
    for lid in range(g_num_layers):
        model = MAPPOIndustrialModel(
            task_input_dim=obs_space['task_queue'].shape[1],
            worker_load_input_dim=obs_space['worker_loads'].shape[1],
            worker_profile_input_dim=g_profile_dim,
            n_worker=g_n_worker,
            num_pad_tasks=g_num_pad,
            hidden_dim=hidden_dim
        )
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
            use_contrastive=False  # 采样端只用模型，不训练 encoder
        )

        g_algs[lid] = alg
        g_agents[lid] = IndustrialAgent(alg, alg_name, device="cpu", num_pad_tasks=g_num_pad)


def process_obs(raw_obs, lid):
    obs = raw_obs[lid]
    return obs['task_queue'], obs['worker_loads'], obs['worker_profile']


def build_macro_feature(env, prev_layer_tasks_done=None):
    """
    使用 env 的内部结构(IndustrialChain) 构造当前 step 的宏观结构向量 x_t。

    输入:
      env: MultiplexEnv 实例，内部包含 env.chain, env.config 等
      prev_layer_tasks_done: 长度为 num_layers 的列表/ndarray，表示上一 step
          各层的累计 tasks_done，用于计算本步 wip_l 和 Δ_l。
          - 如果为 None，则认为上一 step 为 0，此时 Δ_l 全为 0。

    返回:
      macro_feat: np.array, shape = [F, ], 当前 step 的宏观结构特征向量
      curr_layer_tasks_done: np.array, shape = [num_layers, ],
          当前 step 各层的累计 tasks_done（方便外部更新 prev_layer_tasks_done）
    """
    chain = env.chain
    config = env.config
    layers = chain.layers
    num_layers = env.num_layers
    task_types = config["task_types"]

    current_step = env.current_step
    max_steps = env.max_steps if env.max_steps > 0 else 1

    feats = []

    # 1) 第一维: 任务–Agent 负载结构 (用负载而不是剩余负载)
    #    (layer l, task_type c) -> [total_load, max_ratio, variance]
    for layer in layers:
        workers = layer.workers
        for t_name in task_types:
            loads_c = np.array(
                [w.current_load_map.get(t_name, 0.0) for w in workers],
                dtype=np.float32
            )  # [n_worker]

            total_load = float(loads_c.sum())
            if total_load > 1e-8:
                p = loads_c / total_load  # 归一化为分布
                max_ratio = float(p.max())
                var = float(((p - p.mean()) ** 2).mean())
            else:
                max_ratio = 0.0
                var = 0.0

            feats.append(total_load)
            feats.append(max_ratio)
            feats.append(var)

    # 2) 第二维: 任务排队结构
    #    (layer l, task_type c) -> [queue_len, remain_unassigned, wait_mean]
    for layer in layers:
        queue = layer.task_queue  # List[Task]

        for t_name in task_types:
            tasks_c = [task for task in queue if task.task_type == t_name]

            queue_len = float(len(tasks_c))

            if queue_len > 0:
                remain_unassigned = float(sum(t.unassigned_amount for t in tasks_c))
                waits = [current_step - t.arrival_time for t in tasks_c]
                wait_mean = float(np.mean(waits)) if len(waits) > 0 else 0.0
            else:
                remain_unassigned = 0.0
                wait_mean = 0.0

            feats.append(queue_len)
            feats.append(remain_unassigned)
            feats.append(wait_mean)

    # 3) 第三维: 跨层流量不平衡 Δ_l
    #    使用 env.chain.cumulative_kpis['per_layer'][l]['tasks_done']
    #    wip_l(t) = tasks_done_l(t) - tasks_done_l(t-1)
    #    Δ_l = wip_{l-1} - wip_l, l=1..L-1; Δ_0 = 0
    kpis = chain.cumulative_kpis
    per_layer = kpis["per_layer"]

    curr_layer_tasks_done = np.array(
        [per_layer[lid]["tasks_done"] for lid in range(num_layers)],
        dtype=np.float32
    )

    if prev_layer_tasks_done is None:
        wip_per_layer = np.zeros(num_layers, dtype=np.float32)
    else:
        prev_layer_tasks_done = np.asarray(prev_layer_tasks_done, dtype=np.float32)
        wip_per_layer = curr_layer_tasks_done - prev_layer_tasks_done

    flow_delta = np.zeros(num_layers, dtype=np.float32)
    for lid in range(1, num_layers):
        flow_delta[lid] = wip_per_layer[lid - 1] - wip_per_layer[lid]
    for lid in range(num_layers):
        feats.append(float(flow_delta[lid]))

    # 4) 时间相位
    time_frac = float(current_step) / float(max_steps)
    feats.append(time_frac)

    macro_feat = np.array(feats, dtype=np.float32)
    return macro_feat, curr_layer_tasks_done


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
    mode = "load"
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

    # NEW: CReSCENT 探索相关超参（先从 config 里读，找不到就用默认）
    warmup_episodes = ppo_config.get("crescent_warmup_episodes", 0)
    num_clusters = ppo_config.get("crescent_num_clusters", 64)
    cluster_ema = ppo_config.get("crescent_cluster_ema", 0.99)
    count_smoothing = ppo_config.get("crescent_count_smoothing", 0.1)
    intrinsic_coef = ppo_config.get("crescent_intrinsic_coef", 0.1)

    # ===== Init per-layer models and agents =====
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    n_worker, _ = act_space.shape
    n_task_types = len(env_config["task_types"])
    profile_dim = 2 * n_task_types

    # NEW: 预先用 env 算出宏观特征维度
    macro_feat_example, _ = build_macro_feature(env, prev_layer_tasks_done=None)
    macro_feat_dim = macro_feat_example.shape[0]

    agents, algs, return_rms, buffers = {}, {}, {}, {}

    for lid in range(num_layers):
        model = MAPPOIndustrialModel(
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
            device=device,
            macro_feat_dim=macro_feat_dim
        )

        agents[lid] = IndustrialAgent(alg, alg_name, device, env.num_pad_tasks)
        algs[lid] = alg
        return_rms[lid] = RunningMeanStd()
        buffers[lid] = {k: [] for k in [
            'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
            'actions', 'logprobs', 'rewards', 'dones', 'values', 'macro_feat']
                        }

    # NEW: 全局 CReSCENT 聚类器（一个就够了，用 layer 0 的 repr_dim）
    clusterer = CrescentClusterer(
        repr_dim=algs[0].repr_dim,
        num_clusters=num_clusters,
        ema_momentum=cluster_ema,
        count_smoothing=count_smoothing,
        intrinsic_coef=intrinsic_coef,
        device=device,
    )

    pool = mp.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(dire, env_config_path, schedule_path, worker_config_path, alg_name, hidden_dim)
    ) if args.num_workers > 1 else None

    # ===== Training loop =====
    for episode in range(int(num_episodes / args.num_workers)):
        if args.num_workers == 1:
            if episode % reset_schedule_interval == 0:
                obs = env.reset(with_new_schedule=True)
            else:
                obs = env.reset()
            prev_layer_tasks_done = np.zeros(env.num_layers, dtype=np.float32)
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

                macro_feat, curr_done = build_macro_feature(env, prev_layer_tasks_done)
                prev_layer_tasks_done = curr_done
                for lid in range(num_layers):
                    buffers[lid]['macro_feat'].append(macro_feat)

                obs, (_, reward_detail), done, _ = env.step(actions)
                for lid in range(num_layers):
                    r = reward_detail['layer_rewards'][lid]['reward']
                    buffers[lid]['rewards'].append(r)
                    buffers[lid]['dones'].append(done)
                if done:
                    break
        else:
            # ——分布式并发采样：N 个进程各跑一条 episode 然后拼回 buffers——
            policy_states = _snapshot_policy_states(algs)
            with_new_schedule = (episode % reset_schedule_interval == 0)

            # 为每个 worker 分配独立随机种子
            seeds = np.random.randint(0, 2 ** 31 - 1, size=args.num_workers).tolist()

            results = pool.starmap(
                _episode_worker,
                [(policy_states, with_new_schedule, int(seeds[wid]))
                 for wid in range(args.num_workers)]
            )

            # 先清空，再把所有 worker 的按步样本拼接进来（逐层、逐键拼）
            for lid in range(num_layers):
                for k in buffers[lid]:
                    buffers[lid][k].clear()

                def _cat_list(key):
                    return sum([res[lid][key] for res in results], [])

                for key in ['task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
                            'actions', 'logprobs', 'values', 'rewards', 'dones', 'macro_feat']:
                    buffers[lid][key].extend(_cat_list(key))

        # ===== 在 GAE 之前：基于结构表示计算内在奖励，并叠加到 reward 上 =====
        # 使用 layer 0 的 macro_feat 序列作为全局结构时间序列
        T_global = len(buffers[0]['rewards'])
        if T_global > 0:
            macro_seq = np.array(buffers[0]['macro_feat'], dtype=np.float32)  # [T, macro_dim]
            # 使用 target encoder 得到结构表示 z_seq
            z_seq = algs[0].encode_macro_for_cluster(macro_seq)  # [T, repr_dim]

            global_step = episode * steps_per_episode * (args.num_workers if args.num_workers > 1 else 1)

            if episode >= warmup_episodes:
                r_int_seq = clusterer.update_and_compute_intrinsic(z_seq, global_step)
            else:
                r_int_seq = np.zeros(T_global, dtype=np.float32)

            # 将同一条结构内在奖励加到各层 reward 上
            for lid in range(num_layers):
                for t in range(len(buffers[lid]['rewards'])):
                    buffers[lid]['rewards'][t] += float(r_int_seq[t])

        # ===== Learn each agent independently =====
        for lid in range(num_layers):
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

            T = len(buffers[lid]['rewards'])
            # episode_id 这里简单全 0，step_id = [0..T-1]
            episode_ids = np.zeros(T, dtype=np.int64)
            step_ids = np.arange(T, dtype=np.int64)

            dataset = list(zip(
                buffers[lid]['task_obs'],
                buffers[lid]['worker_loads'],
                buffers[lid]['worker_profile'],
                buffers[lid]['valid_mask'],
                buffers[lid]['macro_feat'],
                episode_ids,
                step_ids,
                buffers[lid]['actions'],
                buffers[lid]['values'],
                rets,
                buffers[lid]['logprobs'],
                advs
            ))

            for _ in range(update_epochs):
                random.shuffle(dataset)
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i + batch_size]
                    (task_batch,
                     load_batch,
                     prof_batch,
                     mask_batch,
                     macro_batch,
                     ep_batch,
                     step_batch,
                     act_batch,
                     val_batch,
                     ret_batch,
                     logp_batch,
                     adv_batch) = zip(*batch)

                    task_batch = torch.tensor(np.array(task_batch, dtype=np.float32))
                    load_batch = torch.tensor(np.array(load_batch, dtype=np.float32))
                    prof_batch = torch.tensor(np.array(prof_batch, dtype=np.float32))
                    mask_batch = torch.tensor(np.array(mask_batch, dtype=np.float32))
                    macro_batch = torch.tensor(np.array(macro_batch, dtype=np.float32))
                    ep_batch = torch.tensor(np.array(ep_batch, dtype=np.int64))
                    step_batch = torch.tensor(np.array(step_batch, dtype=np.int64))
                    act_batch = torch.tensor(np.array(act_batch, dtype=np.float32))
                    val_batch = torch.tensor(np.array(val_batch, dtype=np.float32))
                    ret_batch = torch.tensor(np.array(ret_batch, dtype=np.float32))
                    logp_batch = torch.tensor(np.array(logp_batch, dtype=np.float32))
                    adv_batch = torch.tensor(np.array(adv_batch, dtype=np.float32))

                    current_steps = episode * steps_per_episode * (args.num_workers if args.num_workers > 1 else 1)

                    algs[lid].learn(
                        task_batch,
                        load_batch,
                        prof_batch,
                        mask_batch,
                        macro_batch,
                        ep_batch,
                        step_batch,
                        act_batch,
                        val_batch,
                        ret_batch,
                        logp_batch,
                        adv_batch,
                        current_steps
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
    log_dir = f'../logs2/{alg_name}/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    mp.set_start_method("spawn", force=True)
    main()
