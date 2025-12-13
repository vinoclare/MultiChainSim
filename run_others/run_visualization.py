import os
import torch
import numpy as np
import argparse
import json
import time
import random
from math import ceil
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

# ==== t-SNE + 画图 ====
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.crescent_model import CrescentIndustrialModel
from algs.crescent import CRESCENT
from agents.mappo_agent import IndustrialAgent
from utils.utils import RunningMeanStd
from explore.crescent_cluster import CrescentClusterer

# ===== 默认 t-SNE 可视化相关超参 =====
WARMUP_STEPS_FOR_VIZ = 50000    # 先让策略跑到 50k step 再开始记录
VIZ_SAMPLE_STEPS = 10000        # 只记录接下来的 10k step 做 t-SNE
TSNE_DUMP_DIR = "tsne_dump"
FIG_DIR = "figs"
# ====================================

# ===== Load configurations / args =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
parser.add_argument("--alg_name", type=str, default="crescent")
parser.add_argument("--num_workers", type=int, default=1, help="Parallel env workers for sampling")
parser.add_argument("--mode", type=str, default="load", help="save or load configs")
parser.add_argument("--use_dump", type=int, default=0,
                    help="1: load dumped traj and plot only (no training). 0: train, dump and plot.")
parser.add_argument("--dump_path", type=str, default="",
                    help="When --use_dump=1, load this .npz. If empty, use default naming.")
parser.add_argument("--viz_warmup", type=int, default=WARMUP_STEPS_FOR_VIZ,
                    help="warmup steps before collecting viz samples")
parser.add_argument("--viz_len", type=int, default=VIZ_SAMPLE_STEPS,
                    help="how many steps to collect for visualization")
parser.add_argument("--tsne_perplexity", type=int, default=30)
parser.add_argument("--tsne_seed", type=int, default=0)
parser.add_argument("--viz_plot", type=str, default="step1d",
                    choices=["step1d", "tsne2d"],
                    help="step1d: y=step in episode, x=1D embedding(PCA) of macro_feat; tsne2d: original t-SNE scatter.")
parser.add_argument("--max_plot_episodes", type=int, default=100,
                    help="Max number of episodes to plot (avoid clutter).")
parser.add_argument("--plot_stride", type=int, default=10,
                    help="Subsample steps within an episode when plotting (e.g., 5/10/20).")
parser.add_argument("--pca_seed", type=int, default=0)
# ==================================================================

args, _ = parser.parse_known_args()
dire = args.dire
alg_name = args.alg_name.lower()

with open(f'../configs/{dire}/env_config.json') as f:
    env_config = json.load(f)
with open('../configs/ppo_config.json') as f:
    ppo_config = json.load(f)

# ===== Setup environment paths =====
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


def _default_dump_path():
    warm = int(args.viz_warmup)
    ln = int(args.viz_len)
    return os.path.join(TSNE_DUMP_DIR, f"traj_{alg_name}_{dire}_{warm}_{warm + ln}.npz")


def _ensure_dirs():
    os.makedirs(TSNE_DUMP_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


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
    feats = []
    total_valid = 0.0
    total_slots = 0.0

    for lid in range(num_layers):
        layer_obs = obs[lid]
        task_obs = layer_obs['task_queue']
        valid_mask = task_obs[:, 3].astype(np.float32)
        num_valid = float(valid_mask.sum())
        num_slots = float(valid_mask.shape[0])

        backlog_ratio = num_valid / (num_slots + 1e-8)
        feats.append(backlog_ratio)

        total_valid += num_valid
        total_slots += num_slots

        worker_loads = layer_obs['worker_loads']
        wl = worker_loads.reshape(-1).astype(np.float32)
        if wl.size > 0:
            feats.append(float(wl.mean()))
            feats.append(float(wl.max()))
            feats.append(float(wl.std()))
        else:
            feats.extend([0.0, 0.0, 0.0])

    if total_slots > 0.0:
        global_backlog_ratio = total_valid / (total_slots + 1e-8)
    else:
        global_backlog_ratio = 0.0
    feats.append(float(global_backlog_ratio))

    if max_steps > 0:
        time_frac = float(step_idx) / float(max_steps)
    else:
        time_frac = 0.0
    feats.append(time_frac)

    return np.array(feats, dtype=np.float32)


def dump_viz_data(macro_feats, ep_ids, step_ids, dump_path):
    _ensure_dirs()
    macro_feats = np.array(macro_feats, dtype=np.float32)
    ep_ids = np.array(ep_ids, dtype=np.int32)
    step_ids = np.array(step_ids, dtype=np.int32)
    np.savez(dump_path, macro_feats=macro_feats, episode_ids=ep_ids, step_ids=step_ids)
    print(f"[t-SNE] Trajectory dump saved: {dump_path} (N={macro_feats.shape[0]})")


def load_viz_data(dump_path):
    data = np.load(dump_path)
    X = data["macro_feats"].astype(np.float32)
    ep_ids = data["episode_ids"].astype(np.int32)
    step_ids = data["step_ids"].astype(np.int32)
    return X, ep_ids, step_ids


def _traj_to_vec(macro_seq: np.ndarray) -> np.ndarray:
    """
    把一个 episode 的轨迹 (T, F) 压成一个固定维度向量 (D,).
    这样就能对“轨迹”做 t-SNE（每条轨迹=一个样本）。
    """
    # macro_seq: [T, F]
    mean = macro_seq.mean(axis=0)
    std = macro_seq.std(axis=0)
    mn = macro_seq.min(axis=0)
    mx = macro_seq.max(axis=0)
    first = macro_seq[0]
    last = macro_seq[-1]
    # D = 6F
    return np.concatenate([mean, std, mn, mx, first, last], axis=0).astype(np.float32)


def visualize_tsne_from_macro(macro_feats, ep_ids, step_ids, alg_name, dire):
    """
    两种画法：
    1) step1d（默认）：y=step in episode, x=PCA-1D(macro_feat). 轨迹“细/粗”直接表示 intra-episode 的状态多样性。
    2) tsne2d：原来的 t-SNE 2D scatter.
    """
    if len(macro_feats) == 0:
        print("[viz] No macro features collected for visualization.")
        return

    _ensure_dirs()

    X = np.array(macro_feats, dtype=np.float32)                 # [N, F]
    ep_ids = np.array(ep_ids, dtype=np.int32)                   # [N]
    step_ids = np.array(step_ids, dtype=np.int32)               # [N]

    if args.viz_plot == "tsne2d":
        # ===== 原 t-SNE 2D scatter（保留）=====
        print(f"[t-SNE] Running t-SNE on {X.shape[0]} samples ...")
        tsne = TSNE(
            n_components=2,
            perplexity=int(args.tsne_perplexity),
            learning_rate="auto",
            init="random",
            random_state=int(args.tsne_seed),
        )
        Y = tsne.fit_transform(X)

        # 这里的 step_norm 是 episode 内 step（仅作颜色展示）
        step_norm = (step_ids - step_ids.min()) / (step_ids.max() - step_ids.min() + 1e-8)

        plt.figure(figsize=(6, 5))
        plt.scatter(
            Y[:, 0], Y[:, 1],
            c=step_norm,
            cmap="magma_r",
            s=8,
            alpha=0.9,
            linewidths=0
        )
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.title(f"Training Trajectory t-SNE (crescent)")
        plt.colorbar(label="Normalized step index (in-episode)")
        plt.tight_layout()

        out_path = os.path.join(FIG_DIR, f"tsne_traj_{alg_name}_{dire}.png")
        plt.savefig(out_path, dpi=300)
        print(f"[t-SNE] Figure saved to {out_path}")
        return

    print(f"[step1d] Fitting PCA-1D on {X.shape[0]} samples ...")
    pca = PCA(n_components=1, random_state=int(args.pca_seed))
    x1 = pca.fit_transform(X).reshape(-1).astype(np.float32)
    x1 = (x1 - x1.mean()) / (x1.std() + 1e-8)

    # 选一些 episode（避免极端 outlier 统治分位数）
    uniq_eps = np.unique(ep_ids)
    if uniq_eps.size > int(args.max_plot_episodes):
        rng = np.random.RandomState(0)
        uniq_eps = rng.choice(uniq_eps, size=int(args.max_plot_episodes), replace=False)

    # 收集每个 step 的 x 分布
    # 注意：有的 episode 可能提前结束，所以某些 step 样本数会少
    max_step = int(step_ids.max())
    steps = np.arange(0, max_step + 1, dtype=np.int32)

    q05, q25, q50, q75, q95 = [], [], [], [], []
    valid_steps = []

    for s in steps:
        mask = (step_ids == s) & np.isin(ep_ids, uniq_eps)
        xs = x1[mask]
        if xs.size < 5:
            continue
        valid_steps.append(s)
        q05.append(np.percentile(xs, 5))
        q25.append(np.percentile(xs, 25))
        q50.append(np.percentile(xs, 50))
        q75.append(np.percentile(xs, 75))
        q95.append(np.percentile(xs, 95))

    valid_steps = np.array(valid_steps, dtype=np.float32)
    q05 = np.array(q05, dtype=np.float32)
    q25 = np.array(q25, dtype=np.float32)
    q50 = np.array(q50, dtype=np.float32)
    q75 = np.array(q75, dtype=np.float32)
    q95 = np.array(q95, dtype=np.float32)

    plt.figure(figsize=(6, 9))

    # 分位数带：外层更透明，内层更明显
    plt.fill_betweenx(valid_steps, q05, q95, alpha=0.25)
    plt.fill_betweenx(valid_steps, q25, q75, alpha=0.45)

    # 中位数曲线
    plt.plot(q50, valid_steps, linewidth=2.5, alpha=1.0)

    plt.xlabel("1D embedding of macro-state (PCA-1D, z-scored)")
    plt.ylabel("Step in episode")
    plt.title(f"Step-vs-Embedding (quantile bands) ({alg_name}, {dire})")
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, f"step1d_band_{alg_name}_{dire}.png")
    plt.savefig(out_path, dpi=300)
    print(f"[step1d] Figure saved to {out_path}")


def main():
    # ========== 仅画图模式：直接从 dump 读取 ==========
    if int(args.use_dump) == 1:
        dump_path = args.dump_path.strip() or _default_dump_path()
        if not os.path.exists(dump_path):
            raise FileNotFoundError(f"[t-SNE] dump not found: {dump_path}")
        X, ep_ids, step_ids = load_viz_data(dump_path)
        visualize_tsne_from_macro(X, ep_ids, step_ids, alg_name, dire)
        return
    # ===============================================

    mode = args.mode
    if mode == "save":
        env = MultiplexEnv(env_config_path, schedule_save_path=schedule_path,
                           worker_config_save_path=worker_config_path)
    else:
        env = MultiplexEnv(env_config_path, schedule_load_path=schedule_path,
                           worker_config_load_path=worker_config_path)

    # ===== Hyperparameters =====
    num_layers = env_config["num_layers"]
    steps_per_episode = env_config["max_steps"]

    total_need = int(args.viz_warmup) + int(args.viz_len)
    num_episodes = int(ceil(total_need / float(steps_per_episode)))

    update_epochs = ppo_config["update_epochs"]
    gamma = ppo_config["gamma"]
    lam = ppo_config["lam"]
    batch_size = ppo_config["batch_size"]
    hidden_dim = ppo_config["hidden_dim"]
    device = ppo_config["device"]
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
            use_contrastive=(lid == 0),
            train_contrastive=(lid == 0),
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

    repr_dim = getattr(algs[0], "repr_dim", 64)
    clusterer = CrescentClusterer(
        repr_dim=repr_dim,
        num_clusters=64,
        ema_momentum=0.99,
        count_smoothing=0.1,
        intrinsic_coef=0.0,
        device=device,
    )

    pool = mp.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(dire, env_config_path, schedule_path, worker_config_path,
                  alg_name, hidden_dim, macro_feat_dim)
    ) if args.num_workers > 1 else None

    # ===== 训练过程可视化缓存 =====
    viz_macro_feats = []
    viz_ep_ids = []
    viz_step_ids = []

    warmup = int(args.viz_warmup)
    viz_len = int(args.viz_len)
    viz_end = warmup + viz_len

    for episode in range(num_episodes):
        print(f"Episode {episode} / {num_episodes}")
        if args.num_workers == 1:
            if episode % reset_schedule_interval == 0:
                obs = env.reset(with_new_schedule=True)
            else:
                obs = env.reset()

            for step in range(steps_per_episode):
                actions = {}
                for lid in range(num_layers):
                    task_obs = obs[lid]['task_queue']
                    worker_loads = obs[lid]['worker_loads']
                    profile = obs[lid]['worker_profile']
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

                raw_macro = build_struct_macro_feature(
                    obs=obs,
                    num_layers=num_layers,
                    step_idx=step,
                    max_steps=steps_per_episode
                )

                global_step = episode * steps_per_episode + step
                if warmup <= global_step < viz_end:
                    viz_macro_feats.append(raw_macro.copy())
                    viz_ep_ids.append(episode)
                    viz_step_ids.append(step)

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
            policy_states = _snapshot_policy_states(algs)
            with_new_schedule = (episode % reset_schedule_interval == 0)
            seeds = np.random.randint(0, 2 ** 31 - 1, size=args.num_workers).tolist()

            results = pool.starmap(
                _episode_worker,
                [(policy_states, with_new_schedule, int(seeds[wid]), wid)
                 for wid in range(args.num_workers)]
            )

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

        # ========= IR + credit assignment =========
        macro_seq = np.array(buffers[0]['macro_feat'], dtype=np.float32)
        z_seq = algs[0].encode_macro_for_cluster(macro_seq)
        r_int_seq = clusterer.update_and_compute_intrinsic(z_seq)

        T = len(buffers[0]['rewards'])
        assert T == len(r_int_seq), f"IR 长度 {len(r_int_seq)} 和 reward 序列 {T} 不一致"

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

                v_int = algs[lid].int_value(task_t, load_t, prof_t, mask_t)
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

        for lid in range(num_layers):
            assert len(buffers[lid]['int_rewards']) == 0
            w_l = float(weights[lid])
            for t in range(T):
                r_int = float(r_int_seq[t] * w_l)
                buffers[lid]['int_rewards'].append(r_int)
                buffers[lid]['rewards'][t] += r_int

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

            int_rews = buffers[lid]['int_rewards']
            dones = buffers[lid]['dones']
            int_rets = []
            int_gae = 0.0
            for t in reversed(range(len(int_rews))):
                int_gae = int_rews[t] + gamma * int_gae * (1 - dones[t])
                int_rets.insert(0, float(int_gae))
            int_rets = np.array(int_rets, dtype=np.float32)

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
                    int_ret_batch_t = torch.tensor(np.array(int_ret_batch, dtype=np.float32))
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

        for lid in buffers:
            for k in buffers[lid]:
                buffers[lid][k].clear()

        # 提前结束：已经收集够可视化窗口就没必要再训练了（你本来就是为了画图）
        global_progress = (episode + 1) * steps_per_episode
        if global_progress >= viz_end:
            print(f"[INFO] Reached viz_end={viz_end} steps, stop training early for visualization.")
            break

    if pool is not None:
        pool.close()
        pool.join()

    # ===== 训练结束：先 dump，再画图 =====
    dump_path = args.dump_path.strip() or _default_dump_path()
    dump_viz_data(viz_macro_feats, viz_ep_ids, viz_step_ids, dump_path)
    visualize_tsne_from_macro(viz_macro_feats, viz_ep_ids, viz_step_ids, alg_name, dire)


if __name__ == "__main__":
    _ensure_dirs()
    log_dir = f'../logs2/{alg_name}/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    mp.set_start_method("spawn", force=True)
    main()
