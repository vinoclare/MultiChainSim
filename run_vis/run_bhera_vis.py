import torch
import numpy as np
import argparse
import json
import time
import random
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from collections import deque
from pathlib import Path

from envs import IndustrialChain
from envs.env import MultiplexEnv

from models.bhera_model import BHERAJointModel
from algs.bhera import BHERA

from utils.utils_bhera import WindowBuffer, build_belief_token
from utils.utils import RunningMeanStd


# ===== Load configurations =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
parser.add_argument("--alg_name", type=str, default="bhera")
parser.add_argument("--num_workers", type=int, default=10, help="Parallel env workers for sampling")

parser.add_argument("--vis_keep_last_k", type=int, default=100, help="Save data of last K training episodes for visualization")

parser.add_argument("--bhera_lambda_q", type=float, default=1.0)
parser.add_argument("--bhera_decoder_coef", type=float, default=1.0)
parser.add_argument("--bhera_decoder_sparse_weight", type=float, default=1.0)
parser.add_argument("--bhera_n_step", type=int, default=5)

parser.add_argument("--bhera_beta_init", type=float, default=0.1)
parser.add_argument("--bhera_kl_capacity", type=float, default=1.0)
parser.add_argument("--bhera_beta_lr", type=float, default=1e-3)
parser.add_argument("--bhera_kl_ema_decay", type=float, default=0.99)
parser.add_argument("--bhera_slow_window_len", type=int, default=15)

args, _ = parser.parse_known_args()

dire = args.dire
alg_name = args.alg_name.lower()

with open(f"../configs/{dire}/env_config.json") as f:
    env_config = json.load(f)
with open("../configs/ppo_config.json") as f:
    ppo_config = json.load(f)

# ===== Setup environment =====
env_config_path = f"../configs/{dire}/env_config.json"
schedule_path = f"../configs/{dire}/train_schedule.json"
eval_schedule_path = f"../configs/{dire}/eval_schedule.json"
worker_config_path = f"../configs/{dire}/worker_config.json"

# ===== 子进程全局（仅采样推理用）=====
g_env = None
g_alg = None
g_num_layers = None
g_obs_dim_raw = None
g_action_dim = None
g_action_shape = None
g_num_pad = None
g_valid_index = 3
g_token_dim = None
g_slow_window_len = None


def _state_dict_to_cpu(sd):
    """把 state_dict 的 tensor 全搬到 cpu，避免主进程用 cuda 时跨进程序列化/加载出问题。"""
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def _snapshot_policy_state(alg_obj):
    """把当前 joint policy 权重做成只含 CPU tensor 的 state_dict，便于跨进程传递。"""
    return _state_dict_to_cpu(alg_obj.model.state_dict())


def _build_raw_obs_vec(task_obs, worker_loads, profile, valid_mask):
    # task_obs: [num_pad, task_dim]
    # worker_loads: [n_worker, load_dim]
    # profile: [profile_dim] 或 [..]（这里直接 flatten）
    # valid_mask: [num_pad]
    return np.concatenate(
        [
            task_obs.reshape(-1).astype(np.float32),
            worker_loads.reshape(-1).astype(np.float32),
            np.array(profile, dtype=np.float32).reshape(-1),
            valid_mask.reshape(-1).astype(np.float32),
        ],
        axis=0
    ).astype(np.float32)


def _episode_worker(policy_state, with_new_schedule, seed, collect_vis):
    """
    子进程：加载主进程广播的 policy snapshot，跑 1 条 episode，返回每层的 trajectory。
    额外：当 collect_vis=True 时，返回 __vis__（用于最后K条episode的可视化数据）。
    """
    import numpy as np, torch, random
    global g_env, g_alg, g_num_layers, g_valid_index, g_action_shape, g_token_dim, g_slow_window_len, g_action_dim

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载主进程广播的策略快照（只读推理）
    g_alg.model.load_state_dict(policy_state)

    device = torch.device("cpu")  # worker 只推理：cpu

    # Joint belief states (per-episode reset)
    h_fast = torch.zeros((1, g_alg.model.fast_hidden), dtype=torch.float32, device=device)
    a_prev_pool = torch.zeros((1, g_action_dim), dtype=torch.float32, device=device)
    r_prev = torch.zeros((1, 1), dtype=torch.float32, device=device)
    done_prev = torch.zeros((1, 1), dtype=torch.float32, device=device)

    token_buf = WindowBuffer(max_len=int(g_slow_window_len))
    pad_value = 0.0

    # buffers：按“序列”存（每层一份）
    buffers_local = {lid: {k: [] for k in [
        "obs_raw",
        "actions_flat", "logp_old", "value_old",
        "rewards", "dones"
    ]} for lid in range(g_num_layers)}

    # ===== vis buffers（仅当 collect_vis=True 才启用）=====
    if collect_vis:
        latent_mu_list = []
        action_pool_list = []
        entropy_list = []
        q_aux1_list = []
        q_aux2_list = []

    obs = g_env.reset(with_new_schedule=with_new_schedule)
    done = False

    while not done:
        # build obs_raw stack: [L, obs_dim]
        obs_raw_list = []
        for lid in range(g_num_layers):
            task_obs = obs[lid]["task_queue"]
            worker_loads = obs[lid]["worker_loads"]
            profile = obs[lid]["worker_profile"]
            valid_mask = task_obs[:, g_valid_index].astype(np.float32)
            raw_obs = _build_raw_obs_vec(task_obs, worker_loads, profile, valid_mask)
            obs_raw_list.append(raw_obs)

        obs_raw_bl = torch.tensor(np.stack(obs_raw_list, axis=0), dtype=torch.float32, device=device).unsqueeze(0)  # [1,L,obs_dim]

        # build token_window for slow inference (we keep our own buffer)
        with torch.no_grad():
            # compute x_pool from current obs
            h = g_alg.model.encode_obs_stack(obs_raw_bl)     # [1,L,D]
            h_cpl = g_alg.model.couple_layers(h)            # [1,L,D]
            x_pool = h_cpl.mean(dim=1)                      # [1,D]
            token_t = build_belief_token(x_pool, a_prev_pool, r_prev, done_prev)  # [1,token_dim]
            token_buf.append(token_t.detach())

            token_window, pad_mask = token_buf.get_with_mask(pad_value=pad_value)

            # ===== 采集 latent(mu) 与 entropy（不影响真实采样随机性）=====
            if collect_vis:
                # fork_rng: 保证额外的 reparameterize 不会改变 sample_joint 的随机性
                with torch.random.fork_rng(devices=[]):
                    # slow belief
                    try:
                        mu_s, logvar_s, _ = g_alg.model.belief_slow(token_window, padding_mask=pad_mask)  # [1,Ds]
                    except TypeError:
                        mu_s, logvar_s, _ = g_alg.model.belief_slow(token_window)  # [1,Ds]
                    # fast belief (do NOT overwrite h_fast used by sample_joint)
                    _, mu_f, logvar_f, _ = g_alg.model.belief_fast_step(token_t, h_fast, done_prev)  # [1,Df]
                    mu_lat = torch.cat([mu_s, mu_f], dim=-1)  # [1, D_lat]
                    latent_mu_list.append(mu_lat.squeeze(0).cpu().numpy().astype(np.float32))

                    # entropy: use z=mu (deterministic, for visualization)
                    z_mu = mu_lat
                    ent_layers = []
                    for lid in range(g_num_layers):
                        mean, std = g_alg.model.forward_actor(lid, h_cpl[:, lid, :], z_mu)  # [1,A], [1,A]
                        std = torch.clamp(std, min=1e-8)
                        ent = (0.5 * (np.log(2.0 * np.pi) + 1.0) + torch.log(std)).sum(dim=-1)  # [1]
                        ent_layers.append(ent.item())
                    entropy_list.append(float(np.mean(ent_layers)))

            # sample actions (真实交互用这个，不改行为)
            actions_bla, logp_bl1, aux3, vmean_bl1, aux5, h_fast_new = g_alg.sample_joint(
                obs_raw_bl, a_prev_pool, r_prev, done_prev, token_window, h_fast, pad_mask
            )

        actions_np = actions_bla[0].cpu().numpy().astype(np.float32)     # [L, act_dim]
        logp_np = logp_bl1[0, :, 0].cpu().numpy().astype(np.float32)     # [L]
        v_np = vmean_bl1[0, :, 0].cpu().numpy().astype(np.float32)       # [L]

        # vis: action pool + q aux
        if collect_vis:
            action_pool_list.append(actions_np.mean(axis=0).astype(np.float32))

            def _scalarize(x):
                if x is None:
                    return float("nan")
                if torch.is_tensor(x):
                    x = x.detach().cpu().numpy()
                x = np.asarray(x)
                if x.size == 0:
                    return float("nan")
                return float(np.mean(x))

            q_aux1_list.append(_scalarize(aux3))
            q_aux2_list.append(_scalarize(aux5))

        # env actions need shape act_space.shape
        actions_env = {lid: actions_np[lid].reshape(g_action_shape) for lid in range(g_num_layers)}

        # 环境推进一步
        next_obs, (_, reward_detail), done, _ = g_env.step(actions_env)

        # 回填 reward / done，并更新 joint prev feedback
        r_global = 0.0
        for lid in range(g_num_layers):
            r = reward_detail["layer_rewards"][lid]["reward"]
            d = float(done)
            r_global += float(r)

            buffers_local[lid]["obs_raw"].append(obs_raw_list[lid])
            buffers_local[lid]["actions_flat"].append(actions_np[lid].reshape(-1))
            buffers_local[lid]["logp_old"].append(float(logp_np[lid]))
            buffers_local[lid]["value_old"].append(float(v_np[lid]))
            buffers_local[lid]["rewards"].append(float(r))
            buffers_local[lid]["dones"].append(d)

        # update prev feedback for next step
        with torch.no_grad():
            h_fast = h_fast_new
            a_prev_pool = torch.tensor(actions_np.mean(axis=0, keepdims=True), dtype=torch.float32, device=device)
            r_prev = torch.tensor([[r_global]], dtype=torch.float32, device=device)
            done_prev = torch.tensor([[float(done)]], dtype=torch.float32, device=device)

        obs = next_obs

    if collect_vis:
        buffers_local["__vis__"] = {
            "latent_mu": np.stack(latent_mu_list, axis=0).astype(np.float32) if len(latent_mu_list) > 0 else np.zeros((0, 0), dtype=np.float32),
            "action_pool": np.stack(action_pool_list, axis=0).astype(np.float32) if len(action_pool_list) > 0 else np.zeros((0, 0), dtype=np.float32),
            "entropy": np.array(entropy_list, dtype=np.float32),
            "q_aux1": np.array(q_aux1_list, dtype=np.float32),
            "q_aux2": np.array(q_aux2_list, dtype=np.float32),
        }

    return buffers_local


def _init_worker(
    dire,
    env_config_path,
    schedule_path,
    worker_config_path,
    # model hparams
    obs_embed_dim,
    coupling_hidden,
    belief_in_dim,
    slow_window_len,
    slow_n_layers,
    slow_n_heads,
    slow_ff_dim,
    fast_hidden,
    z_slow_dim,
    z_fast_dim,
    policy_hidden,
    value_hidden,
    value_ensemble,
    decoder_hidden,
    # alg hparams
    lambda_q,
    decoder_coef,
    decoder_sparse_weight,
    gamma,
    n_step,
    beta_init,
    kl_capacity,
    beta_lr,
    kl_ema_decay,
):
    """
    每个子进程启动时调用一次：创建本进程持久复用的 env / alg（仅推理）
    """
    global g_env, g_alg, g_num_layers, g_obs_dim_raw, g_action_dim, g_action_shape, g_num_pad, g_valid_index
    global g_token_dim, g_slow_window_len

    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)

    g_env = MultiplexEnv(
        env_config_path,
        schedule_load_path=schedule_path,
        worker_config_load_path=worker_config_path
    )
    g_env.chain = IndustrialChain(g_env.worker_config)

    g_num_layers = env_cfg["num_layers"]
    obs_space = g_env.observation_space[0]
    act_space = g_env.action_space[0]
    g_action_shape = act_space.shape

    n_worker, act_dim_per_worker = act_space.shape
    task_dim = obs_space["task_queue"].shape[1]
    load_dim = obs_space["worker_loads"].shape[1]
    profile_shape = obs_space["worker_profile"].shape
    profile_dim_flat = int(np.prod(profile_shape))

    g_num_pad = g_env.num_pad_tasks
    g_action_dim = n_worker * act_dim_per_worker

    # raw obs dim：flatten(task_queue) + flatten(worker_loads) + flatten(profile) + flatten(valid_mask)
    g_obs_dim_raw = g_num_pad * task_dim + n_worker * load_dim + profile_dim_flat + g_num_pad

    # token dim = obs_embed_dim + action_dim + 2
    g_token_dim = int(obs_embed_dim) + int(g_action_dim) + 2
    g_slow_window_len = int(slow_window_len)

    # 子进程只做推理：cpu
    infer_device = "cpu"

    model = BHERAJointModel(
        obs_dim=g_obs_dim_raw,
        action_dim=g_action_dim,
        num_layers=g_num_layers,
        obs_embed_dim=int(obs_embed_dim),
        coupling_hidden=int(coupling_hidden),
        belief_in_dim=int(belief_in_dim),
        slow_window_len=int(slow_window_len),
        slow_n_layers=int(slow_n_layers),
        slow_n_heads=int(slow_n_heads),
        slow_ff_dim=int(slow_ff_dim),
        fast_hidden=int(fast_hidden),
        z_slow_dim=int(z_slow_dim),
        z_fast_dim=int(z_fast_dim),
        policy_hidden=int(policy_hidden),
        value_hidden=int(value_hidden),
        value_ensemble=int(value_ensemble),
        decoder_hidden=int(decoder_hidden),
    )

    g_alg = BHERA(
        model,
        clip_param=ppo_config["clip_param"],
        value_loss_coef=ppo_config["value_loss_coef"],
        entropy_coef=ppo_config["entropy_coef"],
        initial_lr=ppo_config["initial_lr"],
        max_grad_norm=ppo_config["max_grad_norm"],
        device=infer_device,
        lambda_q=float(lambda_q),
        decoder_coef=float(decoder_coef),
        decoder_sparse_weight=float(decoder_sparse_weight),
        gamma=float(gamma),
        n_step=int(n_step),
        beta_init=float(beta_init),
        kl_capacity=float(kl_capacity),
        beta_lr=float(beta_lr),
        kl_ema_decay=float(kl_ema_decay),
        writer=None,
        global_step_ref=[0],
        total_training_steps=1,
    )


def evaluate_policy(alg, eval_env, num_episodes, writer, global_step):
    """
    - 每条 episode 重置 joint belief 状态
    - 动作使用 mean（deterministic）
    """
    total_reward, total_cost, total_utility, total_wait_penalty = 0, 0, 0, 0

    num_layers = env_config["num_layers"]
    act_space = eval_env.action_space[0]
    action_shape = act_space.shape

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # window
    slow_window_len = int(getattr(alg.model, "slow_window_len", int(ppo_config.get("bhera_slow_window", 16))))

    for _ in range(num_episodes):
        obs = eval_env.reset()
        done = False

        h_fast = torch.zeros((1, alg.model.fast_hidden), dtype=torch.float32, device=device)
        a_prev_pool = torch.zeros((1, alg.model.action_dim), dtype=torch.float32, device=device)
        r_prev = torch.zeros((1, 1), dtype=torch.float32, device=device)
        done_prev = torch.zeros((1, 1), dtype=torch.float32, device=device)

        token_buf = WindowBuffer(max_len=slow_window_len)

        while not done:
            # obs_raw_bl
            obs_raw_list = []
            for lid in range(num_layers):
                task_obs = obs[lid]["task_queue"]
                worker_loads = obs[lid]["worker_loads"]
                profile = obs[lid]["worker_profile"]

                # 你原来的逻辑：环境里没有 obs[lid]["valid_mask"]，所以这里自己算
                valid_mask = task_obs[:, g_valid_index].astype(np.float32)
                obs_raw_list.append(_build_raw_obs_vec(task_obs, worker_loads, profile, valid_mask))

            obs_raw_bl = torch.tensor(
                np.stack(obs_raw_list, axis=0),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)  # [1,L,obs_dim]

            # encode/couple + token window
            h = alg.model.encode_obs_stack(obs_raw_bl)   # [1,L,D]
            h_cpl = alg.model.couple_layers(h)          # [1,L,D]
            x_pool = h_cpl.mean(dim=1)                  # [1,D]

            token_t = build_belief_token(x_pool, a_prev_pool, r_prev, done_prev)  # [1,token_dim]
            token_buf.append(token_t.detach())

            # ====== 只改这里：拿 window + padding mask ======
            token_window, pad_mask = token_buf.get_with_mask(pad_value=0.0)  # [1,K,token_dim], [1,K] bool(True=PAD)

            try:
                mu_s, logvar_s, z_s = alg.model.belief_slow(token_window, padding_mask=pad_mask)  # [1,Ds]
            except TypeError:
                mu_s, logvar_s, z_s = alg.model.belief_slow(token_window)  # [1,Ds]

            h_fast, mu_f, logvar_f, z_f = alg.model.belief_fast_step(token_t, h_fast, done_prev)  # [1,Hf], [1,Df]...
            z = torch.cat([z_s, z_f], dim=-1)  # [1,Z]

            # deterministic action = mean
            actions = {}
            actions_flat = []
            for lid in range(num_layers):
                mean, std = alg.model.forward_actor(lid, h_cpl[:, lid, :], z)
                a = mean  # deterministic
                a_np = a[0].detach().cpu().numpy().astype(np.float32)
                actions_flat.append(a_np)
                actions[lid] = a_np.reshape(action_shape)

            obs, (_, reward_detail), done, _ = eval_env.step(actions)

            for lid, layer_stats in reward_detail["layer_rewards"].items():
                total_reward += layer_stats.get("reward", 0)
                total_cost += layer_stats.get("cost", 0)
                total_utility += layer_stats.get("utility", 0)
                total_wait_penalty += layer_stats.get("waiting_penalty", 0)

            # update prev feedback
            r_global = 0.0
            for lid in range(num_layers):
                r_global += float(reward_detail["layer_rewards"][lid]["reward"])

            a_prev_pool = torch.tensor(
                np.mean(np.stack(actions_flat, axis=0), axis=0, keepdims=True),
                dtype=torch.float32,
                device=device
            )
            r_prev = torch.tensor([[r_global]], dtype=torch.float32, device=device)
            done_prev = torch.tensor([[float(done)]], dtype=torch.float32, device=device)

    writer.add_scalar("eval/reward", total_reward / num_episodes, global_step)
    writer.add_scalar("eval/cost", total_cost / num_episodes, global_step)
    writer.add_scalar("eval/utility", total_utility / num_episodes, global_step)
    writer.add_scalar("eval/waiting_penalty", total_wait_penalty / num_episodes, global_step)


def _compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards: [T]
    values:  [T] (old value)
    dones:   [T] 0/1
    return: advs [T], rets [T]
    """
    T = len(rewards)
    advs = np.zeros((T,), dtype=np.float32)
    gae = 0.0

    # 末端 bootstrap 设为 0
    values_ext = np.concatenate([values, np.array([0.0], dtype=np.float32)], axis=0)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        gae = delta + gamma * lam * mask * gae
        advs[t] = gae

    rets = advs + values
    return advs, rets



def _build_sparse_label(T, segments):
    """Build binary label y[t]=1 for sparse segments (t in [start,end))."""
    y = np.zeros((int(T),), dtype=np.int64)
    if segments is None:
        return y
    for seg in segments:
        if seg is None or len(seg) != 2:
            continue
        s, e = int(seg[0]), int(seg[1])
        if e <= s:
            continue
        s = max(0, s)
        e = min(int(T), e)
        if e > s:
            y[s:e] = 1
    return y


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    # clip for numeric stability
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _to_prob(x):
    """Convert an array to probability in [0,1]. If it's already prob-like, just clip; else sigmoid."""
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, np.nan, dtype=np.float32)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if mn >= -1e-3 and mx <= 1.0 + 1e-3:
        return np.clip(x, 0.0, 1.0)
    return _sigmoid(x)


def _compute_f1(y_true, y_pred):
    """Binary F1 for positive class=1."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return float(2.0 * precision * recall / (precision + recall + eps))


def _pca1_project(mu_all, mask_all):
    """
    mu_all:  [E,T,D] (nan padded ok)
    mask_all:[E,T]   1=valid
    Return:
      y_all: [E,T] (nan padded)
      mean_t, var_t: [T]
      pca_mean: [D], pca_pc1: [D]
    """
    E, T, D = mu_all.shape
    X = mu_all.reshape(-1, D)
    m = mask_all.reshape(-1) > 0.5
    Xv = X[m]
    if Xv.shape[0] < 2:
        y_all = np.full((E, T), np.nan, dtype=np.float32)
        mean_t = np.full((T,), np.nan, dtype=np.float32)
        var_t = np.full((T,), np.nan, dtype=np.float32)
        return y_all, mean_t, var_t, np.zeros((D,), dtype=np.float32), np.zeros((D,), dtype=np.float32)

    pca_mean = Xv.mean(axis=0, keepdims=True)
    Xc = Xv - pca_mean
    # SVD for PCA1
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = vt[0]
    pc1 = pc1 / (np.linalg.norm(pc1) + 1e-8)

    y_flat = np.full((E * T,), np.nan, dtype=np.float32)
    X_all_c = (X - pca_mean.squeeze(0)) @ pc1
    y_flat[m] = X_all_c[m].astype(np.float32)
    y_all = y_flat.reshape(E, T)

    mean_t = np.nanmean(y_all, axis=0).astype(np.float32)
    var_t = np.nanvar(y_all, axis=0).astype(np.float32)
    return y_all, mean_t, var_t, pca_mean.squeeze(0).astype(np.float32), pc1.astype(np.float32)


def _export_last_k_vis(vis_deque, T_max, segments, out_dir):
    """Export last-K visualization data to out_dir/vis_lastK.npz"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    E = len(vis_deque)
    if E == 0:
        return None

    # infer dims
    first = vis_deque[0]
    mu0 = first.get("latent_mu", None)
    a0 = first.get("action_pool", None)
    D_lat = int(mu0.shape[1]) if mu0 is not None and mu0.ndim == 2 else 0
    A_dim = int(a0.shape[1]) if a0 is not None and a0.ndim == 2 else 0

    mu_all = np.full((E, T_max, D_lat), np.nan, dtype=np.float32) if D_lat > 0 else None
    mask_all = np.zeros((E, T_max), dtype=np.float32)

    action_pool_all = np.full((E, T_max, A_dim), np.nan, dtype=np.float32) if A_dim > 0 else None
    entropy_all = np.full((E, T_max), np.nan, dtype=np.float32)
    q_aux1_all = np.full((E, T_max), np.nan, dtype=np.float32)
    q_aux2_all = np.full((E, T_max), np.nan, dtype=np.float32)

    for i, vis in enumerate(vis_deque):
        mu = vis.get("latent_mu", None)
        ap = vis.get("action_pool", None)
        ent = vis.get("entropy", None)
        qa1 = vis.get("q_aux1", None)
        qa2 = vis.get("q_aux2", None)

        # length
        Ti = 0
        if mu is not None and getattr(mu, "shape", None) is not None and mu.ndim == 2:
            Ti = max(Ti, mu.shape[0])
        if ap is not None and getattr(ap, "shape", None) is not None and ap.ndim == 2:
            Ti = max(Ti, ap.shape[0])
        if ent is not None and getattr(ent, "shape", None) is not None:
            Ti = max(Ti, len(ent))
        if qa1 is not None and getattr(qa1, "shape", None) is not None:
            Ti = max(Ti, len(qa1))
        if qa2 is not None and getattr(qa2, "shape", None) is not None:
            Ti = max(Ti, len(qa2))

        Ti = int(min(Ti, T_max))
        if Ti <= 0:
            continue
        mask_all[i, :Ti] = 1.0

        if mu_all is not None and mu is not None and mu.ndim == 2:
            mu_all[i, :Ti, :] = mu[:Ti]
        if action_pool_all is not None and ap is not None and ap.ndim == 2:
            action_pool_all[i, :Ti, :] = ap[:Ti]
        if ent is not None:
            ent = np.asarray(ent, dtype=np.float32)
            entropy_all[i, :Ti] = ent[:Ti]
        if qa1 is not None:
            qa1 = np.asarray(qa1, dtype=np.float32)
            q_aux1_all[i, :Ti] = qa1[:Ti]
        if qa2 is not None:
            qa2 = np.asarray(qa2, dtype=np.float32)
            q_aux2_all[i, :Ti] = qa2[:Ti]

    # build q true
    y_t = _build_sparse_label(T_max, segments).astype(np.int64)
    y_all = np.tile(y_t.reshape(1, T_max), (E, 1))

    # choose which aux corresponds to q prediction (by global F1 @ 0.5)
    valid = mask_all > 0.5
    def f1_of_aux(q_aux):
        prob = _to_prob(q_aux)
        pred = (prob >= 0.5).astype(np.int64)
        return _compute_f1(y_all[valid], pred[valid])

    f1_aux1 = f1_of_aux(q_aux1_all)
    f1_aux2 = f1_of_aux(q_aux2_all)

    if f1_aux2 >= f1_aux1:
        q_pred_all = _to_prob(q_aux2_all)
        q_aux_choice = 2
    else:
        q_pred_all = _to_prob(q_aux1_all)
        q_aux_choice = 1

    # action 1D: first component + norm
    if action_pool_all is not None:
        action_1d_first = action_pool_all[:, :, 0].astype(np.float32)
        action_1d_norm = np.linalg.norm(action_pool_all, axis=-1).astype(np.float32)
    else:
        action_1d_first = None
        action_1d_norm = None

    # latent 1D PCA + mean/var
    if mu_all is not None and D_lat > 0:
        latent_1d_all, latent_mean_t, latent_var_t, pca_mean, pca_pc1 = _pca1_project(mu_all, mask_all)
    else:
        latent_1d_all = None
        latent_mean_t = None
        latent_var_t = None
        pca_mean = None
        pca_pc1 = None

    # summary curves (mean/std over episodes)
    def mean_std(arr):
        m = np.nanmean(arr, axis=0).astype(np.float32)
        s = np.sqrt(np.nanvar(arr, axis=0) + 1e-12).astype(np.float32)
        return m, s

    q_mean_t, q_std_t = mean_std(np.where(valid, q_pred_all, np.nan))
    ent_mean_t, ent_std_t = mean_std(np.where(valid, entropy_all, np.nan))

    out_path = out_dir / f"vis_last{E}.npz"
    np.savez_compressed(
        out_path,
        # common
        mask=mask_all,
        sparse_label=y_t.astype(np.int64),
        # latent
        latent_mu=mu_all,
        latent_1d=latent_1d_all,
        latent_1d_mean=latent_mean_t,
        latent_1d_var=latent_var_t,
        latent_pca_mean=pca_mean,
        latent_pca_pc1=pca_pc1,
        # q
        q_aux1=q_aux1_all,
        q_aux2=q_aux2_all,
        q_aux_choice=np.array([q_aux_choice], dtype=np.int64),
        q_f1_aux1=np.array([f1_aux1], dtype=np.float32),
        q_f1_aux2=np.array([f1_aux2], dtype=np.float32),
        q_pred=q_pred_all.astype(np.float32),
        q_mean=q_mean_t,
        q_std=q_std_t,
        # actions
        action_pool=action_pool_all,
        action_1d_first=action_1d_first,
        action_1d_norm=action_1d_norm,
        # entropy
        entropy=entropy_all.astype(np.float32),
        entropy_mean=ent_mean_t,
        entropy_std=ent_std_t,
    )
    # also write a tiny json summary
    summary = {
        "episodes_saved": int(E),
        "T_max": int(T_max),
        "q_aux_choice": int(q_aux_choice),
        "q_f1_aux1": float(f1_aux1),
        "q_f1_aux2": float(f1_aux2),
    }
    (out_dir / f"vis_last{E}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def main():
    mode = "load"
    if mode == "save":
        env = MultiplexEnv(env_config_path, schedule_save_path=schedule_path,
                           worker_config_save_path=worker_config_path)
        eval_env = MultiplexEnv(env_config_path, schedule_save_path=eval_schedule_path)
    else:
        env = MultiplexEnv(env_config_path, schedule_load_path=schedule_path,
                           worker_config_load_path=worker_config_path)
        eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path)

    eval_env.worker_config = env.worker_config
    eval_env.chain = IndustrialChain(eval_env.worker_config)

    # ===== Hyperparameters（沿用 ppo_config）=====
    num_layers = env_config["num_layers"]
    num_episodes = 4000
    steps_per_episode = env_config["max_steps"]
    update_epochs = ppo_config["update_epochs"]
    gamma = ppo_config["gamma"]
    lam = ppo_config["lam"]
    hidden_dim = ppo_config["hidden_dim"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log_interval = ppo_config["log_interval"]
    eval_interval = max(1, int(ppo_config["eval_interval"] / max(1, args.num_workers)))
    eval_episodes = ppo_config["eval_episodes"]
    reset_schedule_interval = ppo_config["reset_schedule_interval"]

    # ===== BHERA extra hparams（不强依赖 config，有默认值）=====
    obs_embed_dim = hidden_dim
    coupling_hidden = hidden_dim
    belief_in_dim = 128

    slow_window_len = args.bhera_slow_window_len
    slow_n_layers = 2
    slow_n_heads = 4
    slow_ff_dim = 256

    fast_hidden = 128
    z_slow_dim = 32
    z_fast_dim = 32

    policy_hidden = 256
    value_hidden = 256
    value_ensemble = 5
    decoder_hidden = 256

    lambda_q = float(args.bhera_lambda_q)
    decoder_coef = float(args.bhera_decoder_coef)
    decoder_sparse_weight = float(args.bhera_decoder_sparse_weight)
    n_step = int(args.bhera_n_step)

    beta_init = float(args.bhera_beta_init)
    kl_capacity = float(args.bhera_kl_capacity)
    beta_lr = float(args.bhera_beta_lr)
    kl_ema_decay = float(args.bhera_kl_ema_decay)

    # ===== Init model/alg (joint) =====
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    n_worker, act_dim_per_worker = act_space.shape
    task_dim = obs_space["task_queue"].shape[1]
    load_dim = obs_space["worker_loads"].shape[1]
    num_pad = env.num_pad_tasks
    profile_shape = obs_space["worker_profile"].shape
    profile_dim_flat = int(np.prod(profile_shape))

    obs_dim_raw = num_pad * task_dim + n_worker * load_dim + profile_dim_flat + num_pad
    action_dim = n_worker * act_dim_per_worker

    model = BHERAJointModel(
        obs_dim=obs_dim_raw,
        action_dim=action_dim,
        num_layers=num_layers,
        obs_embed_dim=obs_embed_dim,
        coupling_hidden=coupling_hidden,
        belief_in_dim=belief_in_dim,
        slow_window_len=slow_window_len,
        slow_n_layers=slow_n_layers,
        slow_n_heads=slow_n_heads,
        slow_ff_dim=slow_ff_dim,
        fast_hidden=fast_hidden,
        z_slow_dim=z_slow_dim,
        z_fast_dim=z_fast_dim,
        policy_hidden=policy_hidden,
        value_hidden=value_hidden,
        value_ensemble=value_ensemble,
        decoder_hidden=decoder_hidden,
    )

    alg = BHERA(
        model,
        clip_param=ppo_config["clip_param"],
        value_loss_coef=ppo_config["value_loss_coef"],
        entropy_coef=ppo_config["entropy_coef"],
        initial_lr=ppo_config["initial_lr"],
        max_grad_norm=ppo_config["max_grad_norm"],
        device=device,
        lambda_q=lambda_q,
        decoder_coef=decoder_coef,
        decoder_sparse_weight=decoder_sparse_weight,
        gamma=gamma,
        n_step=n_step,
        beta_init=beta_init,
        kl_capacity=kl_capacity,
        beta_lr=beta_lr,
        kl_ema_decay=kl_ema_decay,
        writer=writer,
        global_step_ref=[0],
        total_training_steps=ppo_config["num_episodes"] * env_config["max_steps"],
    )

    # return normalization：按 layer 维护（和 run_varibad 一致）
    return_rms = {lid: RunningMeanStd() for lid in range(num_layers)}

    # ===== Worker pool =====
    pool = mp.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(dire, env_config_path, schedule_path, worker_config_path,
                  obs_embed_dim, coupling_hidden, belief_in_dim,
                  slow_window_len, slow_n_layers, slow_n_heads, slow_ff_dim,
                  fast_hidden, z_slow_dim, z_fast_dim,
                  policy_hidden, value_hidden, value_ensemble, decoder_hidden,
                  lambda_q, decoder_coef, decoder_sparse_weight,
                  gamma, n_step, beta_init, kl_capacity, beta_lr, kl_ema_decay)
    ) if args.num_workers > 1 else None

    # ===== Training loop =====
    outer_loops = int(num_episodes / max(1, args.num_workers)) + 1
    vis_deque = deque(maxlen=int(args.vis_keep_last_k))

    for episode in range(outer_loops):
        with_new_schedule = (episode % reset_schedule_interval == 0)

        # collect visualization only near the end (keep last K episodes)
        B_samp = max(1, args.num_workers)
        episodes_done = episode * B_samp
        collect_vis = episodes_done >= max(0, int(num_episodes) - int(args.vis_keep_last_k) - B_samp)

        policy_state = _snapshot_policy_state(alg)

        if args.num_workers == 1:
            result = _episode_worker(policy_state, with_new_schedule, seed=np.random.randint(0, 2**31 - 1), collect_vis=collect_vis)
            results = [result]
        else:
            seeds = np.random.randint(0, 2 ** 31 - 1, size=args.num_workers).tolist()
            results = pool.starmap(
                _episode_worker,
                [(policy_state, with_new_schedule, int(seeds[wid]), collect_vis) for wid in range(args.num_workers)]
            )

        B = len(results)
        if collect_vis:
            for res in results:
                vis = res.get("__vis__", None)
                if vis is not None:
                    vis_deque.append(vis)

        global_step = episode * steps_per_episode * max(1, args.num_workers)

        # 统一序列长度：裁到最短
        lengths = [len(res[0]["rewards"]) for res in results]
        if min(lengths) == 0:
            continue
        T = min(lengths)

        segments = env_config.get("sparse_reward_segments", [])
        q_t = np.zeros((T,), dtype=np.float32)
        for seg in segments:
            if seg is None or len(seg) != 2:
                continue
            s, e = int(seg[0]), int(seg[1])
            s = max(0, s)
            e = min(T, e)
            if e > s:
                q_t[s:e] = 1.0  # 这里按 [start, end) 处理
        q_labels = np.tile(q_t.reshape(1, T, 1), (B, 1, 1))  # [B,T,1]

        # pack joint tensors [B,T,L,*]
        obs_raw = np.zeros((B, T, num_layers, obs_dim_raw), dtype=np.float32)
        actions_flat = np.zeros((B, T, num_layers, action_dim), dtype=np.float32)
        logp_old = np.zeros((B, T, num_layers, 1), dtype=np.float32)
        value_old = np.zeros((B, T, num_layers, 1), dtype=np.float32)
        rewards = np.zeros((B, T, num_layers, 1), dtype=np.float32)
        dones = np.zeros((B, T, num_layers, 1), dtype=np.float32)

        for b in range(B):
            for lid in range(num_layers):
                obs_raw[b, :, lid, :] = np.stack(results[b][lid]["obs_raw"][:T], axis=0)
                actions_flat[b, :, lid, :] = np.stack(results[b][lid]["actions_flat"][:T], axis=0)
                logp_old[b, :, lid, 0] = np.array(results[b][lid]["logp_old"][:T], dtype=np.float32)
                value_old[b, :, lid, 0] = np.array(results[b][lid]["value_old"][:T], dtype=np.float32)
                rewards[b, :, lid, 0] = np.array(results[b][lid]["rewards"][:T], dtype=np.float32)
                dones[b, :, lid, 0] = np.array(results[b][lid]["dones"][:T], dtype=np.float32)

        # GAE per episode per layer
        advs = np.zeros((B, T, num_layers, 1), dtype=np.float32)
        rets = np.zeros((B, T, num_layers, 1), dtype=np.float32)

        for lid in range(num_layers):
            for b in range(B):
                adv_b, ret_b = _compute_gae(
                    rewards[b, :, lid, 0],
                    value_old[b, :, lid, 0],
                    dones[b, :, lid, 0],
                    gamma, lam
                )
                advs[b, :, lid, 0] = adv_b
                rets[b, :, lid, 0] = ret_b

            if ppo_config.get("return_normalization", False):
                return_rms[lid].update(rets[:, :, lid, 0].reshape(-1))
                rets[:, :, lid, 0] = return_rms[lid].normalize(rets[:, :, lid, 0])

        # episode minibatch（RNN/序列友好）
        ep_batch_size = int(ppo_config.get("bhera_ep_batch_size", B))
        ep_batch_size = max(1, min(ep_batch_size, B))

        idx = np.arange(B)
        for _ in range(update_epochs):
            np.random.shuffle(idx)
            for start in range(0, B, ep_batch_size):
                mb = idx[start:start + ep_batch_size]

                info = alg.learn_joint(
                    torch.tensor(obs_raw[mb], dtype=torch.float32),
                    torch.tensor(actions_flat[mb], dtype=torch.float32),
                    torch.tensor(value_old[mb], dtype=torch.float32),
                    torch.tensor(rets[mb], dtype=torch.float32),
                    torch.tensor(logp_old[mb], dtype=torch.float32),
                    torch.tensor(advs[mb], dtype=torch.float32),
                    torch.tensor(rewards[mb], dtype=torch.float32),
                    torch.tensor(dones[mb], dtype=torch.float32),
                    torch.tensor(q_labels[mb], dtype=torch.float32),
                    global_step
                )

        if episode % eval_interval == 0:
            evaluate_policy(alg, eval_env, eval_episodes, writer, global_step)


    # ===== Export visualization data (last K training episodes) =====
    try:
        _export_last_k_vis(
            vis_deque,
            T_max=int(steps_per_episode),
            segments=env_config.get("sparse_reward_segments", []),
            out_dir=Path(".")
        )
    except Exception as e:
        print(f"[WARN] export vis failed: {e}")

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    log_dir = f"../logs/{alg_name}/{dire}/" + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    mp.set_start_method("spawn", force=True)
    main()
