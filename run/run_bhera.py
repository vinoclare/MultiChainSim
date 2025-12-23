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

from models.bhera_model import BHERAJointModel
from algs.bhera import BHERA

from utils.utils_bhera import WindowBuffer, build_belief_token
from utils.utils import RunningMeanStd


# ===== Load configurations =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
parser.add_argument("--alg_name", type=str, default="bhera")
parser.add_argument("--num_workers", type=int, default=10, help="Parallel env workers for sampling")

parser.add_argument("--bhera_lambda_q", type=float, default=1.0)
parser.add_argument("--bhera_decoder_coef", type=float, default=1.0)
parser.add_argument("--bhera_decoder_sparse_weight", type=float, default=1.0)
parser.add_argument("--bhera_n_step", type=int, default=5)

parser.add_argument("--bhera_beta_init", type=float, default=0.1)
parser.add_argument("--bhera_kl_capacity", type=float, default=1.0)
parser.add_argument("--bhera_beta_lr", type=float, default=1e-3)
parser.add_argument("--bhera_kl_ema_decay", type=float, default=0.99)

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


def _episode_worker(policy_state, with_new_schedule, seed):
    """
    子进程：加载主进程广播的 policy snapshot，跑 1 条 episode，返回每层的 trajectory。
    """
    import numpy as np, torch, random
    global g_env, g_alg, g_num_layers, g_valid_index, g_action_shape, g_token_dim, g_slow_window_len

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
            token_window = token_buf.get(pad_value=pad_value)  # [1,K,token_dim]

            # sample actions
            actions_bla, logp_bl1, _, vmean_bl1, _, h_fast_new = g_alg.sample_joint(
                obs_raw_bl, a_prev_pool, r_prev, done_prev, token_window, h_fast
            )

        actions_np = actions_bla[0].cpu().numpy().astype(np.float32)     # [L, act_dim]
        logp_np = logp_bl1[0, :, 0].cpu().numpy().astype(np.float32)     # [L]
        v_np = vmean_bl1[0, :, 0].cpu().numpy().astype(np.float32)       # [L]

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
    按 run_varibad.py 风格写 eval：
    - 每条 episode 重置 joint belief 状态
    - 动作使用 mean（deterministic）
    """
    total_reward, total_cost, total_utility, total_wait_penalty = 0, 0, 0, 0

    num_layers = env_config["num_layers"]
    obs_space = eval_env.observation_space[0]
    act_space = eval_env.action_space[0]
    action_shape = act_space.shape

    device = torch.device(ppo_config["device"])

    # token dim / window
    token_dim = int(getattr(alg.model, "obs_embed_dim", ppo_config["hidden_dim"])) + int(np.prod(action_shape)) + 2
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
                valid_mask = task_obs[:, g_valid_index].astype(np.float32)
                obs_raw_list.append(_build_raw_obs_vec(task_obs, worker_loads, profile, valid_mask))

            obs_raw_bl = torch.tensor(np.stack(obs_raw_list, axis=0), dtype=torch.float32, device=device).unsqueeze(0)  # [1,L,obs_dim]

            # encode/couple + token window
            h = alg.model.encode_obs_stack(obs_raw_bl)   # [1,L,D]
            h_cpl = alg.model.couple_layers(h)          # [1,L,D]
            x_pool = h_cpl.mean(dim=1)                  # [1,D]
            token_t = build_belief_token(x_pool, a_prev_pool, r_prev, done_prev)  # [1,token_dim]
            token_buf.append(token_t.detach())
            token_window = token_buf.get(pad_value=0.0)  # [1,K,token_dim]

            # belief slow/fast
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
            a_prev_pool = torch.tensor(np.mean(np.stack(actions_flat, axis=0), axis=0, keepdims=True),
                                       dtype=torch.float32, device=device)
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
    num_episodes = ppo_config["num_episodes"]
    steps_per_episode = env_config["max_steps"]
    update_epochs = ppo_config["update_epochs"]
    gamma = ppo_config["gamma"]
    lam = ppo_config["lam"]
    hidden_dim = ppo_config["hidden_dim"]
    device = ppo_config["device"]
    log_interval = ppo_config["log_interval"]
    eval_interval = max(1, int(ppo_config["eval_interval"] / max(1, args.num_workers)))
    eval_episodes = ppo_config["eval_episodes"]
    reset_schedule_interval = ppo_config["reset_schedule_interval"]

    # ===== BHERA extra hparams（不强依赖 config，有默认值）=====
    obs_embed_dim = hidden_dim
    coupling_hidden = hidden_dim
    belief_in_dim = 128

    slow_window_len = 16
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

    for episode in range(outer_loops):
        with_new_schedule = (episode % reset_schedule_interval == 0)

        policy_state = _snapshot_policy_state(alg)

        if args.num_workers == 1:
            result = _episode_worker(policy_state, with_new_schedule, seed=np.random.randint(0, 2**31 - 1))
            results = [result]
        else:
            seeds = np.random.randint(0, 2 ** 31 - 1, size=args.num_workers).tolist()
            results = pool.starmap(
                _episode_worker,
                [(policy_state, with_new_schedule, int(seeds[wid])) for wid in range(args.num_workers)]
            )

        B = len(results)
        global_step = episode * steps_per_episode * max(1, args.num_workers)

        # 统一序列长度：裁到最短
        lengths = [len(res[0]["rewards"]) for res in results]
        if min(lengths) == 0:
            continue
        T = min(lengths)

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
                    global_step
                )

        if episode % eval_interval == 0:
            evaluate_policy(alg, eval_env, eval_episodes, writer, global_step)

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    log_dir = f"../logs/{alg_name}/{dire}/" + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    mp.set_start_method("spawn", force=True)
    main()
