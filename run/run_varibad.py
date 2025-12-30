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

from models.varibad_model import VariBADIndustrialModel
from algs.varibad import VariBAD
from agents.varibad_agent import VariBADIndustrialAgent

from utils.utils import RunningMeanStd


# ===== Load configurations =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
parser.add_argument("--alg_name", type=str, default="varibad")
parser.add_argument("--num_workers", type=int, default=10, help="Parallel env workers for sampling")
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

# ===== 子进程全局（仅采样推理用）=====
g_env = None
g_agents = None
g_algs = None
g_num_layers = None
g_obs_dim_raw = None
g_action_dim = None
g_num_pad = None
g_valid_index = 3


def _state_dict_to_cpu(sd):
    """把 state_dict 的 tensor 全搬到 cpu，避免主进程用 cuda 时跨进程序列化/加载出问题。"""
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def _snapshot_policy_states(algs_dict):
    """把每层当前策略权重做成只含 CPU tensor 的 state_dict，便于跨进程传递。"""
    return {lid: _state_dict_to_cpu(algs_dict[lid].model.state_dict()) for lid in algs_dict}


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


def _episode_worker(policy_states, with_new_schedule, seed):
    """
    子进程：加载主进程广播的 policy snapshot，跑 1 条 episode，返回每层的 trajectory。
    """
    import numpy as np, torch, random
    global g_env, g_agents, g_algs, g_num_layers, g_valid_index

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载主进程广播的策略快照（只读推理）
    for lid in range(g_num_layers):
        g_algs[lid].model.load_state_dict(policy_states[lid])

    # 每条 episode 开始前重置 belief
    for lid in range(g_num_layers):
        g_agents[lid].reset_belief()

    # buffers：按“序列”存
    buffers_local = {lid: {k: [] for k in [
        "obs_raw", "next_obs_raw",
        "actions_flat", "logp_old", "value_old",
        "rewards", "dones"
    ]} for lid in range(g_num_layers)}

    obs = g_env.reset(with_new_schedule=with_new_schedule)
    done = False

    while not done:
        actions_env = {}

        # 先基于当前 obs 采样动作（每层各自采样）
        cache_this_step = {}

        for lid in range(g_num_layers):
            task_obs = obs[lid]["task_queue"]
            worker_loads = obs[lid]["worker_loads"]
            profile = obs[lid]["worker_profile"]

            # valid_mask 用你 run_mappo.py 的习惯：task_obs[:, 3]
            valid_mask = task_obs[:, g_valid_index].astype(np.float32)

            # agent.sample 返回 (value, action_env, logp, entropy)
            value, action_env, logp, _ = g_agents[lid].sample(
                task_obs, worker_loads, profile,
                deterministic=False,
                return_belief=False
            )

            actions_env[lid] = action_env

            # 先把当前步要存的东西缓存起来（next_obs 得 env.step 后才知道）
            raw_obs = _build_raw_obs_vec(task_obs, worker_loads, profile, valid_mask)

            cache_this_step[lid] = (raw_obs, action_env, float(logp), float(np.array(value).reshape(-1)[0]))

        # 环境推进一步
        next_obs, (_, reward_detail), done, _ = g_env.step(actions_env)

        # 回填 next_obs / reward / done，并把 reward/done 喂回 belief（下一步用）
        for lid in range(g_num_layers):
            raw_obs, action_env, logp_old, value_old = cache_this_step[lid]

            r = reward_detail["layer_rewards"][lid]["reward"]
            d = float(done)

            # next obs raw
            n_task_obs = next_obs[lid]["task_queue"]
            n_worker_loads = next_obs[lid]["worker_loads"]
            n_profile = next_obs[lid]["worker_profile"]
            n_valid_mask = n_task_obs[:, g_valid_index].astype(np.float32)
            raw_next_obs = _build_raw_obs_vec(n_task_obs, n_worker_loads, n_profile, n_valid_mask)

            # 训练用 action_flat（把 env action reshape 成一维）
            a_flat = np.array(action_env, dtype=np.float32).reshape(-1)

            buffers_local[lid]["obs_raw"].append(raw_obs)
            buffers_local[lid]["next_obs_raw"].append(raw_next_obs)
            buffers_local[lid]["actions_flat"].append(a_flat)
            buffers_local[lid]["logp_old"].append(logp_old)
            buffers_local[lid]["value_old"].append(value_old)
            buffers_local[lid]["rewards"].append(float(r))
            buffers_local[lid]["dones"].append(d)

            # 把 reward/done 喂回 agent（影响下一步 belief 更新）
            g_agents[lid].set_prev_feedback(r, d)

        obs = next_obs

    return buffers_local


def _init_worker(dire, env_config_path, schedule_path, worker_config_path, hidden_dim, z_dim,
                 belief_hidden, policy_hidden, value_hidden, decoder_hidden, elbo_coef, device_str):
    """
    每个子进程启动时调用一次：创建本进程持久复用的 env / agents（仅推理）
    """
    global g_env, g_agents, g_algs, g_num_layers, g_obs_dim_raw, g_action_dim, g_num_pad, g_valid_index

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

    n_worker, act_dim_per_worker = act_space.shape
    task_dim = obs_space["task_queue"].shape[1]
    load_dim = obs_space["worker_loads"].shape[1]
    n_task_types = len(env_cfg["task_types"])
    profile_shape = obs_space["worker_profile"].shape
    profile_dim_flat = int(np.prod(profile_shape))

    g_num_pad = g_env.num_pad_tasks
    g_action_dim = n_worker * act_dim_per_worker

    # raw obs dim：flatten(task_queue) + flatten(worker_loads) + flatten(profile) + flatten(valid_mask)
    g_obs_dim_raw = g_num_pad * task_dim + n_worker * load_dim + profile_dim_flat + g_num_pad

    g_agents, g_algs = {}, {}

    # 子进程只做推理：用 cpu
    infer_device = "cpu"

    for lid in range(g_num_layers):
        model = VariBADIndustrialModel(
            obs_dim=g_obs_dim_raw,
            action_dim=g_action_dim,
            obs_embed_dim=hidden_dim,
            belief_hidden=belief_hidden,
            z_dim=z_dim,
            policy_hidden=policy_hidden,
            value_hidden=value_hidden,
            decoder_hidden=decoder_hidden,
        )

        alg = VariBAD(
            model,
            clip_param=ppo_config["clip_param"],
            value_loss_coef=ppo_config["value_loss_coef"],
            entropy_coef=ppo_config["entropy_coef"],
            initial_lr=ppo_config["initial_lr"],
            max_grad_norm=ppo_config["max_grad_norm"],
            elbo_coef=elbo_coef,
            device=infer_device,
            writer=None,
            global_step_ref=[0],
            total_training_steps=1,
        )

        agent = VariBADIndustrialAgent(
            alg,
            device=infer_device,
            num_pad_tasks=g_num_pad,
            valid_index=g_valid_index,
            action_shape=act_space.shape,
        )

        g_algs[lid] = alg
        g_agents[lid] = agent


def evaluate_policy(agent_dict, eval_env, num_episodes, writer, global_step):
    total_reward, total_cost, total_utility, total_wait_penalty = 0, 0, 0, 0

    # 每条 eval episode 都要 reset belief
    for _ in range(num_episodes):
        obs = eval_env.reset()
        for lid in agent_dict:
            agent_dict[lid].reset_belief()

        done = False
        while not done:
            actions = {}
            for lid in obs:
                task_obs = obs[lid]["task_queue"]
                worker_loads = obs[lid]["worker_loads"]
                profile = obs[lid]["worker_profile"]

                _, act, _, _ = agent_dict[lid].sample(
                    task_obs, worker_loads, profile,
                    deterministic=True,
                    return_belief=False
                )
                actions[lid] = act

            obs, (_, reward_detail), done, _ = eval_env.step(actions)

            for lid, layer_stats in reward_detail["layer_rewards"].items():
                total_reward += layer_stats.get("reward", 0)
                total_cost += layer_stats.get("cost", 0)
                total_utility += layer_stats.get("utility", 0)
                total_wait_penalty += layer_stats.get("waiting_penalty", 0)

            # 把 reward/done 喂回 belief
            for lid in agent_dict:
                r = reward_detail["layer_rewards"][lid]["reward"]
                agent_dict[lid].set_prev_feedback(r, float(done))

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

    # ===== VariBAD extra hparams（不强依赖 config，有默认值）=====
    z_dim = int(ppo_config.get("varibad_z_dim", 64))
    belief_hidden = int(ppo_config.get("varibad_belief_hidden", 128))
    policy_hidden = int(ppo_config.get("varibad_policy_hidden", 256))
    value_hidden = int(ppo_config.get("varibad_value_hidden", 256))
    decoder_hidden = int(ppo_config.get("varibad_decoder_hidden", 256))
    elbo_coef = float(ppo_config.get("varibad_elbo_coef", 0.1))

    # ===== Init per-layer models and agents =====
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    n_worker, act_dim_per_worker = act_space.shape
    task_dim = obs_space["task_queue"].shape[1]
    load_dim = obs_space["worker_loads"].shape[1]
    n_task_types = len(env_config["task_types"])
    num_pad = env.num_pad_tasks
    profile_shape = obs_space["worker_profile"].shape
    profile_dim_flat = int(np.prod(profile_shape))

    obs_dim_raw = num_pad * task_dim + n_worker * load_dim + profile_dim_flat + num_pad
    action_dim = n_worker * act_dim_per_worker

    agents, algs, return_rms = {}, {}, {}

    for lid in range(num_layers):
        model = VariBADIndustrialModel(
            obs_dim=obs_dim_raw,
            action_dim=action_dim,
            obs_embed_dim=hidden_dim,
            belief_hidden=belief_hidden,
            z_dim=z_dim,
            policy_hidden=policy_hidden,
            value_hidden=value_hidden,
            decoder_hidden=decoder_hidden,
        )

        alg = VariBAD(
            model,
            clip_param=ppo_config["clip_param"],
            value_loss_coef=ppo_config["value_loss_coef"],
            entropy_coef=ppo_config["entropy_coef"],
            initial_lr=ppo_config["initial_lr"],
            max_grad_norm=ppo_config["max_grad_norm"],
            elbo_coef=elbo_coef,
            device=device,
            writer=writer,
            global_step_ref=[0],
            total_training_steps=ppo_config["num_episodes"] * env_config["max_steps"],
        )

        agent = VariBADIndustrialAgent(
            alg,
            device=device,
            num_pad_tasks=num_pad,
            valid_index=3,
            action_shape=act_space.shape,
        )

        agents[lid] = agent
        algs[lid] = alg
        return_rms[lid] = RunningMeanStd()

    # ===== Worker pool =====
    pool = mp.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(dire, env_config_path, schedule_path, worker_config_path,
                  hidden_dim, z_dim, belief_hidden, policy_hidden, value_hidden, decoder_hidden,
                  elbo_coef, device)
    ) if args.num_workers > 1 else None

    # ===== Training loop =====
    outer_loops = int(num_episodes / max(1, args.num_workers)) + 1

    for episode in range(outer_loops):
        with_new_schedule = (episode % reset_schedule_interval == 0)

        if args.num_workers == 1:
            # 单进程采样（与 worker 同逻辑）
            policy_states = _snapshot_policy_states(algs)
            result = _episode_worker(policy_states, with_new_schedule, seed=np.random.randint(0, 2**31 - 1))
            results = [result]
        else:
            policy_states = _snapshot_policy_states(algs)
            seeds = np.random.randint(0, 2 ** 31 - 1, size=args.num_workers).tolist()
            results = pool.starmap(
                _episode_worker,
                [(policy_states, with_new_schedule, int(seeds[wid])) for wid in range(args.num_workers)]
            )

        # ===== Learn each layer independently =====
        B = len(results)
        global_step = episode * steps_per_episode * max(1, args.num_workers)

        for lid in range(num_layers):
            # 统一序列长度：裁到最短（最省事，跟你之前的做法一致）
            lengths = [len(res[lid]["rewards"]) for res in results]
            if min(lengths) == 0:
                continue
            T = min(lengths)

            obs_raw = np.stack([np.stack(res[lid]["obs_raw"][:T], axis=0) for res in results], axis=0)          # [B,T,obs_dim]
            next_obs_raw = np.stack([np.stack(res[lid]["next_obs_raw"][:T], axis=0) for res in results], axis=0)  # [B,T,obs_dim]
            actions_flat = np.stack([np.stack(res[lid]["actions_flat"][:T], axis=0) for res in results], axis=0)  # [B,T,act_dim]
            logp_old = np.stack([np.array(res[lid]["logp_old"][:T], dtype=np.float32) for res in results], axis=0)[:, :, None]  # [B,T,1]
            value_old = np.stack([np.array(res[lid]["value_old"][:T], dtype=np.float32) for res in results], axis=0)[:, :, None]  # [B,T,1]
            rewards = np.stack([np.array(res[lid]["rewards"][:T], dtype=np.float32) for res in results], axis=0)[:, :, None]      # [B,T,1]
            dones = np.stack([np.array(res[lid]["dones"][:T], dtype=np.float32) for res in results], axis=0)[:, :, None]          # [B,T,1]

            # GAE per episode
            advs = np.zeros((B, T, 1), dtype=np.float32)
            rets = np.zeros((B, T, 1), dtype=np.float32)

            for b in range(B):
                adv_b, ret_b = _compute_gae(
                    rewards[b, :, 0],
                    value_old[b, :, 0],
                    dones[b, :, 0],
                    gamma, lam
                )
                advs[b, :, 0] = adv_b
                rets[b, :, 0] = ret_b

            # return normalization（按你 ppo_config 开关来）
            if ppo_config.get("return_normalization", False):
                return_rms[lid].update(rets.reshape(-1))
                rets = return_rms[lid].normalize(rets)

            # 以 episode 为单位做 minibatch（RNN 友好）
            ep_batch_size = int(ppo_config.get("varibad_ep_batch_size", B))
            ep_batch_size = max(1, min(ep_batch_size, B))

            idx = np.arange(B)
            for _ in range(1):
                np.random.shuffle(idx)
                for start in range(0, B, ep_batch_size):
                    mb = idx[start:start + ep_batch_size]

                    info = algs[lid].learn(
                        torch.tensor(obs_raw[mb], dtype=torch.float32),
                        torch.tensor(next_obs_raw[mb], dtype=torch.float32),
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
            evaluate_policy(agents, eval_env, eval_episodes, writer, global_step)

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    log_dir = f'../logs/{alg_name}/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    mp.set_start_method("spawn", force=True)
    main()
