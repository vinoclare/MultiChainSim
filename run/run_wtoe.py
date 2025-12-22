# run/run_wtoe.py
import torch
import numpy as np
import argparse
import json
import time
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.crescent_model import CrescentIndustrialModel
from algs.wtoe import WTOE
from agents.mappo_agent import IndustrialAgent
from utils.utils import RunningMeanStd


parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
parser.add_argument("--alg_name", type=str, default="wtoe")
parser.add_argument("--num_workers", type=int, default=10, help="Parallel env workers for sampling")
parser.add_argument("--mode", type=str, default="load", help="save or load configs")

# WToE hyperparams
parser.add_argument("--hist_len", type=int, default=4, help="macro history length (K)")
parser.add_argument("--z_dim", type=int, default=16)
parser.add_argument("--explore_scale", type=float, default=2.0)
parser.add_argument("--kl_threshold", type=float, default=0.05)
parser.add_argument("--kl_beta", type=float, default=10.0)
parser.add_argument("--entropy_boost", type=float, default=1.0)
parser.add_argument("--vae_lr", type=float, default=3e-4)
parser.add_argument("--vae_kl_coef", type=float, default=0.1)

args, _ = parser.parse_known_args()
dire = args.dire
alg_name = args.alg_name.lower()

with open(f"../configs/{dire}/env_config.json") as f:
    env_config = json.load(f)
with open("../configs/ppo_config.json") as f:
    ppo_config = json.load(f)

env_config_path = f"../configs/{dire}/env_config.json"
schedule_path = f"../configs/{dire}/train_schedule.json"
eval_schedule_path = f"../configs/{dire}/eval_schedule.json"
worker_config_path = f"../configs/{dire}/worker_config.json"

# ===== multiprocessing globals =====
g_env = None
g_agents = None
g_algs = None
g_num_layers = None
g_obs_space = None
g_profile_dim = None
g_n_worker = None
g_num_pad = None
g_max_steps = None
g_macro_feat_dim = None
g_hist_len = None


def build_struct_macro_feature(obs, num_layers, step_idx, max_steps):
    feats = []
    total_valid = 0.0
    total_slots = 0.0

    for lid in range(num_layers):
        layer_obs = obs[lid]
        task_obs = layer_obs["task_queue"]
        valid_mask = task_obs[:, 3].astype(np.float32)
        num_valid = float(valid_mask.sum())
        num_slots = float(valid_mask.shape[0])

        backlog_ratio = num_valid / (num_slots + 1e-8)
        feats.append(backlog_ratio)

        total_valid += num_valid
        total_slots += num_slots

        worker_loads = layer_obs["worker_loads"]
        wl = worker_loads.reshape(-1).astype(np.float32)
        if wl.size > 0:
            feats.append(float(wl.mean()))
            feats.append(float(wl.max()))
            feats.append(float(wl.std()))
        else:
            feats.extend([0.0, 0.0, 0.0])

    feats.append(float(total_valid / (total_slots + 1e-8)) if total_slots > 0 else 0.0)
    feats.append(float(step_idx) / float(max_steps) if max_steps > 0 else 0.0)
    return np.array(feats, dtype=np.float32)


def macro_hist_flatten(hist_deque, hist_len, macro_feat_dim):
    """
    hist_deque: deque of np.ndarray [F]
    return: np.ndarray [hist_len * F]
    """
    buf = np.zeros((hist_len, macro_feat_dim), dtype=np.float32)
    h = list(hist_deque)
    # right-aligned
    for i, v in enumerate(h[-hist_len:]):
        buf[hist_len - len(h[-hist_len:]) + i] = v
    return buf.reshape(-1)


def _snapshot_states(algs_dict):
    """
    需要把 actor(model) + vae 都传给 worker（采样时要用 divergence->explore_prob）
    """
    out = {}
    for lid, alg in algs_dict.items():
        out[lid] = {
            "model": alg.model.state_dict(),
            "vae": alg.vae.state_dict() if getattr(alg, "vae", None) is not None else None,
        }
    return out


def _init_worker(dire, env_config_path, schedule_path, worker_config_path,
                 hidden_dim, macro_feat_dim, hist_len):
    global g_env, g_agents, g_algs, g_num_layers, g_obs_space, g_profile_dim
    global g_n_worker, g_num_pad, g_max_steps, g_macro_feat_dim, g_hist_len

    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)

    g_env = MultiplexEnv(env_config_path,
                         schedule_load_path=schedule_path,
                         worker_config_load_path=worker_config_path)
    g_env.chain = IndustrialChain(g_env.worker_config)

    g_num_layers = env_cfg["num_layers"]
    g_max_steps = env_cfg["max_steps"]
    g_macro_feat_dim = macro_feat_dim
    g_hist_len = hist_len

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
            task_input_dim=obs_space["task_queue"].shape[1],
            worker_load_input_dim=obs_space["worker_loads"].shape[1],
            worker_profile_input_dim=g_profile_dim,
            n_worker=g_n_worker,
            num_pad_tasks=g_num_pad,
            hidden_dim=hidden_dim
        )

        alg = WTOE(
            model,
            clip_param=ppo_config["clip_param"],
            value_loss_coef=ppo_config["value_loss_coef"],
            entropy_coef=ppo_config["entropy_coef"],
            initial_lr=ppo_config["initial_lr"],
            vae_lr=args.vae_lr,
            max_grad_norm=ppo_config["max_grad_norm"],
            writer=None,
            global_step_ref=[0],
            total_training_steps=ppo_config["num_episodes"] * env_cfg["max_steps"],
            device="cpu",
            inner_k=1,
            macro_hist_dim=hist_len * macro_feat_dim,
            z_dim=args.z_dim,
            explore_scale=args.explore_scale,
            kl_threshold=args.kl_threshold,
            kl_beta=args.kl_beta,
            entropy_boost=args.entropy_boost,
            vae_kl_coef=args.vae_kl_coef,
        )

        g_algs[lid] = alg
        g_agents[lid] = IndustrialAgent(alg, "wtoe", device="cpu", num_pad_tasks=g_num_pad)


def _episode_worker(policy_states, with_new_schedule, seed, worker_id):
    global g_env, g_agents, g_algs, g_num_layers, g_max_steps, g_macro_feat_dim, g_hist_len

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load snapshot
    for lid in range(g_num_layers):
        g_algs[lid].model.load_state_dict(policy_states[lid]["model"])
        if policy_states[lid]["vae"] is not None and getattr(g_algs[lid], "vae", None) is not None:
            g_algs[lid].vae.load_state_dict(policy_states[lid]["vae"])

    buffers_local = {
        lid: {k: [] for k in [
            "task_obs", "worker_loads", "worker_profile", "valid_mask",
            "actions", "logprobs",
            "rewards", "dones", "values",
            "macro_hist", "episode_ids", "step_ids",
            "explore_prob", "explore_kl"
        ]}
        for lid in range(g_num_layers)
    }

    obs = g_env.reset(with_new_schedule=with_new_schedule)
    done = False
    step_idx = 0
    episode_id = worker_id
    hist = deque(maxlen=g_hist_len)

    while not done:
        # build macro + hist first
        raw_macro = build_struct_macro_feature(obs, g_num_layers, step_idx, g_max_steps)
        hist.append(raw_macro)
        mh = macro_hist_flatten(hist, g_hist_len, g_macro_feat_dim)  # [K*F]

        actions = {}
        for lid in range(g_num_layers):
            task_obs = obs[lid]["task_queue"]
            worker_loads = obs[lid]["worker_loads"]
            profile = obs[lid]["worker_profile"]

            value, action, logprob, _ = g_agents[lid].sample(task_obs, worker_loads, profile, macro_hist_np=mh)
            actions[lid] = action
            valid_mask = task_obs[:, 3].astype(np.float32)

            buffers_local[lid]["task_obs"].append(task_obs)
            buffers_local[lid]["worker_loads"].append(worker_loads)
            buffers_local[lid]["worker_profile"].append(profile)
            buffers_local[lid]["valid_mask"].append(valid_mask)
            buffers_local[lid]["actions"].append(action)
            buffers_local[lid]["logprobs"].append(logprob)
            buffers_local[lid]["values"].append(value)
            buffers_local[lid]["macro_hist"].append(mh)
            buffers_local[lid]["episode_ids"].append(episode_id)
            buffers_local[lid]["step_ids"].append(step_idx)

            buffers_local[lid]["explore_prob"].append(float(g_algs[lid].last_explore_prob))
            buffers_local[lid]["explore_kl"].append(float(g_algs[lid].last_kl))

        obs, (_, reward_detail), done, _ = g_env.step(actions)
        for lid in range(g_num_layers):
            r = reward_detail["layer_rewards"][lid]["reward"]
            buffers_local[lid]["rewards"].append(r)
            buffers_local[lid]["dones"].append(done)

        step_idx += 1

    return buffers_local


def evaluate_policy(agent_dict, eval_env, num_episodes, writer, global_step):
    total_reward, total_cost, total_utility, total_wait_penalty = 0, 0, 0, 0
    for _ in range(num_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            actions = {}
            for lid in obs:
                task_obs = obs[lid]["task_queue"]
                worker_loads = obs[lid]["worker_loads"]
                profile = obs[lid]["worker_profile"]
                _, act, _, _ = agent_dict[lid].sample(task_obs, worker_loads, profile)
                actions[lid] = act
            obs, (_, reward_detail), done, _ = eval_env.step(actions)
            for lid, layer_stats in reward_detail["layer_rewards"].items():
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
    else:
        env = MultiplexEnv(env_config_path, schedule_load_path=schedule_path,
                           worker_config_load_path=worker_config_path)
        eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path)

    eval_env.worker_config = env.worker_config
    eval_env.chain = IndustrialChain(eval_env.worker_config)

    num_layers = env_config["num_layers"]
    num_episodes = ppo_config["num_episodes"]
    steps_per_episode = env_config["max_steps"]
    update_epochs = ppo_config["update_epochs"]
    gamma = ppo_config["gamma"]
    lam = ppo_config["lam"]
    batch_size = ppo_config["batch_size"]
    hidden_dim = ppo_config["hidden_dim"]
    device = ppo_config["device"]
    eval_interval = ppo_config["eval_interval"] / max(1, args.num_workers)
    eval_episodes = ppo_config["eval_episodes"]
    reset_schedule_interval = ppo_config["reset_schedule_interval"]

    # macro_feat_dim probe
    probe_obs = env.reset(with_new_schedule=True)
    macro_probe = build_struct_macro_feature(probe_obs, num_layers, 0, steps_per_episode)
    macro_feat_dim = int(macro_probe.shape[0])
    macro_hist_dim = args.hist_len * macro_feat_dim

    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    n_worker, _ = act_space.shape
    n_task_types = len(env_config["task_types"])
    profile_dim = 2 * n_task_types

    agents, algs, return_rms, buffers = {}, {}, {}, {}

    for lid in range(num_layers):
        model = CrescentIndustrialModel(
            task_input_dim=obs_space["task_queue"].shape[1],
            worker_load_input_dim=obs_space["worker_loads"].shape[1],
            worker_profile_input_dim=profile_dim,
            n_worker=n_worker,
            num_pad_tasks=env.num_pad_tasks,
            hidden_dim=hidden_dim
        )

        alg = WTOE(
            model,
            clip_param=ppo_config["clip_param"],
            value_loss_coef=ppo_config["value_loss_coef"],
            entropy_coef=ppo_config["entropy_coef"],
            initial_lr=ppo_config["initial_lr"],
            vae_lr=args.vae_lr,
            max_grad_norm=ppo_config["max_grad_norm"],
            writer=writer,
            global_step_ref=[0],
            total_training_steps=ppo_config["num_episodes"] * env_config["max_steps"],
            device=device,
            inner_k=1,
            macro_hist_dim=macro_hist_dim,
            z_dim=args.z_dim,
            explore_scale=args.explore_scale,
            kl_threshold=args.kl_threshold,
            kl_beta=args.kl_beta,
            entropy_boost=args.entropy_boost,
            vae_kl_coef=args.vae_kl_coef,
        )

        agents[lid] = IndustrialAgent(alg, "wtoe", device, env.num_pad_tasks)
        algs[lid] = alg
        return_rms[lid] = RunningMeanStd()
        buffers[lid] = {k: [] for k in [
            "task_obs", "worker_loads", "worker_profile", "valid_mask",
            "actions", "logprobs", "rewards", "dones", "values",
            "macro_hist", "episode_ids", "step_ids",
            "explore_prob", "explore_kl"
        ]}

    pool = mp.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(dire, env_config_path, schedule_path, worker_config_path,
                  hidden_dim, macro_feat_dim, args.hist_len)
    ) if args.num_workers > 1 else None

    for episode in range(int(num_episodes / max(1, args.num_workers)) + 1):
        if args.num_workers == 1:
            obs = env.reset(with_new_schedule=(episode % reset_schedule_interval == 0))
            done = False
            hist = deque(maxlen=args.hist_len)

            for step in range(steps_per_episode):
                raw_macro = build_struct_macro_feature(obs, num_layers, step, steps_per_episode)
                hist.append(raw_macro)
                mh = macro_hist_flatten(hist, args.hist_len, macro_feat_dim)

                actions = {}
                for lid in range(num_layers):
                    task_obs = obs[lid]["task_queue"]
                    worker_loads = obs[lid]["worker_loads"]
                    profile = obs[lid]["worker_profile"]

                    value, action, logprob, _ = agents[lid].sample(task_obs, worker_loads, profile, macro_hist_np=mh)
                    actions[lid] = action
                    valid_mask = task_obs[:, 3].astype(np.float32)

                    buffers[lid]["task_obs"].append(task_obs)
                    buffers[lid]["worker_loads"].append(worker_loads)
                    buffers[lid]["worker_profile"].append(profile)
                    buffers[lid]["valid_mask"].append(valid_mask)
                    buffers[lid]["actions"].append(action)
                    buffers[lid]["logprobs"].append(logprob)
                    buffers[lid]["values"].append(value)
                    buffers[lid]["macro_hist"].append(mh)
                    buffers[lid]["episode_ids"].append(episode)
                    buffers[lid]["step_ids"].append(step)

                    buffers[lid]["explore_prob"].append(float(algs[lid].last_explore_prob))
                    buffers[lid]["explore_kl"].append(float(algs[lid].last_kl))

                obs, (_, reward_detail), done, _ = env.step(actions)
                for lid in range(num_layers):
                    r = reward_detail["layer_rewards"][lid]["reward"]
                    buffers[lid]["rewards"].append(r)
                    buffers[lid]["dones"].append(done)

                if done:
                    break
        else:
            policy_states = _snapshot_states(algs)
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

                for key in buffers[lid].keys():
                    buffers[lid][key].extend(_cat_list(key))

        # ====== Learn ======
        for lid in range(num_layers):
            advs = []
            vals = buffers[lid]["values"] + [0.0]
            gae = 0.0
            for t in reversed(range(len(buffers[lid]["rewards"]))):
                delta = (buffers[lid]["rewards"][t]
                         + gamma * vals[t + 1] * (1 - buffers[lid]["dones"][t])
                         - vals[t])
                gae = delta + gamma * lam * (1 - buffers[lid]["dones"][t]) * gae
                advs.insert(0, gae.copy())
            rets = [a + v for a, v in zip(advs, buffers[lid]["values"])]

            advs = np.array(advs, dtype=np.float32)
            rets = np.array(rets, dtype=np.float32)

            # [T] -> [T,W]
            advs_w = np.repeat(advs[:, None], n_worker, axis=1).reshape(-1, n_worker)
            rets_b = rets.reshape(-1)

            if ppo_config["return_normalization"]:
                return_rms[lid].update(np.array(rets_b))
                rets_b = return_rms[lid].normalize(np.array(rets_b))

            dataset = list(zip(
                buffers[lid]["task_obs"],
                buffers[lid]["worker_loads"],
                buffers[lid]["worker_profile"],
                buffers[lid]["valid_mask"],
                buffers[lid]["macro_hist"],
                buffers[lid]["actions"],
                buffers[lid]["values"],
                rets_b,
                buffers[lid]["logprobs"],
                advs_w
            ))

            for _ in range(update_epochs):
                random.shuffle(dataset)
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i + batch_size]
                    (task_b, load_b, prof_b, mask_b,
                     mh_b, act_b, val_old_b, ret_b, logp_old_b, adv_b) = zip(*batch)

                    task_t = torch.tensor(np.array(task_b, dtype=np.float32))
                    load_t = torch.tensor(np.array(load_b, dtype=np.float32))
                    prof_t = torch.tensor(np.array(prof_b, dtype=np.float32))
                    mask_t = torch.tensor(np.array(mask_b, dtype=np.float32))
                    mh_t = torch.tensor(np.array(mh_b, dtype=np.float32))

                    act_t = torch.tensor(np.array(act_b, dtype=np.float32))
                    val_old_t = torch.tensor(np.array(val_old_b, dtype=np.float32))
                    ret_t = torch.tensor(np.array(ret_b, dtype=np.float32))
                    logp_old_t = torch.tensor(np.array(logp_old_b, dtype=np.float32))
                    adv_t = torch.tensor(np.array(adv_b, dtype=np.float32))

                    current_steps = episode * steps_per_episode * args.num_workers
                    stats = algs[lid].learn(
                        task_t, load_t, prof_t, mask_t,
                        mh_t, act_t, val_old_t, ret_t, logp_old_t, adv_t,
                        current_steps
                    )

                    # logging (only lid=0 to reduce noise)
                    if lid == 0 and episode % max(1, int(eval_interval)) == 0:
                        writer.add_scalar("train/value_loss", stats["value_loss"], current_steps)
                        writer.add_scalar("train/policy_loss", stats["policy_loss"], current_steps)
                        writer.add_scalar("train/entropy", stats["entropy"], current_steps)
                        writer.add_scalar("train/vae_loss", stats["vae_loss"], current_steps)
                        writer.add_scalar("train/vae_recon", stats["vae_recon"], current_steps)
                        writer.add_scalar("train/vae_kl", stats["vae_kl"], current_steps)

        # extra logs
        if episode % max(1, int(eval_interval)) == 0:
            lid0_prob = float(np.mean(buffers[0]["explore_prob"])) if len(buffers[0]["explore_prob"]) > 0 else 0.0
            lid0_kl = float(np.mean(buffers[0]["explore_kl"])) if len(buffers[0]["explore_kl"]) > 0 else 0.0
            writer.add_scalar("train/explore_prob_l0", lid0_prob, episode * steps_per_episode * args.num_workers)
            writer.add_scalar("train/explore_kl_l0", lid0_kl, episode * steps_per_episode * args.num_workers)

        # clear buffers
        for lid in buffers:
            for k in buffers[lid]:
                buffers[lid][k].clear()

        if episode % max(1, int(eval_interval)) == 0:
            evaluate_policy(agents, eval_env, eval_episodes, writer,
                            episode * steps_per_episode * args.num_workers)

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    log_dir = f"../logs/{alg_name}/{dire}/" + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    mp.set_start_method("spawn", force=True)
    main()
