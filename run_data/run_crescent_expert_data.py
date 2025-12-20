import argparse
import json
import random
import time
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch

from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.crescent_model import CrescentIndustrialModel
from algs.crescent import CRESCENT
from agents.mappo_agent import IndustrialAgent


# ========= Worker globals (sampling only) =========
g_env = None
g_agents = None
g_algs = None
g_num_layers = None
g_max_steps = None
g_profile_dim = None
g_n_worker = None
g_num_pad = None
g_macro_feat_dim = None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _stack_or_array(lst, dtype=None):
    """
    Trajectory saving helper:
    prefer np.stack for consistent shapes, fallback to object array if shapes differ.
    """
    if len(lst) == 0:
        return np.array([], dtype=np.float32 if dtype is None else dtype)
    try:
        arr = np.stack(lst, axis=0)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr
    except Exception:
        return np.array(lst, dtype=object)


def build_struct_macro_feature(obs, num_layers, step_idx, max_steps):
    """
    Build 1D macro structural feature vector (length F).
    Used only to probe macro_feat_dim for alg init compatibility.
    """
    feats = []
    total_valid = 0.0
    total_slots = 0.0

    for lid in range(num_layers):
        layer_obs = obs[lid]
        task_obs = layer_obs['task_queue']              # [N_slots, feat_dim]
        valid_mask = task_obs[:, 3].astype(np.float32)  # 4th dim: task exists
        num_valid = float(valid_mask.sum())
        num_slots = float(valid_mask.shape[0])

        backlog_ratio = num_valid / (num_slots + 1e-8)
        feats.append(backlog_ratio)

        total_valid += num_valid
        total_slots += num_slots

        worker_loads = layer_obs['worker_loads']        # [n_worker, k]
        wl = worker_loads.reshape(-1).astype(np.float32)
        if wl.size > 0:
            feats.append(float(wl.mean()))
            feats.append(float(wl.max()))
            feats.append(float(wl.std()))
        else:
            feats.extend([0.0, 0.0, 0.0])

    global_backlog_ratio = total_valid / (total_slots + 1e-8) if total_slots > 0.0 else 0.0
    feats.append(float(global_backlog_ratio))

    time_frac = float(step_idx) / float(max_steps) if max_steps > 0 else 0.0
    feats.append(time_frac)

    return np.array(feats, dtype=np.float32)  # [F]


def _episode_ext_return(buffers_local, num_layers: int) -> float:
    """Episode extrinsic return: sum over layers of l{lid}_reward."""
    s = 0.0
    for lid in range(num_layers):
        s += float(np.sum(np.array(buffers_local[lid]['reward'], dtype=np.float32)))
    return s


def _must_get(layer_info: dict, key: str, lid: int) -> float:
    """Strict getter: if missing, raise to avoid silently writing wrong data."""
    if key not in layer_info:
        raise KeyError(
            f"[reward_detail missing] layer_rewards[{lid}] has no key '{key}'. "
            f"Available keys = {list(layer_info.keys())}"
        )
    return float(layer_info[key])


def _save_offline_episode_npz(
        buffers_local,
        out_dir: Path,
        num_layers: int,
        episode_id: int,
        tag: str = "expert",
        dire: str = ""
) -> None:
    """
    Save ONE episode into a compressed npz.

    Saved keys (NO fused fields):
      global:
        - episode_id, dire, T, ext_return
      per-layer (for each lid):
        - l{lid}_task_obs, l{lid}_worker_loads, l{lid}_worker_profile, l{lid}_valid_mask
        - l{lid}_actions
        - l{lid}_reward, l{lid}_cost, l{lid}_utility, l{lid}_assign_bonus, l{lid}_wait_penalty
        - l{lid}_reward_u, l{lid}_reward_c
        - l{lid}_dones
        - l{lid}_next_task_obs, l{lid}_next_worker_loads, l{lid}_next_worker_profile, l{lid}_next_valid_mask
    """
    _ensure_dir(out_dir)

    T = int(len(buffers_local[0]['reward']))
    ext_ret = _episode_ext_return(buffers_local, num_layers)

    npz_dict = {
        "episode_id": np.int64(int(episode_id)),
        "dire": np.array(dire),
        "T": np.int64(T),
        "ext_return": np.float32(ext_ret),
    }

    for lid in range(num_layers):
        prefix = f"l{lid}_"

        # obs
        npz_dict[prefix + "task_obs"] = _stack_or_array(buffers_local[lid]['task_obs'], dtype=np.float32)
        npz_dict[prefix + "worker_loads"] = _stack_or_array(buffers_local[lid]['worker_loads'], dtype=np.float32)
        npz_dict[prefix + "worker_profile"] = _stack_or_array(buffers_local[lid]['worker_profile'], dtype=np.float32)
        npz_dict[prefix + "valid_mask"] = _stack_or_array(buffers_local[lid]['valid_mask'], dtype=np.float32)

        # action
        npz_dict[prefix + "actions"] = _stack_or_array(buffers_local[lid]['actions'], dtype=np.float32)

        # reward + decomposition
        npz_dict[prefix + "rewards"] = _stack_or_array(buffers_local[lid]['reward'], dtype=np.float32)
        npz_dict[prefix + "cost"] = _stack_or_array(buffers_local[lid]['cost'], dtype=np.float32)
        npz_dict[prefix + "utility"] = _stack_or_array(buffers_local[lid]['utility'], dtype=np.float32)
        npz_dict[prefix + "assign_bonus"] = _stack_or_array(buffers_local[lid]['assign_bonus'], dtype=np.float32)
        npz_dict[prefix + "wait_penalty"] = _stack_or_array(buffers_local[lid]['wait_penalty'], dtype=np.float32)

        # done
        npz_dict[prefix + "dones"] = _stack_or_array(buffers_local[lid]['dones'], dtype=np.float32)

        # next obs
        npz_dict[prefix + "next_task_obs"] = _stack_or_array(buffers_local[lid]['next_task_obs'], dtype=np.float32)
        npz_dict[prefix + "next_worker_loads"] = _stack_or_array(buffers_local[lid]['next_worker_loads'], dtype=np.float32)
        npz_dict[prefix + "next_worker_profile"] = _stack_or_array(buffers_local[lid]['next_worker_profile'], dtype=np.float32)
        npz_dict[prefix + "next_valid_mask"] = _stack_or_array(buffers_local[lid]['next_valid_mask'], dtype=np.float32)

    filename = f"{tag}_ep{int(episode_id):07d}_T{T:03d}.npz"
    np.savez_compressed(str(out_dir / filename), **npz_dict)


def _load_best_model_state_dicts(best_model_path: Path) -> dict:
    ckpt = torch.load(str(best_model_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dicts" in ckpt:
        return ckpt["model_state_dicts"]
    # allow fallback: directly a dict of {lid: state_dict}
    if isinstance(ckpt, dict) and all(isinstance(k, (int, str)) for k in ckpt.keys()):
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {best_model_path}")


def _init_worker_expert(dire, env_config_path, schedule_path, worker_config_path,
                        alg_name, hidden_dim, macro_feat_dim, best_model_path):
    """
    Worker initializer:
    create env/agents (sampling only), then load best_model weights ONCE.
    """
    global g_env, g_agents, g_algs, g_num_layers, g_max_steps
    global g_profile_dim, g_n_worker, g_num_pad, g_macro_feat_dim

    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)

    g_env = MultiplexEnv(env_config_path,
                         schedule_load_path=schedule_path,
                         worker_config_load_path=worker_config_path)
    g_env.chain = IndustrialChain(g_env.worker_config)

    g_num_layers = int(env_cfg["num_layers"])
    g_max_steps = int(env_cfg["max_steps"])
    g_macro_feat_dim = int(macro_feat_dim)

    obs_space = g_env.observation_space[0]
    act_space = g_env.action_space[0]
    g_n_worker, _ = act_space.shape
    n_task_types = len(env_cfg["task_types"])
    g_profile_dim = 2 * n_task_types
    g_num_pad = g_env.num_pad_tasks

    # Build per-layer agents/algs (CPU, inference only)
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
            macro_feat_dim=g_macro_feat_dim,
            use_contrastive=False,
            train_contrastive=False,
        )
        g_algs[lid] = alg
        g_agents[lid] = IndustrialAgent(alg, alg_name, device="cpu", num_pad_tasks=g_num_pad)

    # Load best_model weights once
    model_state_dicts = _load_best_model_state_dicts(Path(best_model_path))
    for lid in range(g_num_layers):
        sd = model_state_dicts[lid] if lid in model_state_dicts else model_state_dicts[str(lid)]
        g_algs[lid].model.load_state_dict(sd)


def _episode_worker_expert(with_new_schedule, seed, episode_id):
    """
    Worker: rollout ONE episode using loaded best_model, and return buffers_local.
    Saved fields are aligned with your required npz keys (except fused fields).
    """
    global g_env, g_agents, g_num_layers, g_max_steps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Only keep keys that will be saved (strict format)
    buffers_local = {
        lid: {k: [] for k in [
            'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
            'actions',
            'reward', 'cost', 'utility', 'assign_bonus', 'wait_penalty', 'reward_u', 'reward_c',
            'dones',
            'next_task_obs', 'next_worker_loads', 'next_worker_profile', 'next_valid_mask',
        ]}
        for lid in range(g_num_layers)
    }

    obs = g_env.reset(with_new_schedule=with_new_schedule)
    done = False
    step_idx = 0

    while (not done) and (step_idx < g_max_steps):
        actions = {}

        # store current obs + action
        for lid in range(g_num_layers):
            task_obs = obs[lid]['task_queue']
            worker_loads = obs[lid]['worker_loads']
            profile = obs[lid]['worker_profile']

            _, action, _, _ = g_agents[lid].sample(task_obs, worker_loads, profile)
            actions[lid] = action

            valid_mask = task_obs[:, 3].astype(np.float32)

            buffers_local[lid]['task_obs'].append(task_obs)
            buffers_local[lid]['worker_loads'].append(worker_loads)
            buffers_local[lid]['worker_profile'].append(profile)
            buffers_local[lid]['valid_mask'].append(valid_mask)
            buffers_local[lid]['actions'].append(action)

        # env step
        obs_next, (_, reward_detail), done, _ = g_env.step(actions)

        # store reward decomposition + next obs
        for lid in range(g_num_layers):
            n_task_obs = obs_next[lid]['task_queue']
            n_worker_loads = obs_next[lid]['worker_loads']
            n_profile = obs_next[lid]['worker_profile']
            n_valid_mask = n_task_obs[:, 3].astype(np.float32)

            buffers_local[lid]['next_task_obs'].append(n_task_obs)
            buffers_local[lid]['next_worker_loads'].append(n_worker_loads)
            buffers_local[lid]['next_worker_profile'].append(n_profile)
            buffers_local[lid]['next_valid_mask'].append(n_valid_mask)

            layer_info = reward_detail['layer_rewards'][lid]

            # strict: must exist, otherwise crash loudly
            r = _must_get(layer_info, 'reward', lid)
            c = _must_get(layer_info, 'cost', lid)
            u = _must_get(layer_info, 'utility', lid)
            ab = _must_get(layer_info, 'assign_bonus', lid)
            wp = _must_get(layer_info, 'wait_penalty', lid)

            buffers_local[lid]['reward'].append(r)
            buffers_local[lid]['cost'].append(c)
            buffers_local[lid]['utility'].append(u)
            buffers_local[lid]['assign_bonus'].append(ab)
            buffers_local[lid]['wait_penalty'].append(wp)
            buffers_local[lid]['dones'].append(float(done))

        obs = obs_next
        step_idx += 1

    return buffers_local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dire", type=str, default="standard")
    parser.add_argument("--alg_name", type=str, default="crescent")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--expert_episodes", type=int, default=500)
    parser.add_argument("--start_episode_id", type=int, default=0, help="Episode id offset for saved filenames")
    parser.add_argument("--with_new_schedule", action="store_true", help="Reset with new schedule every episode")
    parser.add_argument("--offline_data_root", type=str, default="../offline_data/crescent")
    parser.add_argument("--ckpt_root", type=str, default="../checkpoints/crescent")
    parser.add_argument("--best_model_path", type=str, default="", help="Optional explicit path to best_model.pt")
    parser.add_argument("--tag", type=str, default="expert")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="load", choices=["load", "save"],
                        help="Keep consistent with training env init (load schedule or save schedule)")
    args = parser.parse_args()

    dire = args.dire
    alg_name = args.alg_name.lower()

    env_config_path = f"../configs/{dire}/env_config.json"
    schedule_path = f"../configs/{dire}/eval_schedule.json"
    worker_config_path = f"../configs/{dire}/worker_config.json"

    # config
    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)
    with open("../configs/ppo_config.json", "r", encoding="utf-8") as f:
        ppo_cfg = json.load(f)

    num_layers = int(env_cfg["num_layers"])
    steps_per_episode = int(env_cfg["max_steps"])
    hidden_dim = int(ppo_cfg.get("hidden_dim", 256))

    # Determine macro_feat_dim (only for alg init compatibility)
    if args.mode == "save":
        tmp_env = MultiplexEnv(env_config_path, schedule_save_path=schedule_path,
                               worker_config_save_path=worker_config_path)
    else:
        tmp_env = MultiplexEnv(env_config_path, schedule_load_path=schedule_path,
                               worker_config_load_path=worker_config_path)
    tmp_env.chain = IndustrialChain(tmp_env.worker_config)
    probe_obs = tmp_env.reset()
    raw_macro_probe = build_struct_macro_feature(
        obs=probe_obs, num_layers=num_layers, step_idx=0, max_steps=steps_per_episode
    )
    macro_feat_dim = int(raw_macro_probe.shape[0])

    # Resolve best_model path
    if args.best_model_path:
        best_model_path = Path(args.best_model_path)
    else:
        best_model_path = Path(args.ckpt_root) / dire / "best_model.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"best_model.pt not found: {best_model_path}")

    # Output dir
    out_dir = Path(args.offline_data_root) / dire / "expert"
    _ensure_dir(out_dir)

    # Seeds
    rng = np.random.RandomState(args.seed)

    # Multiprocessing pool
    pool = mp.Pool(
        processes=args.num_workers,
        initializer=_init_worker_expert,
        initargs=(dire, env_config_path, schedule_path, worker_config_path,
                  alg_name, hidden_dim, macro_feat_dim, str(best_model_path))
    ) if args.num_workers > 1 else None

    print(f"[collect_expert] dire={dire} num_workers={args.num_workers} expert_episodes={args.expert_episodes}")
    print(f"[collect_expert] best_model_path={best_model_path}")
    print(f"[collect_expert] out_dir={out_dir} macro_feat_dim={macro_feat_dim} hidden_dim={hidden_dim}")

    collected = 0
    start_id = int(args.start_episode_id)
    t0 = time.time()

    while collected < args.expert_episodes:
        batch_k = min(args.num_workers, args.expert_episodes - collected)
        episode_ids = [start_id + collected + i for i in range(batch_k)]
        seeds = rng.randint(0, 2 ** 31 - 1, size=batch_k).tolist()
        with_new_schedule = bool(args.with_new_schedule)

        if pool is None:
            raise RuntimeError("Please use --num_workers >= 2 for expert collection (this script is designed for MP sampling).")
        else:
            results = pool.starmap(
                _episode_worker_expert,
                [(with_new_schedule, int(seeds[i]), int(episode_ids[i])) for i in range(batch_k)]
            )

        # Save each episode (bind episode_id explicitly to avoid relying on buffers)
        ext_returns = []
        for i, res in enumerate(results):
            ep_id = int(episode_ids[i])
            ext_r = _episode_ext_return(res, num_layers)
            ext_returns.append(ext_r)

            _save_offline_episode_npz(
                res,
                out_dir=out_dir,
                num_layers=num_layers,
                episode_id=ep_id,
                tag=args.tag,
                dire=dire
            )

        collected += batch_k
        avg_ret = float(np.mean(ext_returns)) if ext_returns else 0.0
        mx_ret = float(np.max(ext_returns)) if ext_returns else 0.0
        elapsed = time.time() - t0
        print(f"[collect_expert] {collected}/{args.expert_episodes} "
              f"batch_avg_ext_return={avg_ret:.3f} batch_max={mx_ret:.3f} elapsed={elapsed:.1f}s")

    if pool is not None:
        pool.close()
        pool.join()

    print(f"[collect_expert] done. saved {args.expert_episodes} episodes to: {out_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
