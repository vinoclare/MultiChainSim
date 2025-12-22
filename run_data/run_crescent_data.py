import torch
import numpy as np
import argparse
import json
import time
import random
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from pathlib import Path

from envs import IndustrialChain
from envs.env import MultiplexEnv
from models.crescent_model import CrescentIndustrialModel
from algs.crescent import CRESCENT
from agents.mappo_agent import IndustrialAgent
from utils.utils import RunningMeanStd
from explore.crescent_cluster import CrescentClusterer

# ===== Load configurations =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
parser.add_argument("--alg_name", type=str, default="crescent")
parser.add_argument("--num_workers", type=int, default=10, help="Parallel env workers for sampling")
parser.add_argument("--mode", type=str, default="load", help="save or load configs")

# ===== Offline data collection (HiTAC-consistent) =====
parser.add_argument("--offline_save_interval", type=int, default=10,
                    help="Save ONE episode trajectory every N episodes (1-indexed). <=0 disables saving.")
parser.add_argument("--offline_data_root", type=str, default="../offline_data/crescent",
                    help="Root dir to save offline trajectories (will append /{dire}/)")
parser.add_argument("--ckpt_root", type=str, default="../checkpoints/crescent",
                    help="Root dir to save best_model.pt (will append /{dire}/)")

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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _snapshot_policy_states(algs_dict):
    """把每层当前策略权重做成只含张量的 state_dict，便于跨进程传递。"""
    return {lid: algs_dict[lid].model.state_dict() for lid in algs_dict}


def _episode_ext_return(buffers_local, num_layers: int) -> float:
    """episode 的 extrinsic return：对所有层求和。"""
    s = 0.0
    for lid in range(num_layers):
        s += float(np.sum(np.array(buffers_local[lid]['ext_rewards'], dtype=np.float32)))
    return s


def _save_best_model(algs_dict, ckpt_dir: Path, best_ext_return: float,
                     best_episode_id: int, outer_idx: int, worker_id: int) -> None:
    """
    实时保存 best_model：仅保存模型参数（每层一个 state_dict），用于后续采专家数据。
    """
    _ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / "best_model.pt"
    meta_path = ckpt_dir / "best_model_meta.json"

    state = {
        "model_state_dicts": {lid: algs_dict[lid].model.state_dict() for lid in algs_dict},
        "best_ext_return": float(best_ext_return),
        "best_episode_id": int(best_episode_id),
        "outer_idx": int(outer_idx),
        "worker_id": int(worker_id),
        "saved_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dire": dire,
        "alg_name": alg_name,
    }
    torch.save(state, str(ckpt_path))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_ext_return": float(best_ext_return),
            "best_episode_id": int(best_episode_id),
            "outer_idx": int(outer_idx),
            "worker_id": int(worker_id),
            "saved_time": state["saved_time"],
            "dire": dire,
            "alg_name": alg_name,
            "ckpt_path": str(ckpt_path),
        }, f, ensure_ascii=False, indent=2)


def _stack_or_array(lst, dtype=None):
    """
    轨迹保存：优先 stack（保证 shape 统一），失败则退化为 np.array(object)。
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


def _save_offline_episode_npz(buffers_local, out_dir: Path, num_layers: int, global_episode_id: int, tag: str = "traj") -> None:
    """
    保存一个 episode（包含 L 个 layer 的轨迹）到 npz。
    要求 buffers_local 已经包含：
      - obs: task_obs/worker_loads/worker_profile/valid_mask
      - next_obs: next_task_obs/next_worker_loads/next_worker_profile/next_valid_mask
      - actions, logprobs, values
      - rewards/ext_rewards/int_rewards, dones
      - macro_feat, episode_ids, step_ids
    """
    _ensure_dir(out_dir)

    # 注意：global_episode_id 用于离线数据命名与频率计数；episode_ids 保持原训练逻辑（多进程下通常是 worker_id）
    ep_id = int(global_episode_id)
    T = len(buffers_local[0]['rewards'])

    ext_ret = _episode_ext_return(buffers_local, num_layers)
    tot_ret = 0.0
    int_ret = 0.0
    for lid in range(num_layers):
        tot_ret += float(np.sum(np.array(buffers_local[lid]['rewards'], dtype=np.float32)))
        int_ret += float(np.sum(np.array(buffers_local[lid]['int_rewards'], dtype=np.float32)))

    npz_dict = {
        "episode_id": np.int64(ep_id),
        "T": np.int64(T),
        "ext_return": np.float32(ext_ret),
        "int_return": np.float32(int_ret),
        "total_return": np.float32(tot_ret),
        "dire": np.array(dire),
        "alg_name": np.array(alg_name),
        "valid_mask_col_idx": np.int64(3),
    }

    for lid in range(num_layers):
        prefix = f"l{lid}_"
        npz_dict[prefix + "task_obs"] = _stack_or_array(buffers_local[lid]['task_obs'], dtype=np.float32)
        npz_dict[prefix + "worker_loads"] = _stack_or_array(buffers_local[lid]['worker_loads'], dtype=np.float32)
        npz_dict[prefix + "worker_profile"] = _stack_or_array(buffers_local[lid]['worker_profile'], dtype=np.float32)
        npz_dict[prefix + "valid_mask"] = _stack_or_array(buffers_local[lid]['valid_mask'], dtype=np.float32)

        npz_dict[prefix + "actions"] = _stack_or_array(buffers_local[lid]['actions'], dtype=np.float32)
        npz_dict[prefix + "logprobs"] = _stack_or_array(buffers_local[lid]['logprobs'], dtype=np.float32)
        npz_dict[prefix + "values"] = _stack_or_array(buffers_local[lid]['values'], dtype=np.float32)

        npz_dict[prefix + "rewards"] = _stack_or_array(buffers_local[lid]['rewards'], dtype=np.float32)
        npz_dict[prefix + "ext_rewards"] = _stack_or_array(buffers_local[lid]['ext_rewards'], dtype=np.float32)
        npz_dict[prefix + "int_rewards"] = _stack_or_array(buffers_local[lid]['int_rewards'], dtype=np.float32)
        npz_dict[prefix + "dones"] = _stack_or_array(buffers_local[lid]['dones'], dtype=np.float32)

        npz_dict[prefix + "macro_feat"] = _stack_or_array(buffers_local[lid]['macro_feat'], dtype=np.float32)
        npz_dict[prefix + "episode_ids"] = _stack_or_array(buffers_local[lid]['episode_ids'], dtype=np.int64)
        npz_dict[prefix + "step_ids"] = _stack_or_array(buffers_local[lid]['step_ids'], dtype=np.int64)

        npz_dict[prefix + "next_task_obs"] = _stack_or_array(buffers_local[lid]['next_task_obs'], dtype=np.float32)
        npz_dict[prefix + "next_worker_loads"] = _stack_or_array(buffers_local[lid]['next_worker_loads'], dtype=np.float32)
        npz_dict[prefix + "next_worker_profile"] = _stack_or_array(buffers_local[lid]['next_worker_profile'], dtype=np.float32)
        npz_dict[prefix + "next_valid_mask"] = _stack_or_array(buffers_local[lid]['next_valid_mask'], dtype=np.float32)

    filename = f"{tag}_ep{ep_id:07d}_T{T:03d}.npz"
    np.savez_compressed(str(out_dir / filename), **npz_dict)


def _episode_worker(policy_states, with_new_schedule, seed, base_episode_id, worker_id):
    """
    子进程：加载主进程广播的策略快照（只读），采样 1 条 episode 并返回 buffers_local。
    注意：episode_id 使用全局唯一编号 base_episode_id + worker_id，避免跨 batch 重复。
    """
    global g_env, g_agents, g_algs, g_num_layers, g_max_steps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载主进程广播的策略快照（只读）
    for lid in range(g_num_layers):
        g_algs[lid].model.load_state_dict(policy_states[lid])

    buffers_local = {
        lid: {k: [] for k in [
            'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
            'actions', 'logprobs',
            'rewards', 'ext_rewards', 'int_rewards',
            'dones', 'values',
            'macro_feat', 'episode_ids', 'step_ids',
            # next state for offline RL
            'next_task_obs', 'next_worker_loads', 'next_worker_profile', 'next_valid_mask',
        ]}
        for lid in range(g_num_layers)
    }

    obs = g_env.reset(with_new_schedule=with_new_schedule)
    done = False
    step_idx = 0
    # 保持原训练逻辑：episode_id 仍然用 worker_id（0..num_workers-1），不要改成全局 id
    episode_id = int(worker_id)

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

        # 宏观结构特征（真实维度）
        raw_macro = build_struct_macro_feature(
            obs=obs,
            num_layers=g_num_layers,
            step_idx=step_idx,
            max_steps=g_max_steps
        )
        for lid in range(g_num_layers):
            buffers_local[lid]['macro_feat'].append(raw_macro)

        obs_next, (_, reward_detail), done, _ = g_env.step(actions)

        for lid in range(g_num_layers):
            # next obs for offline
            n_task_obs = obs_next[lid]['task_queue']
            n_worker_loads = obs_next[lid]['worker_loads']
            n_profile = obs_next[lid]['worker_profile']
            n_valid_mask = n_task_obs[:, 3].astype(np.float32)

            buffers_local[lid]['next_task_obs'].append(n_task_obs)
            buffers_local[lid]['next_worker_loads'].append(n_worker_loads)
            buffers_local[lid]['next_worker_profile'].append(n_profile)
            buffers_local[lid]['next_valid_mask'].append(n_valid_mask)

            r = reward_detail['layer_rewards'][lid]['reward']
            buffers_local[lid]['rewards'].append(r)       # will add int later in main
            buffers_local[lid]['ext_rewards'].append(r)
            buffers_local[lid]['dones'].append(done)

        obs = obs_next
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

    return np.array(feats, dtype=np.float32)  # [F]


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
            for _, layer_stats in reward_detail['layer_rewards'].items():
                total_reward += layer_stats.get("reward", 0)
                total_cost += layer_stats.get("cost", 0)
                total_utility += layer_stats.get("utility", 0)
                total_wait_penalty += layer_stats.get("waiting_penalty", 0)

    writer.add_scalar("eval/reward", total_reward / num_episodes, global_step)
    writer.add_scalar("eval/cost", total_cost / num_episodes, global_step)
    writer.add_scalar("eval/utility", total_utility / num_episodes, global_step)
    writer.add_scalar("eval/waiting_penalty", total_wait_penalty / num_episodes, global_step)


def _rollout_episode_main(env, agents, num_layers, steps_per_episode, with_new_schedule, episode_id):
    """
    主进程单线程采样 1 条 episode，返回与 worker 同格式的 buffers_local（含 next-state）。
    """
    buffers_local = {
        lid: {k: [] for k in [
            'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
            'actions', 'logprobs',
            'rewards', 'ext_rewards', 'int_rewards',
            'dones', 'values',
            'macro_feat', 'episode_ids', 'step_ids',
            'next_task_obs', 'next_worker_loads', 'next_worker_profile', 'next_valid_mask',
        ]}
        for lid in range(num_layers)
    }

    obs = env.reset(with_new_schedule=with_new_schedule)
    done = False
    step_idx = 0

    while not done and step_idx < steps_per_episode:
        actions = {}
        for lid in range(num_layers):
            task_obs = obs[lid]['task_queue']
            worker_loads = obs[lid]['worker_loads']
            profile = obs[lid]['worker_profile']
            value, action, logprob, _ = agents[lid].sample(task_obs, worker_loads, profile)
            actions[lid] = action
            valid_mask = task_obs[:, 3].astype(np.float32)

            buffers_local[lid]['task_obs'].append(task_obs)
            buffers_local[lid]['worker_loads'].append(worker_loads)
            buffers_local[lid]['worker_profile'].append(profile)
            buffers_local[lid]['valid_mask'].append(valid_mask)
            buffers_local[lid]['actions'].append(action)
            buffers_local[lid]['logprobs'].append(logprob)
            buffers_local[lid]['values'].append(value)
            buffers_local[lid]['episode_ids'].append(int(episode_id))
            buffers_local[lid]['step_ids'].append(step_idx)

        raw_macro = build_struct_macro_feature(
            obs=obs, num_layers=num_layers, step_idx=step_idx, max_steps=steps_per_episode
        )
        for lid in range(num_layers):
            buffers_local[lid]['macro_feat'].append(raw_macro)

        obs_next, (_, reward_detail), done, _ = env.step(actions)

        for lid in range(num_layers):
            n_task_obs = obs_next[lid]['task_queue']
            n_worker_loads = obs_next[lid]['worker_loads']
            n_profile = obs_next[lid]['worker_profile']
            n_valid_mask = n_task_obs[:, 3].astype(np.float32)

            buffers_local[lid]['next_task_obs'].append(n_task_obs)
            buffers_local[lid]['next_worker_loads'].append(n_worker_loads)
            buffers_local[lid]['next_worker_profile'].append(n_profile)
            buffers_local[lid]['next_valid_mask'].append(n_valid_mask)

            r = reward_detail['layer_rewards'][lid]['reward']
            buffers_local[lid]['rewards'].append(r)
            buffers_local[lid]['ext_rewards'].append(r)
            buffers_local[lid]['dones'].append(done)

        obs = obs_next
        step_idx += 1

    return buffers_local


def _apply_intrinsic_and_credit_assignment(results, algs, clusterer, num_layers, gamma):
    """
    对本次采样得到的 results（可能包含多个 episode）：
      1) 用 macro_feat 做聚类得到 global IR
      2) 基于每层 int_value 估算重要性权重
      3) 将 global IR 按权重分配到每层，并写回 results[lid]['int_rewards'] 和 results[lid]['rewards']
    注意：这里的权重是“本 batch 全局”的（与原训练代码一致）。
    """
    # concat macro seq
    macro_list = []
    lens = []
    for res in results:
        l = len(res[0]['macro_feat'])
        lens.append(l)
        macro_list.extend(res[0]['macro_feat'])

    if len(macro_list) == 0:
        return

    macro_seq = np.array(macro_list, dtype=np.float32)  # [T_total, macro_feat_dim]
    z_seq = algs[0].encode_macro_for_cluster(macro_seq)  # [T_total, repr_dim]
    r_int_seq = clusterer.update_and_compute_intrinsic(z_seq)  # [T_total]

    T_total = int(len(r_int_seq))
    assert T_total == sum(lens), f"T_total={T_total}, sum(lens)={sum(lens)} mismatch"

    # 1) per-layer importance based on int_value TD deltas
    layer_importances = []
    with torch.no_grad():
        for lid in range(num_layers):
            task_list = sum([res[lid]['task_obs'] for res in results], [])
            load_list = sum([res[lid]['worker_loads'] for res in results], [])
            prof_list = sum([res[lid]['worker_profile'] for res in results], [])
            mask_list = sum([res[lid]['valid_mask'] for res in results], [])
            done_list = sum([res[lid]['dones'] for res in results], [])

            task_arr = np.array(task_list, dtype=np.float32)
            load_arr = np.array(load_list, dtype=np.float32)
            prof_arr = np.array(prof_list, dtype=np.float32)
            mask_arr = np.array(mask_list, dtype=np.float32)
            done_arr = np.array(done_list, dtype=np.float32)

            task_t = torch.tensor(task_arr)
            load_t = torch.tensor(load_arr)
            prof_t = torch.tensor(prof_arr)
            mask_t = torch.tensor(mask_arr)

            v_int = algs[lid].int_value(task_t, load_t, prof_t, mask_t)  # [T_total]
            v_int = v_int.detach().cpu().numpy().astype(np.float32)

            deltas = []
            for t in range(T_total):
                v_now = v_int[t]
                v_next = v_int[t + 1] if t < T_total - 1 else 0.0
                done_flag = done_arr[t]
                delta = float(r_int_seq[t] + gamma * v_next * (1.0 - done_flag) - v_now)
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

    # 2) distribute global IR into each episode result
    offset = 0
    for res, l in zip(results, lens):
        for lid in range(num_layers):
            w_l = float(weights[lid])
            # ensure empty
            if len(res[lid]['int_rewards']) != 0:
                res[lid]['int_rewards'].clear()
            for t in range(l):
                r_int = float(r_int_seq[offset + t] * w_l)
                res[lid]['int_rewards'].append(r_int)
                res[lid]['rewards'][t] = float(res[lid]['rewards'][t] + r_int)
        offset += l


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

    # offline dirs
    offline_dir = Path(args.offline_data_root) / dire / "suboptimal"
    ckpt_dir = Path(args.ckpt_root) / dire
    _ensure_dir(offline_dir)
    _ensure_dir(ckpt_dir)

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

    # 初始化结构聚类器
    repr_dim = getattr(algs[0], "repr_dim", 64)
    clusterer = CrescentClusterer(
        repr_dim=repr_dim,
        num_clusters=64,
        ema_momentum=0.99,
        count_smoothing=0.1,
        intrinsic_coef=1.0,
        device=device,
    )

    pool = mp.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(dire, env_config_path, schedule_path, worker_config_path,
                  alg_name, hidden_dim, macro_feat_dim)
    ) if args.num_workers > 1 else None

    # ===== Best model tracking (extrinsic return) =====
    best_ext_return = -1e18
    best_episode_id = -1
    best_outer_idx = -1
    best_worker_id = -1

    # ===== Training loop =====
    num_outer = int(num_episodes / args.num_workers) + 1
    for outer_idx in range(num_outer):
        with_new_schedule = True
        base_episode_id = outer_idx * (args.num_workers if args.num_workers > 1 else 1)

        # 1) sample episodes (results: list of episode buffers)
        if args.num_workers == 1:
            results = [_rollout_episode_main(
                env=env,
                agents=agents,
                num_layers=num_layers,
                steps_per_episode=steps_per_episode,
                with_new_schedule=with_new_schedule,
                episode_id=base_episode_id
            )]
        else:
            policy_states = _snapshot_policy_states(algs)
            seeds = np.random.randint(0, 2 ** 31 - 1, size=args.num_workers).tolist()

            results = pool.starmap(
                _episode_worker,
                [(policy_states, with_new_schedule, int(seeds[wid]), base_episode_id, wid)
                 for wid in range(args.num_workers)]
            )

        # 2) intrinsic + credit assignment (write back into results)
        _apply_intrinsic_and_credit_assignment(
            results=results,
            algs=algs,
            clusterer=clusterer,
            num_layers=num_layers,
            gamma=gamma
        )

        # 3) best model saving (use extrinsic return)
        batch_best_ret = -1e18
        batch_best_ep = -1
        batch_best_wid = -1
        # results 的顺序就是 worker_id（starmap 按 wid 传参的顺序返回）
        for wid, res in enumerate(results):
            global_ep_id = base_episode_id + wid if args.num_workers > 1 else base_episode_id
            ext_ret = _episode_ext_return(res, num_layers)
            if ext_ret > batch_best_ret:
                batch_best_ret = ext_ret
                batch_best_ep = global_ep_id
                batch_best_wid = wid

        if batch_best_ret > best_ext_return:
            best_ext_return = float(batch_best_ret)
            best_episode_id = int(batch_best_ep)
            best_outer_idx = int(outer_idx)
            best_worker_id = int(batch_best_wid)
            _save_best_model(
                algs_dict=algs,
                ckpt_dir=ckpt_dir,
                best_ext_return=best_ext_return,
                best_episode_id=best_episode_id,
                outer_idx=best_outer_idx,
                worker_id=best_worker_id
            )

        # 4) offline data saving: save ONE episode every N episodes (1-indexed)
        if args.offline_save_interval and args.offline_save_interval > 0:
            for wid, res in enumerate(results):
                global_ep_id = base_episode_id + wid if args.num_workers > 1 else base_episode_id
                if global_ep_id >= 0 and ((global_ep_id + 1) % args.offline_save_interval == 0):
                    _save_offline_episode_npz(
                        buffers_local=res,
                        out_dir=offline_dir,
                        num_layers=num_layers,
                        global_episode_id=global_ep_id,
                        tag="traj"
                    )

        # 5) merge results -> buffers (for training update)
        for lid in range(num_layers):
            for k in buffers[lid]:
                buffers[lid][k].clear()

        for res in results:
            for lid in range(num_layers):
                for k in buffers[lid]:
                    buffers[lid][k].extend(res[lid][k])

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

                    current_steps = outer_idx * steps_per_episode * (
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

        # eval
        if outer_idx % eval_interval == 0:
            evaluate_policy(
                agents, eval_env, eval_episodes, writer,
                outer_idx * steps_per_episode * args.num_workers
            )

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    log_dir = f'../logs/{alg_name}/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    mp.set_start_method("spawn", force=True)
    main()
