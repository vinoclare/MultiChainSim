import os
import json
import time
import argparse
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs import IndustrialChain
from envs.env import MultiplexEnv
from agents.hitac_muse_agent import HiTACMuSEAgent


def _is_state_dict_like(obj: Any) -> bool:
    """Heuristic: a PyTorch state_dict is typically a dict[str, Tensor]."""
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    tensor_cnt = 0
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        if isinstance(v, torch.Tensor):
            tensor_cnt += 1
    return tensor_cnt >= max(1, int(0.7 * len(obj)))


def load_best_model(agent: Any, best_model_path: str, device: str) -> None:
    """
    Robustly load a 'best_model.pt'.

    Supported layouts (best-effort):
      1) Direct state_dict (dict[str, Tensor]) -> agent.load_state_dict
      2) {'agent_state_dict': ...} -> agent.load_state_dict
      3) {'state_dict': ...} -> agent.load_state_dict
      4) {'muses_state_dict': ..., 'distillers_state_dict': ..., 'hitac_state_dict': ...} -> load to submodules if present
    """
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"best_model not found: {best_model_path}")

    ckpt = torch.load(best_model_path, map_location=device)

    # Case 1: direct state_dict
    if _is_state_dict_like(ckpt):
        agent.load_state_dict(ckpt, strict=False)
        return

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)} in {best_model_path}")

    # Case 2/3: wrapped state_dict
    for key in ("agent_state_dict", "state_dict"):
        if key in ckpt and _is_state_dict_like(ckpt[key]):
            agent.load_state_dict(ckpt[key], strict=False)
            return

    # Case 4: submodule dicts (best-effort)
    loaded_any = False

    if "muses_state_dict" in ckpt and hasattr(agent, "muses"):
        msd = ckpt["muses_state_dict"]
        try:
            if isinstance(msd, (list, tuple)):
                for lid, sd in enumerate(msd):
                    agent.muses[lid].load_state_dict(sd, strict=False)
            elif isinstance(msd, dict):
                for lid in range(len(agent.muses)):
                    if lid in msd:
                        agent.muses[lid].load_state_dict(msd[lid], strict=False)
                    elif str(lid) in msd:
                        agent.muses[lid].load_state_dict(msd[str(lid)], strict=False)
            loaded_any = True
        except Exception as e:
            print(f"[WARN] failed loading muses_state_dict: {e}")

    if "distillers_state_dict" in ckpt and hasattr(agent, "distillers"):
        dsd = ckpt["distillers_state_dict"]
        try:
            if isinstance(dsd, (list, tuple)):
                for lid, sd in enumerate(dsd):
                    agent.distillers[lid].load_state_dict(sd, strict=False)
            elif isinstance(dsd, dict):
                for lid in range(len(agent.distillers)):
                    if lid in dsd:
                        agent.distillers[lid].load_state_dict(dsd[lid], strict=False)
                    elif str(lid) in dsd:
                        agent.distillers[lid].load_state_dict(dsd[str(lid)], strict=False)
            loaded_any = True
        except Exception as e:
            print(f"[WARN] failed loading distillers_state_dict: {e}")

    if "hitac_state_dict" in ckpt and hasattr(agent, "hitac"):
        try:
            agent.hitac.load_state_dict(ckpt["hitac_state_dict"], strict=False)
            loaded_any = True
        except Exception as e:
            print(f"[WARN] failed loading hitac_state_dict: {e}")

    if not loaded_any:
        raise ValueError(
            "Checkpoint format not recognized. "
            "Expected a state_dict, or keys in {'agent_state_dict','state_dict',"
            "'muses_state_dict','distillers_state_dict','hitac_state_dict'}."
        )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_agent(env_config: Dict[str, Any], algo_config: Dict[str, Any],
                 dire: str, device: str, total_steps: int) -> HiTACMuSEAgent:
    num_layers = env_config["num_layers"]
    num_workers = env_config["workers_per_layer"]
    num_pad_tasks = env_config["num_pad_tasks"]
    task_types = env_config["task_types"]
    n_task_types = len(task_types)

    obs_shapes = []
    for lid in range(num_layers):
        obs_shapes.append({
            "task": 4 + n_task_types,
            "worker_load": 1 + n_task_types,
            "worker_profile": 2 * n_task_types,
            "n_worker": num_workers[lid],
            "num_pad_tasks": num_pad_tasks
        })

    act_spaces = [
        (obs_shapes[lid]["n_worker"], obs_shapes[lid]["num_pad_tasks"])
        for lid in range(num_layers)
    ]

    log_dir = f'../logs/hitac_muse_expert/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    agent = HiTACMuSEAgent(
        muse_cfg=algo_config["muse"],
        hitac_cfg=algo_config["hitac"],
        distill_cfg=algo_config["distill"],
        obs_spaces=obs_shapes,
        act_spaces=act_spaces,
        num_layers=num_layers,
        device=device,
        writer=writer,
        total_training_steps=total_steps
    )
    return agent


def _extract_obs_np(obs: Dict[int, Dict[str, np.ndarray]], lid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    layer_obs = obs[lid]
    task_q = np.asarray(layer_obs["task_queue"], dtype=np.float32)
    w_loads = np.asarray(layer_obs["worker_loads"], dtype=np.float32)
    w_prof = np.asarray(layer_obs["worker_profile"], dtype=np.float32)
    valid_mask = task_q[:, 3].astype(np.float32)  # consistent with your main loop / eval
    return task_q, w_loads, w_prof, valid_mask


@torch.no_grad()
def rollout_one_episode(agent: HiTACMuSEAgent,
                        env: MultiplexEnv,
                        device: str,
                        steps_per_episode: int,
                        alpha: float,
                        beta: float,
                        with_new_schedule: bool) -> Dict[str, Any]:
    num_layers = env.num_layers
    obs = env.reset(with_new_schedule=with_new_schedule)
    done = False

    ep = {"per_layer": {}, "T": 0, "ext_return": 0.0, "fused_return": 0.0}
    for lid in range(num_layers):
        ep["per_layer"][lid] = {
            "task_obs": [], "worker_loads": [], "worker_profile": [], "valid_mask": [],
            "actions": [],
            "reward": [], "cost": [], "utility": [], "assign_bonus": [], "wait_penalty": [],
            "reward_u": [], "reward_c": [], "fused_rewards": [],
            "dones": [],
            "next_task_obs": [], "next_worker_loads": [], "next_worker_profile": [], "next_valid_mask": [],
        }

    for t in range(steps_per_episode):
        actions = {}

        for lid in range(num_layers):
            task_q, w_loads, w_prof, vmask = _extract_obs_np(obs, lid)

            ep["per_layer"][lid]["task_obs"].append(task_q)
            ep["per_layer"][lid]["worker_loads"].append(w_loads)
            ep["per_layer"][lid]["worker_profile"].append(w_prof)
            ep["per_layer"][lid]["valid_mask"].append(vmask)

            task_t = torch.tensor(task_q, dtype=torch.float32, device=device).unsqueeze(0)
            wl_t = torch.tensor(w_loads, dtype=torch.float32, device=device).unsqueeze(0)
            wp_t = torch.tensor(w_prof, dtype=torch.float32, device=device).unsqueeze(0)
            vm_t = torch.tensor(vmask, dtype=torch.float32, device=device).unsqueeze(0)

            obs_dict = {
                "task_obs": task_t,
                "worker_loads": wl_t,
                "worker_profiles": wp_t,
                "valid_mask": vm_t,
            }

            _, _, act_t, _ = agent.main_policy_predict(lid, obs_dict)
            act_np = act_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            actions[lid] = act_np
            ep["per_layer"][lid]["actions"].append(act_np)

        obs_next, (total_reward, reward_detail), done, _ = env.step(actions)

        for lid in range(num_layers):
            r = reward_detail["layer_rewards"][lid]
            rew = float(r["reward"])
            cost = float(r["cost"])
            util = float(r["utility"])
            ab = float(r["assign_bonus"])
            wp = float(r["wait_penalty"])

            reward_u = beta * util + ab
            reward_c = -alpha * cost - wp
            fused = reward_u + reward_c

            pl = ep["per_layer"][lid]
            pl["reward"].append(rew)
            pl["cost"].append(cost)
            pl["utility"].append(util)
            pl["assign_bonus"].append(ab)
            pl["wait_penalty"].append(wp)
            pl["reward_u"].append(reward_u)
            pl["reward_c"].append(reward_c)
            pl["fused_rewards"].append(fused)
            pl["dones"].append(float(done))

            task_q2, w_loads2, w_prof2, vmask2 = _extract_obs_np(obs_next, lid)
            pl["next_task_obs"].append(task_q2)
            pl["next_worker_loads"].append(w_loads2)
            pl["next_worker_profile"].append(w_prof2)
            pl["next_valid_mask"].append(vmask2)

        ep["T"] = t + 1
        ep["ext_return"] += sum(float(reward_detail["layer_rewards"][lid]["reward"]) for lid in range(num_layers))
        ep["fused_return"] += sum(float(ep["per_layer"][lid]["fused_rewards"][-1]) for lid in range(num_layers))

        obs = obs_next
        if done:
            break

    return ep


def save_episode_npz(ep: Dict[str, Any], out_path: str, episode_id: int, dire: str) -> None:
    num_layers = len(ep["per_layer"])
    T = ep["T"]

    data: Dict[str, Any] = {
        "episode_id": np.int32(episode_id),
        "dire": np.array(dire),
        "T": np.int32(T),
        "ext_return": np.float32(ep["ext_return"]),
        "fused_return": np.float32(ep["fused_return"]),
    }

    for lid in range(num_layers):
        pl = ep["per_layer"][lid]

        def _as_np(key: str, dtype=np.float32):
            return np.asarray(pl[key], dtype=dtype)

        data[f"l{lid}_task_obs"] = _as_np("task_obs")
        data[f"l{lid}_worker_loads"] = _as_np("worker_loads")
        data[f"l{lid}_worker_profile"] = _as_np("worker_profile")
        data[f"l{lid}_valid_mask"] = _as_np("valid_mask")

        data[f"l{lid}_actions"] = _as_np("actions")

        data[f"l{lid}_reward"] = _as_np("reward")
        data[f"l{lid}_cost"] = _as_np("cost")
        data[f"l{lid}_utility"] = _as_np("utility")
        data[f"l{lid}_assign_bonus"] = _as_np("assign_bonus")
        data[f"l{lid}_wait_penalty"] = _as_np("wait_penalty")

        data[f"l{lid}_reward_u"] = _as_np("reward_u")
        data[f"l{lid}_reward_c"] = _as_np("reward_c")
        data[f"l{lid}_fused_rewards"] = _as_np("fused_rewards")
        data[f"l{lid}_dones"] = _as_np("dones")

        data[f"l{lid}_next_task_obs"] = _as_np("next_task_obs")
        data[f"l{lid}_next_worker_loads"] = _as_np("next_worker_loads")
        data[f"l{lid}_next_worker_profile"] = _as_np("next_worker_profile")
        data[f"l{lid}_next_valid_mask"] = _as_np("next_valid_mask")

    np.savez_compressed(out_path, **data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dire', type=str, default='standard',
                        help='name of sub-folder under ../configs/ 作为本次实验的配置目录')

    parser.add_argument('--expert_episodes', type=int, default=200)
    parser.add_argument('--with_new_schedule', action='store_true',
                        help='每个episode reset时使用 with_new_schedule=True，增加多样性（但可能更难复现）')

    parser.add_argument('--best_model_path', type=str, default=None,
                        help='best_model.pt 路径；默认 ../checkpoints/hitac_muse/{dire}/best_model.pt')

    parser.add_argument('--offline_data_root', type=str, default='../offline_data/hitac_muse',
                        help='输出目录根；最终保存到 {offline_data_root}/{dire}/expert/')

    parser.add_argument('--schedule', type=str, default='eval', choices=['eval', 'train'],
                        help="用 eval_schedule 还是 train_schedule 来 rollout 专家数据（默认 eval 更稳定）")

    parser.add_argument('--device', type=str, default=None, help='cuda/cpu；默认自动判断')

    args = parser.parse_args()
    dire = args.dire

    env_config_path = f'../configs/{dire}/env_config.json'
    algo_config_path = f'../configs/hitac_muse_config.json'
    train_schedule_path = f"../configs/{dire}/train_schedule.json"
    eval_schedule_path = f"../configs/{dire}/eval_schedule.json"
    worker_config_path = f"../configs/{dire}/worker_config.json"

    with open(env_config_path, 'r', encoding='utf-8') as f:
        env_config = json.load(f)
    with open(algo_config_path, 'r', encoding='utf-8') as f:
        algo_config = json.load(f)

    device = args.device
    if device is None:
        device = algo_config["training"]["device"] if torch.cuda.is_available() else "cpu"

    schedule_path = eval_schedule_path if args.schedule == 'eval' else train_schedule_path

    # build env
    env = MultiplexEnv(env_config_path, schedule_load_path=schedule_path, worker_config_load_path=worker_config_path)
    # keep consistent with training: ensure worker_config + chain exist
    base_env = MultiplexEnv(env_config_path, schedule_load_path=train_schedule_path, worker_config_load_path=worker_config_path)
    env.worker_config = base_env.worker_config
    env.chain = IndustrialChain(env.worker_config)

    steps_per_episode = int(env_config["max_steps"])
    alpha = float(env_config["alpha"])
    beta = float(env_config["beta"])

    agent = _build_agent(env_config, algo_config, dire, device, total_steps=steps_per_episode * max(1, args.expert_episodes))

    best_model_path = args.best_model_path or f'../checkpoints/hitac_muse/{dire}/best_model.pt'
    print(f"[Load] best_model_path = {best_model_path}")
    load_best_model(agent, best_model_path, device)

    if hasattr(agent, "eval"):
        agent.eval()

    out_dir = os.path.join(args.offline_data_root, dire, 'expert')
    _ensure_dir(out_dir)

    print(f"[Collect] schedule={args.schedule}, expert_episodes={args.expert_episodes}, steps_per_episode={steps_per_episode}")
    print(f"[Collect] out_dir={out_dir}")

    for ep_id in range(args.expert_episodes):
        ep = rollout_one_episode(
            agent=agent,
            env=env,
            device=device,
            steps_per_episode=steps_per_episode,
            alpha=alpha,
            beta=beta,
            with_new_schedule=args.with_new_schedule
        )
        out_path = os.path.join(out_dir, f"expert_ep{ep_id:06d}_T{ep['T']:03d}.npz")
        save_episode_npz(ep, out_path, episode_id=ep_id, dire=dire)

        if (ep_id + 1) % 10 == 0:
            print(f"[{ep_id+1}/{args.expert_episodes}] ext_return={ep['ext_return']:.2f}, fused_return={ep['fused_return']:.2f}")

    print("[Done] Expert data collection finished.")


if __name__ == "__main__":
    main()
