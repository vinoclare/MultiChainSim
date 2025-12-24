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
from algs.informed_switch import IS
from agents.mappo_agent import IndustrialAgent
from utils.utils import RunningMeanStd
from utils.utils_is import InformedSwitchConfig, InformedSwitchController

# ===== Load configurations =====
parser = argparse.ArgumentParser()
parser.add_argument("--dire", type=str, default="standard")
parser.add_argument("--alg_name", type=str, default="is")  # for log dir only
parser.add_argument("--num_workers", type=int, default=10, help="Parallel env workers for sampling")

# ---- Informed Switching hyper-params (no extra config file) ----
parser.add_argument("--is_k", type=int, default=10, help="k-step window for value promise discrepancy")
parser.add_argument("--is_explore_len", type=int, default=20, help="explore segment length")
parser.add_argument("--is_target_rate", type=float, default=0.01, help="homeostatic target trigger rate")
parser.add_argument("--is_theta_lr", type=float, default=0.01, help="homeostatic threshold lr")
parser.add_argument("--is_init_theta", type=float, default=1.0, help="initial threshold theta")
parser.add_argument("--is_metric", type=str, default="abs", choices=["abs", "sq"], help="discrepancy metric")
parser.add_argument("--is_start_mode", type=int, default=0, choices=[0, 1], help="0 exploit, 1 explore")
parser.add_argument("--is_std_scale", type=float, default=2.0, help="explore std scale (>1 means noisier)")

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

# ===== Globals for subprocess sampling =====
g_env = None
g_agents = None
g_algs = None
g_switchers = None
g_num_layers = None


def _snapshot_policy_states(algs_dict):
    """Make a pure-tensor snapshot state_dict for each layer policy."""
    return {lid: algs_dict[lid].model.state_dict() for lid in algs_dict}


def _pad_inputs(agent: IndustrialAgent, task_obs_np, worker_loads_np, profile_np):
    """
    Use IndustrialAgent's padding + mask logic (same as baseline sampling).
    NOTE: sampling mask uses last column (agent implementation), consistent with run_mappo baseline.
    """
    task_obs = agent._pad_task_obs(task_obs_np)
    worker_loads = agent._pad_worker_obs(worker_loads_np)
    profile = agent._pad_worker_profile(profile_np)
    valid_mask = agent._get_valid_mask(task_obs)
    return task_obs, worker_loads, profile, valid_mask


def _episode_worker(policy_states, with_new_schedule, seed):
    import numpy as np, torch, random

    global g_env, g_agents, g_algs, g_switchers, g_num_layers

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load policy snapshot (read-only)
    for lid in range(g_num_layers):
        g_algs[lid].model.load_state_dict(policy_states[lid])
        # reset episode state (threshold persists)
        g_switchers[lid].reset_episode()

    # Local buffers (same keys as run_mappo, plus modes)
    buffers_local = {
        lid: {k: [] for k in [
            'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
            'actions', 'logprobs', 'rewards', 'dones', 'values', 'modes'
        ]}
        for lid in range(g_num_layers)
    }

    obs = g_env.reset(with_new_schedule=with_new_schedule)
    done = False
    prev_done = False
    prev_rewards = {lid: None for lid in range(g_num_layers)}
    step_idx = 0

    while not done:
        actions = {}

        # decide mode -> sample action for each layer
        for lid in range(g_num_layers):
            task_obs_np = obs[lid]['task_queue']
            worker_loads_np = obs[lid]['worker_loads']
            profile_np = obs[lid]['worker_profile']

            # tensors for value + action sampling
            task_t, load_t, prof_t, valid_t = _pad_inputs(g_agents[lid], task_obs_np, worker_loads_np, profile_np)

            # value for informed switching (critic doesn't depend on explore/exploit)
            v_t = g_algs[lid].value(task_t, load_t, prof_t, valid_t)  # (1,)
            v_scalar = float(v_t[0].detach().cpu().numpy())

            if step_idx == 0:
                # bootstrap tracker with V(s0); keep start_mode for step0
                g_switchers[lid].tracker.observe(v_scalar, None, False)
                mode = g_switchers[lid].current_mode()
            else:
                mode = g_switchers[lid].observe(v_scalar, prev_rewards[lid], prev_done)

            # sample action/logprob under selected mode
            _, a_t, logp_t, _ = g_algs[lid].sample(task_t, load_t, prof_t, valid_t, mode=mode)
            act_np = a_t[0].detach().cpu().numpy()
            logp_np = float(logp_t[0].detach().cpu().numpy())

            actions[lid] = act_np

            # store buffers (match run_mappo)
            valid_mask_np = task_obs_np[:, 3].astype(np.float32)  # keep baseline's stored mask
            buffers_local[lid]['task_obs'].append(task_obs_np)
            buffers_local[lid]['worker_loads'].append(worker_loads_np)
            buffers_local[lid]['worker_profile'].append(profile_np)
            buffers_local[lid]['valid_mask'].append(valid_mask_np)
            buffers_local[lid]['actions'].append(act_np)
            buffers_local[lid]['logprobs'].append(logp_np)
            buffers_local[lid]['values'].append(v_scalar)
            buffers_local[lid]['modes'].append(int(mode))

        # env step
        obs, (_, reward_detail), done, _ = g_env.step(actions)

        for lid in range(g_num_layers):
            r = reward_detail['layer_rewards'][lid]['reward']
            buffers_local[lid]['rewards'].append(r)
            buffers_local[lid]['dones'].append(done)
            prev_rewards[lid] = r

        prev_done = done
        step_idx += 1

    return buffers_local


def _init_worker(dire,
                 env_config_path,
                 schedule_path,
                 worker_config_path,
                 hidden_dim,
                 gamma,
                 is_k,
                 is_explore_len,
                 is_target_rate,
                 is_theta_lr,
                 is_init_theta,
                 is_metric,
                 is_start_mode,
                 is_std_scale):
    """Called once per subprocess: create persistent env / agents / switchers (sampling only)."""
    global g_env, g_agents, g_algs, g_switchers, g_num_layers

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
    n_worker, _ = act_space.shape
    n_task_types = len(env_cfg["task_types"])
    profile_dim = 2 * n_task_types
    num_pad_tasks = g_env.num_pad_tasks

    # switch cfg (threshold persists across episodes within the worker process)
    sw_cfg = InformedSwitchConfig(
        k=int(is_k),
        gamma=float(gamma),
        explore_len=int(is_explore_len),
        init_theta=float(is_init_theta),
        target_rate=float(is_target_rate),
        theta_lr=float(is_theta_lr),
        metric=str(is_metric),
        start_mode=int(is_start_mode),
    )

    g_agents, g_algs, g_switchers = {}, {}, {}
    for lid in range(g_num_layers):
        model = MAPPOIndustrialModel(
            task_input_dim=obs_space['task_queue'].shape[1],
            worker_load_input_dim=obs_space['worker_loads'].shape[1],
            worker_profile_input_dim=profile_dim,
            n_worker=n_worker,
            num_pad_tasks=num_pad_tasks,
            hidden_dim=hidden_dim
        )
        alg = IS(
            model,
            explore_std_scale=float(is_std_scale),
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.0,
            initial_lr=3e-4,
            max_grad_norm=0.5,
            writer=None,
            global_step_ref=[0],
            total_training_steps=1,
            device="cpu",
        )
        g_algs[lid] = alg
        g_agents[lid] = IndustrialAgent(alg, "mappo", device="cpu", num_pad_tasks=num_pad_tasks)
        g_switchers[lid] = InformedSwitchController(sw_cfg)


def process_obs(raw_obs, lid):
    obs = raw_obs[lid]
    return obs['task_queue'], obs['worker_loads'], obs['worker_profile']


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
                # default mode=0 (exploit) because agent.sample calls alg.sample without mode
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

    # ===== Init per-layer models and agents =====
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    n_worker, _ = act_space.shape
    n_task_types = len(env_config["task_types"])
    profile_dim = 2 * n_task_types

    # IS switchers in main process (threshold persists across episodes)
    sw_cfg = InformedSwitchConfig(
        k=int(args.is_k),
        gamma=float(gamma),
        explore_len=int(args.is_explore_len),
        init_theta=float(args.is_init_theta),
        target_rate=float(args.is_target_rate),
        theta_lr=float(args.is_theta_lr),
        metric=str(args.is_metric),
        start_mode=int(args.is_start_mode),
    )
    switchers = {lid: InformedSwitchController(sw_cfg) for lid in range(num_layers)}

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

        alg = IS(
            model,
            explore_std_scale=float(args.is_std_scale),
            clip_param=ppo_config["clip_param"],
            value_loss_coef=ppo_config["value_loss_coef"],
            entropy_coef=ppo_config["entropy_coef"],
            initial_lr=ppo_config["initial_lr"],
            max_grad_norm=ppo_config["max_grad_norm"],
            writer=writer,
            global_step_ref=[0],
            total_training_steps=ppo_config["num_episodes"] * env_config["max_steps"],
            device=device
        )

        agents[lid] = IndustrialAgent(alg, "mappo", device, env.num_pad_tasks)
        algs[lid] = alg
        return_rms[lid] = RunningMeanStd()
        buffers[lid] = {k: [] for k in [
            'task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
            'actions', 'logprobs', 'rewards', 'dones', 'values', 'modes'
        ]}

    pool = mp.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(
            dire, env_config_path, schedule_path, worker_config_path,
            hidden_dim, gamma,
            args.is_k, args.is_explore_len, args.is_target_rate, args.is_theta_lr,
            args.is_init_theta, args.is_metric, args.is_start_mode, args.is_std_scale
        )
    ) if args.num_workers > 1 else None

    # ===== Training loop =====
    for episode in range(int(num_episodes / args.num_workers) + 1):
        if args.num_workers == 1:
            for lid in range(num_layers):
                switchers[lid].reset_episode()

            if episode % reset_schedule_interval == 0:
                obs = env.reset(with_new_schedule=True)
            else:
                obs = env.reset()

            prev_done = False
            prev_rewards = {lid: None for lid in range(num_layers)}

            for step in range(steps_per_episode):
                actions = {}

                for lid in range(num_layers):
                    task_obs_np, worker_loads_np, profile_np = process_obs(obs, lid)

                    # tensors for value + action sampling (use agent's padding/mask)
                    task_t, load_t, prof_t, valid_t = _pad_inputs(agents[lid], task_obs_np, worker_loads_np, profile_np)

                    v_t = algs[lid].value(task_t, load_t, prof_t, valid_t)
                    v_scalar = float(v_t[0].detach().cpu().numpy())

                    if step == 0:
                        switchers[lid].tracker.observe(v_scalar, None, False)
                        mode_sel = switchers[lid].current_mode()
                    else:
                        mode_sel = switchers[lid].observe(v_scalar, prev_rewards[lid], prev_done)

                    _, a_t, logp_t, _ = algs[lid].sample(task_t, load_t, prof_t, valid_t, mode=mode_sel)
                    act_np = a_t[0].detach().cpu().numpy()
                    logp_np = float(logp_t[0].detach().cpu().numpy())

                    actions[lid] = act_np
                    valid_mask_np = task_obs_np[:, 3].astype(np.float32)

                    buffers[lid]['task_obs'].append(task_obs_np)
                    buffers[lid]['worker_loads'].append(worker_loads_np)
                    buffers[lid]['worker_profile'].append(profile_np)
                    buffers[lid]['valid_mask'].append(valid_mask_np)
                    buffers[lid]['actions'].append(act_np)
                    buffers[lid]['logprobs'].append(logp_np)
                    buffers[lid]['values'].append(v_scalar)
                    buffers[lid]['modes'].append(int(mode_sel))

                obs, (_, reward_detail), done, _ = env.step(actions)

                for lid in range(num_layers):
                    r = reward_detail['layer_rewards'][lid]['reward']
                    buffers[lid]['rewards'].append(r)
                    buffers[lid]['dones'].append(done)
                    prev_rewards[lid] = r

                prev_done = done
                if done:
                    break

        else:
            # ---- distributed sampling ----
            policy_states = _snapshot_policy_states(algs)
            with_new_schedule = (episode % reset_schedule_interval == 0)
            seeds = np.random.randint(0, 2 ** 31 - 1, size=args.num_workers).tolist()

            results = pool.starmap(
                _episode_worker,
                [(policy_states, with_new_schedule, int(seeds[wid]))
                 for wid in range(args.num_workers)]
            )

            for lid in range(num_layers):
                for k in buffers[lid]:
                    buffers[lid][k].clear()

                def _cat_list(key):
                    return sum([res[lid][key] for res in results], [])

                for key in ['task_obs', 'worker_loads', 'worker_profile', 'valid_mask',
                            'actions', 'logprobs', 'values', 'rewards', 'dones', 'modes']:
                    buffers[lid][key].extend(_cat_list(key))

        # ===== Learn each layer independently (MAPPO-style GAE) =====
        for lid in range(num_layers):
            advs = []
            vals = buffers[lid]['values'] + [0.0]
            gae = 0.0
            for t in reversed(range(len(buffers[lid]['rewards']))):
                delta = buffers[lid]['rewards'][t] + gamma * vals[t + 1] * (1 - buffers[lid]['dones'][t]) - vals[t]
                gae = delta + gamma * lam * (1 - buffers[lid]['dones'][t]) * gae
                advs.insert(0, gae)

            rets = [a + v for a, v in zip(advs, buffers[lid]['values'])]

            if ppo_config["return_normalization"]:
                return_rms[lid].update(np.array(rets, dtype=np.float32))
                rets = return_rms[lid].normalize(np.array(rets, dtype=np.float32))

            dataset = list(zip(
                buffers[lid]['task_obs'],
                buffers[lid]['worker_loads'],
                buffers[lid]['worker_profile'],
                buffers[lid]['valid_mask'],
                buffers[lid]['actions'],
                buffers[lid]['values'],
                rets,
                buffers[lid]['logprobs'],
                advs,
                buffers[lid]['modes'],
            ))

            for _ in range(update_epochs):
                random.shuffle(dataset)
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i + batch_size]
                    (task_batch, load_batch, prof_batch, mask_batch,
                     act_batch, val_batch, ret_batch, logp_batch, adv_batch, mode_batch) = zip(*batch)

                    algs[lid].learn(
                        torch.tensor(np.array(task_batch, dtype=np.float32)),
                        torch.tensor(np.array(load_batch, dtype=np.float32)),
                        torch.tensor(np.array(prof_batch, dtype=np.float32)),
                        torch.tensor(np.array(mask_batch, dtype=np.float32)),
                        torch.tensor(np.array(act_batch, dtype=np.float32)),
                        torch.tensor(np.array(val_batch, dtype=np.float32)),
                        torch.tensor(np.array(ret_batch, dtype=np.float32)),
                        torch.tensor(np.array(logp_batch, dtype=np.float32)),
                        torch.tensor(np.array(adv_batch, dtype=np.float32)),
                        episode * steps_per_episode * (args.num_workers if args.num_workers > 1 else 1),
                        modes=torch.tensor(np.array(mode_batch, dtype=np.float32)),
                    )

            # optional: log explore ratio per layer (cheap debugging signal)
            if len(buffers[lid]['modes']) > 0:
                explore_ratio = float(np.mean(np.array(buffers[lid]['modes'], dtype=np.float32)))
                writer.add_scalar(f"is/layer{lid}_explore_ratio", explore_ratio,
                                  episode * steps_per_episode * args.num_workers)

        # clear buffers
        for lid in buffers:
            for k in buffers[lid]:
                buffers[lid][k].clear()

        if episode % eval_interval == 0:
            evaluate_policy(agents, eval_env, eval_episodes, writer, episode * steps_per_episode * args.num_workers)

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    log_dir = f'../logs/{alg_name}/{dire}/' + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    mp.set_start_method("spawn", force=True)
    main()
