# run/run_mibrew.py
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_td3bc import Actor, Critic, load_offline_data_per_layer, build_state_from_env_obs, soft_update

from envs import IndustrialChain
from envs.env import MultiplexEnv

from utils.utils_mibrew import (
    RewardHeuristicQInferencer,
    VariBADQInferencer,
    build_q_cache,
    attach_qcache_to_datasets,
    compute_regime_weights,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *a, **kw): pass

        def add_scalar(self, *a, **kw): pass

        def close(self): pass


@torch.no_grad()
def evaluate_policy_mibrew(
        actors,
        eval_env,
        num_layers: int,
        state_dims: dict,
        action_shapes: dict,
        q_inferencer,
        q_lid: int,
        eval_episodes: int,
        device: torch.device,
):
    total_returns = []

    for _ in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_ret = 0.0

        # 在线 q：如果是 VariBAD inferencer，就 reset；否则用 reward-heuristic 的 reward history
        if isinstance(q_inferencer, VariBADQInferencer):
            q_inferencer.reset_online()
        rew_hist = []

        while not done:
            # 计算 q_t（全局，用某一层 obs 估）
            if isinstance(q_inferencer, VariBADQInferencer):
                q_t = q_inferencer.compute_q_from_env_obs(obs[q_lid])
            else:
                # fallback heuristic：用总回报历史算 q
                if len(rew_hist) == 0:
                    q_t = 0.5
                else:
                    w = rew_hist[-q_inferencer.window:]
                    m = float(np.mean(np.abs(w)))
                    q_t = 1.0 / (1.0 + np.exp((m - q_inferencer.thr) / max(q_inferencer.temp, 1e-8)))
                q_t = float(np.clip(q_t, 1e-6, 1.0 - 1e-6))

            q_tensor = torch.tensor([[q_t]], dtype=torch.float32, device=device)

            actions_env = {}
            action_flat_cache = {}

            for lid in range(num_layers):
                state_np = build_state_from_env_obs(obs[lid], state_dims[lid])  # [1, D]
                state_t = torch.tensor(state_np, dtype=torch.float32, device=device)
                state_aug = torch.cat([state_t, q_tensor], dim=1)

                a_flat = actors[lid](state_aug)  # [1, A]
                a_np_flat = a_flat[0].detach().cpu().numpy().astype(np.float32)

                actions_env[lid] = a_np_flat.reshape(action_shapes[lid])
                action_flat_cache[lid] = a_np_flat

            next_obs, (_, reward_detail), done, _ = eval_env.step(actions_env)

            # 统计回报（你环境 reward_detail 的结构沿用 run_varibad / run_td3bc）
            step_ret = 0.0
            for lid in range(num_layers):
                step_ret += float(reward_detail["layer_rewards"][lid]["reward"])
            ep_ret += step_ret
            rew_hist.append(step_ret)

            # 喂回 belief（用 q_lid 的动作/回报，避免分布不一致）
            if isinstance(q_inferencer, VariBADQInferencer):
                r_l = float(reward_detail["layer_rewards"][q_lid]["reward"])
                q_inferencer.set_prev_feedback(action_flat_cache[q_lid], r_l, float(done))

            obs = next_obs

        total_returns.append(ep_ret)

    return float(np.mean(total_returns)), float(np.std(total_returns))


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = args.seed if args.seed else np.random.randint(0, 10000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dire = args.dire
    dataset_tag = args.dataset

    env_config_path = f"../configs/{dire}/env_config.json"
    eval_schedule_path = f"../configs/{dire}/eval_schedule.json"
    worker_config_path = f"../configs/{dire}/worker_config.json"

    eval_env = MultiplexEnv(env_config_path, schedule_load_path=eval_schedule_path,
                            worker_config_load_path=worker_config_path)
    eval_env.chain = IndustrialChain(eval_env.worker_config)

    num_layers = args.num_layers

    # ---------- 离线数据（复用 TD3BC loader） ----------
    root_dir = Path(args.offline_data_root) / dire / dataset_tag
    datasets, state_dims, action_shapes = load_offline_data_per_layer(root_dir, num_layers, device=device)

    # ---------- q inferencer（优先用 VariBAD ckpt，否则 heuristic） ----------
    if args.bad_ckpt:
        # 从第一条 npz 推断 obs_dim_raw / action_dim（不依赖 env space）
        f0 = sorted(root_dir.glob("*.npz"))[0]
        d0 = np.load(str(f0), allow_pickle=True)
        T0 = int(d0["T"])
        p = f"l{args.q_obs_lid}_"
        task_obs0 = d0[p + "task_obs"]
        worker_loads0 = d0[p + "worker_loads"]
        worker_profile0 = d0[p + "worker_profile"]
        num_pad = task_obs0.shape[1]
        task_dim = task_obs0.shape[2]
        n_worker = worker_loads0.shape[1]
        load_dim = worker_loads0.shape[2]
        profile_dim_flat = int(np.prod(worker_profile0.shape[1:]))

        obs_dim_raw = num_pad * task_dim + n_worker * load_dim + profile_dim_flat + num_pad

        pa = f"l{args.q_action_lid}_"
        actions0 = d0[pa + "actions"]
        action_dim = int(np.prod(actions0.shape[1:]))

        bad_ckpt = f"{args.bad_ckpt}/{args.dire}/bad.pt"

        q_infer = VariBADQInferencer(
            ckpt_path=bad_ckpt,
            obs_dim_raw=obs_dim_raw,
            action_dim=action_dim,
            obs_embed_dim=args.bad_obs_embed_dim,
            belief_hidden=args.bad_belief_hidden,
            z_dim=args.bad_z_dim,
            decoder_hidden=args.bad_decoder_hidden,
            policy_hidden=args.bad_policy_hidden,
            value_hidden=args.bad_value_hidden,
            device=str(device),
            valid_index=args.valid_index,
            thr=args.q_thr,
            temp=args.q_temp,
            use_abs=True,
        )
    else:
        q_infer = RewardHeuristicQInferencer(window=args.heur_window, thr=args.q_thr, temp=args.q_temp)

    # ---------- q-cache（和 TD3BC 数据顺序严格对齐） ----------
    cache_dir = root_dir / "_mibrew_cache"
    qcache_path = Path(args.qcache_path) if args.qcache_path else (cache_dir / "mibrew_qcache.npz")

    qcache = build_q_cache(
        root_dir=root_dir,
        num_layers=num_layers,
        inferencer=q_infer,
        out_path=qcache_path,
        q_obs_lid=args.q_obs_lid,
        q_action_lid=args.q_action_lid,
        q_reward_source=args.q_reward_source,
        valid_index=args.valid_index,
        force_rebuild=args.force_rebuild_qcache,
    )
    print(f"[qcache] path={qcache_path}, p_s={qcache.p_s:.4f}, N={qcache.q.shape[0]}")

    datasets_q = attach_qcache_to_datasets(datasets, qcache, device=device)

    # ---------- actor/critic：输入维度 +1（拼接 q） ----------
    actors, actor_targets = {}, {}
    critics, critic_targets = {}, {}
    actor_opt, critic_opt = {}, {}

    max_action = 1.0

    for lid in range(num_layers):
        state_dim_aug = int(state_dims[lid]) + 1
        action_dim = int(np.prod(action_shapes[lid]))

        actor = Actor(state_dim_aug, action_dim, hidden_dim=args.hidden_dim, max_action=max_action).to(device)
        actor_t = Actor(state_dim_aug, action_dim, hidden_dim=args.hidden_dim, max_action=max_action).to(device)
        actor_t.load_state_dict(actor.state_dict())

        critic = Critic(state_dim_aug, action_dim, hidden_dim=args.hidden_dim).to(device)
        critic_t = Critic(state_dim_aug, action_dim, hidden_dim=args.hidden_dim).to(device)
        critic_t.load_state_dict(critic.state_dict())

        actors[lid] = actor
        actor_targets[lid] = actor_t
        critics[lid] = critic
        critic_targets[lid] = critic_t

        actor_opt[lid] = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic_opt[lid] = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    file_name = f"{dataset_tag}_" + time.strftime("%m%d-%H%M%S")
    log_dir = Path(args.log_dir) / dire / file_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    best_ret = -1e18
    best_state = None
    t0 = time.time()

    for it in range(args.train_steps + 1):
        # eval
        if it % args.eval_interval == 0:
            avg_ret, std_ret = evaluate_policy_mibrew(
                actors=actors,
                eval_env=eval_env,
                num_layers=num_layers,
                state_dims=state_dims,
                action_shapes=action_shapes,
                q_inferencer=q_infer,
                q_lid=args.q_obs_lid,
                eval_episodes=args.eval_episodes,
                device=device,
            )
            writer.add_scalar("eval/avg_return", avg_ret, it)
            writer.add_scalar("eval/std_return", std_ret, it)
            print(f"[eval] it={it} avg_return={avg_ret:.3f} std={std_ret:.3f}")

            if avg_ret > best_ret:
                best_ret = avg_ret
                best_state = {
                    "actors": {lid: actors[lid].state_dict() for lid in range(num_layers)},
                    "critics": {lid: critics[lid].state_dict() for lid in range(num_layers)},
                }

        critic_losses = []
        actor_losses = []
        lambdas = []

        # critic update (每层独立)
        for lid in range(num_layers):
            ds = datasets_q[lid]
            s, a, r, s2, d, q, qn = ds.sample(args.batch_size)

            s_aug = torch.cat([s, q], dim=1)
            s2_aug = torch.cat([s2, qn], dim=1)

            w = compute_regime_weights(q, p_s=qcache.p_s, eps=args.weight_eps)  # [B,1]

            with torch.no_grad():
                noise = (torch.randn_like(a) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                a2 = (actor_targets[lid](s2_aug) + noise).clamp(-max_action, max_action)

                tq1, tq2 = critic_targets[lid](s2_aug, a2)
                tq = torch.min(tq1, tq2)
                target = r + (1.0 - d) * args.gamma * tq

            cq1, cq2 = critics[lid](s_aug, a)
            td1 = cq1 - target
            td2 = cq2 - target
            critic_loss = (w * td1.pow(2)).mean() + (w * td2.pow(2)).mean()

            critic_opt[lid].zero_grad()
            critic_loss.backward()
            critic_opt[lid].step()

            critic_losses.append(float(critic_loss.item()))

        # delayed actor update
        if it % args.policy_delay == 0:
            for lid in range(num_layers):
                ds = datasets_q[lid]
                s, a, _, _, _, q, _ = ds.sample(args.batch_size)
                s_aug = torch.cat([s, q], dim=1)

                pi = actors[lid](s_aug)
                q_pi = critics[lid].q1_only(s_aug, pi)

                lambda_coef = args.alpha / (q_pi.abs().mean().detach() + 1e-6)
                bc_loss = F.mse_loss(pi, a)
                actor_loss = (-lambda_coef * q_pi.mean() + bc_loss)

                actor_opt[lid].zero_grad()
                actor_loss.backward()
                actor_opt[lid].step()

                soft_update(actor_targets[lid], actors[lid], args.tau)
                soft_update(critic_targets[lid], critics[lid], args.tau)

                actor_losses.append(float(actor_loss.item()))
                lambdas.append(float(lambda_coef.item()))

        # log
        if it % args.log_interval == 0:
            if critic_losses:
                writer.add_scalar("train/critic_loss", float(np.mean(critic_losses)), it)
            if actor_losses:
                writer.add_scalar("train/actor_loss", float(np.mean(actor_losses)), it)
            if lambdas:
                writer.add_scalar("train/lambda", float(np.mean(lambdas)), it)

            mins = (time.time() - t0) / 60.0
            print(
                f"[it {it:07d}] critic={np.mean(critic_losses):.4f} "
                f"actor={np.mean(actor_losses) if actor_losses else 0:.4f} "
                f"best={best_ret:.3f} elapsed={mins:.1f}min"
            )

    # save best
    ckpt_dir = Path(args.ckpt_dir) / args.dire
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"mibrew_{dataset_tag}_seed{seed}.pt"

    torch.save(
        {
            "best_return": float(best_ret),
            "best_state": best_state,
            "qcache_path": str(qcache_path),
            "qcache_p_s": float(qcache.p_s),
            "config": vars(args),
        },
        str(ckpt_path),
    )
    print(f"[done] best_return={best_ret:.3f}, saved: {ckpt_path}")
    writer.close()


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dire", type=str, default="standard")
    p.add_argument("--dataset", type=str, default="mixed")
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--offline_data_root", type=str, default="../offline_data/crescent")

    # TD3BC hparams
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy_noise", type=float, default=0.2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--policy_delay", type=int, default=2)
    p.add_argument("--alpha", type=float, default=2.5)

    p.add_argument("--train_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=256)

    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=1000)

    p.add_argument("--log_dir", type=str, default="../logs/mibrew")
    p.add_argument("--ckpt_dir", type=str, default="../checkpoints/mibrew")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # q-cache
    p.add_argument("--qcache_path", type=str, default="")
    p.add_argument("--force_rebuild_qcache", action="store_true")

    # q 来源（选哪一层的 obs/action/reward 来推 q）
    p.add_argument("--q_obs_lid", type=int, default=0)
    p.add_argument("--q_action_lid", type=int, default=0)
    p.add_argument("--q_reward_source", type=str, default="l0", choices=["l0", "l1", "l2", "sum"])
    p.add_argument("--valid_index", type=int, default=3)

    # regime-balanced weight
    p.add_argument("--weight_eps", type=float, default=1e-3)

    # q 的 logistic 超参
    p.add_argument("--q_thr", type=float, default=1e-3)
    p.add_argument("--q_temp", type=float, default=5e-3)

    # fallback heuristic window
    p.add_argument("--heur_window", type=int, default=10)

    # BAD(VariBAD) ckpt：为空则用 heuristic
    p.add_argument("--bad_ckpt", type=str, default="../checkpoints/bad")

    # 下面这些必须和你训练 VariBAD 时的模型结构一致，否则 load_state_dict 会报错
    p.add_argument("--bad_obs_embed_dim", type=int, default=256)
    p.add_argument("--bad_belief_hidden", type=int, default=128)
    p.add_argument("--bad_z_dim", type=int, default=64)
    p.add_argument("--bad_decoder_hidden", type=int, default=256)
    p.add_argument("--bad_policy_hidden", type=int, default=256)
    p.add_argument("--bad_value_hidden", type=int, default=256)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
