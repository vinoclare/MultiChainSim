import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_step_reward_matrix(json_path: str) -> np.ndarray:
    """
    读取 eval_last_step_rewards_*.json
    返回形状 [N, T] 的矩阵：N=eval episodes(比如10), T=max_steps(比如50)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    eps = data.get("step_rewards", [])
    if not eps:
        raise ValueError(f"No 'step_rewards' found in: {json_path}")

    seqs = []
    for ep in eps:
        tr = ep.get("total_reward", None)
        if tr is None:
            continue
        seqs.append(np.array(tr, dtype=np.float32))

    if not seqs:
        raise ValueError(f"No valid 'total_reward' sequences in: {json_path}")

    # 统一长度：裁到最短，避免有的 episode 提前 done
    T = min(len(s) for s in seqs)
    mat = np.stack([s[:T] for s in seqs], axis=0)  # [N, T]
    return mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="step_reward_compare.png", help="输出图片路径")
    parser.add_argument("--title", type=str, default="Per-step Reward (Eval, last 10 episodes)", help="图标题")
    parser.add_argument("--show_std", action="store_true", help="是否绘制标准差阴影")
    parser.add_argument("--vline", type=int, default=25, help="分段切换位置（例如 25 表示 0-24 dense, 25-... sparse）")
    args = parser.parse_args()

    cres = load_step_reward_matrix("eval_last_step_rewards_crescent.json")      # [N, T]
    muse = load_step_reward_matrix("eval_last_step_rewards_hitac_muse.json")    # [N, T]
    T = min(cres.shape[1], muse.shape[1])
    cres = cres[:, :T]
    muse = muse[:, :T]

    x = np.arange(T, dtype=np.int32)
    cres_mean = cres.mean(axis=0)
    muse_mean = muse.mean(axis=0)

    cres_std = cres.std(axis=0)
    muse_std = muse.std(axis=0)

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, cres_mean, label="Crescent")
    plt.plot(x, muse_mean, label="HiTAC-MuSE")

    if args.show_std:
        plt.fill_between(x, cres_mean - cres_std, cres_mean + cres_std, alpha=0.2)
        plt.fill_between(x, muse_mean - muse_std, muse_mean + muse_std, alpha=0.2)

    # 画出你设定的 dense/sparse 切换点
    if args.vline is not None and 0 < args.vline < T:
        plt.axvline(args.vline, linestyle="--", linewidth=1.0)
        plt.text(args.vline + 0.3, plt.ylim()[1] * 0.95, "switch", fontsize=10)

    plt.xlabel("Step")
    plt.ylabel("Total Reward (sum over layers)")
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
