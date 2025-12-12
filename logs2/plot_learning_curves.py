import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

import argparse

# ===== 命令行参数 =====
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["multi", "standard"],
    default="multi",
    help="multi: layer/task/worker/step 多环境对比; standard: 只画 standard 三张图"
)
args = parser.parse_args()

# ===== 手动配置 =====

# logs 根目录（第二层 logs2 那个目录）
BASE_LOG_DIR = r"E:\Codes\MultiChainSim\logs2\exp"

# 参与对比的算法及图例名字（顺序会影响颜色）
ALG_ORDER = ["eta_psi", "emu", "mimex", "crescent"]

ALG_LEGEND = {
    "eta_psi": "ETA-PSI",
    "emu": "EMU",
    "mimex": "MIMEx",
    "crescent": "CReSCENT",
}

# 哪些指标、标签，对应 tensorboard 里的 tag
# 你现在 run_crescent 里 eval 部分写的是 eval/reward, eval/cost, eval/utility :contentReference[oaicite:0]{index=0}
METRICS = [
    ("cost", "Cost", "eval/cost"),
    ("reward", "Reward", "eval/reward"),
    ("utility", "Utility", "eval/utility"),
]

# 布局配置：每一行画两个对象（如 layer 2 & layer 4），每个对象画 3 个指标（Cost/Reward/Utility）
if args.mode == "multi":
    # 多环境对比：4 行 * 2 列 (layer/task/worker/step)
    ROW_CONFIG = [
        ("layer", [2, 4], "layer"),
        ("task", ["1.0", "3.5"], "task"),
        ("worker", [6, 10], "worker"),
        ("step", [50, 200], "step"),
    ]
else:
    # standard 模式：只画一行，每个算法一条标准曲线
    ROW_CONFIG = [
        ("standard", [None], "standard"),
    ]


# ===== 工具函数 =====
def get_leaf_dir(alg_name: str, subdir: str, key):
    """
    根据算法名 + 维度类别 + 具体 id，返回存放若干 seed 子目录的目录路径。
    - crescent：  logs2/logs2/crescent/layer/2/...
    - 其他算法： logs2/logs2/happo/mappo/layer/2/...
    - standard： logs2/logs2/<alg>[/mappo]/standard/   （下层直接是时间戳子目录）
    """
    alg_root = os.path.join(BASE_LOG_DIR, alg_name)
    if alg_name != "crescent":
        alg_root = os.path.join(alg_root, "mappo")

    if subdir == "standard":
        return os.path.join(alg_root, "standard")

    return os.path.join(alg_root, subdir, str(key))


def load_scalar_from_run(run_dir: str, tag: str):
    """
    从单个 run（一个时间戳目录）中读取某个 scalar tag，返回 (steps, values)。
    """
    if not os.path.isdir(run_dir):
        return None

    event_files = [
        f for f in os.listdir(run_dir)
        if f.startswith("events.out.tfevents")
    ]
    if not event_files:
        return None

    event_path = os.path.join(run_dir, event_files[0])
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return None

    scalars = ea.Scalars(tag)
    steps = np.array([s.step for s in scalars], dtype=np.float64)
    values = np.array([s.value for s in scalars], dtype=np.float64)
    return steps, values


def aggregate_runs(leaf_dir: str, tag: str):
    """
    给定某个叶子目录（例如 logs2/.../crescent/layer/2），
    聚合其下所有 run（若干时间戳子目录），计算均值和标准差。

    返回 (steps, mean, std)，如果一个都读不到就返回 None。
    """
    if not os.path.isdir(leaf_dir):
        return None

    run_dirs = [
        os.path.join(leaf_dir, d)
        for d in os.listdir(leaf_dir)
        if os.path.isdir(os.path.join(leaf_dir, d))
    ]
    run_dirs.sort()
    if not run_dirs:
        return None

    all_steps = []
    all_values = []

    for rd in run_dirs:
        res = load_scalar_from_run(rd, tag)
        if res is None:
            continue
        steps, vals = res
        all_steps.append(steps)
        all_values.append(vals)

    if not all_values:
        return None

    # 简化假设：不同 seed 的 steps 基本对齐，只截取到最短长度
    min_len = min(len(s) for s in all_steps)
    steps_ref = all_steps[0][:min_len]
    values_stacked = np.stack([v[:min_len] for v in all_values], axis=0)

    mean = values_stacked.mean(axis=0)
    std = values_stacked.std(axis=0)
    return steps_ref, mean, std


# ===== 主绘图逻辑 =====

def main():
    n_rows = len(ROW_CONFIG)
    n_metrics = len(METRICS)
    # 每行的对象数量由 ROW_CONFIG 决定（multi=2, standard=1）
    n_entities_per_row = len(ROW_CONFIG[0][1])
    n_cols = n_metrics * n_entities_per_row

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
        sharex=False
    )

    # 逐行、逐列填图
    for row_idx, (subdir, keys, prefix) in enumerate(ROW_CONFIG):
        for ent_idx, key in enumerate(keys):
            for m_idx, (metric_name, metric_label, tag) in enumerate(METRICS):
                col_idx = ent_idx * n_metrics + m_idx
                ax = axes[row_idx, col_idx]

                # 每个子图画所有算法
                for alg in ALG_ORDER:
                    leaf_dir = get_leaf_dir(alg, subdir, key)
                    agg = aggregate_runs(leaf_dir, tag)
                    if agg is None:
                        continue
                    steps, mean, std = agg
                    line, = ax.plot(steps, mean, label=ALG_LEGEND.get(alg, alg))
                    ax.fill_between(steps, mean - std, mean + std, alpha=0.2)

                # 轴标签 & 标题
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Steps")
                if m_idx == 0:
                    ax.set_ylabel(metric_label)

                title_key = "" if key is None else f" {key}"
                ax.set_title(f"{metric_label} - {prefix}{title_key}")

                # === 根据实验类型设置不同的总步数范围 ===
                if subdir == "step":
                    # step/50 -> 1M, step/200 -> 4M
                    if str(key) == "50":
                        ax.set_xlim(0, 1_000_000)
                    elif str(key) == "200":
                        ax.set_xlim(0, 4_000_000)
                else:
                    # layer / task / worker 统一 2M
                    ax.set_xlim(0, 2_000_000)

    # 只放一个总 legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ALG_ORDER))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
