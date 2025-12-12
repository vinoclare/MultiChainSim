import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


# ====== 命令行参数 ======

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["ablation", "pl"],
    required=True,
    help="绘图模式: ablation 消融实验, pl 参数学习"
)
parser.add_argument(
    "--out",
    default=None,
    help="输出图片路径 (如 figs/ablation.pdf); 不设则直接 show()"
)
parser.add_argument(
    "--max_steps",
    type=float,
    default=2_000_000,
    help="x 轴最大步数, 默认 2e6"
)
args = parser.parse_args()


# ====== 指标配置 (和主实验保持一致) ======

# (内部名字, y 轴标签, tensorboard tag)
METRICS = [
    ("cost", "Cost", "eval/cost"),
    ("reward", "Reward", "eval/reward"),
    ("utility", "Utility", "eval/utility"),
]

# ablation 的图例映射（需要的话可以自己继续加/改）
ABLA_LEGEND = {
    "crescent": "CReSCENT",
    "no_ir": "No-IR",
    "no_cl": "No-CL",
    "no_cluster": "No-Cluster",
    "no_ca": "No-CA",
}

# pl 模式下就直接用系数, 或者在这里自定义
PL_LEGEND = {
    # "0": "coef=0",
    # "0.1": "coef=0.1",
}


# ====== 工具函数 ======

def load_scalar_from_run(run_dir: str, tag: str):
    """
    从单个 seed 目录中读取某个 scalar tag, 返回 (steps, values),
    若失败则返回 None。
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


def aggregate_variant(variant_dir: str, tag: str):
    """
    聚合某个 variant (一种消融 / 一种参数配置) 下的所有 seed。
    目录结构形如：

      variant_dir/
        20251209-110546/
        20251210-093837/
        ...

    返回 (steps, mean, std)，若全都读不到则返回 None。
    """
    if not os.path.isdir(variant_dir):
        return None

    run_dirs = [
        os.path.join(variant_dir, d)
        for d in os.listdir(variant_dir)
        if os.path.isdir(os.path.join(variant_dir, d))
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

    # 对齐 step：截断到最短长度
    min_len = min(len(s) for s in all_steps)
    steps_ref = all_steps[0][:min_len]
    values_stacked = np.stack([v[:min_len] for v in all_values], axis=0)

    mean = values_stacked.mean(axis=0)
    std = values_stacked.std(axis=0)
    return steps_ref, mean, std


# ====== 主绘图逻辑 ======

def main():
    if args.mode == "ablation":
        root = "ablations"
    else:
        root = "pl"

    if not os.path.isdir(root):
        raise ValueError(f"log_dir 不存在: {root}")

    # 根目录下的每个子目录都是一个 variant:
    #  - ablation: crescent / no_ca / no_cl / no_cluster / ...
    #  - pl: 0 / 0.1 / 1.0 / 10 / ...
    variants = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    variants.sort()
    if not variants:
        raise ValueError(f"{root} 下没有找到任何子目录 (variant)")

    if args.mode == "ablation":
        legend_map = ABLA_LEGEND
        title_prefix = "Ablation"
    else:
        legend_map = PL_LEGEND
        title_prefix = "Parameter Learning"

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(4 * n_metrics, 3),
        squeeze=False
    )
    axes = axes[0]  # 取出第一行

    for m_idx, (metric_name, metric_label, tag) in enumerate(METRICS):
        ax = axes[m_idx]

        for var in variants:
            variant_dir = os.path.join(root, var)
            agg = aggregate_variant(variant_dir, tag)
            if agg is None:
                continue
            steps, mean, std = agg

            # 图例标签
            if args.mode == "ablation":
                label = legend_map.get(var, var)
            else:
                label = legend_map.get(var, f"coef={var}")

            # 画均值 + 阴影
            line, = ax.plot(steps, mean, label=label)
            ax.fill_between(
                steps,
                mean - std,
                mean + std,
                alpha=0.2,  # 和你主实验脚本保持一致
            )

        ax.set_xlabel("Steps")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} - {title_prefix}")
        if args.max_steps is not None:
            ax.set_xlim(0, args.max_steps)

    # 合并 legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(variants))

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    if args.out is not None:
        out_dir = os.path.dirname(args.out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(args.out, dpi=400, bbox_inches="tight")
        print(f"[Saved] {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
