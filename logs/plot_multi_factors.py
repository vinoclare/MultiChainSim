"""
读取 exp2/<factor>/<value>/<algo>/<run>/*.csv
画 (1) 相对 Round-Robin (%) 折线, (2) 训练曲线 (此脚本只含折线)
"""
import argparse
import pathlib
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# -------- 参数 --------
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='exp2', help='CSV 根目录 (exp2)')
parser.add_argument('--out_dir', default='figs/factors', help='输出图片目录')
parser.add_argument('--ema', type=float, default=0.05, help='训练曲线 EMA 平滑系数')
parser.add_argument('--baseline', default='rr', help='Round-Robin 目录名')
args = parser.parse_args()

FACTORS = ['layer', 'task', 'worker', 'step']
# METRICS = ['eval_avg_cost', 'eval_avg_reward', 'eval_avg_utility', 'waiting_time']
# Y_LABELS = {
#     'eval_avg_cost': 'Cost Gain (%)',
#     'eval_avg_reward': 'Reward Gain (%)',
#     'eval_avg_utility': 'Utility Gain (%)',
#     'waiting_time': 'Waiting Time (step)'
# }
# TITLE = {
#     'eval_avg_cost': 'Cost',
#     'eval_avg_reward': 'Reward',
#     'eval_avg_utility': 'Utility',
#     'waiting_time': 'Waiting Time'
# }
# SHORT = {
#     'eval_avg_cost': 'cost',
#     'eval_avg_reward': 'reward',
#     'eval_avg_utility': 'utility',
#     'waiting_time': 'waiting_time'
# }

METRICS = ['eval_avg_cost', 'eval_avg_reward', 'eval_avg_utility']
Y_LABELS = {
    'eval_avg_cost': 'Cost Gain (%)',
    'eval_avg_reward': 'Reward Gain (%)',
    'eval_avg_utility': 'Utility Gain (%)'
}
TITLE = {
    'eval_avg_cost': 'Cost',
    'eval_avg_reward': 'Reward',
    'eval_avg_utility': 'Utility'
}
SHORT = {
    'eval_avg_cost': 'cost',
    'eval_avg_reward': 'reward',
    'eval_avg_utility': 'utility'
}

plt.rcParams.update({
    'font.size': 13,
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 2,
})


def build_color_dict(algo_dirs):
    cmap = cm.get_cmap('tab10')
    algos = sorted(a for a in algo_dirs if a != args.baseline)
    return {a: cmap(i % 10) for i, a in enumerate(algos)}


def load_run_final_values(algo_dir, metric, is_baseline=False):
    """
    返回每个 run 的最好值
    - 非 baseline：
        - cost 和 waiting_time 用最小值
        - reward 和 utility 用最大值
    - baseline：
        - 始终只取最后一行的值
    """
    values = []

    if is_baseline:
        csv_path = algo_dir / f'{metric}.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                values.append(df['value'].iloc[-1])
        return values

    # 非 baseline，遍历子目录
    for run_dir in algo_dir.iterdir():
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / f'{metric}.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue

        if metric in ['eval_avg_cost', 'waiting_time']:
            final_value = df['value'].min()
        else:
            final_value = df['value'].max()

        values.append(final_value)
    return values


out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ========== PART: 相对提升折线 ==========
for fac in FACTORS:
    fac_root = pathlib.Path(args.log_dir) / fac
    if not fac_root.exists():
        continue

    values = sorted([d.name for d in fac_root.iterdir() if d.is_dir()],
                    key=lambda x: float(re.sub('[^0-9.]', '', x)))
    # 收集算法集合
    algo_set = set()
    for v in values:
        for algo_dir in (fac_root / v).iterdir():
            if algo_dir.is_dir():
                algo_set.add(algo_dir.name)
    color_dict = build_color_dict(algo_set)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        vals_float = [float(re.sub('[^0-9.]', '', v)) for v in values]
        xs = list(range(len(vals_float)))
        x_labels = [f'{v:g}' for v in vals_float]

        if metric == 'waiting_time':
            # 画绝对值
            for algo in sorted(algo_set):
                if algo == args.baseline:
                    continue
                ys_mean, ys_std = [], []
                for v in values:
                    algo_path = fac_root / v / algo
                    run_values = load_run_final_values(algo_path, metric, is_baseline=True)
                    if len(run_values) == 0:
                        ys_mean.append(None)
                        ys_std.append(None)
                    else:
                        ys_mean.append(sum(run_values) / len(run_values))
                        ys_std.append(pd.Series(run_values).std())

                if all(y is None for y in ys_mean):
                    continue
                ax.errorbar(xs, ys_mean, yerr=ys_std,
                            marker='o', label=algo,
                            color=color_dict[algo],
                            capsize=5, elinewidth=2, alpha=0.9)
            ax.set_ylabel('Waiting Time')
            ax.legend(loc='best', frameon=False, fontsize=10)
        else:
            # 相对提升
            ax.axhline(0, color='#888888', linewidth=1, linestyle='--')
            baseline_vals = {}
            for v in values:
                base_path = fac_root / v / args.baseline
                run_vals = load_run_final_values(base_path, metric, is_baseline=True)
                if len(run_vals) == 0:
                    continue
                baseline_vals[v] = sum(run_vals) / len(run_vals)

            for algo in sorted(algo_set):
                if algo == args.baseline:
                    continue
                ys, yerr = [], []
                for v in values:
                    base = baseline_vals.get(v)
                    if base is None:
                        ys.append(None)
                        yerr.append(None)
                        continue
                    algo_path = fac_root / v / algo
                    run_vals = load_run_final_values(algo_path, metric)
                    if len(run_vals) == 0:
                        ys.append(None)
                        yerr.append(None)
                        continue
                    mean_a = sum(run_vals) / len(run_vals)
                    rel_gain = (mean_a - base) / (abs(base) + 1e-8) * 100
                    ys.append(rel_gain)
                    yerr.append(pd.Series(run_vals).std() / (abs(base) + 1e-8) * 100)

                if all(y is None for y in ys):
                    continue
                ax.errorbar(xs, ys, yerr=yerr,
                            marker='o', label=algo,
                            color=color_dict[algo],
                            capsize=5, elinewidth=2, alpha=0.9)

        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(f'Number of {fac.capitalize()}s')
        ax.set_ylabel(Y_LABELS[metric])
        ax.set_title(TITLE[metric])

        ax.legend(frameon=False, loc='best', fontsize=10)
        fig.tight_layout()
        fname = f'{fac}_{SHORT[metric]}_gain.pdf'
        fig.savefig(out_dir / fname, bbox_inches='tight')
        plt.close(fig)
        print(f'[Saved] {fname}')

print('\n✅ 全部完成')
