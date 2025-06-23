#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取 exp2/<factor>/<value>/<algo>/*.csv
画 (1) 相对 Round-Robin (%) 折线, (2) 训练曲线
"""
import argparse, os, pathlib, re
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# -------- 参数 --------
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='exp2',
                    help='CSV 根目录 (exp2)')
parser.add_argument('--out_dir', default='figs/factors',
                    help='输出图片目录')
parser.add_argument('--ema', type=float, default=0.05,
                    help='训练曲线 EMA 平滑系数')
parser.add_argument('--baseline', default='rr',
                    help='Round-Robin 目录名')
args = parser.parse_args()

FACTORS = ['layer', 'task', 'worker', 'step']
METRICS = ['eval_avg_cost', 'eval_avg_reward', 'eval_avg_utility', 'waiting_time']
Y_LABELS = {'eval_avg_cost': 'Cost Gain (%)',
            'eval_avg_reward': 'Reward Gain (%)',
            'eval_avg_utility': 'Utility Gain (%)',
            'waiting_time': 'Waiting Time (step)'}
TITLE = {'eval_avg_cost': 'Cost',
         'eval_avg_reward': 'Reward',
         'eval_avg_utility': 'Utility',
         'waiting_time': 'Waiting Time (step)'}
SHORT = {'eval_avg_cost': 'cost',
         'eval_avg_reward': 'reward',
         'eval_avg_utility': 'utility',
         'waiting_time': 'waiting_time'}

plt.rcParams.update({
    'font.size': 13,
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 2,
})


# -------- 颜色映射 --------
def build_color_dict(algo_dirs):
    cmap = cm.get_cmap('tab10')
    algos = sorted(a for a in algo_dirs if a != args.baseline)
    return {a: cmap(i % 10) for i, a in enumerate(algos)}


# -------- 读取单个 csv --------
def load_csv(path):
    return pd.read_csv(path)


# -------- EMA 平滑 --------
def smooth(series, alpha):
    return series if alpha <= 0 else series.ewm(alpha=alpha, adjust=False).mean()


out_dir = pathlib.Path(args.out_dir);
out_dir.mkdir(parents=True, exist_ok=True)

# ========== PART 1: 相对提升折线 ==========
for fac in FACTORS:
    fac_root = pathlib.Path(args.log_dir) / fac
    if not fac_root.exists(): continue

    values = sorted([d.name for d in fac_root.iterdir() if d.is_dir()],
                    key=lambda x: float(re.sub('[^0-9.]', '', x)))
    # 收集算法集合
    algo_set = set()
    for v in values:
        for algo_dir in (fac_root / v).iterdir():
            if algo_dir.is_dir():
                algo_set.add(algo_dir.name)
    color_dict = build_color_dict(algo_set)

    # 逐指标画单独 png
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        vals_float = [float(re.sub('[^0-9.]', '', v)) for v in values]
        xs = list(range(len(vals_float)))
        x_labels = [f'{v:g}' for v in vals_float]

        if metric == 'waiting_time':
            # === Waiting-time 直接画绝对值 ===
            ax.axhline(0, color='#888888', linestyle='--')  # RR baseline = 0
            for algo in sorted(algo_set):
                if algo == args.baseline:  # RR 无数据 => 不画曲线
                    continue
                ys = []
                for v in values:
                    csv_path = fac_root / v / algo / 'waiting_time.csv'
                    if csv_path.exists():
                        df = load_csv(csv_path)
                        ys.append(df['value'].min())  # 取最大或平均均可
                    else:
                        ys.append(None)
                if any(y is not None for y in ys):
                    ax.plot(xs, ys, marker='o',
                            label=algo, color=color_dict[algo])
            ax.set_ylabel('Waiting Time')
            ax.legend(loc='best', frameon=False, fontsize=10)
        else:
            # 画 0% 基线
            ax.axhline(0, color='#888888', linewidth=1, linestyle='--')

            # 取 baseline 值字典 baseline[value] = (val, std)
            baseline_vals, baseline_stds = {}, {}
            for v in values:
                csv_base = fac_root / v / args.baseline / f'{metric}.csv'
                if not csv_base.exists(): continue
                dfb = load_csv(csv_base)
                best_row = dfb.loc[dfb['value'].idxmax()]
                baseline_vals[v] = best_row['value']
                if metric == 'eval_avg_reward':
                    std_csv = fac_root / v / args.baseline / 'eval_reward_std.csv'
                    if std_csv.exists():
                        std_df = load_csv(std_csv)
                        std_near = std_df.iloc[(std_df['step'] - best_row['step']).abs().argmin()]
                        baseline_stds[v] = std_near['value']

            for algo in sorted(algo_set):
                if algo == args.baseline:  # baseline 不画
                    continue
                ys, yerr = [], []
                for v in values:
                    base = baseline_vals.get(v)
                    if base is None:  # 无 baseline 数据
                        ys.append(None);
                        yerr.append(None);
                        continue
                    algo_csv = fac_root / v / algo / f'{metric}.csv'
                    if not algo_csv.exists():
                        ys.append(None);
                        yerr.append(None);
                        continue
                    dfa = load_csv(algo_csv)
                    best_a = dfa.loc[dfa['value'].idxmax()]
                    rel_gain = (best_a['value'] - base) / (abs(base) + 1e-8) * 100
                    ys.append(rel_gain)
                    if metric == 'eval_avg_reward':
                        std_csv_a = fac_root / v / algo / 'eval_reward_std.csv'
                        std_rel = None
                        if std_csv_a.exists():
                            std_df_a = load_csv(std_csv_a)
                            std_a = std_df_a.iloc[(std_df_a['step'] - best_a['step']).abs().argmin()]['value']
                            # 同样转百分比
                            std_rel = std_a / (abs(base) + 1e-8) * 100
                        yerr.append(std_rel)
                    else:
                        yerr.append(None)

                if all(y is None for y in ys): continue
                if metric == 'eval_avg_reward':
                    ax.errorbar(xs, ys, yerr=yerr,
                                marker='o', label=algo, color=color_dict[algo],
                                capsize=5, elinewidth=2, alpha=0.9)
                else:
                    ax.plot(xs, ys, marker='o',
                            label=algo, color=color_dict[algo])

        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(f'Number of {fac.capitalize()}s')
        ax.set_ylabel(Y_LABELS[metric])
        ax.set_title(TITLE[metric])

        # 只在 Utility 图显示 legend
        if metric == 'eval_avg_reward':
            ax.legend(frameon=False, loc='best', fontsize=10)

        fig.tight_layout()
        fname = f'{fac}_{SHORT[metric]}_gain.png'
        fig.savefig(out_dir / fname, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f'[Saved] {fname}')
