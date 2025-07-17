import argparse, pathlib, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='exp2', help='CSV 根目录')
parser.add_argument('--out_dir', default='figs/training_curves', help='输出图片目录')
parser.add_argument('--ema', type=float, default=0.3, help='EMA 平滑系数')
args = parser.parse_args()

FACTORS = ['layer', 'task', 'worker', 'step']
METRICS = ['eval_avg_cost', 'eval_avg_reward', 'eval_avg_utility']
SHORT = {'eval_avg_cost': 'cost',
         'eval_avg_reward': 'reward',
         'eval_avg_utility': 'utility'}

plt.rcParams.update({
    'font.size': 18,   # 总字体放大
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 2,
})


def smooth(series, alpha):
    return series if alpha <= 0 else series.ewm(alpha=alpha, adjust=False).mean()


def build_color_dict(algo_names):
    cmap = cm.get_cmap('tab10')
    return {a: cmap(i % 10) for i, a in enumerate(sorted(algo_names))}


out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# 先收集所有子图信息
subplot_info = []

for fac in FACTORS:
    fac_root = pathlib.Path(args.log_dir) / fac
    if not fac_root.exists():
        continue

    values = sorted([d.name for d in fac_root.iterdir() if d.is_dir()],
                    key=lambda x: float(re.sub('[^0-9.]', '', x)))
    for v in values:
        cur_dir = fac_root / v
        algo_dirs = [d for d in cur_dir.iterdir() if d.is_dir()]
        algo_names = [d.name for d in algo_dirs]
        color_dict = build_color_dict(algo_names)

        for metric in METRICS:
            subplot_info.append((fac, v, metric, cur_dir, algo_dirs, color_dict))

# 固定 4 行 6 列
rows = 4
cols = 6

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
axes = axes.flatten()

# 全局 legend 收集器
all_handles = {}
color_dict_global = {}

for idx, (fac, v, metric, cur_dir, algo_dirs, color_dict) in enumerate(subplot_info):
    if idx >= rows * cols:
        break  # 防止超出

    ax = axes[idx]

    for ad in algo_dirs:
        algo = ad.name
        run_dirs = [d for d in ad.iterdir() if d.is_dir()]

        all_dfs = []
        for run_d in run_dirs:
            csv_path = run_d / f'{metric}.csv'
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df = df.copy()
            df['value'] = smooth(df['value'], args.ema)
            all_dfs.append(df)

        if not all_dfs:
            continue

        min_len = min(len(df) for df in all_dfs)
        steps = all_dfs[0]['step'][:min_len]
        values = np.stack([df['value'][:min_len].values for df in all_dfs], axis=0)

        mean_vals = values.mean(axis=0)
        std_vals = values.std(axis=0)

        line, = ax.plot(steps, mean_vals, label=algo, color=color_dict.get(algo, '#000000'))
        ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals,
                        color=color_dict.get(algo, '#000000'), alpha=0.2)

        # 收集 handle，用于全局 legend
        if algo not in all_handles:
            all_handles[algo] = line
            color_dict_global[algo] = color_dict.get(algo, '#000000')

    ax.set_title(f'{metric.replace("eval_avg_", "").capitalize()} - {fac} {v}', fontsize=16)
    ax.set_xlabel('Steps', fontsize=14)
    ax.set_ylabel(metric.replace('eval_avg_', '').capitalize(), fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

# 删除多余空子图
for j in range(len(subplot_info), len(axes)):
    fig.delaxes(axes[j])

# 在大图上放 legend
fig.legend(all_handles.values(), all_handles.keys(),
           loc='upper center', ncol=len(all_handles),
           fontsize=18, frameon=False)

fig.tight_layout(rect=[0, 0, 1, 0.95])  # 为上方 legend 留空间
fig.savefig(out_dir / 'training_all_merged.pdf', dpi=400, bbox_inches='tight')
plt.close(fig)

print(f'✅ 已生成 4x6 大图（全局 legend）: {out_dir / "training_all_merged.pdf"}')
