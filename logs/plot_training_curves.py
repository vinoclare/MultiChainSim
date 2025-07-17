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
    'font.size': 13,
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
            fig, ax = plt.subplots(figsize=(6, 4.5))

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

                # 对齐 step
                min_len = min(len(df) for df in all_dfs)
                steps = all_dfs[0]['step'][:min_len]
                values = np.stack([df['value'][:min_len].values for df in all_dfs], axis=0)

                mean_vals = values.mean(axis=0)
                std_vals = values.std(axis=0)

                ax.plot(steps, mean_vals, label=algo, color=color_dict.get(algo, '#000000'))
                ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals,
                                color=color_dict.get(algo, '#000000'), alpha=0.3)

            ax.set_xlabel('Training steps')
            ax.set_ylabel(metric.replace('eval_avg_', '').capitalize())
            ax.set_title(f'{metric.replace("eval_avg_", "").capitalize()} - {fac} {v}')
            ax.legend(loc='best', frameon=False, fontsize=10)

            fig.tight_layout()
            fname = f'training_{fac}_{v}_{SHORT[metric]}.pdf'
            fig.savefig(out_dir / fname, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f'[Saved] {fname}')
