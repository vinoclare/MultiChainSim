import argparse, os, pathlib, re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='exp2',
                    help='CSV 根目录')
parser.add_argument('--out_dir', default='figs/training_curves',
                    help='输出图片目录')
parser.add_argument('--ema', type=float, default=0.05,
                    help='训练曲线 EMA 平滑系数')
parser.add_argument('--baseline', default='rr',
                    help='Round-Robin 目录名')
args = parser.parse_args()

FACTORS = ['layer', 'task', 'worker', 'step']
METRICS = ['eval_avg_cost', 'eval_avg_reward', 'eval_avg_utility', 'waiting_time']
SHORT = {'eval_avg_cost': 'cost',
         'eval_avg_reward': 'reward',
         'eval_avg_utility': 'utility',
         'waiting_time': 'waiting time'}

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

for fac in FACTORS:
    fac_root = pathlib.Path(args.log_dir) / fac
    if not fac_root.exists(): continue

    values = sorted([d.name for d in fac_root.iterdir() if d.is_dir()],
                    key=lambda x: float(re.sub('[^0-9.]', '', x)))
    for v in values:
        cur_dir = fac_root / v
        algo_dirs = [d for d in cur_dir.iterdir() if d.is_dir()]
        color_dict = build_color_dict([d.name for d in algo_dirs])
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for ad in algo_dirs:
                algo = ad.name
                if algo == args.baseline:
                    if metric == 'waiting_time':
                        rr_val = 0  # RR 没有 waiting time → 记 0
                    else:
                        csv_path = ad / f'{metric}.csv'
                        if not csv_path.exists(): continue
                        df = load_csv(csv_path)
                        if df.empty: continue
                        rr_val = df.iloc[0]['value']
                    ax.axhline(rr_val, color='gray', linestyle='--', label='RR baseline')
                    continue
                csv_path = ad / f'{metric}.csv'
                if not csv_path.exists(): continue
                df = load_csv(csv_path)
                df['value'] = smooth(df['value'], args.ema)
                ax.plot(df['step'] / 1e6, df['value'], label=algo,
                        color=color_dict.get(algo, '#000000'))
            ax.set_xlabel('Training steps (×10⁶)')
            ax.set_ylabel(metric.replace('eval_avg_', '').capitalize())
            ax.set_title(metric.replace('eval_avg_', '').capitalize())
            if metric == 'eval_avg_reward':
                ax.legend(loc='best', frameon=False, fontsize=10)
            fig.tight_layout()
            fname = f'training_{fac}_{v}_{SHORT[metric]}.png'
            fig.savefig(out_dir / fname, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f'[Saved] {fname}')
