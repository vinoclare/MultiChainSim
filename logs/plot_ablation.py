import os
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorboard.backend.event_processing import event_accumulator

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='exp/PolicyNumber', help='TensorBoard 日志根目录')
parser.add_argument('--out_dir', default='figs/policy_number', help='输出图片目录')
parser.add_argument('--ema', type=float, default=0.3, help='EMA 平滑系数')
args = parser.parse_args()

METRICS = {
    'global/eval_avg_cost': 'cost',
    'global/eval_avg_reward': 'reward',
    'global/eval_avg_utility': 'utility'
}

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


def latest_event_file(tb_dir):
    files = [f for f in os.listdir(tb_dir) if 'tfevents' in f]
    if not files:
        return None
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(tb_dir, f)))[-1]


out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

algo_dirs = [d for d in pathlib.Path(args.log_dir).iterdir() if d.is_dir()]
algo_names = [d.name for d in algo_dirs]
color_dict = build_color_dict(algo_names)

for metric_tag, short_name in METRICS.items():
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for algo_dir in algo_dirs:
        algo_name = algo_dir.name
        run_dirs = [d for d in algo_dir.iterdir() if d.is_dir()]

        all_dfs = []

        for run_dir in run_dirs:
            event_file = latest_event_file(run_dir)
            if not event_file:
                continue
            event_path = run_dir / event_file
            try:
                ea = event_accumulator.EventAccumulator(str(event_path), size_guidance={'scalars': 0})
                ea.Reload()
            except Exception as e:
                print(f'[Skip] {algo_name}/{run_dir.name}: 读取失败 → {e}')
                continue

            if metric_tag not in ea.Tags().get('scalars', []):
                continue

            events = ea.Scalars(metric_tag)
            steps = [e.step for e in events if e.step <= 1_001_000]
            values = [e.value for e in events if e.step <= 1_001_000]

            if not steps:
                continue

            df = pd.DataFrame({'step': steps, 'value': values})
            df['value'] = smooth(df['value'], args.ema)
            all_dfs.append(df)

        if not all_dfs:
            continue

        # 对齐 step
        min_len = min(len(df) for df in all_dfs)
        steps = all_dfs[0]['step'][:min_len]
        values_arr = np.stack([df['value'][:min_len].values for df in all_dfs], axis=0)

        mean_vals = values_arr.mean(axis=0)
        std_vals = values_arr.std(axis=0)

        ax.plot(steps, mean_vals, label=algo_name, color=color_dict.get(algo_name, '#000000'))
        ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals,
                        color=color_dict.get(algo_name, '#000000'), alpha=0.15)

    ax.set_xlabel('Training steps')
    ax.set_ylabel(short_name.capitalize())
    ax.set_title(short_name.capitalize())
    ax.legend(loc='best', frameon=False, fontsize=10)

    fig.tight_layout()
    fname = f'training_{short_name}.pdf'
    fig.savefig(out_dir / fname, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {fname}')
