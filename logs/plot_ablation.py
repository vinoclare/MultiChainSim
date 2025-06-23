import argparse, os, re, pathlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from matplotlib import cm

# ---------- 参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='exp/ablation', help='包含子实验日志的总目录')
parser.add_argument('--out_dir', default='figs', help='输出图片保存目录')
parser.add_argument('--ema', type=float, default=0.05, help='指数滑动平均系数（如0.6）')
args = parser.parse_args()

# ---------- 绘图标签 ----------
TAGS = [
    'global/episode_wait_penalty',
    'global/eval_avg_cost',
    'global/eval_avg_reward',
    'global/eval_avg_utility',
]
TITLE = {
    'global/episode_wait_penalty': 'Wait Time',
    'global/eval_avg_cost':        'Cost',
    'global/eval_avg_reward':      'Reward',
    'global/eval_avg_utility':     'Utility',
}

# ---------- Matplotlib 全局风格 ----------
plt.rcParams.update({
    'font.size': 13,
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 2,
})

# ---------- 扫描所有实验子目录 ----------
exp_dirs = [os.path.join(args.log_dir, d) for d in os.listdir(args.log_dir)
            if os.path.isdir(os.path.join(args.log_dir, d))]
exp_names = sorted([os.path.basename(d) for d in exp_dirs])

# ---------- 固定颜色映射：所有图中一致 ----------
COLORMAP = cm.get_cmap('tab10')
COLOR_DICT = {
    name: COLORMAP(i % 10) for i, name in enumerate(exp_names)
}

# ---------- 加载单个实验目录下的 scalars ----------
def load_scalars(tb_dir):
    ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={'scalars': 0})
    ea.Reload()
    out = {}
    for tag in TAGS:
        if tag not in ea.Tags()['scalars']:
            continue
        events = ea.Scalars(tag)
        df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value'])
        if args.ema > 0:
            df['value'] = df['value'].ewm(alpha=args.ema, adjust=False).mean()
        out[tag] = df
    return out

# ---------- 加载全部数据 ----------
all_data = {tag:{} for tag in TAGS}
for d in exp_dirs:
    name = os.path.basename(d)
    try:
        scalars = load_scalars(d)
        for tag, df in scalars.items():
            all_data[tag][name] = df
    except Exception as e:
        print(f'[Warn] 读取失败 {d}: {e}')

# ---------- 创建输出目录 ----------
pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

# ---------- 绘图 ----------
for tag, runs in all_data.items():
    if not runs:
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in runs.items():
        ax.plot(df['step'] / 1e6, df['value'],
                label=name,
                color=COLOR_DICT[name])

    ax.set_xlabel('Training steps (×10⁶)')
    ax.set_ylabel(TITLE[tag])
    ax.set_title(TITLE[tag])

    # 仅 eutility 图加 legend
    if tag.endswith('eval_avg_utility'):
        preferred_locs = ['upper left', 'lower right']
        placed = False
        for loc in preferred_locs:
            legend = ax.legend(loc=loc, frameon=False, fontsize=10)
            fig.canvas.draw()
            if not legend.get_window_extent().overlaps(ax.get_window_extent()):
                placed = True
                break
        if not placed:
            ax.legend(loc='best', frameon=False, fontsize=10)
    else:
        # 不显示 legend
        if ax.get_legend():
            ax.get_legend().remove()

    fig.tight_layout()
    fname = re.sub(r'[^\w\-]+', '_', tag.split('/')[-1]) + '.png'
    out_path = os.path.join(args.out_dir, fname)
    fig.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {out_path}')

print('✅ 所有图完成。')
