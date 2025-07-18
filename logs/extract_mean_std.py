import argparse
import pathlib
import re
import pandas as pd

# -------- 参数 --------
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='exp2', help='CSV 根目录 (exp2)')
parser.add_argument('--out_file', default='factor_summary.csv', help='输出表格文件路径')
parser.add_argument('--baseline', default='rr', help='Round-Robin 目录名')
args = parser.parse_args()

FACTORS = ['layer', 'task', 'worker', 'step']
METRICS = ['eval_avg_cost', 'eval_avg_reward', 'eval_avg_utility', 'eval_avg_wp']


def load_run_final_values(algo_dir, metric, is_baseline=False):
    values = []
    if is_baseline:
        if metric == 'eval_avg_wp':
            values.append(0.0)
            return values

        csv_path = algo_dir / f'{metric}.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                if metric in ['eval_avg_cost']:
                    values.append(df['value'].min())
                else:
                    values.append(df['value'].max())
        return values

    for run_dir in algo_dir.iterdir():
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / f'{metric}.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue
        if metric in ['eval_avg_cost', 'eval_avg_wp']:
            final_value = df['value'].min()
        else:
            final_value = df['value'].max()
        values.append(final_value)
    return values

# ========== 输出所有因子的指标表格 ==========
rows = []

for fac in FACTORS:
    fac_root = pathlib.Path(args.log_dir) / fac
    if not fac_root.exists():
        continue

    values = sorted([d.name for d in fac_root.iterdir() if d.is_dir()],
                    key=lambda x: float(re.sub('[^0-9.]', '', x)))
    algo_set = set()
    for v in values:
        for algo_dir in (fac_root / v).iterdir():
            if algo_dir.is_dir():
                algo_set.add(algo_dir.name)

    for metric in METRICS:
        for v in values:
            val_float = float(re.sub('[^0-9.]', '', v))
            for algo in sorted(algo_set):
                algo_path = fac_root / v / algo
                run_vals = load_run_final_values(
                    algo_path, metric,
                    is_baseline=(algo == args.baseline)
                )
                if len(run_vals) == 0:
                    continue
                mean = sum(run_vals) / len(run_vals)
                std = pd.Series(run_vals).std()
                rows.append({
                    'Factor': fac,
                    'Value': val_float,
                    'Metric': metric,
                    'Algorithm': algo,
                    'Mean': round(mean, 4),
                    'Std': round(std, 4)
                })

# 存成表格
df = pd.DataFrame(rows)
df.to_csv(args.out_file, index=False)
print(f'✅ 表格写入 {args.out_file}')
