import os
import argparse
import pathlib
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='exp', help='原始 TensorBoard 日志目录')
parser.add_argument('--dst', default='exp2', help='CSV 输出目录')
args = parser.parse_args()

# 保留的 scalar tag
KEEP_TAGS = {
    'global/eval_avg_cost':    'eval_avg_cost',
    'global/eval_avg_reward':  'eval_avg_reward',
    'global/eval_avg_utility': 'eval_avg_utility'
}

MAX_STEP = 1_001_000  # 新增：最多只保留到 1.001M 步

src_root = pathlib.Path(args.src).resolve()
dst_root = pathlib.Path(args.dst).resolve()
dst_root.mkdir(parents=True, exist_ok=True)


def latest_event_file(tb_dir):
    """找到目录下最新的 tfevents 文件"""
    files = [f for f in os.listdir(tb_dir) if 'tfevents' in f]
    if not files:
        return None
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(tb_dir, f)))[-1]


count = 0

# 遍历四大任务类别
for task_type in ['layer', 'worker', 'step', 'task']:
    type_path = src_root / task_type
    if not type_path.exists():
        continue
    # 遍历每个 setting 子文件夹
    for setting in type_path.iterdir():
        if not setting.is_dir():
            continue
        # 遍历每个算法
        for algo in ['agent57', 'hitac-muse', 'ppo', 'rascl']:
            algo_path = setting / algo
            if not algo_path.exists():
                continue
            # 遍历每个 run 文件夹（0,1,2,3）
            for run_dir in algo_path.iterdir():
                if not run_dir.is_dir():
                    continue
                rel_path = run_dir.relative_to(src_root)
                event_file = latest_event_file(run_dir)
                if not event_file:
                    continue
                event_path = run_dir / event_file
                try:
                    ea = event_accumulator.EventAccumulator(str(event_path), size_guidance={'scalars': 0})
                    ea.Reload()
                except Exception as e:
                    print(f'[Skip] {rel_path}: 读取失败 → {e}')
                    continue

                tags = ea.Tags().get('scalars', [])
                tags_to_save = [t for t in tags if t in KEEP_TAGS]
                if not tags_to_save:
                    continue

                out_dir = dst_root / rel_path
                out_dir.mkdir(parents=True, exist_ok=True)
                for tag in tags_to_save:
                    events = ea.Scalars(tag)
                    # 新增：只保留 step <= MAX_STEP 的数据
                    filtered = [(e.step, e.value) for e in events if e.step <= MAX_STEP]
                    if not filtered:
                        continue
                    df = pd.DataFrame(filtered, columns=['step', 'value'])
                    out_path = out_dir / f"{KEEP_TAGS[tag]}.csv"
                    df.to_csv(out_path, index=False)
                print(f'[OK] 保存 {rel_path} 中 {len(tags_to_save)} 个指标（已限制到 {MAX_STEP:,} 步）')
                count += 1

print(f'\n✅ 全部完成，共处理 {count} 个 run 文件夹')
