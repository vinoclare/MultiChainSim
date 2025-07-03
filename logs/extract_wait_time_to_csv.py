"""
将 TensorBoard 事件文件中的各层 waiting-time 指标相加，
输出 waiting_time.csv 到镜像目录结构下。

用法：
python extract_wait_time_to_csv.py --src exp --dst exp2
"""
import argparse
import pathlib
import re
from collections import defaultdict

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='exp', help='原始 events 根目录')
parser.add_argument('--dst', default='exp2', help='CSV 输出根目录')
args = parser.parse_args()

# 等待时间的 tag 正则匹配
WAIT_RE = re.compile(r'^eval/layer_?(\d+)_avg_wait_penalty$')

src_root = pathlib.Path(args.src).resolve()
dst_root = pathlib.Path(args.dst).resolve()
dst_root.mkdir(parents=True, exist_ok=True)


def latest_event_file(dir_path: pathlib.Path):
    """找到目录下最新的 tfevents 文件"""
    files = [f for f in dir_path.iterdir() if 'tfevents' in f.name]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


processed = 0

# 遍历四大任务类别
for task_type in ['layer', 'worker', 'step', 'task']:
    type_path = src_root / task_type
    if not type_path.exists():
        continue
    # 遍历每个 setting 子目录
    for setting in type_path.iterdir():
        if not setting.is_dir():
            continue
        # 遍历每个算法
        for algo in ['agent57', 'hitac-muse', 'ppo', 'rascl']:
            algo_path = setting / algo
            if not algo_path.exists():
                continue
            # 遍历 run 目录（0,1,2,3）
            for run_dir in algo_path.iterdir():
                if not run_dir.is_dir():
                    continue
                rel = run_dir.relative_to(src_root)
                ev_path = latest_event_file(run_dir)
                if ev_path is None:
                    continue

                # 载入事件文件
                try:
                    ea = event_accumulator.EventAccumulator(str(ev_path), size_guidance={'scalars': 0})
                    ea.Reload()
                except Exception as e:
                    print(f'[Skip] {rel}: 解析失败 -> {e}')
                    continue

                # 找所有等待时间 tags
                wait_tags = [t for t in ea.Tags()['scalars'] if WAIT_RE.match(t)]
                if not wait_tags:
                    continue

                # 聚合：step → 累加值
                step_sum = defaultdict(float)
                for tag in wait_tags:
                    for ev in ea.Scalars(tag):
                        step_sum[ev.step] += ev.value

                # 保存 CSV
                dst_dir = dst_root / rel
                dst_dir.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(sorted(step_sum.items()), columns=['step', 'value'])
                df.to_csv(dst_dir / 'waiting_time.csv', index=False)
                processed += 1
                print(f'[OK] {rel} -> waiting_time.csv ({len(df)} rows)')

print(f'\n✅ 全部完成，共处理 {processed} 个 run 文件夹')
