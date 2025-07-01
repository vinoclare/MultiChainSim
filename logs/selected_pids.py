#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计 HiTAC-MuSE 训练日志中 pid 0/1/2 的被选次数
用法:
    python count_selected_pid.py --path /path/to/tb_log_dir
"""
import os
import glob
from collections import defaultdict, Counter
import argparse
from tensorboard.backend.event_processing import event_accumulator


def collect_event_files(logdir):
    """递归搜集目录下全部 *.event* 文件"""
    pattern = os.path.join(logdir, "**", "events.*")
    return glob.glob(pattern, recursive=True)


def count_pids_in_event(event_file, layer_counts):
    """
    读取单个 event 文件，将 layer_counts 更新为:
        layer_counts[layer_id][pid] += 1
    """
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    for tag in ea.Tags()['scalars']:
        if not tag.endswith("/selected_pid"):
            continue
        layer_id = tag.split('/')[0]    # 'layer_0'
        for evt in ea.Scalars(tag):
            pid = int(evt.value)
            if pid in (0, 1, 2):
                layer_counts[layer_id][pid] += 1


def main(logdir):
    event_files = collect_event_files(logdir)
    print(os.getcwd())
    if not event_files:
        print(f"未在 {logdir} 找到 event 文件")
        return

    layer_counts = defaultdict(Counter)

    for evf in event_files:
        count_pids_in_event(evf, layer_counts)

    # 总体统计
    overall = Counter()
    for c in layer_counts.values():
        overall.update(c)

    # ----------- 打印结果 -----------
    print("=== PID 统计结果 ===")
    print(">> 全局计数 (所有 layer 加总)")
    for pid in (0, 1, 2):
        print(f"  pid {pid}: {overall[pid]}")

    print("\n>> 分层计数")
    for layer in sorted(layer_counts.keys()):
        cnts = layer_counts[layer]
        print(f"  {layer}: " + ", ".join(f"pid {pid}={cnts[pid]}" for pid in (0, 1, 2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 selected_pid 的出现次数")
    parser.add_argument("--path", type=str, required=True, help="TensorBoard 日志目录路径")
    args = parser.parse_args()

    main(args.path)
