#!/usr/bin/env python3
"""
检查 exp 目录下所有实验是否跑满 5M step
"""

import os
from pathlib import Path
from typing import Optional

try:
    # TensorBoard 自带的事件读取工具，不依赖 TensorFlow
    from tensorboard.backend.event_processing import event_accumulator as ea
except ImportError:
    raise ImportError(
        "找不到 tensorboard 库，请先执行:  pip install tensorboard"
    )

TAG = "global/eval_avg_reward"  # 需要检测的标量名
TARGET_STEP = 5_000_000  # 目标 step
TOLERANCE = 10_000  # 允许的误差


def find_event_files(run_dir: Path):
    """返回 run_dir 里所有 tfevents 文件的路径列表"""
    return list(run_dir.rglob("events.out.tfevents.*"))


def get_max_step(event_path: Path, tag: str) -> Optional[int]:
    """
    读取单个 tfevents 文件，返回指定 tag 的最大 step
    如果文件里没有该 tag 或无法读取，则返回 None
    """
    try:
        acc = ea.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            return None
        # 取最后一条 scalar 记录即可
        events = acc.Scalars(tag)
        return events[-1].step if events else None
    except Exception as e:
        print(f"[警告] 读取 {event_path} 失败：{e}")
        return None


def max_step_in_run(run_dir: Path, tag: str) -> int:
    """
    返回某个 run 目录内所有 tfevents 文件中指定 tag 的最大 step
    如果没有找到，返回 0
    """
    max_step = 0
    for ev in find_event_files(run_dir):
        step = get_max_step(ev, tag)
        if step is not None:
            max_step = max(max_step, step)
    return max_step


def scan_exp(root: Path):
    """
    扫描 exp 目录，依次进入
      exp/{layer,worker,step,task}/*/{agent57,hitac-muse,ppo,rascl}/{0,1,2,3}
    对每个 run 目录计算 TAG 的最大 step
    打印未达标的项
    """
    for task_type in ["layer", "worker", "step", "task"]:
        type_path = root / task_type
        if not type_path.exists():
            continue
        for setting in type_path.iterdir():  # 例如 2 3 4
            if not setting.is_dir():
                continue
            for algo in ["agent57", "hitac-muse", "ppo", "rascl"]:
                algo_path = setting / algo
                if not algo_path.exists():
                    continue
                # 预期有 0 1 2 3 四个 run
                for run_dir in algo_path.iterdir():
                    if not run_dir.is_dir():
                        continue
                    max_step = max_step_in_run(run_dir, TAG)
                    if max_step < TARGET_STEP - TOLERANCE:
                        print(f"未完成: {run_dir}  当前 step={max_step}")


if __name__ == "__main__":
    EXP_ROOT = Path("exp").resolve()  # 根据需要修改
    if not EXP_ROOT.exists():
        print(f"找不到目录 {EXP_ROOT}")
    else:
        scan_exp(EXP_ROOT)
