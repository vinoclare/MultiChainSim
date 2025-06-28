"""
run_ppo_dist.py
—————————
批量并行启动 run_ppo.py
"""

import concurrent.futures as cf
import os
import subprocess

CFG_ROOT = "../configs"  # 根目录
CATEGORIES = ["task", "layer", "worker", "step"]
REPEAT_EACH_EXP = 1  # 同一实验重复次数
MAX_WORKERS = 12  # 并行进程数


def list_exp_dirs(root, cats):
    """yield (cat, exp_name, exp_dir)"""
    for cat in cats:
        cat_dir = os.path.join(root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for exp_name in sorted(os.listdir(cat_dir)):
            exp_dir = os.path.join(cat_dir, exp_name)
            if os.path.isdir(exp_dir):
                yield cat, exp_name, exp_dir


def run_once(exp_dir, run_idx):
    """
    单次实验：把相对路径 <cat/exp> 作为 --dire 传给 run_ppo.py
    并用 run_idx 作为 --seed 保证每次不同 seed。
    """
    dire = os.path.relpath(exp_dir, CFG_ROOT)  # 例如 "worker/4" 或 "task/expA"
    tag = f"{dire} (run {run_idx})"
    print(f"▶️  开始 {tag}")

    env = os.environ.copy()

    # 这里我们把 run_idx 当作 seed 传入
    subprocess.run(
        [
            "python", "run_ppo.py",
            "--dire", dire,
            "--seed", str(run_idx)
        ],
        env=env,
        check=True
    )

    print(f"✅ 完成 {tag}")


if __name__ == "__main__":
    print("\n=== PPO 批量实验开始 ===\n")

    # ---------- 收集任务 ----------
    tasks = []
    for cat, exp_name, exp_dir in list_exp_dirs(CFG_ROOT, CATEGORIES):
        for k in range(REPEAT_EACH_EXP):
            tasks.append((exp_dir, k))

    # ---------- 并行执行 ----------
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(run_once, exp_dir, run_idx)
            for exp_dir, run_idx in tasks
        ]

        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                fut.result()
            except Exception as e:
                print(f"❌ 任务出错：{e}")
            print(f"✔️  已完成 {i}/{len(tasks)}")

    print("🎉 全部 PPO 实验已结束\n")
