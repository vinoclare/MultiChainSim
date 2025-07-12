"""
run_hitac_muse_dist.py
——————————
批量并行启动 run_hitac_muse.py
"""

import time
import concurrent.futures as cf
import os
import subprocess

CFG_ROOT = "../configs"  # 所有实验配置的根目录
CATEGORIES = ["step"]  # 与之前保持一致
REPEAT_EACH_EXP = 8  # 同一实验重复次数
MAX_WORKERS = 12  # 并行进程数


def list_exp_dirs(root, cats):
    """遍历 configs 下各类别子目录，产出 (cat, exp_name, exp_dir)"""
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
    单次实验：
      - 通过 --dire 把相对路径传给 run_hitac_muse.py
      - log_dir 在脚本内部按时间戳自动区分
    """
    dire = os.path.relpath(exp_dir, CFG_ROOT)  # 例子: "task/expA"
    tag = f"{dire} (run {run_idx})"
    print(f"▶️  开始 {tag}")

    # 如有需要，可在 env 中指定 CUDA_VISIBLE_DEVICES 防止 GPU 抢占
    env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = ""          # ← 取消注释可强制 CPU 运行

    # 调用子进程
    subprocess.run(
        ["python", "run_hitac_muse_new.py", "--dire", dire],
        env=env,
        check=True
    )
    print(f"✅ 完成 {tag}")


if __name__ == "__main__":
    print("\n=== HiTAC-MuSE 批量实验开始 ===\n")

    # -------- 打包所有任务 --------
    tasks = []
    for cat, exp_name, exp_dir in list_exp_dirs(CFG_ROOT, CATEGORIES):
        for k in range(REPEAT_EACH_EXP):
            tasks.append((exp_dir, k))

    # -------- 并行执行 --------
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(run_once, exp_dir, run_idx)
                   for (exp_dir, run_idx) in tasks]

        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                fut.result()
            except Exception as e:
                print(f"❌ 任务出错：{e}")
            print(f"✔️  已完成 {i}/{len(tasks)}")

    print("🎉 全部 HiTAC-MuSE 实验已结束\n")
