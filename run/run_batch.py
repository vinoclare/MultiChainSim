import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--alg", type=str, default="crescent")
parser.add_argument("--dire_list", nargs="+", required=True)
parser.add_argument("--repeat", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=10)
args = parser.parse_args()

# 根据 alg 自动选择正确的 run_xxx.py
alg = args.alg.lower()
run_file = f"run_{alg}.py"

if not os.path.exists(run_file):
    raise FileNotFoundError(f"找不到 {run_file}，请确认你的项目里有这个脚本。")

for dire in args.dire_list:
    print(f"=== Running dire={dire} for {args.repeat} times (alg={alg}) ===")
    for i in range(args.repeat):
        print(f"[{dire}] Run {i+1}/{args.repeat}")

        cmd = [
            "python", run_file,
            "--dire", dire,
            "--num_workers", str(args.num_workers),
            "--mode", "load",
        ]

        subprocess.run(cmd)
