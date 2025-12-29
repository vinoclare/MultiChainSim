import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--alg", type=str, nargs="+", required=True,
    help="支持传入一个或多个算法名，例如: --alg crescent emu mimex"
)
parser.add_argument("--dire_list", nargs="+", default=["standard"], required=True)
parser.add_argument("--repeat", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=10)
args = parser.parse_args()

for alg in args.alg:
    alg_lower = alg.lower()
    run_file = f"run_{alg_lower}.py"

    if not os.path.exists(run_file):
        raise FileNotFoundError(f"找不到 {run_file}，请确认你的项目里有这个脚本。")

    for dire in args.dire_list:
        print(f"=== Running alg={alg_lower}, dire={dire} for {args.repeat} times ===")
        for i in range(args.repeat):
            print(f"[alg={alg_lower} dire={dire}] Run {i+1}/{args.repeat}")

            cmd = [
                "python", run_file,
                "--dire", dire,
                "--num_workers", str(args.num_workers),
                "--mode", "load",
            ]

            subprocess.run(cmd)
