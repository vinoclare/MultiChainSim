import os, argparse, pathlib, re
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='exp', help='原始 TensorBoard 日志目录')
parser.add_argument('--dst', default='exp2', help='CSV 输出目录')
args = parser.parse_args()

KEEP_TAGS = {
    'global/eval_avg_cost':    'eval_avg_cost',
    'global/eval_avg_reward':  'eval_avg_reward',
    'global/eval_avg_utility': 'eval_avg_utility',
    'global/eval_reward_std':  'eval_reward_std',
}

src_root = pathlib.Path(args.src).resolve()
dst_root = pathlib.Path(args.dst).resolve()
dst_root.mkdir(parents=True, exist_ok=True)

def latest_event_file(tb_dir):
    files = [f for f in os.listdir(tb_dir) if 'tfevents' in f]
    if not files:
        return None
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(tb_dir, f)))[-1]

count = 0
for root, _, files in os.walk(src_root):
    rel_path = pathlib.Path(root).relative_to(src_root)
    event_file = latest_event_file(root)
    if not event_file:
        continue
    event_path = os.path.join(root, event_file)
    try:
        ea = event_accumulator.EventAccumulator(event_path, size_guidance={'scalars': 0})
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
        df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value'])
        out_path = out_dir / f"{KEEP_TAGS[tag]}.csv"
        df.to_csv(out_path, index=False)
    print(f'[OK] 保存 {rel_path} 中 {len(tags_to_save)} 个指标')
    count += 1

print(f'\n✅ 全部完成，共处理 {count} 个目录')
