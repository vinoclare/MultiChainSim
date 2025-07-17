import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# === 全局字体设置 ===
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# === 修改这里：存放 pkl 文件的目录 ===
pkl_dir = "./eval_trajectories"
pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")])

embeddings = []
labels = []

for file_name in pkl_files:
    with open(os.path.join(pkl_dir, file_name), "rb") as f:
        eval_trajectories = pickle.load(f)

    for episode in eval_trajectories:
        for lid, traj in episode.items():
            task_obs_list = traj["task_obs"]
            worker_loads_list = traj["worker_loads"]
            worker_profile_list = traj["worker_profile"]
            global_context_list = traj["global_context"]
            actions_list = traj["actions"]

            if len(task_obs_list) == 0:
                continue

            task_obs_all = np.stack(task_obs_list, axis=0).mean(axis=(0, 1))
            worker_loads_all = np.stack(worker_loads_list, axis=0).mean(axis=(0, 1))
            worker_profile_all = np.stack(worker_profile_list, axis=0).mean(axis=(0, 1))
            global_context_all = np.stack(global_context_list, axis=0).mean(axis=0)
            actions_all = np.stack(actions_list, axis=0).mean(axis=(0, 1))

            feat_vec = np.concatenate([
                task_obs_all.flatten(),
                worker_loads_all.flatten(),
                worker_profile_all.flatten(),
                global_context_all.flatten(),
                actions_all.flatten()
            ], axis=0)

            embeddings.append(feat_vec)
            labels.append(lid)

embeddings = np.stack(embeddings, axis=0)
labels = np.array(labels)

# === t-SNE 降维 ===
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

# === 绘图 ===
plt.figure(figsize=(10, 6))
unique_lids = np.unique(labels)
colors = plt.cm.tab10.colors

for i, lid in enumerate(unique_lids):
    idx = labels == lid
    plt.scatter(
        embeddings_tsne[idx, 0], embeddings_tsne[idx, 1],
        label=f"Layer {lid}",
        alpha=0.7,
        edgecolors='k',
        linewidths=0.3,
        s=30,
        color=colors[i % len(colors)]
    )

plt.xticks([])
plt.yticks([])
plt.grid(linestyle='--', alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()

# === 保存图片 ===
save_path = "./tsne_trajectories.pdf"
plt.savefig(save_path, dpi=300, format="pdf")
print(f"t-SNE figure saved to {save_path}")

plt.show()
