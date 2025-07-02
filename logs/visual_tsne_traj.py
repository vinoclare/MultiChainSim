import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# === 修改这里：存放 pkl 文件的目录 ===
pkl_dir = "./eval_trajectories"

# === 选择要分析的文件，例如最后几次 ===
pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")])

# === 存储所有 embedding 和标签 ===
embeddings = []
labels = []

for file_name in pkl_files:
    with open(os.path.join(pkl_dir, file_name), "rb") as f:
        eval_trajectories = pickle.load(f)

    # 每个文件是一个 list（每个 eval_episodes）
    for episode in eval_trajectories:
        for lid, traj in episode.items():
            task_obs_list = traj["task_obs"]  # list of (num_pad_tasks, task_dim)
            worker_loads_list = traj["worker_loads"]
            worker_profile_list = traj["worker_profile"]
            global_context_list = traj["global_context"]
            actions_list = traj["actions"]

            if len(task_obs_list) == 0:
                continue

            # === 把所有 step 拼接起来后做 mean pooling ===
            task_obs_all = np.stack(task_obs_list, axis=0).mean(axis=(0, 1))  # (task_dim,)
            worker_loads_all = np.stack(worker_loads_list, axis=0).mean(axis=(0, 1))  # (load_dim,)
            worker_profile_all = np.stack(worker_profile_list, axis=0).mean(axis=(0, 1))  # (profile_dim,)
            global_context_all = np.stack(global_context_list, axis=0).mean(axis=0)  # (global_context_dim,)
            actions_all = np.stack(actions_list, axis=0).mean(axis=(0, 1))  # (action_dim,)

            # === 拼成一个大向量 ===
            feat_vec = np.concatenate([task_obs_all.flatten(),
                                       worker_loads_all.flatten(),
                                       worker_profile_all.flatten(),
                                       global_context_all.flatten(),
                                       actions_all.flatten()], axis=0)

            embeddings.append(feat_vec)
            labels.append(lid)

# === 转 array ===
embeddings = np.stack(embeddings, axis=0)
labels = np.array(labels)

# === t-SNE 降维 ===
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

# === 绘图 ===
plt.figure(figsize=(8, 6))
for lid in np.unique(labels):
    idx = labels == lid
    plt.scatter(embeddings_tsne[idx, 0], embeddings_tsne[idx, 1], label=f"Layer {lid}", alpha=0.7)

plt.legend()
plt.title("t-SNE of Trajectories")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()

# === 保存图片 ===
save_path = "./tsne_trajectories.pdf"
plt.savefig(save_path, dpi=300, format="pdf")
print(f"t-SNE figure saved to {save_path}")

plt.show()
