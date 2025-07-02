import matplotlib.pyplot as plt
import numpy as np

# === 子策略数 ===
num_policies = 3

# === 层名 ===
layer_names = ["Layer 0", "Layer 1", "Layer 2"]

# === 训练策略（Current Pids）数据 ===
train_counts = np.array([
    [44, 16, 392],   # Layer 0
    [252, 40, 160],  # Layer 1
    [28, 412, 12]    # Layer 2
])

# === 蒸馏策略（Distill）数据 ===
distill_counts = np.array([
    [44, 12, 396],   # Layer 0
    [263, 42, 147],  # Layer 1
    [18, 431, 3]     # Layer 2
])

num_layers = train_counts.shape[0]
x = np.arange(num_layers)

# 每个 policy 内的两根小柱之间宽度
bar_width = 0.15

# 每个 policy 在一组 layer 内总宽度
policy_group_width = 2 * bar_width  # Train + Distill

# 所有 policy 的总宽度
total_policies_width = num_policies * policy_group_width

# 起始偏移量（使柱子整体居中）
start_offset = -total_policies_width / 2 + bar_width / 2

fig, ax = plt.subplots(figsize=(10, 6))

for i in range(num_policies):
    # 每个 policy 在 x 上的偏移量
    offset = start_offset + i * policy_group_width

    # Train 柱子
    ax.bar(x + offset - bar_width / 2, train_counts[:, i], bar_width, label=f"Policy {i} (Train)")
    # Distill 柱子
    ax.bar(x + offset + bar_width / 2, distill_counts[:, i], bar_width, label=f"Policy {i} (Distill)", hatch='//', alpha=0.7)

ax.set_ylabel("Selected Times")
ax.set_title("Policy Selection per Layer (Train vs Distill)")
ax.set_xticks(x)
ax.set_xticklabels(layer_names)
ax.legend(ncol=3)

plt.tight_layout()

# === 保存 PDF 文件 ===
plt.savefig("policy_selection_comparison.pdf", dpi=300, format="pdf")
print("Figure saved to policy_selection_comparison.pdf")

plt.show()
