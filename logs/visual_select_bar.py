import matplotlib.pyplot as plt
import numpy as np

# === 全局字体设置 ===
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# === 子策略数 ===
num_policies = 3

# === 层名 ===
layer_names = ["Layer 0", "Layer 1", "Layer 2"]

# === 训练策略数据 ===
train_counts = np.array([
    [44, 16, 392],   # Layer 0
    [252, 40, 160],  # Layer 1
    [28, 412, 12]    # Layer 2
])

# === 蒸馏策略数据 ===
distill_counts = np.array([
    [44, 12, 396],   # Layer 0
    [263, 42, 147],  # Layer 1
    [18, 431, 3]     # Layer 2
])

num_layers = train_counts.shape[0]
x = np.arange(num_layers)

# 每个 policy 内的两根小柱之间宽度
bar_width = 0.15
policy_group_width = 2 * bar_width
total_policies_width = num_policies * policy_group_width
start_offset = -total_policies_width / 2 + bar_width / 2

fig, ax = plt.subplots(figsize=(10, 6))

# 为每个 policy 固定颜色
colors = plt.cm.tab10.colors

for i in range(num_policies):
    offset = start_offset + i * policy_group_width

    # Train 柱子（纯色）
    ax.bar(x + offset - bar_width / 2, train_counts[:, i], bar_width,
           label=f"Policy {i} (Train)", color=colors[i], alpha=0.9)

    # Distill 柱子（相同颜色 + hatch）
    ax.bar(x + offset + bar_width / 2, distill_counts[:, i], bar_width,
           label=f"Policy {i} (Distill)", color=colors[i], hatch='//', alpha=0.7)

ax.set_ylabel("Selected Times")
ax.set_xticks(x)
ax.set_xticklabels(layer_names)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# legend 放在顶部，分两行，防止遮挡
ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))

plt.tight_layout()

# 保存
plt.savefig("policy_selection_comparison.pdf", dpi=300, format="pdf")
print("Figure saved to policy_selection_comparison.pdf")

plt.show()
