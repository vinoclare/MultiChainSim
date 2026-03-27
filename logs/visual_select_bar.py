import matplotlib.pyplot as plt
import numpy as np

# === 全局字体设置 ===
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,   # 保证导出 PDF 时字体更友好
    "ps.fonttype": 42
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

# === 更学术化的配色方案 ===
# Policy 0: 稳重蓝
# Policy 1: 柔和砖红
# Policy 2: 灰青色
colors = ['#4E79A7', '#E15759', '#76B7B2']

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
ax.set_facecolor('white')

for i in range(num_policies):
    offset = start_offset + i * policy_group_width

    # Train 柱子：实心填充
    ax.bar(
        x + offset - bar_width / 2,
        train_counts[:, i],
        bar_width,
        label=f"Policy {i} (Train)",
        color=colors[i],
        edgecolor='black',
        linewidth=0.6
    )

    # Distill 柱子：白底 + 同色边框 + 斜线
    ax.bar(
        x + offset + bar_width / 2,
        distill_counts[:, i],
        bar_width,
        label=f"Policy {i} (Distill)",
        facecolor='white',
        edgecolor=colors[i],
        linewidth=1.0,
        hatch='//'
    )

# === 坐标轴与网格 ===
ax.set_ylabel("Selected Times")
ax.set_xticks(x)
ax.set_xticklabels(layer_names)
ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.35)
ax.set_axisbelow(True)

# 去掉上边和右边框，让图更清爽
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# === 图例 ===
ax.legend(
    ncol=3,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.16),
    frameon=False
)

plt.tight_layout()

# === 保存 ===
plt.savefig("policy_selection_comparison.pdf", dpi=300, format="pdf", bbox_inches="tight")
plt.savefig("policy_selection_comparison.png", dpi=300, format="png", bbox_inches="tight")

print("Figure saved to policy_selection_comparison.pdf and policy_selection_comparison.png")

plt.show()