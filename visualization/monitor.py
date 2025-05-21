import matplotlib.pyplot as plt
from typing import List, Optional
from envs.core_chain import Task


def plot_task_trajectories(task_list: List[Task], save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"A": "blue", "B": "green", "C": "orange"}
    markers = {False: "o", True: "x"}

    plotted = 0
    max_time = 0
    max_layer = 0
    for task in task_list:
        if not task.trajectory or plotted >= 10:
            continue
        layers, times = zip(*task.trajectory)
        max_time = max(max_time, max(times))
        max_layer = max(max_layer, max(layers))

        # 只对前10条轨迹使用通用标签，避免ID过大
        label = f"Task {plotted+1}" if plotted < 10 else None
        ax.plot(
            times, layers,
            linestyle="-", marker=markers[task.failed],
            color=colors.get(task.task_type, "gray"),
            label=label,
            alpha=0.7
        )
        plotted += 1

    if plotted == 0:
        print("[Warning] 无可绘制轨迹")

    ax.grid(True, which="major", linestyle="-")
    ax.grid(False, which="minor")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Layer")
    ax.set_title("Task Execution Trajectories")

    ax.set_xlim(0, max_time + 1)
    ax.set_ylim(-0.5, max_layer + 0.5)
    ax.set_yticks(list(range(0, max_layer + 1)))

    # 仅当有标签时显示图例
    if plotted:
        ax.legend(loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
