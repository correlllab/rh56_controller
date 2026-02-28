import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def plot_phase2_normalized_spaghetti_and_mean(
    data_dir, file_pattern="ur5_task_data_*.csv"
):
    file_list = glob.glob(os.path.join(data_dir, file_pattern))

    if not file_list:
        print("未找到任何数据文件，请检查路径和文件名格式！")
        return

    print(f"共找到 {len(file_list)} 个文件，开始处理和绘制 Phase 2 的数据...")

    # 创建通用的时间轴 (0 到 1 代表 Phase 2 进度的 0% 到 100%)
    num_points = 500
    common_t = np.linspace(0, 1, num_points)

    wrist_all = []
    index_all = []
    thumb_all = []

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 力的标准化函数 (仅在传入的 Series 内做 Min-Max)
    def normalize_force(series):
        mn, mx = series.min(), series.max()
        if mx - mn == 0:
            return np.zeros_like(series)
        return (series - mn) / (mx - mn)

    valid_files_count = 0

    for file in file_list:
        df = pd.read_csv(file)

        # 1. 核心修改：仅保留 hand_phase 为 2 的数据
        df_phase2 = df[df["hand_phase"] == 2].copy()

        # 如果这个文件没有 phase 2 的数据，则跳过
        if df_phase2.empty:
            print(f"文件 {file} 中没有 hand_phase == 2 的数据，已跳过。")
            continue

        valid_files_count += 1

        # 2. 重新计算相对时间，使得 phase 2 的第一点时间为 0，并标准化到 0 ~ 1
        t = df_phase2["timestamp_epoch"] - df_phase2["timestamp_epoch"].iloc[0]
        t_max = t.iloc[-1]
        t_norm = t / t_max if t_max > 0 else t

        # 3. 仅对 phase 2 的力进行标准化（完全不受 phase 1 的极值影响）
        w_norm = normalize_force(df_phase2["wrist_f_norm_N"])
        i_norm = normalize_force(df_phase2["index_force_N"])
        th_norm = normalize_force(df_phase2["thumb_force_N"])

        # 4. 插值到我们统一的 common_t (500个点) 上
        w_interp = np.interp(common_t, t_norm, w_norm)
        i_interp = np.interp(common_t, t_norm, i_norm)
        th_interp = np.interp(common_t, t_norm, th_norm)

        # 5. 保存到列表供计算均值
        wrist_all.append(w_interp)
        index_all.append(i_interp)
        thumb_all.append(th_interp)

        # 6. 绘制背景的 Spaghetti
        axes[0].plot(common_t, w_interp, color="tab:blue", alpha=0.15)
        axes[1].plot(common_t, i_interp, color="tab:orange", alpha=0.15)
        axes[2].plot(common_t, th_interp, color="tab:green", alpha=0.15)

    if valid_files_count == 0:
        print("所有文件中都没有 phase 2 的数据！绘图终止。")
        return

    # 计算试验的平均值
    wrist_mean = np.mean(wrist_all, axis=0)
    index_mean = np.mean(index_all, axis=0)
    thumb_mean = np.mean(thumb_all, axis=0)

    # 绘制平均值，用更粗的线条和深色突出显示
    axes[0].plot(
        common_t, wrist_mean, color="darkblue", linewidth=2.5, label="Mean Wrist Force"
    )
    axes[1].plot(
        common_t, index_mean, color="darkred", linewidth=2.5, label="Mean Index Force"
    )
    axes[2].plot(
        common_t, thumb_mean, color="darkgreen", linewidth=2.5, label="Mean Thumb Force"
    )

    # 图表格式化设置
    axes[0].set_title("Phase 2 - Normalized Wrist Force (Spaghetti + Mean)")
    axes[0].set_ylabel("Normalized Force (0-1)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("Phase 2 - Normalized Index Force (Spaghetti + Mean)")
    axes[1].set_ylabel("Normalized Force (0-1)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].set_title("Phase 2 - Normalized Thumb Force (Spaghetti + Mean)")
    axes[2].set_xlabel("Phase 2 Progression (Normalized Time 0 - 1)")
    axes[2].set_ylabel("Normalized Force (0-1)")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("phase2_normalized_spaghetti_mean.png", dpi=300)
    plt.show()

    print(f"成功绘制了 {valid_files_count} 个文件的 Phase 2 均值图。")


if __name__ == "__main__":
    # 指定你的CSV文件所在的文件夹路径
    DATA_DIRECTORY = "."
    plot_phase2_normalized_spaghetti_and_mean(DATA_DIRECTORY)
