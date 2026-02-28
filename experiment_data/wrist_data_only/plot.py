import os
import glob
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use("pgf")

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.5,
})

def plot_phase2_normalized_spaghetti_and_mean(data_dir, file_pattern="ur5_task_data_*.csv"):
    file_list = glob.glob(os.path.join(data_dir, file_pattern))
    if not file_list:
        print("未找到任何数据文件，请检查路径和文件名格式！")
        return

    print(f"共找到 {len(file_list)} 个文件，开始处理和绘制 Phase 2 的数据...")

    num_points = 500
    common_t = np.linspace(0, 1, num_points)

    wrist_all, index_all, thumb_all = [], [], []

    # 关键修改 1: 用论文单栏宽度做 figsize，避免 LaTeX 再大幅缩放
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 5.2), sharex=True)

    def normalize_force(series):
        mn, mx = series.min(), series.max()
        if mx - mn == 0:
            return np.zeros_like(series)
        return (series - mn) / (mx - mn)

    valid_files_count = 0

    for file in file_list:
        df = pd.read_csv(file)

        df_phase2 = df[df["hand_phase"] == 2].copy()
        if df_phase2.empty:
            print(f"文件 {file} 中没有 hand_phase == 2 的数据，已跳过。")
            continue

        # 关键修改 2: phase2 时间排序 + 去重，保证 interp 的 x 单调
        df_phase2 = df_phase2.sort_values("timestamp_epoch")
        df_phase2 = df_phase2.drop_duplicates(subset="timestamp_epoch", keep="first")

        if len(df_phase2) < 2:
            continue

        valid_files_count += 1

        t = df_phase2["timestamp_epoch"] - df_phase2["timestamp_epoch"].iloc[0]
        t_max = t.iloc[-1]
        t_norm = (t / t_max).to_numpy() if t_max > 0 else t.to_numpy()

        w_norm = normalize_force(df_phase2["wrist_f_norm_N"]).to_numpy()
        i_norm = normalize_force(df_phase2["index_force_N"]).to_numpy()
        th_norm = normalize_force(df_phase2["thumb_force_N"]).to_numpy()

        w_interp = np.interp(common_t, t_norm, w_norm)
        i_interp = np.interp(common_t, t_norm, i_norm)
        th_interp = np.interp(common_t, t_norm, th_norm)

        wrist_all.append(w_interp)
        index_all.append(i_interp)
        thumb_all.append(th_interp)

        axes[0].plot(common_t, w_interp, color="tab:blue", alpha=0.15)
        axes[1].plot(common_t, i_interp, color="tab:orange", alpha=0.15)
        axes[2].plot(common_t, th_interp, color="tab:green", alpha=0.15)

    if valid_files_count == 0:
        print("所有文件中都没有 phase 2 的数据！绘图终止。")
        return

    wrist_mean = np.mean(wrist_all, axis=0)
    index_mean = np.mean(index_all, axis=0)
    thumb_mean = np.mean(thumb_all, axis=0)

    axes[0].plot(common_t, wrist_mean, color="darkblue", linewidth=2.0, label="Mean Wrist Force")
    axes[1].plot(common_t, index_mean, color="darkred", linewidth=2.0, label="Mean Index Force")
    axes[2].plot(common_t, thumb_mean, color="darkgreen", linewidth=2.0, label="Mean Thumb Force")

    axes[0].set_title("Phase 2 - Normalized Wrist Force (Spaghetti + Mean)")
    axes[0].set_ylabel("Normalized (0-1)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("Phase 2 - Normalized Index Force (Spaghetti + Mean)")
    axes[1].set_ylabel("Normalized (0-1)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].set_title("Phase 2 - Normalized Thumb Force (Spaghetti + Mean)")
    axes[2].set_xlabel("Phase 2 Progression (Normalized Time 0-1)")
    axes[2].set_ylabel("Normalized (0-1)")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout(pad=0.2)

    fig.savefig("wrist_only.pgf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig("wrist_only.png", dpi=300, bbox_inches="tight", pad_inches=0.01)

    plt.show()
    print(f"成功绘制了 {valid_files_count} 个文件的 Phase 2 均值图。")

if __name__ == "__main__":
    plot_phase2_normalized_spaghetti_and_mean(".")