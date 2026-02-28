import os
import glob
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use("pgf")  # 输出 LaTeX 原生 pgf

import matplotlib.pyplot as plt

# ==========================================
# 1. 设置 IROS/ICRA 常用 LaTeX/PGF 绘图风格
# ==========================================
plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",

    # 更接近双栏论文图常用字号
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,

    "figure.dpi": 300,
    "lines.linewidth": 1.5,
})

# ==========================================
# 2. 数据处理函数
# ==========================================
def analyze_pair(hand_file, ur5_file):
    try:
        h_df = pd.read_csv(hand_file).sort_values("Timestamp_Epoch")
        u_df = pd.read_csv(ur5_file).sort_values("timestamp_epoch")

        results = {}

        # --- A. 计算灵敏度 (MDF) ---
        p0_h = h_df[h_df["Phase"] == 0]
        if p0_h.empty:
            return None

        t_end_p0 = p0_h["Timestamp_Epoch"].max()
        p0_u = u_df[u_df["timestamp_epoch"] <= t_end_p0]

        h_noise = (p0_h["Index_Force_N"] + p0_h["Thumb_Force_N"]).std()
        u_noise = p0_u["wrist_force_N"].std()
        print(
            f"Debug: {os.path.basename(hand_file)} - Hand Noise: {h_noise:.3f} N, Wrist Noise: {u_noise:.3f} N"
        )

        results["Hand_MDF"] = 3 * h_noise
        results["Wrist_MDF"] = 3 * u_noise

        # --- B. 计算稳定性 (CV) ---
        p2_h = h_df[h_df["Phase"] == 2]
        if not p2_h.empty:
            t_start, t_end = p2_h["Timestamp_Epoch"].min(), p2_h["Timestamp_Epoch"].max()
            p2_u = u_df[(u_df["timestamp_epoch"] >= t_start) & (u_df["timestamp_epoch"] <= t_end)]

            h_force = p2_h["Index_Force_N"] + p2_h["Thumb_Force_N"]
            u_force = p2_u["wrist_force_N"]

            if h_force.mean() > 0.1:
                results["Hand_CV"] = h_force.std() / h_force.mean()
            if u_force.mean() > 0.1:
                results["Wrist_CV"] = u_force.std() / u_force.mean()

        return results

    except Exception as e:
        print(f"Skipping {hand_file}: {e}")
        return None


# ==========================================
# 3. 批量执行
# ==========================================
all_hand_files = glob.glob("hand_data_*.csv")
all_ur5_files = glob.glob("ur5_task_data_*.csv")

hand_files = sorted([f for f in all_hand_files if "_old" not in f])
ur5_files = sorted([f for f in all_ur5_files if "_old" not in f])

print(f"Found {len(all_hand_files)} total files.")
print(f"Processing {len(hand_files)} clean files (excluded {len(all_hand_files) - len(hand_files)} '_old' files)...")
print(f"Processing {len(hand_files)} experiments...")

all_results = []

for h_f in hand_files:
    try:
        h_start = pd.read_csv(h_f, nrows=1)["Timestamp_Epoch"].iloc[0]
        best_u = min(
            ur5_files,
            key=lambda u: abs(pd.read_csv(u, nrows=1)["timestamp_epoch"].iloc[0] - h_start),
        )
        res = analyze_pair(h_f, best_u)
        if res:
            all_results.append(res)
    except:
        continue

df = pd.DataFrame(all_results)
print(f"Analysis done. Valid trials: {len(df)}")

# ==========================================
# 4. 生成两张核心图表 (PGF)
# ==========================================
def create_boxplot(data_col1, data_col2, title, ylabel, filename_base):
    if df.empty or data_col1 not in df.columns or data_col2 not in df.columns:
        return

    # 单栏宽度 3.35in，boxplot 这种高度 2.4in 左右通常合适
    fig, ax = plt.subplots(figsize=(3.35, 2.4))

    data = [df[data_col1].dropna(), df[data_col2].dropna()]
    labels = ["Hand", "Wrist"]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
    )

    colors = ["#aec7e8", "#ff9896"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Improvement 标注改成轴坐标放置，避免不同数据范围导致位置乱跳
    med_hand = df[data_col1].median()
    med_wrist = df[data_col2].median()
    if med_hand > 0:
        ratio = med_wrist / med_hand
        if "MDF" in data_col1:
            text = f"{ratio:.1f}x More Sensitive"
        else:
            text = f"{ratio:.1f}x More Stable"

        ax.text(
            0.5, 0.92, text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7,
            fontweight="bold",
            color="darkred",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
        )

    fig.tight_layout(pad=0.2)

    fig.savefig(f"{filename_base}.pgf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{filename_base}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)

    plt.close(fig)
    print(f"Saved {filename_base}.pgf")


create_boxplot(
    "Hand_MDF",
    "Wrist_MDF",
    "Fig. 1: Minimum Detectable Force (Sensitivity)",
    r"MDF ($N$) [Lower is Better]",
    "Sensitivity_MDF",
)

create_boxplot(
    "Hand_CV",
    "Wrist_CV",
    "Fig. 2: Insert Stability",
    "Coefficient of Variation (CV)",
    "Stability_CV",
)

print("\nSuccess! Generated 2 figures.")

# ==========================================
# 5. PRINT FINAL STATISTICS FOR PAPER
# ==========================================
print("\n" + "=" * 40)
print("FINAL RESULTS (For Paper / Manual Record)")
print("=" * 40)

hand_sigmas = df["Hand_MDF"] / 3
wrist_sigmas = df["Wrist_MDF"] / 3

print(f"\n[TYPICAL NOISE] (Use these for 'Sensors have a noise floor of X')")
print(f"Hand Average Sigma:  {hand_sigmas.mean():.4f} N")
print(f"Wrist Average Sigma: {wrist_sigmas.mean():.4f} N")

print(f"\n[WORST CASE] (Use these for setting safe thresholds)")
print(f"Hand Max Sigma:      {hand_sigmas.max():.4f} N")
print(f"Wrist Max Sigma:     {wrist_sigmas.max():.4f} N")

improvement = wrist_sigmas.mean() / hand_sigmas.mean()
print(f"\n[CONCLUSION]")
print(f"The Hand is {improvement:.1f}x more sensitive than the Wrist.")
print("=" * 40)