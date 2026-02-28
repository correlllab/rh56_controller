import os
import glob
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("pgf")  # output LaTeX-native PGF

import matplotlib.pyplot as plt

# ==========================================
# 0) IROS/ICRA style for PGF
# ==========================================
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

def _first_timestamp(csv_path: str, col: str):
    try:
        return float(pd.read_csv(csv_path, nrows=1)[col].iloc[0])
    except Exception:
        return None

def _normalize_minmax(values: np.ndarray):
    mn = np.nanmin(values)
    mx = np.nanmax(values)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mn) / (mx - mn)

def _prep_time_and_values(t: np.ndarray, y: np.ndarray):
    # sort by time and drop duplicate timestamps (keep first)
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    # drop duplicates in t
    uniq_t, uniq_idx = np.unique(t, return_index=True)
    t = uniq_t
    y = y[uniq_idx]
    return t, y

def plot_phase2_normalized_spaghetti_and_mean_separate(
    data_dir=".",
    hand_pattern="hand_data_*.csv",
    ur5_pattern="ur5_task_data_*.csv",
    exclude_old=True,
    num_points=500,
    out_base="phase2_normalized_spaghetti_mean_separate",
):
    # ---------- collect files ----------
    hand_files = glob.glob(os.path.join(data_dir, hand_pattern))
    ur5_files = glob.glob(os.path.join(data_dir, ur5_pattern))

    if exclude_old:
        hand_files = [f for f in hand_files if "_old" not in os.path.basename(f)]
        ur5_files = [f for f in ur5_files if "_old" not in os.path.basename(f)]

    hand_files = sorted(hand_files)
    ur5_files = sorted(ur5_files)

    if not hand_files:
        print("未找到 hand_data 文件，请检查路径和文件名格式！")
        return
    if not ur5_files:
        print("未找到 ur5_task_data 文件，请检查路径和文件名格式！")
        return

    # precompute ur5 start times
    ur5_starts = []
    for u in ur5_files:
        ts = _first_timestamp(u, "timestamp_epoch")
        if ts is not None:
            ur5_starts.append((u, ts))

    if not ur5_starts:
        print("未能读取任何 ur5 文件的 timestamp_epoch，绘图终止。")
        return

    print(f"Hand files: {len(hand_files)}, UR5 files: {len(ur5_files)}")
    print("开始处理并绘制 Phase 2 (separate files) 的 normalized spaghetti + mean...")

    common_t = np.linspace(0, 1, num_points)

    wrist_all = []
    index_all = []
    thumb_all = []

    # Paper-friendly size (you can scale in LaTeX anyway)
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 5.4), sharex=True)

    valid_trials = 0

    for h_file in hand_files:
        h_start = _first_timestamp(h_file, "Timestamp_Epoch")
        if h_start is None:
            continue

        # pair with closest UR5 by start time
        best_u, _ = min(ur5_starts, key=lambda it: abs(it[1] - h_start))

        try:
            h_df = pd.read_csv(h_file).sort_values("Timestamp_Epoch")
            u_df = pd.read_csv(best_u).sort_values("timestamp_epoch")
        except Exception as e:
            print(f"Skipping pair: {os.path.basename(h_file)} + {os.path.basename(best_u)}: {e}")
            continue

        # phase 2 window from hand
        p2 = h_df[h_df["Phase"] == 2].copy()
        if p2.empty:
            continue

        t_start = float(p2["Timestamp_Epoch"].min())
        t_end = float(p2["Timestamp_Epoch"].max())
        if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
            continue

        # ur5 data within phase 2 window
        u_p2 = u_df[(u_df["timestamp_epoch"] >= t_start) & (u_df["timestamp_epoch"] <= t_end)].copy()
        if u_p2.empty:
            continue

        # hand time normalize (0..1)
        t_h = p2["Timestamp_Epoch"].to_numpy(dtype=float)
        t_h = t_h - t_h[0]
        dur = t_h[-1] if t_h.size > 0 else 0.0
        if dur <= 0:
            continue
        t_h_norm = t_h / dur

        # ur5 time normalize based on same window
        t_u = u_p2["timestamp_epoch"].to_numpy(dtype=float)
        t_u = t_u - t_start
        dur_u = (t_end - t_start)
        if dur_u <= 0:
            continue
        t_u_norm = t_u / dur_u

        # signals
        idx = p2["Index_Force_N"].to_numpy(dtype=float)
        th = p2["Thumb_Force_N"].to_numpy(dtype=float)
        wf = u_p2["wrist_force_N"].to_numpy(dtype=float)

        # normalize within phase 2
        idx_n = _normalize_minmax(idx)
        th_n = _normalize_minmax(th)
        wf_n = _normalize_minmax(wf)

        # prep for interp (sorted, unique time)
        t_h_norm, idx_n = _prep_time_and_values(t_h_norm, idx_n)
        _, th_n = _prep_time_and_values(t_h_norm, th_n)  # same time base after dedup
        t_u_norm, wf_n = _prep_time_and_values(t_u_norm, wf_n)

        if t_h_norm.size < 2 or t_u_norm.size < 2:
            continue

        # interpolate onto common_t
        idx_i = np.interp(common_t, t_h_norm, idx_n)
        th_i = np.interp(common_t, t_h_norm, th_n)
        wf_i = np.interp(common_t, t_u_norm, wf_n)

        wrist_all.append(wf_i)
        index_all.append(idx_i)
        thumb_all.append(th_i)

        # spaghetti
        axes[0].plot(common_t, wf_i, color="tab:blue", alpha=0.15)
        axes[1].plot(common_t, idx_i, color="tab:orange", alpha=0.15)
        axes[2].plot(common_t, th_i, color="tab:green", alpha=0.15)

        valid_trials += 1

    if valid_trials == 0:
        print("没有任何有效 trial (phase 2 + pairing + window data) 可用于绘图。")
        return

    # mean curves
    wrist_mean = np.mean(np.vstack(wrist_all), axis=0)
    index_mean = np.mean(np.vstack(index_all), axis=0)
    thumb_mean = np.mean(np.vstack(thumb_all), axis=0)

    axes[0].plot(common_t, wrist_mean, color="darkblue", linewidth=2.2, label="Mean Wrist Force")
    axes[1].plot(common_t, index_mean, color="darkred", linewidth=2.2, label="Mean Index Force")
    axes[2].plot(common_t, thumb_mean, color="darkgreen", linewidth=2.2, label="Mean Thumb Force")

    # formatting
    axes[0].set_title("Phase 2 - Normalized Wrist Force (Spaghetti + Mean)")
    axes[1].set_title("Phase 2 - Normalized Index Force (Spaghetti + Mean)")
    axes[2].set_title("Phase 2 - Normalized Thumb Force (Spaghetti + Mean)")

    for ax in axes:
        ax.set_ylabel("Normalized (0-1)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axes[2].set_xlabel("Phase 2 Progression (Normalized Time 0-1)")

    plt.tight_layout()

    # outputs
    plt.savefig(f"{out_base}.pgf")
    plt.savefig(f"{out_base}.png", dpi=300)
    plt.show()

    print(f"成功绘制 {valid_trials} 个 trials。Saved: {out_base}.pgf")

if __name__ == "__main__":
    DATA_DIRECTORY = "."
    plot_phase2_normalized_spaghetti_and_mean_separate(DATA_DIRECTORY)