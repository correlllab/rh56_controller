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
plt.rcParams.update(
    {
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
    }
)


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


def _trim_window_from_end(df, time_col: str, duration: float):
    t_end = float(df[time_col].max())
    t_start = t_end - duration
    return df[df[time_col] >= t_start].copy(), t_start, t_end


def _find_first_available_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _first_timestamp(csv_path: str, col: str):
    try:
        return float(pd.read_csv(csv_path, nrows=1)[col].iloc[0])
    except Exception:
        return None


def plot_phase2_mean_separate(
    data_dir=".",
    hand_pattern="hand_data_*.csv",
    ur5_exp1_pattern="ur5_data_*.csv",
    ur5_exp2_pattern="ur5_task_data_*.csv",
    exclude_old=True,
    num_points=500,
    out_base="phase2_mean_separate",
):
    wrist_band_scale = 1.0
    finger_band_scale = 0.5

    # ---------- collect files ----------
    hand_files = glob.glob(os.path.join(data_dir, hand_pattern))
    ur5_exp1_files = glob.glob(os.path.join(data_dir, ur5_exp1_pattern))
    ur5_exp2_files = glob.glob(os.path.join(data_dir, ur5_exp2_pattern))

    if exclude_old:
        hand_files = [f for f in hand_files if "_old" not in os.path.basename(f)]
        ur5_exp1_files = [
            f for f in ur5_exp1_files if "_old" not in os.path.basename(f)
        ]
        ur5_exp2_files = [
            f for f in ur5_exp2_files if "_old" not in os.path.basename(f)
        ]

    hand_files = sorted(hand_files)
    ur5_exp1_files = sorted(ur5_exp1_files)
    ur5_exp2_files = sorted(ur5_exp2_files)

    if not hand_files:
        print("未找到 hand_data 文件，请检查路径和文件名格式！")
        return
    if not ur5_exp1_files:
        print("未找到 ur5_data 文件，请检查路径和文件名格式！")
        return
    if not ur5_exp2_files:
        print("未找到 ur5_task_data 文件，请检查路径和文件名格式！")
        return

    pair_count = min(len(hand_files), len(ur5_exp2_files))
    if pair_count == 0:
        print("没有可配对的 hand / ur5_task 文件。")
        return
    if len(hand_files) != len(ur5_exp2_files):
        print(
            f"警告: hand 文件数 ({len(hand_files)}) 与 ur5_task 文件数 ({len(ur5_exp2_files)}) 不一致，"
            f"将按排序后的前 {pair_count} 对文件进行绘图。"
        )

    exp1_ur5_starts = []
    for ur5_exp1_file in ur5_exp1_files:
        ts = _first_timestamp(ur5_exp1_file, "timestamp_epoch")
        if ts is not None:
            exp1_ur5_starts.append((ur5_exp1_file, ts))

    if not exp1_ur5_starts:
        print("未能读取任何 ur5_data 文件的 timestamp_epoch。")
        return

    print(
        "Hand files: "
        f"{len(hand_files)}, Exp1 wrist files: {len(ur5_exp1_files)}, "
        f"Exp2 files: {len(ur5_exp2_files)}, Pairs used: {pair_count}"
    )
    print("开始处理并绘制 Phase 2 (separate experiments) 的 mean comparison...")

    common_t = np.linspace(0, 1, num_points)

    exp1_wrist_all = []
    exp1_index_all = []
    exp1_thumb_all = []
    exp2_wrist_all = []
    exp2_index_all = []
    exp2_thumb_all = []

    # Paper-friendly size (you can scale in LaTeX anyway)
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 5.4), sharex=True)

    valid_trials = 0

    for h_file, exp2_file in zip(hand_files[:pair_count], ur5_exp2_files[:pair_count]):
        h_start = _first_timestamp(h_file, "Timestamp_Epoch")
        if h_start is None:
            continue

        exp1_file, _ = min(exp1_ur5_starts, key=lambda it: abs(it[1] - h_start))

        try:
            h_df = pd.read_csv(h_file).sort_values("Timestamp_Epoch")
            exp1_df = pd.read_csv(exp1_file).sort_values("timestamp_epoch")
            exp2_df = pd.read_csv(exp2_file).sort_values("timestamp_epoch")
        except Exception as e:
            print(
                f"Skipping pair: {os.path.basename(h_file)} + "
                f"{os.path.basename(exp1_file)} + {os.path.basename(exp2_file)}: {e}"
            )
            continue

        hand_p2 = h_df[h_df["Phase"] == 2].copy()
        exp2_p2 = exp2_df[exp2_df["hand_phase"] == 2].copy()
        if hand_p2.empty or exp2_p2.empty:
            continue

        exp1_wrist_col = _find_first_available_column(
            exp1_df, ["wrist_force_N", "wrist_f_norm_N", "wrist_fz_N"]
        )
        exp2_wrist_col = _find_first_available_column(
            exp2_p2, ["wrist_force_N", "wrist_f_norm_N", "wrist_fz_N"]
        )
        if exp1_wrist_col is None or exp2_wrist_col is None:
            print(
                f"Skipping pair: {os.path.basename(h_file)} + "
                f"{os.path.basename(exp1_file)} + {os.path.basename(exp2_file)}: "
                "no wrist force column found."
            )
            continue

        hand_start = float(hand_p2["Timestamp_Epoch"].min())
        hand_end = float(hand_p2["Timestamp_Epoch"].max())
        hand_duration = hand_end - hand_start
        if (
            not np.isfinite(hand_start)
            or not np.isfinite(hand_end)
            or hand_duration <= 0
        ):
            continue

        # Exp1 wrist is from the same session as hand_data, so slice it with the
        # exact phase-2 absolute time window from hand_data.
        exp1_p2 = exp1_df[
            (exp1_df["timestamp_epoch"] >= hand_start)
            & (exp1_df["timestamp_epoch"] <= hand_end)
        ].copy()
        if exp1_p2.empty:
            continue

        # Use the full hand phase-2 duration as the reference and align both
        # datasets to the same phase-2 end by trimming Exp2 from the front.
        exp2_p2, exp2_start, _ = _trim_window_from_end(
            exp2_p2, "timestamp_epoch", hand_duration
        )
        if exp2_p2.empty:
            continue

        # hand time aligned to phase-2 progression (0..1)
        t_h = hand_p2["Timestamp_Epoch"].to_numpy(dtype=float)
        t_h_raw_norm = (t_h - hand_start) / hand_duration

        # Exp1 wrist shares the same absolute phase-2 window as hand_data.
        t_exp1 = exp1_p2["timestamp_epoch"].to_numpy(dtype=float)
        t_exp1_raw_norm = (t_exp1 - hand_start) / hand_duration

        # Exp2 is from a separate experiment, so use its own phase-2 end as the
        # fixed alignment point and keep only the last hand_duration seconds.
        t_exp2 = exp2_p2["timestamp_epoch"].to_numpy(dtype=float)
        t_exp2_raw_norm = (t_exp2 - exp2_start) / hand_duration

        # signals
        hand_idx = hand_p2["Index_Force_N"].to_numpy(dtype=float)
        hand_th = hand_p2["Thumb_Force_N"].to_numpy(dtype=float)
        exp1_wf = exp1_p2[exp1_wrist_col].to_numpy(dtype=float)
        exp2_wf = exp2_p2[exp2_wrist_col].to_numpy(dtype=float)
        exp2_idx = exp2_p2["index_force_N"].to_numpy(dtype=float)
        exp2_th = exp2_p2["thumb_force_N"].to_numpy(dtype=float)

        # prep for interp (sorted, unique time)
        t_h_norm, hand_idx = _prep_time_and_values(t_h_raw_norm, hand_idx)
        _, hand_th = _prep_time_and_values(t_h_raw_norm, hand_th)
        t_exp1_norm, exp1_wf = _prep_time_and_values(t_exp1_raw_norm, exp1_wf)
        t_exp2_norm, exp2_wf = _prep_time_and_values(t_exp2_raw_norm, exp2_wf)
        _, exp2_idx = _prep_time_and_values(t_exp2_raw_norm, exp2_idx)
        _, exp2_th = _prep_time_and_values(t_exp2_raw_norm, exp2_th)

        if t_h_norm.size < 2 or t_exp1_norm.size < 2 or t_exp2_norm.size < 2:
            continue

        # interpolate onto common_t
        exp1_wf_i = np.interp(common_t, t_exp1_norm, exp1_wf)
        exp1_idx_i = np.interp(common_t, t_h_norm, hand_idx)
        exp1_th_i = np.interp(common_t, t_h_norm, hand_th)
        exp2_wf_i = np.interp(common_t, t_exp2_norm, exp2_wf)
        exp2_idx_i = np.interp(common_t, t_exp2_norm, exp2_idx)
        exp2_th_i = np.interp(common_t, t_exp2_norm, exp2_th)

        exp1_wrist_all.append(exp1_wf_i)
        exp1_index_all.append(exp1_idx_i)
        exp1_thumb_all.append(exp1_th_i)
        exp2_wrist_all.append(exp2_wf_i)
        exp2_index_all.append(exp2_idx_i)
        exp2_thumb_all.append(exp2_th_i)

        valid_trials += 1

    if valid_trials == 0:
        print("没有任何有效 trial (phase 2 + pairing + window data) 可用于绘图。")
        return

    # mean curves
    exp1_wrist_stack = np.vstack(exp1_wrist_all)
    exp1_index_stack = np.vstack(exp1_index_all)
    exp1_thumb_stack = np.vstack(exp1_thumb_all)
    exp2_wrist_stack = np.vstack(exp2_wrist_all)
    exp2_index_stack = np.vstack(exp2_index_all)
    exp2_thumb_stack = np.vstack(exp2_thumb_all)

    exp1_wrist_mean = np.mean(exp1_wrist_stack, axis=0)
    exp1_index_mean = np.mean(exp1_index_stack, axis=0)
    exp1_thumb_mean = np.mean(exp1_thumb_stack, axis=0)
    exp2_wrist_mean = np.mean(exp2_wrist_stack, axis=0)
    exp2_index_mean = np.mean(exp2_index_stack, axis=0)
    exp2_thumb_mean = np.mean(exp2_thumb_stack, axis=0)

    exp1_wrist_std = np.std(exp1_wrist_stack, axis=0)
    exp1_index_std = np.std(exp1_index_stack, axis=0)
    exp1_thumb_std = np.std(exp1_thumb_stack, axis=0)
    exp2_wrist_std = np.std(exp2_wrist_stack, axis=0)
    exp2_index_std = np.std(exp2_index_stack, axis=0)
    exp2_thumb_std = np.std(exp2_thumb_stack, axis=0)

    axes[0].fill_between(
        common_t,
        exp1_wrist_mean - wrist_band_scale * exp1_wrist_std,
        exp1_wrist_mean + wrist_band_scale * exp1_wrist_std,
        color="tab:blue",
        alpha=0.18,
        linewidth=0,
    )
    axes[0].fill_between(
        common_t,
        exp2_wrist_mean - wrist_band_scale * exp2_wrist_std,
        exp2_wrist_mean + wrist_band_scale * exp2_wrist_std,
        color="tab:red",
        alpha=0.18,
        linewidth=0,
    )
    axes[1].fill_between(
        common_t,
        exp1_index_mean - finger_band_scale * exp1_index_std,
        exp1_index_mean + finger_band_scale * exp1_index_std,
        color="tab:blue",
        alpha=0.18,
        linewidth=0,
    )
    axes[1].fill_between(
        common_t,
        exp2_index_mean - finger_band_scale * exp2_index_std,
        exp2_index_mean + finger_band_scale * exp2_index_std,
        color="tab:red",
        alpha=0.18,
        linewidth=0,
    )
    axes[2].fill_between(
        common_t,
        exp1_thumb_mean - finger_band_scale * exp1_thumb_std,
        exp1_thumb_mean + finger_band_scale * exp1_thumb_std,
        color="tab:blue",
        alpha=0.18,
        linewidth=0,
    )
    axes[2].fill_between(
        common_t,
        exp2_thumb_mean - finger_band_scale * exp2_thumb_std,
        exp2_thumb_mean + finger_band_scale * exp2_thumb_std,
        color="tab:red",
        alpha=0.18,
        linewidth=0,
    )

    axes[0].plot(
        common_t,
        exp1_wrist_mean,
        color="tab:blue",
        linewidth=1.0,
        label="Exp 1 Wrist Mean",
    )
    axes[0].plot(
        common_t,
        exp2_wrist_mean,
        color="tab:red",
        linewidth=1.0,
        label="Exp 2 Wrist Mean",
    )
    axes[1].plot(
        common_t,
        exp1_index_mean,
        color="tab:blue",
        linewidth=1.0,
        label="Exp 1 Index Mean",
    )
    axes[1].plot(
        common_t,
        exp2_index_mean,
        color="tab:red",
        linewidth=1.0,
        label="Exp 2 Index Mean",
    )
    axes[2].plot(
        common_t,
        exp1_thumb_mean,
        color="tab:blue",
        linewidth=1.0,
        label="Exp 1 Thumb Mean",
    )
    axes[2].plot(
        common_t,
        exp2_thumb_mean,
        color="tab:red",
        linewidth=1.0,
        label="Exp 2 Thumb Mean",
    )

    # formatting
    axes[0].set_title("Phase 2 - Wrist Force Mean Comparison")
    axes[1].set_title("Phase 2 - Index Force Mean Comparison")
    axes[2].set_title("Phase 2 - Thumb Force Mean Comparison")

    for ax in axes:
        ax.set_ylabel("Force (N)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axes[2].set_xlabel("Phase 2 Progression (Aligned Time 0-1)")

    plt.tight_layout()

    # outputs
    plt.savefig(f"{out_base}.pgf")
    plt.savefig(f"{out_base}.png", dpi=300)
    plt.show()

    print(f"成功绘制 {valid_trials} 个 trials。Saved: {out_base}.pgf")


if __name__ == "__main__":
    DATA_DIRECTORY = "."
    plot_phase2_mean_separate(DATA_DIRECTORY)
