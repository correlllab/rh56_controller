import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import glob
import os
import matplotlib as mpl

# LaTeX-native output (PGF). Requires a working LaTeX (pdflatex).
mpl.use("pgf")

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage{newtxtext,newtxmath}",
})

import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FOLDER = "."
OUTPUT_FOLDER = "final_fixed_trim"
HAND_FILE_PATTERN = "hand_data*.csv"
UR5_FILE_PATTERN = "ur5_task_data*.csv"
INTERP_POINTS = 500

TRIM_SECONDS_FROM_END = 1.8

# Force mode:
#   "norm_01": min-max normalize each trial within the trimmed window (legacy behavior).
#   "baseline_N": subtract per-trial baseline and keep units in N (recommended to avoid misreading).
FORCE_MODE = "baseline_N"

# Baseline estimation if Phase=0 is unavailable or too short.
BASELINE_WINDOW_S = 0.30  # seconds from the beginning of the trimmed window

# Visualization-only smoothing for wrist median (does not affect spaghetti or statistics)
SMOOTH_WRIST_MEDIAN = True
SAVGOL_WINDOW = 21   # must be odd
SAVGOL_POLYORDER = 3

# Phase emphasis on the hand plot:
# Phase 0 and 2: Thumb is dominant
# Phase 1: Index is dominant
USE_PHASE_SHADING = True
PHASE_SHADE_ALPHA = 0.06
PHASE_LINE_ALPHA = 0.25

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize_01(series: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1]."""
    if series.size == 0:
        return series
    mn = np.nanmin(series)
    mx = np.nanmax(series)
    rng = mx - mn
    if not np.isfinite(rng) or rng == 0:
        return series - mn
    return (series - mn) / rng

def baseline_from_phase0(df: pd.DataFrame, col: str):
    if "Phase" not in df.columns:
        return None
    ph0 = df[df["Phase"] == 0]
    if len(ph0) < 5:
        return None
    val = float(np.nanmedian(ph0[col].to_numpy()))
    return val if np.isfinite(val) else None

def baseline_from_window(df: pd.DataFrame, t_col: str, col: str, t0: float, window_s: float):
    seg = df[(df[t_col] >= t0) & (df[t_col] <= (t0 + window_s))]
    if len(seg) < 5:
        return None
    val = float(np.nanmedian(seg[col].to_numpy()))
    return val if np.isfinite(val) else None

def apply_force_mode(df_hand: pd.DataFrame, df_wrist: pd.DataFrame, wrist_start: float, cutoff_time: float):
    """
    Returns (thumb, index, wrist) forces after applying FORCE_MODE.
    Arrays are aligned to df_hand and df_wrist rows (no interpolation here).
    """
    thumb = df_hand["Thumb_Force_N"].to_numpy(dtype=float)
    index = df_hand["Index_Force_N"].to_numpy(dtype=float) if "Index_Force_N" in df_hand.columns else None
    wrist = df_wrist["wrist_force_N"].to_numpy(dtype=float)

    if FORCE_MODE == "norm_01":
        thumb_out = normalize_01(thumb)
        index_out = normalize_01(index) if index is not None else None
        wrist_out = normalize_01(wrist)
        return thumb_out, index_out, wrist_out

    if FORCE_MODE == "baseline_N":
        # Hand baseline: prefer Phase 0, fallback to initial window
        b_thumb = baseline_from_phase0(df_hand, "Thumb_Force_N")
        if b_thumb is None:
            b_thumb = baseline_from_window(df_hand, "Timestamp_Epoch", "Thumb_Force_N", wrist_start, BASELINE_WINDOW_S) or 0.0

        b_index = None
        if index is not None:
            b_index = baseline_from_phase0(df_hand, "Index_Force_N")
            if b_index is None:
                b_index = baseline_from_window(df_hand, "Timestamp_Epoch", "Index_Force_N", wrist_start, BASELINE_WINDOW_S) or 0.0

        # Wrist baseline: use initial window in the trimmed segment
        b_wrist = baseline_from_window(df_wrist, "timestamp_epoch", "wrist_force_N", wrist_start, BASELINE_WINDOW_S) or 0.0

        thumb_out = thumb - b_thumb
        index_out = (index - b_index) if index is not None else None
        wrist_out = wrist - b_wrist
        return thumb_out, index_out, wrist_out

    raise ValueError(f"Unknown FORCE_MODE: {FORCE_MODE}")

def calculate_lag_normalized(t1, y1, t2, y2) -> float:
    """Estimate constant time shift to align y2 to y1 using cross-correlation on normalized signals."""
    y1_n = normalize_01(np.asarray(y1, dtype=float))
    y2_n = normalize_01(np.asarray(y2, dtype=float))

    dt = 0.01
    t_start = max(float(np.nanmin(t1)), float(np.nanmin(t2)))
    t_end = min(float(np.nanmax(t1)), float(np.nanmax(t2)))

    if not np.isfinite(t_start) or not np.isfinite(t_end) or (t_end - t_start) < 0.5:
        return 0.0

    t_grid = np.arange(t_start, t_end, dt)

    f1 = interp1d(t1, y1_n, kind="linear", fill_value="extrapolate")
    f2 = interp1d(t2, y2_n, kind="linear", fill_value="extrapolate")

    y1_g = f1(t_grid)
    y2_g = f2(t_grid)

    y1_d = signal.detrend(y1_g)
    y2_d = signal.detrend(y2_g)

    corr = signal.correlate(y1_d, y2_d, mode="full")
    lags = signal.correlation_lags(len(y1_d), len(y2_d), mode="full")
    return float(lags[int(np.argmax(corr))] * dt)

def band_iqr(mat: np.ndarray):
    """Median and IQR along trials, ignoring NaNs."""
    valid = np.any(~np.isnan(mat), axis=0)
    med = np.full(mat.shape[1], np.nan)
    p25 = np.full(mat.shape[1], np.nan)
    p75 = np.full(mat.shape[1], np.nan)
    if np.any(valid):
        med[valid] = np.nanmedian(mat[:, valid], axis=0)
        p25[valid] = np.nanpercentile(mat[:, valid], 25, axis=0)
        p75[valid] = np.nanpercentile(mat[:, valid], 75, axis=0)
    return med, p25, p75

def mask_interval(x: np.ndarray, y: np.ndarray, intervals):
    """Keep y only inside intervals, NaN elsewhere."""
    out = y.copy()
    keep = np.zeros_like(x, dtype=bool)
    for a, b in intervals:
        keep |= (x >= a) & (x <= b)
    out[~keep] = np.nan
    return out

# ==========================================
# MAIN PROCESSING
# ==========================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

hand_files = sorted(glob.glob(os.path.join(DATA_FOLDER, HAND_FILE_PATTERN)))
ur5_files = sorted(glob.glob(os.path.join(DATA_FOLDER, UR5_FILE_PATTERN)))
pairs = list(zip(hand_files, ur5_files))

all_thumb = []
all_index = []
all_wrist = []
all_lags = []

# phase boundaries per trial (in percent)
b01_list = []  # phase0 -> phase1 boundary
b12_list = []  # phase1 -> phase2 boundary

print(f"Processing {len(pairs)} pairs. Trimming last {TRIM_SECONDS_FROM_END}s...")

for i, (hand_f, ur5_f) in enumerate(pairs):
    try:
        df_hand = pd.read_csv(hand_f)
        df_ur5 = pd.read_csv(ur5_f)

        # Base timeline (wrist)
        wrist_start = float(df_ur5["timestamp_epoch"].min())
        wrist_end = float(df_ur5["timestamp_epoch"].max())

        cutoff_time = wrist_end - TRIM_SECONDS_FROM_END
        if cutoff_time <= wrist_start:
            print(f"  -> Skipping Pair {i+1}: recording too short for trim.")
            continue

        # Slice data to the trimmed window
        hand_slice = df_hand[
            (df_hand["Timestamp_Epoch"] >= wrist_start) &
            (df_hand["Timestamp_Epoch"] <= cutoff_time)
        ].copy()

        wrist_slice = df_ur5[
            (df_ur5["timestamp_epoch"] >= wrist_start) &
            (df_ur5["timestamp_epoch"] <= cutoff_time)
        ].copy()

        if hand_slice.empty or wrist_slice.empty:
            continue

        # Estimate log offset between streams (cross-correlation on thumb vs wrist)
        lag = calculate_lag_normalized(
            hand_slice["Timestamp_Epoch"].values,
            hand_slice["Thumb_Force_N"].values,
            wrist_slice["timestamp_epoch"].values,
            wrist_slice["wrist_force_N"].values,
        )
        all_lags.append(lag)

        # Apply force mode
        h_thumb, h_index, w_force = apply_force_mode(hand_slice, wrist_slice, wrist_start, cutoff_time)

        duration = cutoff_time - wrist_start

        # Time normalization (0-100% task progress)
        h_time_pct = ((hand_slice["Timestamp_Epoch"].values - wrist_start) / duration) * 100.0
        w_time_shifted = wrist_slice["timestamp_epoch"].values + lag
        w_time_pct = ((w_time_shifted - wrist_start) / duration) * 100.0

        # Interpolate to common grid
        x_grid = np.linspace(0, 100, INTERP_POINTS)
        f_thumb = interp1d(h_time_pct, h_thumb, kind="linear", bounds_error=False, fill_value=np.nan)(x_grid)
        all_thumb.append(f_thumb)

        if h_index is not None:
            f_index = interp1d(h_time_pct, h_index, kind="linear", bounds_error=False, fill_value=np.nan)(x_grid)
            all_index.append(f_index)

        f_w = interp1d(w_time_pct, w_force, kind="linear", bounds_error=False, fill_value=np.nan)(x_grid)
        all_wrist.append(f_w)

        # Phase boundaries from the hand log (convert to progress percent)
        if "Phase" in hand_slice.columns:
            if np.any(hand_slice["Phase"].values == 0) and np.any(hand_slice["Phase"].values == 1):
                t01 = float(hand_slice.loc[hand_slice["Phase"] == 0, "Timestamp_Epoch"].max())
                b01 = ((t01 - wrist_start) / duration) * 100.0
                if np.isfinite(b01):
                    b01_list.append(b01)

            if np.any(hand_slice["Phase"].values == 1) and np.any(hand_slice["Phase"].values == 2):
                t12 = float(hand_slice.loc[hand_slice["Phase"] == 1, "Timestamp_Epoch"].max())
                b12 = ((t12 - wrist_start) / duration) * 100.0
                if np.isfinite(b12):
                    b12_list.append(b12)

    except Exception as e:
        print(f"Error pair {i}: {e}")

# ==========================================
# PLOTTING (Scheme A)
# ==========================================
if all_thumb and all_wrist:
    thumb = np.asarray(all_thumb)              # (n_trials, n_points)
    wrist = np.asarray(all_wrist)

    has_index = len(all_index) == len(all_thumb) and len(all_index) > 0
    index = np.asarray(all_index) if has_index else None

    x = np.linspace(0, 100, INTERP_POINTS)
    n_trials = thumb.shape[0]
    avg_lag = float(np.mean(all_lags)) if len(all_lags) else 0.0

    # Robust summaries
    th_med, th_p25, th_p75 = band_iqr(thumb)
    wr_med, wr_p25, wr_p75 = band_iqr(wrist)

    if has_index:
        ix_med, ix_p25, ix_p75 = band_iqr(index)

    # Phase boundaries: median across trials
    b01 = float(np.median(b01_list)) if len(b01_list) else None
    b12 = float(np.median(b12_list)) if len(b12_list) else None

    # Optional smoothing of wrist median for visualization only
    wr_med_plot = wr_med.copy()
    if SMOOTH_WRIST_MEDIAN and np.sum(np.isfinite(wr_med_plot)) > SAVGOL_WINDOW and (SAVGOL_WINDOW % 2 == 1):
        good = np.isfinite(wr_med_plot)
        wr_med_plot[good] = signal.savgol_filter(wr_med_plot[good], SAVGOL_WINDOW, SAVGOL_POLYORDER)

    fig_w = 3.5
    fig_h = 2.9
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

    ylabel = "Norm. force" if FORCE_MODE == "norm_01" else "Force (N), baseline-corrected"

    # Hand panel
    ax = axes[0]

    for k in range(n_trials):
        ax.plot(x, thumb[k], linewidth=0.6, alpha=0.08)
    if has_index:
        for k in range(n_trials):
            ax.plot(x, index[k], linewidth=0.6, alpha=0.06)

    ax.plot(x, th_med, linewidth=1.0, label="Thumb (median)")
    if has_index:
        ax.plot(x, ix_med, linewidth=1.0, linestyle="--", label="Index (median)")

    if has_index and (b01 is not None) and (b12 is not None) and (0 < b01 < b12 < 100):
        if USE_PHASE_SHADING:
            ax.axvspan(0, b01, alpha=PHASE_SHADE_ALPHA)
            ax.axvspan(b01, b12, alpha=PHASE_SHADE_ALPHA)
            ax.axvspan(b12, 100, alpha=PHASE_SHADE_ALPHA)

        ax.axvline(b01, alpha=PHASE_LINE_ALPHA, linewidth=0.8)
        ax.axvline(b12, alpha=PHASE_LINE_ALPHA, linewidth=0.8)

        th_dom = mask_interval(x, th_med, [(0, b01), (b12, 100)])
        ix_dom = mask_interval(x, ix_med, [(b01, b12)])
        ax.plot(x, th_dom, linewidth=1.6)
        ax.plot(x, ix_dom, linewidth=1.6, linestyle="--")

        th_p25_dom = mask_interval(x, th_p25, [(0, b01), (b12, 100)])
        th_p75_dom = mask_interval(x, th_p75, [(0, b01), (b12, 100)])
        ix_p25_dom = mask_interval(x, ix_p25, [(b01, b12)])
        ix_p75_dom = mask_interval(x, ix_p75, [(b01, b12)])

        ax.fill_between(x, th_p25_dom, th_p75_dom, alpha=0.16)
        ax.fill_between(x, ix_p25_dom, ix_p75_dom, alpha=0.14)

        ax.text(0.01, 0.92, "Hand (Thumb in Ph0,2; Index in Ph1)", transform=ax.transAxes)
    else:
        ax.fill_between(x, th_p25, th_p75, alpha=0.16)
        ax.text(0.01, 0.92, "Hand", transform=ax.transAxes)

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    # Wrist panel
    ax = axes[1]
    for k in range(n_trials):
        ax.plot(x, wrist[k], linewidth=0.6, alpha=0.10)

    ax.plot(x, wr_med_plot, linewidth=1.4, label=f"Wrist (median, offset {avg_lag:.2f}s)")
    ax.fill_between(x, wr_p25, wr_p75, alpha=0.16)
    ax.set_xlabel("Task progress (\\%)")
    ax.set_ylabel(ylabel)
    ax.text(0.01, 0.92, "Wrist", transform=ax.transAxes)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    plt.tight_layout(pad=0.2)

    suffix = "schemeA_baselineN" if FORCE_MODE == "baseline_N" else "schemeA_norm01"
    base = f"{OUTPUT_FOLDER}/spaghetti_median_iqr_{suffix}"
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.savefig(base + ".pgf", bbox_inches="tight")
    plt.show()
    print(f"Saved to {base}.pdf and {base}.pgf")
else:
    print("No valid data found (need matching hand + ur5 files).")