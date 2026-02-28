#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Index position + raw force (_g) versus time across multiple trials:
- Align trials to a common time grid (interpolation)
- Plot mean line with IQR (25-75%) band
- Invert position y-axis with 1000 at the bottom
- Crop to (peak force time + 1s)
- Save both PNG and PGF (IROS-friendly rcParams)

Usage:
  1) Put all CSVs in the same folder (default: ./)
  2) python plot_index_mean_iqr.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
DATA_DIR = "."  # folder containing CSVs
CSV_GLOB = "*.csv"  # pattern
OUT_STEM = "index_pos_force_mean_iqr"  # output filename stem

POSITION_BASELINE = 1000.0  # y-axis bottom for position (inverted axis)
CROP_AFTER_PEAK_S = 1.0  # crop to peak + 1 sec

MAKE_COMPACT = True  # also save a compact version for paper
FIGSIZE = (8.0, 4.8)  # inches for compact
FIGSIZE_BIG = (10.0, 6.0)  # inches for big preview

DPI_PNG = 300

# IROS-ish styling (adjust if you already set these globally)
mpl.rcParams.update(
    {
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
    }
)


# ----------------------------
# Column detection helpers
# ----------------------------
def pick_time_column(df: pd.DataFrame) -> str | None:
    preferred = ["timestamp_epoch", "timestamp", "time", "t", "ts", "sec", "seconds"]
    cols_lower = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p in cols_lower:
            c = cols_lower[p]
            if pd.api.types.is_numeric_dtype(df[c]):
                return c

    # fallback: numeric column that is mostly increasing
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    best, best_score = None, -1
    for c in numeric_cols:
        s = df[c].dropna().to_numpy()
        if len(s) < 10:
            continue
        dif = np.diff(s)
        mono = np.mean(dif > 0)
        uniq = len(np.unique(s)) / len(s)
        score = mono * 0.7 + min(1.0, uniq) * 0.3
        if score > best_score:
            best_score = score
            best = c
    return best


def pick_index_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    cols = list(df.columns)
    cols_l = [c.lower() for c in cols]

    # Force raw: contains "index" and ends with "_g"
    force_candidates = [
        cols[i] for i, c in enumerate(cols_l) if ("index" in c and c.endswith("_g"))
    ]
    if not force_candidates:
        # slightly broader fallback
        force_candidates = [
            cols[i]
            for i, c in enumerate(cols_l)
            if ("index" in c and re.search(r"_g($|[^a-z0-9])", c))
        ]

    # Position: contains index and pos/position/angle/joint/q
    pos_candidates = []
    for i, c in enumerate(cols_l):
        if "index" not in c:
            continue
        if c.endswith("_g"):
            continue
        if any(
            k in c for k in ["pos", "position", "angle", "joint", "q", "rad", "deg"]
        ):
            pos_candidates.append(cols[i])
    if not pos_candidates:
        pos_candidates = [
            cols[i]
            for i, c in enumerate(cols_l)
            if ("index" in c and not c.endswith("_g"))
        ]

    pos_col = pos_candidates[0] if pos_candidates else None
    force_col = force_candidates[0] if force_candidates else None
    return pos_col, force_col


def to_seconds(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt_med = np.median(dt) if len(dt) else 0.01

    # Heuristic: detect epoch units by step size
    if dt_med > 1e6:
        scale = 1e-9  # ns
    elif dt_med > 1e3:
        scale = 1e-6  # us
    elif dt_med > 1:
        scale = 1e-3  # ms
    else:
        scale = 1.0  # seconds

    return (t - t[0]) * scale


# ----------------------------
# Data loading / alignment
# ----------------------------
def load_trial(path: str) -> dict:
    df = pd.read_csv(path)

    tcol = pick_time_column(df)
    pos_col, force_col = pick_index_columns(df)
    if tcol is None:
        raise ValueError(f"No suitable time column in {os.path.basename(path)}")
    if pos_col is None or force_col is None:
        raise ValueError(
            f"Missing index columns in {os.path.basename(path)}.\n"
            f"Detected pos_col={pos_col}, force_col={force_col}\n"
            f"Columns={list(df.columns)}"
        )

    t = to_seconds(pd.to_numeric(df[tcol], errors="coerce").to_numpy())
    pos = pd.to_numeric(df[pos_col], errors="coerce").to_numpy()
    force = pd.to_numeric(df[force_col], errors="coerce").to_numpy()

    return {
        "path": path,
        "t": t,
        "pos": pos,
        "force": force,
        "tcol": tcol,
        "pos_col": pos_col,
        "force_col": force_col,
    }


def build_common_grid(trials: list[dict]) -> np.ndarray:
    dts, durations = [], []
    for tr in trials:
        t = tr["t"]
        t = t[np.isfinite(t)]
        if len(t) > 3:
            dt = np.diff(t)
            dt = dt[np.isfinite(dt)]
            dt = dt[dt > 0]
            if len(dt):
                dts.append(np.median(dt))
        durations.append(np.nanmax(t) if len(t) else 0.0)

    dt_target = float(np.median(dts)) if dts else 0.01
    t_end = float(np.min(durations)) if durations else 0.0

    # cap points
    max_points = 6000
    n = int(np.floor(t_end / dt_target)) + 1 if dt_target > 0 else 0
    if n > max_points and t_end > 0:
        dt_target = t_end / (max_points - 1)

    return np.arange(0.0, t_end + 1e-12, dt_target)


def interp_on_grid(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t2, y2 = t[mask], y[mask]
    if len(t2) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)

    order = np.argsort(t2)
    t2, y2 = t2[order], y2[order]

    # remove duplicate t
    _, idx = np.unique(t2, return_index=True)
    t2, y2 = t2[idx], y2[idx]
    if len(t2) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)

    return np.interp(t_grid, t2, y2, left=np.nan, right=np.nan)


def mean_iqr(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(mat, axis=0)
    q25 = np.nanquantile(mat, 0.25, axis=0)
    q75 = np.nanquantile(mat, 0.75, axis=0)
    return mean, q25, q75


# ----------------------------
# Plotting
# ----------------------------
def plot_and_save(
    t: np.ndarray,
    pos_mean: np.ndarray,
    pos_q25: np.ndarray,
    pos_q75: np.ndarray,
    force_mean: np.ndarray,
    force_q25: np.ndarray,
    force_q75: np.ndarray,
    pos_label: str,
    force_label: str,
    out_png: str,
    out_pgf: str,
    figsize: tuple[float, float],
    show_legend: bool = True,
):
    plt.figure(figsize=figsize)

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, pos_mean, label="Mean")
    ax1.fill_between(t, pos_q25, pos_q75, alpha=0.25, label="IQR (25-75%)")
    ax1.set_ylabel(pos_label)
    ax1.grid(True, alpha=0.3)
    if show_legend:
        ax1.legend()

    # Invert y-axis with 1000 at bottom
    span = np.nanmax(pos_q75) - np.nanmin(pos_q25)
    y_top = np.nanmin(pos_q25) - 0.02 * (span + 1e-9)
    ax1.set_ylim(POSITION_BASELINE, y_top)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(t, force_mean, label="Mean")
    ax2.fill_between(t, force_q25, force_q75, alpha=0.25, label="IQR (25-75%)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(force_label)
    ax2.grid(True, alpha=0.3)
    if show_legend:
        ax2.legend()

    plt.tight_layout(pad=0.6)

    # Save
    plt.savefig(out_png, dpi=DPI_PNG, bbox_inches="tight")
    plt.savefig(out_pgf, bbox_inches="tight")
    plt.close()


def main():
    csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, CSV_GLOB)))
    if not csv_paths:
        raise RuntimeError(f"No CSV files found: {os.path.join(DATA_DIR, CSV_GLOB)}")

    trials = [load_trial(p) for p in csv_paths]
    t_grid = build_common_grid(trials)

    pos_mat = np.vstack([interp_on_grid(tr["t"], tr["pos"], t_grid) for tr in trials])
    force_mat = np.vstack(
        [interp_on_grid(tr["t"], tr["force"], t_grid) for tr in trials]
    )

    pos_mean, pos_q25, pos_q75 = mean_iqr(pos_mat)
    force_mean, force_q25, force_q75 = mean_iqr(force_mat)

    # Crop to peak + 1s (use median peak time across trials)
    peak_times = []
    for row in force_mat:
        if np.all(~np.isfinite(row)):
            continue
        peak_times.append(t_grid[int(np.nanargmax(row))])
    peak_t = (
        float(np.median(peak_times))
        if peak_times
        else float(t_grid[int(np.nanargmax(force_mean))])
    )
    t_crop_end = min(float(t_grid[-1]), peak_t + CROP_AFTER_PEAK_S)

    mask = t_grid <= t_crop_end
    t_plot = t_grid[mask]

    pos_mean_p, pos_q25_p, pos_q75_p = pos_mean[mask], pos_q25[mask], pos_q75[mask]
    force_mean_p, force_q25_p, force_q75_p = (
        force_mean[mask],
        force_q25[mask],
        force_q75[mask],
    )

    pos_col = trials[0]["pos_col"]
    force_col = trials[0]["force_col"]

    # Big (with legends)
    out_png = f"{OUT_STEM}_cropped_invert.png"
    out_pgf = f"{OUT_STEM}_cropped_invert.pgf"
    plot_and_save(
        t_plot,
        pos_mean_p,
        pos_q25_p,
        pos_q75_p,
        force_mean_p,
        force_q25_p,
        force_q75_p,
        pos_label=f"Index position ({pos_col})",
        force_label=f"Index force raw (g) ({force_col})",
        out_png=out_png,
        out_pgf=out_pgf,
        figsize=FIGSIZE_BIG,
        show_legend=True,
    )

    if MAKE_COMPACT:
        out_png_c = f"{OUT_STEM}_cropped_invert_compact.png"
        out_pgf_c = f"{OUT_STEM}_cropped_invert_compact.pgf"
        plot_and_save(
            t_plot,
            pos_mean_p,
            pos_q25_p,
            pos_q75_p,
            force_mean_p,
            force_q25_p,
            force_q75_p,
            pos_label="Index position",
            force_label="Index force raw (g)",
            out_png=out_png_c,
            out_pgf=out_pgf_c,
            figsize=FIGSIZE,
            show_legend=False,
        )

    print(f"Saved: {out_png}, {out_pgf}")
    if MAKE_COMPACT:
        print(f"Saved: {out_png_c}, {out_pgf_c}")


if __name__ == "__main__":
    main()
