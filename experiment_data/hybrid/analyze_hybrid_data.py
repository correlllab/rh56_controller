#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze RH56 hybrid experiment CSVs.

Outputs:
- analysis/contact_position_trial_metrics.csv
- analysis/contact_position_group_summary.csv
- analysis/force_time_overview.png
- analysis/force_time_overview.pgf

Definitions:
- Contact position = the angle at the first sustained, noticeable force rise during
  the `contact` stage.
- Force-time traces are aligned to that detected onset (t = 0) so overshoot is
  easier to compare across speeds.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev, pvariance
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "This script requires numpy and matplotlib. "
        "Install them first, then rerun analyze_hybrid_data.py."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "analysis"

FILE_RE = re.compile(
    r"^middlefinger_(?P<speed_label>speed\d+|hybrid)_force(?P<force>\d+)_20trials_.*\.csv$"
)

DEFAULT_BASELINE_WINDOW_S = 0.12
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_SIGMA_MULT = 4.0
DEFAULT_ABS_RISE_G = 20.0
DEFAULT_RISE_CONFIRM_G = 8.0
DEFAULT_MIN_CONSECUTIVE = 3
DEFAULT_PLOT_PRE_S = 0.15
DEFAULT_PLOT_POST_S = 3.0
DEFAULT_PLOT_DT_S = 0.01
DEFAULT_OUTLIER_Z_THRESHOLD = 3.5
DEFAULT_HYBRID_OUTLIER_Z_THRESHOLD = DEFAULT_OUTLIER_Z_THRESHOLD
DEFAULT_OUTLIER_MIN_GROUP = 6
PNG_DPI = 400
COMBINED_NORMALIZED_POINTS = 220

IROS_WIDTH_IN = 7.16
SUBPLOT_HEIGHT_IN = 1.95

SPEED_COLORS = {
    "speed25": "#4E79A7",
    "speed50": "#59A14F",
    "speed100": "#9C755F",
    "speed250": "#F28E2B",
    "speed500": "#E15759",
    "speed750": "#76B7B2",
    "speed1000": "#B07AA1",
    "hybrid": "#111111",
}

FORCE_COLORS = {
    100: "#5B8E7D",
    250: "#4E79A7",
    500: "#F28E2B",
    750: "#E15759",
    1000: "#7A4D8B",
}


@dataclass
class Sample:
    trial_index: int
    trial_name: str
    mode: str
    commanded_speed: int
    stage: str
    trial_elapsed_s: float
    angle: int
    force_g: float
    force_limit_g: int
    force_trigger_g: int


@dataclass
class TrialMetric:
    source_file: str
    mode: str
    speed_label: str
    speed_value: str
    force_limit_g: int
    force_trigger_g: int
    trial_index: int
    trial_name: str
    sample_count: int
    dt_median_s: float
    baseline_mean_g: float
    baseline_std_g: float
    onset_threshold_g: float
    onset_trial_elapsed_s: float
    onset_force_g: float
    onset_angle: int
    peak_force_g: float
    peak_force_trial_elapsed_s: float
    peak_force_angle: int
    overshoot_g: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(SCRIPT_DIR),
        help="Directory containing experiment CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for generated analysis files",
    )
    parser.add_argument(
        "--baseline-window-s",
        type=float,
        default=DEFAULT_BASELINE_WINDOW_S,
        help="Initial contact-stage window used to estimate baseline force noise",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Rolling mean window size for force smoothing",
    )
    parser.add_argument(
        "--sigma-mult",
        type=float,
        default=DEFAULT_SIGMA_MULT,
        help="Multiplier on baseline std for onset detection",
    )
    parser.add_argument(
        "--abs-rise-g",
        type=float,
        default=DEFAULT_ABS_RISE_G,
        help="Minimum absolute force rise above baseline for onset detection",
    )
    parser.add_argument(
        "--rise-confirm-g",
        type=float,
        default=DEFAULT_RISE_CONFIRM_G,
        help="Required rise across the sustained onset confirmation window",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=DEFAULT_MIN_CONSECUTIVE,
        help="Number of consecutive points required for sustained onset",
    )
    parser.add_argument(
        "--plot-pre-s",
        type=float,
        default=DEFAULT_PLOT_PRE_S,
        help="Time shown before onset in the force-time plot",
    )
    parser.add_argument(
        "--plot-post-s",
        type=float,
        default=DEFAULT_PLOT_POST_S,
        help="Time shown after onset in the force-time plot",
    )
    parser.add_argument(
        "--plot-dt-s",
        type=float,
        default=DEFAULT_PLOT_DT_S,
        help="Interpolation step for the mean force-time plot",
    )
    parser.add_argument(
        "--outlier-z-threshold",
        type=float,
        default=DEFAULT_OUTLIER_Z_THRESHOLD,
        help="Modified z-score threshold for plot-only outlier rejection",
    )
    parser.add_argument(
        "--hybrid-outlier-z-threshold",
        type=float,
        default=DEFAULT_HYBRID_OUTLIER_Z_THRESHOLD,
        help="Modified z-score threshold applied only to hybrid plot traces",
    )
    parser.add_argument(
        "--outlier-min-group",
        type=int,
        default=DEFAULT_OUTLIER_MIN_GROUP,
        help="Minimum trials per force-speed group before outlier filtering is applied",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "savefig.dpi": PNG_DPI,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
        }
    )


def discover_csvs(input_dir: Path) -> List[Path]:
    files = []
    for path in sorted(input_dir.glob("*.csv")):
        if FILE_RE.match(path.name):
            files.append(path)
    return files


def parse_filename(path: Path) -> Tuple[str, str, int]:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected filename format: {path.name}")
    speed_label = match.group("speed_label")
    force_limit_g = int(match.group("force"))
    speed_value = "hybrid" if speed_label == "hybrid" else speed_label.replace("speed", "")
    return speed_label, speed_value, force_limit_g


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = max(1, int(window))
    if window == 1:
        return arr.copy()
    kernel = np.ones(window, dtype=float)
    smoothed = np.convolve(arr, kernel, mode="full")[: arr.size]
    counts = np.minimum(np.arange(1, arr.size + 1), window)
    return smoothed / counts


def median_dt(times: Sequence[float]) -> float:
    if len(times) < 2:
        return 0.0
    deltas = [times[i] - times[i - 1] for i in range(1, len(times)) if times[i] > times[i - 1]]
    if not deltas:
        return 0.0
    return median(deltas)


def load_trials_for_stages(
    csv_path: Path,
    stages: Sequence[str],
) -> Dict[int, List[Sample]]:
    stage_set = set(stages)
    trials: Dict[int, List[Sample]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Stage") not in stage_set:
                continue
            sample = Sample(
                trial_index=int(row["Trial_Index"]),
                trial_name=row["Trial"],
                mode=row["Mode"],
                commanded_speed=int(row["Commanded_Speed"]),
                stage=row["Stage"],
                trial_elapsed_s=float(row["Trial_Elapsed_s"]),
                angle=int(row["Middle_Angle"]),
                force_g=float(row["Middle_Force_g"]),
                force_limit_g=int(row["Force_Limit_g"]),
                force_trigger_g=int(row["Force_Trigger_g"]),
            )
            trials[sample.trial_index].append(sample)
    for samples in trials.values():
        samples.sort(key=lambda item: item.trial_elapsed_s)
    return dict(sorted(trials.items()))


def load_contact_trials(csv_path: Path) -> Dict[int, List[Sample]]:
    return load_trials_for_stages(csv_path, ["contact"])


def load_motion_trials(csv_path: Path) -> Dict[int, List[Sample]]:
    return load_trials_for_stages(csv_path, ["approach", "contact"])


def choose_onset_index(
    forces_smooth: np.ndarray,
    dt_s: float,
    baseline_window_s: float,
    sigma_mult: float,
    abs_rise_g: float,
    rise_confirm_g: float,
    min_consecutive: int,
) -> Tuple[int, float, float, float]:
    if forces_smooth.size == 0:
        raise ValueError("No contact-stage samples available")

    dt_ref = dt_s if dt_s > 0 else 0.006
    baseline_count = max(min_consecutive + 1, int(math.ceil(baseline_window_s / dt_ref)))
    baseline_count = min(forces_smooth.size, max(5, baseline_count))

    baseline = forces_smooth[:baseline_count]
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))
    threshold = baseline_mean + max(abs_rise_g, sigma_mult * max(baseline_std, 1.0))

    last_start = max(0, forces_smooth.size - min_consecutive)
    for i in range(baseline_count, last_start + 1):
        window = forces_smooth[i : i + min_consecutive]
        if window.size < min_consecutive:
            break
        if float(np.min(window)) < threshold:
            continue
        if float(window[-1] - window[0]) < rise_confirm_g:
            continue
        prev_value = float(forces_smooth[i - 1]) if i > 0 else baseline_mean
        if float(window[0]) < prev_value:
            continue
        return i, baseline_mean, baseline_std, threshold

    above = np.flatnonzero(forces_smooth >= threshold)
    if above.size > 0:
        idx = int(above[0])
        return idx, baseline_mean, baseline_std, threshold

    peak_idx = int(np.argmax(forces_smooth))
    return peak_idx, baseline_mean, baseline_std, threshold


def build_trial_metric(
    csv_path: Path,
    speed_label: str,
    speed_value: str,
    force_limit_g: int,
    samples: Sequence[Sample],
    args: argparse.Namespace,
) -> TrialMetric:
    if not samples:
        raise ValueError(f"Empty contact-stage trial in {csv_path.name}")

    times = [sample.trial_elapsed_s for sample in samples]
    forces = [sample.force_g for sample in samples]
    dt_s = median_dt(times)
    forces_smooth = rolling_mean(forces, args.smooth_window)
    onset_idx, baseline_mean, baseline_std, threshold = choose_onset_index(
        forces_smooth,
        dt_s,
        args.baseline_window_s,
        args.sigma_mult,
        args.abs_rise_g,
        args.rise_confirm_g,
        args.min_consecutive,
    )
    peak_idx = int(np.argmax(forces_smooth))
    onset_sample = samples[onset_idx]
    peak_sample = samples[peak_idx]

    return TrialMetric(
        source_file=csv_path.name,
        mode=onset_sample.mode,
        speed_label=speed_label,
        speed_value=speed_value,
        force_limit_g=force_limit_g,
        force_trigger_g=onset_sample.force_trigger_g,
        trial_index=onset_sample.trial_index,
        trial_name=onset_sample.trial_name,
        sample_count=len(samples),
        dt_median_s=dt_s,
        baseline_mean_g=baseline_mean,
        baseline_std_g=baseline_std,
        onset_threshold_g=threshold,
        onset_trial_elapsed_s=onset_sample.trial_elapsed_s,
        onset_force_g=float(forces_smooth[onset_idx]),
        onset_angle=onset_sample.angle,
        peak_force_g=float(forces_smooth[peak_idx]),
        peak_force_trial_elapsed_s=peak_sample.trial_elapsed_s,
        peak_force_angle=peak_sample.angle,
        overshoot_g=float(forces_smooth[peak_idx]) - float(force_limit_g),
    )


def write_trial_metrics(output_path: Path, metrics: Sequence[TrialMetric]) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Source_File",
                "Mode",
                "Speed_Label",
                "Speed_Value",
                "Force_Limit_g",
                "Force_Trigger_g",
                "Trial_Index",
                "Trial_Name",
                "Sample_Count",
                "Median_dt_s",
                "Baseline_Mean_g",
                "Baseline_Std_g",
                "Onset_Threshold_g",
                "Onset_Trial_Elapsed_s",
                "Onset_Force_g",
                "Onset_Angle",
                "Peak_Force_g",
                "Peak_Force_Trial_Elapsed_s",
                "Peak_Force_Angle",
                "Overshoot_g",
            ]
        )
        for item in metrics:
            writer.writerow(
                [
                    item.source_file,
                    item.mode,
                    item.speed_label,
                    item.speed_value,
                    item.force_limit_g,
                    item.force_trigger_g,
                    item.trial_index,
                    item.trial_name,
                    item.sample_count,
                    f"{item.dt_median_s:.6f}",
                    f"{item.baseline_mean_g:.6f}",
                    f"{item.baseline_std_g:.6f}",
                    f"{item.onset_threshold_g:.6f}",
                    f"{item.onset_trial_elapsed_s:.6f}",
                    f"{item.onset_force_g:.6f}",
                    item.onset_angle,
                    f"{item.peak_force_g:.6f}",
                    f"{item.peak_force_trial_elapsed_s:.6f}",
                    item.peak_force_angle,
                    f"{item.overshoot_g:.6f}",
                ]
            )


def summarize_group(metrics: Sequence[TrialMetric]) -> Dict[str, float]:
    onset_angles = [item.onset_angle for item in metrics]
    onset_times = [item.onset_trial_elapsed_s for item in metrics]
    overshoots = [item.overshoot_g for item in metrics]
    peak_forces = [item.peak_force_g for item in metrics]
    return {
        "trial_count": float(len(metrics)),
        "onset_angle_mean": mean(onset_angles),
        "onset_angle_variance": pvariance(onset_angles) if len(onset_angles) > 1 else 0.0,
        "onset_angle_std": pstdev(onset_angles) if len(onset_angles) > 1 else 0.0,
        "onset_angle_min": float(min(onset_angles)),
        "onset_angle_max": float(max(onset_angles)),
        "onset_time_mean_s": mean(onset_times),
        "peak_force_mean_g": mean(peak_forces),
        "peak_force_variance_g2": pvariance(peak_forces) if len(peak_forces) > 1 else 0.0,
        "overshoot_mean_g": mean(overshoots),
        "overshoot_variance_g2": pvariance(overshoots) if len(overshoots) > 1 else 0.0,
    }


def pooled_variance_from_groups(groups: Sequence[Sequence[TrialMetric]]) -> Dict[str, float]:
    valid_groups = [list(group) for group in groups if group]
    if not valid_groups:
        return {
            "group_count": 0.0,
            "total_count": 0.0,
            "grand_mean": 0.0,
            "within_group_var_mean": 0.0,
            "between_group_var_of_means": 0.0,
            "pooled_total_var": 0.0,
        }

    group_ns = [len(group) for group in valid_groups]
    group_means = [mean(item.onset_angle for item in group) for group in valid_groups]
    group_vars = [
        pvariance([item.onset_angle for item in group]) if len(group) > 1 else 0.0
        for group in valid_groups
    ]
    total_count = sum(group_ns)
    grand_mean = sum(n * mu for n, mu in zip(group_ns, group_means)) / total_count

    within_group_var_mean = (
        sum(n * var for n, var in zip(group_ns, group_vars)) / total_count
        if total_count > 0
        else 0.0
    )
    between_group_var_of_means = (
        sum(n * ((mu - grand_mean) ** 2) for n, mu in zip(group_ns, group_means)) / total_count
        if total_count > 0
        else 0.0
    )
    pooled_total_var = within_group_var_mean + between_group_var_of_means

    return {
        "group_count": float(len(valid_groups)),
        "total_count": float(total_count),
        "grand_mean": grand_mean,
        "within_group_var_mean": within_group_var_mean,
        "between_group_var_of_means": between_group_var_of_means,
        "pooled_total_var": pooled_total_var,
    }


def modified_z_scores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad > 1e-12:
        return list(0.6745 * (arr - med) / mad)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    if iqr > 1e-12:
        robust_scale = iqr / 1.349
        return list((arr - med) / robust_scale)
    return [0.0 for _ in values]


def detect_plot_outliers(
    metrics: Sequence[TrialMetric],
    motion_trials_by_file: Dict[str, Dict[int, List[Sample]]],
    z_threshold: float,
    hybrid_z_threshold: float,
    min_group_size: int,
) -> Tuple[set[Tuple[str, int]], List[Dict[str, object]]]:
    grouped: Dict[Tuple[int, str], List[TrialMetric]] = defaultdict(list)
    for item in metrics:
        grouped[(item.force_limit_g, item.speed_label)].append(item)

    excluded: set[Tuple[str, int]] = set()
    rows: List[Dict[str, object]] = []

    for (force_limit_g, speed_label), group in grouped.items():
        if len(group) < min_group_size:
            continue
        group_threshold = hybrid_z_threshold if speed_label == "hybrid" else z_threshold

        peak_rel_times = []
        for item in group:
            motion_samples = motion_trials_by_file[item.source_file][item.trial_index]
            motion_start_s = min(sample.trial_elapsed_s for sample in motion_samples)
            peak_rel_times.append(item.peak_force_trial_elapsed_s - motion_start_s)

        feature_values = {
            "onset_angle": [float(item.onset_angle) for item in group],
            "peak_force_g": [float(item.peak_force_g) for item in group],
            "overshoot_g": [float(item.overshoot_g) for item in group],
            "peak_time_s": peak_rel_times,
        }
        feature_scores = {
            name: modified_z_scores(values)
            for name, values in feature_values.items()
        }

        for idx, item in enumerate(group):
            reasons = []
            for feature_name, scores in feature_scores.items():
                score = scores[idx]
                if abs(score) > group_threshold:
                    reasons.append(f"{feature_name}:{score:.2f}")
            if reasons:
                key = (item.source_file, item.trial_index)
                excluded.add(key)
                rows.append(
                    {
                        "source_file": item.source_file,
                        "force_limit_g": force_limit_g,
                        "speed_label": speed_label,
                        "trial_index": item.trial_index,
                        "trial_name": item.trial_name,
                        "threshold": group_threshold,
                        "reasons": ";".join(reasons),
                    }
                )

    rows.sort(
        key=lambda row: (
            int(row["force_limit_g"]),
            overshoot_speed_order(str(row["speed_label"])),
            int(row["trial_index"]),
        )
    )
    return excluded, rows


def write_plot_outliers(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Source_File",
                "Force_Limit_g",
                "Speed_Label",
                "Trial_Index",
                "Trial_Name",
                "Threshold",
                "Reasons",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["source_file"],
                    row["force_limit_g"],
                    row["speed_label"],
                    row["trial_index"],
                    row["trial_name"],
                    row["threshold"],
                    row["reasons"],
                ]
            )


def write_group_summary(output_path: Path, metrics: Sequence[TrialMetric]) -> None:
    by_file: Dict[str, List[TrialMetric]] = defaultdict(list)
    by_speed: Dict[str, List[TrialMetric]] = defaultdict(list)
    by_force_speed: Dict[Tuple[int, str], List[TrialMetric]] = defaultdict(list)

    for item in metrics:
        by_file[item.source_file].append(item)
        by_speed[item.speed_label].append(item)
        by_force_speed[(item.force_limit_g, item.speed_label)].append(item)

    rows: List[List[object]] = []
    file_groups = [group for _, group in sorted(by_file.items())]

    def append_rows(
        level: str,
        source_file: str,
        mode: str,
        speed_label: str,
        speed_value: str,
        force_limit_g: str,
        group_metrics: Sequence[TrialMetric],
    ) -> None:
        s = summarize_group(group_metrics)
        rows.append(
            [
                level,
                source_file,
                mode,
                speed_label,
                speed_value,
                force_limit_g,
                int(s["trial_count"]),
                f"{s['onset_angle_mean']:.6f}",
                f"{s['onset_angle_variance']:.6f}",
                f"{s['onset_angle_std']:.6f}",
                int(s["onset_angle_min"]),
                int(s["onset_angle_max"]),
                f"{s['onset_time_mean_s']:.6f}",
                f"{s['peak_force_mean_g']:.6f}",
                f"{s['peak_force_variance_g2']:.6f}",
                f"{s['overshoot_mean_g']:.6f}",
                f"{s['overshoot_variance_g2']:.6f}",
                "",
                "",
            ]
        )

    for source_file, group_metrics in sorted(by_file.items()):
        first = group_metrics[0]
        append_rows(
            "file",
            source_file,
            first.mode,
            first.speed_label,
            first.speed_value,
            str(first.force_limit_g),
            group_metrics,
        )

    for speed_label, group_metrics in sorted(by_speed.items(), key=lambda item: nice_speed_order(item[0])):
        first = group_metrics[0]
        append_rows(
            "speed",
            "",
            first.mode,
            speed_label,
            first.speed_value,
            "",
            group_metrics,
        )

    append_rows("overall", "", "all", "all", "all", "", metrics)

    pooled = pooled_variance_from_groups(file_groups)
    rows.append(
        [
            "overall_decomposed",
            "",
            "all",
            "all",
            "all",
            "",
            int(pooled["total_count"]),
            f"{pooled['grand_mean']:.6f}",
            f"{pooled['pooled_total_var']:.6f}",
            f"{math.sqrt(pooled['pooled_total_var']):.6f}",
            "",
            "",
            "",
            "",
            "",
            "",
            f"{pooled['within_group_var_mean']:.6f}",
            f"{pooled['between_group_var_of_means']:.6f}",
        ]
    )

    for (force_limit_g, speed_label), group_metrics in sorted(
        by_force_speed.items(),
        key=lambda item: (item[0][0], nice_speed_order(item[0][1])),
    ):
        first = group_metrics[0]
        append_rows(
            "force_speed",
            "",
            first.mode,
            speed_label,
            first.speed_value,
            str(force_limit_g),
            group_metrics,
        )

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Summary_Level",
                "Source_File",
                "Mode",
                "Speed_Label",
                "Speed_Value",
                "Force_Limit_g",
                "Trial_Count",
                "Onset_Angle_Mean",
                "Onset_Angle_Variance",
                "Onset_Angle_Std",
                "Onset_Angle_Min",
                "Onset_Angle_Max",
                "Onset_Time_Mean_s",
                "Peak_Force_Mean_g",
                "Peak_Force_Variance_g2",
                "Overshoot_Mean_g",
                "Overshoot_Variance_g2",
                "Within_CSV_Var_Mean",
                "Between_CSV_Var_Of_Means",
            ]
        )
        writer.writerows(rows)


def make_time_grid(pre_s: float, post_s: float, dt_s: float) -> np.ndarray:
    count = int(math.floor((pre_s + post_s) / dt_s)) + 1
    return np.linspace(-pre_s, post_s, count)


def interpolate_trace(
    samples: Sequence[Sample],
    motion_start_s: float,
    peak_time_s: float,
    grid: np.ndarray,
) -> np.ndarray:
    times = np.asarray([sample.trial_elapsed_s - motion_start_s for sample in samples], dtype=float)
    forces = np.asarray([sample.force_g for sample in samples], dtype=float)
    if times.size == 0:
        return np.full(grid.shape, np.nan)
    peak_rel_s = max(0.0, peak_time_s - motion_start_s)
    valid = (grid >= times[0]) & (grid <= times[-1]) & (grid <= peak_rel_s)
    values = np.full(grid.shape, np.nan)
    values[valid] = np.interp(grid[valid], times, forces)
    return values


def aggregate_plot_data(
    motion_trials_by_file: Dict[str, Dict[int, List[Sample]]],
    metrics: Sequence[TrialMetric],
    pre_s: float,
    post_s: float,
    dt_s: float,
    excluded_keys: Optional[set[Tuple[str, int]]] = None,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Dict[str, np.ndarray]]], float]:
    metric_map = {(item.source_file, item.trial_index): item for item in metrics}
    grouped_trials: Dict[int, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    max_peak_rel_s = 0.0
    pending_traces: List[Tuple[int, str, Sequence[Sample], float, float]] = []
    excluded = excluded_keys or set()

    for source_file, trials in motion_trials_by_file.items():
        for trial_index, samples in trials.items():
            if (source_file, trial_index) in excluded:
                continue
            metric = metric_map.get((source_file, trial_index))
            if metric is None:
                continue
            motion_start_s = min(sample.trial_elapsed_s for sample in samples)
            peak_rel_s = max(
                0.0, metric.peak_force_trial_elapsed_s - motion_start_s
            )
            max_peak_rel_s = max(max_peak_rel_s, peak_rel_s)
            pending_traces.append(
                (
                    metric.force_limit_g,
                    metric.speed_label,
                    samples,
                    motion_start_s,
                    metric.peak_force_trial_elapsed_s,
                )
            )

    effective_post_s = max(post_s, max_peak_rel_s + dt_s)
    grid = make_time_grid(pre_s, effective_post_s, dt_s)

    for force_limit_g, speed_label, samples, motion_start_s, peak_time_s in pending_traces:
        trace = interpolate_trace(
            samples,
            motion_start_s,
            peak_time_s,
            grid,
        )
        grouped_trials[force_limit_g][speed_label].append(trace)

    aggregated: Dict[int, Dict[str, Dict[str, np.ndarray]]] = defaultdict(dict)
    for force_limit_g, speed_map in grouped_trials.items():
        for speed_label, traces in speed_map.items():
            matrix = np.vstack(traces)
            aggregated[force_limit_g][speed_label] = {
                "mean": np.nanmean(matrix, axis=0),
                "std": np.nanstd(matrix, axis=0),
                "count": np.sum(~np.isnan(matrix), axis=0),
            }
    return grid, dict(sorted(aggregated.items())), max_peak_rel_s


def nice_speed_order(speed_label: str) -> Tuple[int, str]:
    if speed_label == "hybrid":
        return (10**9, speed_label)
    return (int(speed_label.replace("speed", "")), speed_label)


def overshoot_speed_order(speed_label: str) -> Tuple[int, str]:
    if speed_label == "hybrid":
        return (-1, speed_label)
    return (int(speed_label.replace("speed", "")), speed_label)


def display_speed_label(speed_label: str) -> str:
    return "Hybrid" if speed_label == "hybrid" else speed_label.replace("speed", "Speed ")


def plot_force_time_overview(
    output_base: Path,
    grid: np.ndarray,
    aggregated: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    x_max: float,
) -> None:
    configure_plot_style()
    force_levels = sorted(aggregated.keys())
    fig_h = 0.9 + SUBPLOT_HEIGHT_IN * len(force_levels)
    fig, axes = plt.subplots(
        len(force_levels),
        1,
        figsize=(IROS_WIDTH_IN, fig_h),
        sharex=True,
        constrained_layout=True,
    )
    if len(force_levels) == 1:
        axes = [axes]

    global_max = 0.0
    for force_limit_g, speed_map in aggregated.items():
        global_max = max(global_max, force_limit_g * 1.1)
        for payload in speed_map.values():
            global_max = max(global_max, float(np.nanmax(payload["mean"] + payload["std"])) * 1.05)
    y_max = max(global_max, 100.0)
    y_min = -50.0

    legend_handles = []
    legend_labels = []

    for ax, force_limit_g in zip(axes, force_levels):
        speed_map = aggregated[force_limit_g]
        ax.set_title(f"Force Set = {force_limit_g} g", loc="left", pad=6, fontweight="bold")
        ax.axhline(force_limit_g, color="#B22222", linestyle=(0, (6, 4)), linewidth=1.4, alpha=0.95)
        ax.axvline(0.0, color="#9A8F7A", linestyle=(0, (2, 3)), linewidth=1.0, alpha=0.9)
        ax.grid(True, which="major", axis="both", color="#E6E2DB", linewidth=0.7)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(grid[0], x_max)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for speed_label in sorted(speed_map.keys(), key=nice_speed_order):
            payload = speed_map[speed_label]
            color = SPEED_COLORS.get(speed_label, "#444444")
            mean_trace = payload["mean"]
            valid = np.isfinite(mean_trace)
            if not np.any(valid):
                continue

            label = display_speed_label(speed_label)
            zorder = 6 if speed_label == "hybrid" else 3

            if speed_label == "hybrid":
                line, = ax.plot(
                    grid,
                    mean_trace,
                    color=color,
                    linewidth=2.9,
                    zorder=zorder,
                    label=label,
                )
                line.set_path_effects(
                    [
                        pe.Stroke(linewidth=5.8, foreground="white", alpha=0.98),
                        pe.Normal(),
                    ]
                )
                peak_idx = int(np.nanargmax(mean_trace))
                ax.scatter(
                    [grid[peak_idx]],
                    [mean_trace[peak_idx]],
                    s=42,
                    color=color,
                    edgecolors="white",
                    linewidths=1.3,
                    zorder=zorder + 1,
                )
            else:
                linestyle = "--" if speed_label in {"speed25", "speed50"} else "-"
                line, = ax.plot(
                    grid,
                    mean_trace,
                    color=color,
                    linewidth=1.8,
                    linestyle=linestyle,
                    alpha=0.97,
                    zorder=zorder,
                    label=label,
                )
                peak_idx = int(np.nanargmax(mean_trace))
                ax.scatter(
                    [grid[peak_idx]],
                    [mean_trace[peak_idx]],
                    s=18,
                    color=color,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=zorder + 1,
                )

            if speed_label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(speed_label)

        ax.text(
            x_max,
            force_limit_g,
            f" setpoint {force_limit_g} g",
            color="#B22222",
            fontsize=8,
            ha="right",
            va="bottom",
        )

    axes[-1].set_xlabel("Time from motion start (s)")
    for ax in axes:
        ax.set_ylabel("Force (g)")

    ordered = sorted(zip(legend_labels, legend_handles), key=lambda item: nice_speed_order(item[0]))
    fig.legend(
        [handle for _, handle in ordered],
        [display_speed_label(label) for label, _ in ordered],
        loc="upper center",
        ncol=min(4, len(ordered)),
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.2,
        handlelength=2.8,
    )
    fig.suptitle(
        "RH56 Time-Force Response by Force Set and Speed",
        y=1.03,
        fontsize=11,
        fontweight="bold",
    )

    save_figure(fig, output_base)


def plot_normalized_force_time_overview(
    output_base: Path,
    grid: np.ndarray,
    aggregated: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    x_max: float,
) -> None:
    configure_plot_style()
    force_levels = sorted(aggregated.keys())
    fig_h = 0.9 + SUBPLOT_HEIGHT_IN * len(force_levels)
    fig, axes = plt.subplots(
        len(force_levels),
        1,
        figsize=(IROS_WIDTH_IN, fig_h),
        sharex=True,
        constrained_layout=True,
    )
    if len(force_levels) == 1:
        axes = [axes]

    global_max = 1.1
    for force_limit_g, speed_map in aggregated.items():
        denom = max(float(force_limit_g), 1.0)
        for payload in speed_map.values():
            if np.any(np.isfinite(payload["mean"])):
                global_max = max(global_max, float(np.nanmax(payload["mean"] / denom)) * 1.05)
    y_max = max(global_max, 1.2)
    y_min = -0.05

    legend_handles = []
    legend_labels = []

    for ax, force_limit_g in zip(axes, force_levels):
        speed_map = aggregated[force_limit_g]
        denom = max(float(force_limit_g), 1.0)
        ax.set_title(f"Force Set = {force_limit_g} g", loc="left", pad=6, fontweight="bold")
        ax.axhline(1.0, color="#B22222", linestyle=(0, (6, 4)), linewidth=1.4, alpha=0.95)
        ax.axvline(0.0, color="#9A8F7A", linestyle=(0, (2, 3)), linewidth=1.0, alpha=0.9)
        ax.grid(True, which="major", axis="both", color="#E6E2DB", linewidth=0.7)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(grid[0], x_max)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for speed_label in sorted(speed_map.keys(), key=nice_speed_order):
            payload = speed_map[speed_label]
            color = SPEED_COLORS.get(speed_label, "#444444")
            mean_trace = payload["mean"] / denom
            valid = np.isfinite(mean_trace)
            if not np.any(valid):
                continue

            label = display_speed_label(speed_label)
            zorder = 6 if speed_label == "hybrid" else 3

            if speed_label == "hybrid":
                line, = ax.plot(
                    grid,
                    mean_trace,
                    color=color,
                    linewidth=2.9,
                    zorder=zorder,
                    label=label,
                )
                line.set_path_effects(
                    [
                        pe.Stroke(linewidth=5.8, foreground="white", alpha=0.98),
                        pe.Normal(),
                    ]
                )
                peak_idx = int(np.nanargmax(mean_trace))
                ax.scatter(
                    [grid[peak_idx]],
                    [mean_trace[peak_idx]],
                    s=42,
                    color=color,
                    edgecolors="white",
                    linewidths=1.3,
                    zorder=zorder + 1,
                )
            else:
                linestyle = "--" if speed_label in {"speed25", "speed50"} else "-"
                line, = ax.plot(
                    grid,
                    mean_trace,
                    color=color,
                    linewidth=1.8,
                    linestyle=linestyle,
                    alpha=0.97,
                    zorder=zorder,
                    label=label,
                )
                peak_idx = int(np.nanargmax(mean_trace))
                ax.scatter(
                    [grid[peak_idx]],
                    [mean_trace[peak_idx]],
                    s=18,
                    color=color,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=zorder + 1,
                )

            if speed_label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(speed_label)

        ax.text(
            x_max,
            1.0,
            " setpoint = 1.0",
            color="#B22222",
            fontsize=8,
            ha="right",
            va="bottom",
        )

    axes[-1].set_xlabel("Time from motion start (s)")
    for ax in axes:
        ax.set_ylabel("Normalized force")

    ordered = sorted(zip(legend_labels, legend_handles), key=lambda item: nice_speed_order(item[0]))
    fig.legend(
        [handle for _, handle in ordered],
        [display_speed_label(label) for label, _ in ordered],
        loc="upper center",
        ncol=min(4, len(ordered)),
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.2,
        handlelength=2.8,
    )
    fig.suptitle(
        "RH56 Normalized Time-Force Response by Force Set and Speed",
        y=1.03,
        fontsize=11,
        fontweight="bold",
    )

    save_figure(fig, output_base)


def aggregate_combined_normalized_by_speed(
    aggregated: Dict[int, Dict[str, Dict[str, np.ndarray]]],
) -> Dict[str, np.ndarray]:
    by_speed: Dict[str, List[np.ndarray]] = defaultdict(list)
    for force_limit_g, speed_map in aggregated.items():
        denom = max(float(force_limit_g), 1.0)
        for speed_label, payload in speed_map.items():
            by_speed[speed_label].append(payload["mean"] / denom)

    combined: Dict[str, np.ndarray] = {}
    for speed_label, traces in by_speed.items():
        combined[speed_label] = np.nanmean(np.vstack(traces), axis=0)

    return combined


def build_representative_normalized_by_speed(
    aggregated: Dict[int, Dict[str, Dict[str, np.ndarray]]],
) -> Dict[str, Dict[str, Dict[str, object]]]:
    per_speed: Dict[str, List[Tuple[int, float, np.ndarray]]] = defaultdict(list)
    for force_limit_g, speed_map in aggregated.items():
        denom = max(float(force_limit_g), 1.0)
        for speed_label, payload in speed_map.items():
            trace = payload["mean"] / denom
            if not np.any(np.isfinite(trace)):
                continue
            peak_value = float(np.nanmax(trace))
            per_speed[speed_label].append((force_limit_g, peak_value, trace))

    representatives: Dict[str, Dict[str, Dict[str, object]]] = {
        "best": {},
        "median": {},
        "worst": {},
    }

    for speed_label, items in per_speed.items():
        ranked = sorted(items, key=lambda item: item[1])
        best_item = ranked[0]
        median_item = ranked[len(ranked) // 2]
        worst_item = ranked[-1]
        representatives["best"][speed_label] = {
            "force_limit_g": best_item[0],
            "peak_value": best_item[1],
            "trace": best_item[2],
        }
        representatives["median"][speed_label] = {
            "force_limit_g": median_item[0],
            "peak_value": median_item[1],
            "trace": median_item[2],
        }
        representatives["worst"][speed_label] = {
            "force_limit_g": worst_item[0],
            "peak_value": worst_item[1],
            "trace": worst_item[2],
        }

    return representatives


def plot_combined_normalized_by_speed(
    output_base: Path,
    grid: np.ndarray,
    aggregated: Dict[str, np.ndarray],
    x_max: float,
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(IROS_WIDTH_IN, 3.5), constrained_layout=True)
    ax.grid(True, which="major", axis="both", color="#E6E2DB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(1.0, color="#B22222", linestyle=(0, (6, 4)), linewidth=1.3, alpha=0.95)
    ax.set_xlim(0.0, x_max)

    y_max = 1.15
    for trace in aggregated.values():
        if np.any(np.isfinite(trace)):
            y_max = max(y_max, float(np.nanmax(trace)) * 1.05)
    ax.set_ylim(-0.05, y_max)

    handles = []
    labels = []
    for speed_label in sorted(aggregated.keys(), key=nice_speed_order):
        color = SPEED_COLORS.get(speed_label, "#444444")
        trace = aggregated[speed_label]
        zorder = 6 if speed_label == "hybrid" else 3
        label = display_speed_label(speed_label)

        if speed_label == "hybrid":
            line, = ax.plot(
                grid,
                trace,
                color=color,
                linewidth=2.9,
                zorder=zorder,
                label=label,
            )
            line.set_path_effects(
                [
                    pe.Stroke(linewidth=5.8, foreground="white", alpha=0.98),
                    pe.Normal(),
                ]
            )
            peak_idx = int(np.nanargmax(trace))
            ax.scatter(
                [grid[peak_idx]],
                [trace[peak_idx]],
                s=42,
                color=color,
                edgecolors="white",
                linewidths=1.3,
                zorder=zorder + 1,
            )
        else:
            linestyle = "--" if speed_label in {"speed25", "speed50"} else "-"
            line, = ax.plot(
                grid,
                trace,
                color=color,
                linewidth=1.9,
                linestyle=linestyle,
                alpha=0.98,
                zorder=zorder,
                label=label,
            )
            peak_idx = int(np.nanargmax(trace))
            ax.scatter(
                [grid[peak_idx]],
                [trace[peak_idx]],
                s=18,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=zorder + 1,
            )

        handles.append(line)
        labels.append(label)

    ax.set_xlabel("Time from motion start (s)")
    ax.set_ylabel("Normalized force (force / setpoint)")
    ax.set_title(
        "Normalized Response Averaged Across Force Sets",
        loc="left",
        pad=6,
        fontweight="bold",
    )
    ax.legend(handles, labels, ncol=min(4, len(labels)), loc="upper left")
    ax.text(
        x_max,
        1.0,
        " setpoint = 1.0",
        color="#B22222",
        fontsize=8,
        ha="right",
        va="bottom",
    )

    save_figure(fig, output_base)


def plot_representative_normalized_by_speed(
    output_base: Path,
    grid: np.ndarray,
    representatives: Dict[str, Dict[str, Dict[str, object]]],
    x_max: float,
) -> None:
    configure_plot_style()
    panel_order = ["best", "median", "worst"]
    panel_titles = {
        "best": "Best Case per Speed (lowest normalized peak across force sets)",
        "median": "Median Case per Speed",
        "worst": "Worst Case per Speed (highest normalized peak across force sets)",
    }
    fig, axes = plt.subplots(
        len(panel_order),
        1,
        figsize=(IROS_WIDTH_IN, 1.8 + 2.25 * len(panel_order)),
        sharex=True,
        constrained_layout=True,
    )
    if len(panel_order) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    for ax, panel_name in zip(axes, panel_order):
        panel = representatives[panel_name]
        y_max = 1.15
        for payload in panel.values():
            trace = payload["trace"]
            if np.any(np.isfinite(trace)):
                y_max = max(y_max, float(np.nanmax(trace)) * 1.05)

        ax.grid(True, which="major", axis="both", color="#E6E2DB", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(1.0, color="#B22222", linestyle=(0, (6, 4)), linewidth=1.3, alpha=0.95)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(-0.05, y_max)
        ax.set_title(panel_titles[panel_name], loc="left", pad=6, fontweight="bold")

        for speed_label in sorted(panel.keys(), key=nice_speed_order):
            payload = panel[speed_label]
            trace = payload["trace"]
            color = SPEED_COLORS.get(speed_label, "#444444")
            label = display_speed_label(speed_label)
            zorder = 6 if speed_label == "hybrid" else 3

            if speed_label == "hybrid":
                line, = ax.plot(
                    grid,
                    trace,
                    color=color,
                    linewidth=2.9,
                    zorder=zorder,
                    label=label,
                )
                line.set_path_effects(
                    [
                        pe.Stroke(linewidth=5.8, foreground="white", alpha=0.98),
                        pe.Normal(),
                    ]
                )
                peak_idx = int(np.nanargmax(trace))
                ax.scatter(
                    [grid[peak_idx]],
                    [trace[peak_idx]],
                    s=42,
                    color=color,
                    edgecolors="white",
                    linewidths=1.3,
                    zorder=zorder + 1,
                )
            else:
                linestyle = "--" if speed_label in {"speed25", "speed50"} else "-"
                line, = ax.plot(
                    grid,
                    trace,
                    color=color,
                    linewidth=1.9,
                    linestyle=linestyle,
                    alpha=0.98,
                    zorder=zorder,
                    label=label,
                )
                peak_idx = int(np.nanargmax(trace))
                ax.scatter(
                    [grid[peak_idx]],
                    [trace[peak_idx]],
                    s=18,
                    color=color,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=zorder + 1,
                )

            if speed_label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(speed_label)

            ax.text(
                x_max,
                trace[int(np.nanargmax(trace))],
                f" F={int(payload['force_limit_g'])}",
                color=color,
                fontsize=7,
                ha="right",
                va="bottom",
            )

        ax.set_ylabel("Norm. force")

    axes[-1].set_xlabel("Time from motion start (s)")
    ordered = sorted(zip(legend_labels, legend_handles), key=lambda item: nice_speed_order(item[0]))
    fig.legend(
        [handle for _, handle in ordered],
        [display_speed_label(label) for label, _ in ordered],
        loc="upper center",
        ncol=min(4, len(ordered)),
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.2,
        handlelength=2.8,
    )
    fig.suptitle(
        "Representative Normalized Responses Across Force Sets",
        y=1.02,
        fontsize=11,
        fontweight="bold",
    )

    save_figure(fig, output_base)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    png_path = output_base.with_suffix(".png")
    pgf_path = output_base.with_suffix(".pgf")
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(pgf_path, bbox_inches="tight", facecolor="white")
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(
            "PNG export succeeded, but PGF export failed. "
            "PGF output usually requires a working LaTeX installation "
            "(for example pdflatex) on this machine."
        ) from exc
    plt.close(fig)


def ordered_speed_labels(metrics: Sequence[TrialMetric]) -> List[str]:
    labels = sorted({item.speed_label for item in metrics}, key=overshoot_speed_order)
    return labels


def plot_overshoot_vs_speed(
    output_base: Path,
    metrics: Sequence[TrialMetric],
    excluded_keys: Optional[set[Tuple[str, int]]] = None,
) -> None:
    configure_plot_style()
    excluded = excluded_keys or set()
    plot_metrics = [
        item for item in metrics if (item.source_file, item.trial_index) not in excluded
    ]
    speed_labels = ordered_speed_labels(plot_metrics)
    x_positions = np.arange(len(speed_labels), dtype=float)

    by_force_speed: Dict[Tuple[int, str], List[TrialMetric]] = defaultdict(list)
    force_levels = sorted({item.force_limit_g for item in plot_metrics})
    for item in plot_metrics:
        by_force_speed[(item.force_limit_g, item.speed_label)].append(item)

    fig, ax = plt.subplots(figsize=(IROS_WIDTH_IN, 3.2), constrained_layout=True)
    ax.grid(True, axis="y", color="#E6E2DB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for force_limit_g in force_levels:
        color = FORCE_COLORS.get(force_limit_g, "#444444")
        xs: List[float] = []
        ys: List[float] = []
        yerrs: List[float] = []
        for pos, speed_label in zip(x_positions, speed_labels):
            group = by_force_speed.get((force_limit_g, speed_label), [])
            if not group:
                continue
            overshoots = [item.overshoot_g for item in group]
            xs.append(float(pos))
            ys.append(float(mean(overshoots)))
            yerrs.append(float(pstdev(overshoots)) if len(overshoots) > 1 else 0.0)

        if not xs:
            continue

        line, caplines, barlinecols = ax.errorbar(
            xs,
            ys,
            yerr=yerrs,
            color=color,
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=3.0,
            label=f"Force set {force_limit_g}",
            zorder=3,
        )
        for barcol in barlinecols:
            barcol.set_alpha(0.85)
        for cap in caplines:
            cap.set_alpha(0.85)

        if "hybrid" in speed_labels:
            hybrid_pos = float(speed_labels.index("hybrid"))
            group = by_force_speed.get((force_limit_g, "hybrid"), [])
            if group:
                hybrid_mean = float(mean(item.overshoot_g for item in group))
                hybrid_std = float(
                    pstdev([item.overshoot_g for item in group]) if len(group) > 1 else 0.0
                )
                ax.scatter(
                    [hybrid_pos],
                    [hybrid_mean],
                    s=64,
                    marker="D",
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=5,
                )
                ax.vlines(
                    hybrid_pos,
                    hybrid_mean - hybrid_std,
                    hybrid_mean + hybrid_std,
                    colors="black",
                    linewidth=1.0,
                    zorder=4,
                )

    ax.axhline(0.0, color="#7D766C", linewidth=1.0, linestyle=(0, (3, 3)))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([display_speed_label(label) for label in speed_labels])
    ax.set_ylabel("Overshoot (g)")
    ax.set_xlabel("Command condition")
    ax.set_title("Overshoot vs Speed / Hybrid", loc="left", pad=6, fontweight="bold")
    ax.legend(ncol=min(3, len(force_levels)), loc="upper left")

    save_figure(fig, output_base)


def print_console_summary(metrics: Sequence[TrialMetric]) -> None:
    by_speed: Dict[str, List[TrialMetric]] = defaultdict(list)
    by_file: Dict[str, List[TrialMetric]] = defaultdict(list)
    for item in metrics:
        by_speed[item.speed_label].append(item)
        by_file[item.source_file].append(item)

    print("Detected contact-position variability by speed:")
    for speed_label, group in sorted(by_speed.items(), key=lambda item: nice_speed_order(item[0])):
        s = summarize_group(group)
        print(
            f"  {speed_label:>9} | n={int(s['trial_count']):>3} | "
            f"onset angle mean={s['onset_angle_mean']:.2f} | "
            f"var={s['onset_angle_variance']:.2f} | "
            f"overshoot mean={s['overshoot_mean_g']:.2f} g"
        )

    overall = summarize_group(metrics)
    pooled = pooled_variance_from_groups([group for group in by_file.values()])
    print(
        "Overall | "
        f"n={int(overall['trial_count'])} | "
        f"onset angle mean={overall['onset_angle_mean']:.2f} | "
        f"raw var={overall['onset_angle_variance']:.2f}"
    )
    print(
        "Decomposed overall | "
        f"within-csv var mean={pooled['within_group_var_mean']:.2f} | "
        f"between-csv var of means={pooled['between_group_var_of_means']:.2f} | "
        f"pooled total var={pooled['pooled_total_var']:.2f}"
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = discover_csvs(input_dir)
    if not csv_files:
        raise SystemExit(f"No experiment CSVs found in {input_dir}")

    all_metrics: List[TrialMetric] = []
    contact_trials_by_file: Dict[str, Dict[int, List[Sample]]] = {}
    motion_trials_by_file: Dict[str, Dict[int, List[Sample]]] = {}

    for csv_path in csv_files:
        speed_label, speed_value, force_limit_g = parse_filename(csv_path)
        contact_trials = load_contact_trials(csv_path)
        motion_trials = load_motion_trials(csv_path)
        contact_trials_by_file[csv_path.name] = contact_trials
        motion_trials_by_file[csv_path.name] = motion_trials
        for samples in contact_trials.values():
            metric = build_trial_metric(
                csv_path,
                speed_label,
                speed_value,
                force_limit_g,
                samples,
                args,
            )
            all_metrics.append(metric)

    all_metrics.sort(
        key=lambda item: (
            item.force_limit_g,
            nice_speed_order(item.speed_label),
            item.source_file,
            item.trial_index,
        )
    )

    trial_metrics_path = output_dir / "contact_position_trial_metrics.csv"
    summary_path = output_dir / "contact_position_group_summary.csv"
    outlier_path = output_dir / "plot_outliers.csv"
    plot_base = output_dir / "force_time_overview"
    normalized_plot_base = output_dir / "normalized_force_time_overview"
    combined_normalized_plot_base = output_dir / "normalized_force_time_combined"
    overshoot_plot_base = output_dir / "overshoot_vs_speed"

    write_trial_metrics(trial_metrics_path, all_metrics)
    write_group_summary(summary_path, all_metrics)
    excluded_keys, outlier_rows = detect_plot_outliers(
        all_metrics,
        motion_trials_by_file,
        args.outlier_z_threshold,
        args.hybrid_outlier_z_threshold,
        args.outlier_min_group,
    )
    write_plot_outliers(outlier_path, outlier_rows)
    grid, aggregated, max_peak_rel_s = aggregate_plot_data(
        motion_trials_by_file,
        all_metrics,
        0.0,
        args.plot_post_s,
        args.plot_dt_s,
        excluded_keys,
    )
    x_max = max_peak_rel_s + args.plot_dt_s
    plot_force_time_overview(plot_base, grid, aggregated, x_max)
    plot_normalized_force_time_overview(normalized_plot_base, grid, aggregated, x_max)
    combined_grid = grid
    combined_aggregated = aggregate_combined_normalized_by_speed(aggregated)
    plot_combined_normalized_by_speed(
        combined_normalized_plot_base,
        combined_grid,
        combined_aggregated,
        x_max,
    )
    plot_overshoot_vs_speed(overshoot_plot_base, all_metrics, excluded_keys)

    print_console_summary(all_metrics)
    print(f"Wrote: {trial_metrics_path}")
    print(f"Wrote: {summary_path}")
    print(
        f"Wrote: {outlier_path} ({len(outlier_rows)} plot outliers excluded; "
        f"default z={args.outlier_z_threshold}, hybrid z={args.hybrid_outlier_z_threshold})"
    )
    print(f"Wrote: {plot_base.with_suffix('.png')}")
    print(f"Wrote: {plot_base.with_suffix('.pgf')}")
    print(f"Wrote: {normalized_plot_base.with_suffix('.png')}")
    print(f"Wrote: {normalized_plot_base.with_suffix('.pgf')}")
    print(f"Wrote: {combined_normalized_plot_base.with_suffix('.png')}")
    print(f"Wrote: {combined_normalized_plot_base.with_suffix('.pgf')}")
    print(f"Wrote: {overshoot_plot_base.with_suffix('.png')}")
    print(f"Wrote: {overshoot_plot_base.with_suffix('.pgf')}")
    print(f"Wrote: {overshoot_plot_base.with_suffix('.png')}")
    print(f"Wrote: {overshoot_plot_base.with_suffix('.pgf')}")


if __name__ == "__main__":
    main()
