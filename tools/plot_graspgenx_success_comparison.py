#!/usr/bin/env python3
"""Plot RH56 analytical-versus-GraspGen-X lift benchmark results."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "analytical": "Analytical synchronized closure",
    "graspgenx": "Pretrained GraspGen-X",
}
METHOD_COLORS = {
    "analytical": "#1976d2",
    "graspgenx": "#ef6c00",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot success-rate and directional-lift figures from the RH56 comparison."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/graspgenx_success_comparison"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_success_rate(summary_rows: list[dict[str, str]], path: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for method in METHOD_LABELS:
        rows = sorted(
            (row for row in summary_rows if row["method"] == method),
            key=lambda row: float(row["error_mm"]),
        )
        errors = np.array([float(row["error_mm"]) for row in rows])
        rates = 100.0 * np.array([float(row["success_rate"]) for row in rows])
        axis.plot(
            errors,
            rates,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for error, rate, row in zip(errors, rates, rows):
            axis.annotate(
                f"{row['successes']}/{row['trials']}",
                (error, rate),
                xytext=(0, 7 if method == "analytical" else -13),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=METHOD_COLORS[method],
            )
    axis.set(xlabel="Horizontal object-position error (mm)", ylabel="Lift success rate (%)")
    axis.set_ylim(-4, 106)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower left", frameon=True)
    axis.set_title("RH56 cube-lift robustness (40 mm cube, 8 directions/error)")
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _direction_matrix(
    trial_rows: list[dict[str, str]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row for row in trial_rows
        if row["method"] == method and float(row["error_mm"]) > 0.0
    ]
    errors = np.array(sorted({float(row["error_mm"]) for row in selected}))
    directions = np.array(sorted({float(row["direction_deg"]) for row in selected}))
    lifts = np.full((len(errors), len(directions)), np.nan)
    success = np.zeros_like(lifts, dtype=bool)
    for row in selected:
        i = int(np.flatnonzero(np.isclose(errors, float(row["error_mm"])))[0])
        j = int(np.flatnonzero(np.isclose(directions, float(row["direction_deg"])))[0])
        lifts[i, j] = float(row["final_lift_mm"])
        success[i, j] = row["success"].lower() == "true"
    return errors, directions, lifts, success


def plot_directional_lift(trial_rows: list[dict[str, str]], path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), constrained_layout=True, sharey=True)
    image = None
    for axis, method in zip(axes, METHOD_LABELS):
        errors, directions, lifts, success = _direction_matrix(trial_rows, method)
        image = axis.imshow(
            np.clip(lifts, 0.0, 70.0),
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=70.0,
        )
        for row_index in range(len(errors)):
            for column_index in range(len(directions)):
                lift = lifts[row_index, column_index]
                label = f"{lift:.0f}" if success[row_index, column_index] else "×"
                color = "white" if not success[row_index, column_index] or lift < 35 else "black"
                axis.text(column_index, row_index, label, ha="center", va="center", color=color, fontsize=8)
        axis.set_xticks(np.arange(len(directions)), [f"{value:.0f}°" for value in directions], rotation=45)
        axis.set_yticks(np.arange(len(errors)), [f"{value:.0f}" for value in errors])
        axis.set_xlabel("Error direction")
        axis.set_title(METHOD_LABELS[method])
    axes[0].set_ylabel("Error magnitude (mm)")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("Final object lift (mm); × = below 40 mm success threshold")
    figure.suptitle("Directional robustness of the two RH56 grasp pipelines")
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def plot_object_displacement(
    summary_rows: list[dict[str, str]], path: Path, dpi: int
) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for method in METHOD_LABELS:
        rows = sorted(
            (row for row in summary_rows if row["method"] == method),
            key=lambda row: float(row["error_mm"]),
        )
        errors = np.array([float(row["error_mm"]) for row in rows])
        displacement = np.array(
            [float(row["mean_max_xy_displacement_mm"]) for row in rows]
        )
        axis.plot(
            errors,
            displacement,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set(
        xlabel="Horizontal object-position error (mm)",
        ylabel="Mean maximum object XY displacement (mm)",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left", frameon=True)
    axis.set_title("Object disturbance during RH56 approach, closure, lift, and hold")
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    summary_path = args.input_dir / "summary.csv"
    trials_path = args.input_dir / "trials.csv"
    summary_rows = read_rows(summary_path)
    trial_rows = read_rows(trials_path)
    success_path = args.input_dir / "success_rate.png"
    directional_path = args.input_dir / "directional_lift.png"
    displacement_path = args.input_dir / "object_displacement.png"
    plot_success_rate(summary_rows, success_path, args.dpi)
    plot_directional_lift(trial_rows, directional_path, args.dpi)
    plot_object_displacement(summary_rows, displacement_path, args.dpi)
    print(f"Wrote {success_path}")
    print(f"Wrote {directional_path}")
    print(f"Wrote {displacement_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
