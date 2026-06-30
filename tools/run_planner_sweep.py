#!/usr/bin/env python3
"""Run a paper-facing RH56 analytical planner sweep.

This script is simulation-only: it uses the existing MuJoCo/FK-backed
ClosureGeometry API and writes CSV/metadata/plots under an artifact directory.
It does not require a physical RH56 hand, ROS2, Unitree SDKs, or a real robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import numpy as np

from rh56_controller.grasp_geometry import (
    ClosureGeometry,
    ClosureResult,
    GRASP_FINGER_SETS,
    InspireHandFK,
    NON_THUMB_FINGERS,
)


MODE_LABELS = {
    "line": "2-finger line",
    "plane3": "3-finger plane",
    "plane4": "4-finger plane",
    "plane5": "5-finger plane",
    "cylinder": "cylinder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the RH56 analytical width-to-grasp planner without hardware."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["line", "plane3", "plane4", "plane5"],
        choices=["line", "plane", "plane3", "plane4", "plane5", "cylinder"],
        help=(
            "Closure modes to evaluate. 'plane' is an alias for plane4. "
            "Cylinder is available but excluded from the paper-facing default "
            "because it needs object-aware palm-centering path planning."
        ),
    )
    parser.add_argument("--width-min-mm", type=float, default=5.0)
    parser.add_argument("--width-max-mm", type=float, default=115.0)
    parser.add_argument("--width-step-mm", type=float, default=1.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/planner_sweep"),
        help="Output directory for summary.csv, assumptions.json, and plots.",
    )
    parser.add_argument(
        "--object-width-offset-mm",
        type=float,
        default=20.0,
        help=(
            "Subtract this from internal site-to-site widths for precision modes "
            "to estimate object-facing fingertip width. Not applied to cylinder."
        ),
    )
    parser.add_argument("--xml", default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--rebuild-fk", action="store_true", help="Rebuild FK cache.")
    parser.add_argument(
        "--width-match-tolerance-mm",
        "--success-tolerance-mm",
        dest="width_match_tolerance_mm",
        type=float,
        default=1.0,
        help=(
            "Requested-vs-achieved width tolerance for the width-match diagnostic. "
            "This is not a grasp-success metric."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV/JSON only.",
    )
    return parser.parse_args()


def width_values(min_mm: float, max_mm: float, step_mm: float) -> list[float]:
    if step_mm <= 0:
        raise ValueError("--width-step-mm must be positive")
    values: list[float] = []
    current = min_mm
    while current <= max_mm + 1e-9:
        values.append(round(current, 6))
        current += step_mm
    return values


def active_actuators(mode: str) -> list[str]:
    if mode == "line":
        return ["index", "thumb_proximal", "thumb_yaw"]
    if mode.startswith("plane"):
        n = int(mode[-1])
        return GRASP_FINGER_SETS[n] + ["thumb_proximal", "thumb_yaw"]
    return NON_THUMB_FINGERS + ["thumb_proximal", "thumb_yaw"]


def joint_margin(result: ClosureResult, fk: InspireHandFK, mode: str) -> float:
    margins: list[float] = []
    for key in active_actuators(mode):
        value = float(result.ctrl_values[key])
        ctrl_key = key if key != "thumb_yaw" else "thumb_yaw"
        lo = float(fk.ctrl_min[ctrl_key])
        hi = float(fk.ctrl_max[ctrl_key])
        span = hi - lo
        if span <= 0:
            continue
        margins.append(min((value - lo) / span, (hi - value) / span))
    if not margins:
        return 0.0
    return float(max(0.0, min(margins)))


def z_metrics(result: ClosureResult) -> tuple[float, float]:
    wtips = result.world_tips(world_grasp_z=0.0)
    zvals = np.array([pos[2] for pos in wtips.values()], dtype=float)
    z_span_mm = float((zvals.max() - zvals.min()) * 1000.0) if len(zvals) else 0.0
    coplanarity_mm = float(np.std(zvals) * 1000.0) if len(zvals) else 0.0
    return coplanarity_mm, z_span_mm


def solve_mode(closure: ClosureGeometry, mode: str, width_m: float) -> ClosureResult:
    if mode == "line":
        return closure.line(width_m)
    if mode.startswith("plane"):
        return closure.plane(width_m, n_fingers=int(mode[-1]))
    if mode == "cylinder":
        return closure.cylinder(width_m)
    raise ValueError(f"Unsupported mode: {mode}")


def mode_range(closure: ClosureGeometry, mode: str) -> tuple[float, float]:
    if mode == "line":
        return closure.width_range("2-finger line", n_fingers=2)
    if mode.startswith("plane"):
        n = int(mode[-1])
        return closure.width_range(f"{n}-finger plane", n_fingers=n)
    return closure.width_range("cylinder", n_fingers=5)


def classify_result(
    width_m: float,
    achieved_m: float,
    lo: float,
    hi: float,
    tol_m: float,
) -> tuple[bool, bool, str]:
    """Separate nominal geometric feasibility from width-tracking quality."""
    nominal_feasible = lo - tol_m <= width_m <= hi + tol_m
    width_match = nominal_feasible and abs(achieved_m - width_m) <= tol_m
    if not nominal_feasible:
        if width_m < lo - tol_m:
            return nominal_feasible, width_match, "below_nominal_range"
        if width_m > hi + tol_m:
            return nominal_feasible, width_match, "above_nominal_range"
        return nominal_feasible, width_match, "outside_nominal_range"
    if not width_match:
        return nominal_feasible, width_match, "width_tracking_error"
    return nominal_feasible, width_match, "ok"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "mode",
        "width_mm",
        "object_width_mm",
        "internal_width_mm",
        "width_basis",
        "solve_success",
        "nominal_feasible",
        "width_match",
        "solve_time_ms",
        "width_error_mm",
        "coplanarity_error_mm",
        "z_span_mm",
        "tilt_deg",
        "joint_margin",
        "reason",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_assumptions(path: Path, args: argparse.Namespace, ranges: dict[str, tuple[float, float]]) -> None:
    payload = {
        "script": "tools/run_planner_sweep.py",
        "simulation_only": True,
        "uses_hardware": False,
        "modes": args.modes,
        "width_min_mm": args.width_min_mm,
        "width_max_mm": args.width_max_mm,
        "width_step_mm": args.width_step_mm,
        "object_width_offset_mm": args.object_width_offset_mm,
        "width_match_tolerance_mm": args.width_match_tolerance_mm,
        "reachable_ranges_mm": {
            mode: [lo * 1000.0, hi * 1000.0] for mode, (lo, hi) in ranges.items()
        },
        "object_corrected_ranges_mm": {
            mode: [
                lo * 1000.0 - (args.object_width_offset_mm if mode != "cylinder" else 0.0),
                hi * 1000.0 - (args.object_width_offset_mm if mode != "cylinder" else 0.0),
            ]
            for mode, (lo, hi) in ranges.items()
        },
        "nominal_feasible_definition": (
            "requested internal scalar width lies inside the mode-specific nominal solver range "
            "and the solver returns a finite configuration"
        ),
        "object_width_definition": (
            "for line/plane modes, object_width_mm = internal_width_mm - "
            "object_width_offset_mm because MuJoCo sites are embedded within the fingertips; "
            "for cylinder, object_width_mm is the requested cylinder diameter"
        ),
        "width_match_definition": (
            "absolute achieved internal scalar width error is within width_match_tolerance_mm; "
            "this is a planner diagnostic, not a grasp-success metric"
        ),
        "grasp_success_definition": "not evaluated by this simulation-only planner sweep",
        "cylinder_note": (
            "cylinder remains available as an explicit mode, but it is not included "
            "in the paper-facing default because rollout quality depends on "
            "object-aware palm-centering path planning"
        ),
        "collision_model": "not evaluated in this sweep",
        "joint_margin": "minimum normalized distance to active actuator limits",
        "runtime_note": (
            "solve_time_ms measures offline analytical solve/table-generation time; "
            "online execution can use a precomputed lookup table indexed by mode and width"
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def maybe_write_plots(out_dir: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_skipped.txt").write_text(
            f"matplotlib unavailable; plots skipped: {exc}\n"
        )
        return

    def _as_bool(value: object) -> bool:
        return str(value).strip().lower() == "true"

    def _width(row: dict[str, object]) -> float:
        return float(row["width_mm"])

    width_axis_label = (
        "Object width [mm] (internal site width corrected where applicable)"
    )

    modes = list(dict.fromkeys(str(row["mode"]) for row in rows))
    rows_by_mode = {mode: [row for row in rows if str(row["mode"]) == mode] for mode in modes}

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.62 * len(modes) + 1.2)))
    y_positions = list(range(len(modes)))
    for y, mode in zip(y_positions, modes):
        mode_rows = sorted(rows_by_mode[mode], key=_width)
        widths = [_width(row) for row in mode_rows]
        feasible = [_as_bool(row["nominal_feasible"]) for row in mode_rows]
        if len(widths) > 1:
            diffs = np.diff(widths)
            step = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
        else:
            step = 1.0

        ax.hlines(y, min(widths), max(widths), color="#d8d8d8", lw=8, label=None)
        segment_start: float | None = None
        last_success: float | None = None
        segments: list[tuple[float, float]] = []
        for width, ok in zip(widths, feasible):
            if ok and segment_start is None:
                segment_start = width
            if ok:
                last_success = width
            if not ok and segment_start is not None and last_success is not None:
                segments.append((segment_start, last_success))
                segment_start = None
                last_success = None
        if segment_start is not None and last_success is not None:
            segments.append((segment_start, last_success))

        for start, end in segments:
            ax.hlines(y, start, end, color="#217a3a", lw=8)
            ax.plot([start, end], [y, y], "o", color="#217a3a", ms=4)

        ok_count = sum(feasible)
        if segments:
            labels = [
                f"{start:.0f}" if abs(start - end) < 0.5 * step else f"{start:.0f}-{end:.0f}"
                for start, end in segments
            ]
            label = f"{ok_count}/{len(mode_rows)} feasible; " + ", ".join(labels) + " mm"
        else:
            label = f"{ok_count}/{len(mode_rows)} feasible"
        ax.text(max(widths) + 2.0, y, label, va="center", fontsize=8)

    ax.set_yticks(y_positions, modes)
    ax.set_xlabel(width_axis_label)
    ax.set_title("Nominal feasible object-width range by grasp mode")
    ax.set_xlim(left=min(_width(row) for row in rows) - 2, right=max(_width(row) for row in rows) + 28)
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(
        0.01,
        -0.18,
        "Gray = swept corrected range; green = internal request inside nominal solver range. This is feasibility, not grasp success.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "planner_feasible_ranges.png", dpi=150)
    fig.savefig(out_dir / "reachable_widths.png", dpi=150)
    fig.savefig(out_dir / "planner_success_intervals.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(modes), 1, figsize=(9, max(5.5, 1.35 * len(modes))), sharex=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        mode_rows = sorted(rows_by_mode[mode], key=_width)
        widths = np.array([_width(row) for row in mode_rows])
        errors = np.array([
            float(row["width_error_mm"]) if row["width_error_mm"] != "" else np.nan
            for row in mode_rows
        ])
        width_matches = np.array([_as_bool(row["width_match"]) for row in mode_rows])
        ax.plot(widths, errors, color="#555555", lw=1.2)
        ax.scatter(
            widths[width_matches],
            errors[width_matches],
            color="#217a3a",
            s=14,
            label="within width tolerance",
        )
        ax.scatter(
            widths[~width_matches],
            errors[~width_matches],
            color="#b6403c",
            s=10,
            alpha=0.55,
            label="outside width tolerance",
        )
        ax.axhline(1.0, color="#1f77b4", lw=1, ls="--", label="1 mm tolerance")
        ax.set_ylabel("Width error [mm]")
        ax.set_title(mode, loc="left", fontsize=10)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(width_axis_label)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", ncols=3, fontsize=8)
    fig.suptitle("Internal width tracking diagnostic after multi-finger corrections", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "width_error_by_mode.png", dpi=150)
    plt.close(fig)

    solve_times_by_mode: list[tuple[str, list[float]]] = []
    for mode in modes:
        values = [
            float(row["solve_time_ms"])
            for row in rows_by_mode[mode]
            if row["solve_time_ms"] != ""
        ]
        if values:
            solve_times_by_mode.append((mode, values))
    if solve_times_by_mode:
        fig, ax = plt.subplots(figsize=(8.6, 4.5))
        labels = [mode for mode, _ in solve_times_by_mode]
        data = [values for _, values in solve_times_by_mode]
        positions = np.arange(1, len(data) + 1)
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1b1b1b", "linewidth": 1.5},
            boxprops={"facecolor": "#dbe7f5", "edgecolor": "#5277a3", "linewidth": 1.2},
            whiskerprops={"color": "#5277a3", "linewidth": 1.1},
            capprops={"color": "#5277a3", "linewidth": 1.1},
        )
        for patch in bp["boxes"]:
            patch.set_alpha(0.9)
        for pos, values in zip(positions, data):
            arr = np.array(values, dtype=float)
            jitter = (((np.arange(len(arr)) % 11) - 5) / 5.0) * 0.075
            ax.scatter(
                np.full(len(arr), pos) + jitter,
                arr,
                s=12,
                color="#3d5f87",
                alpha=0.38,
                linewidths=0,
            )
            median = float(np.median(arr))
            label_y = float(np.percentile(arr, 75) + 0.28)
            ax.text(
                pos,
                label_y,
                f"median {median:.2f} ms",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(positions, [f"{label}\n(n={len(vals)})" for label, vals in solve_times_by_mode])
        ax.set_ylabel("Offline solve time [ms]")
        ax.set_title(
            f"Offline analytical table-generation time by grasp mode (total n={sum(len(vals) for vals in data)})"
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.margins(y=0.12)
        fig.tight_layout()
        fig.savefig(out_dir / "solve_time_by_mode.png", dpi=150)
        fig.savefig(out_dir / "solve_time_hist.png", dpi=150)
        plt.close(fig)


def normalize_modes(modes: Iterable[str]) -> list[str]:
    normalized = ["plane4" if mode == "plane" else mode for mode in modes]
    return list(dict.fromkeys(normalized))


def main() -> int:
    args = parse_args()
    args.modes = normalize_modes(args.modes)
    args.out.mkdir(parents=True, exist_ok=True)

    print("RH56 planner sweep assumptions:")
    print("  simulation_only: true")
    print("  hardware_required: false")
    print(f"  output: {args.out}")

    fk = InspireHandFK(xml_path=args.xml, rebuild=args.rebuild_fk) if args.xml else InspireHandFK(rebuild=args.rebuild_fk)
    closure = ClosureGeometry(fk)
    widths_mm = width_values(args.width_min_mm, args.width_max_mm, args.width_step_mm)
    tol_m = args.width_match_tolerance_mm / 1000.0

    ranges = {mode: mode_range(closure, mode) for mode in args.modes}
    rows: list[dict[str, object]] = []

    for mode in args.modes:
        lo, hi = ranges[mode]
        mode_width_offset_mm = 0.0 if mode == "cylinder" else args.object_width_offset_mm
        width_basis = "cylinder_diameter" if mode == "cylinder" else "object_width_corrected"
        for width_mm in widths_mm:
            width_m = width_mm / 1000.0
            object_width_mm = width_mm - mode_width_offset_mm
            try:
                t0 = time.perf_counter()
                result = solve_mode(closure, mode, width_m)
                solve_time_ms = (time.perf_counter() - t0) * 1000.0
                width_error_mm = abs(result.width - width_m) * 1000.0
                coplanarity_mm, z_span_mm = z_metrics(result)
                nominal_feasible, width_match, reason = classify_result(
                    width_m,
                    result.width,
                    lo,
                    hi,
                    tol_m,
                )
                rows.append(
                    {
                        "mode": MODE_LABELS[mode],
                        "width_mm": f"{object_width_mm:.3f}",
                        "object_width_mm": f"{object_width_mm:.3f}",
                        "internal_width_mm": f"{width_mm:.3f}",
                        "width_basis": width_basis,
                        "solve_success": True,
                        "nominal_feasible": nominal_feasible,
                        "width_match": width_match,
                        "solve_time_ms": f"{solve_time_ms:.6f}",
                        "width_error_mm": f"{width_error_mm:.6f}",
                        "coplanarity_error_mm": f"{coplanarity_mm:.6f}",
                        "z_span_mm": f"{z_span_mm:.6f}",
                        "tilt_deg": f"{result.tilt_deg:.6f}",
                        "joint_margin": f"{joint_margin(result, fk, mode):.6f}",
                        "reason": reason,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "mode": MODE_LABELS[mode],
                        "width_mm": f"{object_width_mm:.3f}",
                        "object_width_mm": f"{object_width_mm:.3f}",
                        "internal_width_mm": f"{width_mm:.3f}",
                        "width_basis": width_basis,
                        "solve_success": False,
                        "nominal_feasible": False,
                        "width_match": False,
                        "solve_time_ms": "",
                        "width_error_mm": "",
                        "coplanarity_error_mm": "",
                        "z_span_mm": "",
                        "tilt_deg": "",
                        "joint_margin": "",
                        "reason": f"exception:{type(exc).__name__}:{exc}",
                    }
                )

    write_csv(args.out / "summary.csv", rows)
    write_assumptions(args.out / "assumptions.json", args, ranges)
    if not args.no_plots:
        maybe_write_plots(args.out, rows)

    n_feasible = sum(1 for row in rows if row["nominal_feasible"] is True)
    n_width_match = sum(1 for row in rows if row["width_match"] is True)
    print(
        f"Wrote {args.out / 'summary.csv'} "
        f"({n_feasible}/{len(rows)} nominally feasible, "
        f"{n_width_match}/{len(rows)} width-matched rows)"
    )
    print(f"Wrote {args.out / 'assumptions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
