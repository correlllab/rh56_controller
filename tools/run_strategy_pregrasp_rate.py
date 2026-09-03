#!/usr/bin/env python3
"""Coarse feasible-rate sweep for strategy-specific RH56 pre-grasp poses.

This is the first-rate version of the strategy pre-grasp demo.  For one fixed
object and analytical final grasp pose, it samples hand-base start positions and
reports the fraction that can reach each strategy-specific pre-grasp target
without object or floor collision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rh56_controller.capsule_hand_proxy import (
    COLLISION_NUMERICAL_EPSILON_M,
    closure_base_rotation,
)
from rh56_controller.grasp_geometry import ClosureGeometry, InspireHandFK
from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS
from tools.demo_strategy_pregrasp_collision import (
    build_capsules_for_ctrl,
    closure_base_position_for_center,
    evaluate_strategy,
    grasp_center_base,
    pregrasp_width_policy,
    solve_mode,
    strategy_poses,
    tabletop_grasp_target,
    tabletop_object_center,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute coarse feasible rates for RH56 strategy pre-grasp poses."
    )
    parser.add_argument("--object", choices=sorted(BUILTIN_OBJECTS), default="debug_40mm_cube")
    parser.add_argument(
        "--mode",
        choices=["object-default", "line", "plane3", "plane4", "plane5", "cylinder"],
        default="plane4",
        help="Analytical final grasp mode. Use object-default to use the object metadata.",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/strategy_pregrasp_rate_40mm"))
    parser.add_argument("--xml", default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--rebuild-fk", action="store_true", help="Rebuild FK cache.")
    parser.add_argument("--object-width-offset-mm", type=float, default=20.0)
    parser.add_argument(
        "--grasp-center-policy",
        choices=["antipodal", "contact-centroid"],
        default="antipodal",
        help=(
            "How to align the object AABB to the analytical grasp. antipodal "
            "uses the midpoint between thumb and non-thumb fingertip centroid; "
            "contact-centroid uses ClosureResult.midpoint for legacy viewer parity."
        ),
    )
    parser.add_argument(
        "--grasp-target-z-fraction",
        type=float,
        default=None,
        help=(
            "Override the object metadata z fraction for the analytical grasp "
            "target. Fractions are measured from object bottom=0 to top=1."
        ),
    )
    parser.add_argument(
        "--grasp-target-top-offset-mm",
        type=float,
        default=None,
        help=(
            "Override the object metadata and place the grasp target this "
            "distance below the object top. Mutually exclusive with "
            "--grasp-target-z-fraction."
        ),
    )
    parser.add_argument(
        "--grasp-target-approach-offset-mm",
        type=float,
        default=0.0,
        help=(
            "Shift the grasp target toward the approach side by this distance. "
            "This models upper-side/near-side grasps for bulky tabletop objects."
        ),
    )
    parser.add_argument(
        "--preopen-mm",
        type=float,
        default=5.0,
        help=(
            "Width margin added to the final grasp width for iterative_closure "
            "pre-grasp when --iterative-pregrasp-policy=final-plus-preopen."
        ),
    )
    parser.add_argument(
        "--iterative-pregrasp-policy",
        choices=["planner-max-width", "final-plus-preopen"],
        default="planner-max-width",
        help=(
            "How to choose the Plan/iterative_closure pre-grasp width when "
            "--iterative-width-mm is not set. planner-max-width matches the "
            "interactive planner default."
        ),
    )
    parser.add_argument(
        "--iterative-width-mm",
        type=float,
        default=None,
        help="Explicit iterative_closure pre-grasp width. Overrides --preopen-mm.",
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=0.0,
        help="Single yaw angle, or the first yaw angle when --yaw-samples > 1.",
    )
    parser.add_argument(
        "--yaw-samples",
        type=int,
        default=1,
        help=(
            "Number of yaw angles. 1 uses --yaw-deg. Values >1 sample "
            "--yaw-span-deg evenly starting at --yaw-deg."
        ),
    )
    parser.add_argument(
        "--yaw-span-deg",
        type=float,
        default=360.0,
        help="Yaw span used when --yaw-samples > 1.",
    )
    parser.add_argument(
        "--start-sampler",
        choices=[
            "final-offset-grid",
            "object-top-grid",
            "approach-plane-grid",
            "paper-approach-points",
        ],
        default="final-offset-grid",
        help=(
            "final-offset-grid preserves the original final_grasp_base + offset "
            "sampling. object-top-grid samples a horizontal grid above the "
            "object tabletop AABB. approach-plane-grid samples an initial hand "
            "plane offset from the object along an approach axis. "
            "paper-approach-points samples the 10-point paper-style setup."
        ),
    )
    parser.add_argument(
        "--start-reference",
        choices=["grasp-center", "base"],
        default="grasp-center",
        help=(
            "For object-top-grid, approach-plane-grid, and paper-approach-points, "
            "interpret sampled points as either the strategy grasp-center "
            "waypoint or the raw hand-base waypoint."
        ),
    )
    parser.add_argument("--x-offset-range-mm", type=float, nargs=2, default=[-120.0, 120.0])
    parser.add_argument("--y-offset-range-mm", type=float, nargs=2, default=[-120.0, 120.0])
    parser.add_argument("--z-offset-range-mm", type=float, nargs=2, default=[40.0, 160.0])
    parser.add_argument("--xy-step-mm", type=float, default=120.0)
    parser.add_argument("--z-step-mm", type=float, default=60.0)
    parser.add_argument("--top-x-range-mm", type=float, nargs=2, default=[-40.0, 40.0])
    parser.add_argument("--top-y-range-mm", type=float, nargs=2, default=[-40.0, 40.0])
    parser.add_argument(
        "--top-height-mm",
        type=float,
        default=80.0,
        help="Height above the object top face for object-top-grid.",
    )
    parser.add_argument(
        "--top-height-range-mm",
        type=float,
        nargs=2,
        default=None,
        help="Optional height range above the object top face for object-top-grid.",
    )
    parser.add_argument("--top-xy-step-mm", type=float, default=40.0)
    parser.add_argument("--top-z-step-mm", type=float, default=40.0)
    parser.add_argument(
        "--approach-axis",
        choices=["x-", "x+", "y-", "y+"],
        default="y-",
        help=(
            "World-frame side from which the hand starts for approach-plane-grid "
            "or paper-approach-points."
        ),
    )
    parser.add_argument(
        "--approach-distance-mm",
        type=float,
        default=150.0,
        help="Single distance from the object center along --approach-axis.",
    )
    parser.add_argument(
        "--approach-distance-range-mm",
        type=float,
        nargs=2,
        default=None,
        help="Optional distance range along --approach-axis.",
    )
    parser.add_argument("--approach-distance-step-mm", type=float, default=100.0)
    parser.add_argument(
        "--approach-lateral-range-mm",
        type=float,
        nargs=2,
        default=[-100.0, 100.0],
        help="Lateral grid range in the approach plane.",
    )
    parser.add_argument(
        "--approach-lateral-step-mm",
        type=float,
        default=100.0,
        help="Lateral grid spacing in the approach plane.",
    )
    parser.add_argument(
        "--approach-height-range-mm",
        type=float,
        nargs=2,
        default=[40.0, 240.0],
        help="Height range above the object top face for approach-plane-grid.",
    )
    parser.add_argument(
        "--approach-height-step-mm",
        type=float,
        default=100.0,
        help="Vertical spacing for approach-plane-grid.",
    )
    parser.add_argument(
        "--paper-p1-distance-mm",
        type=float,
        default=250.0,
        help="P1 distance from the grasp point along the approach axis.",
    )
    parser.add_argument(
        "--paper-p2-height-mm",
        type=float,
        default=250.0,
        help="P2 height above the final grasp center.",
    )
    parser.add_argument(
        "--paper-level1-height-mm",
        type=float,
        default=100.0,
        help="Height for the first intermediate paper point level.",
    )
    parser.add_argument(
        "--paper-level2-height-mm",
        type=float,
        default=250.0,
        help="Height for the second intermediate paper point level.",
    )
    parser.add_argument(
        "--paper-d-list-mm",
        type=float,
        nargs="+",
        default=[-150.0, -50.0, 50.0, 150.0],
        help=(
            "Lateral d coordinates for the four points on each paper level. "
            "These are perpendicular to the P1/P2/grasp-point plane."
        ),
    )
    parser.add_argument(
        "--paper-hand-yaw-offset-deg",
        type=float,
        default=-90.0,
        help=(
            "Additional hand yaw for paper-approach-points after the hand is "
            "aligned to the P1-to-grasp direction. Negative values rotate right "
            "in the world xy plane."
        ),
    )
    parser.add_argument("--path-samples", type=int, default=10)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--floor-z-mm", type=float, default=0.0)
    parser.add_argument(
        "--floor-tolerance-mm",
        type=float,
        default=3.0,
        help=(
            "Small negative floor clearance tolerated for the capsule proxy. "
            "This absorbs proxy-vs-mesh conservatism; reported clearances remain raw."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def frange_mm(bounds: list[float], step_mm: float) -> np.ndarray:
    if step_mm <= 0.0:
        raise ValueError("step must be positive")
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi < lo:
        raise ValueError("range upper bound must be >= lower bound")
    values: list[float] = []
    current = lo
    while current <= hi + 1e-9:
        values.append(current)
        current += step_mm
    return np.asarray(values, dtype=float)


def yaw_values_deg(args: argparse.Namespace) -> np.ndarray:
    if args.yaw_samples <= 0:
        raise ValueError("--yaw-samples must be positive")
    if args.yaw_samples == 1:
        return np.asarray([float(args.yaw_deg)], dtype=float)
    step = float(args.yaw_span_deg) / float(args.yaw_samples)
    return float(args.yaw_deg) + step * np.arange(args.yaw_samples, dtype=float)


def top_height_values_mm(args: argparse.Namespace) -> np.ndarray:
    if args.top_height_range_mm is None:
        return np.asarray([float(args.top_height_mm)], dtype=float)
    return frange_mm(args.top_height_range_mm, args.top_z_step_mm)


def approach_distance_values_mm(args: argparse.Namespace) -> np.ndarray:
    if args.approach_distance_range_mm is None:
        return np.asarray([float(args.approach_distance_mm)], dtype=float)
    return frange_mm(args.approach_distance_range_mm, args.approach_distance_step_mm)


def approach_basis(axis: str) -> tuple[np.ndarray, np.ndarray]:
    if axis == "x-":
        return np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if axis == "x+":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if axis == "y-":
        return np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    if axis == "y+":
        return np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    raise ValueError(f"Unsupported approach axis: {axis}")


def facing_yaw_for_approach_axis(axis: str) -> float:
    """Yaw that points the hand along the P1-to-grasp movement direction."""

    if axis == "y-":
        return 0.0
    if axis == "x-":
        return -90.0
    if axis == "y+":
        return 180.0
    if axis == "x+":
        return 90.0
    raise ValueError(f"Unsupported approach axis: {axis}")


def paper_approach_points_mm(args: argparse.Namespace) -> list[dict[str, float | str]]:
    points: list[dict[str, float | str]] = [
        {
            "point_id": "P1",
            "order": 1.0,
            "approach_distance_mm": float(args.paper_p1_distance_mm),
            "d_mm": 0.0,
            "h_mm": 0.0,
        },
        {
            "point_id": "P2",
            "order": 2.0,
            "approach_distance_mm": 0.0,
            "d_mm": 0.0,
            "h_mm": float(args.paper_p2_height_mm),
        },
    ]
    order = 3
    for level_name, height in (
        ("L1", float(args.paper_level1_height_mm)),
        ("L2", float(args.paper_level2_height_mm)),
    ):
        for d_mm in args.paper_d_list_mm:
            points.append(
                {
                    "point_id": f"{level_name}_d{float(d_mm):+g}",
                    "order": float(order),
                    "approach_distance_mm": float(args.paper_p1_distance_mm),
                    "d_mm": float(d_mm),
                    "h_mm": height,
                }
            )
            order += 1
    return points


def grasp_center_error_mm(
    result,
    base_position: np.ndarray,
    rotation: np.ndarray,
    object_center: np.ndarray,
    policy: str,
) -> float:
    center_world = base_position + rotation @ grasp_center_base(result, policy)
    return float(np.linalg.norm(center_world - object_center) * 1000.0)


def write_volume(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "strategy",
        "valid",
        "dominant_blocker",
        "yaw_deg",
        "start_sampler",
        "start_reference",
        "point_id",
        "point_order",
        "point_d_mm",
        "point_h_mm",
        "approach_axis",
        "approach_distance_mm",
        "grid_lateral_mm",
        "grid_height_mm",
        "offset_x_mm",
        "offset_y_mm",
        "offset_z_mm",
        "start_center_x_mm",
        "start_center_y_mm",
        "start_center_z_mm",
        "target_center_x_mm",
        "target_center_y_mm",
        "target_center_z_mm",
        "start_x_mm",
        "start_y_mm",
        "start_z_mm",
        "target_x_mm",
        "target_y_mm",
        "target_z_mm",
        "target_width_mm",
        "object_path_collision",
        "object_target_collision",
        "floor_path_collision",
        "floor_target_collision",
        "min_object_clearance_mm",
        "min_floor_clearance_mm",
        "nearest_blocker",
        "max_active_object_collisions",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "strategy",
        "start_sampler",
        "start_reference",
        "position_samples",
        "yaw_samples",
        "total_samples",
        "valid_samples",
        "feasible_rate",
        "object_path_collision_count",
        "object_target_collision_count",
        "floor_path_collision_count",
        "floor_target_collision_count",
        "mean_object_clearance_mm",
        "mean_floor_clearance_mm",
        "dominant_blocker",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_plots(
    out_dir: Path,
    summary_rows: list[dict[str, object]],
    volume_rows: list[dict[str, object]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_skipped.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return

    names = [str(row["strategy"]) for row in summary_rows]
    rates = [float(row["feasible_rate"]) for row in summary_rows]
    colors = ["#d55e00", "#0072b2", "#009e73"]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(names, rates, color=colors[: len(names)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("feasible start fraction")
    ax.set_title("Strategy pre-grasp feasible rate")
    for i, rate in enumerate(rates):
        ax.text(i, min(rate + 0.03, 0.96), f"{rate:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_feasible_rate.png", dpi=150)
    plt.close(fig)

    top_rows = [
        row for row in volume_rows
        if row.get("start_sampler") == "object-top-grid"
    ]
    strategy_names = [str(row["strategy"]) for row in summary_rows]

    if top_rows:
        z_values = sorted({float(row["offset_z_mm"]) for row in top_rows})
        x_values = sorted({float(row["offset_x_mm"]) for row in top_rows})
        y_values = sorted({float(row["offset_y_mm"]) for row in top_rows})

        for z_value in z_values:
            fig, axes = plt.subplots(
                1,
                len(strategy_names),
                figsize=(4.2 * len(strategy_names), 3.8),
                squeeze=False,
            )
            image = None
            for idx, strategy_name in enumerate(strategy_names):
                ax = axes[0][idx]
                grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
                for yi, y_value in enumerate(y_values):
                    for xi, x_value in enumerate(x_values):
                        matches = [
                            float(row["valid"])
                            for row in top_rows
                            if (
                                str(row["strategy"]) == strategy_name
                                and float(row["offset_x_mm"]) == x_value
                                and float(row["offset_y_mm"]) == y_value
                                and float(row["offset_z_mm"]) == z_value
                            )
                        ]
                        if matches:
                            grid[yi, xi] = float(np.mean(matches))
                image = ax.imshow(grid, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
                ax.set_title(strategy_name)
                ax.set_xlabel("x from object center [mm]")
                ax.set_ylabel("y from object center [mm]")
                ax.set_xticks(range(len(x_values)), [f"{value:g}" for value in x_values])
                ax.set_yticks(range(len(y_values)), [f"{value:g}" for value in y_values])
                for yi, y_value in enumerate(y_values):
                    for xi, x_value in enumerate(x_values):
                        value = grid[yi, xi]
                        if not np.isnan(value):
                            ax.text(
                                xi,
                                yi,
                                f"{value:.2f}",
                                ha="center",
                                va="center",
                                color="white" if value < 0.55 else "black",
                                fontsize=8,
                            )
            if image is not None:
                fig.colorbar(image, ax=axes.ravel().tolist(), label="valid fraction")
            fig.suptitle(
                f"Object-top start grid, {z_value:g} mm above object top",
                fontsize=12,
            )
            if len(z_values) == 1:
                plot_path = out_dir / "strategy_top_grid_yaw_fraction.png"
            else:
                plot_path = out_dir / f"strategy_top_grid_yaw_fraction_z_{z_value:g}mm.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)

    approach_rows = [
        row for row in volume_rows
        if row.get("start_sampler") in ("approach-plane-grid", "paper-approach-points")
    ]
    if not approach_rows:
        return

    distances = sorted({float(row["approach_distance_mm"]) for row in approach_rows})
    lateral_values = sorted({float(row["grid_lateral_mm"]) for row in approach_rows})
    height_values = sorted({float(row["grid_height_mm"]) for row in approach_rows})
    approach_axis = str(approach_rows[0].get("approach_axis", ""))
    is_paper_points = any(row.get("start_sampler") == "paper-approach-points" for row in approach_rows)

    if is_paper_points:
        p1_distance = max(distances) if distances else 0.0
        fig, axes = plt.subplots(
            1,
            len(strategy_names),
            figsize=(4.2 * len(strategy_names), 3.8),
            squeeze=False,
            constrained_layout=True,
        )
        image = None
        for idx, strategy_name in enumerate(strategy_names):
            ax = axes[0][idx]
            grid = np.full((len(height_values), len(lateral_values)), np.nan, dtype=float)
            labels: dict[tuple[int, int], str] = {}
            for hi, height in enumerate(height_values):
                for li, lateral in enumerate(lateral_values):
                    matches = [
                        float(row["valid"])
                        for row in approach_rows
                        if (
                            str(row["strategy"]) == strategy_name
                            and float(row["grid_lateral_mm"]) == lateral
                            and float(row["grid_height_mm"]) == height
                        )
                    ]
                    point_labels = [
                        str(row["point_id"])
                        for row in approach_rows
                        if (
                            str(row["strategy"]) == strategy_name
                            and float(row["grid_lateral_mm"]) == lateral
                            and float(row["grid_height_mm"]) == height
                            and str(row.get("point_id", ""))
                        )
                    ]
                    if matches:
                        grid[hi, li] = float(np.mean(matches))
                    if point_labels:
                        labels[(hi, li)] = "/".join(sorted(set(point_labels)))
            image = ax.imshow(grid, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_title(strategy_name)
            ax.set_xlabel("lateral d [mm]")
            if idx == 0:
                ax.set_ylabel("h above final grasp center [mm]")
            ax.set_xticks(range(len(lateral_values)), [f"{value:g}" for value in lateral_values])
            ax.set_yticks(range(len(height_values)), [f"{value:g}" for value in height_values])
            for hi, height in enumerate(height_values):
                for li, lateral in enumerate(lateral_values):
                    value = grid[hi, li]
                    if not np.isnan(value):
                        label = labels.get((hi, li), "")
                        text = f"{value:.2f}" if not label else f"{value:.2f}\n{label}"
                        ax.text(
                            li,
                            hi,
                            text,
                            ha="center",
                            va="center",
                            color="white" if value < 0.55 else "black",
                            fontsize=7,
                        )
        if image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), label="valid fraction")
        fig.suptitle(
            (
                "Paper approach points: P1/level plane "
                f"{p1_distance:g} mm along {approach_axis}"
            ),
            fontsize=12,
        )
        fig.savefig(out_dir / "strategy_paper_approach_points_validity.png", dpi=150)
        plt.close(fig)
        return

    for distance in distances:
        fig, axes = plt.subplots(
            1,
            len(strategy_names),
            figsize=(4.2 * len(strategy_names), 3.8),
            squeeze=False,
        )
        image = None
        for idx, strategy_name in enumerate(strategy_names):
            ax = axes[0][idx]
            grid = np.full((len(height_values), len(lateral_values)), np.nan, dtype=float)
            for hi, height in enumerate(height_values):
                for li, lateral in enumerate(lateral_values):
                    matches = [
                        float(row["valid"])
                        for row in approach_rows
                        if (
                            str(row["strategy"]) == strategy_name
                            and float(row["approach_distance_mm"]) == distance
                            and float(row["grid_lateral_mm"]) == lateral
                            and float(row["grid_height_mm"]) == height
                        )
                    ]
                    if matches:
                        grid[hi, li] = float(np.mean(matches))
            image = ax.imshow(grid, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_title(strategy_name)
            ax.set_xlabel("d along approach axis [mm]" if is_paper_points else "lateral offset [mm]")
            ax.set_ylabel("h above final grasp center [mm]" if is_paper_points else "height above object top [mm]")
            ax.set_xticks(range(len(lateral_values)), [f"{value:g}" for value in lateral_values])
            ax.set_yticks(range(len(height_values)), [f"{value:g}" for value in height_values])
            for hi, height in enumerate(height_values):
                for li, lateral in enumerate(lateral_values):
                    value = grid[hi, li]
                    if not np.isnan(value):
                        ax.text(
                            li,
                            hi,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="white" if value < 0.55 else "black",
                            fontsize=8,
                        )
        if image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), label="valid fraction")
        title = (
            f"Paper approach points, P1 distance {distance:g} mm along {approach_axis}"
            if is_paper_points
            else f"Approach-plane grid, {distance:g} mm along {approach_axis}"
        )
        fig.suptitle(title, fontsize=12)
        if is_paper_points:
            plot_path = out_dir / "strategy_paper_approach_points_validity.png"
        elif len(distances) == 1:
            plot_path = out_dir / "strategy_approach_plane_validity.png"
        else:
            plot_path = out_dir / f"strategy_approach_plane_validity_d_{distance:g}mm.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    if (
        args.grasp_target_z_fraction is not None
        and args.grasp_target_top_offset_mm is not None
    ):
        raise SystemExit(
            "--grasp-target-z-fraction and --grasp-target-top-offset-mm "
            "are mutually exclusive"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    obj = BUILTIN_OBJECTS[args.object]
    mode = obj.mode if args.mode == "object-default" else args.mode
    target_width_m = obj.grasp_width_m + args.object_width_offset_mm / 1000.0
    preopen_m = args.preopen_mm / 1000.0
    iterative_width_m = (
        args.iterative_width_mm / 1000.0
        if args.iterative_width_mm is not None
        else None
    )
    yaw_degrees = yaw_values_deg(args)
    if args.start_sampler == "paper-approach-points":
        yaw_degrees = (
            yaw_degrees
            + facing_yaw_for_approach_axis(args.approach_axis)
            + float(args.paper_hand_yaw_offset_deg)
        )
    floor_z_m = args.floor_z_mm / 1000.0
    floor_tolerance_m = args.floor_tolerance_mm / 1000.0

    fk = (
        InspireHandFK(xml_path=args.xml, rebuild=args.rebuild_fk)
        if args.xml
        else InspireHandFK(rebuild=args.rebuild_fk)
    )
    closure = ClosureGeometry(fk)
    final_result = solve_mode(closure, mode, target_width_m)
    aabb_center = tabletop_object_center(obj)
    grasp_target = tabletop_grasp_target(
        obj,
        z_fraction=args.grasp_target_z_fraction,
        top_offset_m=(
            args.grasp_target_top_offset_mm / 1000.0
            if args.grasp_target_top_offset_mm is not None
            else None
        ),
    )
    half_extents = np.asarray(obj.size_m, dtype=float) / 2.0

    strategies = strategy_poses(
        fk,
        closure,
        mode=mode,
        final_result=final_result,
        target_width_m=target_width_m,
        preopen_m=preopen_m,
        iterative_width_m=iterative_width_m,
        iterative_pregrasp_policy=args.iterative_pregrasp_policy,
    )

    model = mujoco.MjModel.from_xml_path(str(args.xml or fk.xml_path))
    data = mujoco.MjData(model)
    capsules_by_strategy = {
        strategy.name: build_capsules_for_ctrl(
            model,
            data,
            strategy.ctrl_values,
            radius_scale=args.radius_scale,
        )
        for strategy in strategies
    }

    if args.start_sampler == "final-offset-grid":
        x_offsets = frange_mm(args.x_offset_range_mm, args.xy_step_mm)
        y_offsets = frange_mm(args.y_offset_range_mm, args.xy_step_mm)
        z_offsets = frange_mm(args.z_offset_range_mm, args.z_step_mm)
        row_start_reference = "base"
    elif args.start_sampler == "object-top-grid":
        x_offsets = frange_mm(args.top_x_range_mm, args.top_xy_step_mm)
        y_offsets = frange_mm(args.top_y_range_mm, args.top_xy_step_mm)
        z_offsets = top_height_values_mm(args)
        row_start_reference = args.start_reference
    elif args.start_sampler == "approach-plane-grid":
        x_offsets = frange_mm(args.approach_lateral_range_mm, args.approach_lateral_step_mm)
        y_offsets = frange_mm(args.approach_height_range_mm, args.approach_height_step_mm)
        z_offsets = approach_distance_values_mm(args)
        row_start_reference = args.start_reference
    else:
        paper_points = paper_approach_points_mm(args)
        x_offsets = np.arange(len(paper_points), dtype=float)
        y_offsets = np.asarray([0.0], dtype=float)
        z_offsets = np.asarray([0.0], dtype=float)
        row_start_reference = args.start_reference
    position_samples_per_yaw = len(x_offsets) * len(y_offsets) * len(z_offsets)

    volume_rows: list[dict[str, object]] = []
    stats = {
        strategy.name: {
            "total": 0,
            "valid": 0,
            "object_path": 0,
            "object_target": 0,
            "floor_path": 0,
            "floor_target": 0,
            "object_clearance": [],
            "floor_clearance": [],
            "blockers": {},
        }
        for strategy in strategies
    }
    final_center_errors: list[float] = []
    target_center_errors_by_strategy: dict[str, list[float]] = {
        strategy.name: [] for strategy in strategies
    }

    object_top_z_m = float(aabb_center[2] + half_extents[2])
    approach_vector, lateral_vector = approach_basis(args.approach_axis)
    grasp_target = grasp_target + approach_vector * (
        args.grasp_target_approach_offset_mm / 1000.0
    )

    for yaw_deg in yaw_degrees:
        yaw_rad = math.radians(float(yaw_deg))
        final_grasp = closure_base_position_for_center(
            final_result,
            yaw_rad,
            object_center=grasp_target,
            policy=args.grasp_center_policy,
        )
        final_rotation = closure_base_rotation(final_result, yaw_rad)
        final_center_errors.append(
            grasp_center_error_mm(
                final_result,
                final_grasp,
                final_rotation,
                grasp_target,
                args.grasp_center_policy,
            )
        )
        target_by_strategy = {
            strategy.name: closure_base_position_for_center(
                strategy.target_result,
                yaw_rad,
                object_center=grasp_target,
                policy=args.grasp_center_policy,
            )
            for strategy in strategies
        }
        rotation_by_strategy = {
            strategy.name: closure_base_rotation(strategy.target_result, yaw_rad)
            for strategy in strategies
        }
        for strategy in strategies:
            target_center_errors_by_strategy[strategy.name].append(
                grasp_center_error_mm(
                    strategy.target_result,
                    target_by_strategy[strategy.name],
                    rotation_by_strategy[strategy.name],
                    grasp_target,
                    args.grasp_center_policy,
                )
            )

        for dx in x_offsets:
            for dy in y_offsets:
                for dz in z_offsets:
                    if args.start_sampler == "final-offset-grid":
                        offset = np.array([dx, dy, dz], dtype=float) / 1000.0
                        start_center = final_grasp + offset
                        row_offset_x, row_offset_y, row_offset_z = dx, dy, dz
                        row_approach_distance = ""
                        row_grid_lateral = ""
                        row_grid_height = ""
                        row_point_id = ""
                        row_point_order = ""
                        row_point_d = ""
                        row_point_h = ""
                    elif args.start_sampler == "object-top-grid":
                        start_center = np.array(
                            [
                                aabb_center[0] + dx / 1000.0,
                                aabb_center[1] + dy / 1000.0,
                                object_top_z_m + dz / 1000.0,
                            ],
                            dtype=float,
                        )
                        row_offset_x, row_offset_y, row_offset_z = dx, dy, dz
                        row_approach_distance = ""
                        row_grid_lateral = ""
                        row_grid_height = ""
                        row_point_id = ""
                        row_point_order = ""
                        row_point_d = ""
                        row_point_h = ""
                    elif args.start_sampler == "approach-plane-grid":
                        start_center = (
                            grasp_target
                            + approach_vector * (dz / 1000.0)
                            + lateral_vector * (dx / 1000.0)
                        )
                        start_center[2] = object_top_z_m + dy / 1000.0
                        row_offset = (start_center - grasp_target) * 1000.0
                        row_offset_x = float(row_offset[0])
                        row_offset_y = float(row_offset[1])
                        row_offset_z = float(row_offset[2])
                        row_approach_distance = f"{dz:.3f}"
                        row_grid_lateral = f"{dx:.3f}"
                        row_grid_height = f"{dy:.3f}"
                        row_point_id = ""
                        row_point_order = ""
                        row_point_d = ""
                        row_point_h = ""
                    else:
                        point = paper_points[int(dx)]
                        point_id = str(point["point_id"])
                        point_order = float(point["order"])
                        point_approach_distance_mm = float(point["approach_distance_mm"])
                        point_d_mm = float(point["d_mm"])
                        point_h_mm = float(point["h_mm"])
                        start_center = (
                            grasp_target
                            + approach_vector * (point_approach_distance_mm / 1000.0)
                            + lateral_vector * (point_d_mm / 1000.0)
                        )
                        start_center[2] = grasp_target[2] + point_h_mm / 1000.0
                        row_offset = (start_center - grasp_target) * 1000.0
                        row_offset_x = float(row_offset[0])
                        row_offset_y = float(row_offset[1])
                        row_offset_z = float(row_offset[2])
                        row_approach_distance = f"{point_approach_distance_mm:.3f}"
                        row_grid_lateral = f"{point_d_mm:.3f}"
                        row_grid_height = f"{point_h_mm:.3f}"
                        row_point_id = point_id
                        row_point_order = f"{point_order:.0f}"
                        row_point_d = f"{point_d_mm:.3f}"
                        row_point_h = f"{point_h_mm:.3f}"
                    for strategy in strategies:
                        if args.start_sampler != "final-offset-grid" and args.start_reference == "grasp-center":
                            start = start_center - (
                                rotation_by_strategy[strategy.name]
                                @ grasp_center_base(strategy.target_result, args.grasp_center_policy)
                            )
                        else:
                            start = start_center
                        evaluation = evaluate_strategy(
                            strategy,
                            capsules_base=capsules_by_strategy[strategy.name],
                            start=start,
                            target=target_by_strategy[strategy.name],
                            rotation=rotation_by_strategy[strategy.name],
                            half_extents=half_extents,
                            aabb_center=aabb_center,
                            object_shape=obj.collision_shape,
                            floor_z_m=floor_z_m,
                            floor_tolerance_m=floor_tolerance_m,
                            path_samples=args.path_samples,
                        )
                        stat = stats[strategy.name]
                        stat["total"] += 1
                        stat["valid"] += int(evaluation.valid)
                        stat["object_path"] += int(evaluation.object_path_collision)
                        stat["object_target"] += int(evaluation.object_target_collision)
                        stat["floor_path"] += int(evaluation.floor_path_collision)
                        stat["floor_target"] += int(evaluation.floor_target_collision)
                        stat["object_clearance"].append(evaluation.min_object_clearance_m * 1000.0)
                        stat["floor_clearance"].append(evaluation.min_floor_clearance_m * 1000.0)
                        blockers = stat["blockers"]
                        blockers[evaluation.dominant_blocker] = blockers.get(evaluation.dominant_blocker, 0) + 1
                        volume_rows.append(
                            {
                                "strategy": strategy.name,
                                "valid": int(evaluation.valid),
                                "dominant_blocker": evaluation.dominant_blocker,
                                "yaw_deg": f"{float(yaw_deg):.3f}",
                                "start_sampler": args.start_sampler,
                                "start_reference": row_start_reference,
                                "point_id": row_point_id,
                                "point_order": row_point_order,
                                "point_d_mm": row_point_d,
                                "point_h_mm": row_point_h,
                                "approach_axis": args.approach_axis if args.start_sampler in ("approach-plane-grid", "paper-approach-points") else "",
                                "approach_distance_mm": row_approach_distance,
                                "grid_lateral_mm": row_grid_lateral,
                                "grid_height_mm": row_grid_height,
                                "offset_x_mm": f"{row_offset_x:.3f}",
                                "offset_y_mm": f"{row_offset_y:.3f}",
                                "offset_z_mm": f"{row_offset_z:.3f}",
                                "start_center_x_mm": f"{start_center[0] * 1000.0:.3f}",
                                "start_center_y_mm": f"{start_center[1] * 1000.0:.3f}",
                                "start_center_z_mm": f"{start_center[2] * 1000.0:.3f}",
                                "target_center_x_mm": f"{grasp_target[0] * 1000.0:.3f}",
                                "target_center_y_mm": f"{grasp_target[1] * 1000.0:.3f}",
                                "target_center_z_mm": f"{grasp_target[2] * 1000.0:.3f}",
                                "start_x_mm": f"{start[0] * 1000.0:.3f}",
                                "start_y_mm": f"{start[1] * 1000.0:.3f}",
                                "start_z_mm": f"{start[2] * 1000.0:.3f}",
                                "target_x_mm": f"{evaluation.target[0] * 1000.0:.3f}",
                                "target_y_mm": f"{evaluation.target[1] * 1000.0:.3f}",
                                "target_z_mm": f"{evaluation.target[2] * 1000.0:.3f}",
                                "target_width_mm": f"{strategy.target_result.width * 1000.0:.3f}",
                                "object_path_collision": int(evaluation.object_path_collision),
                                "object_target_collision": int(evaluation.object_target_collision),
                                "floor_path_collision": int(evaluation.floor_path_collision),
                                "floor_target_collision": int(evaluation.floor_target_collision),
                                "min_object_clearance_mm": f"{evaluation.min_object_clearance_m * 1000.0:.3f}",
                                "min_floor_clearance_mm": f"{evaluation.min_floor_clearance_m * 1000.0:.3f}",
                                "nearest_blocker": evaluation.nearest_object,
                                "max_active_object_collisions": evaluation.max_active_object_collisions,
                            }
                        )

    summary_rows: list[dict[str, object]] = []
    for strategy in strategies:
        stat = stats[strategy.name]
        total = int(stat["total"])
        valid = int(stat["valid"])
        blockers = dict(stat["blockers"])
        failure_blockers = {
            key: value for key, value in blockers.items()
            if key != "ok"
        }
        dominant_source = failure_blockers or blockers
        dominant = max(dominant_source.items(), key=lambda item: item[1])[0] if dominant_source else ""
        summary_rows.append(
            {
                "strategy": strategy.name,
                "start_sampler": args.start_sampler,
                "start_reference": row_start_reference,
                "position_samples": position_samples_per_yaw,
                "yaw_samples": len(yaw_degrees),
                "total_samples": total,
                "valid_samples": valid,
                "feasible_rate": f"{(valid / total) if total else 0.0:.6f}",
                "object_path_collision_count": int(stat["object_path"]),
                "object_target_collision_count": int(stat["object_target"]),
                "floor_path_collision_count": int(stat["floor_path"]),
                "floor_target_collision_count": int(stat["floor_target"]),
                "mean_object_clearance_mm": f"{float(np.mean(stat['object_clearance'])):.3f}",
                "mean_floor_clearance_mm": f"{float(np.mean(stat['floor_clearance'])):.3f}",
                "dominant_blocker": dominant,
            }
        )

    write_volume(args.out / "volume.csv", volume_rows)
    write_summary(args.out / "summary.csv", summary_rows)
    final_center_error_max = max(final_center_errors) if final_center_errors else 0.0
    target_center_error_max_by_strategy = {
        name: max(values) if values else 0.0
        for name, values in target_center_errors_by_strategy.items()
    }
    if args.start_sampler == "final-offset-grid":
        start_definition = (
            "Each start is final_grasp_base(yaw) + [dx, dy, dz] from "
            "offset_grid_mm. The sampled point is the hand-base pose."
        )
        start_grid_payload = {
            "offset_grid_mm": {
                "x": x_offsets.tolist(),
                "y": y_offsets.tolist(),
                "z": z_offsets.tolist(),
            }
        }
    elif args.start_sampler == "object-top-grid":
        start_definition = (
            "Each start grid point is sampled on a horizontal plane above the "
            "object top face. With start_reference=grasp-center, the sampled "
            "point is the strategy grasp-center waypoint and is converted to a "
            "hand-base start pose separately for each strategy and yaw."
        )
        start_grid_payload = {
            "top_grid_mm": {
                "x_from_object_center": x_offsets.tolist(),
                "y_from_object_center": y_offsets.tolist(),
                "height_above_object_top": z_offsets.tolist(),
            }
        }
    elif args.start_sampler == "approach-plane-grid":
        start_definition = (
            "Each start grid point is sampled on an initial hand-position "
            "plane offset from the object along approach_axis. With "
            "start_reference=grasp-center, the sampled point is the strategy "
            "grasp-center waypoint and is converted to a hand-base start pose "
            "separately for each strategy and yaw."
        )
        start_grid_payload = {
            "approach_plane_grid_mm": {
                "approach_axis": args.approach_axis,
                "distance_from_object_center": z_offsets.tolist(),
                "lateral_offset": x_offsets.tolist(),
                "height_above_object_top": y_offsets.tolist(),
            }
        }
    else:
        start_definition = (
            "The 10 paper-style start points use two perpendicular planes. "
            "P1, P2, and the grasp point define the vertical approach plane. "
            "P1 is 250 mm from the grasp point along approach_axis with h=0; "
            "P2 is directly above the grasp point with h=250 mm. The other "
            "eight points are in the vertical plane through P1, perpendicular "
            "to the P1/P2/grasp-point plane: they use the same approach "
            "distance as P1, with h=100/250 mm and lateral d=-150/-50/50/150 "
            "mm. With start_reference=grasp-center, each sampled grasp-center "
            "waypoint is converted to a hand-base start pose separately for "
            "each strategy. The yaw is set from approach_axis so the hand "
            "faces the object along the P1-to-grasp movement direction, then "
            "rotated by paper_hand_yaw_offset_deg."
        )
        start_grid_payload = {
            "paper_approach_points_mm": {
                "approach_axis": args.approach_axis,
                "p1_distance": args.paper_p1_distance_mm,
                "p2_height": args.paper_p2_height_mm,
                "level1_height": args.paper_level1_height_mm,
                "level2_height": args.paper_level2_height_mm,
                "d_list": args.paper_d_list_mm,
                "points": paper_approach_points_mm(args),
                "facing_yaw_offset_deg": facing_yaw_for_approach_axis(args.approach_axis),
                "hand_yaw_offset_deg": args.paper_hand_yaw_offset_deg,
            }
        }

    effective_grasp_target_fraction = list(obj.grasp_target_fraction)
    effective_grasp_target_fraction[2] = grasp_target[2] / obj.size_m[2]
    if args.grasp_target_top_offset_mm is not None:
        effective_top_offset_m = args.grasp_target_top_offset_mm / 1000.0
    elif args.grasp_target_z_fraction is None:
        effective_top_offset_m = obj.grasp_target_top_offset_m
    else:
        effective_top_offset_m = None

    assumptions = {
        "script": "tools/run_strategy_pregrasp_rate.py",
        "simulation_only": True,
        "uses_hardware": False,
        "object": {
            "name": obj.name,
            "label": obj.label,
            "size_m": obj.size_m,
            "grasp_width_m": obj.grasp_width_m,
            "collision_shape": obj.collision_shape,
            "aabb_center_m": aabb_center.tolist(),
            "grasp_target_fraction": obj.grasp_target_fraction,
            "grasp_target_top_offset_m": obj.grasp_target_top_offset_m,
            "effective_grasp_target_fraction": effective_grasp_target_fraction,
            "grasp_target_z_fraction_override": args.grasp_target_z_fraction,
            "grasp_target_top_offset_mm_override": args.grasp_target_top_offset_mm,
            "effective_grasp_target_top_offset_mm": (
                effective_top_offset_m * 1000.0
                if effective_top_offset_m is not None
                else None
            ),
            "grasp_target_approach_offset_mm": args.grasp_target_approach_offset_mm,
            "grasp_target_m": grasp_target.tolist(),
            "bottom_z_m": 0.0,
        },
        "mode": mode,
        "target_width_m": target_width_m,
        "object_width_offset_mm": args.object_width_offset_mm,
        "grasp_center_policy": args.grasp_center_policy,
        "preopen_mm": args.preopen_mm,
        "iterative_width_mm": args.iterative_width_mm,
        "iterative_pregrasp_policy": args.iterative_pregrasp_policy,
        "pregrasp_width_policy": pregrasp_width_policy(args),
        "yaw_deg": yaw_degrees.tolist(),
        "yaw_samples": len(yaw_degrees),
        "yaw_span_deg": args.yaw_span_deg,
        "final_grasp_center_error_mm_max": final_center_error_max,
        "target_grasp_center_error_mm_max_by_strategy": target_center_error_max_by_strategy,
        "start_sampler": args.start_sampler,
        "start_reference": row_start_reference,
        "position_samples_per_yaw": position_samples_per_yaw,
        "start_definition": start_definition,
        **start_grid_payload,
        "path_samples": args.path_samples,
        "floor_z_m": floor_z_m,
        "floor_tolerance_mm": args.floor_tolerance_mm,
        "radius_scale": args.radius_scale,
        "object_collision_numerical_epsilon_mm": (
            COLLISION_NUMERICAL_EPSILON_M * 1000.0
        ),
        "viability": (
            "A start is feasible for a strategy if its strategy-specific "
            "pre-grasp capsule path reaches that strategy target without "
            f"{obj.collision_shape}-proxy object or floor collision."
        ),
    }
    (args.out / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")
    if not args.no_plots:
        maybe_write_plots(args.out, summary_rows, volume_rows)

    print("RH56 strategy pre-grasp feasible-rate sweep:")
    print("  simulation_only: true")
    print("  hardware_required: false")
    print(f"  object: {obj.name}")
    print(f"  mode: {mode}")
    print(f"  grasp_center_policy: {args.grasp_center_policy}")
    print(f"  pregrasp_width_policy: {pregrasp_width_policy(args)}")
    print(f"  start_sampler: {args.start_sampler}")
    print(f"  start_reference: {row_start_reference}")
    print(f"  position_samples_per_yaw: {position_samples_per_yaw}")
    print(f"  yaw_samples: {len(yaw_degrees)}")
    print(f"  final_grasp_center_error_mm_max: {final_center_error_max:.6f}")
    for row in summary_rows:
        print(
            f"  {row['strategy']}: rate={row['feasible_rate']} "
            f"valid={row['valid_samples']}/{row['total_samples']} "
            f"dominant={row['dominant_blocker']}"
        )
    print(f"Wrote {args.out / 'summary.csv'}")
    print(f"Wrote {args.out / 'volume.csv'}")
    print(f"Wrote {args.out / 'assumptions.json'}")
    if not args.no_plots:
        print(f"Wrote {args.out / 'strategy_feasible_rate.png'}")
        if args.start_sampler == "object-top-grid":
            print(f"Wrote {args.out / 'strategy_top_grid_yaw_fraction.png'}")
        if args.start_sampler == "approach-plane-grid":
            print(f"Wrote {args.out / 'strategy_approach_plane_validity.png'}")
        if args.start_sampler == "paper-approach-points":
            print(f"Wrote {args.out / 'strategy_paper_approach_points_validity.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
