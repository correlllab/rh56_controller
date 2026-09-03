#!/usr/bin/env python3
"""Characterize no-go volume for analytical RH56 grasps.

This script is a simulation-only reachability proxy. It does not perform robot
arm IK or object-aware path planning. Instead, it asks a paper-facing question:

    From each sampled hand-base position around an object, what fraction of
    sampled yaw poses can reach the analytical grasp pose with a straight
    Cartesian move without the capsule hand proxy colliding with the object
    bounding box?

The output is intended to expose where simple analytical closure plus a linear
approach is plausible, and where object-aware path planning is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import mujoco
import numpy as np

from rh56_controller.capsule_hand_proxy import (
    Capsule,
    build_capsule_proxy,
    closure_base_position,
    closure_base_rotation,
    sample_linear_path_collisions,
    set_closure_qpos,
)
from rh56_controller.grasp_geometry import (
    ACTUATOR_NAMES,
    CTRL_MAX,
    ClosureGeometry,
    ClosureResult,
    InspireHandFK,
)
from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS, ObjectSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate no-go volume heatmaps for analytical RH56 grasps."
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["ycb_cracker_box", "ycb_sugar_box", "ycb_potted_meat_can"],
        choices=sorted(BUILTIN_OBJECTS),
        help="Built-in object proxies to evaluate.",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/analytical_grasp_volume"))
    parser.add_argument("--xml", default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--rebuild-fk", action="store_true", help="Rebuild FK cache.")
    parser.add_argument(
        "--object-width-offset-mm",
        type=float,
        default=20.0,
        help="Added to object width before calling line/plane analytical solvers.",
    )
    parser.add_argument("--mode-override", choices=["line", "plane3", "plane4", "plane5"], default=None)
    parser.add_argument("--x-range-mm", type=float, nargs=2, default=[-240.0, 240.0])
    parser.add_argument("--y-range-mm", type=float, nargs=2, default=[-240.0, 240.0])
    parser.add_argument("--z-range-mm", type=float, nargs=2, default=[40.0, 280.0])
    parser.add_argument("--grid-step-mm", type=float, default=60.0)
    parser.add_argument("--yaw-samples", type=int, default=8)
    parser.add_argument(
        "--path-samples",
        type=int,
        default=8,
        help="Visible intervals along each straight-line path; capsule checks are swept over each interval.",
    )
    parser.add_argument(
        "--collision-model",
        choices=["capsule", "point-inflated"],
        default="capsule",
        help="Capsule swept hand proxy, or legacy hand-base point against inflated AABB.",
    )
    parser.add_argument(
        "--path-hand-shape",
        choices=["open", "closed"],
        default="open",
        help="Hand qpos used when extracting the capsule proxy.",
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to all capsule radii.",
    )
    parser.add_argument(
        "--final-ignore-groups",
        choices=["fingers", "all", "none"],
        default="fingers",
        help="Capsule groups allowed to be ignored in the final-contact region.",
    )
    parser.add_argument(
        "--final-contact-ignore-mm",
        type=float,
        default=0.0,
        help="Only ignore selected capsule contacts this close to the analytical target pose.",
    )
    parser.add_argument(
        "--hand-clearance-mm",
        type=float,
        default=35.0,
        help="Only used by --collision-model point-inflated.",
    )
    parser.add_argument(
        "--max-linear-move-mm",
        type=float,
        default=320.0,
        help="Maximum allowed straight-line base motion from sampled pose to grasp pose.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def frange_mm(bounds_mm: Iterable[float], step_mm: float) -> np.ndarray:
    lo, hi = [float(v) for v in bounds_mm]
    if step_mm <= 0:
        raise ValueError("grid step must be positive")
    n = int(math.floor((hi - lo) / step_mm)) + 1
    return lo + step_mm * np.arange(n, dtype=float)


def solve_object_grasp(
    closure: ClosureGeometry,
    obj: ObjectSpec,
    mode: str,
    width_offset_m: float,
) -> tuple[ClosureResult | None, str]:
    internal_width_m = obj.grasp_width_m + width_offset_m
    try:
        if mode == "line":
            return closure.line(internal_width_m), "ok"
        if mode.startswith("plane"):
            return closure.plane(internal_width_m, n_fingers=int(mode[-1])), "ok"
    except Exception as exc:
        return None, f"solve_failed:{type(exc).__name__}:{exc}"
    return None, f"unsupported_mode:{mode}"


def path_ctrl_values(result: ClosureResult, shape: str) -> dict[str, float]:
    if shape == "closed":
        return dict(result.ctrl_values)
    if shape == "open":
        values = {name: 0.0 for name in ACTUATOR_NAMES}
        values["thumb_yaw"] = CTRL_MAX["thumb_yaw"]
        return values
    raise ValueError(f"Unsupported path hand shape: {shape}")


def tabletop_object_center(obj: ObjectSpec) -> np.ndarray:
    """Place the object AABB on the ground plane instead of centered on it."""

    return np.array([0.0, 0.0, obj.size_m[2] / 2.0], dtype=float)


def final_ignore_groups(mode: str) -> tuple[str, ...] | None:
    if mode == "all":
        return None
    if mode == "none":
        return ()
    if mode == "fingers":
        return ("thumb", "index", "middle", "ring", "pinky")
    raise ValueError(f"Unsupported final-ignore-groups mode: {mode}")


def build_path_capsules(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    result: ClosureResult,
    *,
    shape: str,
    radius_scale: float,
) -> list[Capsule]:
    set_closure_qpos(model, data, path_ctrl_values(result, shape))
    return build_capsule_proxy(model, data, radius_scale=radius_scale)


def path_crosses_inflated_object(
    start: np.ndarray,
    final: np.ndarray,
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    clearance_m: float,
    final_ignore_m: float,
    n_samples: int,
) -> bool:
    inflated = half_extents + clearance_m
    if n_samples < 2:
        raise ValueError("--path-samples must be >= 2")
    for t in np.linspace(0.0, 1.0, n_samples, endpoint=True):
        p = start + t * (final - start)
        if np.linalg.norm(p - final) <= final_ignore_m:
            continue
        if np.all(np.abs(p - aabb_center) <= inflated):
            return True
    return False


def evaluate_capsule_path(
    capsules_base: list[Capsule],
    *,
    start: np.ndarray,
    final: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    path_samples: int,
    final_ignore_m: float,
    final_ignore_group_mode: str,
) -> tuple[bool, float, str, int]:
    rows = sample_linear_path_collisions(
        capsules_base,
        start=start,
        final=final,
        rotation=rotation,
        half_extents=half_extents,
        aabb_center=aabb_center,
        path_samples=path_samples,
        final_ignore_m=final_ignore_m,
        final_ignore_groups=final_ignore_groups(final_ignore_group_mode),
    )
    active_collisions = [row for row in rows if bool(row["collision"])]
    usable = [row for row in rows if not bool(row["ignored_for_final_contact"])] or rows
    min_row = min(usable, key=lambda row: float(row["clearance_m"]))
    nearest = f"{min_row['nearest_group']}/{min_row['nearest_capsule']}"
    return (
        bool(active_collisions),
        float(min_row["clearance_m"]),
        nearest,
        max(int(row["active_collision_count"]) for row in rows),
    )


def evaluate_object(
    closure: ClosureGeometry,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obj: ObjectSpec,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    t0 = time.perf_counter()
    mode = args.mode_override or obj.mode
    width_offset_m = args.object_width_offset_mm / 1000.0
    result, solve_reason = solve_object_grasp(closure, obj, mode, width_offset_m)

    xs = frange_mm(args.x_range_mm, args.grid_step_mm) / 1000.0
    ys = frange_mm(args.y_range_mm, args.grid_step_mm) / 1000.0
    zs = frange_mm(args.z_range_mm, args.grid_step_mm) / 1000.0
    yaws = np.linspace(0.0, 2.0 * math.pi, args.yaw_samples, endpoint=False)

    half_extents = np.array(obj.size_m, dtype=float) / 2.0
    aabb_center = tabletop_object_center(obj)
    clearance_m = args.hand_clearance_mm / 1000.0
    final_ignore_m = args.final_contact_ignore_mm / 1000.0
    max_linear_move_m = args.max_linear_move_mm / 1000.0

    rows: list[dict[str, object]] = []
    capsules_base = (
        build_path_capsules(
            model,
            data,
            result,
            shape=args.path_hand_shape,
            radius_scale=args.radius_scale,
        )
        if result and args.collision_model == "capsule"
        else []
    )
    final_and_rotation_by_yaw = [
        (
            closure_base_position(result, yaw, object_center=aabb_center),
            closure_base_rotation(result, yaw),
            math.degrees(yaw),
        )
        for yaw in yaws
    ] if result else []

    for x in xs:
        for y in ys:
            for z in zs:
                start = np.array([x, y, z], dtype=float)
                viable = 0
                blockers = {
                    "solve_failed": 0,
                    "move_too_long": 0,
                    "path_collision": 0,
                }
                path_lengths: list[float] = []
                clearances: list[float] = []
                nearest_blockers: dict[str, int] = {}
                max_active_collision_count = 0
                for final, rotation, _yaw_deg in final_and_rotation_by_yaw:
                    path_len = float(np.linalg.norm(final - start))
                    path_lengths.append(path_len)
                    if path_len > max_linear_move_m:
                        blockers["move_too_long"] += 1
                        continue

                    if args.collision_model == "capsule":
                        collided, min_clearance_m, nearest, active_count = evaluate_capsule_path(
                            capsules_base,
                            start=start,
                            final=final,
                            rotation=rotation,
                            half_extents=half_extents,
                            aabb_center=aabb_center,
                            path_samples=args.path_samples,
                            final_ignore_m=final_ignore_m,
                            final_ignore_group_mode=args.final_ignore_groups,
                        )
                        clearances.append(min_clearance_m)
                        max_active_collision_count = max(max_active_collision_count, active_count)
                        if collided:
                            blockers["path_collision"] += 1
                            nearest_blockers[nearest] = nearest_blockers.get(nearest, 0) + 1
                            continue
                    elif path_crosses_inflated_object(
                        start,
                        final,
                        half_extents,
                        aabb_center,
                        clearance_m,
                        final_ignore_m,
                        args.path_samples,
                    ):
                        blockers["path_collision"] += 1
                        continue
                    viable += 1
                if result is None:
                    blockers["solve_failed"] = args.yaw_samples
                total = args.yaw_samples
                dominant_blocker = (
                    "ok"
                    if viable
                    else max(blockers.items(), key=lambda item: item[1])[0]
                )
                viability = viable / total if total else 0.0
                rows.append(
                    {
                        "object": obj.name,
                        "label": obj.label,
                        "mode": mode,
                        "x_mm": f"{x * 1000.0:.3f}",
                        "y_mm": f"{y * 1000.0:.3f}",
                        "z_mm": f"{z * 1000.0:.3f}",
                        "object_width_mm": f"{obj.grasp_width_m * 1000.0:.3f}",
                        "object_center_z_mm": f"{aabb_center[2] * 1000.0:.3f}",
                        "internal_width_mm": f"{(obj.grasp_width_m + width_offset_m) * 1000.0:.3f}",
                        "collision_model": args.collision_model,
                        "path_hand_shape": args.path_hand_shape if args.collision_model == "capsule" else "",
                        "viable_poses": viable,
                        "total_poses": total,
                        "viability": f"{viability:.6f}",
                        "move_too_long_count": blockers["move_too_long"],
                        "path_collision_count": blockers["path_collision"],
                        "max_active_collision_count": max_active_collision_count,
                        "best_clearance_mm": (
                            f"{max(clearances) * 1000.0:.3f}" if clearances else ""
                        ),
                        "mean_clearance_mm": (
                            f"{np.mean(clearances) * 1000.0:.3f}" if clearances else ""
                        ),
                        "most_common_blocker": (
                            max(nearest_blockers.items(), key=lambda item: item[1])[0]
                            if nearest_blockers else ""
                        ),
                        "mean_path_mm": (
                            f"{np.mean(path_lengths) * 1000.0:.3f}" if path_lengths else ""
                        ),
                        "dominant_blocker": dominant_blocker if result else solve_reason,
                    }
                )

    viabilities = np.array([float(row["viability"]) for row in rows], dtype=float)
    summary = {
        "object": obj.name,
        "label": obj.label,
        "mode": mode,
        "grid_points": len(rows),
        "yaw_samples": args.yaw_samples,
        "object_width_mm": f"{obj.grasp_width_m * 1000.0:.3f}",
        "object_center_z_mm": f"{aabb_center[2] * 1000.0:.3f}",
        "internal_width_mm": f"{(obj.grasp_width_m + width_offset_m) * 1000.0:.3f}",
        "collision_model": args.collision_model,
        "path_hand_shape": args.path_hand_shape if args.collision_model == "capsule" else "",
        "mean_viability": f"{float(np.mean(viabilities)):.6f}",
        "reachable_voxel_fraction": f"{float(np.mean(viabilities > 0.0)):.6f}",
        "no_go_voxel_fraction": f"{float(np.mean(viabilities <= 0.0)):.6f}",
        "solve_reason": solve_reason,
        "elapsed_s": f"{time.perf_counter() - t0:.3f}",
        "notes": obj.notes,
    }
    return rows, summary


def write_volume_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "object",
        "label",
        "mode",
        "x_mm",
        "y_mm",
        "z_mm",
        "object_width_mm",
        "object_center_z_mm",
        "internal_width_mm",
        "collision_model",
        "path_hand_shape",
        "viable_poses",
        "total_poses",
        "viability",
        "move_too_long_count",
        "path_collision_count",
        "max_active_collision_count",
        "best_clearance_mm",
        "mean_clearance_mm",
        "most_common_blocker",
        "mean_path_mm",
        "dominant_blocker",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "object",
        "label",
        "mode",
        "grid_points",
        "yaw_samples",
        "object_width_mm",
        "object_center_z_mm",
        "internal_width_mm",
        "collision_model",
        "path_hand_shape",
        "mean_viability",
        "reachable_voxel_fraction",
        "no_go_voxel_fraction",
        "solve_reason",
        "elapsed_s",
        "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_assumptions(path: Path, args: argparse.Namespace, objects: list[ObjectSpec]) -> None:
    payload = {
        "script": "tools/run_analytical_grasp_volume.py",
        "simulation_only": True,
        "uses_hardware": False,
        "model_status": (
            "first-pass analytical no-go volume proxy; not full MuJoCo collision, "
            "arm IK, or object-aware path planning"
        ),
        "objects": {
            obj.name: {
                "label": obj.label,
                "size_m": obj.size_m,
                "aabb_center_m": tabletop_object_center(obj).tolist(),
                "bottom_z_m": 0.0,
                "grasp_width_m": obj.grasp_width_m,
                "default_mode": obj.mode,
                "notes": obj.notes,
            }
            for obj in objects
        },
        "sampling": {
            "x_range_mm": args.x_range_mm,
            "y_range_mm": args.y_range_mm,
            "z_range_mm": args.z_range_mm,
            "grid_step_mm": args.grid_step_mm,
            "yaw_samples": args.yaw_samples,
            "path_samples": args.path_samples,
        },
        "assumptions": {
            "object_width_offset_mm": args.object_width_offset_mm,
            "collision_model": args.collision_model,
            "path_hand_shape": args.path_hand_shape,
            "radius_scale": args.radius_scale,
            "final_ignore_groups": args.final_ignore_groups,
            "hand_clearance_mm": args.hand_clearance_mm if args.collision_model == "point-inflated" else None,
            "final_contact_ignore_mm": args.final_contact_ignore_mm,
            "max_linear_move_mm": args.max_linear_move_mm,
            "object_model": "tabletop axis-aligned bounding box",
            "pose_samples": "yaw samples around object with analytical grasp midpoint at object center",
            "viability": (
                "fraction of yaw samples whose straight-line hand-base path to the "
                "analytical grasp pose is collision-free under the selected proxy"
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def maybe_write_plots(out_dir: Path, volume_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_skipped.txt").write_text(
            f"matplotlib unavailable; plots skipped: {exc}\n"
        )
        return

    by_object: dict[str, list[dict[str, object]]] = {}
    for row in volume_rows:
        by_object.setdefault(str(row["object"]), []).append(row)

    for object_name, rows in by_object.items():
        xs = sorted({float(row["x_mm"]) for row in rows})
        ys = sorted({float(row["y_mm"]) for row in rows})
        zs = sorted({float(row["z_mm"]) for row in rows})
        lookup = {
            (float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])): float(row["viability"])
            for row in rows
        }

        slice_indices = sorted({0, len(zs) // 2, len(zs) - 1})
        slice_zs = [zs[index] for index in slice_indices]
        fig, axes = plt.subplots(
            1,
            len(slice_zs),
            figsize=(4.2 * len(slice_zs), 3.8),
            sharex=True,
            sharey=True,
        )
        if len(slice_zs) == 1:
            axes = [axes]
        for ax, z in zip(axes, slice_zs):
            grid = np.array(
                [[lookup.get((x, y, z), np.nan) for x in xs] for y in ys],
                dtype=float,
            )
            im = ax.imshow(
                grid,
                origin="lower",
                extent=[min(xs), max(xs), min(ys), max(ys)],
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                aspect="equal",
            )
            ax.set_title(f"z={z:.0f} mm")
            ax.set_xlabel("hand-base x [mm]")
            ax.grid(False)
        axes[0].set_ylabel("hand-base y [mm]")
        fig.suptitle(f"Analytical linear-approach viability slices: {object_name}", fontsize=11)
        fig.subplots_adjust(left=0.07, right=0.84, bottom=0.16, top=0.84, wspace=0.18)
        cax = fig.add_axes([0.88, 0.20, 0.018, 0.60])
        fig.colorbar(im, cax=cax, label="viable yaw fraction")
        fig.savefig(out_dir / f"{object_name}_viability_slices.png", dpi=150)
        plt.close(fig)

        x = np.array([float(row["x_mm"]) for row in rows])
        y = np.array([float(row["y_mm"]) for row in rows])
        z = np.array([float(row["z_mm"]) for row in rows])
        v = np.array([float(row["viability"]) for row in rows])
        visible = v > 0.0
        fig = plt.figure(figsize=(7, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        if np.any(visible):
            sc = ax.scatter(
                x[visible],
                y[visible],
                z[visible],
                c=v[visible],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=24,
                alpha=0.85,
            )
            fig.colorbar(sc, ax=ax, label="viable yaw fraction", shrink=0.72)
        ax.scatter(x[~visible], y[~visible], z[~visible], c="#d8d8d8", s=8, alpha=0.18)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.set_title(f"No-go volume proxy: {object_name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{object_name}_no_go_volume.png", dpi=150)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.yaw_samples <= 0:
        raise ValueError("--yaw-samples must be positive")
    if args.grid_step_mm <= 0:
        raise ValueError("--grid-step-mm must be positive")

    fk = InspireHandFK(xml_path=args.xml, rebuild=args.rebuild_fk) if args.xml else InspireHandFK(rebuild=args.rebuild_fk)
    closure = ClosureGeometry(fk)
    objects = [BUILTIN_OBJECTS[name] for name in args.objects]

    print("RH56 analytical grasp volume assumptions:")
    print("  simulation_only: true")
    print("  hardware_required: false")
    print(f"  collision_model: {args.collision_model}")
    if args.collision_model == "capsule":
        print(f"  path_hand_shape: {args.path_hand_shape}")
    print(f"  objects: {', '.join(args.objects)}")
    print(f"  output: {args.out}")

    all_volume_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for obj in objects:
        model = mujoco.MjModel.from_xml_path(str(args.xml or fk.xml_path))
        data = mujoco.MjData(model)
        rows, summary = evaluate_object(closure, model, data, obj, args)
        all_volume_rows.extend(rows)
        summary_rows.append(summary)
        print(
            f"  {obj.name}: mean_viability={summary['mean_viability']} "
            f"no_go_fraction={summary['no_go_voxel_fraction']} "
            f"elapsed_s={summary['elapsed_s']}"
        )

    write_volume_csv(args.out / "volume.csv", all_volume_rows)
    write_summary_csv(args.out / "summary.csv", summary_rows)
    write_assumptions(args.out / "assumptions.json", args, objects)
    if not args.no_plots:
        maybe_write_plots(args.out, all_volume_rows)

    print(f"Wrote {args.out / 'summary.csv'}")
    print(f"Wrote {args.out / 'volume.csv'} ({len(all_volume_rows)} voxels)")
    print(f"Wrote {args.out / 'assumptions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
