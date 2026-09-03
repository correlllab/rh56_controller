#!/usr/bin/env python3
"""Render grid start-to-target paths from a strategy pre-grasp sweep.

This is a visual verifier for tools/run_strategy_pregrasp_rate.py.  It reads
the generated volume.csv, then renders the exact start and target poses used by
that sweep, one path at a time, so the sampling geometry can be checked by eye.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rh56_controller.capsule_hand_proxy import closure_base_rotation, set_closure_qpos, transform_capsules
from rh56_controller.grasp_geometry import ClosureGeometry, InspireHandFK
from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS
from tools.demo_strategy_pregrasp_collision import (
    STRATEGIES,
    build_capsules_for_ctrl,
    pregrasp_width_policy,
    solve_mode,
    strategy_poses,
    tabletop_object_center,
)
from tools.render_strategy_pregrasp_video import (
    COLORS,
    add_capsule,
    add_object_proxy,
    add_sphere,
    add_text,
    current_collisions,
    make_camera,
    save_gif,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render sampled RH56 pre-grasp grid paths from volume.csv."
    )
    parser.add_argument(
        "--volume",
        type=Path,
        default=Path("artifacts/current/paper_bottle/volume.csv"),
        help="volume.csv produced by tools/run_strategy_pregrasp_rate.py.",
    )
    parser.add_argument(
        "--assumptions",
        type=Path,
        default=None,
        help="assumptions.json from the same sweep. Defaults to volume parent.",
    )
    parser.add_argument(
        "--strategy",
        choices=["all", *STRATEGIES],
        default="all",
        help="Strategy to render. all writes one GIF per strategy.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/current/paper_bottle/rendered_paths"),
    )
    parser.add_argument("--xml", default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--rebuild-fk", action="store_true", help="Rebuild FK cache.")
    parser.add_argument("--path-samples", type=int, default=10)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--floor-tolerance-mm", type=float, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--frames-per-path", type=int, default=18)
    parser.add_argument("--hold-frames", type=int, default=4)
    parser.add_argument("--azimuth", type=float, default=-48.0)
    parser.add_argument("--elevation", type=float, default=-24.0)
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument(
        "--show-capsules",
        action="store_true",
        help="Overlay the current capsule proxy while the hand moves.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def float_field(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def vec_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array(
        [
            float(row[f"{prefix}_x_mm"]),
            float(row[f"{prefix}_y_mm"]),
            float(row[f"{prefix}_z_mm"]),
        ],
        dtype=float,
    ) / 1000.0


def target_center_from_row(row: dict[str, str], object_center: np.ndarray) -> np.ndarray:
    if "target_center_x_mm" not in row or row.get("target_center_x_mm", "") == "":
        return object_center
    return vec_from_row(row, "target_center")


def sort_key(row: dict[str, str]) -> tuple[float, float, float, float]:
    point_order = row.get("point_order", "")
    if point_order:
        return (
            float(point_order),
            float_field(row, "point_d_mm"),
            float_field(row, "point_h_mm"),
            float_field(row, "yaw_deg"),
        )
    return (
        float_field(row, "approach_distance_mm"),
        float_field(row, "grid_lateral_mm", float_field(row, "offset_x_mm")),
        float_field(row, "grid_height_mm", float_field(row, "offset_z_mm")),
        float_field(row, "yaw_deg"),
    )


def selected_strategy_names(rows: list[dict[str, str]], name: str) -> list[str]:
    names = sorted({row["strategy"] for row in rows})
    if name == "all":
        return [strategy for strategy in STRATEGIES if strategy in names]
    return [name]


def iter_frames_for_strategy(
    *,
    model: mujoco.MjModel,
    strategy,
    rows: list[dict[str, str]],
    capsules_base,
    half_extents: np.ndarray,
    object_center: np.ndarray,
    object_shape: str,
    floor_z_m: float,
    floor_tolerance_m: float,
    args: argparse.Namespace,
    display_name: str | None = None,
) -> Iterator[np.ndarray]:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    starts = [vec_from_row(row, "start") for row in rows]
    targets = [vec_from_row(row, "target") for row in rows]
    grasp_starts = [vec_from_row(row, "start_center") for row in rows]
    grasp_targets = [target_center_from_row(row, object_center) for row in rows]
    points = [object_center, object_center + half_extents, object_center - half_extents]
    points.extend(starts)
    points.extend(targets)
    points.extend(grasp_starts)
    points.extend(grasp_targets)
    camera = make_camera(points, args)

    color = COLORS.get(strategy.name, (0.2, 0.2, 0.2, 1.0))
    total_paths = len(rows)
    total_path_frames = args.frames_per_path + args.hold_frames

    try:
        for path_idx, row in enumerate(rows, start=1):
            start = vec_from_row(row, "start")
            target = vec_from_row(row, "target")
            grasp_start = vec_from_row(row, "start_center")
            grasp_target = target_center_from_row(row, object_center)
            yaw_rad = math.radians(float_field(row, "yaw_deg"))
            rotation = closure_base_rotation(strategy.target_result, yaw_rad)
            valid = int(row["valid"])
            blocker = row["dominant_blocker"]

            for frame_idx in range(total_path_frames):
                if args.frames_per_path <= 1:
                    alpha = 1.0
                elif frame_idx >= args.frames_per_path:
                    alpha = 1.0
                else:
                    alpha = float(frame_idx) / float(args.frames_per_path - 1)
                base = (1.0 - alpha) * start + alpha * target
                grasp_center = (1.0 - alpha) * grasp_start + alpha * grasp_target

                set_closure_qpos(
                    model,
                    data,
                    strategy.ctrl_values,
                    base_position=base,
                    base_rotation=rotation,
                )
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                scn = renderer.scene

                add_object_proxy(
                    scn,
                    object_center,
                    half_extents,
                    object_shape,
                    (0.0, 0.8, 0.25, 0.30),
                )
                add_sphere(scn, grasp_center, 0.006, (*color[:3], 0.95))
                for marker_idx, marker_start in enumerate(grasp_starts, start=1):
                    marker_rgba = (0.06, 0.18, 0.95, 0.90) if marker_idx == path_idx else (0.1, 0.1, 0.1, 0.45)
                    add_sphere(scn, marker_start, 0.005, marker_rgba)
                for marker_idx, marker_target in enumerate(grasp_targets, start=1):
                    marker_rgba = (0.0, 0.0, 0.0, 0.95) if marker_idx == path_idx else (0.0, 0.0, 0.0, 0.35)
                    add_sphere(scn, marker_target, 0.004, marker_rgba)
                for marker_idx, (path_start, path_target) in enumerate(zip(grasp_starts, grasp_targets), start=1):
                    path_rgba = (*color[:3], 0.90) if marker_idx == path_idx else (0.05, 0.05, 0.05, 0.18)
                    path_radius = 0.0014 if marker_idx == path_idx else 0.0007
                    add_capsule(scn, path_start, path_target, path_radius, path_rgba)

                current_status = "not checked"
                if args.show_capsules:
                    capsules = transform_capsules(capsules_base, rotation, base)
                    collided, object_clearance, floor_raw_clearance = current_collisions(
                        capsules,
                        half_extents=half_extents,
                        aabb_center=object_center,
                        object_shape=object_shape,
                        floor_z_m=floor_z_m,
                        floor_tolerance_m=floor_tolerance_m,
                    )
                    current_status = (
                        f"{'collision' if collided else 'clear'} | "
                        f"obj {object_clearance * 1000.0:.1f} mm | "
                        f"floor {floor_raw_clearance * 1000.0:.1f} mm"
                    )
                    rgba = (0.95, 0.04, 0.02, 0.64) if collided else (0.05, 0.72, 1.0, 0.36)
                    for capsule in capsules:
                        add_capsule(scn, capsule.p0, capsule.p1, capsule.radius, rgba)

                frame = renderer.render()
                lateral = row.get("grid_lateral_mm", "")
                height = row.get("grid_height_mm", "")
                distance = row.get("approach_distance_mm", "")
                point_id = row.get("point_id", "")
                point_d = row.get("point_d_mm", "")
                point_h = row.get("point_h_mm", "")
                status = "valid" if valid else f"blocked: {blocker}"
                if point_id:
                    point_line = (
                        f"{point_id} | approach {distance} mm | "
                        f"d {point_d} mm | h {point_h} mm"
                    )
                else:
                    point_line = f"lateral {lateral or 'n/a'} mm | height {height or 'n/a'} mm | distance {distance or 'n/a'} mm"
                title = f"{strategy.name} | path {path_idx}/{total_paths} | {status}"
                if display_name:
                    title = f"{display_name} | {title}"
                frame = add_text(
                    frame,
                    [
                        title,
                        f"alpha {alpha:.2f} | {point_line}",
                        current_status,
                    ],
                )
                yield frame
    finally:
        renderer.close()


def render_rows_for_strategy(
    **kwargs,
) -> list[np.ndarray]:
    return list(iter_frames_for_strategy(**kwargs))


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "strategy",
        "gif_path",
        "paths_rendered",
        "valid_paths",
        "invalid_paths",
        "fps",
        "frames_per_path",
        "hold_frames",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assumptions_path = args.assumptions or (args.volume.parent / "assumptions.json")
    assumptions = json.loads(assumptions_path.read_text())
    rows = load_rows(args.volume)
    if not rows:
        raise ValueError(f"No rows found in {args.volume}")

    obj = BUILTIN_OBJECTS[assumptions["object"]["name"]]
    mode = str(assumptions["mode"])
    target_width_m = float(assumptions["target_width_m"])
    preopen_m = float(assumptions.get("preopen_mm", 0.0)) / 1000.0
    iterative_width_raw = assumptions.get("iterative_width_mm")
    iterative_width_m = (
        float(iterative_width_raw) / 1000.0
        if iterative_width_raw is not None
        else None
    )
    floor_z_m = float(assumptions.get("floor_z_m", 0.0))
    floor_tolerance_m = (
        float(args.floor_tolerance_mm) / 1000.0
        if args.floor_tolerance_mm is not None
        else float(assumptions.get("floor_tolerance_mm", 3.0)) / 1000.0
    )
    grasp_center_policy = str(assumptions.get("grasp_center_policy", "antipodal"))
    iterative_pregrasp_policy = str(
        assumptions.get("iterative_pregrasp_policy", "planner-max-width")
    )

    fk = (
        InspireHandFK(xml_path=args.xml, rebuild=args.rebuild_fk)
        if args.xml
        else InspireHandFK(rebuild=args.rebuild_fk)
    )
    closure = ClosureGeometry(fk)
    final_result = solve_mode(closure, mode, target_width_m)
    strategies = strategy_poses(
        fk,
        closure,
        mode=mode,
        final_result=final_result,
        target_width_m=target_width_m,
        preopen_m=preopen_m,
        iterative_width_m=iterative_width_m,
        iterative_pregrasp_policy=iterative_pregrasp_policy,
    )
    strategy_by_name = {strategy.name: strategy for strategy in strategies}

    model = mujoco.MjModel.from_xml_path(str(args.xml or fk.xml_path))
    data = mujoco.MjData(model)
    half_extents = np.asarray(obj.size_m, dtype=float) / 2.0
    object_center = tabletop_object_center(obj)

    summary_rows: list[dict[str, object]] = []
    for strategy_name in selected_strategy_names(rows, args.strategy):
        strategy = strategy_by_name[strategy_name]
        strategy_rows = sorted(
            [row for row in rows if row["strategy"] == strategy_name],
            key=sort_key,
        )
        if args.max_paths is not None:
            strategy_rows = strategy_rows[: args.max_paths]
        capsules_base = build_capsules_for_ctrl(
            model,
            data,
            strategy.ctrl_values,
            radius_scale=args.radius_scale,
        )
        frames = render_rows_for_strategy(
            model=model,
            strategy=strategy,
            rows=strategy_rows,
            capsules_base=capsules_base,
            half_extents=half_extents,
            object_center=object_center,
            object_shape=obj.collision_shape,
            floor_z_m=floor_z_m,
            floor_tolerance_m=floor_tolerance_m,
            args=args,
        )
        gif_path = args.out / f"{strategy_name}_grid_paths.gif"
        save_gif(gif_path, frames, args.fps)
        valid_count = sum(int(row["valid"]) for row in strategy_rows)
        summary_rows.append(
            {
                "strategy": strategy_name,
                "gif_path": str(gif_path),
                "paths_rendered": len(strategy_rows),
                "valid_paths": valid_count,
                "invalid_paths": len(strategy_rows) - valid_count,
                "fps": args.fps,
                "frames_per_path": args.frames_per_path,
                "hold_frames": args.hold_frames,
            }
        )

    write_summary(args.out / "summary.csv", summary_rows)
    render_assumptions = {
        "script": "tools/render_strategy_grid_paths.py",
        "simulation_only": True,
        "uses_hardware": False,
        "source_volume": str(args.volume),
        "source_assumptions": str(assumptions_path),
        "purpose": (
            "Visual verification of sampled grasp-center start points and "
            "straight-line moves from each sampled grasp center to the fixed grasp target. "
            "The hand mesh is still animated by wrist/base position control."
        ),
        "visual_reference": "grasp_center",
        "motion_control_reference": "wrist_base",
        "object_collision_shape": obj.collision_shape,
        "grasp_center_policy": grasp_center_policy,
        "iterative_pregrasp_policy": iterative_pregrasp_policy,
        "show_capsules": args.show_capsules,
        "rendering": {
            "format": "gif",
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "frames_per_path": args.frames_per_path,
            "hold_frames": args.hold_frames,
            "mujoCo_gl": os.environ.get("MUJOCO_GL"),
        },
    }
    (args.out / "assumptions.json").write_text(json.dumps(render_assumptions, indent=2) + "\n")

    print("RH56 strategy grid path render:")
    print("  simulation_only: true")
    print("  hardware_required: false")
    print(f"  volume: {args.volume}")
    for row in summary_rows:
        print(
            f"  {row['strategy']}: paths={row['paths_rendered']} "
            f"valid={row['valid_paths']} gif={row['gif_path']}"
        )
    print(f"Wrote {args.out / 'summary.csv'}")
    print(f"Wrote {args.out / 'assumptions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
