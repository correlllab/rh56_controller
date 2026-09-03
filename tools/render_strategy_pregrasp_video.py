#!/usr/bin/env python3
"""Render MuJoCo videos for strategy-specific RH56 pre-grasp paths.

This is a visualization/debug companion for the paper-v2 no-go-volume tools.
It renders the exact linear hand-base motion checked by
tools/demo_strategy_pregrasp_collision.py and overlays the capsule proxy used
for collision testing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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

from rh56_controller.capsule_hand_proxy import (
    capsule_collisions_against_object,
    closure_base_rotation,
    set_closure_qpos,
    transform_capsules,
)
from rh56_controller.grasp_geometry import ClosureGeometry, InspireHandFK
from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS
from tools.demo_strategy_pregrasp_collision import (
    STRATEGIES,
    build_capsules_for_ctrl,
    choose_start,
    closure_base_position_for_center,
    evaluate_strategy,
    floor_clearance,
    pregrasp_width_policy,
    solve_mode,
    strategy_poses,
    tabletop_grasp_target,
    tabletop_object_center,
)


COLORS = {
    "naive": (0.85, 0.32, 0.04, 1.0),
    "iterative_closure": (0.00, 0.45, 0.70, 1.0),
    "thumb_reflex": (0.00, 0.62, 0.42, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render MuJoCo GIFs for RH56 strategy pre-grasp collision paths."
        )
    )
    parser.add_argument("--object", choices=sorted(BUILTIN_OBJECTS), default="debug_40mm_cube")
    parser.add_argument(
        "--mode",
        choices=["object-default", "line", "plane3", "plane4", "plane5", "cylinder"],
        default="line",
        help="Analytical final grasp mode. Use object-default to use the object metadata.",
    )
    parser.add_argument(
        "--strategy",
        choices=["all", *STRATEGIES],
        default="all",
        help="Strategy to render. all writes one GIF per strategy plus a combined GIF.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/scratch/strategy_pregrasp_video"),
        help="Output artifact directory.",
    )
    parser.add_argument("--xml", default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--rebuild-fk", action="store_true", help="Rebuild FK cache.")
    parser.add_argument("--object-width-offset-mm", type=float, default=20.0)
    parser.add_argument(
        "--grasp-center-policy",
        choices=["antipodal", "contact-centroid"],
        default="antipodal",
    )
    parser.add_argument(
        "--preopen-mm",
        type=float,
        default=5.0,
        help=(
            "Width margin added to the final grasp width for iterative_closure "
            "when --iterative-pregrasp-policy=final-plus-preopen."
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
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--start-mm",
        type=float,
        nargs=3,
        default=None,
        help="Manual hand-base start position. If omitted, a demo start is selected.",
    )
    parser.add_argument("--path-samples", type=int, default=16)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--floor-z-mm", type=float, default=0.0)
    parser.add_argument("--floor-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=640, help="Per-strategy video width.")
    parser.add_argument("--height", type=int, default=480, help="Per-strategy video height.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=72, help="Moving frames per strategy.")
    parser.add_argument("--hold-frames", type=int, default=18, help="Extra frames at target.")
    parser.add_argument("--azimuth", type=float, default=-55.0)
    parser.add_argument("--elevation", type=float, default=-23.0)
    parser.add_argument(
        "--no-capsules",
        action="store_true",
        help="Render only the MuJoCo hand mesh and object, without capsule overlay.",
    )
    return parser.parse_args()


def add_box(scn, center: np.ndarray, half_extents: np.ndarray, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(half_extents, dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def add_object_proxy(
    scn,
    center: np.ndarray,
    half_extents: np.ndarray,
    object_shape: str,
    rgba,
) -> None:
    """Add a supported object proxy to a MuJoCo render scene."""

    if object_shape == "box":
        add_box(scn, center, half_extents, rgba)
        return
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    if object_shape == "cylinder":
        size = np.array([half_extents[0], half_extents[2], 0.0], dtype=np.float64)
        geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
    elif object_shape == "sphere":
        size = np.repeat(float(half_extents[0]), 3).astype(np.float64)
        geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
    else:
        raise ValueError(f"Unsupported object collision shape: {object_shape}")
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        size,
        np.asarray(center, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def add_sphere(scn, point: np.ndarray, radius: float, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(point, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def add_capsule(scn, p0: np.ndarray, p1: np.ndarray, radius: float, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scn.ngeom += 1


def make_camera(points: list[np.ndarray], args: argparse.Namespace) -> mujoco.MjvCamera:
    pts = np.vstack(points)
    center = pts.mean(axis=0)
    span = float((pts.max(axis=0) - pts.min(axis=0)).max())

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.lookat[2] = max(0.045, float(center[2]))
    cam.distance = max(0.26, span * 2.35)
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    return cam


def selected_strategies(strategies, name: str):
    if name == "all":
        return strategies
    return [strategy for strategy in strategies if strategy.name == name]


def current_collisions(
    capsules,
    *,
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    object_shape: str,
    floor_z_m: float,
    floor_tolerance_m: float,
) -> tuple[bool, float, float]:
    object_records = capsule_collisions_against_object(
        capsules,
        half_extents,
        object_center=aabb_center,
        object_shape=object_shape,
    )
    object_collision = any(record.intersects for record in object_records)
    object_clearance = min(float(record.clearance) for record in object_records)
    floor_raw_clearance, _ = floor_clearance(capsules, floor_z_m)
    floor_collision = floor_raw_clearance < -floor_tolerance_m
    return object_collision or floor_collision, object_clearance, floor_raw_clearance


def add_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    margin = 10
    line_height = 15
    box_height = margin * 2 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0, 130))
    for idx, line in enumerate(lines):
        draw.text((margin, margin + idx * line_height), line, fill=(255, 255, 255, 255))
    return np.asarray(image)


def save_gif(path: Path, frames: list[np.ndarray], fps: int) -> None:
    from PIL import Image

    images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    duration_ms = max(1, int(round(1000.0 / float(fps))))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def render_strategy(
    *,
    model: mujoco.MjModel,
    strategy,
    evaluation,
    capsules_base,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    aabb_center: np.ndarray,
    object_shape: str,
    floor_z_m: float,
    floor_tolerance_m: float,
    camera: mujoco.MjvCamera,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames: list[np.ndarray] = []
    total_frames = args.frames + args.hold_frames

    for frame_idx in range(total_frames):
        if args.frames <= 1:
            alpha = 1.0
        elif frame_idx >= args.frames:
            alpha = 1.0
        else:
            alpha = float(frame_idx) / float(args.frames - 1)
        base = (1.0 - alpha) * evaluation.start + alpha * evaluation.target

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
        color = COLORS.get(strategy.name, (0.2, 0.2, 0.2, 1.0))
        add_object_proxy(
            scn,
            aabb_center,
            half_extents,
            object_shape,
            (0.0, 0.8, 0.25, 0.30),
        )
        add_sphere(scn, evaluation.start, 0.006, (*color[:3], 0.95))
        add_sphere(scn, evaluation.target, 0.006, (0.02, 0.02, 0.02, 0.95))
        add_capsule(scn, evaluation.start, evaluation.target, 0.0012, (*color[:3], 0.65))

        capsules = transform_capsules(capsules_base, rotation, base)
        collided, object_clearance, floor_raw_clearance = current_collisions(
            capsules,
            half_extents=half_extents,
            aabb_center=aabb_center,
            object_shape=object_shape,
            floor_z_m=floor_z_m,
            floor_tolerance_m=floor_tolerance_m,
        )
        if not args.no_capsules:
            rgba = (0.95, 0.04, 0.02, 0.66) if collided else (0.05, 0.72, 1.0, 0.42)
            for capsule in capsules:
                add_capsule(scn, capsule.p0, capsule.p1, capsule.radius, rgba)

        frame = renderer.render()
        status = "valid" if evaluation.valid else f"blocked: {evaluation.dominant_blocker}"
        current = "collision" if collided else "clear"
        frame = add_text(
            frame,
            [
                f"{strategy.name} | {status}",
                f"path alpha {alpha:.2f} | current {current}",
                f"object clearance {object_clearance * 1000.0:.1f} mm | floor clearance {floor_raw_clearance * 1000.0:.1f} mm",
            ],
        )
        frames.append(frame)

    renderer.close()
    return frames


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "strategy",
        "gif_path",
        "valid",
        "dominant_blocker",
        "start_x_mm",
        "start_y_mm",
        "start_z_mm",
        "target_x_mm",
        "target_y_mm",
        "target_z_mm",
        "object_path_collision",
        "object_target_collision",
        "floor_path_collision",
        "floor_target_collision",
        "min_object_clearance_mm",
        "min_floor_clearance_mm",
        "nearest_blocker",
        "frames",
        "hold_frames",
        "fps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_assumptions(
    path: Path,
    *,
    args: argparse.Namespace,
    obj,
    mode: str,
    target_width_m: float,
    start: np.ndarray,
    final_grasp: np.ndarray,
    target_by_strategy: dict[str, np.ndarray],
) -> None:
    payload = {
        "script": "tools/render_strategy_pregrasp_video.py",
        "simulation_only": True,
        "uses_hardware": False,
        "purpose": (
            "Render the MuJoCo hand mesh and the capsule collision proxy for "
            "the same linear pre-grasp path used by the strategy collision demo."
        ),
        "object": {
            "name": obj.name,
            "label": obj.label,
            "size_m": obj.size_m,
            "grasp_width_m": obj.grasp_width_m,
            "collision_shape": obj.collision_shape,
            "aabb_center_m": tabletop_object_center(obj).tolist(),
            "grasp_target_fraction": obj.grasp_target_fraction,
            "grasp_target_top_offset_m": obj.grasp_target_top_offset_m,
            "grasp_target_m": tabletop_grasp_target(obj).tolist(),
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
        "yaw_deg": args.yaw_deg,
        "start_m": start.tolist(),
        "final_grasp_base_m": final_grasp.tolist(),
        "target_by_strategy_m": {
            name: value.tolist() for name, value in target_by_strategy.items()
        },
        "path_samples": args.path_samples,
        "floor_z_m": args.floor_z_mm / 1000.0,
        "floor_tolerance_mm": args.floor_tolerance_mm,
        "radius_scale": args.radius_scale,
        "rendering": {
            "format": "gif",
            "per_strategy_width": args.width,
            "per_strategy_height": args.height,
            "fps": args.fps,
            "frames": args.frames,
            "hold_frames": args.hold_frames,
            "mujoCo_gl": os.environ.get("MUJOCO_GL"),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.width <= 0 or args.height <= 0:
        raise ValueError("render width and height must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.frames <= 0 or args.hold_frames < 0:
        raise ValueError("frames must be positive and hold-frames cannot be negative")

    obj = BUILTIN_OBJECTS[args.object]
    mode = obj.mode if args.mode == "object-default" else args.mode
    target_width_m = obj.grasp_width_m + args.object_width_offset_mm / 1000.0
    preopen_m = args.preopen_mm / 1000.0
    iterative_width_m = (
        args.iterative_width_mm / 1000.0
        if args.iterative_width_mm is not None
        else None
    )
    floor_z_m = args.floor_z_mm / 1000.0
    floor_tolerance_m = args.floor_tolerance_mm / 1000.0
    yaw_rad = math.radians(args.yaw_deg)

    fk = (
        InspireHandFK(xml_path=args.xml, rebuild=args.rebuild_fk)
        if args.xml
        else InspireHandFK(rebuild=args.rebuild_fk)
    )
    closure = ClosureGeometry(fk)
    final_result = solve_mode(closure, mode, target_width_m)
    aabb_center = tabletop_object_center(obj)
    grasp_target = tabletop_grasp_target(obj)
    half_extents = np.asarray(obj.size_m, dtype=float) / 2.0
    final_grasp = closure_base_position_for_center(
        final_result,
        yaw_rad,
        object_center=grasp_target,
        policy=args.grasp_center_policy,
    )

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
    strategies = selected_strategies(strategies, args.strategy)

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

    if args.start_mm is None:
        start, evals = choose_start(
            strategies,
            capsules_by_strategy,
            final_grasp=final_grasp,
            target_by_strategy=target_by_strategy,
            rotation_by_strategy=rotation_by_strategy,
            half_extents=half_extents,
            aabb_center=aabb_center,
            object_shape=obj.collision_shape,
            floor_z_m=floor_z_m,
            floor_tolerance_m=floor_tolerance_m,
            path_samples=args.path_samples,
        )
    else:
        start = np.asarray(args.start_mm, dtype=float) / 1000.0
        evals = [
            evaluate_strategy(
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
            for strategy in strategies
        ]

    camera_points = [aabb_center, grasp_target, final_grasp, start]
    camera_points.extend(target_by_strategy.values())
    camera = make_camera(camera_points, args)

    frames_by_strategy: dict[str, list[np.ndarray]] = {}
    summary_rows: list[dict[str, object]] = []
    for evaluation in evals:
        strategy = evaluation.strategy
        frames = render_strategy(
            model=model,
            strategy=strategy,
            evaluation=evaluation,
            capsules_base=capsules_by_strategy[strategy.name],
            rotation=rotation_by_strategy[strategy.name],
            half_extents=half_extents,
            aabb_center=aabb_center,
            object_shape=obj.collision_shape,
            floor_z_m=floor_z_m,
            floor_tolerance_m=floor_tolerance_m,
            camera=camera,
            args=args,
        )
        frames_by_strategy[strategy.name] = frames
        gif_path = args.out / f"{strategy.name}.gif"
        save_gif(gif_path, frames, args.fps)
        summary_rows.append(
            {
                "strategy": strategy.name,
                "gif_path": str(gif_path),
                "valid": int(evaluation.valid),
                "dominant_blocker": evaluation.dominant_blocker,
                "start_x_mm": f"{evaluation.start[0] * 1000.0:.3f}",
                "start_y_mm": f"{evaluation.start[1] * 1000.0:.3f}",
                "start_z_mm": f"{evaluation.start[2] * 1000.0:.3f}",
                "target_x_mm": f"{evaluation.target[0] * 1000.0:.3f}",
                "target_y_mm": f"{evaluation.target[1] * 1000.0:.3f}",
                "target_z_mm": f"{evaluation.target[2] * 1000.0:.3f}",
                "object_path_collision": int(evaluation.object_path_collision),
                "object_target_collision": int(evaluation.object_target_collision),
                "floor_path_collision": int(evaluation.floor_path_collision),
                "floor_target_collision": int(evaluation.floor_target_collision),
                "min_object_clearance_mm": f"{evaluation.min_object_clearance_m * 1000.0:.3f}",
                "min_floor_clearance_mm": f"{evaluation.min_floor_clearance_m * 1000.0:.3f}",
                "nearest_blocker": evaluation.nearest_object,
                "frames": args.frames,
                "hold_frames": args.hold_frames,
                "fps": args.fps,
            }
        )

    if len(frames_by_strategy) > 1:
        ordered_frames = [frames_by_strategy[e.strategy.name] for e in evals]
        combined = [
            np.concatenate([frames[i] for frames in ordered_frames], axis=1)
            for i in range(len(ordered_frames[0]))
        ]
        combined_path = args.out / "all_strategies.gif"
        save_gif(combined_path, combined, args.fps)

    write_summary(args.out / "summary.csv", summary_rows)
    write_assumptions(
        args.out / "assumptions.json",
        args=args,
        obj=obj,
        mode=mode,
        target_width_m=target_width_m,
        start=start,
        final_grasp=final_grasp,
        target_by_strategy=target_by_strategy,
    )

    print("RH56 strategy pre-grasp video render:")
    print("  simulation_only: true")
    print("  hardware_required: false")
    print(f"  object: {obj.name}")
    print(f"  mode: {mode}")
    print(f"  strategy: {args.strategy}")
    print(f"  start_mm: {[round(v, 3) for v in (start * 1000.0).tolist()]}")
    for row in summary_rows:
        print(
            f"  {row['strategy']}: valid={row['valid']} "
            f"dominant={row['dominant_blocker']} gif={row['gif_path']}"
        )
    if len(frames_by_strategy) > 1:
        print(f"Wrote {args.out / 'all_strategies.gif'}")
    print(f"Wrote {args.out / 'summary.csv'}")
    print(f"Wrote {args.out / 'assumptions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
