#!/usr/bin/env python3
"""Render GraspGen-X grasp poses with the shared RH56 MuJoCo executor.

GraspGen-X selects a 6-D grasp pose.  The video labels that model output
separately from the shared approach, joint interpolation, contact-force,
lift, and hold executor so the learned grasp quality is not confused with
closed-loop hand control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rh56_controller.graspgenx_baseline import load_isaac_grasp_yaml  # noqa: E402
from rh56_controller.paper_v2_objects import (  # noqa: E402
    BUILTIN_OBJECTS,
    tabletop_aabb_center,
)
from tools.run_paper_15_object_grasp_success import (  # noqa: E402
    PAPER_OBJECT_MASS_KG,
    _add_object_model,
    _contact_side_normal_forces,
    _execute_trial,
    _object_position,
    _select_candidate,
    _validate_args,
    build_grid_points,
    build_run_config,
    parse_args as parse_benchmark_args,
)
from tools.run_strategy_pregrasp_paper_batch import PAPER_OBJECTS  # noqa: E402


DEFAULT_OBJECTS = [
    "paper_bottle",
    "paper_can",
    "paper_sugar_box",
    "paper_egg",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render pretrained GraspGen-X pose selection plus the shared RH56 "
            "approach/force/lift executor for paper-object proxies."
        )
    )
    parser.add_argument("--objects", nargs="+", choices=PAPER_OBJECTS, default=DEFAULT_OBJECTS)
    parser.add_argument("--point", default="P2")
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("artifacts/paper_15_object_grasp_success/graspgenx_candidates"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/paper_15_object_grasp_success/graspgenx_video"),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-22.0)
    parser.add_argument("--camera-distance", type=float, default=0.62)
    parser.add_argument(
        "--fallback-rank",
        type=int,
        default=None,
        help=(
            "Diagnostic only: when no collision-free candidate exists, render "
            "this rejected confidence rank and force the outcome to FAIL."
        ),
    )
    parser.add_argument(
        "--no-contact-visualization",
        action="store_true",
        help="Disable MuJoCo contact-point and contact-force overlays.",
    )
    return parser.parse_args(argv)


def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    panel_height = 12 + 17 * len(lines)
    draw.rectangle((0, 0, image.width, panel_height), fill=(0, 0, 0, 160))
    for index, line in enumerate(lines):
        draw.text((8, 7 + 17 * index), line, fill=(255, 255, 255, 255))
    return np.asarray(image)


def _result_overlay(frame: np.ndarray, success: bool, lift_mm: float) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    color = (20, 135, 55, 205) if success else (175, 30, 30, 205)
    label = "PASS" if success else "FAIL"
    draw.rectangle((0, image.height - 38, image.width, image.height), fill=color)
    draw.text(
        (10, image.height - 29),
        f"{label}: final object lift {lift_mm:.1f} mm",
        fill=(255, 255, 255, 255),
    )
    return np.asarray(image)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("MP4 output requires the video extra") from exc
    if not frames:
        raise RuntimeError(f"No frames recorded for {path}")
    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def _montage_frames(
    frames_by_object: dict[str, list[np.ndarray]],
    width: int,
    height: int,
) -> list[np.ndarray]:
    names = list(frames_by_object)
    columns = 2
    rows = (len(names) + columns - 1) // columns
    frame_count = max(len(frames) for frames in frames_by_object.values())
    montage: list[np.ndarray] = []
    for frame_index in range(frame_count):
        panels: list[np.ndarray] = []
        for name in names:
            frames = frames_by_object[name]
            panels.append(frames[min(frame_index, len(frames) - 1)])
        while len(panels) < rows * columns:
            panels.append(np.zeros((height, width, 3), dtype=np.uint8))
        montage.append(
            np.concatenate(
                [
                    np.concatenate(panels[row * columns : (row + 1) * columns], axis=1)
                    for row in range(rows)
                ],
                axis=0,
            )
        )
    return montage


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("--fps, --width, and --height must be positive")

    benchmark_args = parse_benchmark_args(
        [
            "--objects",
            *args.objects,
            "--methods",
            "graspgenx",
            "--candidate-dir",
            str(args.candidate_dir),
            "--points",
            args.point,
            "--workers",
            "1",
            "--no-plots",
        ]
    )
    _validate_args(benchmark_args)
    points = build_grid_points(benchmark_args)
    if len(points) != 1:
        raise ValueError("--point must resolve to exactly one P1-P10 grid point")
    point = points[0]
    config = build_run_config(benchmark_args)

    args.out.mkdir(parents=True, exist_ok=True)
    frames_by_object: dict[str, list[np.ndarray]] = {}
    rows: list[dict[str, object]] = []

    for object_name in args.objects:
        obj = BUILTIN_OBJECTS[object_name]
        yaml_path = args.candidate_dir / f"{object_name}.yml"
        candidates = load_isaac_grasp_yaml(yaml_path)
        selection = _select_candidate(
            config=config,
            obj=obj,
            candidates=candidates,
            topdown_cos=None,
        )
        diagnostic_rejected = selection is None and args.fallback_rank is not None
        if selection is None:
            if diagnostic_rejected:
                if not 0 <= args.fallback_rank < len(candidates):
                    raise ValueError(
                        f"--fallback-rank is outside {object_name}'s candidate list"
                    )
                selection = (args.fallback_rank, candidates[args.fallback_rank])
            else:
                rows.append(
                    {
                        "object": object_name,
                        "label": obj.label,
                        "point": point.point_id,
                        "candidate_rank": "",
                        "candidate_confidence": "",
                        "candidate_status": "none_collision_free",
                        "success": False,
                        "failure_mode": "no_executable_candidate",
                        "final_lift_mm": 0.0,
                        "max_xy_displacement_mm": 0.0,
                        "video": "",
                    }
                )
                print(f"{obj.label}: no executable candidate")
                continue

        if selection is None:  # Narrow the optional type after the branch above.
            raise AssertionError("candidate selection unexpectedly missing")

        rank, candidate = selection
        model, object_geom_id, object_qadr = _add_object_model(
            config.xml,
            obj,
            PAPER_OBJECT_MASS_KG[object_name],
        )
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = tabletop_aabb_center(obj) + np.array([0.0, 0.0, 0.10])
        camera.distance = args.camera_distance
        camera.azimuth = args.camera_azimuth
        camera.elevation = args.camera_elevation
        scene_option = mujoco.MjvOption()
        if not args.no_contact_visualization:
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        frames: list[np.ndarray] = []
        next_frame_time = 0.0
        initial_position = tabletop_aabb_center(obj)

        def record_frame(
            _model: mujoco.MjModel,
            data: mujoco.MjData,
            phase: str,
        ) -> None:
            nonlocal next_frame_time
            if data.time + 1e-9 < next_frame_time:
                return
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            frame = renderer.render()
            position = _object_position(data, object_qadr)
            thumb_force, opposing_force = _contact_side_normal_forces(
                model,
                data,
                object_geom_id,
            )
            frame = _overlay(
                frame,
                [
                    f"{obj.label} | {point.point_id} | GenX rank {rank}, score {candidate.confidence:.3f}",
                    "GenX: target grasp pose | shared executor: motion + force control",
                    (
                        "target status: REJECTED by open-hand collision filter"
                        if diagnostic_rejected
                        else "target status: collision-free at open-hand target"
                    ),
                    f"phase: {phase}",
                    f"lift: {(position[2] - initial_position[2]) * 1000.0:.1f} mm | xy motion: {np.linalg.norm(position[:2] - initial_position[:2]) * 1000.0:.1f} mm",
                    f"normal force: thumb {thumb_force:.1f} N | fingers {opposing_force:.1f} N",
                ],
            )
            frames.append(frame)
            next_frame_time += 1.0 / args.fps

        result = _execute_trial(
            config=config,
            obj=obj,
            method="graspgenx",
            point=point,
            candidate_selection=selection,
            model=model,
            object_geom_id=object_geom_id,
            object_qadr=object_qadr,
            frame_callback=record_frame,
        )
        renderer.close()
        scored_success = result.success and not diagnostic_rejected
        failure_mode = (
            "rejected_target_collision"
            if diagnostic_rejected
            else result.failure_mode
        )
        final_frame = _result_overlay(frames[-1], scored_success, result.final_lift_mm)
        frames.extend([final_frame] * args.fps)
        frames_by_object[object_name] = frames
        video_path = args.out / f"{object_name}_{point.point_id}.mp4"
        _write_video(video_path, frames, args.fps)
        rows.append(
            {
                "object": object_name,
                "label": obj.label,
                "point": point.point_id,
                "candidate_rank": rank,
                "candidate_confidence": candidate.confidence,
                "candidate_status": (
                    "rejected_collision_diagnostic"
                    if diagnostic_rejected
                    else "collision_free"
                ),
                "success": scored_success,
                "failure_mode": failure_mode,
                "final_lift_mm": result.final_lift_mm,
                "max_xy_displacement_mm": result.max_object_xy_displacement_mm,
                "video": str(video_path.resolve()),
            }
        )
        print(
            f"{obj.label}: {'PASS' if scored_success else 'FAIL'}, "
            f"lift={result.final_lift_mm:.1f} mm -> {video_path}"
        )

    if frames_by_object:
        montage_path = args.out / f"graspgenx_{point.point_id}_montage.mp4"
        _write_video(
            montage_path,
            _montage_frames(frames_by_object, args.width, args.height),
            args.fps,
        )
        print(f"Montage -> {montage_path}")

    summary_path = args.out / "summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "scope": "GraspGen-X pose visualization on estimated paper-object proxies",
        "model_role": "GraspGen-X selects the 6-D target grasp pose only",
        "executor_role": (
            "Shared RH56 approach, joint interpolation, approximate 6 N MuJoCo "
            "contact-force control, lift, and hold"
        ),
        "object_proxy_warning": (
            "Primitive geometry and estimated masses are suitable for control-flow "
            "inspection, not a paper-facing 15-object success-rate claim."
        ),
        "fallback_rank_warning": (
            "A fallback rank is rendered only to visualize why an otherwise "
            "unexecutable target is rejected; it is always scored as failure."
            if args.fallback_rank is not None
            else None
        ),
        "point": asdict(point),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "candidate_sha256": {
            name: hashlib.sha256(
                (args.candidate_dir / f"{name}.yml").read_bytes()
            ).hexdigest()
            for name in args.objects
        },
    }
    # NumPy arrays in GridPoint are normalized for JSON output.
    metadata["point"]["offset_m"] = point.offset_m.tolist()
    (args.out / "assumptions.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
