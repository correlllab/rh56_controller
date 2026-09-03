#!/usr/bin/env python3
"""Stream all paper-object pre-grasp path trials into one MP4 video."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rh56_controller.grasp_geometry import ClosureGeometry, InspireHandFK
from rh56_controller.capsule_hand_proxy import COLLISION_NUMERICAL_EPSILON_M
from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS
from tools.demo_strategy_pregrasp_collision import (
    STRATEGIES,
    build_capsules_for_ctrl,
    solve_mode,
    strategy_poses,
    tabletop_object_center,
)
from tools.render_strategy_grid_paths import (
    iter_frames_for_strategy,
    load_rows,
    sort_key,
)
from tools.run_strategy_pregrasp_paper_batch import PAPER_OBJECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render every sampled start-to-pre-grasp-target path for the paper "
            "object set into one streamed MP4."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/current"),
        help="Batch artifact root containing one directory per paper object.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        choices=PAPER_OBJECTS,
        default=PAPER_OBJECTS,
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGIES),
        default=list(STRATEGIES),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/current/all_trials/all_trials.mp4"),
    )
    parser.add_argument("--xml", default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--rebuild-fk", action="store_true")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frames-per-path", type=int, default=8)
    parser.add_argument("--hold-frames", type=int, default=2)
    parser.add_argument("--azimuth", type=float, default=-48.0)
    parser.add_argument("--elevation", type=float, default=-24.0)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--floor-tolerance-mm", type=float, default=None)
    parser.add_argument(
        "--show-capsules",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        help="Limit paths per object/strategy for a quick render smoke test.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=7,
        choices=range(1, 11),
        metavar="1-10",
        help="MP4 encoder quality; higher values produce larger files.",
    )
    return parser.parse_args()


def load_assumptions(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sweep assumptions: {path}")
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def open_video_writer(args: argparse.Namespace):
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit(
            "MP4 rendering requires the video extra: "
            "UV_PROJECT_ENVIRONMENT=.venv312 uv run --extra video "
            "python tools/render_strategy_paper_trials.py"
        ) from exc

    writer = imageio_ffmpeg.write_frames(
        str(args.out),
        (args.width, args.height),
        fps=args.fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=args.quality,
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    return writer


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise ValueError("--width and --height must be positive even numbers")
    if args.fps <= 0 or args.frames_per_path <= 0 or args.hold_frames < 0:
        raise ValueError("fps and frames-per-path must be positive; hold-frames cannot be negative")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fk = (
        InspireHandFK(xml_path=args.xml, rebuild=args.rebuild_fk)
        if args.xml
        else InspireHandFK(rebuild=args.rebuild_fk)
    )
    closure = ClosureGeometry(fk)
    model = mujoco.MjModel.from_xml_path(str(args.xml or fk.xml_path))
    data = mujoco.MjData(model)
    writer = open_video_writer(args)

    frame_count = 0
    trial_count = 0
    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    frames_per_trial = args.frames_per_path + args.hold_frames

    try:
        for object_index, object_name in enumerate(args.objects, start=1):
            object_dir = args.root / object_name
            assumptions_path = object_dir / "assumptions.json"
            volume_path = object_dir / "volume.csv"
            assumptions = load_assumptions(assumptions_path)
            rows = load_rows(volume_path)
            obj = BUILTIN_OBJECTS[object_name]
            mode = str(assumptions["mode"])
            target_width_m = float(assumptions["target_width_m"])
            final_result = solve_mode(closure, mode, target_width_m)
            iterative_width_raw = assumptions.get("iterative_width_mm")
            iterative_width_m = (
                float(iterative_width_raw) / 1000.0
                if iterative_width_raw is not None
                else None
            )
            strategies = strategy_poses(
                fk,
                closure,
                mode=mode,
                final_result=final_result,
                target_width_m=target_width_m,
                preopen_m=float(assumptions.get("preopen_mm", 0.0)) / 1000.0,
                iterative_width_m=iterative_width_m,
                iterative_pregrasp_policy=str(
                    assumptions.get("iterative_pregrasp_policy", "planner-max-width")
                ),
            )
            strategy_by_name = {strategy.name: strategy for strategy in strategies}
            half_extents = np.asarray(obj.size_m, dtype=float) / 2.0
            object_center = tabletop_object_center(obj)
            floor_z_m = float(assumptions.get("floor_z_m", 0.0))
            floor_tolerance_m = (
                args.floor_tolerance_mm / 1000.0
                if args.floor_tolerance_mm is not None
                else float(assumptions.get("floor_tolerance_mm", 3.0)) / 1000.0
            )

            for strategy_name in args.strategies:
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
                group_start_frame = frame_count
                frames = iter_frames_for_strategy(
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
                    display_name=f"{object_index}/{len(args.objects)} {obj.label}",
                )
                for frame in frames:
                    writer.send(np.ascontiguousarray(frame))
                    frame_count += 1

                expected_frames = len(strategy_rows) * frames_per_trial
                if frame_count - group_start_frame != expected_frames:
                    raise RuntimeError(
                        f"Rendered frame count mismatch for {object_name}/{strategy_name}"
                    )

                for local_index, row in enumerate(strategy_rows):
                    start_frame = group_start_frame + local_index * frames_per_trial
                    end_frame = start_frame + frames_per_trial
                    trial_count += 1
                    trial_rows.append(
                        {
                            "trial": trial_count,
                            "object": object_name,
                            "label": obj.label,
                            "strategy": strategy_name,
                            "point_id": row.get("point_id", ""),
                            "valid": row["valid"],
                            "dominant_blocker": row["dominant_blocker"],
                            "start_frame": start_frame,
                            "end_frame_exclusive": end_frame,
                            "start_time_s": f"{start_frame / args.fps:.3f}",
                            "end_time_s": f"{end_frame / args.fps:.3f}",
                        }
                    )

                valid_count = sum(int(row["valid"]) for row in strategy_rows)
                summary_rows.append(
                    {
                        "object": object_name,
                        "label": obj.label,
                        "strategy": strategy_name,
                        "trials": len(strategy_rows),
                        "valid_trials": valid_count,
                        "invalid_trials": len(strategy_rows) - valid_count,
                        "start_time_s": f"{group_start_frame / args.fps:.3f}",
                        "end_time_s": f"{frame_count / args.fps:.3f}",
                    }
                )
                print(
                    f"[{object_index}/{len(args.objects)}] {object_name}/{strategy_name}: "
                    f"{len(strategy_rows)} trials, {valid_count} valid",
                    flush=True,
                )
    finally:
        writer.close()

    write_csv(
        args.out.parent / "trial_index.csv",
        trial_rows,
        [
            "trial",
            "object",
            "label",
            "strategy",
            "point_id",
            "valid",
            "dominant_blocker",
            "start_frame",
            "end_frame_exclusive",
            "start_time_s",
            "end_time_s",
        ],
    )
    write_csv(
        args.out.parent / "summary.csv",
        summary_rows,
        [
            "object",
            "label",
            "strategy",
            "trials",
            "valid_trials",
            "invalid_trials",
            "start_time_s",
            "end_time_s",
        ],
    )
    metadata = {
        "script": "tools/render_strategy_paper_trials.py",
        "simulation_only": True,
        "uses_hardware": False,
        "source_root": str(args.root),
        "video": str(args.out),
        "objects": args.objects,
        "strategies": args.strategies,
        "trial_count": trial_count,
        "frame_count": frame_count,
        "duration_s": frame_count / args.fps,
        "path_definition": (
            "Each trial keeps its strategy-specific pre-grasp hand shape fixed "
            "and moves the wrist/base linearly from the sampled start to the "
            "pre-grasp target. Finger closure after arrival is not rendered or scored."
        ),
        "rendering": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "frames_per_path": args.frames_per_path,
            "hold_frames": args.hold_frames,
            "show_capsules": args.show_capsules,
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "quality": args.quality,
            "object_collision_numerical_epsilon_mm": (
                COLLISION_NUMERICAL_EPSILON_M * 1000.0
            ),
        },
    }
    (args.out.parent / "assumptions.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.parent / 'trial_index.csv'}")
    print(f"Rendered {trial_count} trials in {frame_count / args.fps:.1f} video seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
