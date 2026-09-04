#!/usr/bin/env python3
"""Calibrate a fixed RealSense camera to the UR base without commanding motion.

The operator moves the arm using the lab's normal manual/reduced-speed
procedure.  This tool only reads RGB-D frames and, when requested, the current
UR TCP pose through the RTDE receive interface.  It never imports
``rtde_control`` and never sends a robot, hand, or motion command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rh56_controller.camera_calibration import (  # noqa: E402
    average_transforms,
    calibrate_intrinsics,
    chessboard_object_points,
    detect_chessboard,
    estimate_eye_to_hand,
    evaluate_hand_eye,
    invert_transform,
    pose_vector_xyz_rotvec_to_transform,
    project_target_points,
    rotation_error_deg,
    solve_chessboard_pose,
    transform_to_dict,
)


DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts/ur5_external_camera_calibration_real"


@dataclass(frozen=True)
class CameraMetadata:
    name: str
    serial: str
    firmware: str
    width: int
    height: int
    fps: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    distortion_model: str
    depth_scale_m: float
    depth_to_color_rotation: np.ndarray
    depth_to_color_translation_m: np.ndarray
    depth_visual_preset: str
    emitter_enabled: float
    laser_power: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a fixed RealSense and stationary UR TCP poses to estimate "
            "T_base_camera. No robot or RH56 command is ever sent."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Artifact directory. A timestamped directory is used by default. "
            "For --calibrate-only, point this at an existing capture directory."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Capture one RGB-D frame and check chessboard visibility; do not contact the UR.",
    )
    mode.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Recompute calibration from an existing capture_manifest.yaml without hardware.",
    )
    mode.add_argument(
        "--print-board-only",
        action="store_true",
        help="Generate a physical-size chessboard SVG without opening the camera or UR.",
    )
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--ur-ip", default="192.168.0.4")
    parser.add_argument(
        "--pose-source",
        choices=("rtde", "manual"),
        default="rtde",
        help=(
            "Read getActualTCPPose through RTDE receive, or enter the pendant's "
            "x y z rx ry rz values manually (default: rtde)."
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--settle-frames", type=int, default=5)
    parser.add_argument(
        "--laser-power",
        type=float,
        default=None,
        help="Optional D4xx projector power applied for this process.",
    )
    parser.add_argument("--captures", type=int, default=25)
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument(
        "--page-width-mm",
        type=float,
        default=None,
        help="Optional physical SVG/backing width; must be used with --page-height-mm.",
    )
    parser.add_argument(
        "--page-height-mm",
        type=float,
        default=None,
        help="Optional physical SVG/backing height; must be used with --page-width-mm.",
    )
    parser.add_argument(
        "--intrinsics-source",
        choices=("factory", "estimate"),
        default="factory",
        help=(
            "Use RealSense factory color intrinsics (recommended for aligned RGB-D) "
            "or estimate an OpenCV model from training views."
        ),
    )
    parser.add_argument(
        "--holdout-every",
        type=int,
        default=5,
        help="Reserve every Nth accepted capture for validation (default: 5).",
    )
    parser.add_argument(
        "--motion-translation-mm",
        type=float,
        default=0.5,
        help="Reject an RTDE capture if TCP translation changes more than this during capture.",
    )
    parser.add_argument(
        "--motion-rotation-deg",
        type=float,
        default=0.1,
        help="Reject an RTDE capture if TCP rotation changes more than this during capture.",
    )
    parser.add_argument(
        "--max-holdout-translation-mm",
        type=float,
        default=5.0,
        help="Reported quality gate for held-out transform consistency.",
    )
    parser.add_argument(
        "--max-holdout-rotation-deg",
        type=float,
        default=2.0,
        help="Reported quality gate for held-out transform consistency.",
    )
    parser.add_argument(
        "--max-holdout-reprojection-px",
        type=float,
        default=3.0,
        help="Reported quality gate for held-out robot-plus-camera reprojection.",
    )
    parser.add_argument(
        "--no-save-depth",
        action="store_true",
        help="Do not save aligned uint16 depth PNGs with accepted captures.",
    )
    parser.add_argument(
        "--mount-description",
        default="fixed external D435, approximately vertical downward toward the desk",
    )
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.width < 64 or args.height < 64 or args.fps < 1:
        raise ValueError("image dimensions must be at least 64 and FPS must be positive")
    if args.warmup_frames < 0 or args.settle_frames < 1:
        raise ValueError("frame counts cannot be negative and --settle-frames must be positive")
    if args.laser_power is not None and args.laser_power < 0.0:
        raise ValueError("--laser-power cannot be negative")
    if args.captures < 5:
        raise ValueError("--captures must be at least 5")
    if args.pattern_cols < 2 or args.pattern_rows < 2:
        raise ValueError("the chessboard needs at least 2x2 inner corners")
    if args.square_size_mm <= 0.0:
        raise ValueError("--square-size-mm must be positive")
    if (args.page_width_mm is None) != (args.page_height_mm is None):
        raise ValueError("--page-width-mm and --page-height-mm must be used together")
    if args.page_width_mm is not None:
        if args.page_width_mm <= 0.0 or args.page_height_mm <= 0.0:
            raise ValueError("SVG page dimensions must be positive")
        checker_width_mm = (args.pattern_cols + 1) * args.square_size_mm
        checker_height_mm = (args.pattern_rows + 1) * args.square_size_mm
        if args.page_width_mm < checker_width_mm or args.page_height_mm < checker_height_mm:
            raise ValueError(
                "SVG page is smaller than the checker field: "
                f"need at least {checker_width_mm:g} x {checker_height_mm:g} mm"
            )
    if args.holdout_every < 2:
        raise ValueError("--holdout-every must be at least 2")
    if args.motion_translation_mm <= 0.0 or args.motion_rotation_deg <= 0.0:
        raise ValueError("motion rejection thresholds must be positive")
    if args.calibrate_only and args.out is None:
        raise ValueError("--calibrate-only requires --out pointing at a capture directory")


def _default_output_dir() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_ARTIFACT_ROOT / stamp


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"failed to write {path}")


def _save_depth(path: Path, depth_u16: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if depth_u16.dtype != np.uint16:
        raise ValueError("depth image must be uint16")
    if not cv2.imwrite(str(path), depth_u16):
        raise OSError(f"failed to write {path}")


def _annotate_detection(
    image_rgb: np.ndarray,
    pattern_size: tuple[int, int],
    corners: np.ndarray | None,
    label: str,
) -> np.ndarray:
    import cv2

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if corners is not None:
        cv2.drawChessboardCorners(image_bgr, pattern_size, corners, True)
        columns, rows = pattern_size
        points = np.round(corners.reshape(-1, 2)).astype(int)
        origin = tuple(points[0])
        x_endpoint = tuple(points[columns - 1])
        y_endpoint = tuple(points[(rows - 1) * columns])
        cv2.arrowedLine(image_bgr, origin, x_endpoint, (255, 40, 255), 3, cv2.LINE_AA)
        cv2.arrowedLine(image_bgr, origin, y_endpoint, (255, 255, 40), 3, cv2.LINE_AA)
        for point, text, color in (
            (origin, "ORIGIN", (40, 40, 255)),
            (x_endpoint, "+X", (255, 40, 255)),
            (y_endpoint, "+Y", (255, 255, 40)),
        ):
            cv2.circle(image_bgr, point, 7, color, 2, cv2.LINE_AA)
            cv2.putText(
                image_bgr,
                text,
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        color = (40, 220, 40)
    else:
        color = (20, 20, 240)
    cv2.putText(
        image_bgr,
        label,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _depth_preview(depth_u16: np.ndarray) -> np.ndarray:
    import cv2

    valid = depth_u16[depth_u16 > 0]
    if valid.size == 0:
        scaled = np.zeros(depth_u16.shape, dtype=np.uint8)
    else:
        lower, upper = np.percentile(valid, [2.0, 98.0])
        if upper <= lower:
            upper = lower + 1.0
        scaled = np.clip((depth_u16.astype(float) - lower) * 255.0 / (upper - lower), 0, 255)
        scaled = scaled.astype(np.uint8)
        scaled[depth_u16 == 0] = 0
    colored_bgr = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def _draw_prediction(
    image_rgb: np.ndarray,
    detected: np.ndarray,
    predicted: np.ndarray,
    rmse_px: float,
) -> np.ndarray:
    import cv2

    image = image_rgb.copy()
    for point in detected.reshape(-1, 2):
        cv2.circle(image, tuple(np.round(point).astype(int)), 4, (20, 230, 20), 2, cv2.LINE_AA)
    for point in predicted.reshape(-1, 2):
        x_position, y_position = np.round(point).astype(int)
        cv2.drawMarker(
            image,
            (int(x_position), int(y_position)),
            (255, 30, 220),
            cv2.MARKER_TILTED_CROSS,
            10,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        f"green=detected magenta=held-out prediction RMSE={rmse_px:.2f}px",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _write_printable_chessboard_svg(
    path: Path,
    pattern_size: tuple[int, int],
    square_size_mm: float,
    page_size_mm: tuple[float, float] | None = None,
) -> None:
    columns, rows = pattern_size
    grid_columns = columns + 1
    grid_rows = rows + 1
    checker_width_mm = grid_columns * square_size_mm
    checker_height_mm = grid_rows * square_size_mm
    if page_size_mm is None:
        margin_mm = 0.35 * square_size_mm
        width_mm = checker_width_mm + 2.0 * margin_mm
        height_mm = checker_height_mm + 2.0 * margin_mm
    else:
        width_mm, height_mm = page_size_mm
        if width_mm < checker_width_mm or height_mm < checker_height_mm:
            raise ValueError("page_size_mm cannot be smaller than the checker field")
    offset_x_mm = 0.5 * (width_mm - checker_width_mm)
    offset_y_mm = 0.5 * (height_mm - checker_height_mm)
    rectangles: list[str] = []
    for row in range(grid_rows):
        for column in range(grid_columns):
            if (row + column) % 2:
                continue
            rectangles.append(
                f'  <rect x="{offset_x_mm + column * square_size_mm:.6f}" '
                f'y="{offset_y_mm + row * square_size_mm:.6f}" '
                f'width="{square_size_mm:.6f}" height="{square_size_mm:.6f}" fill="black"/>'
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm:.6f}mm" '
                    f'height="{height_mm:.6f}mm" viewBox="0 0 {width_mm:.6f} {height_mm:.6f}">'
                ),
                f'  <rect width="{width_mm:.6f}" height="{height_mm:.6f}" fill="white"/>',
                *rectangles,
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_print_board(args: argparse.Namespace, output_dir: Path) -> int:
    """Write a dimensioned SVG and metadata without accessing hardware."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    page_size_mm = (
        None
        if args.page_width_mm is None
        else (args.page_width_mm, args.page_height_mm)
    )
    path = output_dir / "printable_chessboard.svg"
    _write_printable_chessboard_svg(
        path,
        pattern_size,
        args.square_size_mm,
        page_size_mm=page_size_mm,
    )
    checker_width_mm = (args.pattern_cols + 1) * args.square_size_mm
    checker_height_mm = (args.pattern_rows + 1) * args.square_size_mm
    summary = {
        "pattern_inner_cols": args.pattern_cols,
        "pattern_inner_rows": args.pattern_rows,
        "square_size_mm": args.square_size_mm,
        "checker_width_mm": checker_width_mm,
        "checker_height_mm": checker_height_mm,
        "page_width_mm": args.page_width_mm,
        "page_height_mm": args.page_height_mm,
        "x_inner_corner_span_mm": (args.pattern_cols - 1) * args.square_size_mm,
        "y_inner_corner_span_mm": (args.pattern_rows - 1) * args.square_size_mm,
        "hardware_contacted": 0,
    }
    _write_csv(output_dir / "summary.csv", [summary])
    (output_dir / "board_metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "rh56_printable_chessboard/v1",
                "inner_corners": list(pattern_size),
                "square_size_mm": args.square_size_mm,
                "checker_size_mm": [checker_width_mm, checker_height_mm],
                "page_size_mm": (
                    None if page_size_mm is None else list(page_size_mm)
                ),
                "print_instructions": [
                    "Print at 100% or actual size; disable fit-to-page scaling.",
                    "Measure several intervals after printing and use the measured average square size.",
                    "Mount the paper flat on a rigid backing without bubbles or wrinkles.",
                ],
                "safety": {"camera_contacted": False, "ur_contacted": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"Chessboard SVG: {path}")
    print(
        f"Pattern: {args.pattern_cols}x{args.pattern_rows} inner corners, "
        f"{args.square_size_mm:g} mm squares, "
        f"{checker_width_mm:g}x{checker_height_mm:g} mm checker field"
    )
    print("Print at 100% / actual size; disable fit-to-page scaling.")
    print("Camera contacted: no; robot contacted: no")
    return 0


def _camera_metadata_to_dict(metadata: CameraMetadata) -> dict[str, object]:
    return {
        "name": metadata.name,
        "serial": metadata.serial,
        "firmware": metadata.firmware,
        "stream": {
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "color_format": "rgb8",
            "depth_format": "z16 aligned to color",
        },
        "factory_color_intrinsics": {
            "camera_matrix": metadata.camera_matrix.tolist(),
            "distortion_coefficients": metadata.distortion.tolist(),
            "distortion_model": metadata.distortion_model,
        },
        "depth": {
            "scale_m_per_unit": metadata.depth_scale_m,
            "depth_to_color_rotation": metadata.depth_to_color_rotation.tolist(),
            "depth_to_color_translation_m": metadata.depth_to_color_translation_m.tolist(),
            "visual_preset": metadata.depth_visual_preset,
            "emitter_enabled": metadata.emitter_enabled,
            "laser_power": metadata.laser_power,
        },
    }


def _start_camera(args: argparse.Namespace):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is required; install the real-ur5-vision profile"
        ) from exc

    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        raise RuntimeError("no RealSense device detected")
    serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
    if args.camera_serial is None:
        if len(serials) != 1:
            raise RuntimeError(
                "multiple RealSense devices detected; select one with --camera-serial: "
                + ", ".join(serials)
            )
        serial = serials[0]
    else:
        serial = args.camera_serial
        if serial not in serials:
            raise RuntimeError(
                f"RealSense serial {serial} not found; available: {', '.join(serials)}"
            )

    selected_device = next(
        device
        for device in devices
        if device.get_info(rs.camera_info.serial_number) == serial
    )
    depth_sensor = selected_device.first_depth_sensor()
    if args.laser_power is not None:
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)
        if not laser_range.min <= args.laser_power <= laser_range.max:
            raise ValueError(
                f"--laser-power must be in [{laser_range.min:g}, {laser_range.max:g}]"
            )
        depth_sensor.set_option(rs.option.laser_power, args.laser_power)

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    device = profile.get_device()
    for _ in range(args.warmup_frames):
        pipeline.wait_for_frames(5000)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    depth_to_color = depth_profile.get_extrinsics_to(color_profile)
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    metadata = CameraMetadata(
        name=device.get_info(rs.camera_info.name),
        serial=serial,
        firmware=device.get_info(rs.camera_info.firmware_version),
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_matrix=camera_matrix,
        distortion=np.asarray(intrinsics.coeffs, dtype=float),
        distortion_model=str(intrinsics.model),
        depth_scale_m=float(depth_sensor.get_depth_scale()),
        depth_to_color_rotation=np.asarray(depth_to_color.rotation, dtype=float).reshape(3, 3),
        depth_to_color_translation_m=np.asarray(depth_to_color.translation, dtype=float),
        depth_visual_preset=depth_sensor.get_option_value_description(
            rs.option.visual_preset,
            depth_sensor.get_option(rs.option.visual_preset),
        ),
        emitter_enabled=float(depth_sensor.get_option(rs.option.emitter_enabled)),
        laser_power=float(depth_sensor.get_option(rs.option.laser_power)),
    )
    return pipeline, align, metadata


def _read_aligned_frame(pipeline, align, settle_frames: int):
    frames = None
    for _ in range(settle_frames):
        frames = align.process(pipeline.wait_for_frames(5000))
    assert frames is not None
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("aligned RGB-D frames were incomplete")
    color = np.asanyarray(color_frame.get_data()).copy()
    depth = np.asanyarray(depth_frame.get_data()).copy()
    return color, depth, float(color_frame.get_timestamp()), int(color_frame.get_frame_number())


def _check_board_coverage(
    corners: np.ndarray | None,
    width: int,
    height: int,
) -> dict[str, float]:
    if corners is None:
        return {"board_width_fraction": 0.0, "board_height_fraction": 0.0, "min_edge_margin_px": 0.0}
    points = corners.reshape(-1, 2)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    margins = [minimum[0], minimum[1], width - maximum[0], height - maximum[1]]
    return {
        "board_width_fraction": float((maximum[0] - minimum[0]) / width),
        "board_height_fraction": float((maximum[1] - minimum[1]) / height),
        "min_edge_margin_px": float(min(margins)),
    }


def depth_validity_metrics(
    depth_u16: np.ndarray,
    depth_scale_m: float,
) -> dict[str, float]:
    """Summarize full-frame and central depth availability."""

    depth = np.asarray(depth_u16)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("depth_u16 must be a two-dimensional uint16 image")
    if depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be positive")
    height, width = depth.shape
    center = depth[
        3 * height // 8 : 5 * height // 8,
        3 * width // 8 : 5 * width // 8,
    ]

    def metrics(prefix: str, values: np.ndarray) -> dict[str, float]:
        valid = values[values > 0]
        result = {f"{prefix}_valid_depth_fraction": float(valid.size / values.size)}
        if valid.size:
            result[f"{prefix}_median_depth_m"] = float(np.median(valid) * depth_scale_m)
        else:
            result[f"{prefix}_median_depth_m"] = float("nan")
        return result

    return metrics("full", depth) | metrics("center", center)


def run_camera_check(args: argparse.Namespace, output_dir: Path) -> int:
    pipeline, align, metadata = _start_camera(args)
    try:
        image_rgb, depth_u16, timestamp_ms, frame_number = _read_aligned_frame(
            pipeline, align, args.settle_frames
        )
    finally:
        pipeline.stop()
    pattern_size = (args.pattern_cols, args.pattern_rows)
    corners = detect_chessboard(image_rgb, pattern_size)
    detected = corners is not None
    coverage = _check_board_coverage(corners, args.width, args.height)
    depth_metrics = depth_validity_metrics(depth_u16, metadata.depth_scale_m)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb(output_dir / "camera_check_rgb.png", image_rgb)
    _save_rgb(
        output_dir / "camera_check_annotated.png",
        _annotate_detection(
            image_rgb,
            pattern_size,
            corners,
            (
                f"{pattern_size[0]}x{pattern_size[1]} chessboard detected"
                if detected
                else f"{pattern_size[0]}x{pattern_size[1]} chessboard NOT detected"
            ),
        ),
    )
    _save_depth(output_dir / "camera_check_depth_u16.png", depth_u16)
    _save_rgb(output_dir / "camera_check_depth_preview.png", _depth_preview(depth_u16))
    summary = {
        "camera_serial": metadata.serial,
        "frame_number": frame_number,
        "camera_timestamp_ms": timestamp_ms,
        "chessboard_detected": int(detected),
        **coverage,
        **depth_metrics,
        "hardware_motion_commands_sent": 0,
    }
    _write_csv(output_dir / "summary.csv", [summary])
    payload = {
        "schema": "rh56_realsense_camera_check/v1",
        "scope": "read-only camera and target visibility check",
        "camera": _camera_metadata_to_dict(metadata),
        "chessboard": {
            "inner_corners": list(pattern_size),
            "square_size_m": args.square_size_mm / 1000.0,
            "detected": detected,
            **coverage,
        },
        "depth_check": depth_metrics,
        "mount_description": args.mount_description,
        "safety": {"hardware_motion_commands_sent": False, "ur_contacted": False},
    }
    (output_dir / "camera_check.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    _write_printable_chessboard_svg(
        output_dir / "printable_chessboard.svg",
        pattern_size,
        args.square_size_mm,
    )
    print(f"Camera: {metadata.name} serial={metadata.serial} firmware={metadata.firmware}")
    print(f"RGB-D stream: {args.width}x{args.height}@{args.fps} OK")
    print(f"Chessboard detected: {detected}")
    print(f"Annotated image: {output_dir / 'camera_check_annotated.png'}")
    print("Robot contacted: no; motion commands sent: no")
    return 0 if detected else 2


def _open_rtde_receiver(ip: str):
    try:
        import rtde_receive
    except ImportError as exc:
        raise RuntimeError(
            "ur-rtde is required for --pose-source rtde; install the real-ur5-vision profile"
        ) from exc
    # Deliberately instantiate only the receive interface.  Do not import or
    # construct RTDEControlInterface anywhere in this tool.
    return rtde_receive.RTDEReceiveInterface(ip)


def _read_rtde_pose(receiver) -> tuple[np.ndarray, np.ndarray]:
    pose_vector = np.asarray(receiver.getActualTCPPose(), dtype=float)
    return pose_vector_xyz_rotvec_to_transform(pose_vector), pose_vector


def _manual_pose() -> tuple[np.ndarray, np.ndarray]:
    while True:
        raw = input("Enter pendant TCP pose x y z rx ry rz (m, rad), or q: ").strip()
        if raw.lower() in {"q", "quit", "stop"}:
            raise EOFError
        try:
            values = np.asarray([float(value) for value in raw.replace(",", " ").split()])
            return pose_vector_xyz_rotvec_to_transform(values), values
        except (ValueError, TypeError) as exc:
            print(f"Invalid pose: {exc}")


def _capture_manifest(
    args: argparse.Namespace,
    metadata: CameraMetadata,
    captures: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": "rh56_ur5_external_camera_capture/v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "real fixed external camera; operator-positioned arm",
        "camera": _camera_metadata_to_dict(metadata),
        "chessboard": {
            "inner_corners": [args.pattern_cols, args.pattern_rows],
            "square_size_m": args.square_size_mm / 1000.0,
        },
        "robot_pose": {
            "source": args.pose_source,
            "frame": "T_base_tcp",
            "rtde_ip": args.ur_ip if args.pose_source == "rtde" else None,
            "vector_convention": "[x,y,z,rx,ry,rz], metres and axis-angle radians",
        },
        "mount_description": args.mount_description,
        "captures": captures,
        "safety": {
            "hardware_motion_commands_sent": False,
            "rtde_interface": "receive-only" if args.pose_source == "rtde" else "not used",
            "operator_moves_arm_manually": True,
        },
        "assumptions": [
            "The RealSense is rigidly fixed for every capture and later experiment.",
            "The chessboard is rigid relative to the reported UR TCP frame.",
            "The robot is stationary while each image/pose pair is recorded.",
            "The reported TCP definition is not changed during or after calibration.",
            "A vertical camera is allowed, but wrist orientations must vary about multiple axes.",
        ],
    }


def _write_capture_state(
    output_dir: Path,
    args: argparse.Namespace,
    metadata: CameraMetadata,
    captures: list[dict[str, object]],
    rows: list[dict[str, object]],
) -> None:
    manifest = _capture_manifest(args, metadata, captures)
    (output_dir / "capture_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    if rows:
        _write_csv(output_dir / "capture_summary.csv", rows)


def capture_dataset(args: argparse.Namespace, output_dir: Path) -> None:
    manifest_path = output_dir / "capture_manifest.yaml"
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing dataset {manifest_path}; choose a new --out"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    pipeline, align, metadata = _start_camera(args)
    receiver = None
    captures: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    attempt = 0
    try:
        if args.pose_source == "rtde":
            receiver = _open_rtde_receiver(args.ur_ip)
        print("\nAssumptions and safety:")
        print("  - fixed external D435; current mount is approximately vertical downward")
        print("  - move the robot only with the lab's manual/reduced-speed procedure")
        print("  - this process opens RTDEReceiveInterface only; it cannot command motion")
        print("  - stop the robot before each capture and vary wrist roll/pitch/yaw")
        print("  - do not move the camera, target mount, or change the UR TCP definition")
        print(f"  - target: {args.pattern_cols}x{args.pattern_rows} inner corners, {args.square_size_mm:.3f} mm squares")

        while len(captures) < args.captures:
            command = input(
                f"\nPose {len(captures) + 1}/{args.captures}: stop robot, then Enter to capture; q to finish: "
            ).strip().lower()
            if command in {"q", "quit", "stop"}:
                break
            attempt += 1
            pre_transform = None
            pre_vector = None
            if receiver is not None:
                pre_transform, pre_vector = _read_rtde_pose(receiver)
            image_rgb, depth_u16, timestamp_ms, frame_number = _read_aligned_frame(
                pipeline, align, args.settle_frames
            )
            corners = detect_chessboard(image_rgb, pattern_size)
            failure_reason = ""
            motion_translation_mm = 0.0
            motion_rotation_deg = 0.0
            base_from_tcp = None
            pose_vector = None
            if corners is None:
                failure_reason = "chessboard_not_detected"
            elif receiver is not None:
                post_transform, post_vector = _read_rtde_pose(receiver)
                assert pre_transform is not None and pre_vector is not None
                motion_translation_mm = 1000.0 * float(
                    np.linalg.norm(post_transform[:3, 3] - pre_transform[:3, 3])
                )
                motion_rotation_deg = rotation_error_deg(pre_transform, post_transform)
                if motion_translation_mm > args.motion_translation_mm:
                    failure_reason = "robot_moved_during_capture_translation"
                elif motion_rotation_deg > args.motion_rotation_deg:
                    failure_reason = "robot_moved_during_capture_rotation"
                else:
                    base_from_tcp = average_transforms([pre_transform, post_transform])
                    pose_vector = 0.5 * (pre_vector + post_vector)
            else:
                try:
                    base_from_tcp, pose_vector = _manual_pose()
                except EOFError:
                    break

            accepted = not failure_reason
            accepted_index = len(captures) if accepted else -1
            attempt_stem = f"attempt_{attempt:03d}"
            annotated = _annotate_detection(
                image_rgb,
                pattern_size,
                corners,
                "accepted" if accepted else failure_reason.replace("_", " "),
            )
            _save_rgb(output_dir / "attempts" / f"{attempt_stem}.png", annotated)
            if accepted:
                assert corners is not None and base_from_tcp is not None and pose_vector is not None
                stem = f"capture_{accepted_index:03d}"
                rgb_rel = Path("captures") / f"{stem}_rgb.png"
                annotated_rel = Path("captures") / f"{stem}_annotated.png"
                depth_rel = Path("captures") / f"{stem}_depth_u16.png"
                _save_rgb(output_dir / rgb_rel, image_rgb)
                _save_rgb(output_dir / annotated_rel, annotated)
                if not args.no_save_depth:
                    _save_depth(output_dir / depth_rel, depth_u16)
                captures.append(
                    {
                        "index": accepted_index,
                        "attempt": attempt,
                        "rgb_path": str(rgb_rel),
                        "annotated_path": str(annotated_rel),
                        "depth_path": None if args.no_save_depth else str(depth_rel),
                        "camera_timestamp_ms": timestamp_ms,
                        "camera_frame_number": frame_number,
                        "base_from_tcp": np.asarray(base_from_tcp).tolist(),
                        "tcp_pose_vector": np.asarray(pose_vector).tolist(),
                        "corners_px": corners.reshape(-1, 2).tolist(),
                        "motion_translation_mm": motion_translation_mm,
                        "motion_rotation_deg": motion_rotation_deg,
                    }
                )
                print(f"Accepted capture {accepted_index}: frame {frame_number}")
            else:
                print(f"Rejected: {failure_reason}; annotated attempt saved")

            row = {
                "attempt": attempt,
                "accepted": int(accepted),
                "accepted_index": accepted_index,
                "failure_reason": failure_reason,
                "camera_frame_number": frame_number,
                "camera_timestamp_ms": timestamp_ms,
                "motion_translation_mm": motion_translation_mm,
                "motion_rotation_deg": motion_rotation_deg,
                "hardware_motion_commands_sent": 0,
            }
            rows.append(row)
            _write_capture_state(output_dir, args, metadata, captures, rows)
    finally:
        pipeline.stop()
        if receiver is not None and hasattr(receiver, "disconnect"):
            receiver.disconnect()

    _write_capture_state(output_dir, args, metadata, captures, rows)
    _write_printable_chessboard_svg(
        output_dir / "printable_chessboard.svg", pattern_size, args.square_size_mm
    )
    print(f"Captured {len(captures)} accepted views in {attempt} attempts")
    print(f"Dataset: {manifest_path}")


def _load_manifest(output_dir: Path) -> dict[str, object]:
    path = output_dir / "capture_manifest.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if payload.get("schema") != "rh56_ur5_external_camera_capture/v1":
        raise ValueError(f"unsupported capture manifest schema in {path}")
    return payload


def split_calibration_indices(count: int, holdout_every: int) -> tuple[list[int], list[int]]:
    """Return deterministic training and held-out capture indices."""

    if count < 5:
        raise ValueError("at least five accepted captures are required")
    if holdout_every < 2:
        raise ValueError("holdout_every must be at least 2")
    holdout = [index for index in range(count) if (index + 1) % holdout_every == 0]
    training = [index for index in range(count) if index not in holdout]
    if not holdout:
        holdout = [count - 1]
        training = training[:-1]
    if len(training) < 3:
        raise ValueError("the split leaves fewer than three training captures")
    return training, holdout


def pose_diversity(base_from_tcp: Sequence[np.ndarray]) -> dict[str, object]:
    """Summarize translation/rotation excitation used by hand-eye calibration."""

    poses = tuple(np.asarray(pose, dtype=float) for pose in base_from_tcp)
    if len(poses) < 2:
        raise ValueError("at least two poses are required")
    translations = np.asarray([pose[:3, 3] for pose in poses])
    translation_span_mm = 1000.0 * float(
        max(
            np.linalg.norm(first - second)
            for index, first in enumerate(translations)
            for second in translations[index + 1 :]
        )
    )
    reference_rotation = poses[0][:3, :3]
    relative_rotvec = np.asarray(
        [Rotation.from_matrix(reference_rotation.T @ pose[:3, :3]).as_rotvec() for pose in poses]
    )
    singular_values_deg = np.degrees(np.linalg.svd(relative_rotvec, compute_uv=False))
    max_rotation_span_deg = float(
        max(
            rotation_error_deg(first, second)
            for index, first in enumerate(poses)
            for second in poses[index + 1 :]
        )
    )
    rotation_axis_rank = int(np.count_nonzero(singular_values_deg > 2.0))
    return {
        "translation_span_mm": translation_span_mm,
        "max_rotation_span_deg": max_rotation_span_deg,
        "rotation_axis_rank_over_2deg": rotation_axis_rank,
        "rotation_singular_values_deg": singular_values_deg.tolist(),
    }


def calibrate_dataset(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    import cv2

    manifest = _load_manifest(output_dir)
    entries = manifest.get("captures", [])
    if len(entries) < 5:
        raise RuntimeError(f"only {len(entries)} accepted captures; at least five are required")
    chessboard = manifest["chessboard"]
    pattern_size = tuple(int(value) for value in chessboard["inner_corners"])
    object_points = chessboard_object_points(
        pattern_size, float(chessboard["square_size_m"])
    )
    camera = manifest["camera"]
    stream = camera["stream"]
    factory = camera["factory_color_intrinsics"]
    factory_matrix = np.asarray(factory["camera_matrix"], dtype=float)
    factory_distortion = np.asarray(factory["distortion_coefficients"], dtype=float)
    factory_model = str(factory["distortion_model"])
    corners = [np.asarray(entry["corners_px"], dtype=np.float32) for entry in entries]
    base_from_tcp = [np.asarray(entry["base_from_tcp"], dtype=float) for entry in entries]
    training_indices, holdout_indices = split_calibration_indices(
        len(entries), args.holdout_every
    )

    intrinsic_rms_px = float("nan")
    if args.intrinsics_source == "estimate":
        intrinsic = calibrate_intrinsics(
            [object_points.copy() for _ in training_indices],
            [corners[index] for index in training_indices],
            (int(stream["width"]), int(stream["height"])),
            estimate_distortion=True,
        )
        camera_matrix = intrinsic.camera_matrix
        distortion = intrinsic.distortion
        intrinsic_rms_px = intrinsic.rms_reprojection_error_px
    else:
        if "inverse_brown_conrady" in factory_model and not np.allclose(
            factory_distortion, 0.0
        ):
            raise RuntimeError(
                "non-zero inverse-Brown factory coefficients cannot be passed "
                "directly to OpenCV solvePnP; use --intrinsics-source estimate"
            )
        camera_matrix = factory_matrix
        distortion = factory_distortion

    camera_from_target = [
        solve_chessboard_pose(object_points, points, camera_matrix, distortion)
        for points in corners
    ]
    hand_eye = estimate_eye_to_hand(
        [base_from_tcp[index] for index in training_indices],
        [camera_from_target[index] for index in training_indices],
    )
    if not np.all(np.isfinite(hand_eye.base_from_camera)):
        raise RuntimeError(
            "hand-eye calibration returned non-finite values; collect more varied wrist rotations"
        )
    train_residuals = evaluate_hand_eye(
        [base_from_tcp[index] for index in training_indices],
        [camera_from_target[index] for index in training_indices],
        hand_eye.base_from_camera,
        hand_eye.gripper_from_target,
    )
    holdout_residuals = evaluate_hand_eye(
        [base_from_tcp[index] for index in holdout_indices],
        [camera_from_target[index] for index in holdout_indices],
        hand_eye.base_from_camera,
        hand_eye.gripper_from_target,
    )

    view_rows: list[dict[str, object]] = []
    reprojection_rmse: list[float] = []
    for index, (entry, base_tcp, detected) in enumerate(zip(entries, base_from_tcp, corners)):
        predicted_camera_target = (
            invert_transform(hand_eye.base_from_camera)
            @ base_tcp
            @ hand_eye.gripper_from_target
        )
        predicted = project_target_points(
            object_points,
            predicted_camera_target,
            camera_matrix,
            distortion,
        )
        error = float(
            np.sqrt(np.mean(np.sum((predicted - detected.reshape(-1, 2)) ** 2, axis=1)))
        )
        reprojection_rmse.append(error)
        direct_target = camera_from_target[index]
        robot_target = hand_eye.base_from_camera @ direct_target
        expected_target = base_tcp @ hand_eye.gripper_from_target
        view_rows.append(
            {
                "capture_index": index,
                "split": "holdout" if index in holdout_indices else "training",
                "rgb_path": entry["rgb_path"],
                "transform_translation_error_mm": 1000.0
                * float(np.linalg.norm(robot_target[:3, 3] - expected_target[:3, 3])),
                "transform_rotation_error_deg": rotation_error_deg(robot_target, expected_target),
                "robot_camera_reprojection_rmse_px": error,
            }
        )
    _write_csv(output_dir / "calibration_views.csv", view_rows)

    holdout_reprojection = np.asarray(
        [reprojection_rmse[index] for index in holdout_indices], dtype=float
    )
    holdout_reprojection_rms_px = float(
        np.sqrt(np.mean(np.square(holdout_reprojection)))
    )
    diversity = pose_diversity([base_from_tcp[index] for index in training_indices])
    quality_pass = bool(
        holdout_residuals.translation_rms_m * 1000.0
        <= args.max_holdout_translation_mm
        and holdout_residuals.rotation_rms_deg <= args.max_holdout_rotation_deg
        and holdout_reprojection_rms_px <= args.max_holdout_reprojection_px
        and diversity["rotation_axis_rank_over_2deg"] >= 2
    )
    summary = {
        "accepted_views": len(entries),
        "training_views": len(training_indices),
        "holdout_views": len(holdout_indices),
        "intrinsics_source": args.intrinsics_source,
        "intrinsic_rms_px": intrinsic_rms_px,
        "training_translation_rms_mm": 1000.0 * train_residuals.translation_rms_m,
        "training_rotation_rms_deg": train_residuals.rotation_rms_deg,
        "holdout_translation_rms_mm": 1000.0 * holdout_residuals.translation_rms_m,
        "holdout_rotation_rms_deg": holdout_residuals.rotation_rms_deg,
        "holdout_reprojection_rms_px": holdout_reprojection_rms_px,
        "translation_span_mm": diversity["translation_span_mm"],
        "max_rotation_span_deg": diversity["max_rotation_span_deg"],
        "rotation_axis_rank_over_2deg": diversity["rotation_axis_rank_over_2deg"],
        "quality_pass": int(quality_pass),
        "hardware_motion_commands_sent": 0,
    }
    _write_csv(output_dir / "summary.csv", [summary])

    worst_index = max(holdout_indices, key=lambda index: reprojection_rmse[index])
    worst_entry = entries[worst_index]
    source_bgr = cv2.imread(str(output_dir / worst_entry["rgb_path"]), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise OSError(f"failed to read {output_dir / worst_entry['rgb_path']}")
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    predicted_camera_target = (
        invert_transform(hand_eye.base_from_camera)
        @ base_from_tcp[worst_index]
        @ hand_eye.gripper_from_target
    )
    predicted = project_target_points(
        object_points, predicted_camera_target, camera_matrix, distortion
    )
    _save_rgb(
        output_dir / "verification_holdout.png",
        _draw_prediction(
            source_rgb,
            corners[worst_index],
            predicted,
            reprojection_rmse[worst_index],
        ),
    )

    calibration_payload = {
        "schema": "rh56_ur5_external_camera_calibration/v2",
        "scope": "real fixed external RealSense eye-to-hand calibration",
        "frames": {
            "base": "UR base",
            "camera": "RealSense color optical frame",
            "gripper": "UR TCP reported by RTDE/pendant",
            "target": "OpenCV chessboard first-inner-corner frame",
            "camera_axes": "+X right, +Y down, +Z forward",
            "equation": "p_base = T_base_camera @ p_camera",
        },
        "camera": camera,
        "selected_intrinsics": {
            "source": args.intrinsics_source,
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": distortion.tolist(),
        },
        "chessboard": chessboard,
        "transforms": {
            "base_from_camera": transform_to_dict(hand_eye.base_from_camera),
            "tcp_from_target": transform_to_dict(hand_eye.gripper_from_target),
        },
        "split": {
            "holdout_every": args.holdout_every,
            "training_indices": training_indices,
            "holdout_indices": holdout_indices,
        },
        "pose_diversity": diversity,
        "metrics": summary,
        "quality_thresholds": {
            "max_holdout_translation_mm": args.max_holdout_translation_mm,
            "max_holdout_rotation_deg": args.max_holdout_rotation_deg,
            "max_holdout_reprojection_px": args.max_holdout_reprojection_px,
        },
        "quality_pass": quality_pass,
        "safety": {
            "hardware_motion_commands_sent": False,
            "calibration_does_not_authorize_robot_motion": True,
        },
        "assumptions": manifest["assumptions"],
        "capture_manifest_sha256": hashlib.sha256(
            (output_dir / "capture_manifest.yaml").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "camera_calibration.yaml").write_text(
        yaml.safe_dump(calibration_payload, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Calibration quality pass: {quality_pass}")
    print(
        "Held-out residual: "
        f"{summary['holdout_translation_rms_mm']:.3f} mm, "
        f"{summary['holdout_rotation_rms_deg']:.3f} deg, "
        f"{holdout_reprojection_rms_px:.3f} px"
    )
    print(
        "Pose diversity: "
        f"{diversity['max_rotation_span_deg']:.1f} deg maximum rotation, "
        f"axis rank {diversity['rotation_axis_rank_over_2deg']}"
    )
    print(f"Calibration: {output_dir / 'camera_calibration.yaml'}")
    print(f"Visual check: {output_dir / 'verification_holdout.png'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (args.out if args.out is not None else _default_output_dir()).resolve()
    print("This tool is read-only with respect to the UR and RH56.")
    print(f"Output directory: {output_dir}")
    if args.print_board_only:
        return run_print_board(args, output_dir)
    if args.check_only:
        return run_camera_check(args, output_dir)
    if not args.calibrate_only:
        capture_dataset(args, output_dir)
    summary = calibrate_dataset(args, output_dir)
    return 0 if summary["quality_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
