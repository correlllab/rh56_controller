#!/usr/bin/env python3
"""Validate a UR-base camera transform with a colored target of known height."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect a colored target in a saved aligned RGB-D camera check, "
            "transform its top center into the UR base frame, and compare its "
            "height with an independent physical measurement. Offline only."
        )
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Directory containing camera_check.yaml and aligned RGB/depth PNGs.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--desk-z-mm", type=float, required=True)
    parser.add_argument(
        "--desk-plane",
        type=Path,
        default=None,
        help=(
            "Optional desk_plane_validation.yaml whose normal accounts for desk "
            "tilt. --desk-z-mm remains the independently measured Z at base XY=0."
        ),
    )
    parser.add_argument("--target-height-mm", type=float, required=True)
    parser.add_argument("--expected-base-x-mm", type=float, default=None)
    parser.add_argument("--expected-base-y-mm", type=float, default=None)
    parser.add_argument("--max-height-error-mm", type=float, default=5.0)
    parser.add_argument("--max-xy-error-mm", type=float, default=5.0)
    parser.add_argument("--hue-min", type=int, default=5)
    parser.add_argument("--hue-max", type=int, default=35)
    parser.add_argument("--min-saturation", type=int, default=140)
    parser.add_argument("--min-value", type=int, default=140)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Optional half-open pixel ROI used only for target selection.",
    )
    parser.add_argument("--erosion-px", type=int, default=9)
    parser.add_argument("--min-component-area-px", type=int, default=500)
    parser.add_argument("--foreground-percentile", type=float, default=10.0)
    parser.add_argument("--foreground-band-mm", type=float, default=10.0)
    parser.add_argument("--min-foreground-points", type=int, default=100)
    args = parser.parse_args(argv)
    if args.target_height_mm <= 0.0:
        raise ValueError("--target-height-mm must be positive")
    if args.max_height_error_mm <= 0.0 or args.max_xy_error_mm <= 0.0:
        raise ValueError("quality thresholds must be positive")
    if not 0 <= args.hue_min < args.hue_max <= 179:
        raise ValueError("HSV hue bounds must satisfy 0 <= min < max <= 179")
    if not 0 <= args.min_saturation <= 255 or not 0 <= args.min_value <= 255:
        raise ValueError("HSV saturation/value bounds must be in [0, 255]")
    if args.erosion_px < 1 or args.erosion_px % 2 == 0:
        raise ValueError("--erosion-px must be a positive odd integer")
    if args.min_component_area_px < 1 or args.min_foreground_points < 1:
        raise ValueError("minimum component/point counts must be positive")
    if not 0.0 <= args.foreground_percentile <= 50.0:
        raise ValueError("--foreground-percentile must be in [0, 50]")
    if args.foreground_band_mm <= 0.0:
        raise ValueError("--foreground-band-mm must be positive")
    if (args.expected_base_x_mm is None) != (args.expected_base_y_mm is None):
        raise ValueError("expected base X and Y must be supplied together")
    return args


def largest_hsv_component(
    image_bgr: np.ndarray,
    hsv_lower: Sequence[int],
    hsv_upper: Sequence[int],
    erosion_px: int,
    roi: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], int]:
    """Return an eroded largest-component mask, centroid, bbox, and raw area."""

    import cv2

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be an HxWx3 uint8 image")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    threshold = cv2.inRange(
        hsv,
        np.asarray(hsv_lower, dtype=np.uint8),
        np.asarray(hsv_upper, dtype=np.uint8),
    )
    if roi is not None:
        x0, y0, x1, y1 = (int(value) for value in roi)
        height, width = threshold.shape
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("ROI must be inside the image")
        roi_mask = np.zeros_like(threshold)
        roi_mask[y0:y1, x0:x1] = 255
        threshold = cv2.bitwise_and(threshold, roi_mask)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold)
    if count <= 1:
        raise RuntimeError("no colored target component was detected")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    x, y, width, height = (
        int(value) for value in stats[component, :4]
    )
    raw_mask = (labels == component).astype(np.uint8)
    kernel = np.ones((erosion_px, erosion_px), dtype=np.uint8)
    eroded = cv2.erode(raw_mask, kernel)
    if not np.any(eroded):
        raise RuntimeError("target mask disappeared after erosion")
    return eroded.astype(bool), centroids[component], (x, y, width, height), area


def nearest_depth_cluster(
    depth_u16: np.ndarray,
    mask: np.ndarray,
    depth_scale_m: float,
    percentile: float,
    band_mm: float,
) -> tuple[float, int, int, float]:
    """Estimate surface depth while rejecting background values in depth holes."""

    depth = np.asarray(depth_u16)
    selected_mask = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or depth.dtype != np.uint16 or depth.shape != selected_mask.shape:
        raise ValueError("depth_u16 and mask must be same-size 2D arrays")
    if depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be positive")
    values_m = depth[selected_mask].astype(float) * depth_scale_m
    valid_m = values_m[values_m > 0.0]
    if not len(valid_m):
        raise RuntimeError("the colored target contains no valid aligned depth")
    seed_m = float(np.percentile(valid_m, percentile))
    foreground_m = valid_m[np.abs(valid_m - seed_m) <= band_mm / 1000.0]
    if not len(foreground_m):
        raise RuntimeError("no depth samples remained in the foreground cluster")
    depth_m = float(np.median(foreground_m))
    mad_mm = 1000.0 * float(np.median(np.abs(foreground_m - depth_m)))
    return depth_m, len(valid_m), len(foreground_m), mad_mm


def pixel_depth_to_base(
    pixel_xy: Sequence[float],
    depth_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    base_from_camera: np.ndarray,
) -> np.ndarray:
    """Deproject an aligned color pixel and transform it into the UR base frame."""

    import cv2

    if depth_m <= 0.0:
        raise ValueError("depth_m must be positive")
    normalized = cv2.undistortPoints(
        np.asarray(pixel_xy, dtype=np.float32).reshape(1, 1, 2),
        np.asarray(camera_matrix, dtype=float),
        np.asarray(distortion, dtype=float),
    ).reshape(2)
    point_camera = depth_m * np.array(
        [normalized[0], normalized[1], 1.0], dtype=float
    )
    transform = np.asarray(base_from_camera, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("base_from_camera must be 4x4")
    return transform[:3, :3] @ point_camera + transform[:3, 3]


def plane_z_at_xy_mm(
    normal_base: Sequence[float],
    z_at_base_origin_mm: float,
    x_mm: float,
    y_mm: float,
) -> float:
    """Evaluate a plane using an independently anchored Z at base XY origin."""

    normal = np.asarray(normal_base, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("normal_base must contain three finite values")
    if normal[2] < 0.0:
        normal = -normal
    if abs(normal[2]) < 1e-9:
        raise ValueError("desk plane is parallel to the base Z axis")
    return float(
        z_at_base_origin_mm
        - (normal[0] * x_mm + normal[1] * y_mm) / normal[2]
    )


def _write_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def analyze(args: argparse.Namespace) -> dict[str, object]:
    import cv2

    calibration_path = args.calibration.resolve()
    capture_dir = args.capture.resolve()
    output_dir = (
        args.out.resolve()
        if args.out is not None
        else capture_dir / "colored_target_analysis"
    )
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    check = yaml.safe_load(
        (capture_dir / "camera_check.yaml").read_text(encoding="utf-8")
    )
    rgb_path = capture_dir / "camera_check_rgb.png"
    depth_path = capture_dir / "camera_check_depth_u16.png"
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if image is None or depth is None:
        raise OSError("failed to read saved aligned RGB-D images")
    mask, centroid, bbox, component_area = largest_hsv_component(
        image,
        (args.hue_min, args.min_saturation, args.min_value),
        (args.hue_max, 255, 255),
        args.erosion_px,
        args.roi,
    )
    if component_area < args.min_component_area_px:
        raise RuntimeError(
            f"colored component area {component_area} px is below the "
            f"{args.min_component_area_px} px minimum"
        )
    depth_scale_m = float(check["camera"]["depth"]["scale_m_per_unit"])
    target_depth_m, valid_depth_points, foreground_points, depth_mad_mm = (
        nearest_depth_cluster(
            depth,
            mask,
            depth_scale_m,
            args.foreground_percentile,
            args.foreground_band_mm,
        )
    )
    if foreground_points < args.min_foreground_points:
        raise RuntimeError(
            f"only {foreground_points} foreground depth points; "
            f"need at least {args.min_foreground_points}"
        )
    intrinsics = calibration["selected_intrinsics"]
    base_point_m = pixel_depth_to_base(
        centroid,
        target_depth_m,
        np.asarray(intrinsics["camera_matrix"], dtype=float),
        np.asarray(intrinsics["distortion_coefficients"], dtype=float),
        np.asarray(
            calibration["transforms"]["base_from_camera"]["matrix"], dtype=float
        ),
    )
    base_point_mm = 1000.0 * base_point_m
    desk_normal = np.array([0.0, 0.0, 1.0])
    desk_plane_path = None
    if args.desk_plane is not None:
        desk_plane_path = args.desk_plane.resolve()
        desk_plane = yaml.safe_load(desk_plane_path.read_text(encoding="utf-8"))
        desk_normal = np.asarray(desk_plane["plane"]["normal_base"], dtype=float)
    desk_z_at_target_mm = plane_z_at_xy_mm(
        desk_normal,
        args.desk_z_mm,
        float(base_point_mm[0]),
        float(base_point_mm[1]),
    )
    expected_top_z_mm = desk_z_at_target_mm + args.target_height_mm
    height_error_mm = float(base_point_mm[2] - expected_top_z_mm)
    absolute_height_error_mm = abs(height_error_mm)
    xy_error_mm = None
    if args.expected_base_x_mm is not None:
        xy_error_mm = float(
            np.linalg.norm(
                base_point_mm[:2]
                - np.array(
                    [args.expected_base_x_mm, args.expected_base_y_mm], dtype=float
                )
            )
        )
    height_quality_pass = absolute_height_error_mm <= args.max_height_error_mm
    xy_quality_pass = None if xy_error_mm is None else xy_error_mm <= args.max_xy_error_mm
    quality_pass = bool(height_quality_pass and xy_quality_pass is not False)
    x, y, width, height = bbox
    row: dict[str, object] = {
        "target_center_pixel_x": float(centroid[0]),
        "target_center_pixel_y": float(centroid[1]),
        "component_area_px": component_area,
        "eroded_mask_area_px": int(np.count_nonzero(mask)),
        "valid_depth_points": valid_depth_points,
        "foreground_depth_points": foreground_points,
        "foreground_depth_m": target_depth_m,
        "foreground_depth_mad_mm": depth_mad_mm,
        "estimated_base_x_mm": float(base_point_mm[0]),
        "estimated_base_y_mm": float(base_point_mm[1]),
        "estimated_base_z_mm": float(base_point_mm[2]),
        "desk_z_mm": args.desk_z_mm,
        "desk_z_at_target_mm": desk_z_at_target_mm,
        "target_height_mm": args.target_height_mm,
        "expected_top_z_mm": expected_top_z_mm,
        "signed_height_error_mm": height_error_mm,
        "absolute_height_error_mm": absolute_height_error_mm,
        "expected_base_x_mm": (
            "" if args.expected_base_x_mm is None else args.expected_base_x_mm
        ),
        "expected_base_y_mm": (
            "" if args.expected_base_y_mm is None else args.expected_base_y_mm
        ),
        "xy_error_mm": "" if xy_error_mm is None else xy_error_mm,
        "height_quality_pass": int(height_quality_pass),
        "xy_quality_pass": "" if xy_quality_pass is None else int(xy_quality_pass),
        "quality_pass": int(quality_pass),
        "hardware_motion_commands_sent": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(output_dir / "summary.csv", row)
    payload = {
        "schema": "rh56_ur5_camera_colored_target_validation/v1",
        "calibration": str(calibration_path),
        "capture": str(capture_dir),
        "segmentation": {
            "hsv_lower": [args.hue_min, args.min_saturation, args.min_value],
            "hsv_upper": [args.hue_max, 255, 255],
            "erosion_px": args.erosion_px,
            "roi_xyxy": None if args.roi is None else list(args.roi),
            "bbox_xywh": [x, y, width, height],
            "component_area_px": component_area,
            "centroid_pixel_xy": centroid.tolist(),
        },
        "depth_foreground": {
            "selection_percentile": args.foreground_percentile,
            "band_mm": args.foreground_band_mm,
            "estimated_depth_m": target_depth_m,
            "mad_mm": depth_mad_mm,
            "valid_points": valid_depth_points,
            "selected_points": foreground_points,
        },
        "estimated_target_top_center_base_mm": base_point_mm.tolist(),
        "physical_reference": {
            "desk_z_at_base_xy_origin_mm": args.desk_z_mm,
            "desk_plane_validation": (
                None if desk_plane_path is None else str(desk_plane_path)
            ),
            "desk_normal_base": desk_normal.tolist(),
            "desk_z_at_target_xy_mm": desk_z_at_target_mm,
            "target_height_mm": args.target_height_mm,
            "expected_top_z_mm": expected_top_z_mm,
            "expected_base_xy_mm": (
                None
                if args.expected_base_x_mm is None
                else [args.expected_base_x_mm, args.expected_base_y_mm]
            ),
        },
        "metrics": row,
        "quality_thresholds": {
            "max_absolute_height_error_mm": args.max_height_error_mm,
            "max_xy_error_mm": args.max_xy_error_mm,
            "min_component_area_px": args.min_component_area_px,
            "min_foreground_points": args.min_foreground_points,
        },
        "quality_pass": quality_pass,
        "assumptions": [
            "The colored component is the intended validation target.",
            "The target top center projects to the color-mask centroid.",
            "The nearest coherent depth cluster belongs to the target top, while farther values are depth-hole background.",
            "Desk Z and target height are independent physical measurements in the UR base frame.",
            (
                "The saved desk-plane normal accounts for local desk height; its "
                "absolute offset is anchored by the independent --desk-z-mm value."
                if desk_plane_path is not None
                else "No desk-plane normal was supplied, so the desk is assumed "
                "parallel to the UR base XY plane."
            ),
            "Height validation alone does not establish absolute XY accuracy.",
        ],
        "safety": {"offline_analysis": True, "hardware_motion_commands_sent": False},
    }
    (output_dir / "target_validation.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )

    annotated = image.copy()
    cv2.rectangle(annotated, (x, y), (x + width - 1, y + height - 1), (40, 220, 40), 3)
    center_int = tuple(int(round(value)) for value in centroid)
    cv2.drawMarker(
        annotated,
        center_int,
        (255, 30, 30),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=3,
    )
    labels = [
        "base XYZ = " + ", ".join(f"{value:.1f}" for value in base_point_mm) + " mm",
        f"expected Z = {expected_top_z_mm:.1f} mm; error = {height_error_mm:+.1f} mm",
        f"quality pass = {quality_pass}",
    ]
    for index, label in enumerate(labels):
        cv2.putText(
            annotated,
            label,
            (max(15, x - 100), max(35, y - 70 + 30 * index)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (40, 220, 40) if quality_pass else (20, 20, 240),
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_dir / "target_validation.png"), annotated):
        raise OSError("failed to write target validation annotation")

    print(f"Colored-target quality pass: {quality_pass}")
    print(
        "Estimated target top center in UR base: "
        + ", ".join(f"{value:.3f} mm" for value in base_point_mm)
    )
    print(
        f"Expected top Z={expected_top_z_mm:.3f} mm; "
        f"signed error={height_error_mm:+.3f} mm"
    )
    if xy_error_mm is None:
        print("Absolute XY reference: not supplied; XY is reported but not quality-gated")
    else:
        print(f"Absolute XY error={xy_error_mm:.3f} mm")
    print(f"Result: {output_dir / 'target_validation.yaml'}")
    print("Hardware motion commands sent: no")
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(args)
    return 0 if summary["quality_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
