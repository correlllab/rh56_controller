#!/usr/bin/env python3
"""Fit a desk plane from saved aligned depth and tape it to a UR base frame."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "artifacts/ur5_external_camera_calibration_real"
    / "hand_eye_9x8_12mm_20260903/camera_calibration.yaml"
)
DEFAULT_CAPTURE = (
    REPO_ROOT / "artifacts/ur5_external_camera_calibration_real/desk_validation"
)


@dataclass(frozen=True)
class PlaneFit:
    point_base_m: np.ndarray
    normal_base: np.ndarray
    inlier_mask: np.ndarray
    inlier_residuals_m: np.ndarray

    @property
    def residual_rms_mm(self) -> float:
        return 1000.0 * float(np.sqrt(np.mean(np.square(self.inlier_residuals_m))))

    @property
    def residual_median_abs_mm(self) -> float:
        return 1000.0 * float(np.median(np.abs(self.inlier_residuals_m)))

    @property
    def residual_p95_abs_mm(self) -> float:
        return 1000.0 * float(np.percentile(np.abs(self.inlier_residuals_m), 95.0))

    @property
    def tilt_from_base_z_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.normal_base[2], -1.0, 1.0))))

    @property
    def z_at_base_origin_m(self) -> float:
        if abs(self.normal_base[2]) < 1e-9:
            raise ValueError("plane is parallel to the base Z axis")
        return float(np.dot(self.normal_base, self.point_base_m) / self.normal_base[2])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a desk plane from a saved aligned RealSense depth frame and a "
            "T_base_camera calibration. This is offline and sends no hardware command."
        )
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=(300, 50, 1150, 650),
        help="Half-open RGB/depth pixel ROI containing only the desk.",
    )
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.54)
    parser.add_argument("--max-depth-m", type=float, default=0.64)
    parser.add_argument(
        "--clip-mm", type=float, nargs="+", default=(5.0, 3.0, 2.5)
    )
    parser.add_argument("--max-rms-mm", type=float, default=2.0)
    parser.add_argument("--max-tilt-deg", type=float, default=2.0)
    parser.add_argument("--min-inlier-fraction", type=float, default=0.5)
    parser.add_argument(
        "--measured-desk-z-mm",
        type=float,
        default=None,
        help=(
            "Independent physical desk height relative to the UR Base z=0 plane. "
            "Positive is along Base +Z."
        ),
    )
    parser.add_argument(
        "--max-height-error-mm",
        type=float,
        default=5.0,
        help="Maximum absolute visual-versus-physical desk-height error.",
    )
    args = parser.parse_args(argv)
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if args.min_depth_m <= 0.0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("depth range must be positive and increasing")
    if any(value <= 0.0 for value in args.clip_mm):
        raise ValueError("--clip-mm values must be positive")
    if (
        args.max_rms_mm <= 0.0
        or args.max_tilt_deg <= 0.0
        or args.max_height_error_mm <= 0.0
    ):
        raise ValueError("quality thresholds must be positive")
    if not 0.0 < args.min_inlier_fraction <= 1.0:
        raise ValueError("--min-inlier-fraction must be in (0, 1]")
    return args


def deproject_aligned_depth(
    depth_u16: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    depth_scale_m: float,
    roi: Sequence[int],
    stride: int,
    depth_range_m: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-frame XYZ and corresponding pixels from aligned depth."""

    import cv2

    depth = np.asarray(depth_u16)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("depth_u16 must be a two-dimensional uint16 array")
    x0, y0, x1, y1 = (int(value) for value in roi)
    height, width = depth.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("ROI must be inside the depth image")
    yy, xx = np.mgrid[y0:y1:stride, x0:x1:stride]
    z_m = depth[yy, xx].astype(float) * depth_scale_m
    valid = (z_m > depth_range_m[0]) & (z_m < depth_range_m[1])
    pixels = np.column_stack((xx[valid], yy[valid])).astype(np.float32)
    if len(pixels) < 3:
        raise ValueError("fewer than three valid depth samples in the selected ROI")
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=float),
        np.asarray(distortion, dtype=float),
    ).reshape(-1, 2)
    z_valid = z_m[valid]
    points_camera = np.column_stack(
        (normalized[:, 0] * z_valid, normalized[:, 1] * z_valid, z_valid)
    )
    return points_camera, pixels


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform_array = np.asarray(transform, dtype=float)
    point_array = np.asarray(points, dtype=float)
    if transform_array.shape != (4, 4) or point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("expected a 4x4 transform and an Nx3 point array")
    return (transform_array[:3, :3] @ point_array.T).T + transform_array[:3, 3]


def fit_plane_robust(points_base_m: np.ndarray, clip_mm: Sequence[float]) -> PlaneFit:
    """Fit a plane with repeated absolute-residual clipping."""

    points = np.asarray(points_base_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points_base_m must contain at least three XYZ points")
    active = np.ones(len(points), dtype=bool)
    for threshold_mm in clip_mm:
        active_points = points[active]
        if len(active_points) < 3:
            raise ValueError("plane clipping left fewer than three points")
        center = active_points.mean(axis=0)
        _u, _s, vh = np.linalg.svd(active_points - center, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0.0:
            normal = -normal
        residuals = (points - center) @ normal
        active = np.abs(residuals) < float(threshold_mm) / 1000.0
    active_points = points[active]
    center = active_points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(active_points - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    residuals = (active_points - center) @ normal
    return PlaneFit(center, normal, active, residuals)


def _write_csv(path: Path, row: dict[str, object]) -> None:
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
        args.out.resolve() if args.out is not None else capture_dir / "plane_analysis"
    )
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    check = yaml.safe_load(
        (capture_dir / "camera_check.yaml").read_text(encoding="utf-8")
    )
    depth = cv2.imread(
        str(capture_dir / "camera_check_depth_u16.png"), cv2.IMREAD_UNCHANGED
    )
    if depth is None:
        raise OSError("failed to read saved uint16 depth image")
    intrinsics = calibration["selected_intrinsics"]
    points_camera, pixels = deproject_aligned_depth(
        depth,
        np.asarray(intrinsics["camera_matrix"], dtype=float),
        np.asarray(intrinsics["distortion_coefficients"], dtype=float),
        float(check["camera"]["depth"]["scale_m_per_unit"]),
        args.roi,
        args.stride,
        (args.min_depth_m, args.max_depth_m),
    )
    points_base = transform_points(
        np.asarray(calibration["transforms"]["base_from_camera"]["matrix"], dtype=float),
        points_camera,
    )
    fit = fit_plane_robust(points_base, args.clip_mm)
    inlier_fraction = float(np.mean(fit.inlier_mask))
    plane_fit_quality_pass = bool(
        fit.residual_rms_mm <= args.max_rms_mm
        and fit.tilt_from_base_z_deg <= args.max_tilt_deg
        and inlier_fraction >= args.min_inlier_fraction
    )
    measured_desk_z_mm = args.measured_desk_z_mm
    absolute_height_error_mm = (
        None
        if measured_desk_z_mm is None
        else abs(1000.0 * fit.z_at_base_origin_m - measured_desk_z_mm)
    )
    absolute_height_quality_pass = (
        None
        if absolute_height_error_mm is None
        else absolute_height_error_mm <= args.max_height_error_mm
    )
    quality_pass = bool(
        plane_fit_quality_pass
        and absolute_height_quality_pass is not False
    )
    summary: dict[str, object] = {
        "sampled_valid_points": len(points_base),
        "plane_inliers": int(np.count_nonzero(fit.inlier_mask)),
        "inlier_fraction": inlier_fraction,
        "residual_rms_mm": fit.residual_rms_mm,
        "residual_median_abs_mm": fit.residual_median_abs_mm,
        "residual_p95_abs_mm": fit.residual_p95_abs_mm,
        "normal_x_base": float(fit.normal_base[0]),
        "normal_y_base": float(fit.normal_base[1]),
        "normal_z_base": float(fit.normal_base[2]),
        "tilt_from_base_z_deg": fit.tilt_from_base_z_deg,
        "plane_z_at_base_origin_m": fit.z_at_base_origin_m,
        "measured_desk_z_mm": (
            "" if measured_desk_z_mm is None else measured_desk_z_mm
        ),
        "absolute_height_error_mm": (
            "" if absolute_height_error_mm is None else absolute_height_error_mm
        ),
        "plane_fit_quality_pass": int(plane_fit_quality_pass),
        "absolute_height_quality_pass": (
            "" if absolute_height_quality_pass is None else int(absolute_height_quality_pass)
        ),
        "quality_pass": int(quality_pass),
        "hardware_motion_commands_sent": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", summary)
    payload = {
        "schema": "rh56_ur5_camera_desk_plane_validation/v1",
        "calibration": str(calibration_path),
        "capture": str(capture_dir),
        "roi_xyxy": list(args.roi),
        "depth_range_m": [args.min_depth_m, args.max_depth_m],
        "sampling_stride_px": args.stride,
        "clip_thresholds_mm": list(args.clip_mm),
        "plane": {
            "point_base_m": fit.point_base_m.tolist(),
            "normal_base": fit.normal_base.tolist(),
            "z_at_base_xy_origin_m": fit.z_at_base_origin_m,
        },
        "metrics": summary,
        "quality_thresholds": {
            "max_residual_rms_mm": args.max_rms_mm,
            "max_tilt_from_base_z_deg": args.max_tilt_deg,
            "min_inlier_fraction": args.min_inlier_fraction,
            "max_absolute_height_error_mm": args.max_height_error_mm,
        },
        "physical_height_check": {
            "measured_desk_z_mm": measured_desk_z_mm,
            "estimated_desk_z_mm": 1000.0 * fit.z_at_base_origin_m,
            "absolute_error_mm": absolute_height_error_mm,
            "quality_pass": absolute_height_quality_pass,
        },
        "plane_fit_quality_pass": plane_fit_quality_pass,
        "quality_pass": quality_pass,
        "assumptions": [
            "The selected ROI contains the same approximately planar desk surface.",
            "The desk is expected to be approximately parallel to the UR base XY plane.",
            "Aligned depth pixels use the saved color-camera intrinsics.",
            (
                "The reported physical desk height is an independent approximate "
                "operator measurement."
                if measured_desk_z_mm is not None
                else "Plane height is not an absolute accuracy check until the "
                "base-to-desk offset is measured independently."
            ),
        ],
        "safety": {"offline_analysis": True, "hardware_motion_commands_sent": False},
    }
    (output_dir / "desk_plane_validation.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )

    rgb_path = capture_dir / "camera_check_rgb.png"
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is not None:
        x0, y0, x1, y1 = args.roi
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (40, 220, 40), 3)
        labels = [
            f"plane RMS={fit.residual_rms_mm:.2f} mm",
            f"tilt from base +Z={fit.tilt_from_base_z_deg:.2f} deg",
            f"z at base origin={1000.0 * fit.z_at_base_origin_m:.1f} mm",
        ]
        if absolute_height_error_mm is not None:
            labels.append(f"absolute height error={absolute_height_error_mm:.1f} mm")
        labels.append(f"quality pass={quality_pass}")
        for index, label in enumerate(labels):
            cv2.putText(
                image,
                label,
                (x0 + 15, y0 + 35 + 32 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (40, 220, 40) if quality_pass else (20, 20, 240),
                2,
                cv2.LINE_AA,
            )
        if not cv2.imwrite(str(output_dir / "desk_plane_roi.png"), image):
            raise OSError("failed to write desk-plane annotation")
    print(f"Desk-plane quality pass: {quality_pass}")
    print(
        f"RMS={fit.residual_rms_mm:.3f} mm, "
        f"p95={fit.residual_p95_abs_mm:.3f} mm, "
        f"tilt={fit.tilt_from_base_z_deg:.3f} deg"
    )
    print(f"Plane z at base XY origin: {fit.z_at_base_origin_m:.6f} m")
    if absolute_height_error_mm is not None:
        print(
            f"Physical desk z={measured_desk_z_mm:.1f} mm, "
            f"absolute height error={absolute_height_error_mm:.3f} mm"
        )
    print(f"Result: {output_dir / 'desk_plane_validation.yaml'}")
    print("Hardware motion commands sent: no")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(args)
    return 0 if summary["quality_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
