#!/usr/bin/env python3
"""Calibrate and use a rigid point probe through RTDE receive-only reads.

This utility never imports ``rtde_control`` and cannot command robot motion.
The operator manually places the probe while the tool records stationary
``getActualTCPPose`` samples.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rh56_controller.camera_calibration import (  # noqa: E402
    average_transforms,
    pose_vector_xyz_rotvec_to_transform,
    rotation_error_deg,
)
from rh56_controller.pointer_calibration import (  # noqa: E402
    pointer_point_in_base,
    solve_pointer_calibration,
)


DEFAULT_OUT = REPO_ROOT / "artifacts/ur_pointer_calibration/calibration_pointer"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a rigid pointer or survey a point using stationary UR "
            "RTDE receive-only samples. No motion command is available."
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--capture-reference",
        metavar="LABEL",
        help="Append one pose while the pointer touches the common fixed reference point.",
    )
    action.add_argument(
        "--solve",
        action="store_true",
        help="Solve the pointer offset from previously captured reference poses.",
    )
    action.add_argument(
        "--capture-point",
        metavar="LABEL",
        help="Use the solved pointer offset to append one surveyed base-frame point.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ur-ip", default="192.168.0.4")
    parser.add_argument("--stationary-delay-s", type=float, default=0.25)
    parser.add_argument("--motion-translation-mm", type=float, default=0.2)
    parser.add_argument("--motion-rotation-deg", type=float, default=0.05)
    parser.add_argument("--max-loo-rms-mm", type=float, default=1.0)
    parser.add_argument("--max-loo-error-mm", type=float, default=2.0)
    parser.add_argument("--min-rotation-span-deg", type=float, default=30.0)
    parser.add_argument(
        "--allow-failed-calibration",
        action="store_true",
        help="Allow point surveying even if pointer quality thresholds failed.",
    )
    args = parser.parse_args(argv)
    for name in (
        "stationary_delay_s",
        "motion_translation_mm",
        "motion_rotation_deg",
        "max_loo_rms_mm",
        "max_loo_error_mm",
        "min_rotation_span_deg",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return args


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_yaml(path: Path, schema: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported schema in {path}")
    return payload


def _save_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _open_receiver(ip: str):
    try:
        import rtde_receive
    except ImportError as exc:
        raise RuntimeError("ur-rtde is required; install real-ur5-vision") from exc
    # Safety invariant: this file never imports rtde_control.
    return rtde_receive.RTDEReceiveInterface(ip)


def read_stationary_pose(args: argparse.Namespace) -> dict[str, object]:
    """Read the reported TCP twice and reject a moving-robot sample."""

    receiver = _open_receiver(args.ur_ip)
    try:
        before_vector = np.asarray(receiver.getActualTCPPose(), dtype=float)
        before_speed = np.asarray(receiver.getActualTCPSpeed(), dtype=float)
        time.sleep(args.stationary_delay_s)
        after_vector = np.asarray(receiver.getActualTCPPose(), dtype=float)
        after_speed = np.asarray(receiver.getActualTCPSpeed(), dtype=float)
    finally:
        receiver.disconnect()
    before = pose_vector_xyz_rotvec_to_transform(before_vector)
    after = pose_vector_xyz_rotvec_to_transform(after_vector)
    translation_change_mm = 1000.0 * float(
        np.linalg.norm(after[:3, 3] - before[:3, 3])
    )
    rotation_change_deg = rotation_error_deg(before, after)
    maximum_speed_norm = float(
        max(np.linalg.norm(before_speed), np.linalg.norm(after_speed))
    )
    if translation_change_mm > args.motion_translation_mm:
        raise RuntimeError(
            f"robot moved {translation_change_mm:.3f} mm during read; sample rejected"
        )
    if rotation_change_deg > args.motion_rotation_deg:
        raise RuntimeError(
            f"robot rotated {rotation_change_deg:.4f} deg during read; sample rejected"
        )
    return {
        "captured_at": datetime.now().astimezone().isoformat(),
        "base_from_reported_tcp": average_transforms([before, after]).tolist(),
        "reported_tcp_pose_before": before_vector.tolist(),
        "reported_tcp_pose_after": after_vector.tolist(),
        "translation_change_mm": translation_change_mm,
        "rotation_change_deg": rotation_change_deg,
        "maximum_reported_speed_norm": maximum_speed_norm,
    }


def _reference_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for index, sample in enumerate(payload["samples"]):
        pose = np.asarray(sample["base_from_reported_tcp"], dtype=float)
        row: dict[str, object] = {
            "index": index,
            "label": sample["label"],
            "captured_at": sample["captured_at"],
            "translation_change_mm": sample["translation_change_mm"],
            "rotation_change_deg": sample["rotation_change_deg"],
            "hardware_motion_commands_sent": 0,
        }
        row.update(
            {
                f"T_base_reported_tcp_r{row_index}c{column_index}": float(
                    pose[row_index, column_index]
                )
                for row_index in range(4)
                for column_index in range(4)
            }
        )
        rows.append(row)
    return rows


def capture_reference(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "reference_poses.yaml"
    if path.exists():
        payload = _load_yaml(path, "rh56_ur_pointer_reference_poses/v1")
        if payload["ur_ip"] != args.ur_ip:
            raise ValueError("--ur-ip does not match the existing session")
    else:
        payload = {
            "schema": "rh56_ur_pointer_reference_poses/v1",
            "created_at": datetime.now().astimezone().isoformat(),
            "ur_ip": args.ur_ip,
            "reported_frame": "active TCP reported by getActualTCPPose",
            "samples": [],
            "safety": {"hardware_motion_commands_sent": False, "rtde_interface": "receive-only"},
            "assumptions": [
                "The pointer is rigid relative to the reported active TCP frame.",
                "The active TCP configuration is unchanged for all reference and surveyed points.",
                "The pointer touches exactly the same fixed physical point in every reference pose.",
                "The operator positions the robot manually and stops before each read.",
            ],
        }
    label = str(args.capture_reference)
    if any(sample["label"] == label for sample in payload["samples"]):
        raise ValueError(f"reference label already exists: {label}")
    sample = read_stationary_pose(args)
    sample["label"] = label
    payload["samples"].append(sample)
    _save_yaml(path, payload)
    _write_csv(output_dir / "reference_poses.csv", _reference_rows(payload))
    _write_csv(
        output_dir / "summary.csv",
        [
            {
                "stage": "reference_capture",
                "reference_samples": len(payload["samples"]),
                "latest_label": label,
                "latest_translation_change_mm": sample["translation_change_mm"],
                "latest_rotation_change_deg": sample["rotation_change_deg"],
                "hardware_motion_commands_sent": 0,
            }
        ],
    )
    print(f"Saved reference pose {label}; total={len(payload['samples'])}")
    print("Motion commands sent: no")


def solve(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    references = _load_yaml(
        output_dir / "reference_poses.yaml", "rh56_ur_pointer_reference_poses/v1"
    )
    poses = [
        np.asarray(sample["base_from_reported_tcp"], dtype=float)
        for sample in references["samples"]
    ]
    result = solve_pointer_calibration(poses)
    quality_pass = bool(
        result.leave_one_out_rms_mm <= args.max_loo_rms_mm
        and result.leave_one_out_max_mm <= args.max_loo_error_mm
        and result.maximum_rotation_span_deg >= args.min_rotation_span_deg
    )
    summary = {
        "stage": "pointer_solve",
        "reference_samples": len(poses),
        "fit_residual_rms_mm": result.residual_rms_mm,
        "fit_residual_max_mm": result.residual_max_mm,
        "leave_one_out_rms_mm": result.leave_one_out_rms_mm,
        "leave_one_out_max_mm": result.leave_one_out_max_mm,
        "maximum_rotation_span_deg": result.maximum_rotation_span_deg,
        "linear_system_rank": result.linear_system_rank,
        "linear_system_condition": result.linear_system_condition,
        "quality_pass": int(quality_pass),
        "hardware_motion_commands_sent": 0,
    }
    payload = {
        "schema": "rh56_ur_pointer_calibration/v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "reported_frame": references["reported_frame"],
        "pointer_in_reported_tcp_m": result.pointer_in_reported_tcp_m.tolist(),
        "fitted_reference_point_base_m": result.reference_point_base_m.tolist(),
        "residuals_mm": result.residuals_mm.tolist(),
        "leave_one_out_errors_mm": result.leave_one_out_errors_mm.tolist(),
        "metrics": summary,
        "quality_thresholds": {
            "max_leave_one_out_rms_mm": args.max_loo_rms_mm,
            "max_leave_one_out_error_mm": args.max_loo_error_mm,
            "min_rotation_span_deg": args.min_rotation_span_deg,
        },
        "quality_pass": quality_pass,
        "safety": {"hardware_motion_commands_sent": False, "rtde_interface": "receive-only"},
        "assumptions": references["assumptions"],
    }
    _save_yaml(output_dir / "pointer_calibration.yaml", payload)
    _write_csv(output_dir / "summary.csv", [summary])
    print(f"Pointer quality pass: {quality_pass}")
    print(
        f"Leave-one-out: RMS={result.leave_one_out_rms_mm:.3f} mm, "
        f"max={result.leave_one_out_max_mm:.3f} mm"
    )
    print(f"Maximum orientation span: {result.maximum_rotation_span_deg:.1f} deg")
    print(f"Pointer offset in reported TCP [m]: {result.pointer_in_reported_tcp_m}")
    return summary


def _survey_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "label": point["label"],
            "x_base_m": point["point_base_m"][0],
            "y_base_m": point["point_base_m"][1],
            "z_base_m": point["point_base_m"][2],
            "captured_at": point["captured_at"],
            "translation_change_mm": point["translation_change_mm"],
            "rotation_change_deg": point["rotation_change_deg"],
            "hardware_motion_commands_sent": 0,
        }
        for index, point in enumerate(payload["points"])
    ]


def capture_point(args: argparse.Namespace, output_dir: Path) -> None:
    calibration = _load_yaml(
        output_dir / "pointer_calibration.yaml", "rh56_ur_pointer_calibration/v1"
    )
    if not calibration["quality_pass"] and not args.allow_failed_calibration:
        raise RuntimeError(
            "pointer calibration did not pass; repeat reference captures or use "
            "--allow-failed-calibration explicitly"
        )
    path = output_dir / "surveyed_points.yaml"
    if path.exists():
        payload = _load_yaml(path, "rh56_ur_pointer_survey/v1")
    else:
        payload = {
            "schema": "rh56_ur_pointer_survey/v1",
            "created_at": datetime.now().astimezone().isoformat(),
            "points": [],
            "safety": {"hardware_motion_commands_sent": False, "rtde_interface": "receive-only"},
        }
    label = str(args.capture_point)
    if any(point["label"] == label for point in payload["points"]):
        raise ValueError(f"survey point label already exists: {label}")
    sample = read_stationary_pose(args)
    point_base = pointer_point_in_base(
        np.asarray(sample["base_from_reported_tcp"], dtype=float),
        calibration["pointer_in_reported_tcp_m"],
    )
    sample["label"] = label
    sample["point_base_m"] = point_base.tolist()
    payload["points"].append(sample)
    _save_yaml(path, payload)
    _write_csv(output_dir / "surveyed_points.csv", _survey_rows(payload))
    print(f"Saved {label}: base XYZ [m] = {point_base}")
    print("Motion commands sent: no")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("RTDE receive-only: no robot or RH56 motion command is available.")
    print("Keep the active TCP configuration unchanged throughout this session.")
    if args.capture_reference is not None:
        capture_reference(args, output_dir)
        return 0
    if args.solve:
        summary = solve(args, output_dir)
        return 0 if summary["quality_pass"] else 2
    capture_point(args, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
