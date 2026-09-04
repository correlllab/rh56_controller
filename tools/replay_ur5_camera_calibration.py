#!/usr/bin/env python3
"""Replay a recorded UR TCP calibration path with explicit motion gating."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rh56_controller.camera_calibration import (  # noqa: E402
    average_transforms,
    detect_chessboard,
    pose_vector_xyz_rotvec_to_transform,
    rotation_error_deg,
)
from tools.calibrate_ur5_external_camera_real import (  # noqa: E402
    _annotate_detection,
    _camera_metadata_to_dict,
    _read_aligned_frame,
    _save_depth,
    _save_rgb,
    _start_camera,
    _write_printable_chessboard_svg,
    calibrate_dataset,
)


DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts/ur5_external_camera_calibration_replay"
)
CONFIRMATION_PHRASE = "MOVE UR5 THROUGH CAMERA POSES"


@dataclass(frozen=True)
class ReplayPose:
    reference_index: int
    tcp_pose_vector: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or explicitly execute a low-speed UR moveL replay of a saved "
            "camera-calibration TCP path. The default is offline dry-run only."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Existing capture_manifest.yaml whose TCP poses will be replayed.",
    )
    parser.add_argument(
        "--baseline-calibration",
        type=Path,
        default=None,
        help=(
            "Calibration used for the camera-drift comparison. Defaults to "
            "camera_calibration.yaml beside --reference."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pose-indices", type=int, nargs="+", default=None)
    parser.add_argument("--ur-ip", default=None)
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--speed-m-s", type=float, default=0.02)
    parser.add_argument("--accel-m-s2", type=float, default=0.05)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--capture-retries", type=int, default=3)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--settle-frames", type=int, default=5)
    parser.add_argument("--laser-power", type=float, default=None)
    parser.add_argument("--no-save-depth", action="store_true")
    parser.add_argument("--intrinsics-source", choices=("factory", "estimate"), default="estimate")
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--max-holdout-translation-mm", type=float, default=5.0)
    parser.add_argument("--max-holdout-rotation-deg", type=float, default=2.0)
    parser.add_argument("--max-holdout-reprojection-px", type=float, default=3.0)
    parser.add_argument("--max-camera-drift-mm", type=float, default=5.0)
    parser.add_argument("--max-camera-drift-deg", type=float, default=1.0)
    parser.add_argument("--min-tcp-z-m", type=float, default=0.10)
    parser.add_argument("--max-tcp-radius-m", type=float, default=0.80)
    parser.add_argument("--max-segment-mm", type=float, default=300.0)
    parser.add_argument("--max-segment-deg", type=float, default=75.0)
    parser.add_argument("--start-tolerance-mm", type=float, default=50.0)
    parser.add_argument("--start-tolerance-deg", type=float, default=20.0)
    parser.add_argument("--max-target-error-mm", type=float, default=3.0)
    parser.add_argument("--max-target-error-deg", type=float, default=2.0)
    parser.add_argument(
        "--expected-tcp-offset",
        type=float,
        nargs=6,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Expected active UR TCP offset in metres and rotation-vector radians.",
    )
    parser.add_argument(
        "--skip-start-proximity-check",
        action="store_true",
        help=(
            "Permit the first move from an arbitrary TCP pose. This removes an "
            "important default safety gate and requires the normal motion confirmation."
        ),
    )
    parser.add_argument(
        "--execute-motion",
        action="store_true",
        help=(
            "Connect RTDE control and execute motion after interactive confirmation. "
            "Without this flag the tool only writes an offline replay plan."
        ),
    )
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.speed_m_s <= 0.05:
        raise ValueError("--speed-m-s must be in (0, 0.05]")
    if not 0.0 < args.accel_m_s2 <= 0.10:
        raise ValueError("--accel-m-s2 must be in (0, 0.10]")
    positive = (
        args.settle_seconds,
        args.max_holdout_translation_mm,
        args.max_holdout_rotation_deg,
        args.max_holdout_reprojection_px,
        args.max_camera_drift_mm,
        args.max_camera_drift_deg,
        args.max_tcp_radius_m,
        args.max_segment_mm,
        args.max_segment_deg,
        args.start_tolerance_mm,
        args.start_tolerance_deg,
        args.max_target_error_mm,
        args.max_target_error_deg,
    )
    if any(value <= 0.0 for value in positive):
        raise ValueError("all safety and quality thresholds must be positive")
    if args.min_tcp_z_m < 0.0:
        raise ValueError("--min-tcp-z-m cannot be negative")
    if args.capture_retries < 1 or args.warmup_frames < 0 or args.settle_frames < 1:
        raise ValueError("capture counts must be positive and warmup cannot be negative")
    if args.holdout_every < 2:
        raise ValueError("--holdout-every must be at least 2")
    if args.laser_power is not None and args.laser_power < 0.0:
        raise ValueError("--laser-power cannot be negative")


def _default_output_dir() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_ARTIFACT_ROOT / stamp


def load_replay_poses(
    path: Path,
    pose_indices: Sequence[int] | None = None,
) -> tuple[dict[str, object], list[ReplayPose]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "rh56_ur5_external_camera_capture/v1":
        raise ValueError(f"unsupported reference manifest schema in {path}")
    entries = payload.get("captures", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("reference manifest contains no captures")
    selected = list(range(len(entries))) if pose_indices is None else list(pose_indices)
    if len(set(selected)) != len(selected):
        raise ValueError("--pose-indices cannot contain duplicates")
    if any(index < 0 or index >= len(entries) for index in selected):
        raise ValueError("--pose-indices contains an out-of-range index")
    poses: list[ReplayPose] = []
    for index in selected:
        vector = np.asarray(entries[index].get("tcp_pose_vector"), dtype=float)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"capture {index} has no finite six-value TCP pose")
        poses.append(ReplayPose(index, vector))
    if len(poses) < 5:
        raise ValueError("select at least five poses so calibration has a holdout")
    return payload, poses


def pose_error(
    actual_vector: Sequence[float],
    target_vector: Sequence[float],
) -> tuple[float, float]:
    actual = pose_vector_xyz_rotvec_to_transform(actual_vector)
    target = pose_vector_xyz_rotvec_to_transform(target_vector)
    translation_mm = 1000.0 * float(
        np.linalg.norm(actual[:3, 3] - target[:3, 3])
    )
    return translation_mm, rotation_error_deg(actual, target)


def trajectory_metrics(poses: Sequence[ReplayPose]) -> dict[str, float]:
    vectors = [pose.tcp_pose_vector for pose in poses]
    transforms = [pose_vector_xyz_rotvec_to_transform(vector) for vector in vectors]
    segment_translation_mm = [
        1000.0 * float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
        for first, second in zip(transforms[:-1], transforms[1:])
    ]
    segment_rotation_deg = [
        rotation_error_deg(first, second)
        for first, second in zip(transforms[:-1], transforms[1:])
    ]
    xyz = np.asarray([vector[:3] for vector in vectors])
    return {
        "pose_count": float(len(poses)),
        "minimum_tcp_z_m": float(np.min(xyz[:, 2])),
        "maximum_tcp_radius_m": float(np.max(np.linalg.norm(xyz[:, :2], axis=1))),
        "maximum_segment_translation_mm": float(max(segment_translation_mm)),
        "maximum_segment_rotation_deg": float(max(segment_rotation_deg)),
    }


def validate_trajectory(
    metrics: dict[str, float], args: argparse.Namespace
) -> list[str]:
    failures: list[str] = []
    if metrics["minimum_tcp_z_m"] < args.min_tcp_z_m:
        failures.append("minimum_tcp_z_below_limit")
    if metrics["maximum_tcp_radius_m"] > args.max_tcp_radius_m:
        failures.append("maximum_tcp_radius_above_limit")
    if metrics["maximum_segment_translation_mm"] > args.max_segment_mm:
        failures.append("segment_translation_above_limit")
    if metrics["maximum_segment_rotation_deg"] > args.max_segment_deg:
        failures.append("segment_rotation_above_limit")
    return failures


def calibration_drift(
    baseline_base_from_camera: np.ndarray,
    observed_base_from_camera: np.ndarray,
) -> tuple[float, float]:
    baseline = np.asarray(baseline_base_from_camera, dtype=float)
    observed = np.asarray(observed_base_from_camera, dtype=float)
    translation_mm = 1000.0 * float(
        np.linalg.norm(observed[:3, 3] - baseline[:3, 3])
    )
    return translation_mm, rotation_error_deg(baseline, observed)


def _write_one_row_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plan(
    output_dir: Path,
    args: argparse.Namespace,
    reference_path: Path,
    baseline_path: Path,
    poses: Sequence[ReplayPose],
    metrics: dict[str, float],
    failures: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, object] = {
        **metrics,
        "speed_m_s": args.speed_m_s,
        "accel_m_s2": args.accel_m_s2,
        "trajectory_preflight_pass": int(not failures),
        "execution_requested": int(args.execute_motion),
        "execution_completed": 0,
        "failure_reason": "",
        "hardware_motion_commands_sent": 0,
    }
    _write_one_row_csv(output_dir / "replay_summary.csv", row)
    if not args.execute_motion:
        _write_one_row_csv(output_dir / "summary.csv", row)
    payload = {
        "schema": "rh56_ur5_camera_calibration_replay_plan/v1",
        "reference_manifest": str(reference_path),
        "baseline_calibration": str(baseline_path),
        "pose_indices": [pose.reference_index for pose in poses],
        "tcp_pose_vectors": [pose.tcp_pose_vector.tolist() for pose in poses],
        "trajectory_metrics": metrics,
        "trajectory_limits": {
            "minimum_tcp_z_m": args.min_tcp_z_m,
            "maximum_tcp_radius_m": args.max_tcp_radius_m,
            "maximum_segment_translation_mm": args.max_segment_mm,
            "maximum_segment_rotation_deg": args.max_segment_deg,
        },
        "trajectory_preflight_failures": list(failures),
        "motion": {
            "command": "moveL",
            "speed_m_s": args.speed_m_s,
            "acceleration_m_s2": args.accel_m_s2,
            "execute_motion": args.execute_motion,
            "interactive_confirmation_phrase": CONFIRMATION_PHRASE,
        },
        "assumptions": [
            "The reference TCP poses and active TCP definition are unchanged.",
            "The chessboard is rigidly attached for the complete replay.",
            "The workspace and every straight TCP segment have been cleared by the operator.",
            "The operator watches the complete low-speed replay with stop controls available.",
            "The dry-run checks scalar bounds; it is not a collision certificate.",
        ],
        "safety": {"hardware_motion_commands_sent": False},
    }
    (output_dir / "replay_plan.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def _confirm_motion(poses: Sequence[ReplayPose]) -> bool:
    print("\nMOTION ENABLE REQUESTED")
    print(f"  TCP poses: {len(poses)}")
    print("  Confirm the board is rigid, the workspace is empty, and stop controls are ready.")
    try:
        answer = input(f"Type exactly '{CONFIRMATION_PHRASE}' to continue: ").strip()
    except EOFError:
        return False
    return answer == CONFIRMATION_PHRASE


def _runtime_settings(
    args: argparse.Namespace, reference: dict[str, object]
) -> tuple[tuple[int, int], float, str]:
    camera = reference["camera"]
    stream = camera["stream"]
    board = reference["chessboard"]
    args.width = int(args.width if args.width is not None else stream["width"])
    args.height = int(args.height if args.height is not None else stream["height"])
    args.fps = int(args.fps if args.fps is not None else stream["fps"])
    if args.camera_serial is None:
        args.camera_serial = str(camera["serial"])
    if args.ur_ip is None:
        args.ur_ip = str(reference.get("robot_pose", {}).get("rtde_ip") or "192.168.0.4")
    pattern_size = tuple(int(value) for value in board["inner_corners"])
    square_size_mm = 1000.0 * float(board["square_size_m"])
    return pattern_size, square_size_mm, str(args.camera_serial)


def _capture_manifest(
    args: argparse.Namespace,
    reference_path: Path,
    metadata,
    pattern_size: tuple[int, int],
    square_size_mm: float,
    captures: list[dict[str, object]],
    motion_command_count: int,
) -> dict[str, object]:
    return {
        "schema": "rh56_ur5_external_camera_capture/v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "real fixed external camera; explicit low-speed UR TCP replay",
        "camera": _camera_metadata_to_dict(metadata),
        "chessboard": {
            "inner_corners": list(pattern_size),
            "square_size_m": square_size_mm / 1000.0,
        },
        "robot_pose": {
            "source": "rtde_replay_actual_pose",
            "frame": "T_base_tcp",
            "rtde_ip": args.ur_ip,
            "vector_convention": "[x,y,z,rx,ry,rz], metres and axis-angle radians",
        },
        "replay": {
            "reference_manifest": str(reference_path),
            "command": "moveL",
            "speed_m_s": args.speed_m_s,
            "acceleration_m_s2": args.accel_m_s2,
        },
        "captures": captures,
        "safety": {
            "hardware_motion_commands_sent": motion_command_count > 0,
            "motion_command_count": motion_command_count,
            "interactive_confirmation_received": True,
            "operator_watches_motion": True,
        },
        "assumptions": [
            "The RealSense remains rigidly fixed during this replay.",
            "The chessboard remains rigid relative to the active UR TCP during this replay.",
            "The active TCP offset matches the explicitly checked expected offset.",
            "The operator cleared and watches every recorded straight TCP segment.",
        ],
    }


def execute_replay(
    args: argparse.Namespace,
    output_dir: Path,
    reference_path: Path,
    baseline_path: Path,
    reference: dict[str, object],
    poses: Sequence[ReplayPose],
    metrics: dict[str, float],
) -> int:
    try:
        import rtde_control
        import rtde_receive
    except ImportError as exc:
        raise RuntimeError("ur-rtde is required for --execute-motion") from exc

    if (output_dir / "capture_manifest.yaml").exists():
        raise FileExistsError(f"refusing to overwrite existing dataset {output_dir}")
    pattern_size, square_size_mm, _serial = _runtime_settings(args, reference)
    pipeline = None
    receiver = None
    control = None
    captures: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    motion_command_count = 0
    metadata = None
    failure_reason = ""
    try:
        pipeline, align, metadata = _start_camera(args)
        receiver = rtde_receive.RTDEReceiveInterface(args.ur_ip)
        control = rtde_control.RTDEControlInterface(args.ur_ip)
        if receiver.isProtectiveStopped():
            raise RuntimeError("UR reports a protective stop; no motion was sent")
        actual_tcp_offset = np.asarray(control.getTCPOffset(), dtype=float)
        expected_tcp_offset = np.asarray(args.expected_tcp_offset, dtype=float)
        if not np.allclose(actual_tcp_offset, expected_tcp_offset, atol=1e-6):
            raise RuntimeError(
                "active TCP offset does not match --expected-tcp-offset: "
                f"actual={actual_tcp_offset.tolist()} expected={expected_tcp_offset.tolist()}"
            )

        current_vector = np.asarray(receiver.getActualTCPPose(), dtype=float)
        start_translation_mm, start_rotation_deg = pose_error(
            current_vector, poses[0].tcp_pose_vector
        )
        if not args.skip_start_proximity_check and (
            start_translation_mm > args.start_tolerance_mm
            or start_rotation_deg > args.start_tolerance_deg
        ):
            raise RuntimeError(
                "current TCP is not near the first replay pose; manually position it "
                f"closer first (difference {start_translation_mm:.1f} mm, "
                f"{start_rotation_deg:.1f} deg)"
            )

        if not args.skip_start_proximity_check:
            preview_rgb, _preview_depth, _timestamp, _frame = _read_aligned_frame(
                pipeline, align, args.settle_frames
            )
            preview_corners = detect_chessboard(preview_rgb, pattern_size)
            _save_rgb(output_dir / "preflight_rgb.png", preview_rgb)
            _save_rgb(
                output_dir / "preflight_annotated.png",
                _annotate_detection(
                    preview_rgb,
                    pattern_size,
                    preview_corners,
                    "preflight board detected"
                    if preview_corners is not None
                    else "preflight board NOT detected",
                ),
            )
            if preview_corners is None:
                raise RuntimeError("chessboard was not detected at the near-start preflight pose")

        for replay_number, pose in enumerate(poses):
            if receiver.isProtectiveStopped():
                raise RuntimeError("UR entered protective stop; replay aborted")
            target = pose.tcp_pose_vector.tolist()
            print(
                f"Pose {replay_number + 1}/{len(poses)}: reference "
                f"{pose.reference_index}, moveL at {args.speed_m_s:.3f} m/s"
            )
            motion_command_count += 1
            if not control.moveL(target, args.speed_m_s, args.accel_m_s2, False):
                raise RuntimeError(f"UR rejected moveL for reference pose {pose.reference_index}")
            time.sleep(args.settle_seconds)
            reached_vector = np.asarray(receiver.getActualTCPPose(), dtype=float)
            target_translation_mm, target_rotation_deg = pose_error(reached_vector, target)
            if (
                target_translation_mm > args.max_target_error_mm
                or target_rotation_deg > args.max_target_error_deg
            ):
                raise RuntimeError(
                    f"pose {pose.reference_index} target error is too large: "
                    f"{target_translation_mm:.2f} mm, {target_rotation_deg:.2f} deg"
                )

            image_rgb = depth_u16 = corners = None
            timestamp_ms = frame_number = None
            retry_used = 0
            for retry in range(args.capture_retries):
                image_rgb, depth_u16, timestamp_ms, frame_number = _read_aligned_frame(
                    pipeline, align, args.settle_frames
                )
                corners = detect_chessboard(image_rgb, pattern_size)
                retry_used = retry
                if corners is not None:
                    break
            assert image_rgb is not None and depth_u16 is not None
            if corners is None:
                _save_rgb(
                    output_dir / "attempts" / f"pose_{pose.reference_index:03d}_failed.png",
                    _annotate_detection(
                        image_rgb, pattern_size, None, "chessboard NOT detected; aborting"
                    ),
                )
                raise RuntimeError(
                    f"chessboard not detected at reference pose {pose.reference_index}"
                )

            post_vector = np.asarray(receiver.getActualTCPPose(), dtype=float)
            pre_transform = pose_vector_xyz_rotvec_to_transform(reached_vector)
            post_transform = pose_vector_xyz_rotvec_to_transform(post_vector)
            base_from_tcp = average_transforms([pre_transform, post_transform])
            motion_translation_mm, motion_rotation_deg = pose_error(
                reached_vector, post_vector
            )
            stem = f"capture_{replay_number:03d}"
            rgb_rel = Path("captures") / f"{stem}_rgb.png"
            annotated_rel = Path("captures") / f"{stem}_annotated.png"
            depth_rel = Path("captures") / f"{stem}_depth_u16.png"
            annotated = _annotate_detection(
                image_rgb,
                pattern_size,
                corners,
                f"replay pose {pose.reference_index} accepted",
            )
            _save_rgb(output_dir / rgb_rel, image_rgb)
            _save_rgb(output_dir / annotated_rel, annotated)
            if not args.no_save_depth:
                _save_depth(output_dir / depth_rel, depth_u16)
            captures.append(
                {
                    "index": replay_number,
                    "reference_capture_index": pose.reference_index,
                    "rgb_path": str(rgb_rel),
                    "annotated_path": str(annotated_rel),
                    "depth_path": None if args.no_save_depth else str(depth_rel),
                    "camera_timestamp_ms": timestamp_ms,
                    "camera_frame_number": frame_number,
                    "base_from_tcp": base_from_tcp.tolist(),
                    "tcp_pose_vector": (0.5 * (reached_vector + post_vector)).tolist(),
                    "target_tcp_pose_vector": target,
                    "corners_px": corners.reshape(-1, 2).tolist(),
                    "target_translation_error_mm": target_translation_mm,
                    "target_rotation_error_deg": target_rotation_deg,
                    "motion_translation_mm": motion_translation_mm,
                    "motion_rotation_deg": motion_rotation_deg,
                    "capture_retry": retry_used,
                }
            )
            rows.append(
                {
                    "replay_number": replay_number,
                    "reference_capture_index": pose.reference_index,
                    "target_translation_error_mm": target_translation_mm,
                    "target_rotation_error_deg": target_rotation_deg,
                    "motion_during_capture_translation_mm": motion_translation_mm,
                    "motion_during_capture_rotation_deg": motion_rotation_deg,
                    "capture_retry": retry_used,
                    "hardware_motion_commands_sent": motion_command_count,
                }
            )
            manifest = _capture_manifest(
                args,
                reference_path,
                metadata,
                pattern_size,
                square_size_mm,
                captures,
                motion_command_count,
            )
            (output_dir / "capture_manifest.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )
            _write_rows_csv(output_dir / "capture_summary.csv", rows)
    except KeyboardInterrupt:
        failure_reason = "operator_interrupted_replay"
        if control is not None:
            control.stopL(0.5, False)
        raise RuntimeError("operator interrupted replay; stopL requested") from None
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if failure_reason:
            failure_row = {
                **metrics,
                "speed_m_s": args.speed_m_s,
                "accel_m_s2": args.accel_m_s2,
                "trajectory_preflight_pass": 1,
                "execution_requested": 1,
                "execution_completed": 0,
                "failure_reason": failure_reason,
                "hardware_motion_commands_sent": motion_command_count,
            }
            _write_one_row_csv(output_dir / "replay_summary.csv", failure_row)
            if metadata is not None:
                partial_manifest = _capture_manifest(
                    args,
                    reference_path,
                    metadata,
                    pattern_size,
                    square_size_mm,
                    captures,
                    motion_command_count,
                )
                partial_manifest["replay"]["execution_completed"] = False
                partial_manifest["replay"]["failure_reason"] = failure_reason
                (output_dir / "capture_manifest.yaml").write_text(
                    yaml.safe_dump(partial_manifest, sort_keys=False), encoding="utf-8"
                )
                _write_rows_csv(output_dir / "capture_summary.csv", rows)
        if pipeline is not None:
            pipeline.stop()
        if control is not None:
            try:
                control.stopScript()
            finally:
                control.disconnect()
        if receiver is not None:
            receiver.disconnect()

    assert metadata is not None
    _write_printable_chessboard_svg(
        output_dir / "printable_chessboard.svg", pattern_size, square_size_mm
    )
    calibration_summary = calibrate_dataset(args, output_dir)
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    observed_path = output_dir / "camera_calibration.yaml"
    observed = yaml.safe_load(observed_path.read_text(encoding="utf-8"))
    drift_translation_mm, drift_rotation_deg = calibration_drift(
        np.asarray(
            baseline["transforms"]["base_from_camera"]["matrix"], dtype=float
        ),
        np.asarray(
            observed["transforms"]["base_from_camera"]["matrix"], dtype=float
        ),
    )
    drift_pass = bool(
        drift_translation_mm <= args.max_camera_drift_mm
        and drift_rotation_deg <= args.max_camera_drift_deg
    )
    drift_row = {
        "camera_translation_drift_mm": drift_translation_mm,
        "camera_rotation_drift_deg": drift_rotation_deg,
        "max_camera_drift_mm": args.max_camera_drift_mm,
        "max_camera_drift_deg": args.max_camera_drift_deg,
        "camera_drift_quality_pass": int(drift_pass),
        "hardware_motion_commands_sent": motion_command_count,
    }
    _write_one_row_csv(output_dir / "drift_summary.csv", drift_row)
    combined_summary = {**calibration_summary, **drift_row}
    combined_summary["overall_quality_pass"] = int(
        bool(calibration_summary["quality_pass"]) and drift_pass
    )
    _write_one_row_csv(output_dir / "summary.csv", combined_summary)
    (output_dir / "camera_drift.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "rh56_ur5_camera_drift_check/v1",
                "baseline_calibration": str(baseline_path),
                "observed_calibration": str(observed_path),
                "translation_drift_mm": drift_translation_mm,
                "rotation_drift_deg": drift_rotation_deg,
                "quality_thresholds": {
                    "max_translation_drift_mm": args.max_camera_drift_mm,
                    "max_rotation_drift_deg": args.max_camera_drift_deg,
                },
                "quality_pass": drift_pass,
                "hardware_motion_commands_sent": motion_command_count,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    replay_row = {
        **metrics,
        "speed_m_s": args.speed_m_s,
        "accel_m_s2": args.accel_m_s2,
        "trajectory_preflight_pass": 1,
        "execution_requested": 1,
        "execution_completed": 1,
        "failure_reason": "",
        "hardware_motion_commands_sent": motion_command_count,
    }
    _write_one_row_csv(output_dir / "replay_summary.csv", replay_row)
    print(
        f"Camera drift: {drift_translation_mm:.3f} mm, "
        f"{drift_rotation_deg:.3f} deg; pass={drift_pass}"
    )
    print(f"Result: {output_dir / 'camera_drift.yaml'}")
    return 0 if combined_summary["overall_quality_pass"] else 3


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference_path = args.reference.resolve()
    baseline_path = (
        args.baseline_calibration.resolve()
        if args.baseline_calibration is not None
        else reference_path.parent / "camera_calibration.yaml"
    )
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    reference, poses = load_replay_poses(reference_path, args.pose_indices)
    metrics = trajectory_metrics(poses)
    failures = validate_trajectory(metrics, args)
    output_dir = (args.out if args.out is not None else _default_output_dir()).resolve()
    _write_plan(
        output_dir,
        args,
        reference_path,
        baseline_path,
        poses,
        metrics,
        failures,
    )
    print(f"Replay poses: {len(poses)}")
    print(
        "Largest recorded segment: "
        f"{metrics['maximum_segment_translation_mm']:.1f} mm, "
        f"{metrics['maximum_segment_rotation_deg']:.1f} deg"
    )
    print(f"Trajectory scalar preflight pass: {not failures}")
    print(f"Plan: {output_dir / 'replay_plan.yaml'}")
    if failures:
        raise RuntimeError("trajectory preflight failed: " + ", ".join(failures))
    if not args.execute_motion:
        print("Dry-run only; RTDE and camera were not opened; motion commands sent: 0")
        return 0
    if not _confirm_motion(poses):
        print("Confirmation not received; motion commands sent: 0")
        return 2
    return execute_replay(
        args,
        output_dir,
        reference_path,
        baseline_path,
        reference,
        poses,
        metrics,
    )


if __name__ == "__main__":
    raise SystemExit(main())
