#!/usr/bin/env python3
"""Calibrate a fixed workspace camera in the UR5 + RH56 MuJoCo scene.

The script deliberately estimates the camera from rendered chessboard images
and robot forward kinematics.  MuJoCo's exact camera pose is used only after
calibration to report the estimation error.

No hardware commands are issued.  The generated calibration numbers belong to
the simulated camera; reuse the workflow and file schema on hardware, not the
simulation numbers themselves.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# EGL keeps the paper-facing tool usable on headless machines.  Users can set
# MUJOCO_GL before launching to select another backend.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rh56_controller.camera_calibration import (  # noqa: E402
    calibrate_intrinsics,
    chessboard_object_points,
    detect_chessboard,
    estimate_eye_to_hand,
    interpolate_joint_positions,
    invert_transform,
    look_at_rotation,
    make_transform,
    mujoco_camera_pose_opencv,
    pinhole_intrinsics_from_fovy,
    rotation_error_deg,
    rotation_matrix_to_wxyz,
    transform_to_dict,
)


DEFAULT_XML = REPO_ROOT / "h1_mujoco/inspire/ur5_inspire.xml"
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HOME_ARM_Q = np.array([0.0, -np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, 0.0, 0.0])
POSE_PERTURBATION_RAD = np.array([0.22, 0.18, 0.22, 0.16, 0.28, 0.35])
CAMERA_NAME = "workspace_camera"
GRIPPER_FRAME = "gripper_attachment"
TARGET_FRAME = "calibration_target"


@dataclass
class Capture:
    attempt: int
    accepted_index: int
    joint_positions: np.ndarray
    image_rgb: np.ndarray
    corners: np.ndarray
    base_from_gripper: np.ndarray
    base_from_target_truth: np.ndarray


@dataclass(frozen=True)
class PathSafetyCheck:
    safe: bool
    samples_checked: int
    failure_reason: str
    contact_descriptions: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a wrist-mounted chessboard in the UR5+RH56 MuJoCo scene, "
            "estimate camera intrinsics and fixed eye-to-hand extrinsics, and "
            "compare the result with simulation ground truth."
        )
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/ur5_external_camera_calibration"),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fovy-deg", type=float, default=48.0)
    parser.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        default=[-0.25, -0.68, 1.18],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--camera-look-at",
        type=float,
        nargs=3,
        default=[-0.25, -0.44, 0.55],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--captures", type=int, default=25)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument(
        "--path-step-deg",
        type=float,
        default=1.0,
        help=(
            "Maximum joint change between collision checks along a simulated "
            "capture-to-capture path (default: 1 degree)."
        ),
    )
    parser.add_argument(
        "--apparatus-clearance-mm",
        type=float,
        default=10.0,
        help=(
            "Clearance margin around the calibration board, bracket, and "
            "camera housing (default: 10 mm)."
        ),
    )
    parser.add_argument(
        "--camera-housing-size-mm",
        type=float,
        nargs=3,
        default=[80.0, 40.0, 30.0],
        metavar=("WIDTH", "HEIGHT", "DEPTH"),
        help="Full XYZ dimensions of the external camera body proxy.",
    )
    parser.add_argument(
        "--estimate-distortion",
        action="store_true",
        help=(
            "Estimate lens distortion. Leave disabled for MuJoCo's ideal "
            "pinhole renderer; enable it for real camera images."
        ),
    )
    parser.add_argument(
        "--save-captures",
        action="store_true",
        help="Save every accepted annotated calibration image under the output directory.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.xml.is_file():
        raise FileNotFoundError(args.xml)
    if args.width < 64 or args.height < 64:
        raise ValueError("image width and height must be at least 64 pixels")
    if not 1.0 < args.fovy_deg < 179.0:
        raise ValueError("--fovy-deg must be in (1, 179)")
    if args.captures < 5:
        raise ValueError("--captures must be at least 5")
    if args.max_attempts < args.captures:
        raise ValueError("--max-attempts cannot be smaller than --captures")
    if args.pattern_cols < 2 or args.pattern_rows < 2:
        raise ValueError("the chessboard needs at least 2x2 inner corners")
    if args.square_size_mm <= 0.0:
        raise ValueError("--square-size-mm must be positive")
    if args.path_step_deg <= 0.0:
        raise ValueError("--path-step-deg must be positive")
    if args.apparatus_clearance_mm < 0.0:
        raise ValueError("--apparatus-clearance-mm cannot be negative")
    if any(dimension <= 0.0 for dimension in args.camera_housing_size_mm):
        raise ValueError("all --camera-housing-size-mm dimensions must be positive")


def _add_chessboard(
    spec: mujoco.MjSpec,
    *,
    pattern_size: tuple[int, int],
    square_size_m: float,
    clearance_m: float,
) -> None:
    """Attach a printed-board and conservative bracket collision proxy."""

    columns, rows = pattern_size
    grid_columns = columns + 1
    grid_rows = rows + 1
    grid_width = grid_columns * square_size_m
    grid_height = grid_rows * square_size_m

    # The simulated bracket moves the board to the side of RH56 so the fingers
    # do not hide its inner corners.  Eye-to-hand calibration solves the unknown
    # rigid gripper-to-target transform, so this bracket pose is not an input to
    # the estimator and need not be reproduced numerically on hardware.
    board = spec.body(GRIPPER_FRAME).add_body(
        name=TARGET_FRAME,
        pos=[0.25, 0.016, 0.17],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    )
    margin = 0.35 * square_size_m
    board.add_geom(
        name="calibration_board_backing",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[grid_width / 2.0 + margin, grid_height / 2.0 + margin, 0.002],
        pos=[0.0, 0.0, -0.002],
        rgba=[1.0, 1.0, 1.0, 1.0],
        contype=1,
        conaffinity=1,
        margin=clearance_m,
        # Keep the white backing visible to the calibration camera while also
        # using this same box as its collision proxy.
        group=0,
        density=0.0,
    )
    board.add_geom(
        name="calibration_board_bracket",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.008, 0.0, 0.0],
        # Stop behind the backing so the physical proxy cannot cover inner
        # chessboard corners in the rendered calibration image.
        fromto=[-0.25, -0.016, -0.17, 0.0, 0.0, -0.015],
        rgba=[0.3, 0.3, 0.3, 1.0],
        contype=1,
        conaffinity=1,
        margin=clearance_m,
        group=3,
        density=0.0,
    )
    for row in range(grid_rows):
        for column in range(grid_columns):
            if (row + column) % 2 != 0:
                continue
            x_position = -grid_width / 2.0 + (column + 0.5) * square_size_m
            y_position = grid_height / 2.0 - (row + 0.5) * square_size_m
            board.add_geom(
                name=f"calibration_square_{row}_{column}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[square_size_m / 2.0, square_size_m / 2.0, 0.0005],
                pos=[x_position, y_position, 0.0005],
                rgba=[0.005, 0.005, 0.005, 1.0],
                contype=0,
                conaffinity=0,
            )


def _build_model(args: argparse.Namespace) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(args.xml.resolve()))
    spec.visual.global_.offwidth = args.width
    spec.visual.global_.offheight = args.height
    _add_chessboard(
        spec,
        pattern_size=(args.pattern_cols, args.pattern_rows),
        square_size_m=args.square_size_mm / 1000.0,
        clearance_m=args.apparatus_clearance_mm / 1000.0,
    )
    camera_rotation = look_at_rotation(args.camera_position, args.camera_look_at)
    camera_body = spec.worldbody.add_body(
        name="workspace_camera_housing",
        pos=args.camera_position,
        quat=rotation_matrix_to_wxyz(camera_rotation),
    )
    camera_size_m = np.asarray(args.camera_housing_size_mm, dtype=float) / 1000.0
    # MuJoCo cameras look down local -Z. Put the housing just behind the
    # optical centre so its collision proxy does not occlude its own image.
    camera_body.add_geom(
        name="workspace_camera_housing_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(camera_size_m / 2.0).tolist(),
        pos=[0.0, 0.0, camera_size_m[2] / 2.0 + 0.005],
        rgba=[0.12, 0.12, 0.14, 1.0],
        contype=1,
        conaffinity=1,
        margin=args.apparatus_clearance_mm / 1000.0,
        group=3,
        density=0.0,
    )
    camera_body.add_camera(
        name=CAMERA_NAME,
        pos=[0.0, 0.0, 0.0],
        fovy=args.fovy_deg,
    )
    return spec.compile()


def _body_pose(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return make_transform(
        data.xmat[body_id].reshape(3, 3),
        data.xpos[body_id],
    )


def _candidate_joint_poses(args: argparse.Namespace):
    yield HOME_ARM_Q.copy()
    random_generator = np.random.default_rng(args.seed)
    for _ in range(args.max_attempts - 1):
        yield HOME_ARM_Q + random_generator.uniform(-1.0, 1.0, 6) * POSE_PERTURBATION_RAD


def _contact_description(model: mujoco.MjModel, contact: mujoco.MjContact) -> str:
    def label(geom_id: int) -> str:
        geom_name = model.geom(geom_id).name or f"geom_{geom_id}"
        body_name = model.body(model.geom_bodyid[geom_id]).name or "world"
        return f"{body_name}/{geom_name}"

    return (
        f"{label(contact.geom1)} <-> {label(contact.geom2)} "
        f"(distance={1000.0 * float(contact.dist):.3f} mm)"
    )


def _is_allowed_fixture_contact(model: mujoco.MjModel, contact: mujoco.MjContact) -> bool:
    """Ignore the base-fixture overlap already encoded by the source scene."""

    geom_names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
    body_names = {
        model.body(model.geom_bodyid[contact.geom1]).name,
        model.body(model.geom_bodyid[contact.geom2]).name,
    }
    return geom_names == {"floor", ""} and body_names == {"world", "shoulder_link"}


def _unsafe_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[str, ...]:
    descriptions = {
        _contact_description(model, data.contact[index])
        for index in range(data.ncon)
        if not _is_allowed_fixture_contact(model, data.contact[index])
    }
    return tuple(sorted(descriptions))


def _check_joint_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_ids: np.ndarray,
    joint_addresses: np.ndarray,
    start: np.ndarray,
    target: np.ndarray,
    max_step_rad: float,
) -> PathSafetyCheck:
    samples = interpolate_joint_positions(start, target, max_step_rad)
    for sample_index, sample in enumerate(samples, start=1):
        for joint_id, value in zip(joint_ids, sample):
            if model.jnt_limited[joint_id]:
                lower, upper = model.jnt_range[joint_id]
                if value < lower or value > upper:
                    return PathSafetyCheck(
                        safe=False,
                        samples_checked=sample_index,
                        failure_reason=(
                            f"joint_limit:{model.joint(int(joint_id)).name}="
                            f"{float(value):.6f} not in [{lower:.6f}, {upper:.6f}]"
                        ),
                        contact_descriptions=(),
                    )
        data.qpos[joint_addresses] = sample
        mujoco.mj_forward(model, data)
        contacts = _unsafe_contacts(model, data)
        if contacts:
            return PathSafetyCheck(
                safe=False,
                samples_checked=sample_index,
                failure_reason="collision_or_clearance",
                contact_descriptions=contacts,
            )
    return PathSafetyCheck(
        safe=True,
        samples_checked=len(samples),
        failure_reason="",
        contact_descriptions=(),
    )


def _draw_detected_corners(
    image_rgb: np.ndarray,
    pattern_size: tuple[int, int],
    corners: np.ndarray,
) -> np.ndarray:
    import cv2

    annotated_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.drawChessboardCorners(annotated_bgr, pattern_size, corners, True)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Failed to write image: {path}")


def _write_montage(path: Path, captures: list[Capture], pattern_size: tuple[int, int]) -> None:
    import cv2

    selected = [
        captures[index]
        for index in np.linspace(0, len(captures) - 1, min(6, len(captures)), dtype=int)
    ]
    tiles: list[np.ndarray] = []
    tile_width = 480
    for capture in selected:
        annotated = _draw_detected_corners(capture.image_rgb, pattern_size, capture.corners)
        scale = tile_width / annotated.shape[1]
        tile = cv2.resize(
            annotated,
            (tile_width, int(round(annotated.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        cv2.putText(
            tile,
            f"capture {capture.accepted_index:02d}",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 80, 30),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    columns = 3
    rows: list[np.ndarray] = []
    for start in range(0, len(tiles), columns):
        row_tiles = tiles[start : start + columns]
        while len(row_tiles) < columns:
            row_tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row_tiles))
    _save_rgb(path, np.vstack(rows))


def _write_printable_chessboard_svg(
    path: Path,
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> None:
    """Write the simulated target as a physical-size SVG for 100% printing."""

    columns, rows = pattern_size
    grid_columns = columns + 1
    grid_rows = rows + 1
    margin_mm = 0.35 * square_size_mm
    width_mm = grid_columns * square_size_mm + 2.0 * margin_mm
    height_mm = grid_rows * square_size_mm + 2.0 * margin_mm
    rectangles: list[str] = []
    for row in range(grid_rows):
        for column in range(grid_columns):
            if (row + column) % 2 != 0:
                continue
            x_position = margin_mm + column * square_size_mm
            y_position = margin_mm + row * square_size_mm
            rectangles.append(
                f'  <rect x="{x_position:.6f}" y="{y_position:.6f}" '
                f'width="{square_size_mm:.6f}" height="{square_size_mm:.6f}" fill="black"/>'
            )
    svg = "\n".join(
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
    )
    path.write_text(svg, encoding="utf-8")


def _project_points(
    object_points: np.ndarray,
    camera_from_target: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    import cv2

    rotation_vector, _ = cv2.Rodrigues(camera_from_target[:3, :3])
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        camera_from_target[:3, 3],
        camera_matrix,
        distortion,
    )
    return np.asarray(projected, dtype=float).reshape(-1, 2)


def _write_verification_image(
    path: Path,
    capture: Capture,
    *,
    pattern_size: tuple[int, int],
    object_points: np.ndarray,
    base_from_camera: np.ndarray,
    gripper_from_target: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    import cv2

    base_from_target = capture.base_from_gripper @ gripper_from_target
    camera_from_target = invert_transform(base_from_camera) @ base_from_target
    predicted = _project_points(
        object_points,
        camera_from_target,
        camera_matrix,
        distortion,
    )
    detected = capture.corners.reshape(-1, 2).astype(float)
    pixel_rmse = float(np.sqrt(np.mean(np.sum((predicted - detected) ** 2, axis=1))))

    image = capture.image_rgb.copy()
    for point in detected:
        cv2.circle(image, tuple(np.round(point).astype(int)), 4, (20, 230, 20), 2, cv2.LINE_AA)
    for point in predicted:
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
        f"green=detected  magenta=robot+calibration  RMSE={pixel_rmse:.3f}px",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    _save_rgb(path, image)
    return pixel_rmse


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is missing. Install the vision profile first:\n"
            "  tools/setup_uv_env.sh --profile sim-ur5-vision --python 3.12 --env .venv312"
        ) from exc
    if not hasattr(cv2, "calibrateHandEye"):
        raise SystemExit("OpenCV 4.x with calibrateHandEye is required (pin OpenCV to <5).")

    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    object_points = chessboard_object_points(pattern_size, args.square_size_mm / 1000.0)

    print("Assumptions:")
    print("  - fixed external eye-to-hand camera")
    print("  - ideal MuJoCo pinhole image; distortion fixed to zero unless requested")
    print("  - chessboard rigidly mounted beside the RH56 hand")
    print(
        "  - board, bracket, and camera housing use collision proxies with "
        f"{args.apparatus_clearance_mm:.1f} mm margin"
    )
    print(
        "  - every simulated path is sampled at no more than "
        f"{args.path_step_deg:.2f} deg per joint step"
    )
    print("  - this check is not authorization to replay poses on real hardware")
    print("  - simulation camera truth is used only for post-calibration validation")

    model = _build_model(args)
    data = mujoco.MjData(model)
    joint_addresses = np.array(
        [model.jnt_qposadr[model.joint(name).id] for name in ARM_JOINTS],
        dtype=int,
    )
    joint_ids = np.array([model.joint(name).id for name in ARM_JOINTS], dtype=int)
    gripper_body_id = model.body(GRIPPER_FRAME).id
    target_body_id = model.body(TARGET_FRAME).id
    camera_id = model.camera(CAMERA_NAME).id
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    captures: list[Capture] = []
    capture_rows: list[dict[str, object]] = []
    current_joint_positions = HOME_ARM_Q.copy()
    path_samples_checked = 0
    collision_rejections = 0
    try:
        for attempt, joint_positions in enumerate(_candidate_joint_poses(args)):
            path_check = _check_joint_path(
                model,
                data,
                joint_ids=joint_ids,
                joint_addresses=joint_addresses,
                start=current_joint_positions,
                target=joint_positions,
                max_step_rad=np.deg2rad(args.path_step_deg),
            )
            path_samples_checked += path_check.samples_checked
            corners = None
            image_rgb = None
            if path_check.safe:
                current_joint_positions = joint_positions.copy()
                renderer.update_scene(data, camera=CAMERA_NAME)
                image_rgb = renderer.render().copy()
                corners = detect_chessboard(image_rgb, pattern_size)
            else:
                collision_rejections += 1
                data.qpos[joint_addresses] = current_joint_positions
                mujoco.mj_forward(model, data)
            detected = corners is not None
            accepted_index = len(captures) if detected else -1
            capture_rows.append(
                {
                    "attempt": attempt,
                    "simulation_path_collision_free": int(path_check.safe),
                    "path_samples_checked": path_check.samples_checked,
                    "path_failure_reason": path_check.failure_reason,
                    "path_contacts": " | ".join(path_check.contact_descriptions),
                    "detected": int(detected),
                    "accepted_index": accepted_index,
                    "hardware_replay_approved": 0,
                    **{
                        f"{name}_rad": float(value)
                        for name, value in zip(ARM_JOINTS, joint_positions)
                    },
                    "board_x_m": float(data.xpos[target_body_id, 0]),
                    "board_y_m": float(data.xpos[target_body_id, 1]),
                    "board_z_m": float(data.xpos[target_body_id, 2]),
                }
            )
            if not detected:
                if attempt + 1 >= args.max_attempts:
                    break
                continue
            capture = Capture(
                attempt=attempt,
                accepted_index=accepted_index,
                joint_positions=joint_positions.copy(),
                image_rgb=image_rgb,
                corners=corners,
                base_from_gripper=_body_pose(data, gripper_body_id),
                base_from_target_truth=_body_pose(data, target_body_id),
            )
            captures.append(capture)
            if args.save_captures:
                _save_rgb(
                    output_dir / "captures" / f"capture_{accepted_index:03d}.png",
                    _draw_detected_corners(image_rgb, pattern_size, corners),
                )
            if len(captures) >= args.captures:
                break
    finally:
        renderer.close()

    # Preserve safety diagnostics even when too few images are usable for
    # calibration (for example, because a camera proxy blocks every path).
    _write_csv(output_dir / "capture_summary.csv", capture_rows)
    if len(captures) < args.captures:
        raise RuntimeError(
            f"Only detected {len(captures)} usable views out of {args.max_attempts}; "
            "adjust the camera or pose range."
        )

    object_point_sets = [object_points.copy() for _ in captures]
    image_point_sets = [capture.corners for capture in captures]
    intrinsic = calibrate_intrinsics(
        object_point_sets,
        image_point_sets,
        (args.width, args.height),
        estimate_distortion=args.estimate_distortion,
    )
    hand_eye = estimate_eye_to_hand(
        [capture.base_from_gripper for capture in captures],
        intrinsic.camera_from_target,
    )

    truth_intrinsics = pinhole_intrinsics_from_fovy(
        args.fovy_deg,
        args.width,
        args.height,
    )
    truth_base_from_camera = mujoco_camera_pose_opencv(
        data.cam_xpos[camera_id],
        data.cam_xmat[camera_id].reshape(3, 3),
    )
    translation_error_mm = float(
        1000.0
        * np.linalg.norm(
            hand_eye.base_from_camera[:3, 3] - truth_base_from_camera[:3, 3]
        )
    )
    rotation_error = rotation_error_deg(hand_eye.base_from_camera, truth_base_from_camera)
    focal_error_px = float(
        np.mean(
            np.abs(
                np.diag(intrinsic.camera_matrix)[:2]
                - np.diag(truth_intrinsics)[:2]
            )
        )
    )
    principal_point_error_px = float(
        np.linalg.norm(
            intrinsic.camera_matrix[:2, 2] - truth_intrinsics[:2, 2]
        )
    )
    verification_rmse_px = _write_verification_image(
        output_dir / "verification.png",
        captures[-1],
        pattern_size=pattern_size,
        object_points=object_points,
        base_from_camera=hand_eye.base_from_camera,
        gripper_from_target=hand_eye.gripper_from_target,
        camera_matrix=intrinsic.camera_matrix,
        distortion=intrinsic.distortion,
    )
    _write_montage(output_dir / "calibration_montage.png", captures, pattern_size)
    _write_printable_chessboard_svg(
        output_dir / "printable_chessboard.svg",
        pattern_size,
        args.square_size_mm,
    )
    summary_row = {
        "accepted_views": len(captures),
        "attempted_views": len(capture_rows),
        "collision_rejected_views": collision_rejections,
        "path_samples_checked": path_samples_checked,
        "path_step_deg": args.path_step_deg,
        "apparatus_clearance_mm": args.apparatus_clearance_mm,
        "intrinsic_rms_px": intrinsic.rms_reprojection_error_px,
        "verification_rmse_px": verification_rmse_px,
        "focal_error_px": focal_error_px,
        "principal_point_error_px": principal_point_error_px,
        "extrinsic_translation_error_mm": translation_error_mm,
        "extrinsic_rotation_error_deg": rotation_error,
        "handeye_residual_translation_mm": 1000.0
        * hand_eye.residual_translation_rms_m,
        "handeye_residual_rotation_deg": hand_eye.residual_rotation_rms_deg,
    }
    _write_csv(output_dir / "summary.csv", [summary_row])

    calibration_payload = {
        "schema": "rh56_ur5_external_camera_calibration/v2",
        "scope": "simulation-only fixed external camera calibration",
        "frames": {
            "base": "ur5_base",
            "camera": f"{CAMERA_NAME}_opencv",
            "gripper": GRIPPER_FRAME,
            "target": "opencv_chessboard_inner_corner_frame",
            "camera_axes": "+X right, +Y down, +Z forward",
        },
        "image": {
            "width": args.width,
            "height": args.height,
            "model": "pinhole",
            "camera_matrix": intrinsic.camera_matrix.tolist(),
            "distortion_coefficients": intrinsic.distortion.tolist(),
            "distortion_estimated": bool(args.estimate_distortion),
        },
        "chessboard": {
            "inner_corners": [args.pattern_cols, args.pattern_rows],
            "square_size_m": args.square_size_mm / 1000.0,
        },
        "transforms": {
            "base_from_camera": transform_to_dict(hand_eye.base_from_camera),
            "gripper_from_target": transform_to_dict(hand_eye.gripper_from_target),
        },
        "metrics": summary_row,
        "simulation_ground_truth": {
            "camera_matrix": truth_intrinsics.tolist(),
            "base_from_camera": transform_to_dict(truth_base_from_camera),
        },
        "reproducibility": {
            "xml": str(args.xml.resolve()),
            "seed": args.seed,
            "camera_position_m": list(map(float, args.camera_position)),
            "camera_look_at_m": list(map(float, args.camera_look_at)),
            "camera_fovy_deg": args.fovy_deg,
            "arm_joint_names": list(ARM_JOINTS),
            "path_step_deg": args.path_step_deg,
            "apparatus_clearance_mm": args.apparatus_clearance_mm,
            "camera_housing_size_mm": list(map(float, args.camera_housing_size_mm)),
            "allowed_fixture_contact": (
                "world/floor <-> shoulder_link/unnamed collision geom"
            ),
        },
        "simulation_safety_check": {
            "collision_rejected_views": collision_rejections,
            "path_samples_checked": path_samples_checked,
            "checks": [
                "UR5 joint limits at every interpolated sample",
                "MuJoCo contacts along every capture-to-capture joint-space segment",
                "clearance margin around calibration board, bracket, and camera housing",
            ],
            "not_modelled": [
                "real table and robot mounting mismatch",
                "camera stand and clamps",
                "cables and soft covers",
                "controller tracking error and stopping distance",
                "unmodelled RH56 or calibration-board mount geometry",
            ],
            "hardware_pose_replay_authorized": False,
        },
        "assumptions": [
            "No artificial image, joint, or calibration noise is injected.",
            "The chessboard is rigid relative to the gripper during all captures.",
            "The saved transform maps OpenCV camera coordinates into UR5 base coordinates.",
            "Simulation calibration numbers must not be copied to real hardware.",
            "Simulation collision-free paths are not certified safe real-robot trajectories.",
        ],
    }
    (output_dir / "camera_calibration.yaml").write_text(
        yaml.safe_dump(calibration_payload, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "command_arguments": vars(args)
                | {"xml": str(args.xml), "out": str(args.out)},
                "opencv_version": cv2.__version__,
                "mujoco_version": mujoco.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Accepted views: {len(captures)}/{len(capture_rows)}")
    print(
        "Simulation path checks: "
        f"{path_samples_checked} samples, {collision_rejections} rejected candidates"
    )
    print(f"Intrinsic RMS: {intrinsic.rms_reprojection_error_px:.4f} px")
    print(f"Verification reprojection RMSE: {verification_rmse_px:.4f} px")
    print(
        "External-camera error against hidden MuJoCo truth: "
        f"{translation_error_mm:.3f} mm, {rotation_error:.4f} deg"
    )
    print(f"Calibration: {output_dir / 'camera_calibration.yaml'}")
    print(f"Visual check: {output_dir / 'verification.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
