#!/usr/bin/env python3
"""Compare pretrained GraspGen-X with RH56 analytical synchronized closure.

The comparison deliberately shares the same accurate RH56 MuJoCo model,
object, approach/lift timing, contact-limited stopping rule, and success
criterion.  The analytical method supplies a width-conditioned coupled-joint
trajectory and synchronizes the virtual wrist pose to keep the antipodal grasp
center fixed.  GraspGen-X supplies a pretrained 6-D grasp pose and uses its
official Inspire-Hand open-to-close joint interpolation while holding that
pose fixed.

This is a simulation benchmark, not a claim about hardware success.  Its first
paper-facing case is a 40 mm cube so the learned baseline can be validated
before adding production objects or robot-arm reachability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

from rh56_controller.capsule_hand_proxy import (  # noqa: E402
    closure_base_rotation,
    rotation_to_grasp_scene_euler_xyz,
    set_closure_qpos,
)
from rh56_controller.grasp_geometry import (  # noqa: E402
    ACTUATOR_NAMES,
    ClosureGeometry,
    ClosureResult,
    InspireHandFK,
)
from rh56_controller.graspgenx_baseline import (  # noqa: E402
    GraspGenXCandidate,
    graspgenx_to_mujoco_base_pose,
    load_isaac_grasp_yaml,
    pregrasp_base_position,
)


DEFAULT_XML = REPO_ROOT / "h1_mujoco" / "inspire" / "inspire_grasp_scene.xml"
METHODS = ("analytical", "graspgenx")
FINGER_ACTUATORS = tuple(ACTUATOR_NAMES)
BASE_ACTUATORS = (
    "right_pos_x_position",
    "right_pos_y_position",
    "right_pos_z_position",
    "right_rot_x_position",
    "right_rot_y_position",
    "right_rot_z_position",
)

GGX_OPEN_CTRL = {
    "pinky": 0.0,
    "ring": 0.0,
    "middle": 0.0,
    "index": 0.0,
    "thumb_proximal": 0.1,  # Accurate MuJoCo model lower limit.
    "thumb_yaw": 1.308,
}
GGX_CLOSE_CTRL = {
    "pinky": 1.47,
    "ring": 1.47,
    "middle": 1.47,
    "index": 1.47,
    "thumb_proximal": 0.57,  # Accurate MuJoCo model upper actuator limit.
    "thumb_yaw": 1.308,
}


@dataclass(frozen=True)
class PoseCommand:
    base_position: np.ndarray
    base_rotation: np.ndarray
    finger_ctrl: dict[str, float]


@dataclass(frozen=True)
class TrialCondition:
    error_mm: float
    direction_deg: float
    object_offset: np.ndarray


@dataclass
class TrialResult:
    method: str
    error_mm: float
    direction_deg: float
    object_dx_mm: float
    object_dy_mm: float
    success: bool
    initial_object_z_mm: float
    final_object_z_mm: float
    max_object_z_mm: float
    final_lift_mm: float
    max_lift_mm: float
    first_opposing_contact_alpha: float | None
    stopped_closure_alpha: float
    opposing_contact_detected: bool
    max_object_xy_displacement_mm: float
    candidate_name: str
    candidate_confidence: float | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RH56 analytical synchronized closure with pretrained "
            "GraspGen-X grasp poses in the same MuJoCo lift test."
        )
    )
    parser.add_argument("--graspgenx-yaml", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=Path, default=Path("artifacts/graspgenx_success_comparison"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--target-width-mm", type=float, default=40.0)
    parser.add_argument("--cube-width-mm", type=float, default=40.0)
    parser.add_argument("--cube-mass-g", type=float, default=50.0)
    parser.add_argument(
        "--error-mm",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 4.0, 6.0],
        help="Horizontal object-position errors evaluated for both methods.",
    )
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--analytical-yaw-deg", type=float, default=0.0)
    parser.add_argument("--standoff-mm", type=float, default=60.0)
    parser.add_argument("--lift-mm", type=float, default=80.0)
    parser.add_argument("--success-lift-mm", type=float, default=40.0)
    parser.add_argument("--settle-s", type=float, default=0.25)
    parser.add_argument("--approach-s", type=float, default=0.8)
    parser.add_argument("--close-s", type=float, default=1.8)
    parser.add_argument("--post-contact-s", type=float, default=0.20)
    parser.add_argument("--lift-s", type=float, default=0.9)
    parser.add_argument("--hold-s", type=float, default=0.8)
    parser.add_argument(
        "--contact-compression",
        type=float,
        default=0.10,
        help="Extra normalized closure after opposing contacts are first detected.",
    )
    parser.add_argument(
        "--opposing-span-fraction",
        type=float,
        default=0.30,
        help="Required contact span along the closing axis as a fraction of cube width.",
    )
    parser.add_argument(
        "--graspgenx-rank",
        type=int,
        default=None,
        help="Use this confidence rank directly; otherwise select the first top-down floor-clear pose.",
    )
    parser.add_argument("--topdown-cos", type=float, default=0.50)
    parser.add_argument("--rebuild-fk", action="store_true")
    parser.add_argument("--video", action="store_true", help="Render the zero-error trial as a side-by-side MP4.")
    parser.add_argument("--video-width", type=int, default=480)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-22.0)
    parser.add_argument("--camera-distance", type=float, default=0.42)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.cube_width_mm <= 0 or args.cube_mass_g <= 0:
        raise ValueError("Cube width and mass must be positive")
    if args.target_width_mm <= 0 or args.directions < 1:
        raise ValueError("Target width and directions must be positive")
    if any(error < 0 for error in args.error_mm):
        raise ValueError("Position errors must be non-negative")
    if not 0.0 <= args.contact_compression <= 0.25:
        raise ValueError("--contact-compression must be in [0, 0.25]")
    if not 0.0 < args.opposing_span_fraction <= 1.0:
        raise ValueError("--opposing-span-fraction must be in (0, 1]")
    if args.video_width <= 0 or args.video_height <= 0 or args.fps <= 0:
        raise ValueError("Video dimensions and FPS must be positive")


def _conditions(error_levels_mm: list[float], directions: int) -> list[TrialCondition]:
    rows: list[TrialCondition] = []
    for error_mm in error_levels_mm:
        if math.isclose(error_mm, 0.0, abs_tol=1e-12):
            rows.append(TrialCondition(0.0, 0.0, np.zeros(3)))
            continue
        for direction_index in range(directions):
            angle = 2.0 * math.pi * direction_index / directions
            offset = np.array([math.cos(angle), math.sin(angle), 0.0]) * error_mm / 1000.0
            rows.append(TrialCondition(float(error_mm), math.degrees(angle), offset))
    return rows


def _add_cube_model(xml_path: Path, width_m: float, mass_kg: float) -> tuple[mujoco.MjModel, int, int]:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    body = spec.worldbody.add_body(name="benchmark_object", pos=[0.0, 0.0, 0.0])
    body.add_freejoint(name="benchmark_object_free")
    body.add_geom(
        name="benchmark_object_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[width_m / 2.0] * 3,
        mass=mass_kg,
        rgba=[0.90, 0.12, 0.08, 1.0],
        friction=[1.5, 0.01, 0.001],
        condim=6,
    )
    model = spec.compile()
    object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "benchmark_object_geom")
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "benchmark_object_free")
    return model, int(object_geom_id), int(model.jnt_qposadr[object_joint_id])


def _actuator_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
    if any(actuator_id < 0 for actuator_id in ids):
        missing = [name for name, actuator_id in zip(names, ids) if actuator_id < 0]
        raise ValueError(f"Missing actuators in RH56 model: {missing}")
    return np.asarray(ids, dtype=int)


def _object_position(data: mujoco.MjData, object_qadr: int) -> np.ndarray:
    return data.qpos[object_qadr : object_qadr + 3].copy()


def _set_object_pose(data: mujoco.MjData, object_qadr: int, center: np.ndarray) -> None:
    data.qpos[object_qadr : object_qadr + 7] = np.r_[center, 1.0, 0.0, 0.0, 0.0]


def _finger_vector(values: dict[str, float]) -> np.ndarray:
    return np.array([float(values[name]) for name in FINGER_ACTUATORS], dtype=float)


def _base_vector(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.r_[position, rotation_to_grasp_scene_euler_xyz(rotation)]


def _write_command(
    data: mujoco.MjData,
    base_ids: np.ndarray,
    finger_ids: np.ndarray,
    command: PoseCommand,
) -> None:
    data.ctrl[base_ids] = _base_vector(command.base_position, command.base_rotation)
    data.ctrl[finger_ids] = _finger_vector(command.finger_ctrl)


def _lerp_dict(start: dict[str, float], end: dict[str, float], alpha: float) -> dict[str, float]:
    return {name: (1.0 - alpha) * start[name] + alpha * end[name] for name in FINGER_ACTUATORS}


def _grasp_center(result: ClosureResult) -> np.ndarray:
    return result.grasp_center("antipodal")


def _analytical_pose(
    result: ClosureResult,
    nominal_object_center: np.ndarray,
    yaw_rad: float,
) -> PoseCommand:
    rotation = closure_base_rotation(result, yaw_rad)
    position = nominal_object_center - rotation @ _grasp_center(result)
    return PoseCommand(position, rotation, dict(result.ctrl_values))


def _analytical_closing_axis(result: ClosureResult, rotation: np.ndarray) -> np.ndarray:
    thumb = result.tip_positions["thumb"]
    opposing = np.vstack(
        [position for name, position in result.tip_positions.items() if name != "thumb"]
    ).mean(axis=0)
    axis = rotation @ (thumb - opposing)
    return axis / np.linalg.norm(axis)


def _object_hand_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_geom_id: int,
) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.dist > 1e-5:
            continue
        if contact.geom1 == object_geom_id:
            other_geom = int(contact.geom2)
        elif contact.geom2 == object_geom_id:
            other_geom = int(contact.geom1)
        else:
            continue
        other_body = int(model.geom_bodyid[other_geom])
        if other_body == 0:  # Object-floor contact is not a hand contact.
            continue
        points.append(np.asarray(contact.pos, dtype=float).copy())
    return points


def _has_opposing_contacts(
    points: list[np.ndarray],
    object_center: np.ndarray,
    closing_axis: np.ndarray,
    required_span_m: float,
) -> bool:
    if len(points) < 2:
        return False
    projections = np.array([(point - object_center) @ closing_axis for point in points])
    return float(projections.max() - projections.min()) >= required_span_m


def _floor_hand_collision(model: mujoco.MjModel, data: mujoco.MjData, object_geom_id: int) -> bool:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor_id not in pair or object_geom_id in pair:
            continue
        if contact.dist < -1e-5:
            return True
    return False


def _initialize_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_qadr: int,
    object_center: np.ndarray,
    command: PoseCommand,
    base_ids: np.ndarray,
    finger_ids: np.ndarray,
) -> None:
    mujoco.mj_resetData(model, data)
    set_closure_qpos(
        model,
        data,
        command.finger_ctrl,
        base_position=command.base_position,
        base_rotation=command.base_rotation,
    )
    _set_object_pose(data, object_qadr, object_center)
    data.qvel[:] = 0.0
    _write_command(data, base_ids, finger_ids, command)
    mujoco.mj_forward(model, data)


def _select_graspgenx_candidate(
    args: argparse.Namespace,
    candidates: list[GraspGenXCandidate],
    nominal_center: np.ndarray,
    width_m: float,
    mass_kg: float,
) -> tuple[int, GraspGenXCandidate]:
    if args.graspgenx_rank is not None:
        if not 0 <= args.graspgenx_rank < len(candidates):
            raise ValueError("--graspgenx-rank is outside the candidate list")
        return args.graspgenx_rank, candidates[args.graspgenx_rank]

    for rank, candidate in enumerate(candidates):
        if candidate.approach_direction[2] > -args.topdown_cos:
            continue
        model, object_geom_id, object_qadr = _add_cube_model(args.xml, width_m, mass_kg)
        data = mujoco.MjData(model)
        base_ids = _actuator_ids(model, BASE_ACTUATORS)
        finger_ids = _actuator_ids(model, FINGER_ACTUATORS)
        position, rotation = graspgenx_to_mujoco_base_pose(candidate, object_center=nominal_center)
        command = PoseCommand(position, rotation, GGX_OPEN_CTRL)
        _initialize_state(
            model,
            data,
            object_qadr,
            nominal_center,
            command,
            base_ids,
            finger_ids,
        )
        if not _floor_hand_collision(model, data, object_geom_id):
            return rank, candidate
    raise RuntimeError("No top-down, floor-clear GraspGen-X candidate was found")


def _make_camera(args: argparse.Namespace) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.085]
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    return camera


def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    height = 12 + 17 * len(lines)
    draw.rectangle((0, 0, image.width, height), fill=(0, 0, 0, 145))
    for index, line in enumerate(lines):
        draw.text((8, 7 + 17 * index), line, fill=(255, 255, 255, 255))
    return np.asarray(image)


def _run_trial(
    *,
    args: argparse.Namespace,
    method: str,
    condition: TrialCondition,
    closure: ClosureGeometry,
    candidate: GraspGenXCandidate,
    record: bool,
) -> tuple[TrialResult, list[np.ndarray]]:
    width_m = args.cube_width_mm / 1000.0
    half_width_m = width_m / 2.0
    mass_kg = args.cube_mass_g / 1000.0
    nominal_center = np.array([0.0, 0.0, half_width_m], dtype=float)
    actual_center = nominal_center + condition.object_offset
    model, object_geom_id, object_qadr = _add_cube_model(args.xml, width_m, mass_kg)
    data = mujoco.MjData(model)
    base_ids = _actuator_ids(model, BASE_ACTUATORS)
    finger_ids = _actuator_ids(model, FINGER_ACTUATORS)

    if method == "analytical":
        minimum, maximum = closure.width_range("4-finger plane", n_fingers=4)
        target_width = float(np.clip(args.target_width_mm / 1000.0, minimum, maximum))
        open_result = closure.plane(maximum, n_fingers=4)
        final_result = closure.plane(target_width, n_fingers=4)
        yaw_rad = math.radians(args.analytical_yaw_deg)
        target_open = _analytical_pose(open_result, nominal_center, yaw_rad)
        standoff_position = target_open.base_position + np.array([0.0, 0.0, args.standoff_mm / 1000.0])
        initial = PoseCommand(standoff_position, target_open.base_rotation, target_open.finger_ctrl)

        def close_command(alpha: float) -> PoseCommand:
            width = (1.0 - alpha) * maximum + alpha * target_width
            return _analytical_pose(closure.plane(float(width), n_fingers=4), nominal_center, yaw_rad)

        closing_axis = _analytical_closing_axis(final_result, closure_base_rotation(final_result, yaw_rad))
        candidate_name = "analytical_plane4"
        candidate_confidence = None
    elif method == "graspgenx":
        target_position, target_rotation = graspgenx_to_mujoco_base_pose(
            candidate, object_center=nominal_center
        )
        standoff_position = pregrasp_base_position(
            target_position,
            candidate.approach_direction,
            args.standoff_mm / 1000.0,
        )
        initial = PoseCommand(standoff_position, target_rotation, dict(GGX_OPEN_CTRL))

        def close_command(alpha: float) -> PoseCommand:
            return PoseCommand(
                target_position,
                target_rotation,
                _lerp_dict(GGX_OPEN_CTRL, GGX_CLOSE_CTRL, alpha),
            )

        closing_axis = candidate.rotation[:, 0].copy()
        closing_axis /= np.linalg.norm(closing_axis)
        candidate_name = candidate.name
        candidate_confidence = candidate.confidence
    else:
        raise ValueError(f"Unsupported method: {method}")

    _initialize_state(
        model,
        data,
        object_qadr,
        actual_center,
        initial,
        base_ids,
        finger_ids,
    )

    renderer = (
        mujoco.Renderer(model, height=args.video_height, width=args.video_width)
        if record
        else None
    )
    camera = _make_camera(args) if record else None
    frames: list[np.ndarray] = []
    next_frame_time = 0.0
    phase = "settle"
    initial_z = float(actual_center[2])
    max_z = initial_z
    max_xy_displacement = 0.0
    first_contact_alpha: float | None = None
    stop_alpha = 1.0
    frozen_command: PoseCommand | None = None

    def sample_frame(label: str) -> None:
        nonlocal next_frame_time
        if renderer is None or camera is None:
            return
        if data.time + 1e-9 < next_frame_time:
            return
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        object_position = _object_position(data, object_qadr)
        frame = _overlay(
            frame,
            [
                "Analytical synchronized closure" if method == "analytical" else "Pretrained GraspGen-X",
                f"phase: {label}",
                f"object lift: {(object_position[2] - initial_z) * 1000.0:.1f} mm",
            ],
        )
        frames.append(frame)
        next_frame_time += 1.0 / args.fps

    def update_metrics() -> None:
        nonlocal max_z, max_xy_displacement
        position = _object_position(data, object_qadr)
        max_z = max(max_z, float(position[2]))
        max_xy_displacement = max(
            max_xy_displacement,
            float(np.linalg.norm(position[:2] - actual_center[:2])),
        )

    def step_for(duration: float, command_fn: Callable[[float], PoseCommand], label: str) -> None:
        nonlocal phase
        phase = label
        start_time = float(data.time)
        duration = max(0.0, duration)
        while data.time - start_time < duration - 1e-12:
            alpha = 1.0 if duration <= 0.0 else min(1.0, (data.time - start_time) / duration)
            _write_command(data, base_ids, finger_ids, command_fn(alpha))
            mujoco.mj_step(model, data)
            update_metrics()
            sample_frame(label)

    step_for(args.settle_s, lambda _alpha: initial, "settle")

    target_at_open = close_command(0.0)
    step_for(
        args.approach_s,
        lambda alpha: PoseCommand(
            (1.0 - alpha) * initial.base_position + alpha * target_at_open.base_position,
            target_at_open.base_rotation,
            target_at_open.finger_ctrl,
        ),
        "approach",
    )

    phase = "close"
    close_start = float(data.time)
    while data.time - close_start < args.close_s - 1e-12:
        alpha = min(1.0, (data.time - close_start) / max(args.close_s, 1e-9))
        points = _object_hand_contacts(model, data, object_geom_id)
        opposing = _has_opposing_contacts(
            points,
            _object_position(data, object_qadr),
            closing_axis,
            args.opposing_span_fraction * width_m,
        )
        if opposing and first_contact_alpha is None:
            first_contact_alpha = alpha
            stop_alpha = min(1.0, alpha + args.contact_compression)
        if first_contact_alpha is not None and alpha >= stop_alpha and frozen_command is None:
            frozen_command = close_command(stop_alpha)
        command = frozen_command if frozen_command is not None else close_command(alpha)
        _write_command(data, base_ids, finger_ids, command)
        mujoco.mj_step(model, data)
        update_metrics()
        sample_frame("close")

    if frozen_command is None:
        frozen_command = close_command(1.0)
        stop_alpha = 1.0

    step_for(args.post_contact_s, lambda _alpha: frozen_command, "stabilize")
    lift_start = frozen_command.base_position.copy()
    lift_delta = np.array([0.0, 0.0, args.lift_mm / 1000.0])
    step_for(
        args.lift_s,
        lambda alpha: PoseCommand(
            lift_start + alpha * lift_delta,
            frozen_command.base_rotation,
            frozen_command.finger_ctrl,
        ),
        "lift",
    )
    lifted_command = PoseCommand(
        lift_start + lift_delta,
        frozen_command.base_rotation,
        frozen_command.finger_ctrl,
    )
    step_for(args.hold_s, lambda _alpha: lifted_command, "hold")

    sample_frame("done")
    if renderer is not None:
        renderer.close()

    final_position = _object_position(data, object_qadr)
    final_lift_m = float(final_position[2] - initial_z)
    max_lift_m = float(max_z - initial_z)
    success = final_lift_m >= args.success_lift_mm / 1000.0
    result = TrialResult(
        method=method,
        error_mm=condition.error_mm,
        direction_deg=condition.direction_deg,
        object_dx_mm=condition.object_offset[0] * 1000.0,
        object_dy_mm=condition.object_offset[1] * 1000.0,
        success=success,
        initial_object_z_mm=initial_z * 1000.0,
        final_object_z_mm=float(final_position[2]) * 1000.0,
        max_object_z_mm=max_z * 1000.0,
        final_lift_mm=final_lift_m * 1000.0,
        max_lift_mm=max_lift_m * 1000.0,
        first_opposing_contact_alpha=first_contact_alpha,
        stopped_closure_alpha=stop_alpha,
        opposing_contact_detected=first_contact_alpha is not None,
        max_object_xy_displacement_mm=max_xy_displacement * 1000.0,
        candidate_name=candidate_name,
        candidate_confidence=candidate_confidence,
    )
    return result, frames


def _write_trials(path: Path, results: list[TrialResult]) -> None:
    rows = [asdict(result) for result in results]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(results: list[TrialResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = sorted({(result.method, result.error_mm) for result in results})
    for method, error_mm in keys:
        selected = [result for result in results if result.method == method and result.error_mm == error_mm]
        successes = sum(result.success for result in selected)
        rows.append(
            {
                "method": method,
                "error_mm": error_mm,
                "trials": len(selected),
                "successes": successes,
                "success_rate": successes / len(selected),
                "opposing_contact_rate": sum(result.opposing_contact_detected for result in selected) / len(selected),
                "mean_final_lift_mm": float(np.mean([result.final_lift_mm for result in selected])),
                "mean_max_lift_mm": float(np.mean([result.max_lift_mm for result in selected])),
                "mean_max_xy_displacement_mm": float(
                    np.mean([result.max_object_xy_displacement_mm for result in selected])
                ),
            }
        )
    return rows


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_video(path: Path, frames_by_method: dict[str, list[np.ndarray]], fps: int) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("MP4 output requires: uv sync --extra video") from exc

    available = [frames_by_method[method] for method in METHODS if method in frames_by_method]
    if not available or any(not frames for frames in available):
        raise RuntimeError("Representative frames were not recorded")
    frame_count = max(len(frames) for frames in available)
    height = available[0][0].shape[0]
    width = sum(frames[0].shape[1] for frames in available)
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
        for index in range(frame_count):
            panels = [frames[min(index, len(frames) - 1)] for frames in available]
            writer.send(np.concatenate(panels, axis=1))
    finally:
        writer.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)

    candidates = load_isaac_grasp_yaml(args.graspgenx_yaml)
    width_m = args.cube_width_mm / 1000.0
    nominal_center = np.array([0.0, 0.0, width_m / 2.0])
    candidate_rank, candidate = _select_graspgenx_candidate(
        args,
        candidates,
        nominal_center,
        width_m,
        args.cube_mass_g / 1000.0,
    )
    print(
        f"Selected GraspGen-X {candidate.name} at confidence rank {candidate_rank} "
        f"(score={candidate.confidence:.3f}, approach_z={candidate.approach_direction[2]:.3f})"
    )

    fk = InspireHandFK(rebuild=args.rebuild_fk)
    closure = ClosureGeometry(fk)
    conditions = _conditions([float(value) for value in args.error_mm], args.directions)
    results: list[TrialResult] = []
    representative_frames: dict[str, list[np.ndarray]] = {}

    for method in args.methods:
        for condition in conditions:
            record = args.video and condition.error_mm == 0.0 and method not in representative_frames
            result, frames = _run_trial(
                args=args,
                method=method,
                condition=condition,
                closure=closure,
                candidate=candidate,
                record=record,
            )
            results.append(result)
            if record:
                representative_frames[method] = frames
            state = "SUCCESS" if result.success else "fail"
            print(
                f"{method:10s} error={condition.error_mm:4.1f} mm "
                f"dir={condition.direction_deg:6.1f} deg -> {state}, "
                f"final lift={result.final_lift_mm:6.1f} mm"
            )

    _write_trials(args.out / "trials.csv", results)
    summaries = _summary_rows(results)
    _write_summary(args.out / "summary.csv", summaries)
    assumptions = {
        "scope": "simulation-only 40 mm cube lift benchmark",
        "comparison": {
            "analytical": "4-finger plane width trajectory with synchronized RH56 base pose",
            "graspgenx": "pretrained 6-D pose with official Inspire-Hand linear open/close endpoints",
            "shared_executor": "linear approach, opposing-contact stop plus normalized compression, vertical lift, hold",
        },
        "success_definition": f"final object center at least {args.success_lift_mm:.1f} mm above its initial height",
        "object_position_error": "planner uses nominal center while actual cube is shifted in the horizontal plane",
        "graspgenx_yaml": str(args.graspgenx_yaml.resolve()),
        "graspgenx_yaml_sha256": hashlib.sha256(args.graspgenx_yaml.read_bytes()).hexdigest(),
        "graspgenx_candidate": {
            "rank": candidate_rank,
            "name": candidate.name,
            "confidence": candidate.confidence,
            "approach_direction": candidate.approach_direction.tolist(),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "limitations": [
            "No H12/UR5 arm IK, balance, sensing, or navigation is modeled.",
            "The GraspGen-X pose is generated once for the nominal cube; this benchmark does not retrain it.",
            "MuJoCo contact success is not equivalent to real-hardware success.",
        ],
    }
    (args.out / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")

    if args.video:
        video_path = args.out / "nominal_comparison.mp4"
        _write_video(video_path, representative_frames, args.fps)
        print(f"Video: {video_path}")

    print(f"Trials: {args.out / 'trials.csv'}")
    print(f"Summary: {args.out / 'summary.csv'}")
    for row in summaries:
        print(
            f"  {row['method']:10s} @ {row['error_mm']:4.1f} mm: "
            f"{row['successes']}/{row['trials']} = {100.0 * row['success_rate']:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
