#!/usr/bin/env python3
"""Compare H12 screw-task camera layouts for Magpie and RH56.

This simulation-only screening study uses a low crowned-cover proxy beginning
shortly beyond the screw.  It compares the checked-in Magpie palm camera with
four RH56 scenarios: head-only, under-hand, close dorsal, and a close outboard
wrist mount.  The RH56 camera bodies are transparent assumptions, not
production mount designs.

For every screw-head-to-cover-edge gap, the script reports:

* closure-phase target visibility and the first ray occluder;
* hand/arm/camera clearance to the pack and sloped cover;
* camera-housing clearance to the RH56 hand (installation interference);
* wrist/arm compensation and quasi-static support margin; and
* a scenario-level predicted feasibility flag using explicit localization-
  error assumptions.

Outputs include summary.csv, trials.csv, assumptions.json, a Chinese result
guide, plots, camera-view snapshots, and an optional composite MP4/GIF.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_h12_screw_accessibility_benchmark as accessibility  # noqa: E402
from tools.run_h12_gripper_motion_comparison import add_text, named_id  # noqa: E402
from tools.run_h12_practical_gripper_comparison import (  # noqa: E402
    foot_support_polygon,
    signed_support_margin_m,
    whole_body_com,
)


@dataclass(frozen=True)
class CameraConfig:
    key: str
    label: str
    hand: str
    kind: str
    localization_error_mm: float
    reference_offset_world_m: tuple[float, float, float] | None = None
    color: tuple[float, float, float, float] = (0.1, 0.8, 0.2, 1.0)
    reference_origin: str = "target"


@dataclass
class CameraTrial:
    config: CameraConfig
    cover_edge_gap_mm: float
    base_result: accessibility.TrialResult
    closure_visibility_fraction: float
    path_visibility_fraction: float
    visible_at_closure_start: bool
    visible_at_closure_end: bool
    dominant_occluder: str
    minimum_environment_clearance_mm: float
    minimum_camera_environment_clearance_mm: float | None
    minimum_camera_hand_clearance_mm: float | None
    minimum_support_margin_mm: float
    maximum_camera_com_shift_mm: float
    added_camera_mass_kg: float
    localization_ok: bool
    visibility_ok: bool
    installation_ok: bool
    predicted_feasible: bool


CONFIG_LABELS = {
    "magpie_integrated": "Magpie integrated palm camera",
    "rh56_head": "RH56 head camera only",
    "rh56_palm_under": "RH56 under-hand long camera",
    "rh56_dorsal": "RH56 close dorsal long camera",
    "rh56_close_wrist": "RH56 close outboard wrist camera",
}

CONFIG_COLORS = {
    "magpie_integrated": "#0072b2",
    "rh56_head": "#6c757d",
    "rh56_palm_under": "#009e73",
    "rh56_dorsal": "#cc79a7",
    "rh56_close_wrist": "#e69f00",
}

CONFIG_STYLES = {
    "magpie_integrated": "-",
    "rh56_head": "--",
    "rh56_palm_under": "-.",
    "rh56_dorsal": ":",
    "rh56_close_wrist": (0, (5, 1)),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Magpie and RH56 camera placement trade-offs in an H12 "
            "nearby low crowned-cover surrogate task."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/h12_camera_mount_tradeoff"),
    )
    parser.add_argument(
        "--cover-edge-gaps-mm",
        "--obstacle-gaps-mm",
        dest="cover_edge_gaps_mm",
        type=float,
        nargs="+",
        default=(5.0, 10.0, 15.0, 20.0, 30.0, 40.0),
    )
    parser.add_argument("--representative-gap-mm", type=float, default=15.0)
    parser.add_argument("--cover-width-mm", type=float, default=120.0)
    parser.add_argument("--cover-slope-run-mm", type=float, default=15.0)
    parser.add_argument("--cover-rise-mm", type=float, default=10.0)
    parser.add_argument("--cover-thickness-mm", type=float, default=2.0)
    parser.add_argument("--rh56-cutoff-mm", type=float, default=40.0)
    parser.add_argument("--closure-frames", type=int, default=13)
    parser.add_argument("--approach-frames", type=int, default=5)
    parser.add_argument("--extraction-frames", type=int, default=5)
    parser.add_argument("--head-localization-error-mm", type=float, default=50.0)
    parser.add_argument("--eye-in-hand-error-mm", type=float, default=5.0)
    parser.add_argument("--capture-radius-mm", type=float, default=10.0)
    parser.add_argument("--minimum-closure-visibility", type=float, default=0.80)
    parser.add_argument("--camera-fovy-deg", type=float, default=58.0)
    parser.add_argument("--camera-width-mm", type=float, default=80.0)
    parser.add_argument("--camera-height-mm", type=float, default=20.0)
    parser.add_argument("--camera-depth-mm", type=float, default=20.0)
    parser.add_argument("--camera-mass-g", type=float, default=50.0)
    parser.add_argument(
        "--palm-camera-height-mm",
        type=float,
        default=55.0,
        help="Lens height above the screw at the closure reference frame.",
    )
    parser.add_argument("--dorsal-wrist-offset-mm", type=float, default=20.0)
    parser.add_argument("--close-wrist-outboard-mm", type=float, default=40.0)
    parser.add_argument("--close-wrist-down-mm", type=float, default=20.0)
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--panel-height", type=int, default=300)
    parser.add_argument("--external-camera-distance", type=float, default=0.85)
    parser.add_argument("--external-camera-azimuth", type=float, default=150.0)
    parser.add_argument("--external-camera-elevation", type=float, default=-28.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--hold-frames", type=int, default=24)
    parser.add_argument("--video-format", choices=("auto", "mp4", "gif"), default="auto")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if any(gap <= 0.0 for gap in args.cover_edge_gaps_mm):
        raise ValueError("--cover-edge-gaps-mm values must be positive")
    if args.closure_frames < 2:
        raise ValueError("--closure-frames must be at least 2")
    if args.approach_frames < 1 or args.extraction_frames < 1:
        raise ValueError("approach/extraction frame counts must be positive")
    if not 0.0 <= args.minimum_closure_visibility <= 1.0:
        raise ValueError("--minimum-closure-visibility must be in [0, 1]")
    if min(args.camera_width_mm, args.camera_height_mm, args.camera_depth_mm) <= 0.0:
        raise ValueError("camera dimensions must be positive")
    if args.camera_mass_g < 0.0:
        raise ValueError("--camera-mass-g cannot be negative")
    if min(
        args.cover_width_mm,
        args.cover_slope_run_mm,
        args.cover_rise_mm,
        args.cover_thickness_mm,
    ) <= 0.0:
        raise ValueError("sloped-cover dimensions must be positive")


def camera_configs(args: argparse.Namespace) -> tuple[CameraConfig, ...]:
    eye_error = args.eye_in_hand_error_mm
    return (
        CameraConfig(
            "magpie_integrated",
            CONFIG_LABELS["magpie_integrated"],
            "magpie",
            "existing_magpie",
            eye_error,
        ),
        CameraConfig(
            "rh56_head",
            CONFIG_LABELS["rh56_head"],
            "rh56",
            "head",
            args.head_localization_error_mm,
        ),
        CameraConfig(
            "rh56_palm_under",
            CONFIG_LABELS["rh56_palm_under"],
            "rh56",
            "wrist",
            eye_error,
            (0.0, 0.0, args.palm_camera_height_mm / 1000.0),
            (0.05, 0.72, 0.31, 1.0),
        ),
        CameraConfig(
            "rh56_dorsal",
            CONFIG_LABELS["rh56_dorsal"],
            "rh56",
            "wrist",
            eye_error,
            (0.0, 0.0, args.dorsal_wrist_offset_mm / 1000.0),
            (0.72, 0.24, 0.66, 1.0),
            "wrist",
        ),
        CameraConfig(
            "rh56_close_wrist",
            CONFIG_LABELS["rh56_close_wrist"],
            "rh56",
            "wrist",
            eye_error,
            (
                0.0,
                -args.close_wrist_outboard_mm / 1000.0,
                -args.close_wrist_down_mm / 1000.0,
            ),
            (0.90, 0.58, 0.05, 1.0),
            "wrist",
        ),
    )


def base_benchmark_args(args: argparse.Namespace) -> argparse.Namespace:
    base = accessibility.parse_args([])
    base.rh56_cutoff_mm = args.rh56_cutoff_mm
    base.closure_frames = args.closure_frames
    base.approach_frames = args.approach_frames
    base.extraction_frames = args.extraction_frames
    base.capture_radius_mm = args.capture_radius_mm
    base.cover_edge_gap_mm = args.representative_gap_mm
    base.cover_edge_gap_sweep_mm = tuple(sorted(set(args.cover_edge_gaps_mm)))
    base.cover_width_mm = args.cover_width_mm
    base.cover_slope_run_mm = args.cover_slope_run_mm
    base.cover_rise_mm = args.cover_rise_mm
    base.cover_thickness_mm = args.cover_thickness_mm
    base.no_video = True
    base.live = False
    return base


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=float).reshape(-1))
    return quaternion


def look_at_rotation(camera_world: np.ndarray, target_world: np.ndarray) -> np.ndarray:
    """Return camera-to-world rotation; MuJoCo cameras look along local -z."""
    view = np.asarray(target_world) - np.asarray(camera_world)
    view /= np.linalg.norm(view)
    z_axis = -view
    image_up_hint = np.array([1.0, 0.0, 0.0])
    y_axis = image_up_hint - np.dot(image_up_hint, z_axis) * z_axis
    if np.linalg.norm(y_axis) < 1e-8:
        image_up_hint = np.array([0.0, 1.0, 0.0])
        y_axis = image_up_hint - np.dot(image_up_hint, z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def relative_pose(
    parent_position: np.ndarray,
    parent_rotation: np.ndarray,
    child_position: np.ndarray,
    child_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        parent_rotation.T @ (child_position - parent_position),
        parent_rotation.T @ child_rotation,
    )


def magpie_head_pose_in_torso() -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(accessibility.DEFAULT_MAGPIE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    torso_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    camera_id = named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    return relative_pose(
        data.xpos[torso_id],
        data.xmat[torso_id].reshape(3, 3),
        data.cam_xpos[camera_id],
        data.cam_xmat[camera_id].reshape(3, 3),
    )


def wrist_camera_pose(
    result: accessibility.TrialResult,
    config: CameraConfig,
    reference_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(result.condition.model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = result.condition.qpos[reference_index]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    wrist_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    offset = np.asarray(config.reference_offset_world_m, dtype=float)
    if config.reference_origin == "target":
        lens_world = result.actual_screw_world_m + offset
    elif config.reference_origin == "wrist":
        lens_world = data.xpos[wrist_id] + offset
    else:
        raise ValueError(f"Unknown camera reference origin: {config.reference_origin}")
    camera_world_rotation = look_at_rotation(lens_world, result.actual_screw_world_m)
    local_position, local_rotation = relative_pose(
        data.xpos[wrist_id],
        data.xmat[wrist_id].reshape(3, 3),
        lens_world,
        camera_world_rotation,
    )
    return local_position, local_rotation, lens_world, camera_world_rotation


def add_fixed_camera_body(
    spec: mujoco.MjSpec,
    *,
    parent_name: str,
    body_name: str,
    camera_name: str,
    local_position: np.ndarray,
    local_rotation: np.ndarray,
    fovy_deg: float,
    housing_size_m: np.ndarray | None,
    housing_mass_kg: float,
    housing_rgba: tuple[float, float, float, float],
) -> None:
    parent = spec.body(parent_name)
    if parent is None:
        raise ValueError(f"Missing camera parent body: {parent_name}")
    body = parent.add_body()
    body.name = body_name
    body.pos = local_position
    body.quat = rotation_to_quaternion(local_rotation)
    camera = body.add_camera()
    camera.name = camera_name
    camera.fovy = fovy_deg
    camera.pos = np.zeros(3)
    camera.quat = np.array([1.0, 0.0, 0.0, 0.0])
    if housing_size_m is not None:
        housing = body.add_geom()
        housing.name = "evaluation_camera_housing"
        housing.type = mujoco.mjtGeom.mjGEOM_BOX
        housing.size = 0.5 * housing_size_m
        # The lens is at the front face. MuJoCo looks along local -z, so the
        # housing extends behind the lens in local +z.
        housing.pos = np.array([0.0, 0.0, 0.5 * housing_size_m[2]])
        housing.mass = housing_mass_kg
        housing.rgba = np.asarray(housing_rgba)
        housing.contype = 1
        housing.conaffinity = 1


def build_camera_model(
    config: CameraConfig,
    result: accessibility.TrialResult,
    args: argparse.Namespace,
    base_args: argparse.Namespace,
) -> tuple[mujoco.MjModel, dict[str, object]]:
    spec = mujoco.MjSpec.from_file(str(result.condition.model_path))
    accessibility.add_surrogate_geoms(spec, base_args, result.workcell)
    metadata: dict[str, object] = {}
    if config.kind == "existing_magpie":
        camera_name = "hand_cam"
        camera_body_name = "gripper_attachment"
    elif config.kind == "head":
        local_position, local_rotation = magpie_head_pose_in_torso()
        add_fixed_camera_body(
            spec,
            parent_name="torso_link",
            body_name="evaluation_head_camera_body",
            camera_name="evaluation_camera",
            local_position=local_position,
            local_rotation=local_rotation,
            fovy_deg=args.camera_fovy_deg,
            housing_size_m=None,
            housing_mass_kg=0.0,
            housing_rgba=config.color,
        )
        camera_name = "evaluation_camera"
        camera_body_name = "evaluation_head_camera_body"
    else:
        reference_index = base_args.approach_frames
        local_position, local_rotation, lens_world, camera_world_rotation = wrist_camera_pose(
            result,
            config,
            reference_index,
        )
        housing_size = np.array([
            args.camera_width_mm,
            args.camera_height_mm,
            args.camera_depth_mm,
        ]) / 1000.0
        add_fixed_camera_body(
            spec,
            parent_name="right_wrist_yaw_link",
            body_name="evaluation_wrist_camera_body",
            camera_name="evaluation_camera",
            local_position=local_position,
            local_rotation=local_rotation,
            fovy_deg=args.camera_fovy_deg,
            housing_size_m=housing_size,
            housing_mass_kg=args.camera_mass_g / 1000.0,
            housing_rgba=config.color,
        )
        camera_name = "evaluation_camera"
        camera_body_name = "evaluation_wrist_camera_body"
        metadata.update({
            "reference_lens_world_m": lens_world.tolist(),
            "reference_camera_rotation": camera_world_rotation.tolist(),
            "wrist_local_position_m": local_position.tolist(),
            "wrist_local_quaternion_wxyz": rotation_to_quaternion(local_rotation).tolist(),
        })
    model = spec.compile()
    metadata["camera_name"] = camera_name
    metadata["camera_body_name"] = camera_body_name
    return model, metadata


def camera_frame_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_name: str,
    target_world: np.ndarray,
    aspect: float,
) -> tuple[bool, bool, str]:
    camera_id = named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera_position = data.cam_xpos[camera_id]
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    target_local = camera_rotation.T @ (target_world - camera_position)
    depth = -float(target_local[2])
    vertical_tangent = math.tan(math.radians(float(model.cam_fovy[camera_id])) / 2.0)
    horizontal_tangent = aspect * vertical_tangent
    in_fov = bool(
        depth > 0.0
        and abs(float(target_local[0])) <= depth * horizontal_tangent
        and abs(float(target_local[1])) <= depth * vertical_tangent
    )
    if not in_fov:
        return False, False, "out_of_fov"

    ray = target_world - camera_position
    target_distance = float(np.linalg.norm(ray))
    ray /= target_distance
    hit_geom = np.array([-1], dtype=np.int32)
    hit_distance = float(mujoco.mj_ray(
        model,
        data,
        camera_position,
        ray,
        None,
        1,
        int(model.cam_bodyid[camera_id]),
        hit_geom,
    ))
    geom_id = int(hit_geom[0])
    if hit_distance < 0.0:
        return True, True, "clear_ray"
    geom_name = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        or f"geom_{geom_id}"
    )
    if geom_name.startswith("surrogate_screw"):
        return True, True, "target"
    # A hit at or behind the target does not occlude the target point.
    if hit_distance >= target_distance - 0.002:
        return True, True, "clear_ray"
    body_id = int(model.geom_bodyid[geom_id])
    body_name = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        or "world"
    )
    return True, False, f"{body_name}/{geom_name}"


def geom_pair_minimum_distance(
    model: mujoco.MjModel,
    qpos_rows: np.ndarray,
    first_geom_id: int,
    other_geom_ids: list[int],
) -> float:
    data = mujoco.MjData(model)
    minimum = float("inf")
    for qpos in qpos_rows:
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for other_geom_id in other_geom_ids:
            distance = float(mujoco.mj_geomDistance(
                model,
                data,
                first_geom_id,
                other_geom_id,
                0.5,
                None,
            ))
            minimum = min(minimum, distance)
    return 1000.0 * minimum


def obb_signed_separation(
    first_center: np.ndarray,
    first_rotation: np.ndarray,
    first_half_size: np.ndarray,
    second_center: np.ndarray,
    second_rotation: np.ndarray,
    second_half_size: np.ndarray,
) -> float:
    """Signed separating-axis margin for two oriented boxes.

    Positive values prove separation. Negative values mean overlap on every
    candidate separating axis. For separated diagonal boxes this is a lower
    bound, rather than the exact Euclidean surface distance.
    """
    axes = [first_rotation[:, index] for index in range(3)]
    axes.extend(second_rotation[:, index] for index in range(3))
    for first_index in range(3):
        for second_index in range(3):
            axis = np.cross(
                first_rotation[:, first_index],
                second_rotation[:, second_index],
            )
            norm = float(np.linalg.norm(axis))
            if norm > 1e-10:
                axes.append(axis / norm)
    delta = np.asarray(second_center) - np.asarray(first_center)
    separations = []
    for axis in axes:
        axis = np.asarray(axis) / np.linalg.norm(axis)
        first_radius = float(np.sum(
            first_half_size * np.abs(first_rotation.T @ axis)
        ))
        second_radius = float(np.sum(
            second_half_size * np.abs(second_rotation.T @ axis)
        ))
        separations.append(abs(float(np.dot(delta, axis))) - first_radius - second_radius)
    return float(max(separations))


def box_pair_minimum_separation(
    model: mujoco.MjModel,
    qpos_rows: np.ndarray,
    first_geom_id: int,
    other_geom_ids: list[int],
) -> float:
    data = mujoco.MjData(model)
    minimum = float("inf")
    first_size = np.asarray(model.geom_size[first_geom_id], dtype=float)
    for qpos in qpos_rows:
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        first_rotation = data.geom_xmat[first_geom_id].reshape(3, 3)
        for other_geom_id in other_geom_ids:
            if int(model.geom_type[other_geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError("SAT camera clearance expects box obstacles")
            separation = obb_signed_separation(
                data.geom_xpos[first_geom_id],
                first_rotation,
                first_size,
                data.geom_xpos[other_geom_id],
                data.geom_xmat[other_geom_id].reshape(3, 3),
                np.asarray(model.geom_size[other_geom_id], dtype=float),
            )
            minimum = min(minimum, separation)
    return 1000.0 * minimum


def camera_clearances(
    model: mujoco.MjModel,
    qpos_rows: np.ndarray,
) -> tuple[float | None, float | None]:
    housing_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "evaluation_camera_housing",
    )
    if housing_id < 0:
        return None, None
    environment_ids = [
        named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "surrogate_pack",
            "surrogate_cover_near_slope",
            "surrogate_cover_far_slope",
        )
    ]
    environment_clearance = box_pair_minimum_separation(
        model,
        qpos_rows,
        housing_id,
        environment_ids,
    )
    wrist_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    housing_body_id = int(model.geom_bodyid[housing_id])
    hand_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if geom_id != housing_id
        and int(model.geom_contype[geom_id]) != 0
        and accessibility.body_is_descendant(
            model,
            int(model.geom_bodyid[geom_id]),
            wrist_id,
        )
        and int(model.geom_bodyid[geom_id]) != housing_body_id
    ]
    hand_clearance = geom_pair_minimum_distance(
        model,
        qpos_rows,
        housing_id,
        hand_ids,
    )
    return environment_clearance, hand_clearance


def balance_metrics(
    model: mujoco.MjModel,
    qpos_rows: np.ndarray,
    base_com_world: np.ndarray,
    *,
    sole_tolerance_m: float,
) -> tuple[float, float, float]:
    data = mujoco.MjData(model)
    polygon = None
    margins = []
    shifts = []
    for frame_index, qpos in enumerate(qpos_rows):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if polygon is None:
            polygon = foot_support_polygon(
                model,
                data,
                sole_tolerance_m=sole_tolerance_m,
            )
        com = whole_body_com(model, data)
        margins.append(1000.0 * signed_support_margin_m(com[:2], polygon))
        shifts.append(1000.0 * float(np.linalg.norm(com - base_com_world[frame_index])))
    return (
        float(np.min(margins)),
        float(np.max(shifts)),
        float(np.asarray(model.body_mass[1:]).sum()),
    )


def evaluate_camera_trial(
    config: CameraConfig,
    result: accessibility.TrialResult,
    args: argparse.Namespace,
    base_args: argparse.Namespace,
) -> tuple[CameraTrial, dict[str, object]]:
    model, mount_metadata = build_camera_model(config, result, args, base_args)
    data = mujoco.MjData(model)
    camera_name = str(mount_metadata["camera_name"])
    visible = []
    occluders = []
    aspect = args.panel_width / args.panel_height
    for qpos in result.condition.qpos:
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        _, frame_visible, occluder = camera_frame_observation(
            model,
            data,
            camera_name=camera_name,
            target_world=result.actual_screw_world_m,
            aspect=aspect,
        )
        visible.append(frame_visible)
        if not frame_visible:
            occluders.append(occluder)
    visible_array = np.asarray(visible, dtype=bool)
    closure_slice = slice(
        base_args.approach_frames,
        base_args.approach_frames + base_args.closure_frames,
    )
    closure_visible = visible_array[closure_slice]
    dominant_occluder = Counter(occluders).most_common(1)[0][0] if occluders else "none"
    environment_clearance, hand_clearance = camera_clearances(
        model,
        result.condition.qpos,
    )
    overall_clearance, _, _ = accessibility.collision_metrics(
        model,
        result.condition.qpos,
        result.phases,
    )
    min_support, camera_com_shift, total_mass = balance_metrics(
        model,
        result.condition.qpos,
        result.condition.com_world,
        sole_tolerance_m=base_args.foot_sole_tolerance_mm / 1000.0,
    )
    base_model = mujoco.MjModel.from_xml_path(str(result.condition.model_path))
    base_mass = float(np.asarray(base_model.body_mass[1:]).sum())
    localization_ok = config.localization_error_mm <= args.capture_radius_mm
    visibility_ok = bool(
        closure_visible[0]
        and closure_visible[-1]
        and float(np.mean(closure_visible)) >= args.minimum_closure_visibility
    )
    installation_ok = bool(
        hand_clearance is None or hand_clearance >= 0.0
    )
    predicted_feasible = bool(
        result.reachable
        and result.joint_margin_ok
        and result.closure_sufficient
        and overall_clearance >= base_args.collision_clearance_mm
        and min_support > 0.0
        and localization_ok
        and visibility_ok
        and installation_ok
    )
    return CameraTrial(
        config=config,
        cover_edge_gap_mm=float(result.spec.cover_edge_gap_mm),
        base_result=result,
        closure_visibility_fraction=float(np.mean(closure_visible)),
        path_visibility_fraction=float(np.mean(visible_array)),
        visible_at_closure_start=bool(closure_visible[0]),
        visible_at_closure_end=bool(closure_visible[-1]),
        dominant_occluder=dominant_occluder,
        minimum_environment_clearance_mm=overall_clearance,
        minimum_camera_environment_clearance_mm=environment_clearance,
        minimum_camera_hand_clearance_mm=hand_clearance,
        minimum_support_margin_mm=min_support,
        maximum_camera_com_shift_mm=camera_com_shift,
        added_camera_mass_kg=total_mass - base_mass,
        localization_ok=localization_ok,
        visibility_ok=visibility_ok,
        installation_ok=installation_ok,
        predicted_feasible=predicted_feasible,
    ), mount_metadata


def trial_row(trial: CameraTrial) -> dict[str, object]:
    condition = trial.base_result.condition
    return {
        "configuration": trial.config.key,
        "configuration_label": trial.config.label,
        "hand": trial.config.hand,
        "cover_edge_gap_mm": trial.cover_edge_gap_mm,
        "localization_error_assumption_mm": trial.config.localization_error_mm,
        "localization_ok": trial.localization_ok,
        "closure_visibility_fraction": trial.closure_visibility_fraction,
        "path_visibility_fraction": trial.path_visibility_fraction,
        "visible_at_closure_start": trial.visible_at_closure_start,
        "visible_at_closure_end": trial.visible_at_closure_end,
        "dominant_occluder": trial.dominant_occluder,
        "visibility_ok": trial.visibility_ok,
        "minimum_environment_clearance_mm": trial.minimum_environment_clearance_mm,
        "minimum_camera_environment_clearance_mm": (
            "" if trial.minimum_camera_environment_clearance_mm is None
            else trial.minimum_camera_environment_clearance_mm
        ),
        "minimum_camera_hand_clearance_mm": (
            "" if trial.minimum_camera_hand_clearance_mm is None
            else trial.minimum_camera_hand_clearance_mm
        ),
        "installation_ok": trial.installation_ok,
        "wrist_path_mm": float(condition.wrist_path_mm[-1]),
        "wrist_rotation_deg": float(condition.wrist_rotation_path_deg[-1]),
        "arm_joint_travel_rad": float(condition.arm_joint_travel_rad[-1]),
        "minimum_static_support_margin_mm": trial.minimum_support_margin_mm,
        "maximum_camera_induced_com_shift_mm": trial.maximum_camera_com_shift_mm,
        "added_camera_mass_kg": trial.added_camera_mass_kg,
        "base_reachable": trial.base_result.reachable,
        "base_closure_sufficient": trial.base_result.closure_sufficient,
        "predicted_feasible": trial.predicted_feasible,
        "failure_reason": failure_reason(trial),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def failure_reason(trial: CameraTrial) -> str:
    if not trial.base_result.reachable:
        return "unreachable"
    if not trial.base_result.joint_margin_ok:
        return "joint_limit"
    if not trial.base_result.closure_sufficient:
        return "insufficient_closure"
    if trial.minimum_environment_clearance_mm < 0.0:
        return "workcell_collision"
    if trial.minimum_support_margin_mm <= 0.0:
        return "balance_margin"
    if not trial.installation_ok:
        return "camera_hand_interference"
    if not trial.visibility_ok:
        return f"visibility:{trial.dominant_occluder}"
    if not trial.localization_ok:
        return "localization_error_assumption"
    return "none"


def summary_rows(trials: list[CameraTrial]) -> list[dict[str, object]]:
    rows = []
    for key in CONFIG_LABELS:
        subset = [trial for trial in trials if trial.config.key == key]
        rows.append({
            "configuration": key,
            "configuration_label": CONFIG_LABELS[key],
            "trials": len(subset),
            "predicted_feasible_trials": sum(trial.predicted_feasible for trial in subset),
            "predicted_feasible_rate": float(np.mean([
                trial.predicted_feasible for trial in subset
            ])),
            "mean_closure_visibility_fraction": float(np.mean([
                trial.closure_visibility_fraction for trial in subset
            ])),
            "minimum_environment_clearance_mm": min(
                trial.minimum_environment_clearance_mm for trial in subset
            ),
            "minimum_camera_environment_clearance_mm": min(
                (
                    trial.minimum_camera_environment_clearance_mm
                    for trial in subset
                    if trial.minimum_camera_environment_clearance_mm is not None
                ),
                default="",
            ),
            "minimum_camera_hand_clearance_mm": min(
                (
                    trial.minimum_camera_hand_clearance_mm
                    for trial in subset
                    if trial.minimum_camera_hand_clearance_mm is not None
                ),
                default="",
            ),
            "minimum_static_support_margin_mm": min(
                trial.minimum_support_margin_mm for trial in subset
            ),
            "maximum_camera_induced_com_shift_mm": max(
                trial.maximum_camera_com_shift_mm for trial in subset
            ),
            "localization_error_assumption_mm": subset[0].config.localization_error_mm,
            "dominant_failure_or_occluder": Counter(
                failure_reason(trial)
                for trial in subset
                if not trial.predicted_feasible
            ).most_common(1)[0][0] if any(
                not trial.predicted_feasible for trial in subset
            ) else "none",
        })
    return rows


def plot_camera_tradeoffs(path: Path, trials: list[CameraTrial]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.5), sharex=True)
    for key in CONFIG_LABELS:
        subset = sorted(
            (trial for trial in trials if trial.config.key == key),
            key=lambda trial: trial.cover_edge_gap_mm,
        )
        gaps = np.asarray([trial.cover_edge_gap_mm for trial in subset])
        color = CONFIG_COLORS[key]
        linestyle = CONFIG_STYLES[key]
        label = CONFIG_LABELS[key]
        axes[0, 0].plot(
            gaps,
            100.0 * np.asarray([trial.closure_visibility_fraction for trial in subset]),
            marker="o",
            color=color,
            linestyle=linestyle,
            label=label,
        )
        axes[0, 1].plot(
            gaps,
            [
                np.nan
                if trial.minimum_camera_environment_clearance_mm is None
                else trial.minimum_camera_environment_clearance_mm
                for trial in subset
            ],
            marker="o",
            color=color,
            linestyle=linestyle,
            label=label,
        )
        hand_clearance = [
            np.nan if trial.minimum_camera_hand_clearance_mm is None
            else trial.minimum_camera_hand_clearance_mm
            for trial in subset
        ]
        axes[1, 0].plot(
            gaps,
            hand_clearance,
            marker="o",
            color=color,
            linestyle=linestyle,
            label=label,
        )
        axes[1, 1].step(
            gaps,
            [100.0 if trial.predicted_feasible else 0.0 for trial in subset],
            where="mid",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    axes[0, 0].set_ylabel("Closure frames with screw visible (%)")
    axes[0, 0].set_title("A. View continuity during coupled closure")
    axes[0, 0].set_ylim(-3.0, 103.0)
    axes[0, 1].axhline(0.0, color="#333333", linestyle=":")
    axes[0, 1].set_ylabel("Camera housing to pack/cover separation (mm)")
    axes[0, 1].set_title("B. Added camera clearance (negative = collision)")
    axes[1, 0].axhline(0.0, color="#333333", linestyle=":")
    axes[1, 0].set_ylabel("Camera housing to RH56 clearance (mm)")
    axes[1, 0].set_title("C. Installation interference (negative = overlap)")
    axes[1, 1].set_ylabel("Predicted feasible (0 or 100%)")
    axes[1, 1].set_ylim(-3.0, 103.0)
    axes[1, 1].set_title("D. Combined scenario gate")
    for axis in axes[1, :]:
        axis.set_xlabel("Screw-head edge to sloped-cover edge gap (mm)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("H12 camera placement trade-offs near a low crowned cover")
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_motion_comparison(path: Path, trials: list[CameraTrial], representative_gap: float) -> None:
    import matplotlib.pyplot as plt

    selected = representative_trials(trials, representative_gap)
    labels = [trial.config.key.replace("rh56_", "RH56\n").replace("magpie_", "Magpie\n") for trial in selected]
    metrics = (
        ("wrist_path_mm", "Wrist path (mm)"),
        ("wrist_rotation_deg", "Wrist rotation (deg)"),
        ("arm_joint_travel_rad", "Arm joint travel (rad)"),
    )
    rows = [trial_row(trial) for trial in selected]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    colors = [CONFIG_COLORS[trial.config.key] for trial in selected]
    for axis, (field, title) in zip(axes.flat, metrics):
        axis.bar(np.arange(len(rows)), [float(row[field]) for row in rows], color=colors)
        axis.set_xticks(np.arange(len(rows)), labels, fontsize=8)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle(
        f"Motion at {selected[0].cover_edge_gap_mm:.0f} mm screw-to-cover edge gap"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def representative_trials(
    trials: list[CameraTrial],
    requested_gap_mm: float,
) -> list[CameraTrial]:
    gaps = sorted({trial.cover_edge_gap_mm for trial in trials})
    gap = min(gaps, key=lambda value: abs(value - requested_gap_mm))
    lookup = {
        (trial.config.key, trial.cover_edge_gap_mm): trial
        for trial in trials
    }
    return [lookup[(key, gap)] for key in CONFIG_LABELS]


def render_panel(
    trial: CameraTrial,
    args: argparse.Namespace,
    base_args: argparse.Namespace,
    *,
    frame_index: int,
    camera_view: bool,
) -> np.ndarray:
    model, metadata = build_camera_model(trial.config, trial.base_result, args, base_args)
    data = mujoco.MjData(model)
    data.qpos[:] = trial.base_result.condition.qpos[frame_index]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=args.panel_height, width=args.panel_width)
    try:
        if camera_view:
            renderer.update_scene(data, camera=str(metadata["camera_name"]))
        else:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = trial.base_result.actual_screw_world_m + np.array([0.0, 0.0, 0.08])
            camera.distance = args.external_camera_distance
            camera.azimuth = args.external_camera_azimuth
            camera.elevation = args.external_camera_elevation
            renderer.update_scene(data, camera=camera)
        frame = renderer.render()
    finally:
        renderer.close()
    return add_text(frame, [
        trial.config.label,
        f"cover gap {trial.cover_edge_gap_mm:.0f} mm | closure visible {100.0 * trial.closure_visibility_fraction:.0f}%",
        f"occluder {trial.dominant_occluder}",
        f"predicted feasible {trial.predicted_feasible}",
    ])


def tiled_canvas(frames: list[np.ndarray], *, columns: int = 3) -> np.ndarray:
    rows = math.ceil(len(frames) / columns)
    height, width = frames[0].shape[:2]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    canvas[:] = 28
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    return canvas


def save_snapshots(
    out: Path,
    trials: list[CameraTrial],
    args: argparse.Namespace,
    base_args: argparse.Namespace,
) -> None:
    from PIL import Image

    frame_index = base_args.approach_frames
    camera_frames = [
        render_panel(trial, args, base_args, frame_index=frame_index, camera_view=True)
        for trial in trials
    ]
    external_frames = [
        render_panel(trial, args, base_args, frame_index=frame_index, camera_view=False)
        for trial in trials
    ]
    Image.fromarray(tiled_canvas(camera_frames)).save(out / "camera_views.png")
    Image.fromarray(tiled_canvas(external_frames)).save(out / "camera_mounts.png")


def video_extension(video_format: str) -> str:
    if video_format == "gif":
        return ".gif"
    if video_format == "mp4":
        return ".mp4"
    try:
        import imageio_ffmpeg  # noqa: F401
        return ".mp4"
    except ImportError:
        return ".gif"


def render_video(
    out: Path,
    trials: list[CameraTrial],
    args: argparse.Namespace,
    base_args: argparse.Namespace,
) -> Path:
    extension = video_extension(args.video_format)
    path = out / f"camera_mount_comparison{extension}"
    models_and_metadata = [
        build_camera_model(trial.config, trial.base_result, args, base_args)
        for trial in trials
    ]
    models = [item[0] for item in models_and_metadata]
    data_rows = [mujoco.MjData(model) for model in models]
    renderers = [
        mujoco.Renderer(model, height=args.panel_height, width=args.panel_width)
        for model in models
    ]
    cameras = []
    for trial in trials:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = trial.base_result.actual_screw_world_m + np.array([0.0, 0.0, 0.08])
        camera.distance = args.external_camera_distance
        camera.azimuth = args.external_camera_azimuth
        camera.elevation = args.external_camera_elevation
        cameras.append(camera)
    total_frames = len(trials[0].base_result.condition.qpos) + args.hold_frames
    writer = None
    gif_frames: list[np.ndarray] = []
    canvas_size = (args.panel_width * 3, args.panel_height * 2)
    if extension == ".mp4":
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            str(path),
            canvas_size,
            fps=args.fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
        )
        writer.send(None)
    try:
        for output_index in range(total_frames):
            panels = []
            for trial, model, data, renderer, camera in zip(
                trials,
                models,
                data_rows,
                renderers,
                cameras,
            ):
                frame_index = min(
                    output_index,
                    len(trial.base_result.condition.qpos) - 1,
                )
                data.qpos[:] = trial.base_result.condition.qpos[frame_index]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                panels.append(add_text(renderer.render(), [
                    trial.config.label,
                    (
                        f"cover gap {trial.cover_edge_gap_mm:.0f} mm | "
                        f"phase {trial.base_result.phases[frame_index]}"
                    ),
                    (
                        f"closure visible "
                        f"{100.0 * trial.closure_visibility_fraction:.0f}%"
                    ),
                    f"result {failure_reason(trial)}",
                ]))
            canvas = np.ascontiguousarray(tiled_canvas(panels))
            if writer is not None:
                writer.send(canvas)
            else:
                gif_frames.append(canvas)
    finally:
        if writer is not None:
            writer.close()
        for renderer in renderers:
            renderer.close()
    if extension == ".gif":
        from PIL import Image

        images = [Image.fromarray(frame).convert("RGB") for frame in gif_frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=max(1, int(round(1000.0 / args.fps))),
            loop=0,
        )
    return path


def live_worker(
    trial: CameraTrial,
    args: argparse.Namespace,
    base_args: argparse.Namespace,
) -> None:
    os.environ.pop("MUJOCO_GL", None)
    import mujoco.viewer

    model, _ = build_camera_model(trial.config, trial.base_result, args, base_args)
    data = mujoco.MjData(model)
    frame_period = 1.0 / args.fps
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = trial.base_result.actual_screw_world_m + np.array([0.0, 0.0, 0.08])
        viewer.cam.distance = args.external_camera_distance
        viewer.cam.azimuth = args.external_camera_azimuth
        viewer.cam.elevation = args.external_camera_elevation
        while viewer.is_running():
            for qpos in trial.base_result.condition.qpos:
                if not viewer.is_running():
                    return
                start = time.monotonic()
                data.qpos[:] = qpos
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                delay = frame_period - (time.monotonic() - start)
                if delay > 0.0:
                    time.sleep(delay)
            if args.once:
                while viewer.is_running():
                    time.sleep(0.05)
                return


def run_live(
    trials: list[CameraTrial],
    args: argparse.Namespace,
    base_args: argparse.Namespace,
) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=live_worker, args=(trial, args, base_args))
        for trial in trials
    ]
    for process in processes:
        process.start()
    print("[camera] live viewers started; close all windows to stop")
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()


def write_assumptions(
    path: Path,
    *,
    args: argparse.Namespace,
    base_args: argparse.Namespace,
    mount_metadata: dict[str, dict[str, object]],
    trials: list[CameraTrial],
    video_path: Path | None,
) -> None:
    payload = {
        "script": "tools/run_h12_camera_mount_tradeoff.py",
        "simulation_only": True,
        "uses_hardware": False,
        "workcell_layout": (
            "robot -> screw -> short gap -> low crowned cover along robot +x"
        ),
        "workcell_geometry": {
            "real_battery_cad_used": False,
            "cover_edge_gaps_mm": sorted(set(args.cover_edge_gaps_mm)),
            "cover_width_mm": base_args.cover_width_mm,
            "cover_slope_run_mm_each_side": base_args.cover_slope_run_mm,
            "cover_rise_mm": base_args.cover_rise_mm,
            "cover_thickness_mm": base_args.cover_thickness_mm,
        },
        "camera_models": {
            "magpie": (
                "Uses the checked-in hand_cam on the Magpie central/palm axis."
            ),
            "rh56_head": (
                "Transfers the checked-in H12 Magpie head_cam optical pose to "
                "the same torso body in the RH56 model; no added wrist mass."
            ),
            "rh56_mounts": (
                "Parameterized long-cuboid webcam-like housings rigidly "
                "attached to the right wrist at poses constructed at closure "
                "start. They are design proxies, not validated brackets."
            ),
            "fovy_deg": args.camera_fovy_deg,
            "housing_dimensions_mm": [
                args.camera_width_mm,
                args.camera_height_mm,
                args.camera_depth_mm,
            ],
            "housing_mass_g": args.camera_mass_g,
            "mount_reference_offsets_world_mm": {
                config.key: (
                    None
                    if config.reference_offset_world_m is None
                    else (1000.0 * np.asarray(config.reference_offset_world_m)).tolist()
                )
                for config in camera_configs(args)
            },
            "mount_reference_origins": {
                config.key: config.reference_origin
                for config in camera_configs(args)
            },
            "resolved_mounts": mount_metadata,
        },
        "external_visualization_camera": {
            "distance": args.external_camera_distance,
            "azimuth_deg": args.external_camera_azimuth,
            "elevation_deg": args.external_camera_elevation,
        },
        "localization_scenarios_not_sim_measurements": {
            "head_camera_error_mm": args.head_localization_error_mm,
            "eye_in_hand_error_mm": args.eye_in_hand_error_mm,
            "capture_radius_mm": args.capture_radius_mm,
            "warning": (
                "These values connect the geometry study to the prior injected-"
                "error sweep. They are explicit scenario assumptions; this "
                "script does not simulate a detector or visual servo."
            ),
        },
        "visibility_gate": {
            "minimum_closure_visibility_fraction": args.minimum_closure_visibility,
            "requires_visible_at_closure_start": True,
            "requires_visible_at_closure_end": True,
            "occlusion_method": "MuJoCo ray cast from optical center to screw",
        },
        "clearance_methods": {
            "whole_right_limb_to_workcell": (
                "Minimum MuJoCo signed geom distance. Negative means "
                "penetration; zero is treated as no detected penetration, not "
                "as a certified positive safety margin."
            ),
            "camera_box_to_pack_and_sloped_cover": (
                "Signed oriented-box separating-axis margin; a positive value "
                "proves separation and is a lower bound on diagonal "
                "Euclidean clearance."
            ),
            "camera_box_to_rh56": "Minimum MuJoCo signed geom distance.",
        },
        "balance": {
            "metric": "whole-body CoM signed distance to double-support polygon",
            "dynamic_balance_evaluated": False,
            "reporting_priority": (
                "Recorded as a secondary safety check; omitted from the main "
                "motion figure because the camera is a small close-wrist unit."
            ),
        },
        "prediction_gate": [
            "base kinematics reachable",
            "arm joints inside limits",
            "closure sufficient for screw proxy",
            "hand/arm/camera workcell clearance >= 0 mm",
            "static support margin > 0 mm",
            "assumed localization error <= capture radius",
            "closure visibility gate passes",
            "camera proxy does not overlap collidable RH56 geometry",
        ],
        "not_evaluated": [
            "camera image noise, depth noise, calibration, or detector accuracy",
            "camera cable, connector, bracket, thermal, or structural design",
            "real battery CAD and material/contact fidelity",
            "closed-loop visual servo dynamics",
            "dynamic standing recovery",
        ],
        "trial_count": len(trials),
        "video_path": str(video_path.resolve()) if video_path else None,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_chinese_guide(path: Path, trials: list[CameraTrial]) -> None:
    summaries = {row["configuration"]: row for row in summary_rows(trials)}
    lines = [
        "# H12 螺丝任务摄像头对比：如何读结果",
        "",
        "场景沿机器人前向排列为：**机器人 → 螺丝 → 低矮双斜面盖板**。横轴是螺丝头边缘到盖板近侧低边的距离。盖板每侧斜坡长 15 mm，中央比低边高 10 mm。",
        "",
        "## 为什么做这四项比较",
        "",
        "- 图 A（闭合可见率）：检查耦合关节闭合时，手是否挡住螺丝。100% 表示整个闭合阶段都能看见。",
        "- 图 B（新增相机净空）：只看新增 RH56 摄像头外壳到台面/低矮斜盖板的分离距离，用来判断小盖板是否会限制相机；负值表示穿模碰撞。",
        "- 图 C（安装净空）：只测摄像头外壳与 RH56 本体。负值表示这个假设安装位会占用现有手部实体空间，需要改支架或改位置。",
        "- 图 D（综合可行性）：只有运动可达、无碰撞、看得见、假设定位误差小于抓取容差且静态 CoM 仍在支撑区内时才为 100%。它是筛选门，不是实际成功率。",
        "- arm_motion_comparison.png：三张柱图回答 RH56 耦合闭合让腕和手臂多动多少。平衡仅保留在 CSV 中作为次要安全检查。",
        "- camera_views.png：每个方案在闭合开始时实际渲染出的相机画面，用来直观看视场和遮挡。",
        "- camera_mounts.png / 视频：从机器人正面观察安装位置和整个抓取动作。",
        "",
        "## 重要解释边界",
        "",
        "头部与眼在手上的定位误差是显式情景假设，不是这段 MuJoCo 仿真测出来的相机精度。综合可行性也不是成功率。RH56 摄像头是 80×20×20 mm 的长条代理，不是已经设计完成的支架。",
        "",
        "所有 RH56 摄像头方案使用同一条手臂补偿轨迹，所以三项 RH56 运动量相同；当前实验只改变摄像头几何和质量，没有为了某个安装位重新规划手臂。近腕方案的镜头参考点位于腕中心向外 40 mm、向下 20 mm，尚未加入真实支架和线缆。整手碰撞检查返回的 0 mm 只表示未检测到穿透，并不代表已经留出工程安全距离。",
        "",
        "## 本次汇总",
        "",
        "| 方案 | 闭合平均可见率 | 可行距离点 | 相机-环境最小分离 | 最小安装净空 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in CONFIG_LABELS:
        row = summaries[key]
        environment = row["minimum_camera_environment_clearance_mm"]
        environment_text = (
            "不适用" if environment == "" else f"{float(environment):.1f} mm"
        )
        install = row["minimum_camera_hand_clearance_mm"]
        install_text = "不适用" if install == "" else f"{float(install):.1f} mm"
        lines.append(
            f"| {CONFIG_LABELS[key]} | "
            f"{100.0 * float(row['mean_closure_visibility_fraction']):.1f}% | "
            f"{row['predicted_feasible_trials']}/{row['trials']} | "
            f"{environment_text} | "
            f"{install_text} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    base_args = base_benchmark_args(args)
    accessibility.validate_args(base_args)
    configs = camera_configs(args)
    templates, _ = accessibility.build_hand_templates(base_args)

    reference_model = mujoco.MjModel.from_xml_path(str(base_args.rh56_xml))
    reference_data = mujoco.MjData(reference_model)
    mujoco.mj_forward(reference_model, reference_data)
    pelvis_id = named_id(reference_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    pelvis_world = reference_data.xpos[pelvis_id].copy()

    base_results: dict[tuple[str, float], accessibility.TrialResult] = {}
    environment_cache: dict = {}
    gaps = sorted(set(float(gap) for gap in args.cover_edge_gaps_mm))
    for gap in gaps:
        spec = accessibility.TrialSpec(
            "cover_edge_gap",
            0.0,
            0.0,
            0.0,
            0.0,
            cover_edge_gap_mm=gap,
        )
        for hand in ("magpie", "rh56"):
            print(f"[camera] solving {hand} base motion at {gap:.0f} mm gap")
            base_results[(hand, gap)] = accessibility.evaluate_trial(
                templates[hand],
                spec,
                base_args,
                pelvis_world_m=pelvis_world,
                environment_cache=environment_cache,
            )

    trials = []
    resolved_mounts: dict[str, dict[str, object]] = {}
    for config in configs:
        for gap in gaps:
            print(f"[camera] evaluating {config.key} at {gap:.0f} mm gap")
            trial, metadata = evaluate_camera_trial(
                config,
                base_results[(config.hand, gap)],
                args,
                base_args,
            )
            trials.append(trial)
            resolved_mounts.setdefault(config.key, metadata)

    write_csv(args.out / "trials.csv", [trial_row(trial) for trial in trials])
    write_csv(args.out / "summary.csv", summary_rows(trials))
    plot_camera_tradeoffs(args.out / "camera_tradeoffs.png", trials)
    plot_motion_comparison(
        args.out / "arm_motion_comparison.png",
        trials,
        args.representative_gap_mm,
    )
    selected = representative_trials(trials, args.representative_gap_mm)
    print("[camera] rendering camera and mount snapshots")
    save_snapshots(args.out, selected, args, base_args)
    video_path = None
    if not args.no_video:
        print("[camera] rendering composite front-view video")
        video_path = render_video(args.out, selected, args, base_args)
        print(f"[camera] video: {video_path}")
    write_assumptions(
        args.out / "assumptions.json",
        args=args,
        base_args=base_args,
        mount_metadata=resolved_mounts,
        trials=trials,
        video_path=video_path,
    )
    write_chinese_guide(args.out / "README_zh.md", trials)
    for row in summary_rows(trials):
        print(
            f"[camera] {row['configuration']}: feasible="
            f"{row['predicted_feasible_trials']}/{row['trials']}, "
            f"mean closure visibility="
            f"{100.0 * row['mean_closure_visibility_fraction']:.1f}%"
        )
    print(f"[camera] results: {args.out}")
    if args.live:
        run_live(selected, args, base_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
